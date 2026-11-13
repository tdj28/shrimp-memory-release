#!/usr/bin/env python3
"""Part B of docs/public-toggle-fold.md: does the offset from the pulse's slow line throw the copy out?

Five fixed groups at the toggle-check controls (30 degrees, radius 3.5), each one stage of six updates, both lanes
together, started from the states in a bound start file (tools/prepare_public_toggle_offset.py; lambda = 0, 0.25,
0.5, 0.75, 1). No hold at H and no readout. Instrument and inputs are those of tools/run_public_toggle_check.py (and so
of the sweep); only the groups and their starts differ. One group per shard, one single-thread process per shard.
"""
import copy
import hashlib
import signal
import time
import traceback
from pathlib import Path

import numpy as np

import run_public_toggle_check as chk

sweep, pilot, prior, memory, engine, reference = chk.sweep, chk.pilot, chk.prior, chk.memory, chk.engine, chk.reference
PROFILE = 'public-toggle-offset'
PLAN = 'docs/public-toggle-fold.md'
CANDIDATE = 't030-r3.5-d012'
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
PULSE = 6


def plans():
    groups, stages, rows = [], [], []
    ct = sweep.target(30, 3.5)
    for gid, lam in enumerate(LAMBDAS):
        name = f't030-r3.5-d006-offset-lambda{lam:.2f}'
        group = dict(profile=PROFILE, group_id=gid, name=name, kind='probe', role='offset', candidate=CANDIDATE,
                     controls=ct, control_bits=pilot.control_bits(ct), pulse_steps=PULSE, theta_degrees=30,
                     radius=3.5, start_phase=0, perturbation_seed=None, offset_lambda=lam, start_index=gid,
                     stage_ids=[gid], row_ids=[2*gid, 2*gid + 1], status='not_started', mapping='not_read',
                     decoded_pair=[None, None], closed=False, observations=[])
        groups.append(group)
        stage = copy.deepcopy(prior.stage_plan(prior.WritePlan(0))[0])
        stage.update(profile=PROFILE, stage_id=gid, group_id=gid, group=name, local_stage_id=0, name='pulse',
                     controls=dict(ct), control_bits=pilot.control_bits(ct), nominal_controls=dict(ct), steps=PULSE,
                     historical_comparison_scope='not_applicable_fixed_toggle_exploration')
        stages.append(stage)
        for lane in (0, 1):
            rows.append(dict(profile=PROFILE, row_id=len(rows), group_id=gid, group=name, lane=lane,
                             source_reference_index=lane, source_reference_phase=0, offset_lambda=lam,
                             status='not_started', readout=None, decoded_bit=None))
    return groups, stages, rows


def load_starts(path, sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise pilot.InstrumentFailure('start file SHA-256 differs from the one given')
    with np.load(path) as z:
        lambdas, states = z['lambdas'], z['states']
    if tuple(float(x) for x in lambdas) != LAMBDAS or states.shape != (len(LAMBDAS), 2, 896) or states.dtype != np.float64:
        raise pilot.InstrumentFailure('start file does not match the fixed design')
    if not np.isfinite(states).all():
        raise pilot.InstrumentFailure('start file has non-finite values')
    return states


def run(args, backend_factory=prior.Backend, qualifier=pilot.qualify_inputs, clock=time.monotonic,
        rss=reference.rss_bytes):
    groups, stages, rows = plans()
    ids = sweep.shard_group_ids(len(groups), args.shard, args.shards)
    slots = sweep.shard_calls(groups, stages, ids)
    for c in slots:
        c['profile'] = PROFILE
    mine = [groups[g] for g in ids]
    recorder = sweep.Recorder(args.output_dir, clock, rss)
    budget = sweep.Budget(recorder, stages, slots)
    result = dict(profile=PROFILE, shard=args.shard, shards=args.shards, status='execution_failed',
                  outcome='not_evaluated', model_loaded=False, groups=mine, stages=stages, rows=rows, calls=slots,
                  initializations=0)
    handlers, active = {}, None

    def stop(signum, _frame):
        if signum == signal.SIGALRM:
            raise TimeoutError('internal wall cap')
        raise InterruptedError(f'signal {signum}')
    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        handlers[sig] = signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, sweep.MAX_SECONDS)
    try:
        starts = load_starts(args.starts, args.starts_sha256)
        recorder.json('design.json', dict(profile=PROFILE, rules=PLAN + ' (Part B)', lambdas=list(LAMBDAS),
                      pulse_steps=PULSE, starts=memory.binding(Path(args.starts)), shard=args.shard,
                      shards=args.shards, shard_group_ids=ids, all_groups=len(groups), shard_calls=len(slots),
                      artifact_envelope=sweep.artifact_envelope(stages, ids, groups)))
        recorder.json('planned-groups.json', mine); recorder.json('planned-calls.json', slots)
        here = Path(__file__).resolve()
        result['source'] = pilot.source_metadata(args, extra_sources=(
            here, Path(chk.__file__).resolve(), Path(sweep.__file__).resolve(), pilot.ROOT/PLAN,
            pilot.ROOT/'tools/prepare_public_toggle_offset.py'))
        recorder.json('source.json', result['source'])
        if any(v != '1' for v in pilot.STARTUP_THREADS.values()):
            raise pilot.InstrumentFailure('four thread limits must equal one before process startup')
        refs, provenance = qualifier(args); engine.shared.check_references(refs)
        recorder.json('qualified-inputs.json', provenance)
        recorder.npz('reference-cycles.npz', states=refs)
        if starts[-1].tobytes() != refs[:, 0].tobytes():
            raise pilot.InstrumentFailure('lambda = 1 start is not the phase-0 reference state')
        if args.preflight_only:
            result.update(status='preflight_passed', outcome='not_run_preflight')
        else:
            backend = pilot.guarded_backend_factory(backend_factory, require_runtime=True)(args, recorder, budget)
            result['model_loaded'] = True
            if hasattr(backend, 'bindings'):
                recorder.json('model-inputs.json', backend.bindings)
            for group in mine:
                active = group
                state = starts[group['start_index']].copy()
                recorder.state(f'group-{group["group_id"]:03d}-start.npy', state)
                result['initializations'] += 1
                group.update(status='running', initial_state_sha256=prior.array_hash(state))
                for sid in group['stage_ids']:
                    engine.execute_stage(backend, state, stages[sid], recorder, budget)
                for rid in group['row_ids']:
                    rows[rid].update(status='completed_no_readout')
                group.update(status='completed', terminal_state_sha256=prior.array_hash(state),
                             diagnostics_passed=all(stages[i]['cache_comparison']['passed'] for i in group['stage_ids']))
                pilot.close_group(group, result, recorder); active = None
            result.update(status='completed', outcome='run')
    except BaseException as exc:
        status = pilot.failure_status(exc)
        result.update(status=status, error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        if active is not None:
            active.update(status=status, error=result['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        recorder.finalizing = True
        for group in mine:
            if group['status'] == 'not_started':
                group.update(status='not_run', not_run_reason=result['status'])
            pilot.close_group(group, result, recorder)
        result.update(model_calls=budget.calls, call_counts=budget.counts,
                      elapsed_seconds=clock() - recorder.started, peak_rss_bytes=rss())
        recorder.stream.close()
        recorder.json('result.json', result, final=True)
        recorder.json('manifest.json', dict(files=[memory.binding(p) for p in sorted(recorder.directory.iterdir())
                        if p.is_file() and p.name != 'manifest.json']), final=True)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return result


def main(argv=None):
    p = sweep.parser()
    p.description = __doc__
    p.add_argument('--starts', type=Path, required=True)
    p.add_argument('--starts-sha256', required=True)
    result = run(p.parse_args(argv))
    return 0 if result['status'] in ('completed', 'preflight_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
