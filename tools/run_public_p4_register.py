#!/usr/bin/env python3
"""Second register, period-4 window (docs/public-p4-register.md, approved 12 November 2026: "Proceed").

Runs the groups of a bound design file (stage R: one relaxation group; stage W: hold and write groups) from the states
of a bound start file, both lanes together, one group per shard, one single-thread process per shard. Model calls go
through the toggle check's instrument unchanged (same backend, inputs, guards and call accounting); only the groups,
their controls and their starts differ. No readout here: period-4 readouts are recomputed from saved states by
tools/analyze_public_p4_register.py.
"""
import copy
import hashlib
import json
import signal
import time
import traceback
from pathlib import Path

import numpy as np

import run_public_toggle_check as chk

sweep, pilot, prior, memory, engine, reference = chk.sweep, chk.pilot, chk.prior, chk.memory, chk.engine, chk.reference
PROFILE = 'public-p4-register'
PLAN = 'docs/public-p4-register.md'
ALPHA = 0.06484


def load_bound(path, sha256, what):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise pilot.InstrumentFailure(f'{what} SHA-256 differs from the one given')
    return raw


def validate_design(design):
    if design.get('profile') != PROFILE or design.get('stage_set') not in ('R', 'W'):
        raise pilot.InstrumentFailure('design profile or stage set differs')
    for g in design['groups']:
        if not g['stages']:
            raise pilot.InstrumentFailure('group without stages')
        for s in g['stages']:
            c = s['controls']
            if set(c) != {'alpha', 'log10_temperature'} or c['alpha'] != ALPHA:
                raise pilot.InstrumentFailure('controls must fix alpha at 0.06484')
            if not (0.06 < c['log10_temperature'] < 0.07) or type(s['steps']) is not int or not 0 < s['steps'] <= 400:
                raise pilot.InstrumentFailure('controls or steps outside the plan')
    return design


def plans(design):
    groups, stages, rows = [], [], []
    for gid, g in enumerate(design['groups']):
        sids = []
        for local, s in enumerate(g['stages']):
            stage = copy.deepcopy(prior.stage_plan(prior.WritePlan(0))[0])
            ct = dict(s['controls'])
            stage.update(profile=PROFILE, stage_id=len(stages), group_id=gid, group=g['name'], local_stage_id=local,
                         name=s['name'], controls=ct, control_bits=pilot.control_bits(ct), nominal_controls=dict(ct),
                         steps=s['steps'], historical_comparison_scope='not_applicable_p4_register')
            sids.append(len(stages))
            stages.append(stage)
        ct0 = dict(g['stages'][0]['controls'])
        groups.append(dict(profile=PROFILE, group_id=gid, name=g['name'], kind='probe', role=design['stage_set'],
                           controls=ct0, control_bits=pilot.control_bits(ct0),
                           pulse_steps=g['stages'][0]['steps'], start_index=g['start_index'], stage_ids=sids,
                           row_ids=[2*gid, 2*gid + 1], status='not_started', mapping='not_read',
                           decoded_pair=[None, None], closed=False, observations=[]))
        for lane in (0, 1):
            rows.append(dict(profile=PROFILE, row_id=len(rows), group_id=gid, group=g['name'], lane=lane,
                             status='not_started', readout=None, decoded_bit=None))
    return groups, stages, rows


def run(args, backend_factory=prior.Backend, qualifier=pilot.qualify_inputs, clock=time.monotonic,
        rss=reference.rss_bytes):
    raw = load_bound(args.design, args.design_sha256, 'design file')
    design = validate_design(json.loads(raw))
    groups, stages, rows = plans(design)
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
        load_bound(args.starts, args.starts_sha256, 'start file')
        with np.load(args.starts) as z:
            starts = z['states']
        if starts.ndim != 3 or starts.shape[1:] != (2, 896) or starts.dtype != np.float64 or not np.isfinite(starts).all():
            raise pilot.InstrumentFailure('start file shape, type or values invalid')
        if max(g['start_index'] for g in groups) >= starts.shape[0]:
            raise pilot.InstrumentFailure('design refers to a missing start')
        recorder.json('design.json', dict(profile=PROFILE, rules=PLAN, stage_set=design['stage_set'],
                      design=memory.binding(Path(args.design)), starts=memory.binding(Path(args.starts)),
                      shard=args.shard, shards=args.shards, shard_group_ids=ids, all_groups=len(groups),
                      shard_calls=len(slots), artifact_envelope=sweep.artifact_envelope(stages, ids, groups)))
        recorder.json('planned-groups.json', mine); recorder.json('planned-calls.json', slots)
        here = Path(__file__).resolve()
        result['source'] = pilot.source_metadata(args, extra_sources=(
            here, Path(chk.__file__).resolve(), Path(sweep.__file__).resolve(), pilot.ROOT/PLAN,
            pilot.ROOT/'tools/prepare_public_p4_register.py'))
        recorder.json('source.json', result['source'])
        if any(v != '1' for v in pilot.STARTUP_THREADS.values()):
            raise pilot.InstrumentFailure('four thread limits must equal one before process startup')
        refs, provenance = qualifier(args); engine.shared.check_references(refs)   # instrument input checks only
        recorder.json('qualified-inputs.json', provenance)
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
    p.add_argument('--design', type=Path, required=True)
    p.add_argument('--design-sha256', required=True)
    p.add_argument('--starts', type=Path, required=True)
    p.add_argument('--starts-sha256', required=True)
    result = run(p.parse_args(argv))
    return 0 if result['status'] in ('completed', 'preflight_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
