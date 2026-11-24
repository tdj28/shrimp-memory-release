#!/usr/bin/env python3
"""Basin map at H (docs/public-basin-map.md), one shard of an interleaved partition of one start set.

Each group holds two fixed starts, read from a start file written by tools/prepare_public_basin_map.py and bound by
its SHA-256, at H for 240 updates; the toggle sweep's instrument, guarded backend, recorder, budget, caps and neutral
readout are reused unchanged. Nothing observed selects a start, a duration or a retry.
"""
import copy
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import run_public_toggle_sweep as sweep

pilot, engine, prior, memory, reference = sweep.pilot, sweep.engine, sweep.prior, sweep.memory, sweep.reference
PROFILE = 'public-basin-map'
PLAN = 'docs/public-basin-map.md'
LABELS = ('ad', 'td')


def load_starts(path, sha256):
    if memory.binding(Path(path))['sha256'] != sha256:
        raise pilot.InstrumentFailure('start file hash differs from the bound value')
    with np.load(path) as z:
        starts = z['starts']
    if starts.ndim != 3 or starts.shape[1:] != (2, 896) or starts.dtype != np.float64 or not np.isfinite(starts).all():
        raise pilot.InstrumentFailure('start file must hold finite float64 starts of shape (groups, 2, 896)')
    return starts


def plans(n_groups, set_name):
    groups, stages, rows = [], [], []
    for gid in range(n_groups):
        name = f'{set_name}-{gid:04d}'
        groups.append(dict(profile=PROFILE, group_id=gid, name=name, kind='probe', start_set=set_name,
                           controls=dict(sweep.HOLD), control_bits=pilot.control_bits(sweep.HOLD), pulse_steps=0,
                           stage_ids=[gid], row_ids=[2*gid, 2*gid + 1], status='not_started', mapping='unavailable',
                           decoded_pair=[None, None], closed=False, observations=[]))
        stage = copy.deepcopy(prior.stage_plan(prior.WritePlan(0))[0])
        stage.update(profile=PROFILE, stage_id=gid, group_id=gid, group=name, local_stage_id=0, name='hold',
                     controls=dict(sweep.HOLD), control_bits=pilot.control_bits(sweep.HOLD),
                     nominal_controls=dict(sweep.HOLD), steps=sweep.HOLD_STEPS,
                     historical_comparison_scope='not_applicable_fixed_basin_map')
        stages.append(stage)
        for lane in (0, 1):
            rows.append(dict(profile=PROFILE, row_id=2*gid + lane, group_id=gid, group=name, lane=lane,
                             start_index=2*gid + lane, status='not_started', readout=None, label=None))
    return groups, stages, rows


def run(args, backend_factory=prior.Backend, qualifier=pilot.qualify_inputs, clock=time.monotonic,
        rss=reference.rss_bytes):
    starts = load_starts(args.starts, args.starts_sha256)
    groups, stages, rows = plans(starts.shape[0], args.start_set)
    ids = sweep.shard_group_ids(len(groups), args.shard, args.shards)
    slots = sweep.shard_calls(groups, stages, ids)
    for c in slots:
        c['profile'] = PROFILE
    mine = [groups[g] for g in ids]
    recorder = sweep.Recorder(args.output_dir, clock, rss)
    budget = sweep.Budget(recorder, stages, slots)
    result = dict(profile=PROFILE, start_set=args.start_set, shard=args.shard, shards=args.shards,
                  status='execution_failed', outcome='not_evaluated', model_loaded=False, groups=mine, stages=stages,
                  rows=rows, calls=slots, initializations=0)
    handlers, active = {}, None

    def stop(signum, _frame):
        if signum == signal.SIGALRM:
            raise TimeoutError('internal wall cap')
        raise InterruptedError(f'signal {signum}')
    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        handlers[sig] = signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, sweep.MAX_SECONDS)
    try:
        envelope = sweep.artifact_envelope(stages, ids, groups)
        recorder.json('design.json', dict(profile=PROFILE, rules=PLAN, start_set=args.start_set,
                      starts=str(args.starts), starts_sha256=args.starts_sha256, shard=args.shard, shards=args.shards,
                      shard_group_ids=ids, all_groups=len(groups), shard_calls=len(slots), hold=sweep.HOLD,
                      hold_steps=sweep.HOLD_STEPS, artifact_envelope=envelope))
        recorder.json('planned-groups.json', mine); recorder.json('planned-calls.json', slots)
        here = Path(__file__).resolve()
        result['source'] = pilot.source_metadata(args, extra_sources=(
            here, Path(sweep.__file__).resolve(), pilot.ROOT/PLAN, pilot.ROOT/'tools/prepare_public_basin_map.py'))
        recorder.json('source.json', result['source'])
        recorder.json('thread-environment.json', dict(startup=pilot.STARTUP_THREADS,
                                                      current={k: os.environ.get(k) for k in pilot.THREAD_KEYS}))
        if any(v != '1' for v in pilot.STARTUP_THREADS.values()):
            raise pilot.InstrumentFailure('four thread limits must equal one before process startup')
        if envelope['total_bytes'] >= sweep.WORKING_OUTPUT:
            raise pilot.ResourceLimit('native evidence estimate exceeds working cap')
        refs, provenance = qualifier(args); engine.shared.check_references(refs)
        recorder.json('qualified-inputs.json', provenance)
        recorder.npz('reference-cycles.npz', states=refs)
        if args.preflight_only:
            result.update(status='preflight_passed', outcome='not_run_preflight')
        else:
            budget.check(); started = clock()
            backend = pilot.guarded_backend_factory(backend_factory, require_runtime=True)(args, recorder, budget)
            result.update(model_loaded=True, cold_initialization_seconds=clock() - started)
            if hasattr(backend, 'bindings'):
                recorder.json('model-inputs.json', backend.bindings)
            for group in mine:
                active = group
                state = starts[group['group_id']].copy(); result['initializations'] += 1
                group.update(status='running', initial_state_sha256=prior.array_hash(state))
                recorder.event('group_started', group_id=group['group_id'], initial_sha256=group['initial_state_sha256'])
                engine.execute_stage(backend, state, stages[group['stage_ids'][0]], recorder, budget)
                saved, _, _ = prior.reconstruct(recorder.directory, stages[group['stage_ids'][0]])
                labels = []
                for rid in group['row_ids']:
                    row = rows[rid]; readout = pilot.neutral_readout(saved[1:, row['lane']], refs)
                    q = readout['qualified_label']
                    row.update(status='completed', readout=readout, label=None if q is None else LABELS.index(q))
                    labels.append(row['label'])
                group.update(status='completed', decoded_pair=labels, mapping='labels',
                             terminal_state_sha256=prior.array_hash(state),
                             terminal_lanes_equal=bool(np.array_equal(state[0], state[1])),
                             diagnostics_passed=bool(stages[group['stage_ids'][0]]['cache_comparison']['passed']))
                pilot.close_group(group, result, recorder); recorder.event('group_finished', group_id=group['group_id'])
                active = None
            if budget.calls != len(slots):
                raise pilot.InstrumentFailure('incomplete fixed call schedule')
            result.update(status='completed', outcome='labelled')
    except BaseException as exc:
        status = pilot.failure_status(exc)
        result.update(status=status, error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        if active is not None:
            active.update(status=status, error=result['error'])
            for rid in active['row_ids']:
                rows[rid]['status'] = status
        recorder.event('execution_failed', _final=True, status=status, error=result['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        recorder.finalizing = True
        for group in mine:
            if group['status'] == 'not_started':
                group.update(status='not_run', not_run_reason=result['status'])
                for rid in group['row_ids']:
                    rows[rid].update(status='not_run', not_run_reason=result['status'])
            pilot.close_group(group, result, recorder)
        result.update(model_calls=budget.calls, call_counts=budget.counts,
                      elapsed_seconds=clock() - recorder.started, peak_rss_bytes=rss())
        result['resource_limits_passed'] = (budget.calls <= len(slots) and result['elapsed_seconds'] < sweep.MAX_SECONDS
                                           and result['peak_rss_bytes'] < sweep.MAX_RSS
                                           and recorder.size() < sweep.MAX_OUTPUT)
        if not result['resource_limits_passed'] and result['status'] in ('completed', 'preflight_passed'):
            result.update(status='incomplete', outcome='resource_limit')
        recorder.event('finished', status=result['status']); recorder.stream.close()
        result['rows'] = [rows[r] for g in mine for r in g['row_ids']]
        result['stages'] = [stages[g['stage_ids'][0]] for g in mine]
        recorder.json('result.json', result, final=True)
        recorder.json('manifest.json', dict(files=[memory.binding(p) for p in sorted(recorder.directory.iterdir())
                        if p.is_file() and p.name != 'manifest.json']), final=True)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return result


def parser():
    p = sweep.parser()
    p.description = __doc__
    p.add_argument('--starts', type=Path, required=True)
    p.add_argument('--starts-sha256', required=True)
    p.add_argument('--start-set', choices=('M', 'Z', 'S', 'X'), required=True)
    return p


def main(argv=None):
    result = run(parser().parse_args(argv))
    return 0 if result['status'] in ('completed', 'preflight_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
