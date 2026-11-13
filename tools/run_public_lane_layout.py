#!/usr/bin/env python3
"""Lane-layout check (docs/public-lane-layout-check.md, fixed 13 Nov 2026 before any call).

Sixteen fixed groups: four operations (W = 12 idle updates at H, E0 and E1 = the audited 120-update writes, T = the
30-degree, radius-3.5, 12-update toggle), each followed by 240 updates at H and the neutral readout, under four
layouts of the phase-0 reference starts: P (cycle 0, cycle 1), R (cycle 1, cycle 0), D0 (cycle 0 twice) and D1
(cycle 1 twice). The pair instrument of tools/run_public_toggle_check.py is unchanged; move controls come from
tools/run_public_adder.py; only the groups and the placement of the starts differ. One group per shard, one
single-thread process per shard.
"""
import copy
import signal
import time
import traceback
from pathlib import Path

import numpy as np

import run_public_adder as adder
import run_public_toggle_check as chk

sweep, pilot, prior, memory, engine, reference = chk.sweep, chk.pilot, chk.prior, chk.memory, chk.engine, chk.reference
PROFILE = 'public-lane-layout'
PLAN = 'docs/public-lane-layout-check.md'
OPERATIONS = ('W', 'E0', 'E1', 'T')
LAYOUTS = {'P': (0, 1), 'R': (1, 0), 'D0': (0, 0), 'D1': (1, 1)}
EXPECTED = {'W': lambda a, b: (a, b), 'E0': lambda a, b: (0, 0), 'E1': lambda a, b: (1, 1),
            'T': lambda a, b: (1 - a, 1 - b)}


def specs():
    return [dict(name=f'{op}-{lay}', operation=op, layout=lay, starts=LAYOUTS[lay],
                 expected=list(EXPECTED[op](*LAYOUTS[lay]))) for op in OPERATIONS for lay in LAYOUTS]


def plans():
    groups, stages, rows = [], [], []
    for gid, s in enumerate(specs()):
        ct, steps = adder.controls(s['operation'])
        group = dict(profile=PROFILE, group_id=gid, name=s['name'], kind='probe', role='lane-layout',
                     operation=s['operation'], layout=s['layout'], start_cycles=list(s['starts']),
                     expected_pair=s['expected'], controls=ct, control_bits=pilot.control_bits(ct), pulse_steps=steps,
                     start_phase=0, perturbation_seed=None, stage_ids=[2*gid, 2*gid + 1],
                     row_ids=[2*gid, 2*gid + 1], status='not_started', mapping='unavailable',
                     decoded_pair=[None, None], closed=False, observations=[])
        groups.append(group)
        for local, (c, n) in enumerate(((ct, steps), (sweep.HOLD, sweep.HOLD_STEPS))):
            stage = copy.deepcopy(prior.stage_plan(prior.WritePlan(0))[0])
            stage.update(profile=PROFILE, stage_id=len(stages), group_id=gid, group=s['name'], local_stage_id=local,
                         name='move' if local == 0 else 'hold', controls=dict(c), control_bits=pilot.control_bits(c),
                         nominal_controls=dict(c), steps=n,
                         historical_comparison_scope='not_applicable_fixed_toggle_exploration')
            stages.append(stage)
        for lane in (0, 1):
            rows.append(dict(profile=PROFILE, row_id=len(rows), group_id=gid, group=s['name'], lane=lane,
                             source_reference_index=s['starts'][lane], source_reference_phase=0,
                             status='not_started', readout=None, decoded_bit=None))
    return groups, stages, rows


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
        recorder.json('design.json', dict(profile=PROFILE, rules=PLAN, specs=specs(), shard=args.shard,
                      shards=args.shards, shard_group_ids=ids, all_groups=len(groups), shard_calls=len(slots),
                      artifact_envelope=sweep.artifact_envelope(stages, ids, groups)))
        recorder.json('planned-groups.json', mine); recorder.json('planned-calls.json', slots)
        here = Path(__file__).resolve()
        result['source'] = pilot.source_metadata(args, extra_sources=(
            here, Path(chk.__file__).resolve(), Path(sweep.__file__).resolve(), Path(adder.__file__).resolve(),
            pilot.ROOT/PLAN))
        recorder.json('source.json', result['source'])
        if any(v != '1' for v in pilot.STARTUP_THREADS.values()):
            raise pilot.InstrumentFailure('four thread limits must equal one before process startup')
        refs, provenance = qualifier(args); engine.shared.check_references(refs)
        recorder.json('qualified-inputs.json', provenance)
        recorder.npz('reference-cycles.npz', states=refs)
        if args.preflight_only:
            result.update(status='preflight_passed', outcome='not_run_preflight')
        else:
            backend = pilot.guarded_backend_factory(backend_factory, require_runtime=True)(args, recorder, budget)
            result['model_loaded'] = True
            if hasattr(backend, 'bindings'):
                recorder.json('model-inputs.json', backend.bindings)
            for group in mine:
                active = group
                a, b = group['start_cycles']
                state = np.stack([refs[a, 0], refs[b, 0]]).copy()
                recorder.state(f'group-{group["group_id"]:03d}-start.npy', state)
                result['initializations'] += 1
                group.update(status='running', initial_state_sha256=prior.array_hash(state))
                for sid in group['stage_ids']:
                    engine.execute_stage(backend, state, stages[sid], recorder, budget)
                saved, _, _ = prior.reconstruct(recorder.directory, stages[group['stage_ids'][-1]])
                for rid in group['row_ids']:
                    row = rows[rid]; readout = pilot.neutral_readout(saved[1:, row['lane']], refs)
                    label = readout['qualified_label']
                    row.update(status='completed', readout=readout,
                               decoded_bit=None if label is None else ('ad', 'td').index(label))
                bits = [rows[rid]['decoded_bit'] for rid in group['row_ids']]
                group.update(status='completed', decoded_pair=bits, matches_expected=bits == group['expected_pair'],
                             mapping=pilot.classify_pair(bits), terminal_state_sha256=prior.array_hash(state),
                             diagnostics_passed=all(stages[i]['cache_comparison']['passed'] for i in group['stage_ids']))
                pilot.close_group(group, result, recorder); active = None
            result.update(status='completed', outcome='checked')
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
    result = run(p.parse_args(argv))
    return 0 if result['status'] in ('completed', 'preflight_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
