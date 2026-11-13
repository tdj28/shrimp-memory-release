#!/usr/bin/env python3
"""Fixed public toggle sweep (docs/public-toggle-sweep.md), one shard of an interleaved partition.

Reuses the toggle pilot's instrument unchanged: guarded native64 backend, recorder, neutral readout and pair mapping.
The candidate list is fixed here, before any model call; nothing observed selects a control, duration or retry.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import run_public_toggle_pilot as pilot

engine, prior, memory, reference, batch = pilot.engine, pilot.prior, pilot.memory, pilot.reference, pilot.batch
PROFILE = 'public-toggle-sweep'
HOLD = pilot.HOLD.copy()
ENDPOINTS = [x.copy() for x in pilot.ENDPOINTS]
DA = ENDPOINTS[1]['alpha'] - HOLD['alpha']
DT = ENDPOINTS[0]['log10_temperature'] - HOLD['log10_temperature']
DIRECTIONS = tuple(range(0, 360, 30))
RADII = (0.5, 1.0, 2.0, 3.5)
DWELLS = (12, 48, 120)
HOLD_STEPS = 240
MAX_SECONDS, MAX_RSS, MAX_VM = 4*3600, 8*1024**3, 16*1024**3
MAX_OUTPUT, FINAL_RESERVE = 1024**3, 32*1024**2
WORKING_OUTPUT = MAX_OUTPUT - FINAL_RESERVE


def target(theta_deg, radius):
    th = np.deg2rad(theta_deg)
    return dict(alpha=float(HOLD['alpha'] + radius*np.cos(th)*DA),
                log10_temperature=float(HOLD['log10_temperature'] + radius*np.sin(th)*DT))


def specs():
    """Fixed order: three baselines, then directions x radii x dwells."""
    out = [dict(name='baseline-hold', controls=HOLD, pulse=120, expected=[0, 1], theta=None, radius=0.0),
           dict(name='baseline-write0', controls=ENDPOINTS[0], pulse=120, expected=[0, 0], theta=90, radius=1.0),
           dict(name='baseline-write1', controls=ENDPOINTS[1], pulse=120, expected=[1, 1], theta=0, radius=1.0)]
    for theta in DIRECTIONS:
        for radius in RADII:
            for dwell in DWELLS:
                out.append(dict(name=f'sweep-t{theta:03d}-r{radius:.1f}-d{dwell:03d}', controls=target(theta, radius),
                                pulse=dwell, expected=None, theta=theta, radius=radius))
    return out


def plans():
    """All groups and stages (global ids); calls are built per shard by shard_calls()."""
    groups, stages, rows = [], [], []
    for gid, s in enumerate(specs()):
        group = dict(profile=PROFILE, group_id=gid, name=s['name'], kind='baseline' if gid < 3 else 'probe',
                     controls=dict(s['controls']), control_bits=pilot.control_bits(s['controls']),
                     pulse_steps=s['pulse'], theta_degrees=s['theta'], radius=s['radius'],
                     offset_units=dict(alpha=DA, log10_temperature=DT), expected_baseline_pair=s['expected'],
                     stage_ids=[2*gid, 2*gid + 1], row_ids=[2*gid, 2*gid + 1], status='not_started',
                     mapping='unavailable', decoded_pair=[None, None], closed=False, observations=[])
        groups.append(group)
        for local, (ct, steps) in enumerate(((s['controls'], s['pulse']), (HOLD, HOLD_STEPS))):
            stage = copy.deepcopy(prior.stage_plan(prior.WritePlan(0))[0])
            stage.update(profile=PROFILE, stage_id=len(stages), group_id=gid, group=s['name'], local_stage_id=local,
                         name='pulse' if local == 0 else 'hold', controls=dict(ct), control_bits=pilot.control_bits(ct),
                         nominal_controls=dict(ct), steps=steps,
                         historical_comparison_scope='not_applicable_fixed_toggle_exploration')
            stages.append(stage)
        for lane in (0, 1):
            rows.append(dict(profile=PROFILE, row_id=len(rows), group_id=gid, group=s['name'], lane=lane,
                             source_reference_index=lane, source_reference_phase=0, status='not_started',
                             readout=None, decoded_bit=None))
    return groups, stages, rows


def shard_group_ids(n_groups, shard, shards):
    if not (shards >= 1 and 0 <= shard < shards):
        raise ValueError('shard index out of range')
    return [g for g in range(n_groups) if g % shards == shard]


def shard_calls(groups, stages, ids):
    calls = []
    for gid in ids:
        for sid in groups[gid]['stage_ids']:
            st = stages[sid]
            for category, step, cached in [('prefix', 0, None), ('comparison', 0, True), ('comparison', 0, False)] + [
                    ('update', i, True) for i in range(1, st['steps'] + 1)]:
                calls.append(dict(profile=PROFILE, call=len(calls) + 1, group_id=gid, group=st['group'],
                                  stage_id=sid, stage=st['name'], category=category, step=step, cached=cached,
                                  status='not_started'))
    return calls


def artifact_envelope(stages, ids, groups):
    updates = sum(stages[s]['steps'] for g in ids for s in groups[g]['stage_ids'])
    n_stages = 2*len(ids)
    parts = dict(state_bytes=(updates + n_stages)*2*896*8, token_time_bytes=updates*24,
                 cache_bytes=n_stages*3*24*2*2*37*64*8, comparison_bytes=2*n_stages*(3*2*896 + 2*151936)*8,
                 container_allowance=8*1024**2, metadata_allowance=48*1024**2 + n_stages*256*1024,
                 failed_return_bytes=prior.FAILED_RETURN_BYTES + 16384)
    return dict(parts=parts, total_bytes=sum(parts.values()), compression_assumed=False)


def design(shard, shards):
    groups, stages, rows = plans()
    ids = shard_group_ids(len(groups), shard, shards)
    calls = shard_calls(groups, stages, ids)
    return dict(profile=PROFILE, evidence='fixed exploratory toggle sweep; docs/public-toggle-sweep.md',
                shard=shard, shards=shards, shard_group_ids=ids, all_groups=len(groups),
                all_calls=sum(len(shard_calls(groups, stages, [g])) for g in range(len(groups))),
                shard_calls=len(calls), directions_degrees=list(DIRECTIONS), radii=list(RADII), dwells=list(DWELLS),
                offset_units=dict(alpha=DA, log10_temperature=DT), hold=HOLD, endpoints=ENDPOINTS,
                lane_identity=['literal references[0,0]', 'literal references[1,0]'],
                readout=dict(hold_steps=HOLD_STEPS, closure_tolerance=1e-8, divisor_separation=1e-4,
                             reference_match_tolerance=1e-6, toggle_pair=[1, 0]),
                limits=dict(internal_seconds=MAX_SECONDS, rss_bytes=MAX_RSS, vm_bytes=MAX_VM, raw_bytes=MAX_OUTPUT,
                            final_reserve_bytes=FINAL_RESERVE, threads=1),
                artifact_envelope=artifact_envelope(stages, ids, groups),
                scope='Public numerical map only, two literal phase-zero starts per pulse. No language interface, '
                      'other-phase, repeated-toggle or reliability claim.')


class Recorder(pilot.Recorder):
    def admit(self, size, final=False):
        if self.size() + size > (MAX_OUTPUT if final or self.finalizing else WORKING_OUTPUT):
            raise pilot.ResourceLimit('raw artifact limit reached')


class Budget(pilot.mechanism.Budget):
    def __init__(self, recorder, stages, slots):
        super().__init__(recorder, 'fixed-public', 0, slots,
                         planner=lambda *_: ([], copy.deepcopy(stages), copy.deepcopy(slots)))

    def check(self):
        if self.recorder.clock() - self.recorder.started >= MAX_SECONDS:
            raise TimeoutError('internal wall cap')
        if self.recorder.rss() >= MAX_RSS:
            raise pilot.ResourceLimit('RSS cap')
        if self.recorder.size() > WORKING_OUTPUT:
            raise pilot.ResourceLimit('working raw cap')


def run(args, backend_factory=prior.Backend, qualifier=pilot.qualify_inputs, clock=time.monotonic,
        rss=reference.rss_bytes, install_signals=True, require_runtime=True):
    groups, stages, rows = plans()
    ids = shard_group_ids(len(groups), args.shard, args.shards)
    slots = shard_calls(groups, stages, ids)
    mine = [groups[g] for g in ids]
    recorder = Recorder(args.output_dir, clock, rss)
    budget = Budget(recorder, stages, slots)
    result = dict(profile=PROFILE, shard=args.shard, shards=args.shards, status='execution_failed',
                  outcome='not_evaluated', model_loaded=False, groups=mine, stages=stages, rows=rows, calls=slots,
                  initializations=0, baseline_qualified=None, toggle_candidate_group_ids=[])
    handlers, state, active = {}, None, None
    if install_signals:
        def stop(signum, _frame):
            if signum == signal.SIGALRM:
                raise TimeoutError('internal wall cap')
            raise InterruptedError(f'signal {signum}')
        for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    try:
        payload = design(args.shard, args.shards)
        recorder.json('design.json', payload); recorder.json('planned-groups.json', mine)
        recorder.json('planned-stages.json', [stages[s] for g in mine for s in g['stage_ids']])
        recorder.json('planned-rows.json', [rows[r] for g in mine for r in g['row_ids']])
        recorder.json('planned-calls.json', slots)
        here = Path(__file__).resolve()
        result['source'] = pilot.source_metadata(args, extra_sources=(
            here, pilot.ROOT/'docs/public-toggle-sweep.md', pilot.ROOT/'tests/test_public_toggle_sweep.py'))
        recorder.json('source.json', result['source'])
        recorder.json('thread-environment.json', dict(startup=pilot.STARTUP_THREADS,
                                                      current={k: os.environ.get(k) for k in pilot.THREAD_KEYS}))
        if any(v != '1' for v in pilot.STARTUP_THREADS.values()):
            raise pilot.InstrumentFailure('four thread limits must equal one before process startup')
        if payload['artifact_envelope']['total_bytes'] >= WORKING_OUTPUT:
            raise pilot.ResourceLimit('native evidence estimate exceeds working cap')
        refs, provenance = qualifier(args); engine.shared.check_references(refs)
        recorder.json('qualified-inputs.json', provenance)
        recorder.npz('reference-cycles.npz', states=refs)
        initial = refs[:, 0].copy(); recorder.state('reference-starts.npy', initial)
        recorder.json('input-preflight.json', dict(status='passed', model_calls=0, model_loaded=False,
                      initial_sha256=prior.array_hash(initial), provenance=provenance))
        if getattr(args, 'preflight_only', False):
            result.update(status='preflight_passed', outcome='not_run_preflight')
        else:
            budget.check(); started = clock()
            backend = pilot.guarded_backend_factory(backend_factory, require_runtime=require_runtime)(args, recorder, budget)
            result.update(model_loaded=True, cold_initialization_seconds=clock() - started)
            if hasattr(backend, 'bindings'):
                recorder.json('model-inputs.json', backend.bindings)
            for group in mine:
                active = group; state = initial.copy(); result['initializations'] += 1
                group.update(status='running', initial_state_sha256=prior.array_hash(state))
                recorder.event('group_started', group_id=group['group_id'], initial_sha256=group['initial_state_sha256'])
                for rid in group['row_ids']:
                    rows[rid]['status'] = 'running'
                for sid in group['stage_ids']:
                    engine.execute_stage(backend, state, stages[sid], recorder, budget)
                saved, _, _ = prior.reconstruct(recorder.directory, stages[group['stage_ids'][-1]])
                for rid in group['row_ids']:
                    row = rows[rid]; readout = pilot.neutral_readout(saved[1:, row['lane']], refs)
                    label = readout['qualified_label']
                    row.update(status='completed', readout=readout,
                               decoded_bit=None if label is None else ('ad', 'td').index(label))
                bits = [rows[rid]['decoded_bit'] for rid in group['row_ids']]
                group.update(status='completed', decoded_pair=bits, mapping=pilot.classify_pair(bits),
                             terminal_state_sha256=prior.array_hash(state),
                             diagnostics_passed=all(stages[i]['cache_comparison']['passed'] for i in group['stage_ids']))
                pilot.close_group(group, result, recorder); recorder.event('group_finished', group_id=group['group_id'])
                active = None
            if budget.calls != len(slots):
                raise pilot.InstrumentFailure('incomplete fixed call schedule')
            base = [g for g in mine if g['kind'] == 'baseline']
            baselines = all(g['decoded_pair'] == g['expected_baseline_pair'] for g in base) if base else None
            toggles = [g['group_id'] for g in mine if g['kind'] == 'probe' and g['mapping'] == 'toggle']
            outcome = ('baseline_unqualified' if baselines is False else 'toggle_observed' if toggles else
                       'toggle_unresolved' if any(g['mapping'] == 'unresolved' for g in mine if g['kind'] == 'probe')
                       else 'no_toggle_observed')
            result.update(status='completed', outcome=outcome, baseline_qualified=baselines,
                          toggle_candidate_group_ids=toggles)
    except BaseException as exc:
        status = pilot.failure_status(exc)
        result.update(status=status, error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        if active is not None:
            active.update(status=status, error=result['error'])
            for rid in active['row_ids']:
                rows[rid]['status'] = status
        recorder.event('execution_failed', _final=True, status=status, error=result['error'])
    finally:
        if install_signals:
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
        result['resource_limits_passed'] = (budget.calls <= len(slots) and result['elapsed_seconds'] < MAX_SECONDS
                                           and result['peak_rss_bytes'] < MAX_RSS and recorder.size() < MAX_OUTPUT)
        if not result['resource_limits_passed'] and result['status'] in ('completed', 'preflight_passed'):
            result.update(status='incomplete', outcome='resource_limit')
        recorder.event('finished', status=result['status']); recorder.stream.close()
        recorder.json('result.json', result, final=True)
        recorder.json('manifest.json', dict(files=[memory.binding(p) for p in sorted(recorder.directory.iterdir())
                        if p.is_file() and p.name != 'manifest.json']), final=True)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return result


def parser():
    p = pilot.parser()
    p.description = __doc__
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--shards', type=int, required=True)
    return p


def main(argv=None):
    result = run(parser().parse_args(argv))
    return 0 if result['status'] in ('completed', 'preflight_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
