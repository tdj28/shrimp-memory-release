#!/usr/bin/env python3
"""One fixed, public phase-zero toggle feasibility panel; no adaptive controls."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

THREAD_KEYS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
STARTUP_THREADS = {key: os.environ.get(key) for key in THREAD_KEYS}

import numpy as np
import run_public_mechanism as mechanism

engine, prior, memory = mechanism.engine, mechanism.prior, mechanism.memory
reference, batch = mechanism.reference, mechanism.batch
ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'public-toggle-pilot'
MAX_CALLS, MAX_UPDATES = 3258, 3192
MAX_SECONDS, MAX_RSS, MAX_VM = 3600, 8*1024**3, 16*1024**3
MAX_OUTPUT, FINAL_RESERVE = 512*1024**2, 32*1024**2
WORKING_OUTPUT = MAX_OUTPUT-FINAL_RESERVE
PINS = {'references_sha256': 'd96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0',
        'reference_manifest_sha256': '17b8e826a63a01d442877b6cf837272f760f49f8f4ba26aa6192294f47b6853f',
        'recovery_results_sha256': '5de613795c4bc10e0584072fd7cb12586de4db869286d15cbbdc7f57dab37ef5'}
HOLD = mechanism.HOLD.copy()
ENDPOINTS = [x.copy() for x in mechanism.ENDPOINTS]


class ResourceLimit(MemoryError):
    """A preserved local resource failure, never a scientific unresolved row."""


class InstrumentFailure(ValueError):
    pass


def _resource_error(error):
    text = str(error).lower()
    return isinstance(error, RuntimeError) and any(x in text for x in
        ('raw cap', 'raw budget', 'raw artifact', 'artifact limit'))


def guarded_backend_factory(factory=prior.Backend, *, require_runtime=True):
    """Reuse the mechanism engine with guards on run-local objects only.

    Both comparison returns are saved by the existing evaluator. A failed
    comparison prevents the following update claim, without spending a call.
    Legacy raw-cap RuntimeError exceptions become ResourceLimit (incomplete).
    This factory also works with mechanism.run and its fixed-plan injection API.
    """
    def create(args, recorder, budget):
        for obj, name in ((recorder, 'admit'), (budget, 'check')):
            original = getattr(obj, name)
            def guarded(*a, _original=original, **kw):
                try:
                    return _original(*a, **kw)
                except RuntimeError as exc:
                    if _resource_error(exc):
                        raise ResourceLimit(str(exc)) from exc
                    raise
            setattr(obj, name, guarded)
        backend = factory(args, recorder, budget)
        if require_runtime:
            patch = backend.patch
            norms, rotaries = patch['norms'], patch['rotaries']
            ready = dict(norms=norms, rotaries=rotaries,
                         torch_threads=backend.torch.get_num_threads(),
                         torch_interop_threads=backend.torch.get_num_interop_threads(),
                         mode=backend.loop.mode,
                         thread_environment={k: os.environ.get(k) for k in THREAD_KEYS})
            recorder.json('backend-readiness.json', ready)
            if (len(norms) != 49 or len(set(norms)) != 49 or rotaries != ['model.rotary_emb']
                    or ready['torch_threads'] != 1 or ready['torch_interop_threads'] != 1
                    or ready['mode'] != 'native64'
                    or any(v != '1' for v in ready['thread_environment'].values())):
                raise InstrumentFailure('native64 patch/thread readiness failed')
        original_claim = budget.claim

        class Guard:
            comparison = None
            cached_comparison = None

            def __getattr__(self, name):
                return getattr(backend, name)

            def prepare(self, controls):
                self.comparison = self.cached_comparison = None
                return backend.prepare(controls)

            def forward(self, state, cached):
                returned = backend.forward(state, cached)
                slot = budget.slots[budget.calls-1]
                if slot['category'] == 'comparison':
                    pair = tuple(np.asarray(x).copy() for x in returned[:2])
                    if cached:
                        self.cached_comparison = pair
                    else:
                        if self.cached_comparison is None:
                            raise InstrumentFailure('full comparison preceded cached comparison')
                        self.comparison = batch.compare(self.cached_comparison, pair)
                return returned

        guard = Guard()
        def claim(label, category='prefix'):
            if category == 'update' and not (guard.comparison and guard.comparison['passed']):
                recorder.event('update_blocked', _final=True, next_call=budget.calls+1,
                               stage_id=budget.stage_id, comparison=guard.comparison)
                raise InstrumentFailure('cached/full comparison failed before update')
            return original_claim(label, category)
        budget.claim = claim
        return guard
    return create


def control_bits(controls):
    return {k: dict(value=v, hex=float(v).hex(), bits=int(np.asarray(v, dtype=np.float64).view(np.uint64)))
            for k, v in controls.items()}


def plans():
    """Fixed order; this function has no state, readout, label or user-control input."""
    specs = [('baseline-hold', HOLD, 120, [0, 1]),
             ('baseline-write0', ENDPOINTS[0], 120, [0, 0]),
             ('baseline-write1', ENDPOINTS[1], 120, [1, 1])]
    da = ENDPOINTS[1]['alpha']-HOLD['alpha']
    dt = ENDPOINTS[0]['log10_temperature']-HOLD['log10_temperature']
    for sa, st in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        name = ('m' if sa < 0 else 'p')+('m' if st < 0 else 'p')
        controls = dict(alpha=HOLD['alpha']+sa*da,
                        log10_temperature=HOLD['log10_temperature']+st*dt)
        for duration in (12, 36):
            specs.append((f'diagonal-{name}-{duration}', controls, duration, None))
    groups, stages, rows, calls = [], [], [], []
    for gid, (name, controls, duration, expected) in enumerate(specs):
        group = dict(profile=PROFILE, group_id=gid, name=name, kind='baseline' if gid < 3 else 'probe',
                     controls=controls.copy(), control_bits=control_bits(controls), pulse_steps=duration,
                     expected_baseline_pair=expected, stage_ids=[2*gid, 2*gid+1],
                     row_ids=[2*gid, 2*gid+1], status='not_started', mapping='unavailable',
                     decoded_pair=[None, None], closed=False)
        groups.append(group)
        for local, (ct, steps) in enumerate(((controls, duration), (HOLD, 240))):
            stage = copy.deepcopy(prior.stage_plan(prior.WritePlan(0))[0])
            stage.update(profile=PROFILE, stage_id=len(stages), group_id=gid, group=name,
                         local_stage_id=local, name='pulse' if local == 0 else 'hold',
                         controls=ct.copy(), control_bits=control_bits(ct), nominal_controls=ct.copy(),
                         steps=steps, historical_comparison_scope='not_applicable_fixed_toggle_exploration')
            stages.append(stage)
            for category, step, cached in [('prefix', 0, None), ('comparison', 0, True), ('comparison', 0, False)]+[
                    ('update', i, True) for i in range(1, steps+1)]:
                calls.append(dict(profile=PROFILE, call=len(calls)+1, group_id=gid, group=name,
                                  stage_id=stage['stage_id'], stage=stage['name'], category=category,
                                  step=step, cached=cached, status='not_started'))
        for lane in (0, 1):
            rows.append(dict(profile=PROFILE, row_id=len(rows), group_id=gid, group=name,
                             lane=lane, source_reference_index=lane, source_reference_phase=0,
                             status='not_started', readout=None, decoded_bit=None))
    if len(calls) != MAX_CALLS or sum(s['steps'] for s in stages) != MAX_UPDATES:
        raise ValueError('fixed toggle ledger changed')
    return groups, stages, rows, calls


def classify_pair(bits):
    if len(bits) != 2 or any(x is not None and (type(x) is not int or x not in (0, 1)) for x in bits):
        raise ValueError('neutral two-lane bit pair required')
    if None in bits:
        return 'unresolved'
    return {(0, 1): 'identity', (1, 0): 'toggle', (0, 0): 'reset0', (1, 1): 'reset1'}[tuple(bits)]


def neutral_readout(trajectory, refs):
    result = memory.neutral_readout(trajectory, refs)
    result['all_reference_alignments'] = [memory.hold_test.separation_gate(trajectory[-6:], cycle)
                                          for cycle in refs]
    return result


def artifact_envelope():
    # Native uncompressed payloads, with deliberately generous separate metadata.
    parts = dict(state_bytes=(MAX_UPDATES+22)*2*896*8,
                 token_time_bytes=MAX_UPDATES*24,
                 cache_bytes=22*3*24*2*2*37*64*8,
                 comparison_bytes=44*(3*2*896+2*151936)*8,
                 container_allowance=8*1024**2, metadata_allowance=48*1024**2,
                 failed_return_bytes=prior.FAILED_RETURN_BYTES+16384)
    return dict(parts=parts, total_bytes=sum(parts.values()), compression_assumed=False,
                scope='All states/tokens/times, scalar and expanded caches, full diagnostic/failure probabilities; successful-update probabilities are hash-only.')


def design():
    groups, stages, rows, calls = plans()
    return dict(profile=PROFILE, evidence='fixed exploratory phase-zero feasibility pilot',
                groups=groups, stages=stages, rows=rows,
                schedule=dict(groups=11, stages=22, terminal_rows=22, batch_size=2, batch_updates=MAX_UPDATES,
                              row_updates=2*MAX_UPDATES, prefix_calls=22, comparison_calls=44,
                              total_calls=len(calls), within_group_carries=11, between_group_resets=10),
                lane_identity=['literal references[0,0]', 'literal references[1,0]'],
                corner_offsets=dict(alpha=ENDPOINTS[1]['alpha']-HOLD['alpha'],
                                    log10_temperature=ENDPOINTS[0]['log10_temperature']-HOLD['log10_temperature']),
                readout=dict(hold_steps=240, closure_indices=[235, 240], previous_indices=[229, 234],
                             closure_tolerance=1e-8, proper_divisors=[1, 2, 3], divisor_separation=1e-4,
                             reference_match_tolerance=1e-6, toggle_pair=[1, 0]),
                limits=dict(model_calls=MAX_CALLS, internal_seconds=MAX_SECONDS, rss_bytes=MAX_RSS,
                            vm_bytes=MAX_VM, raw_bytes=MAX_OUTPUT, final_reserve_bytes=FINAL_RESERVE, threads=1),
                artifact_envelope=artifact_envelope(),
                scope='Public numerical map only. No language interface, causal hysteresis, repeated-toggle reliability or other-phase claim.')


class Recorder(mechanism.Recorder):
    def admit(self, size, final=False):
        if self.size()+size > (MAX_OUTPUT if final or self.finalizing else WORKING_OUTPUT):
            raise ResourceLimit('raw artifact limit reached')

    def npz(self, name, final=False, **arrays):
        # The inherited evaluator omits final=True for failed-call arrays.
        # These are the precise payload for which claim() reserved space.
        return super().npz(name, final=final or (name.startswith('call-') and '-failed.npz' in name), **arrays)

    def json(self, name, value, final=False):
        if name.startswith('group-') and name.endswith('.json') and '-summary' not in name:
            raw = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
            self.admit(len(raw), final)
            path = self.directory/name
            if path.exists():
                raise FileExistsError(path)
            temporary = self.directory/(name+'.pending')
            with temporary.open('xb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            os.link(temporary, path); temporary.unlink()
            return memory.binding(path)
        return super().json(name, value, final)


class Budget(mechanism.Budget):
    def __init__(self, recorder, stages, slots):
        super().__init__(recorder, 'fixed-public', 0, slots,
                         planner=lambda *_: ([], copy.deepcopy(stages), copy.deepcopy(slots)))

    def check(self):
        if self.recorder.clock()-self.recorder.started >= MAX_SECONDS:
            raise TimeoutError('internal wall cap')
        if self.recorder.rss() >= MAX_RSS:
            raise ResourceLimit('RSS cap')
        if self.recorder.size() > WORKING_OUTPUT:
            raise ResourceLimit('working raw cap')


def failure_status(exc):
    if isinstance(exc, (TimeoutError, InterruptedError, MemoryError, KeyboardInterrupt)) or _resource_error(exc):
        return 'incomplete'
    return 'instrument_failed' if isinstance(exc, ValueError) else 'execution_failed'


def source_metadata(args, extra_sources=()):
    modules = (mechanism, engine, engine.continuation, engine.shared, prior, memory,
               engine.contract, memory.writes, memory.recovery, memory.hold_test,
               reference, batch, prior.historical_source, prior.trajectory_source)
    paths = [Path(m.__file__) for m in modules]+[Path(__file__), ROOT/'tests/test_public_toggle_pilot.py',
             ROOT/'docs/public-toggle-pilot.md', ROOT/'docs/receipts/p6-write-path-design.json',
             ROOT/'pyproject.toml', ROOT/'uv.lock']+list(extra_sources)
    for name in ('tools/audit_public_toggle_pilot.py', 'tests/test_public_toggle_pilot_audit.py',
                 'tools/audit_public_repair_half_pulse.py'):
        path = ROOT/name
        if path.is_file():
            paths.append(path)
    return dict(utc=reference.utc_now(), source_commit=subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        working_tree_status=subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True),
        files=[{**memory.binding(p), 'repository_path': str(p.relative_to(ROOT))} for p in dict.fromkeys(paths)],
        arguments={k: str(v) for k, v in vars(args).items()})


def qualify_inputs(args):
    if any(getattr(args, key) != value for key, value in PINS.items()):
        raise ValueError('original reference/recovery pins required')
    refs, provenance = memory.qualify_inputs(args)
    engine.shared.check_references(refs)
    return refs, provenance


def close_group(group, result, recorder):
    if group['closed']:
        return
    stages, slots = result['stages'], result['calls']
    for rid in group['row_ids']:
        recorder.json(f'row-{rid:02d}-completed.json', result['rows'][rid], final=True)
    summary = dict(group_id=group['group_id'], status=group['status'], mapping=group['mapping'],
                   rows=[result['rows'][i] for i in group['row_ids']],
                   stages=[stages[i] for i in group['stage_ids']],
                   calls=[s for s in slots if s['group_id'] == group['group_id']])
    item = recorder.json(f'group-{group["group_id"]:02d}-summary.json', summary, final=True)
    artifacts = engine.shared.immutable_group_artifacts(recorder.directory, group, stages, slots)
    artifacts += [memory.binding(recorder.directory/f'stage-{sid:02d}-observation.json')
                  for sid in group['stage_ids'] if (recorder.directory/f'stage-{sid:02d}-observation.json').is_file()]
    group.update(closed=True, model_calls=sum(s['status'] != 'not_started' for s in summary['calls']),
                 artifacts=artifacts+[item])
    recorder.json(f'group-{group["group_id"]:02d}.json', group, final=True)


def run_fixed(args, *, profile, plan_factory, design_factory, qualifier, observe_stage=None,
              extra_sources=(), backend_factory=prior.Backend, clock=time.monotonic,
              rss=reference.rss_bytes, install_signals=True, require_runtime=True):
    """Shared closure adapter for exactly the two fixed public pilots.

    plan_factory and design_factory take no arguments. The toggle plan returns
    groups/stages/rows/calls; the path plan returns groups/stages/calls. The
    optional observer receives immutable saved states, references and a stage;
    it cannot select or alter any later controls. No model or control CLI is added.
    """
    expected = {PROFILE: (11, 22, 3258, 3192), 'public-path-comparison': (3, 9, 2907, 2880)}
    if profile not in expected:
        raise ValueError('only the two fixed public pilot profiles are supported')
    planned = plan_factory()
    if len(planned) == 4:
        groups, stages, rows, slots = planned
    else:
        groups, stages, slots = planned; rows = []
        for group in groups:
            group['row_ids'] = [len(rows), len(rows)+1]
            for lane in (0, 1):
                rows.append(dict(profile=profile, row_id=len(rows), group_id=group['group_id'],
                                 group=group['name'], lane=lane, source_reference_index=lane,
                                 source_reference_phase=0, original_history_index=lane,
                                 reference_label_scope='Original P-history indices; not H memory bits. decoded_bit is a legacy numeric-index field.',
                                 status='not_started', readout=None, decoded_bit=None,
                                 decoded_reference_index=None))
    if (len(groups), len(stages), len(slots), sum(s['steps'] for s in stages)) != expected[profile]:
        raise ValueError('fixed public pilot cardinality changed')
    for group in groups:
        group.setdefault('row_ids', []); group.setdefault('closed', False)
        group.setdefault('mapping', 'not_applicable'); group.setdefault('observations', [])
        group['profile'] = profile
    recorder = Recorder(args.output_dir, clock, rss); budget = Budget(recorder, stages, slots)
    result = dict(profile=profile, status='execution_failed', outcome='not_evaluated', model_loaded=False,
                  groups=groups, stages=stages, rows=rows, calls=slots, initializations=0,
                  baseline_qualified=None, toggle_candidate_group_ids=[])
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
        payload = design_factory()
        if payload['profile'] != profile:
            raise ValueError('design profile differs')
        recorder.json('design.json', payload); recorder.json('planned-groups.json', groups)
        recorder.json('planned-stages.json', stages); recorder.json('planned-rows.json', rows)
        recorder.json('planned-calls.json', slots)
        result['source'] = source_metadata(args, extra_sources); recorder.json('source.json', result['source'])
        recorder.json('thread-environment.json', dict(startup=STARTUP_THREADS, current={k: os.environ.get(k) for k in THREAD_KEYS}))
        if any(v != '1' for v in STARTUP_THREADS.values()):
            raise InstrumentFailure('four thread limits must equal one before process startup')
        if artifact_envelope()['total_bytes'] >= WORKING_OUTPUT:
            raise ResourceLimit('native evidence estimate exceeds working cap')
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
            backend = guarded_backend_factory(backend_factory, require_runtime=require_runtime)(args, recorder, budget)
            result.update(model_loaded=True, cold_initialization_seconds=clock()-started)
            if hasattr(backend, 'bindings'):
                recorder.json('model-inputs.json', backend.bindings)
            for group in groups:
                active = group; state = initial.copy(); result['initializations'] += 1
                group.update(status='running', initial_state_sha256=prior.array_hash(state))
                recorder.event('group_started', group_id=group['group_id'], initial_sha256=group['initial_state_sha256'])
                for rid in group['row_ids']:
                    rows[rid]['status'] = 'running'
                for sid in group['stage_ids']:
                    engine.execute_stage(backend, state, stages[sid], recorder, budget)
                    if observe_stage is not None:
                        saved, _, _ = prior.reconstruct(recorder.directory, stages[sid])
                        observation = observe_stage(saved.copy(), refs.copy(), copy.deepcopy(stages[sid]))
                        observation = {**observation, 'stage_id': sid, 'controls': stages[sid]['controls'].copy()}
                        group['observations'].append(observation)
                        recorder.json(f'stage-{sid:02d}-observation.json', observation)
                saved, _, _ = prior.reconstruct(recorder.directory, stages[group['stage_ids'][-1]])
                for rid in group['row_ids']:
                    row = rows[rid]; readout = neutral_readout(saved[1:, row['lane']], refs)
                    label = readout['qualified_label']
                    row.update(status='completed', readout=readout,
                               decoded_bit=None if label is None else ('ad', 'td').index(label))
                    if profile == 'public-path-comparison':
                        row['decoded_reference_index'] = row['decoded_bit']
                bits = [rows[rid]['decoded_bit'] for rid in group['row_ids']]
                group.update(status='completed', decoded_pair=bits,
                             mapping=classify_pair(bits) if profile == PROFILE else 'not_applicable',
                             terminal_state_sha256=prior.array_hash(state),
                             diagnostics_passed=all(stages[i]['cache_comparison']['passed'] for i in group['stage_ids']))
                close_group(group, result, recorder); recorder.event('group_finished', group_id=group['group_id'])
                active = None
            if budget.calls != len(slots):
                raise InstrumentFailure('incomplete fixed call schedule')
            baselines = None if profile != PROFILE else all(g['decoded_pair'] == g['expected_baseline_pair'] for g in groups[:3])
            toggles = [g['group_id'] for g in groups[3:] if g['mapping'] == 'toggle'] if profile == PROFILE else []
            outcome = ('path_comparison_completed' if profile != PROFILE else
                       'baseline_unqualified' if not baselines else 'toggle_observed' if toggles else
                       'toggle_unresolved' if any(g['mapping'] == 'unresolved' for g in groups[3:]) else 'no_toggle_observed')
            result.update(status='completed', outcome=outcome, baseline_qualified=baselines,
                          toggle_candidate_group_ids=toggles)
    except BaseException as exc:
        status = failure_status(exc)
        # The inherited stage finally block records a cache-inspection error
        # before raising a generic ValueError. Preserve that original category.
        if str(exc) == 'stage failed final cache gate' and active is not None:
            causes = [stages[sid].get('final_cache_error', '') for sid in active['stage_ids']]
            if any(x.startswith(('TimeoutError:', 'MemoryError:', 'ResourceLimit:', 'InterruptedError:')) for x in causes):
                status = 'incomplete'
            elif any(x.startswith(('OSError:', 'IOError:')) for x in causes):
                status = 'execution_failed'
        result.update(status=status, error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        if active is not None:
            active.update(status=result['status'], error=result['error'])
            for rid in active['row_ids']:
                rows[rid]['status'] = result['status']
        recorder.event('execution_failed', _final=True, status=result['status'], error=result['error'])
    finally:
        if install_signals:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig in handlers:
                signal.signal(sig, signal.SIG_IGN)
        recorder.finalizing = True
        for group in groups:
            if group['status'] == 'not_started':
                group.update(status='not_run', not_run_reason=result['status'])
                for rid in group['row_ids']:
                    rows[rid].update(status='not_run', not_run_reason=result['status'])
            close_group(group, result, recorder)
        result.update(model_calls=budget.calls, call_counts=budget.counts,
                      elapsed_seconds=clock()-recorder.started, peak_rss_bytes=rss())
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


def run(args, **kwargs):
    qualifier = kwargs.pop('qualifier', qualify_inputs)
    return run_fixed(args, profile=PROFILE, plan_factory=plans, design_factory=design,
                     qualifier=qualifier, **kwargs)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache-dir', 'vectors', 'references', 'reference-manifest', 'source-hold-results',
                 'recovery-results', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('references-sha256', 'reference-manifest-sha256', 'recovery-results-sha256'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--preflight-only', action='store_true')
    return p


def main(argv=None):
    result = run(parser().parse_args(argv))
    return 0 if result['status'] in ('completed', 'preflight_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
