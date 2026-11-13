#!/usr/bin/env python3
"""Two fixed public-control continuations from identical saved endpoint states."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import signal
import subprocess
import time
import traceback

import run_public_control_tolerance as shared
prior = shared.prior
import numpy as np

memory, reference, contract, batch = prior.memory, prior.reference, prior.contract, prior.batch
ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'public-endpoint-continuation'
GROUPS = ('continue-perturbed-endpoint', 'restore-nominal-endpoint')
MAX_CALLS, MAX_UPDATES = 732, 720
MAX_SECONDS, EXTERNAL_SECONDS, GRACE = 900, 1080, 30
MAX_RSS, MAX_VM = 8*1024**3, 16*1024**3
MAX_OUTPUT, FINAL_RESERVE = 160*1024**2, 8*1024**2
WORKING_OUTPUT = MAX_OUTPUT-FINAL_RESERVE
ARCHIVE_CAP, TRANSFER_RESERVATION, SOURCE_UPLOAD_CAP = 224*1024**2, 320*1024**2, 8*1024**2
PRECISION_SCOPE = ('Existing native64 normalization/rotary patches, per-call observations, '
    'CPU math SDPA and one thread; four stage-specific cached/full batch comparisons. '
    'No new derivative-convergence test or global precision proof.')


def source_index(bit):
    prior.validate_plan(prior.WritePlan(bit))
    return 1-bit


PACKET_HASHES = (
    'dd1337dfeef8a1f321bd36c948cc625149796b8815cb8fe56cb1dbc090fafd10',
    'd5c25d476e718be48f115aa8783be0a1c5b64cec275fba0c215c9f79870d7314',
)
SOURCE_PHASE_COMMIT = '739a4fd678e35364236799d3701463a4d7b0009d'
HISTORY_SCOPE = 'not_applicable_endpoint_continuation'


def endpoint_controls(bit, perturbed):
    controls = prior.stage_plan(prior.WritePlan(bit))[0]['controls'].copy()
    if perturbed: controls['log10_temperature' if bit == 0 else 'alpha'] -= 1e-4 if bit == 0 else 1e-5
    return controls


def same_array(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def qualify_endpoints(args, refs):
    """Load only the pinned, audited pre-hold endpoints and recheck their origin."""
    bit = args.requested_bit; source_index(bit); shared.check_references(refs)
    if args.endpoint_packet_sha256 != PACKET_HASHES[bit]: raise ValueError('endpoint packet is not the fixed selected input')
    packet_binding = reference.require_hash(args.endpoint_packet,args.endpoint_packet_sha256)
    packet = json.loads(Path(args.endpoint_packet).read_text())
    if (packet['status'] != 'saved_arrays_verified_no_model_replay' or packet['requested_bit'] != bit
            or packet['source_commit'] != SOURCE_PHASE_COMMIT or packet['source_stage_id'] != 2
            or packet['source_group'] != 'full-negative-offset' or packet['source_completed_endpoint_updates'] != 120
            or packet['source_endpoint_controls'] != endpoint_controls(bit,True)
            or packet['nominal_endpoint_controls'] != endpoint_controls(bit,False)
            or packet['hold_controls'] != memory.writes.HOLD):
        raise ValueError('fixed endpoint packet identity or controls differ')
    def bound(path, expected):
        path = Path(path)
        if not path.is_file() or path.is_symlink(): raise ValueError('missing or symlink endpoint evidence')
        value = reference.require_hash(path,expected['sha256'])
        if value['bytes'] != expected['bytes']: raise ValueError('endpoint evidence byte count differs')
        return value
    audit_binding = bound(args.source_audit,packet['source_audit'])
    audit = json.loads(Path(args.source_audit).read_text())
    if (audit['status'] != 'passed' or audit['profile'] != 'public-two-phase-challenge'
            or audit['requested_bit'] != bit or audit['audited_source_commit'] != SOURCE_PHASE_COMMIT
            or audit['all_rows_preserved'] is not True or audit['rows'] != packet['full_source_rows']):
        raise ValueError('full source audit differs from packet')
    source_run = Path(args.source_run)
    artifacts = []; names = set()
    for original in packet['source_artifact_bindings']:
        name = Path(original['path']).name
        if name in names: raise ValueError('duplicate packet source artifact')
        names.add(name); artifacts.append(bound(source_run/name,original))
    raw = json.loads((source_run/'result.json').read_text())
    bound(source_run/'result.json',packet['source_result'])
    if raw['profile'] != 'public-two-phase-challenge' or raw['requested_bit'] != bit or raw['source']['source_commit'] != SOURCE_PHASE_COMMIT:
        raise ValueError('source result identity differs')
    source_stage = raw['stages'][2]
    if (source_stage['stage_id'] != 2 or source_stage['group'] != 'full-negative-offset'
            or source_stage['controls'] != packet['source_endpoint_controls'] or source_stage['status'] != 'completed'
            or source_stage['steps'] != source_stage['completed_steps'] or source_stage['completed_steps'] != 120):
        raise ValueError('source endpoint stage differs')
    expected_names = {'result.json','source.json','design.json','phase-inputs.json','qualified-inputs.json',
        'stage-02-initial.npy','stage-02-trajectory.json','stage-03-initial.npy'}|{r['path'] for r in source_stage['blocks']}
    if names != expected_names: raise ValueError('source endpoint artifact inventory differs')
    states,_,_ = prior.reconstruct(source_run,source_stage)
    original = np.stack([refs[1-bit,p] for p in (0,1)])
    if (states.shape != (121,2,896) or not same_array(states[0],original)
            or prior.array_hash(states) != packet['endpoint_trajectory_array_sha256']):
        raise ValueError('source endpoint trajectory or phase origin differs')
    hold_initial = np.load(source_run/'stage-03-initial.npy',allow_pickle=False)
    if not same_array(states[-1],hold_initial) or packet['hold_stage3_initial_bitwise_equal'] is not True:
        raise ValueError('source handoff continuity differs')
    endpoint_binding = bound(args.endpoint_states,packet['endpoint_npz'])
    with np.load(args.endpoint_states,allow_pickle=False) as z:
        if set(z.files) != {'states','origin_reference_index','origin_reference_phase'}: raise ValueError('endpoint NPZ schema differs')
        initial = z['states'].copy(); indices, phases = z['origin_reference_index'],z['origin_reference_phase']
    if (not same_array(initial,states[-1]) or initial.dtype != np.float64 or initial.shape != (2,896)
            or not np.isfinite(initial).all() or indices.dtype != np.int64 or phases.dtype != np.int64
            or indices.tolist() != [1-bit,1-bit] or phases.tolist() != [0,1]
            or prior.array_hash(initial) != packet['endpoint_batch_array_sha256']):
        raise ValueError('selected endpoint array or origin ordering differs')
    ordering = [dict(batch_index=i,origin_reference_index=1-bit,origin_reference_label=('ad','td')[1-bit],
        origin_reference_phase=i,source_result_row_id=2+i,initial_reference_array_sha256=prior.array_hash(refs[1-bit,i]),
        endpoint_array_sha256=prior.array_hash(initial[i])) for i in (0,1)]
    if packet['ordering'] != ordering or packet['source_endpoint_call_ids'] != list(range(367,490)):
        raise ValueError('endpoint origin metadata or historical call accounting differs')
    initial_rows = [dict(**row,shape=[896],dtype='float64',array_sha256=prior.array_hash(initial[i])) for i,row in enumerate(ordering)]
    return initial,dict(packet_binding=packet_binding,packet=packet,source_audit_binding=audit_binding,
        source_artifact_bindings=artifacts,endpoint_binding=endpoint_binding,initial_rows=initial_rows,
        batch_initial_sha256=prior.array_hash(initial),scope='Literal states after 120 historical perturbed endpoint updates; before any hold. '
            'Origin phase labels identify earlier reference rows, not the current endpoint states.')


def plans(requested_bit):
    profile = PROFILE
    plan = prior.validate_plan(prior.WritePlan(requested_bit))
    groups, stages, rows, calls = [], [], [], []
    names = GROUPS
    deltas = ((0.,-1e-4) if requested_bit == 0 else (-1e-5,0.), (0.,0.))
    for group_id, (name, delta) in enumerate(zip(names, deltas, strict=True)):
        group = dict(profile=profile, group_id=group_id, name=name, endpoint_delta=dict(alpha=delta[0], log10_temperature=delta[1]),
            status='not_started', row_ids=[2*group_id, 2*group_id+1],
            stage_ids=[2*group_id, 2*group_id+1], initial_state_sha256=None, terminal_state_sha256=None)
        groups.append(group)
        for local_id, canonical in enumerate(prior.stage_plan(plan)):
            stage = copy.deepcopy(canonical)
            stage.update(profile=profile, stage_id=len(stages), group_id=group_id, group=name, local_stage_id=local_id,
                nominal_controls=canonical['controls'].copy(), historical_comparison=None,
                historical_comparison_scope=HISTORY_SCOPE)
            if local_id == 0:
                for key, value in group['endpoint_delta'].items(): stage['controls'][key] += value
            stages.append(stage)
            for category, step, cached in [('prefix', 0, None), ('comparison', 0, True), ('comparison', 0, False)]+[
                    ('update', step, True) for step in range(1, stage['steps']+1)]:
                calls.append(dict(profile=profile, call=len(calls)+1, group_id=group_id, group=name, stage_id=stage['stage_id'],
                    stage=stage['name'], category=category, step=step, cached=cached, status='not_started'))
        for start in range(2):
            rows.append(dict(profile=profile, row_id=len(rows), group_id=group_id, group=name, batch_index=start,
                message=plan.messages[start], requested_bit=requested_bit, origin_reference_index=source_index(requested_bit),
                origin_reference_label=('ad', 'td')[source_index(requested_bit)], origin_reference_phase=start,
                source_result_row_id=2+start, source_stage_id=2, historical_endpoint_updates=120, status='not_started',
                parser_outcome='not_started', public_action=None, realized_stages=[dict(profile=profile, stage_id=s['stage_id'],
                    name=s['name'], steps=s['steps'], controls=s['controls'].copy()) for s in stages[-2:]],
                dispatch_count=0, completed_updates=0, readout=None, outcome='not_evaluated',
                historical_comparison=None, historical_comparison_scope=HISTORY_SCOPE))
    if len(calls) != MAX_CALLS or sum(s['steps'] for s in stages) != MAX_UPDATES:
        raise ValueError('fixed endpoint-continuation schedule changed')
    return plan, groups, stages, rows, calls


def derive_design(requested_bit):
    plan, groups, stages, rows, calls = plans(requested_bit)
    envelope = 2*prior.artifact_envelope()['estimated_uncompressed_envelope_bytes']
    if envelope+prior.FAILED_RETURN_BYTES+16384 >= WORKING_OUTPUT:
        raise ValueError('uncompressed evidence does not fit working allowance')
    return dict(schema_version=1, profile=PROFILE,
        evidence='adapted exploratory public-controller endpoint continuation after all eight phase outcomes', evidence_rung=0,
        requested_bit=requested_bit, public_messages=list(plan.messages),
        origin_reference_index=source_index(requested_bit), origin_reference_label=('ad','td')[source_index(requested_bit)],
        origin_reference_phases=[0,1], source_stage_id=2, source_result_row_ids=[2,3], historical_endpoint_updates=120,
        endpoint_packet_sha256=PACKET_HASHES[requested_bit], source_phase_commit=SOURCE_PHASE_COMMIT,
        groups=groups, realized_stages=[{k:s[k] for k in ('profile','stage_id','group_id','name','steps',
            'controls','nominal_controls')} for s in stages],
        schedule=dict(groups=2,batch_size=2,rows=4,stages=4,endpoint_steps=120,hold_steps=240,
            batch_updates=720,row_updates=1440,prefix_calls=4,comparison_calls=8,total_calls=732,
            endpoint_initializations=2,between_arm_resets=1,reference_initializations=0,
            historical_endpoint_updates_per_row=120,new_updates_per_row=360,total_path_updates_per_row=480),
        limits=dict(model_calls=MAX_CALLS,internal_seconds=MAX_SECONDS,external_seconds=EXTERNAL_SECONDS,
            termination_grace_seconds=GRACE,threads=1,peak_rss_bytes=MAX_RSS,external_vm_bytes=MAX_VM,
            working_raw_bytes=WORKING_OUTPUT,finalization_reserve_bytes=FINAL_RESERVE,total_raw_bytes=MAX_OUTPUT,
            archive_bytes=ARCHIVE_CAP,transfer_reservation_bytes=TRANSFER_RESERVATION,source_upload_bytes=SOURCE_UPLOAD_CAP),
        artifact_envelope=dict(estimated_uncompressed_envelope_bytes=envelope,
            failed_return_reserved_bytes=prior.FAILED_RETURN_BYTES,
            scope='Two original-format complete groups plus one failed native-shape return; no compression assumption.'),
        map=dict(model=reference.MODEL,revision=reference.REVISION,prefix_ids=reference.PREFIX_IDS,
            hold=memory.writes.HOLD,endpoints=memory.writes.ENDPOINTS),
        precision_scope=PRECISION_SCOPE,diagnostic_tolerance=prior.DIAGNOSTIC_TOLERANCE,
        initial_state_scope='Both arms fork identical saved 120-step perturbed endpoint terminal states; no reference reset or settled-hold start.',
        historical_scope='The earlier 120 endpoint updates are retained as historical cost and origin. Neither continuation arm has scalar equivalence.',
        stopping='Fixed two groups, no required offset failure, retry, extension or outcome-selected control. Finite wrong, '
            'unresolved and diagnostic failures remain outcomes; structural/precision/cache/resource errors abort.',
        scope='Two selected endpoint histories per bit and two fixed public continuation arms. No all-phase reliability, '
            'continuous tolerance, language-interface recovery, ECC, hidden channel or speedup claim.',
        timing_scope='Record actual elapsed time and calls. The protocol describes a historical proportional '
            'planning estimate only; no matched timing, deadline prediction or achieved speedup claim.')


class Recorder(memory.Recorder):
    def admit(self, size, final=False):
        if self.size()+size > (MAX_OUTPUT if final or self.finalizing else WORKING_OUTPUT):
            raise RuntimeError('raw artifact limit reached')


class Budget:
    def __init__(self, recorder, slots, requested_bit):
        self.recorder, self.slots, self.requested_bit = recorder, slots, requested_bit
        self.profile = PROFILE
        _, _, self.expected_stages, _, self.expected_slots = plans(requested_bit)
        if slots != self.expected_slots: raise ValueError('call slots differ from fixed profile schedule')
        self.calls = 0; self.counts = dict(prefix=0, comparison=0, update=0)
        self.stage_id = None; self.max_rss = MAX_RSS

    def check(self):
        if self.recorder.clock()-self.recorder.started >= MAX_SECONDS: raise TimeoutError('900-second internal cap')
        if self.recorder.rss() >= MAX_RSS: raise MemoryError('8 GiB RSS cap')
        if self.recorder.size() > WORKING_OUTPUT: raise RuntimeError('working raw artifact limit reached')

    def begin(self, stage):
        expected = self.expected_stages[stage['stage_id']]
        if any(stage[k] != expected[k] for k in ('profile', 'group_id', 'group', 'local_stage_id',
                'name', 'controls', 'nominal_controls', 'steps')):
            raise ValueError('stage differs from fixed realized-control schedule')
        if self.calls >= MAX_CALLS or self.slots[self.calls]['stage_id'] != stage['stage_id']:
            raise ValueError('stage order differs from fixed call schedule')
        self.stage_id = stage['stage_id']

    def claim(self, label, category='prefix'):
        self.check(); self.recorder.admit(prior.FAILED_RETURN_BYTES+16384)
        if self.calls >= MAX_CALLS: raise RuntimeError('model-call cap reached')
        slot = self.slots[self.calls]
        expected = self.expected_slots[self.calls]
        if (slot['stage_id'] != self.stage_id or slot['category'] != category or any(
                slot.get(key) != expected[key] for key in ('profile', 'call', 'group_id', 'group',
                    'stage_id', 'stage', 'category', 'step', 'cached'))):
            raise RuntimeError('model call differs from fixed schedule')
        self.calls += 1; self.counts[category] += 1; slot.update(status='started', label=label)
        self.recorder.event('model_call_started', **slot)
        return slot


def preflight(args, qualifier=memory.qualify_inputs, endpoint_qualifier=qualify_endpoints):
    """Qualify immutable inputs; no weights, model load or numerical forward."""
    refs, provenance = qualifier(args)
    initial, continuation = endpoint_qualifier(args, refs)
    return dict(status='passed',profile=PROFILE,model_calls=0,model_loaded=False,
        design=derive_design(args.requested_bit),qualified_inputs=provenance,continuation_inputs=continuation,
        endpoint_initial_state_sha256=prior.array_hash(initial))


classify_readout = shared.classify_readout
immutable_group_artifacts = shared.immutable_group_artifacts


def execute_stage(backend, state, stage, recorder, budget):
    archive = prefix = None; stage['status'] = 'running'
    try:
        budget.begin(stage); archive = prior.BlockArchive(recorder, stage, state)
        try:
            backend.prepare(stage['controls'])
            if not prior.precision(backend.observations): raise ValueError('prefix native64 observations missing')
            slot = budget.slots[budget.calls-1]; slot['dtype_observations'] = copy.deepcopy(backend.observations)
            prefix = backend.cache_arrays()
            if not batch.valid_cache(prefix): raise ValueError('invalid scalar prefix cache')
            stage['scalar_prefix_cache'] = recorder.npz(f'stage-{stage["stage_id"]:02d}-scalar-prefix.npz', **prefix)
            slot['status'] = 'returned'; budget.check()
        except BaseException as exc:
            if budget.calls and budget.slots[budget.calls-1]['category'] == 'prefix':
                budget.slots[budget.calls-1].update(status=memory.error_status(exc), error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if budget.calls and budget.slots[budget.calls-1]['category'] == 'prefix':
                recorder.event('model_call_completed', _final=True, **budget.slots[budget.calls-1])
        diagnostics = [prior.evaluate(backend, state, stage, recorder, budget, prefix, archive) for _ in range(2)]
        stage['cache_comparison'] = batch.compare(*diagnostics)
        for _ in range(stage['steps']): prior.evaluate(backend, state, stage, recorder, budget, prefix, archive)
        stage['status'] = 'completed'
    except BaseException as exc:
        stage.update(status=memory.error_status(exc), error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        with batch.preserve_call_files():
            if prefix is not None:
                try:
                    after = backend.cache_arrays(); stage['final_scalar_cache_sha256'] = prior.cache_hash(after)
                    stage['final_scalar_cache_unchanged'] = prior.same_bytes(prefix, after)
                    if not stage['final_scalar_cache_unchanged']:
                        stage.update(status='failed', final_cache_error='final scalar cache changed')
                except BaseException as exc:
                    stage.update(status='failed', final_scalar_cache_unchanged=False,
                        final_cache_error=f'{type(exc).__name__}: {exc}')
            if archive is not None:
                archive.finalize()
            recorder.event('stage_finished', _final=True, stage_id=stage['stage_id'], group_id=stage['group_id'],
                status=stage['status'], completed_steps=stage['completed_steps'])
    if stage['status'] != 'completed': raise ValueError('stage failed final cache gate')


def run(args, backend_factory=prior.Backend, qualifier=memory.qualify_inputs,
        endpoint_qualifier=qualify_endpoints, clock=time.monotonic,
        rss=reference.rss_bytes, install_signals=True):
    profile = PROFILE
    plan, groups, stages, rows, slots = plans(args.requested_bit)
    recorder = Recorder(args.output_dir, clock, rss); budget = Budget(recorder, slots, args.requested_bit)
    result = dict(schema_version=1, profile=profile, status='failed', requested_bit=plan.requested_bit, groups=groups,
        rows=rows, stages=stages, calls=slots, reference_initializations=0, endpoint_initializations=0, between_arm_resets=0,
        historical_endpoint_updates_per_row=120, historical_comparison_scope=HISTORY_SCOPE,
        evidence=derive_design(plan.requested_bit)['evidence'], evidence_rung=0,
        precision_scope=PRECISION_SCOPE, historical_scalar_equivalence=None)
    state = backend = None; old_handlers = {}
    if install_signals:
        def stop(signum, frame):
            if signum == signal.SIGALRM: raise TimeoutError('declared internal deadline reached')
            raise InterruptedError(f'received signal {signum}')
        old_handlers = {s:signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
        for s in old_handlers: signal.signal(s, stop)
        signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    try:
        recorder.json('design.json', derive_design(plan.requested_bit)); recorder.json('planned-rows.json', rows)
        recorder.json('planned-groups.json', groups); recorder.json('planned-stages.json', stages)
        recorder.json('planned-calls.json', slots)
        modules = (shared, prior, memory, contract, memory.writes, memory.recovery, memory.hold_test,
            reference, batch, prior.historical_source, prior.trajectory_source)
        paths = [Path(m.__file__) for m in modules]+[Path(__file__), ROOT/'docs/public-control-continuation.md', ROOT/'tests/test_public_control_continuation.py',
            ROOT/'docs/receipts/p6-writes.json', ROOT/'docs/receipts/p6-write-path-design.json',
            ROOT/'docs/receipts/public-control-tolerance.json',
            ROOT/'pyproject.toml', ROOT/'uv.lock']
        result['source'] = dict(utc=reference.utc_now(), source_commit=subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
            files=[memory.binding(p) for p in paths], arguments={**{k:str(v) for k,v in vars(args).items()}, 'profile':profile})
        recorder.json('source.json', result['source'])
        refs, provenance = qualifier(args)
        recorder.json('qualified-inputs.json', provenance)
        initial, continuation = endpoint_qualifier(args, refs)
        recorder.json('continuation-inputs.json', continuation)
        result['existing_history_cost'] = continuation['packet']['existing_history_cost']
        # Freeze every public request before any numerical call. Realized controls
        # are a separately logged engineering perturbation, never a changed parse.
        for row in rows:
            action = prior.actions(plan)[row['batch_index']]
            row['public_action'] = {**asdict(action), 'stages':[asdict(s) for s in action.stages]}
            row['parser_outcome'] = 'accepted'
            def dispatch(parsed):
                contract.verify_action(row['message'], parsed); row['dispatch_count'] += 1
            contract.dispatch_command(row['message'], dispatch, prepared_action=action)
        recorder.json('frozen-rows.json', rows)
        state = initial.copy(); stages[0]['initial'] = recorder.state('stage-00-initial.npy', state)
        recorder.event('inputs_frozen', initial_sha256=prior.array_hash(state), source_phase_commit=continuation['packet']['source_commit'],
            historical_endpoint_updates=120)
        budget.check(); started = clock(); backend = backend_factory(args, recorder, budget)
        result['cold_initialization_seconds'] = clock()-started
        if hasattr(backend, 'bindings'): recorder.json('model-inputs.json', backend.bindings)
        for group in groups:
            state = initial.copy(); result['endpoint_initializations'] += 1
            if group['group_id']: result['between_arm_resets'] += 1
            group.update(status='running', initial_state_sha256=prior.array_hash(state))
            recorder.event('group_started', group_id=group['group_id'], initial_sha256=group['initial_state_sha256'])
            for i in group['row_ids']: rows[i]['status'] = 'running'
            for i in group['stage_ids']: execute_stage(backend, state, stages[i], recorder, budget)
            hold = stages[group['stage_ids'][1]]
            saved, _, _ = prior.reconstruct(recorder.directory, hold)
            for i in group['row_ids']:
                row = rows[i]; index = row['batch_index']
                row['readout'] = memory.neutral_readout(saved[1:,index], refs)
                row['outcome'] = classify_readout(row['readout'], plan.requested_bit)
                row['numerical_diagnostics_passed'] = all(stages[k]['cache_comparison']['passed'] for k in group['stage_ids'])
                row.update(status='completed', completed_updates=360)
                recorder.json(f'row-{i:02d}-completed.json', row)
            group.update(terminal_state_sha256=prior.array_hash(state),
                diagnostics_passed=all(stages[k]['cache_comparison']['passed'] for k in group['stage_ids']))
            group['artifacts'] = immutable_group_artifacts(recorder.directory, group, stages, slots)
            group['status'] = 'completed'
            try:
                recorder.json(f'group-{group["group_id"]:02d}.json', group)
            except BaseException:
                group['status'] = 'failed'
                raise
            recorder.event('group_finished', group_id=group['group_id'], status=group['status'])
        if budget.calls != MAX_CALLS or budget.counts != dict(prefix=4, comparison=8, update=720):
            raise ValueError('completed schedule count differs')
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status=memory.error_status(exc), error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        for item in groups+rows:
            if item['status'] == 'running': item.update(status=memory.error_status(exc), error=result['error'])
    finally:
        if install_signals:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s in old_handlers: signal.signal(s, signal.SIG_IGN)
        recorder.finalizing = True
        try:
            for row in rows:
                group = groups[row['group_id']]
                row['completed_updates'] = sum(stages[i]['completed_steps'] for i in group['stage_ids'])
                recorder.json(f'row-{row["row_id"]:02d}.json', row, final=True)
            result.update(model_calls=budget.calls, call_counts=budget.counts,
                completed_batch_updates=sum(s['completed_steps'] for s in stages),
                completed_row_updates=2*sum(s['completed_steps'] for s in stages),
                elapsed_seconds=clock()-recorder.started, peak_rss_bytes=rss(),
                outcome_counts={k:sum(r['outcome'] == k for r in rows) for k in ('passed','failed','unresolved','not_evaluated')})
            result['terminal_state_sha256'] = prior.array_hash(state) if state is not None else None
            result['resource_limits_passed'] = (budget.calls <= MAX_CALLS and result['elapsed_seconds'] < MAX_SECONDS
                and result['peak_rss_bytes'] < MAX_RSS and recorder.size() <= MAX_OUTPUT)
            recorder.json('result.json', result, final=True)
            recorder.event('finished', status=result['status'], model_calls=budget.calls)
            recorder.stream.close()
            recorder.json('manifest.json', dict(files=[memory.binding(p) for p in sorted(recorder.directory.iterdir())
                if p.is_file() and p.name != 'manifest.json']), final=True)
        finally:
            for s, handler in old_handlers.items(): signal.signal(s, handler)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requested-bit', type=int, choices=(0,1), required=True)
    parser.add_argument('--preflight-only', action='store_true')
    for name in ('cache-dir','vectors','references','reference-manifest','source-hold-results',
                 'recovery-results','endpoint-packet','endpoint-states','source-run','source-audit','output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('references-sha256','reference-manifest-sha256','recovery-results-sha256','endpoint-packet-sha256'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    for digest in (args.references_sha256, args.reference_manifest_sha256, args.recovery_results_sha256, args.endpoint_packet_sha256):
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            parser.error('input hashes must be lowercase SHA-256 digests')
    if args.preflight_only:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        try:
            result = preflight(args)
        except BaseException as exc:
            result = dict(status='failed', model_calls=0, model_loaded=False,
                error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        with (args.output_dir/'input-preflight.json').open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
        return 0 if result['status'] == 'passed' else 1
    return 0 if run(args)['status'] == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
