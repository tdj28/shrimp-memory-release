#!/usr/bin/env python3
"""Fixed public hold controls and nominal repair from literal early and late saved states."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import signal
import subprocess
import time
import traceback

import run_public_control_continuation as continuation
shared = continuation.shared
prior = shared.prior
import numpy as np

memory, reference, contract, batch = prior.memory, prior.reference, prior.contract, prior.batch
ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'public-hold-repair'
GROUPS = ('hold36-continue-hold', 'hold36-restore-nominal',
          'hold240-continue-hold', 'hold240-restore-nominal')
HOLD_AGES = (36, 240)
MAX_CALLS, MAX_UPDATES = 1464, 1440
MAX_SECONDS, EXTERNAL_SECONDS, GRACE = 1800, 2100, 30
MAX_RSS, MAX_VM = 8*1024**3, 16*1024**3
MAX_OUTPUT, FINAL_RESERVE = 256*1024**2, 8*1024**2
WORKING_OUTPUT = MAX_OUTPUT-FINAL_RESERVE
ARCHIVE_CAP, TRANSFER_RESERVATION, SOURCE_UPLOAD_CAP = 352*1024**2, 448*1024**2, 8*1024**2
PRECISION_SCOPE = ('Existing native64 normalization/rotary patches, per-call observations, '
    'CPU math SDPA and one thread; eight stage-specific cached/full batch comparisons. '
    'No new derivative-convergence test or global precision proof.')


def source_index(bit):
    prior.validate_plan(prior.WritePlan(bit))
    return 1-bit


PACKET_HASHES = ('d12b7b61a1bc0add6e2df79ee14a3817965020ce904141f10d6ddce74dfe50a7',
    '7f15a627d070503f8f66b96aed81af127f21f496f3f6222169a8d96fa06279bf')
SOURCE_PHASE_COMMIT = continuation.SOURCE_PHASE_COMMIT
HISTORY_SCOPE = 'not_applicable_new_hold_repair'


def endpoint_controls(bit, perturbed):
    controls = prior.stage_plan(prior.WritePlan(bit))[0]['controls'].copy()
    if perturbed: controls['log10_temperature' if bit == 0 else 'alpha'] -= 1e-4 if bit == 0 else 1e-5
    return controls


def same_array(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def qualify_holds(args, refs):
    """Verify literal saved-state packet before weights; labels never filter rows."""
    bit=args.requested_bit; source_index(bit); shared.check_references(refs)
    if args.hold_packet_sha256 != PACKET_HASHES[bit]: raise ValueError('hold packet is not the fixed selected input')
    packet_binding=reference.require_hash(args.hold_packet,args.hold_packet_sha256)
    packet=json.loads(Path(args.hold_packet).read_text())
    if (packet['status']!='saved_arrays_verified_no_model_replay' or packet['profile']!='public-hold-repair-inputs'
            or packet['requested_bit']!=bit or packet['source_commit']!=SOURCE_PHASE_COMMIT
            or packet['source_stage_id']!=3 or packet['source_endpoint_stage_id']!=2
            or packet['source_group']!='full-negative-offset' or packet['historical_endpoint_updates']!=120
            or packet['source_completed_hold_updates']!=240 or packet['hold_ages']!=list(HOLD_AGES)
            or packet['source_endpoint_controls']!=endpoint_controls(bit,True)
            or packet['nominal_endpoint_controls']!=endpoint_controls(bit,False)
            or packet['hold_controls']!=memory.writes.HOLD):
        raise ValueError('fixed hold packet identity or controls differ')
    def bound(path,expected):
        path=Path(path)
        if not path.is_file() or path.is_symlink(): raise ValueError('missing or symlink hold evidence')
        value=reference.require_hash(path,expected['sha256'])
        if value['bytes']!=expected['bytes']: raise ValueError('hold evidence byte count differs')
        return value
    reference_binding=bound(args.references,packet['reference_binding'])
    audit_binding=bound(args.source_audit,packet['source_audit'])
    audit=json.loads(Path(args.source_audit).read_text())
    if (audit['status']!='passed' or audit['profile']!='public-two-phase-challenge'
            or audit['requested_bit']!=bit or audit['audited_source_commit']!=SOURCE_PHASE_COMMIT
            or audit['all_rows_preserved'] is not True or audit['rows']!=packet['full_source_rows']):
        raise ValueError('full source audit differs from hold packet')
    source_run=Path(args.source_run);artifacts=[];names=set()
    for original in packet['source_artifact_bindings']:
        name=Path(original['path']).name
        if name in names: raise ValueError('duplicate packet source artifact')
        names.add(name);artifacts.append(bound(source_run/name,original))
    source_result_binding=bound(source_run/'result.json',packet['source_result'])
    raw=json.loads((source_run/'result.json').read_text())
    if (raw['profile']!='public-two-phase-challenge' or raw['requested_bit']!=bit
            or raw['source']['source_commit']!=SOURCE_PHASE_COMMIT):
        raise ValueError('source result identity differs')
    endpoint,hold=raw['stages'][2:4]
    for stage,index,count,controls in [(endpoint,2,120,packet['source_endpoint_controls']),
            (hold,3,240,memory.writes.HOLD)]:
        if (stage['stage_id']!=index or stage['group']!='full-negative-offset'
                or stage['controls']!=controls or stage['status']!='completed'
                or stage['steps']!=count or stage['completed_steps']!=count):
            raise ValueError('source endpoint/hold stage differs')
    expected_names={'result.json','source.json','design.json','phase-inputs.json','qualified-inputs.json',
        'stage-02-initial.npy','stage-02-trajectory.json','stage-03-initial.npy','stage-03-trajectory.json'}
    expected_names|={x['path'] for stage in (endpoint,hold) for x in stage['blocks']}
    if names!=expected_names: raise ValueError('source hold artifact inventory differs')
    endpoint_states,_,_=prior.reconstruct(source_run,endpoint)
    hold_states,_,_=prior.reconstruct(source_run,hold)
    origin=np.stack([refs[1-bit,phase] for phase in (0,1)])
    if (endpoint_states.shape!=(121,2,896) or hold_states.shape!=(241,2,896)
            or not same_array(endpoint_states[0],origin)
            or prior.array_hash(endpoint_states)!=packet['endpoint_trajectory_array_sha256']
            or prior.array_hash(hold_states)!=packet['hold_trajectory_array_sha256']):
        raise ValueError('source trajectory or phase origin differs')
    if (not same_array(endpoint_states[-1],hold_states[0])
            or packet['endpoint_to_hold_initial_bitwise_equal'] is not True):
        raise ValueError('source endpoint/hold continuity differs')
    states_binding=bound(args.hold_states,packet['states_npz'])
    with np.load(args.hold_states,allow_pickle=False) as z:
        if set(z.files)!={'states','hold_ages','origin_reference_index','origin_reference_phase'}:
            raise ValueError('hold NPZ schema differs')
        initial=z['states'].copy();ages=z['hold_ages'];indices=z['origin_reference_index'];phases=z['origin_reference_phase']
    expected=np.stack([hold_states[age] for age in HOLD_AGES])
    if (not same_array(initial,expected) or initial.dtype!=np.float64 or initial.shape!=(2,2,896)
            or not np.isfinite(initial).all() or ages.dtype!=np.int64 or ages.tolist()!=list(HOLD_AGES)
            or indices.dtype!=np.int64 or phases.dtype!=np.int64
            or indices.tolist()!=[1-bit,1-bit] or phases.tolist()!=[0,1]
            or prior.array_hash(initial)!=packet['all_states_array_sha256']):
        raise ValueError('selected hold array, age or origin ordering differs')
    age_batches=[]
    for age_index,age in enumerate(HOLD_AGES):
        initial_rows=[dict(batch_index=i,origin_reference_index=1-bit,origin_reference_label=('ad','td')[1-bit],
            origin_reference_phase=i,source_result_row_id=2+i,
            initial_reference_array_sha256=prior.array_hash(refs[1-bit,i]),
            state_array_sha256=prior.array_hash(initial[age_index,i])) for i in (0,1)]
        age_batches.append(dict(age_index=age_index,hold_age=age,
            batch_array_sha256=prior.array_hash(initial[age_index]),initial_rows=initial_rows))
    if packet['age_batches']!=age_batches or packet['source_endpoint_call_ids']!=list(range(367,490)) or packet['source_hold_call_ids']!=list(range(490,733)):
        raise ValueError('hold age/origin metadata or historical call accounting differs')
    expected_observers=['tools/analyze_causal_geometry.py','tools/streaming_readout.py','tools/audit_p6_recovery.py']
    if [x['relative_path'] for x in packet['observer_source_bindings']]!=expected_observers:
        raise ValueError('observer source inventory differs')
    observers=[bound(ROOT/x['relative_path'],x) for x in packet['observer_source_bindings']]
    policy=packet['observer_policy']
    if policy['checkpoint_interval']!=12 or policy['consecutive_labels']!=3 or policy['reference_sha256']!=packet['reference_binding']['sha256']:
        raise ValueError('observer policy differs')
    if len(packet['observations'])!=4: raise ValueError('input observation inventory differs')
    for index,item in enumerate(packet['observations']):
        age_index,phase=divmod(index,2);age=HOLD_AGES[age_index]
        if (item['age_index']!=age_index or item['hold_age']!=age or item['batch_index']!=phase
                or item['origin_reference_index']!=1-bit or item['origin_reference_label']!=('ad','td')[1-bit]
                or item['origin_reference_phase']!=phase or item['source_result_row_id']!=2+phase
                or item['prefix_updates']!=age or item['initial_hold_state_included_but_not_pushed'] is not True
                or item['prefix_array_sha256']!=prior.array_hash(hold_states[:age+1,phase])
                or [x['steps'] for x in item['observation']['checkpoints']]!=list(range(12,age+1,12))):
            raise ValueError('input observation prefix identity differs')
        # Measured labels, including None, do not select or exclude an input.
    return initial,dict(packet_binding=packet_binding,packet=packet,source_audit_binding=audit_binding,
        source_result_binding=source_result_binding,source_artifact_bindings=artifacts,
        reference_binding=reference_binding,states_binding=states_binding,age_batches=age_batches,
        observer_source_bindings=observers,input_observations=packet['observations'],
        all_age_initial_sha256=prior.array_hash(initial),scope='Literal hold36/240states after120historical perturbed endpoint updates. '
            'Origin labels refer to the earlier references. Saved input observer metadata never selects branches or durations; no observer or model replay during qualification.')

def plans(requested_bit):
    plan = prior.validate_plan(prior.WritePlan(requested_bit))
    groups, stages, rows, calls = [], [], [], []
    for group_id, name in enumerate(GROUPS):
        age_index, repair = group_id//2, bool(group_id%2)
        age = HOLD_AGES[age_index]
        controls = endpoint_controls(requested_bit, False) if repair else memory.writes.HOLD.copy()
        group = dict(profile=PROFILE, group_id=group_id, name=name, age_index=age_index,
            hold_age=age, intervention='restore-nominal' if repair else 'continue-hold',
            intervention_controls=controls.copy(), status='not_started',
            row_ids=[2*group_id,2*group_id+1], stage_ids=[2*group_id,2*group_id+1],
            initial_state_sha256=None, terminal_state_sha256=None)
        groups.append(group)
        for local_id, canonical in enumerate(prior.stage_plan(plan)):
            stage = copy.deepcopy(canonical)
            stage.update(profile=PROFILE, stage_id=len(stages), group_id=group_id, group=name,
                local_stage_id=local_id, age_index=age_index, hold_age=age,
                name='intervention' if local_id == 0 else 'hold',
                nominal_controls=canonical['controls'].copy(), historical_comparison=None,
                historical_comparison_scope=HISTORY_SCOPE)
            if local_id == 0: stage['controls'] = controls.copy()
            stages.append(stage)
            for category,step,cached in [('prefix',0,None),('comparison',0,True),('comparison',0,False)]+[
                    ('update',step,True) for step in range(1,stage['steps']+1)]:
                calls.append(dict(profile=PROFILE,call=len(calls)+1,group_id=group_id,group=name,
                    stage_id=stage['stage_id'],stage=stage['name'],category=category,step=step,
                    cached=cached,status='not_started'))
        for start in range(2):
            rows.append(dict(profile=PROFILE,row_id=len(rows),group_id=group_id,group=name,
                batch_index=start,message=plan.messages[start],requested_bit=requested_bit,
                origin_reference_index=source_index(requested_bit),
                origin_reference_label=('ad','td')[source_index(requested_bit)],origin_reference_phase=start,
                source_result_row_id=2+start,source_stage_id=3,age_index=age_index,hold_age=age,
                historical_endpoint_updates=120,historical_hold_updates=age,
                historical_total_updates=120+age,total_path_updates=480+age,status='not_started',
                parser_outcome='not_started',public_action=None,
                realized_stages=[dict(profile=PROFILE,stage_id=x['stage_id'],name=x['name'],
                    steps=x['steps'],controls=x['controls'].copy()) for x in stages[-2:]],
                dispatch_count=0,completed_updates=0,readout=None,outcome='not_evaluated',
                historical_comparison=None,historical_comparison_scope=HISTORY_SCOPE))
    if len(calls) != MAX_CALLS or sum(x['steps'] for x in stages) != MAX_UPDATES:
        raise ValueError('fixed hold-repair schedule changed')
    return plan,groups,stages,rows,calls


def derive_design(requested_bit):
    plan,groups,stages,rows,calls = plans(requested_bit)
    envelope = 4*prior.artifact_envelope()['estimated_uncompressed_envelope_bytes']
    input_metadata_allowance = 4*1024**2
    if envelope+input_metadata_allowance+prior.FAILED_RETURN_BYTES+16384 >= WORKING_OUTPUT:
        raise ValueError('uncompressed evidence does not fit working allowance')
    return dict(schema_version=1,profile=PROFILE,
        evidence='adapted exploratory public hold-state repair after completed endpoint-restoration comparison',
        evidence_rung=0,requested_bit=requested_bit,public_messages=list(plan.messages),
        origin_reference_index=source_index(requested_bit),origin_reference_label=('ad','td')[source_index(requested_bit)],
        origin_reference_phases=[0,1],source_stage_id=3,source_endpoint_stage_id=2,
        source_result_row_ids=[2,3],historical_endpoint_updates=120,hold_ages=list(HOLD_AGES),
        hold_packet_sha256=PACKET_HASHES[requested_bit],source_phase_commit=SOURCE_PHASE_COMMIT,
        groups=groups,realized_stages=[{k:x[k] for k in ('profile','stage_id','group_id','name','steps',
            'controls','nominal_controls','age_index','hold_age')} for x in stages],
        schedule=dict(groups=4,batch_size=2,rows=8,stages=8,intervention_steps=120,hold_steps=240,
            batch_updates=1440,row_updates=2880,prefix_calls=8,comparison_calls=16,total_calls=1464,
            hold_state_initializations=4,between_group_resets=3,reference_initializations=0,
            historical_endpoint_updates_per_row=120,historical_hold_ages_per_group=[36,36,240,240],
            new_updates_per_row=360,total_path_updates_per_group=[516,516,720,720]),
        limits=dict(model_calls=MAX_CALLS,internal_seconds=MAX_SECONDS,external_seconds=EXTERNAL_SECONDS,
            termination_grace_seconds=GRACE,threads=1,peak_rss_bytes=MAX_RSS,external_vm_bytes=MAX_VM,
            working_raw_bytes=WORKING_OUTPUT,finalization_reserve_bytes=FINAL_RESERVE,total_raw_bytes=MAX_OUTPUT,
            archive_bytes=ARCHIVE_CAP,transfer_reservation_bytes=TRANSFER_RESERVATION,source_upload_bytes=SOURCE_UPLOAD_CAP),
        artifact_envelope=dict(original_group_bytes=prior.artifact_envelope()['estimated_uncompressed_envelope_bytes'],
            groups=4,estimated_uncompressed_envelope_bytes=envelope,
            input_metadata_allowance_bytes=input_metadata_allowance,
            failed_return_reserved_bytes=prior.FAILED_RETURN_BYTES,event_allowance_bytes=16384,
            total_working_requirement_bytes=envelope+input_metadata_allowance+prior.FAILED_RETURN_BYTES+16384,
            scope='Four original-format groups plus pinned age packet and one failed native-shape return; no compression assumption.'),
        map=dict(model=reference.MODEL,revision=reference.REVISION,prefix_ids=reference.PREFIX_IDS,
            hold=memory.writes.HOLD,endpoints=memory.writes.ENDPOINTS),
        precision_scope=PRECISION_SCOPE,diagnostic_tolerance=prior.DIAGNOSTIC_TOLERANCE,
        initial_state_scope='Four forks of literal saved hold states at36and240; no reference replacement or extra settling.',
        initial_observation_scope='Packet records neutral geometric and strict observations. No observed label, qualification or success selects inclusion or duration.',
        historical_scope='Selected original120endpoint plus36or240hold updates remain historical cost. No new branch has scalar trajectory equivalence.',
        stopping='Fixed four groups, no required success or failure, adaptive retry, extension or control selection. Finite wrong, unresolved and diagnostic disagreements remain outcomes; structural/precision/cache/resource errors abort.',
        scope='Public corrective-control/readiness challenge on two known origin histories per bit; no general online controller, held-out reliability, language interface, ECC or speedup claim.',
        timing_scope='Actual elapsed and calls retained. Earlier scalar/batch timings inform planning only; no matched speedup or deadline prediction.')


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
        if self.recorder.clock()-self.recorder.started >= MAX_SECONDS: raise TimeoutError('1800-second internal cap')
        if self.recorder.rss() >= MAX_RSS: raise MemoryError('8 GiB RSS cap')
        if self.recorder.size() > WORKING_OUTPUT: raise RuntimeError('working raw artifact limit reached')

    def begin(self, stage):
        expected = self.expected_stages[stage['stage_id']]
        if any(stage[k] != expected[k] for k in ('profile', 'group_id', 'group', 'local_stage_id',
                'name', 'controls', 'nominal_controls', 'steps', 'age_index', 'hold_age')):
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


def preflight(args, qualifier=memory.qualify_inputs, hold_qualifier=qualify_holds):
    """Qualify immutable arrays and metadata; no weights or model forward."""
    refs, provenance = qualifier(args)
    initial, holds = hold_qualifier(args,refs)
    return dict(status='passed',profile=PROFILE,model_calls=0,model_loaded=False,
        design=derive_design(args.requested_bit),qualified_inputs=provenance,hold_inputs=holds,
        all_age_initial_state_sha256=prior.array_hash(initial))


classify_readout = shared.classify_readout
immutable_group_artifacts = shared.immutable_group_artifacts


# This helper uses supplied stage, recorder and budget objects, never source-module
# schedule globals. Keep its original frozen implementation and failure contracts.
execute_stage = continuation.execute_stage


def run(args, backend_factory=prior.Backend, qualifier=memory.qualify_inputs,
        hold_qualifier=qualify_holds, clock=time.monotonic,
        rss=reference.rss_bytes, install_signals=True):
    profile = PROFILE
    plan, groups, stages, rows, slots = plans(args.requested_bit)
    recorder = Recorder(args.output_dir, clock, rss); budget = Budget(recorder, slots, args.requested_bit)
    result = dict(schema_version=1, profile=profile, status='failed', requested_bit=plan.requested_bit, groups=groups,
        rows=rows, stages=stages, calls=slots, reference_initializations=0, hold_state_initializations=0, between_group_resets=0,
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
        modules = (continuation, shared, prior, memory, contract, memory.writes, memory.recovery, memory.hold_test,
            reference, batch, prior.historical_source, prior.trajectory_source)
        paths = [Path(m.__file__) for m in modules]+[Path(__file__), ROOT/'docs/public-hold-repair.md', ROOT/'tests/test_public_hold_repair.py',
            ROOT/'docs/receipts/p6-writes.json', ROOT/'docs/receipts/p6-write-path-design.json',
            ROOT/'docs/receipts/public-control-tolerance.json', ROOT/'docs/receipts/public-control-continuation.json',
            ROOT/'tools/analyze_causal_geometry.py', ROOT/'tools/streaming_readout.py', ROOT/'tools/audit_p6_recovery.py',
            ROOT/'pyproject.toml', ROOT/'uv.lock']
        result['source'] = dict(utc=reference.utc_now(), source_commit=subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
            files=[memory.binding(p) for p in paths], arguments={**{k:str(v) for k,v in vars(args).items()}, 'profile':profile})
        recorder.json('source.json', result['source'])
        refs, provenance = qualifier(args)
        recorder.json('qualified-inputs.json', provenance)
        initial, holds = hold_qualifier(args, refs)
        recorder.json('hold-inputs.json', holds)
        result['existing_history_cost'] = holds['packet']['existing_history_cost']
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
        state = initial[0].copy(); stages[0]['initial'] = recorder.state('stage-00-initial.npy', state)
        recorder.event('inputs_frozen', initial_sha256=prior.array_hash(initial), source_phase_commit=holds['packet']['source_commit'],
            historical_endpoint_updates=120, hold_ages=list(HOLD_AGES))
        budget.check(); started = clock(); backend = backend_factory(args, recorder, budget)
        result['cold_initialization_seconds'] = clock()-started
        if hasattr(backend, 'bindings'): recorder.json('model-inputs.json', backend.bindings)
        for group in groups:
            state = initial[group['age_index']].copy(); result['hold_state_initializations'] += 1
            if group['group_id']: result['between_group_resets'] += 1
            group.update(status='running', initial_state_sha256=prior.array_hash(state))
            recorder.event('group_started', group_id=group['group_id'], age_index=group['age_index'],
                hold_age=group['hold_age'], initial_sha256=group['initial_state_sha256'])
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
        if budget.calls != MAX_CALLS or budget.counts != dict(prefix=8, comparison=16, update=1440):
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
                 'recovery-results','hold-packet','hold-states','source-run','source-audit','output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('references-sha256','reference-manifest-sha256','recovery-results-sha256','hold-packet-sha256'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    for digest in (args.references_sha256, args.reference_manifest_sha256, args.recovery_results_sha256, args.hold_packet_sha256):
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
