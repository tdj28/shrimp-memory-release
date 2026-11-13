"""Model-free endpoint-fork continuity, provenance and bounded failure retention."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_public_control_continuation as pilot
sys.path.pop(0)
from test_public_control_tolerance import FakeBackend as PriorFake
from test_public_batch_writes import OBS,caches,fake_valid,HOLD,ENDPOINT,ZERO_ENDPOINT
from test_public_command_memory import refs
from test_public_phase_challenge import validate_files


def endpoints(bit):
    value = refs()[1-bit,:2].copy()
    value[:,1] = .25  # Deliberately neither a reference start nor a settled hold state.
    return value


def continuation(args,cycles):
    initial = endpoints(args.requested_bit)
    return initial,dict(batch_initial_sha256=pilot.prior.array_hash(initial),initial_rows=[],packet=dict(
        source_commit=pilot.SOURCE_PHASE_COMMIT,existing_history_cost=dict(entire_source_run_model_calls=732,
        selected_endpoint_stage_model_calls=123,selected_endpoint_stage_batch_updates=120,
        selected_endpoint_stage_row_updates=240,entire_source_run_elapsed_seconds=1.)))


class FakeBackend(PriorFake):
    def __init__(self,args,recorder,budget):
        for name in ('planned-groups.json','planned-stages.json','planned-rows.json','planned-calls.json',
                'frozen-rows.json','continuation-inputs.json','stage-00-initial.npy'):
            assert (recorder.directory/name).is_file()
        assert not (recorder.directory/'historical-inputs.json').exists()
        assert not (recorder.directory/'phase-inputs.json').exists()
        self.recorder,self.budget,self.requested_bit = recorder,budget,args.requested_bit
        self.observations=copy.deepcopy(OBS);self.cache=caches();self.controls=None
        self.returned_call=None;self.end_inspections=0


@pytest.fixture(autouse=True)
def cheap_files(monkeypatch):
    monkeypatch.setattr(pilot.memory.os,'fsync',lambda fd:None)
    now=pilot.reference.utc_now();monkeypatch.setattr(pilot.reference,'utc_now',lambda:now)
    monkeypatch.setattr(pilot.batch,'VOCAB',4);monkeypatch.setattr(pilot.batch,'valid_cache',fake_valid)


def run(tmp_path,bit=1,backend=FakeBackend,**kwargs):
    return pilot.run(SimpleNamespace(output_dir=tmp_path/'run',requested_bit=bit),backend_factory=backend,
        qualifier=lambda args:(refs(),{}),endpoint_qualifier=continuation,rss=lambda:0,install_signals=False,**kwargs)


@pytest.mark.parametrize('bit',[0,1])
def test_both_arms_fork_literal_endpoints_carry_to_hold_and_preserve_origin(bit,tmp_path):
    prior_globals=copy.deepcopy((pilot.shared.GROUPS,pilot.shared.MAX_CALLS,pilot.prior.MAX_CALLS,pilot.memory.writes.ENDPOINTS))
    result=run(tmp_path,bit);path=tmp_path/'run'
    assert result['status']=='completed' and result['model_calls']==732
    assert result['call_counts']==dict(prefix=4,comparison=8,update=720)
    assert result['reference_initializations']==0 and result['endpoint_initializations']==2 and result['between_arm_resets']==1
    assert result['historical_scalar_equivalence'] is None and result['historical_endpoint_updates_per_row']==120
    assert result['completed_batch_updates']==720 and result['completed_row_updates']==1440
    assert result['outcome_counts']==dict(passed=4,failed=0,unresolved=0,not_evaluated=0)
    assert result['existing_history_cost']['selected_endpoint_stage_model_calls']==123
    for group in result['groups']:
        endpoint,hold=[result['stages'][i] for i in group['stage_ids']]
        left,_,_=pilot.prior.reconstruct(path,endpoint);right,_,_=pilot.prior.reconstruct(path,hold)
        assert left[0].tobytes()==endpoints(bit).tobytes()
        assert left[0].tobytes()!=refs()[1-bit,:2].tobytes()
        assert left[-1].tobytes()==right[0].tobytes()
        assert hold['controls']==HOLD
        expected=(ZERO_ENDPOINT if bit==0 else ENDPOINT).copy()
        if group['group_id']==0:expected['log10_temperature' if bit==0 else 'alpha']-=1e-4 if bit==0 else 1e-5
        assert endpoint['controls']==expected
        assert all(s['historical_comparison'] is None for s in (endpoint,hold))
        for index in group['row_ids']:
            row=result['rows'][index]
            assert row['origin_reference_phase']==row['batch_index'] and row['origin_reference_index']==1-bit
            assert row['origin_reference_label']==('ad','td')[1-bit]
            assert row['source_result_row_id']==2+row['batch_index'] and row['source_stage_id']==2
            assert row['historical_endpoint_updates']==120 and 'source_phase' not in row
            assert row['historical_comparison'] is None and row['historical_comparison_scope']==pilot.HISTORY_SCOPE
            assert row['public_action']['stages'][0]['alpha']==endpoint['nominal_controls']['alpha']
            assert row['realized_stages'][0]['controls']==endpoint['controls']
    assert prior_globals==(pilot.shared.GROUPS,pilot.shared.MAX_CALLS,pilot.prior.MAX_CALLS,pilot.memory.writes.ENDPOINTS)
    validate_files(path,result)


@pytest.mark.parametrize('fault,outcome',[('wrong','failed'),('unresolved','unresolved'),('finite-diagnostic','passed')])
def test_bad_first_arm_does_not_select_or_skip_second_arm(tmp_path,fault,outcome):
    result=run(tmp_path,backend=type('Fault',(FakeBackend,),{'group_fault':(0,fault)}))
    assert result['status']=='completed' and result['model_calls']==732
    assert [r['outcome'] for r in result['rows'][:2]]==[outcome]*2
    assert [r['outcome'] for r in result['rows'][2:]]==['passed']*2
    if fault=='finite-diagnostic':assert not result['groups'][0]['diagnostics_passed']
    validate_files(tmp_path/'run',result)


@pytest.mark.parametrize('call',[1,8,368])
def test_failure_retains_all_slots_and_any_closed_first_arm(tmp_path,call):
    result=run(tmp_path,backend=type('Fail',(FakeBackend,),{'fail_at':call}))
    assert result['status']=='failed' and result['model_calls']==call
    assert len(result['calls'])==732 and len(result['rows'])==4
    assert all(r['outcome']=='not_evaluated' for r in result['rows'][2:])
    if call>366:
        assert all(r['status']=='completed' for r in result['rows'][:2])
        assert (tmp_path/'run'/'group-00.json').is_file()
        assert np.load(tmp_path/'run'/'stage-02-initial.npy').tobytes()==endpoints(1).tobytes()
    validate_files(tmp_path/'run',result)


def test_returned_state_retained_when_deadline_trips(tmp_path):
    clock=[0.]
    class Timeout(FakeBackend):
        def forward(self,state,cached):
            output=super().forward(state,cached)
            if self.budget.calls==8:clock[0]=900.
            return output
    result=run(tmp_path,backend=Timeout,clock=lambda:clock[0])
    assert result['status']=='incomplete' and result['model_calls']==8
    assert result['stages'][0]['completed_steps']==5 and result['calls'][7]['state_carried']
    assert not result['resource_limits_passed']
    validate_files(tmp_path/'run',result)


def test_initial_write_failure_retains_assignment_without_fabricated_initialization(tmp_path,monkeypatch):
    original=pilot.Recorder.state
    def fail(self,name,value):
        if name=='stage-00-initial.npy':raise OSError('injected initial save failure')
        return original(self,name,value)
    monkeypatch.setattr(pilot.Recorder,'state',fail)
    result=run(tmp_path)
    assert result['model_calls']==result['endpoint_initializations']==0
    assert result['terminal_state_sha256']==pilot.prior.array_hash(endpoints(1))
    assert all(r['status']=='not_started' for r in result['rows'])
    validate_files(tmp_path/'run',result)


def test_marker_failure_keeps_completed_rows_and_original_fork_plan(tmp_path,monkeypatch):
    original=pilot.Recorder.json
    def fail(self,name,value,final=False):
        if name=='group-00.json':raise OSError('injected marker failure')
        return original(self,name,value,final)
    monkeypatch.setattr(pilot.Recorder,'json',fail)
    result=run(tmp_path)
    assert result['status']==result['groups'][0]['status']=='failed' and result['model_calls']==366
    assert all(r['status']=='completed' for r in result['rows'][:2])
    assert all(r['status']=='not_started' for r in result['rows'][2:])
    validate_files(tmp_path/'run',result)


def test_schedule_guards_and_reserve_are_fixed(tmp_path):
    _,_,stages,_,slots=pilot.plans(0);recorder=pilot.Recorder(tmp_path/'budget',clock=lambda:0.,rss=lambda:0)
    try:
        budget=pilot.Budget(recorder,slots,0)
        changed=copy.deepcopy(stages[0]);changed['controls']=copy.deepcopy(stages[2]['controls'])
        with pytest.raises(ValueError,match='realized-control'):budget.begin(changed)
        budget.begin(stages[0]);slots[0]['profile']='public-two-phase-challenge'
        with pytest.raises(RuntimeError,match='fixed schedule'):budget.claim('prefix')
        assert pilot.derive_design(0)['schedule']['total_path_updates_per_row']==480
        assert pilot.derive_design(0)['limits']['internal_seconds']==900
        recorder.size=lambda:pilot.WORKING_OUTPUT
        with pytest.raises(RuntimeError):recorder.admit(1)
        recorder.admit(1,final=True)
    finally:recorder.stream.close()


@pytest.mark.parametrize('bit',[-1,2,True,1.,None])
def test_invalid_bit_refused_before_output(tmp_path,bit):
    with pytest.raises(ValueError):run(tmp_path,bit)
    assert not (tmp_path/'run').exists()


def test_existing_output_cannot_be_reused(tmp_path):
    (tmp_path/'run').mkdir();(tmp_path/'run'/'keep').write_text('keep')
    with pytest.raises(FileExistsError):run(tmp_path)
    assert (tmp_path/'run'/'keep').read_text()=='keep'


def endpoint_fixture(tmp_path,monkeypatch,bit):
    """Independent tiny saved-data packet with the actual filenames and block format."""
    source=tmp_path/'source-run';source.mkdir();packet_path=tmp_path/'endpoint-packet.json'
    endpoint_path=tmp_path/'endpoint-states.npz';audit_path=tmp_path/'independent-review.json'
    nominal=(ZERO_ENDPOINT if bit==0 else ENDPOINT).copy();control=nominal.copy()
    control['log10_temperature' if bit==0 else 'alpha']-=1e-4 if bit==0 else 1e-5
    states=np.repeat(refs()[1-bit,:2][None],121,axis=0)
    for i in range(1,121):
        states[i,:,0]=(states[i,:,0]+i)%6;states[i,:,1]=.25
    np.save(source/'stage-02-initial.npy',states[0]);np.save(source/'stage-03-initial.npy',states[-1])
    def absolute(path):return pilot.reference.require_hash(path,pilot.reference.sha256(path))
    def relative(path):return {**absolute(path),'path':path.name}
    blocks=[]
    for end in range(12,121,12):
        path=source/f'stage-02-block-{end:03d}.npz'
        np.savez(path,states=states[end-11:end+1],token_ids=np.ones((12,2),dtype=np.int64),
            step_seconds=np.full(12,.01,dtype=np.float64),first_state_index=np.asarray(end-11,dtype=np.int64),
            last_state_index=np.asarray(end,dtype=np.int64),**control)
        blocks.append(relative(path))
    stage=dict(stage_id=2,group='full-negative-offset',controls=control,status='completed',steps=120,completed_steps=120,
        initial=relative(source/'stage-02-initial.npy'),blocks=blocks)
    raw=dict(profile='public-two-phase-challenge',requested_bit=bit,source={'source_commit':pilot.SOURCE_PHASE_COMMIT},
        stages=[{}, {}, stage, {}])
    for name in ('source.json','design.json','phase-inputs.json','qualified-inputs.json','stage-02-trajectory.json'):
        (source/name).write_text('{}')
    (source/'result.json').write_text(json.dumps(raw))
    source_rows=[dict(row_id=i,origin_phase=i%2,status='completed') for i in range(4)]
    audit_path.write_text(json.dumps(dict(status='passed',profile='public-two-phase-challenge',requested_bit=bit,
        audited_source_commit=pilot.SOURCE_PHASE_COMMIT,all_rows_preserved=True,rows=source_rows)))
    np.savez(endpoint_path,states=states[-1],origin_reference_index=np.asarray([1-bit]*2,dtype=np.int64),
        origin_reference_phase=np.asarray([0,1],dtype=np.int64))
    packet=dict(status='saved_arrays_verified_no_model_replay',requested_bit=bit,source_commit=pilot.SOURCE_PHASE_COMMIT,
        source_stage_id=2,source_group='full-negative-offset',source_completed_endpoint_updates=120,
        source_endpoint_controls=control,nominal_endpoint_controls=nominal,hold_controls=HOLD,
        source_audit=absolute(audit_path),source_result=absolute(source/'result.json'),endpoint_npz=absolute(endpoint_path),
        source_artifact_bindings=[absolute(p) for p in sorted(source.iterdir())],full_source_rows=source_rows,
        endpoint_trajectory_array_sha256=pilot.prior.array_hash(states),endpoint_batch_array_sha256=pilot.prior.array_hash(states[-1]),
        hold_stage3_initial_bitwise_equal=True,source_endpoint_call_ids=list(range(367,490)),
        existing_history_cost={'entire_source_run_model_calls':732,'selected_endpoint_stage_model_calls':123},
        ordering=[dict(batch_index=i,origin_reference_index=1-bit,origin_reference_label=('ad','td')[1-bit],origin_reference_phase=i,
            source_result_row_id=2+i,initial_reference_array_sha256=pilot.prior.array_hash(refs()[1-bit,i]),
            endpoint_array_sha256=pilot.prior.array_hash(states[-1,i])) for i in (0,1)])
    args=SimpleNamespace(requested_bit=bit,endpoint_packet=packet_path,endpoint_states=endpoint_path,source_run=source,source_audit=audit_path)
    def save_packet():
        packet_path.write_text(json.dumps(packet));args.endpoint_packet_sha256=pilot.reference.sha256(packet_path)
        values=list(pilot.PACKET_HASHES);values[bit]=args.endpoint_packet_sha256;monkeypatch.setattr(pilot,'PACKET_HASHES',tuple(values))
    save_packet()
    return args,packet,states,absolute,save_packet


@pytest.mark.parametrize('bit',[0,1])
def test_actual_packet_file_contract_and_zero_forward_preflight(tmp_path,monkeypatch,bit):
    args,packet,states,_,_=endpoint_fixture(tmp_path,monkeypatch,bit)
    def forbidden(*args,**kwargs):raise AssertionError('model must not load during preflight')
    monkeypatch.setattr(pilot.prior,'Backend',forbidden)
    initial,evidence=pilot.qualify_endpoints(args,refs())
    assert initial.tobytes()==states[-1].tobytes() and initial.tobytes()!=refs()[1-bit,:2].tobytes()
    assert len(evidence['source_artifact_bindings'])==18
    assert [r['origin_reference_phase'] for r in evidence['initial_rows']]==[0,1]
    preflight=pilot.preflight(args,qualifier=lambda args:(refs(),{}))
    assert preflight['status']=='passed' and preflight['model_calls']==0 and preflight['model_loaded'] is False
    assert preflight['continuation_inputs']['packet']==packet


@pytest.mark.parametrize('tamper',['packet-hash','npz-digest','phase-order','origin-label','history-count','source-stage',
    'source-audit','duplicate-artifact','missing-artifact','hold-start','settled-array','gapped-block'])
def test_corrupted_endpoint_origin_or_continuity_fails_before_backend(tmp_path,monkeypatch,tamper):
    args,packet,states,absolute,save_packet=endpoint_fixture(tmp_path,monkeypatch,0)
    def refresh_source(path):
        packet['source_artifact_bindings']=[absolute(path) if p['path']==str(path) else p for p in packet['source_artifact_bindings']]
        if path.name=='result.json':packet['source_result']=absolute(path)
    if tamper=='packet-hash':args.endpoint_packet_sha256='0'*64
    elif tamper=='npz-digest':args.endpoint_states.write_bytes(b'changed')
    elif tamper=='phase-order':packet['ordering'][0]['origin_reference_phase']=1;save_packet()
    elif tamper=='origin-label':packet['ordering'][0]['origin_reference_label']='ad';save_packet()
    elif tamper=='history-count':packet['source_completed_endpoint_updates']=240;save_packet()
    elif tamper=='source-stage':
        path=args.source_run/'result.json';raw=json.loads(path.read_text());raw['stages'][2]['controls']=HOLD.copy()
        path.write_text(json.dumps(raw));refresh_source(path);save_packet()
    elif tamper=='source-audit':
        value=json.loads(args.source_audit.read_text());value['status']='failed';args.source_audit.write_text(json.dumps(value))
        packet['source_audit']=absolute(args.source_audit);save_packet()
    elif tamper=='duplicate-artifact':packet['source_artifact_bindings'].append(packet['source_artifact_bindings'][0]);save_packet()
    elif tamper=='missing-artifact':packet['source_artifact_bindings'].pop();save_packet()
    elif tamper=='hold-start':
        path=args.source_run/'stage-03-initial.npy';np.save(path,states[0]);refresh_source(path);save_packet()
    elif tamper=='settled-array':
        np.savez(args.endpoint_states,states=refs()[0,:2],origin_reference_index=np.asarray([1,1],dtype=np.int64),
            origin_reference_phase=np.asarray([0,1],dtype=np.int64))
        packet['endpoint_npz']=absolute(args.endpoint_states);save_packet()
    elif tamper=='gapped-block':
        path=args.source_run/'stage-02-block-012.npz'
        with np.load(path) as z:values={k:z[k] for k in z.files}
        values['first_state_index']=np.asarray(2,dtype=np.int64);np.savez(path,**values);refresh_source(path)
        result=args.source_run/'result.json';raw=json.loads(result.read_text());raw['stages'][2]['blocks'][0]={**absolute(path),'path':path.name}
        result.write_text(json.dumps(raw));refresh_source(result);save_packet()
    with pytest.raises(ValueError):pilot.qualify_endpoints(args,refs())
