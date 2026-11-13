"""Literal-age forks, neutral controls and complete failure evidence; no model."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_public_hold_repair as pilot
sys.path.pop(0)
from test_public_batch_writes import FakeBackend as BaseFake, OBS,caches,fake_valid,HOLD,ENDPOINT,ZERO_ENDPOINT
from test_public_command_memory import refs


def selected(bit):
    return np.stack([refs()[1-bit,[2,3]],refs()[1-bit,[4,5]]])


def hold_inputs(args,cycles):
    initial=selected(args.requested_bit)
    return initial,dict(all_age_initial_sha256=pilot.prior.array_hash(initial),age_batches=[],
        input_observations=[],packet=dict(source_commit=pilot.SOURCE_PHASE_COMMIT,
        existing_history_cost=dict(entire_source_run_model_calls=732,selected_endpoint_stage_model_calls=123,
        selected_hold_stage_model_calls=243)))


class FakeBackend(BaseFake):
    group_fault=None
    def __init__(self,args,recorder,budget):
        for name in ('planned-groups.json','planned-stages.json','planned-rows.json','planned-calls.json',
                'frozen-rows.json','hold-inputs.json','stage-00-initial.npy'):
            assert (recorder.directory/name).is_file()
        assert not (recorder.directory/'continuation-inputs.json').exists()
        self.recorder,self.budget=recorder,budget
        self.observations=copy.deepcopy(OBS);self.cache=caches();self.controls=None
        self.returned_call=None;self.end_inspections=0
    def forward(self,state,cached):
        previous=self.fault
        slot=self.budget.slots[self.budget.calls-1]
        if self.group_fault is not None and slot['group_id']==self.group_fault[0]:self.fault=self.group_fault[1]
        try:return super().forward(state,cached)
        finally:self.fault=previous


@pytest.fixture(autouse=True)
def cheap_files(monkeypatch):
    monkeypatch.setattr(pilot.memory.os,'fsync',lambda fd:None)
    now=pilot.reference.utc_now();monkeypatch.setattr(pilot.reference,'utc_now',lambda:now)
    monkeypatch.setattr(pilot.batch,'VOCAB',4);monkeypatch.setattr(pilot.batch,'valid_cache',fake_valid)


def run(tmp_path,bit=1,backend=FakeBackend,**kwargs):
    return pilot.run(SimpleNamespace(output_dir=tmp_path/'run',requested_bit=bit),backend_factory=backend,
        qualifier=lambda args:(refs(),{}),hold_qualifier=hold_inputs,rss=lambda:0,install_signals=False,**kwargs)


def validate_files(path,result):
    assert json.loads((path/'result.json').read_text())==result
    assert len(result['groups'])==4 and len(result['rows'])==8 and len(result['stages'])==8 and len(result['calls'])==1464
    manifest=json.loads((path/'manifest.json').read_text())
    assert {x['path'] for x in manifest['files']}=={p.name for p in path.iterdir()}-{'manifest.json'}
    for x in manifest['files']:
        raw=(path/x['path']).read_bytes();assert len(raw)==x['bytes'] and hashlib.sha256(raw).hexdigest()==x['sha256']
    for stage in result['stages']:
        if stage['trajectory']:
            states,tokens,times=pilot.prior.reconstruct(path,stage)
            assert len(states)==stage['completed_steps']+1 and len(tokens)==len(times)==stage['completed_steps']
    for row in result['rows']:assert json.loads((path/f'row-{row["row_id"]:02d}.json').read_text())==row
    for group in result['groups']:
        marker=path/f'group-{group["group_id"]:02d}.json'
        if group['status']!='completed':assert not marker.exists();continue
        assert json.loads(marker.read_text())==group
        expected={p.name for p in path.iterdir() if any(p.name.startswith(f'stage-{i:02d}-') for i in group['stage_ids'])}
        expected|={slot['artifact']['path'] for slot in result['calls'] if slot['group_id']==group['group_id'] and 'artifact' in slot}
        expected|={f'row-{i:02d}-completed.json' for i in group['row_ids']}
        assert {x['path'] for x in group['artifacts']}==expected
        assert all(x in manifest['files'] for x in group['artifacts'])


@pytest.mark.parametrize('bit',[0,1])
def test_complete_fixed_age_forks_controls_carry_and_canonical_action(tmp_path,bit):
    old=copy.deepcopy((pilot.continuation.MAX_CALLS,pilot.shared.GROUPS,pilot.prior.MAX_CALLS,pilot.memory.writes.ENDPOINTS))
    result=run(tmp_path,bit);path=tmp_path/'run'
    assert result['status']=='completed' and result['model_calls']==1464
    assert result['call_counts']==dict(prefix=8,comparison=16,update=1440)
    assert result['hold_state_initializations']==4 and result['between_group_resets']==3 and result['reference_initializations']==0
    assert result['completed_row_updates']==2880
    assert result['outcome_counts']==dict(passed=4,failed=4,unresolved=0,not_evaluated=0)
    frozen=json.loads((path/'planned-groups.json').read_text());assert all(g['status']=='not_started' for g in frozen)
    events=[json.loads(x) for x in (path/'events.jsonl').read_text().splitlines()]
    freeze=next(x for x in events if x['event']=='inputs_frozen')
    assert freeze['initial_sha256']==pilot.prior.array_hash(selected(bit))
    for g in result['groups']:
        age=g['group_id']//2;repair=bool(g['group_id']%2)
        a,h=[result['stages'][i] for i in g['stage_ids']]
        left,_,_=pilot.prior.reconstruct(path,a);right,_,_=pilot.prior.reconstruct(path,h)
        assert left[0].tobytes()==selected(bit)[age].tobytes()
        assert left[-1].tobytes()==right[0].tobytes()
        assert a['name']=='intervention' and a['steps']==120 and h['name']=='hold' and h['steps']==240
        nominal=ZERO_ENDPOINT if bit==0 else ENDPOINT
        assert a['controls']==(nominal if repair else HOLD) and a['nominal_controls']==nominal
        assert h['controls']==h['nominal_controls']==HOLD
        assert g['initial_state_sha256']==pilot.prior.array_hash(selected(bit)[age])
        for i in g['row_ids']:
            row=result['rows'][i]
            assert row['hold_age']==(36,240)[age] and row['historical_hold_updates']==row['hold_age']
            assert row['historical_total_updates']==120+row['hold_age'] and row['total_path_updates']==480+row['hold_age']
            assert row['source_stage_id']==3 and row['source_result_row_id']==2+row['batch_index']
            assert row['public_action']['stages'][0]['alpha']==nominal['alpha']
            assert row['realized_stages'][0]['controls']==a['controls'] and row['dispatch_count']==1
            assert row['numerical_diagnostics_passed']
            assert row['outcome']==('passed' if repair else 'failed')
    assert old==(pilot.continuation.MAX_CALLS,pilot.shared.GROUPS,pilot.prior.MAX_CALLS,pilot.memory.writes.ENDPOINTS)
    validate_files(path,result)


@pytest.mark.parametrize('fault,outcome',[('wrong','failed'),('unresolved','unresolved'),('finite-diagnostic','passed')])
def test_bad_repair_does_not_skip_later_control_or_repair(tmp_path,fault,outcome):
    result=run(tmp_path,backend=type('Fault',(FakeBackend,),{'group_fault':(1,fault)}))
    assert result['status']=='completed' and result['model_calls']==1464
    assert [x['outcome'] for x in result['rows'][2:4]]==[outcome]*2
    assert [x['outcome'] for x in result['rows'][6:]]==['passed']*2
    if fault=='finite-diagnostic':assert not result['groups'][1]['diagnostics_passed']
    validate_files(tmp_path/'run',result)


@pytest.mark.parametrize('call',[1,8,734,1100])
def test_failure_keeps_all_slots_partial_blocks_and_closed_groups(tmp_path,call):
    result=run(tmp_path,backend=type('Fail',(FakeBackend,),{'fail_at':call}))
    assert result['status']=='failed' and result['model_calls']==call
    expected_started=(call-1)//366+1
    assert result['hold_state_initializations']==expected_started
    assert result['between_group_resets']==expected_started-1
    assert all(x['status']=='completed' for x in result['groups'][:expected_started-1])
    validate_files(tmp_path/'run',result)


def test_deadline_after_return_preserves_actual_carried_state(tmp_path):
    clock=[0.]
    class Timeout(FakeBackend):
        def forward(self,state,cached):
            output=super().forward(state,cached)
            if self.budget.calls==8:clock[0]=1800.
            return output
    result=run(tmp_path,backend=Timeout,clock=lambda:clock[0])
    assert result['status']=='incomplete' and result['model_calls']==8
    assert result['stages'][0]['completed_steps']==5 and result['calls'][7]['state_carried']
    assert not result['resource_limits_passed'];validate_files(tmp_path/'run',result)


@pytest.mark.parametrize('fault',['nonfinite','dtype','precision','input','cache','expanded','length','cache-inspect'])
def test_structural_or_precision_fault_aborts_without_row_drop(tmp_path,fault):
    result=run(tmp_path,backend=type('Fault',(FakeBackend,),{'fault':fault}))
    assert result['status']=='failed' and result['model_calls']==8
    assert all(x['status']=='not_started' for x in result['rows'][2:])
    validate_files(tmp_path/'run',result)


def test_reset_mutation_count_precedes_initial_save_failure(tmp_path,monkeypatch):
    old=pilot.Recorder.state
    def fail(self,name,value):
        if name=='stage-04-initial.npy':raise OSError('injected late initial save')
        return old(self,name,value)
    monkeypatch.setattr(pilot.Recorder,'state',fail)
    result=run(tmp_path)
    assert result['model_calls']==732 and result['hold_state_initializations']==3 and result['between_group_resets']==2
    assert result['terminal_state_sha256']==pilot.prior.array_hash(selected(1)[1])
    assert result['groups'][2]['initial_state_sha256']==pilot.prior.array_hash(selected(1)[1])
    validate_files(tmp_path/'run',result)


def test_marker_failure_preserves_completed_rows_but_not_completion_claim(tmp_path,monkeypatch):
    old=pilot.Recorder.json
    def fail(self,name,value,final=False):
        if name=='group-00.json':raise OSError('injected marker failure')
        return old(self,name,value,final)
    monkeypatch.setattr(pilot.Recorder,'json',fail)
    result=run(tmp_path)
    assert result['status']==result['groups'][0]['status']=='failed' and result['model_calls']==366
    assert all(x['status']=='completed' for x in result['rows'][:2])
    validate_files(tmp_path/'run',result)


def test_schedule_and_resource_limits_refuse_mutated_age_or_controls(tmp_path):
    _,_,stages,_,slots=pilot.plans(0);rec=pilot.Recorder(tmp_path/'budget',clock=lambda:0.,rss=lambda:0)
    try:
        budget=pilot.Budget(rec,slots,0)
        for key,value in [('hold_age',240),('age_index',1),('controls',ZERO_ENDPOINT)]:
            changed=copy.deepcopy(stages[0]);changed[key]=value
            with pytest.raises(ValueError,match='realized-control'):budget.begin(changed)
        budget.begin(stages[0]);slots[0]['profile']='public-endpoint-continuation'
        with pytest.raises(RuntimeError,match='fixed schedule'):budget.claim('prefix')
        design=pilot.derive_design(0)
        assert design['schedule']['total_calls']==1464 and design['limits']['internal_seconds']==1800
        assert design['artifact_envelope']['total_working_requirement_bytes']<pilot.WORKING_OUTPUT
        rec.size=lambda:pilot.WORKING_OUTPUT
        with pytest.raises(RuntimeError):rec.admit(1)
        rec.admit(1,final=True)
    finally:rec.stream.close()


@pytest.mark.parametrize('bit',[-1,2,True,1.,None])
def test_bad_bit_precedes_output(tmp_path,bit):
    with pytest.raises(ValueError):run(tmp_path,bit)
    assert not (tmp_path/'run').exists()


def test_existing_output_is_preserved(tmp_path):
    (tmp_path/'run').mkdir();(tmp_path/'run'/'keep').write_text('keep')
    with pytest.raises(FileExistsError):run(tmp_path)
    assert (tmp_path/'run'/'keep').read_text()=='keep'


def packet_fixture(tmp_path,monkeypatch,bit):
    """Extend the established actual-name endpoint fixture with240saved hold updates."""
    from test_public_control_continuation import endpoint_fixture
    old,oldpacket,endpoints,absolute,_=endpoint_fixture(tmp_path,monkeypatch,bit)
    source=old.source_run;hold=np.repeat(endpoints[-1][None],241,axis=0)
    for step in range(1,241):
        hold[step,:,0]=(hold[step,:,0]+step)%6
        hold[step,:,2]=step*1e-7
    blocks=[]
    def relative(p):return {**absolute(p),'path':p.name}
    for end in range(12,241,12):
        p=source/f'stage-03-block-{end:03d}.npz'
        np.savez(p,states=hold[end-11:end+1],token_ids=np.ones((12,2),dtype=np.int64),
            step_seconds=np.full(12,.01,dtype=np.float64),first_state_index=np.asarray(end-11,dtype=np.int64),
            last_state_index=np.asarray(end,dtype=np.int64),**HOLD)
        blocks.append(relative(p))
    (source/'stage-03-trajectory.json').write_text('{}')
    raw=json.loads((source/'result.json').read_text())
    raw['stages'][3]=dict(stage_id=3,group='full-negative-offset',controls=HOLD,status='completed',steps=240,
        completed_steps=240,initial=relative(source/'stage-03-initial.npy'),blocks=blocks)
    (source/'result.json').write_text(json.dumps(raw))
    states_path=tmp_path/'hold-states.npz';initial=hold[[36,240]]
    np.savez(states_path,states=initial,hold_ages=np.asarray([36,240],dtype=np.int64),
        origin_reference_index=np.asarray([1-bit]*2,dtype=np.int64),origin_reference_phase=np.asarray([0,1],dtype=np.int64))
    ref_path=tmp_path/'references.npz';np.savez(ref_path,states=refs())
    age_batches=[];observations=[]
    for ai,age in enumerate((36,240)):
        rows=[]
        for phase in (0,1):
            origin=dict(batch_index=phase,origin_reference_index=1-bit,origin_reference_label=('ad','td')[1-bit],
                origin_reference_phase=phase,source_result_row_id=2+phase)
            rows.append(dict(**origin,initial_reference_array_sha256=pilot.prior.array_hash(refs()[1-bit,phase]),
                state_array_sha256=pilot.prior.array_hash(initial[ai,phase])))
            observations.append(dict(**origin,age_index=ai,hold_age=age,prefix_updates=age,
                initial_hold_state_included_but_not_pushed=True,prefix_array_sha256=pilot.prior.array_hash(hold[:age+1,phase]),
                observation=dict(geometry_decision=None,strict_decision=None,
                    checkpoints=[dict(steps=s,geometry_label=None,strict_label=None) for s in range(12,age+1,12)])))
        age_batches.append(dict(age_index=ai,hold_age=age,batch_array_sha256=pilot.prior.array_hash(initial[ai]),initial_rows=rows))
    source_files=['tools/analyze_causal_geometry.py','tools/streaming_readout.py','tools/audit_p6_recovery.py']
    packet=dict(status='saved_arrays_verified_no_model_replay',profile='public-hold-repair-inputs',requested_bit=bit,
        source_commit=pilot.SOURCE_PHASE_COMMIT,source_stage_id=3,source_endpoint_stage_id=2,
        source_group='full-negative-offset',historical_endpoint_updates=120,source_completed_hold_updates=240,hold_ages=[36,240],
        source_endpoint_controls=oldpacket['source_endpoint_controls'],nominal_endpoint_controls=oldpacket['nominal_endpoint_controls'],
        hold_controls=HOLD,source_audit=oldpacket['source_audit'],full_source_rows=oldpacket['full_source_rows'],
        source_result=absolute(source/'result.json'),source_artifact_bindings=[absolute(p) for p in sorted(source.iterdir())],
        reference_binding=absolute(ref_path),states_npz=absolute(states_path),
        endpoint_trajectory_array_sha256=pilot.prior.array_hash(endpoints),hold_trajectory_array_sha256=pilot.prior.array_hash(hold),
        endpoint_to_hold_initial_bitwise_equal=True,all_states_array_sha256=pilot.prior.array_hash(initial),
        age_batches=age_batches,source_endpoint_call_ids=list(range(367,490)),source_hold_call_ids=list(range(490,733)),
        observer_source_bindings=[dict(**absolute(pilot.ROOT/x),relative_path=x) for x in source_files],
        observer_policy=dict(checkpoint_interval=12,consecutive_labels=3,reference_sha256=pilot.reference.sha256(ref_path)),
        observations=observations,existing_history_cost=oldpacket['existing_history_cost'])
    args=SimpleNamespace(requested_bit=bit,hold_packet=tmp_path/'hold-packet.json',hold_states=states_path,
        source_run=source,source_audit=old.source_audit,references=ref_path)
    def save():
        args.hold_packet.write_text(json.dumps(packet));args.hold_packet_sha256=pilot.reference.sha256(args.hold_packet)
        values=list(pilot.PACKET_HASHES);values[bit]=args.hold_packet_sha256;monkeypatch.setattr(pilot,'PACKET_HASHES',tuple(values))
    save();return args,packet,hold,absolute,save


@pytest.mark.parametrize('bit',[0,1])
def test_actual_file_schema_and_missing_observer_decisions_do_not_filter_inputs(tmp_path,monkeypatch,bit):
    args,packet,hold,_,_=packet_fixture(tmp_path,monkeypatch,bit)
    def forbidden(*a,**kw):raise AssertionError('no model during qualification')
    monkeypatch.setattr(pilot.prior,'Backend',forbidden)
    initial,evidence=pilot.qualify_holds(args,refs())
    assert initial.tobytes()==hold[[36,240]].tobytes()
    assert len(evidence['source_artifact_bindings'])==39 and len(evidence['age_batches'])==2
    assert all(x['observation']['geometry_decision'] is None for x in evidence['input_observations'])
    result=pilot.preflight(args,qualifier=lambda a:(refs(),{}))
    assert result['status']=='passed' and result['model_calls']==0 and not result['model_loaded']
    assert result['hold_inputs']['packet']==packet


@pytest.mark.parametrize('tamper',['hash','npz-digest','age-order','origin-phase','source-hold-control','history-count',
    'missing-artifact','duplicate-artifact','hold-start','age-reference-substitution','observation-prefix','observer-source',
    'reference-digest','audit-status','gapped-block'])
def test_rehashed_tampered_packet_or_source_fails_before_backend(tmp_path,monkeypatch,tamper):
    args,p,hold,absolute,save=packet_fixture(tmp_path,monkeypatch,0)
    def refresh(path):
        p['source_artifact_bindings']=[absolute(path) if x['path']==str(path) else x for x in p['source_artifact_bindings']]
        if path.name=='result.json':p['source_result']=absolute(path)
    if tamper=='hash':args.hold_packet_sha256='0'*64
    elif tamper=='npz-digest':args.hold_states.write_bytes(b'bad')
    elif tamper=='age-order':p['hold_ages']=[240,36];save()
    elif tamper=='origin-phase':p['age_batches'][0]['initial_rows'][1]['origin_reference_phase']=0;save()
    elif tamper=='source-hold-control':
        path=args.source_run/'result.json';r=json.loads(path.read_text());r['stages'][3]['controls']=ZERO_ENDPOINT
        path.write_text(json.dumps(r));refresh(path);save()
    elif tamper=='history-count':p['historical_endpoint_updates']=60;save()
    elif tamper=='missing-artifact':p['source_artifact_bindings'].pop();save()
    elif tamper=='duplicate-artifact':p['source_artifact_bindings'].append(p['source_artifact_bindings'][0]);save()
    elif tamper=='hold-start':
        path=args.source_run/'stage-03-initial.npy';np.save(path,refs()[1,:2]);refresh(path);save()
    elif tamper=='age-reference-substitution':
        with np.load(args.hold_states) as z:v={k:z[k] for k in z.files}
        v['states'][0]=refs()[1,:2];np.savez(args.hold_states,**v);p['states_npz']=absolute(args.hold_states);save()
    elif tamper=='observation-prefix':p['observations'][0]['prefix_array_sha256']='0'*64;save()
    elif tamper=='observer-source':p['observer_source_bindings'][0]['sha256']='0'*64;save()
    elif tamper=='reference-digest':args.references.write_bytes(b'bad')
    elif tamper=='audit-status':
        v=json.loads(args.source_audit.read_text());v['status']='failed';args.source_audit.write_text(json.dumps(v))
        p['source_audit']=absolute(args.source_audit);save()
    elif tamper=='gapped-block':
        path=args.source_run/'stage-03-block-012.npz'
        with np.load(path) as z:v={k:z[k] for k in z.files}
        v['first_state_index']=np.asarray(2,dtype=np.int64);np.savez(path,**v);refresh(path)
        result=args.source_run/'result.json';r=json.loads(result.read_text());r['stages'][3]['blocks'][0]={**absolute(path),'path':path.name}
        result.write_text(json.dumps(r));refresh(result);save()
    with pytest.raises(ValueError):pilot.qualify_holds(args,refs())
