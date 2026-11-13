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
import run_public_repair_handoff as pilot
sys.path.pop(0)
from test_public_batch_writes import FakeBackend as BaseFake, OBS,caches,fake_valid,HOLD,ENDPOINT,ZERO_ENDPOINT
from test_public_command_memory import refs


def selected(bit):
    return np.stack([refs()[1-bit,phases] for phases in ([0,1],[2,3],[4,5],[1,2])])


def hold_inputs(args,cycles):
    initial=selected(args.requested_bit)
    return initial,dict(all_age_initial_sha256=pilot.prior.array_hash(initial),age_batches=[],
        input_observations=[],packet=dict(source_commit=pilot.SOURCE_PHASE_COMMIT,
        source_producer_total_calls=1464,slots=[]))


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
    assert result['outcome_counts']==dict(passed=8,failed=0,unresolved=0,not_evaluated=0)
    frozen=json.loads((path/'planned-groups.json').read_text());assert all(g['status']=='not_started' for g in frozen)
    events=[json.loads(x) for x in (path/'events.jsonl').read_text().splitlines()]
    freeze=next(x for x in events if x['event']=='inputs_frozen')
    assert freeze['initial_sha256']==pilot.prior.array_hash(selected(bit))
    for g in result['groups']:
        age=g['group_id'];repair=True
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
            assert row['hold_age']==(36,240,36,240)[age] and row['historical_hold_updates']==row['hold_age']+(36,240)[age//2]
            assert row['historical_total_updates']==240+row['historical_hold_updates'] and row['total_path_updates']==600+row['historical_hold_updates']
            assert row['source_stage_id']==3+4*(age//2) and row['source_result_row_id']==2+4*(age//2)+row['batch_index']
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
    assert result['terminal_state_sha256']==pilot.prior.array_hash(selected(1)[2])
    assert result['groups'][2]['initial_state_sha256']==pilot.prior.array_hash(selected(1)[2])
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
        for key,value in [('hold_age',240),('age_index',1),('controls',HOLD)]:
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


