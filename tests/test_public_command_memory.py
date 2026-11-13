"""Public dispatch, preservation and actual file contracts, without a real model."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import run_public_command_memory as memory
sys.path.pop(0)

# Independent literal oracle, not imported from producer/parser constants.
TEXT = (
    'Write bit 0 to register R0.',
    'Write bit 2 to register R0.',
    'Write bit 1 to register R0.',
    'Write bit 0 to register R0. Then erase the latch.',
    'Write bit 1 to register R0.',
    'Do not write bit 0 to register R0.',
    'Write bit 0 to register R0.',
)
GOLD = (0, None, 1, None, 1, None, 0)
ENDPOINT = {0:(0.07293697809562831, -0.15350), 1:(0.07299785589927693,-0.15395151820007266)}
HOLD = (0.07293697809562831,-0.15395151820007266)


def refs():
    result = np.zeros((2,6,896), dtype=np.float64)
    result[:,:,0] = np.arange(6)
    result[1,:,1] = .1
    return result


def args(tmp_path):
    return SimpleNamespace(output_dir=tmp_path/'run')


class FakeBackend:
    fail_at = None
    cache_fail = False
    wrong = False
    unresolved = False
    nonfinite = False
    wrong_dtype = False
    wrong_shape = False
    time_trip = None

    def __init__(self, args, recorder, budget):
        self.recorder, self.budget = recorder, budget
        self.observations = {'synthetic':True}
        self.controls = None
        self.inputs = []

    def fail(self):
        if self.budget.calls == self.fail_at: raise RuntimeError('injected model failure')

    def prepare(self, controls):
        self.controls = controls.copy()
        self.budget.claim('native64_prefix')
        self.fail()

    def forward(self, state, cached):
        self.fail(); self.inputs.append((self.budget.calls, state.copy(), self.controls.copy(), cached))
        y = state.copy(); y[0] = (y[0]+1)%6
        pair = (self.controls['alpha'],self.controls['log10_temperature'])
        if pair == ENDPOINT[0]: y[1] = 0
        elif pair == ENDPOINT[1]: y[1] = 0 if self.wrong else .1
        elif pair != HOLD: raise AssertionError('unexpected controls')
        if self.unresolved: y[2] += .01
        if self.cache_fail and not cached: y[4] += .01
        probabilities = np.array([.25,.75], dtype=np.float64)
        if self.budget.calls == 4:
            if self.nonfinite: y[3] = np.nan
            if self.wrong_dtype: y = y.astype(np.float32)
            if self.wrong_shape: y = y[:-1]
        if self.time_trip and self.budget.calls == self.time_trip[0]: self.time_trip[1][0] = 900
        return y, probabilities


@pytest.fixture(autouse=True)
def no_fsync(monkeypatch):
    # All tests use a temporary, synthetic file tree. Production keeps fsync.
    monkeypatch.setattr(memory.os, 'fsync', lambda fd:None)
    frozen_utc=memory.reference.utc_now()
    monkeypatch.setattr(memory.reference, 'utc_now', lambda:frozen_utc)


def run(tmp_path, backend=FakeBackend, **kwargs):
    return memory.run(args(tmp_path), backend_factory=backend,
        qualifier=lambda args:(refs(), {'synthetic':True}), install_signals=False, rss=lambda:0, **kwargs)


def check_inventory(path, result):
    assert {f.name for f in path.glob('message-??.json')} == {f'message-{i:02d}.json' for i in range(7)}
    for i,row in enumerate(result['rows']):
        assert json.loads((path/f'message-{i:02d}.json').read_text()) == row
    manifest=json.loads((path/'manifest.json').read_text())
    assert {r['path'] for r in manifest['files']} == {p.name for p in path.iterdir()}-{'manifest.json'}
    for item in manifest['files']:
        data=(path/item['path']).read_bytes()
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
    assert json.loads((path/'result.json').read_text()) == result


def test_fixed_stream_has_no_gold_fields_and_derived_budget():
    assert tuple(memory.MESSAGES)==TEXT
    plan=memory.message_plan()
    assert all(not any('gold' in key or 'expected_bit' in key or 'desired' in key for key in row) for row in plan)
    d=memory.derive_design()
    assert d['schedule']==dict(accepted_messages=4,rejected_messages=3,stages=8,numerical_updates=1440,
                              prefix_calls=8,comparison_calls=16,total_calls=1464)
    assert d['limits']['internal_seconds']==840 and d['limits']['external_seconds']==900
    assert d['limits']['total_raw_bytes']==72*1024**2


def test_full_stream_real_file_contract_and_independent_control_oracle(tmp_path):
    result=run(tmp_path); path=tmp_path/'run'
    assert result['status']=='completed'
    assert result['model_calls']==1464 and result['call_counts']==dict(prefix=8,comparison=16,update=1440)
    assert result['completed_messages']==7 and result['accepted_messages']==4 and result['rejected_messages']==3
    check_inventory(path,result)
    last=refs()[0,0].tobytes()
    for row,bit in zip(result['rows'],GOLD):
        before=np.load(path/row['state_before']['path'],allow_pickle=False)
        after=np.load(path/row['state_after']['path'],allow_pickle=False)
        assert before.tobytes()==last
        assert hashlib.sha256(before.tobytes()).hexdigest()==row['state_before']['array_sha256']
        if bit is None:
            assert row['parser_outcome']=='rejected' and row['dispatch_count']==row['model_calls']==0
            assert after.tobytes()==before.tobytes() and row['stages']==[] and row['readout'] is None
        else:
            assert row['parser_outcome']=='accepted' and row['dispatch_count']==1 and row['model_calls']==366
            assert row['action']['bit']==bit and row['readout']['qualified_label']==('ad' if bit==0 else 'td')
            assert [s['steps'] for s in row['stages']]==[120,240]
            stage_last=before.tobytes()
            for stage,controls in zip(row['stages'],(ENDPOINT[bit],HOLD)):
                assert (stage['controls']['alpha'],stage['controls']['log10_temperature'])==controls
                assert stage['cache_comparison']['passed']
                assert len(stage['checkpoints'])==stage['steps']//12+1
                with np.load(path/stage['trajectory']['path'],allow_pickle=False) as arrays:
                    assert arrays['states'].shape==(stage['steps']+1,896)
                    assert arrays['states'][0].tobytes()==stage_last
                    assert int(arrays['first_state_index'])==0
                    assert len(arrays['token_ids'])==stage['steps']
                    stage_last=arrays['states'][-1].tobytes()
                for artifact in stage['cache_comparison']['artifacts']:
                    with np.load(path/artifact['path'],allow_pickle=False) as arrays:
                        assert arrays['input'].tobytes()==np.load(path/stage['trajectory']['path'])['states'][0].tobytes()
            assert after.tobytes()==stage_last
        last=after.tobytes()
    assert np.load(path/'terminal-state.npy').tobytes()==last
    digital=json.loads((path/'digital-control.json').read_text())['rows']
    assert [r['dispatch_calls'] for r in digital]==[1,0,1,0,1,0,1]
    assert [r['state_after'] for r in digital]==['00','00','01','01','01','01','00']
    events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
    calls=[e for e in events if e['event']=='model_call_started']
    assert [e['call'] for e in calls]==list(range(1,1465))
    assert {e['message_id'] for e in calls}=={0,2,4,6}


@pytest.mark.parametrize('kind',['wrong','unresolved'])
def test_scientific_failure_continues_actual_state_without_reset(tmp_path,kind):
    backend=type('BadResult',(FakeBackend,),{kind:True})
    result=run(tmp_path,backend)
    assert result['status']=='completed' and result['model_calls']==1464
    if kind=='wrong':
        assert [result['rows'][i]['readout']['qualified_label'] for i in (0,2,4,6)]==['ad']*4
    else:
        assert all(result['rows'][i]['readout']['qualified_label'] is None for i in (0,2,4,6))
    path=tmp_path/'run'
    for previous,current in zip(result['rows'],result['rows'][1:]):
        assert np.load(path/previous['state_after']['path']).tobytes()==np.load(path/current['state_before']['path']).tobytes()
    check_inventory(path,result)


@pytest.mark.parametrize('call',[1,2,3,4,123,124,126,366,367,1464])
def test_execution_failure_preserves_every_row_and_available_stage(tmp_path,call):
    backend=type('Broken',(FakeBackend,),{'fail_at':call})
    result=run(tmp_path,backend)
    assert result['status']=='failed' and result['model_calls']==call
    assert 'injected model failure' in result['error']
    check_inventory(tmp_path/'run',result)
    attempted=[r for r in result['rows'] if r['status']!='not_started']
    assert attempted[-1]['status']=='failed' and attempted[-1]['state_after']
    if call>366: assert result['rows'][0]['status']==result['rows'][1]['status']=='completed'
    for row in attempted:
        for stage in row['stages']:
            if stage['status']!='not_started': assert 'trajectory' in stage


@pytest.mark.parametrize('kind',['nonfinite','wrong_dtype','wrong_shape'])
def test_bad_output_is_preserved_without_updating_carried_state(tmp_path,kind):
    result=run(tmp_path,type('BadOutput',(FakeBackend,),{kind:True}))
    assert result['status']=='failed' and result['model_calls']==4
    path=tmp_path/'run'; row=result['rows'][0]
    assert (path/'invalid-message-00-endpoint-001.npz').exists()
    assert np.load(path/row['state_before']['path']).tobytes()==np.load(path/row['state_after']['path']).tobytes()
    check_inventory(path,result)


def test_cache_failure_aborts_before_updates(tmp_path):
    result=run(tmp_path,type('BadCache',(FakeBackend,),{'cache_fail':True}))
    assert result['status']=='failed' and result['model_calls']==3
    assert result['rows'][0]['completed_steps']==0
    assert not result['rows'][0]['stages'][0]['cache_comparison']['passed']
    check_inventory(tmp_path/'run',result)


def test_timeout_after_last_update_preserves_updated_state_and_all_slots(tmp_path):
    now=[0.0]
    backend=type('LateTimeout',(FakeBackend,),{'time_trip':(1464,now)})
    result=run(tmp_path,backend,clock=lambda:now[0])
    assert result['status']=='incomplete' and result['model_calls']==1464
    assert result['rows'][6]['completed_steps']==360
    assert result['rows'][6]['stages'][1]['status']=='incomplete'
    assert result['rows'][6]['state_after'] and result['rows'][6]['readout'] is None
    assert not result['resource_limits_passed']
    check_inventory(tmp_path/'run',result)


def test_input_failure_never_loads_backend_and_keeps_seven_slots(tmp_path):
    def qualifier(args): raise ValueError('missing input')
    def backend(*args): pytest.fail('backend must not load')
    result=memory.run(args(tmp_path),backend_factory=backend,qualifier=qualifier,
                      install_signals=False,rss=lambda:0)
    assert result['status']=='failed' and result['model_calls']==0
    assert all(r['status']=='not_started' for r in result['rows'])
    check_inventory(tmp_path/'run',result)


def test_working_output_failure_uses_finalization_reserve(tmp_path,monkeypatch):
    monkeypatch.setattr(memory,'WORKING_OUTPUT',50000)
    result=run(tmp_path)
    assert result['status']=='failed' and 'artifact limit' in result['error']
    assert result['model_calls']<1464
    check_inventory(tmp_path/'run',result)


def test_invalid_signed_zero_mutation_is_detected(tmp_path,monkeypatch):
    recorder=memory.Recorder(tmp_path/'run',rss=lambda:0); budget=memory.Budget(recorder)
    state=refs()[0,0].copy(); row=memory.message_plan()[1]
    def corrupt(text):
        state[3]=-0.0
        raise memory.contract.CommandRejected('invalid')
    monkeypatch.setattr(memory.contract,'parse_command',corrupt)
    with pytest.raises(ValueError,match='mutated state'):
        memory.execute_message(row,state,None,refs(),recorder,budget)
    assert row['status']=='failed' and budget.calls==0
    assert row['state_before']['array_sha256']!=row['state_after']['array_sha256']
    recorder.stream.close()


def test_neutral_readout_does_not_take_a_requested_bit():
    cycles=refs()
    for i,label in enumerate(('ad','td')):
        r=memory.neutral_readout(np.tile(cycles[i],(40,1)),cycles)
        assert r['qualified_label']==label and r['cycle_gate']['passed']
    cycles[1]=cycles[0]
    assert memory.neutral_readout(np.tile(cycles[0],(40,1)),cycles)['qualified_label'] is None


def test_budget_per_stage_total_rss_and_time_guards(tmp_path):
    now=[0.0]; rss=[0]
    recorder=memory.Recorder(tmp_path/'run',clock=lambda:now[0],rss=lambda:rss[0]); budget=memory.Budget(recorder)
    with pytest.raises(RuntimeError,match='outside'): budget.claim('unexpected')
    budget.begin(0,dict(name='endpoint',steps=120)); budget.claim('prefix')
    with pytest.raises(RuntimeError,match='limit'): budget.claim('extra prefix')
    for _ in range(2): budget.claim('check','comparison')
    with pytest.raises(RuntimeError,match='limit'): budget.claim('extra check','comparison')
    now[0]=840
    with pytest.raises(TimeoutError): budget.check()
    now[0]=0; rss[0]=8*1024**3
    with pytest.raises(MemoryError): budget.check()
    recorder.stream.close()


def test_before_state_artifact_failure_finalizes_the_attempted_row(tmp_path,monkeypatch):
    original=memory.Recorder.state
    def broken(self,name,state,final=False):
        if name.endswith('-before.npy'): raise OSError('injected initial-state write failure')
        return original(self,name,state,final)
    monkeypatch.setattr(memory.Recorder,'state',broken)
    result=run(tmp_path)
    assert result['status']=='failed' and result['model_calls']==0
    assert result['rows'][0]['status']=='failed' and result['rows'][0]['state_before'] is None
    assert result['rows'][0]['state_after'] is not None
    assert all(r['status']=='not_started' for r in result['rows'][1:])
    check_inventory(tmp_path/'run',result)
