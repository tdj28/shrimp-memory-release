import copy,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_public_mechanism as m
from test_public_batch_writes import FakeBackend as Base,OBS,caches,fake_valid
from test_public_command_memory import refs

class Fake(Base):
    def __init__(self,args,recorder,budget):
        assert (recorder.directory/'reference-starts.npy').is_file()
        self.recorder,self.budget=recorder,budget;self.observations=copy.deepcopy(OBS);self.cache=caches();self.controls=None;self.returned_call=None;self.end_inspections=0
    def forward(self,state,cached):
        original=self.controls
        if self.controls not in [m.HOLD,*m.ENDPOINTS]:self.controls=m.HOLD
        try:return super().forward(state,cached)
        finally:self.controls=original

@pytest.fixture(autouse=True)
def cheap(monkeypatch):
    monkeypatch.setattr(m.memory.os,'fsync',lambda fd:None)
    now=m.reference.utc_now();monkeypatch.setattr(m.reference,'utc_now',lambda:now)
    monkeypatch.setattr(m.batch,'VOCAB',4);monkeypatch.setattr(m.batch,'valid_cache',fake_valid)

def execute(tmp_path,mode,backend=Fake):
    return m.run(SimpleNamespace(mode=mode,dwell=240,output_dir=tmp_path/'run'),backend_factory=backend,qualifier=lambda a:(refs(),{}),rss=lambda:0,install_signals=False)

@pytest.mark.parametrize('mode,calls,updates',[('hysteresis',1944,1920),('and',2928,2880),('or',2928,2880)])
def test_fixed_schedule_and_input_isolation(mode,calls,updates):
    g,s,c=m.plans(mode);assert len(c)==calls and sum(x['steps'] for x in s)==updates
    if mode!='hysteresis':
        for group in g:
            for index,key in ((0,'b'),(2,'c')):assert s[group['stage_ids'][index]]['controls']==m.control(group['case'][key],mode)
    else:
        for group in g:
            a,b,c,d=[s[i] for i in group['stage_ids']]
            assert a['controls']==c['controls'] and d['controls']==m.HOLD
            assert b['controls']==m.ENDPOINTS[group['case']['direction']]

@pytest.mark.parametrize('mode',['and','hysteresis'])
def test_real_file_contract_and_no_within_group_reset(tmp_path,mode):
    r=execute(tmp_path,mode);assert r['status']=='completed',r.get('traceback')
    base=tmp_path/'run';manifest=json.loads((base/'manifest.json').read_text())
    for x in manifest['files']:
        raw=(base/x['path']).read_bytes();assert len(raw)==x['bytes'] and hashlib.sha256(raw).hexdigest()==x['sha256']
    assert r['model_calls']==m.design(mode,240)['calls']
    for g in r['groups']:
        previous=np.stack([refs()[0,0],refs()[1,0]])
        for sid in g['stage_ids']:
            states,_,_=m.prior.reconstruct(base,r['stages'][sid]);assert states[0].tobytes()==previous.tobytes();previous=states[-1]
        assert g['terminal_state_sha256']==m.prior.array_hash(previous)
        for x in g['artifacts']:assert (base/x['path']).is_file()
        if mode=='and':
            for step,key in ((1,'b'),(3,'c')):
                actual=[None if x['qualified_label'] is None else ('ad','td').index(x['qualified_label']) for x in g['observations'][step]['readouts']]
                expected=[a & g['case']['b'] for a in (0,1)]
                if key=='c':expected=[x & g['case']['c'] for x in expected]
                assert actual==expected

@pytest.mark.parametrize('mode',['and','hysteresis'])
def test_failure_preserves_all_planned_cases_and_partial_bytes(tmp_path,mode):
    r=execute(tmp_path,mode,type('Fail',(Fake,),{'fail_at':8}));assert r['status']=='failed' and r['model_calls']==8
    assert len(r['groups'])==len(m.plans(mode)[0]) and all(g['status']=='not_started' for g in r['groups'][1:])
    assert r['stages'][0]['completed_steps']==4 and r['stages'][0]['failed_outputs']
    assert (tmp_path/'run/manifest.json').is_file()

def test_invalid_design_fails_before_execution():
    for args in [('xor',240),('and',480),('hysteresis',12)]:
        with pytest.raises(ValueError):m.plans(*args)
    with pytest.raises(ValueError):m.control(True,'and')
