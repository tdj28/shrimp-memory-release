"""Model-free tests of the exact experiment orchestrator, not scientific data."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_public_periodic_corrector as m

OBS={'norm':{k:'torch.float64' for k in ('input','reduction','weight','output')},
     'rotary':{k:'torch.float64' for k in ('input','inverse_frequency','angles','cos','sin')}}


def fixtures():
    base=np.zeros(m.DIM,dtype=np.float64);base[0]=.2
    seed=base.copy();seed[6]=.0001
    cycles=np.stack([np.r_[np.roll(base[:6],i+1),base[6:]] for i in range(6)])
    p=np.stack([cycles,cycles.copy()],axis=1);p[:,1,6]=.0001
    return dict(seed=seed,comparator=base,p_cycles=p,m_cycles=np.stack([cycles,cycles],axis=1),
        p_terminal=np.stack([base,seed]),trust_radius=.0002)


class FakeBackend:
    fault=None
    def __init__(self,rec,budget):self.rec,self.budget=rec,budget
    def evaluate(self,label,state,tangent=None,cached=True):
        self.budget.claim(label)
        if self.fault=='exception' and label=='control-2':raise RuntimeError('fixture interruption')
        y=.1*state.copy();y[:6]=np.roll(state[:6],1)
        y[6]=.5*state[6]+1e6*state[6]**3
        j=None
        if tangent is not None:
            j=.1*tangent.copy();j[:6]=np.roll(tangent[:6],1)
            j[6]=(.5+3e6*state[6]**2)*tangent[6]
            if self.fault=='jvp':j[6]=np.nan
            if self.fault=='primal':y[6]+=.01
            if self.fault=='fd':j*=1.1
        if self.fault=='cache' and not cached:y[0]+=.01
        data=dict(input=state,output=y)
        if tangent is not None:data.update(direction=tangent,jvp=j)
        self.rec.save(f'call-{self.budget.calls:04d}-{label}.npz',**data)
        return y,j
    @staticmethod
    def offset(x,v,h):return x+h*v


def execute(tmp_path, *,fault=None,protocol=None,preflight=False,inputs=None):
    args=SimpleNamespace(output_dir=tmp_path/'run',preflight_only=preflight)
    def backend(args,snapshot,rec,budget):
        budget.claim('fixture-prefix');instance=FakeBackend(rec,budget);instance.fault=fault
        return instance,copy.deepcopy(OBS)
    kwargs=dict(backend_factory=backend,input_qualifier=lambda args:(inputs or fixtures(),{'fixture':True}),
        model_qualifier=lambda args:(tmp_path,{'fixture':True}),source_factory=lambda args:{'fixture':True})
    if protocol is not None:kwargs['protocol']=protocol
    code=m.run(args,**kwargs)
    return code,json.loads((args.output_dir/'result.json').read_text()),args.output_dir


def assert_artifacts(folder,result):
    manifest=json.loads((folder/'manifest.json').read_text())
    for row in manifest['files']:
        raw=(folder/row['path']).read_bytes()
        assert len(raw)==row['bytes']
        assert hashlib.sha256(raw).hexdigest()==row['sha256']
    assigned=set()
    for i,group in enumerate(result['groups']):
        assert group==json.loads((folder/f'group-{i:02d}.json').read_text())
        assert group['closed']
        for row in group['artifacts']:
            assert row['path'] not in assigned
            assigned.add(row['path'])
            assert hashlib.sha256((folder/row['path']).read_bytes()).hexdigest()==row['sha256']
    assert sum(g['model_calls'] for g in result['groups'])==result['model_calls']


def test_real_protocol_with_known_periodic_map(tmp_path):
    code,result,folder=execute(tmp_path)
    assert code==0,result.get('error')
    assert result['status']=='completed' and result['outcome']=='merged_cycle_candidate'
    assert result['groups'][0]['model_calls']==91
    assert result['groups'][5]['model_calls']==24
    assert result['derivative']['fd']['passed'] and result['final']['passed']
    assert result['model_calls']<=403 and result['resource_limits_passed']
    assert result['seed_repeat_linf']==0
    assert result['final']['merged_cycle_distance']['distance']<1e-10
    assert_artifacts(folder,result)


@pytest.mark.parametrize('fault',['cache','primal','fd','jvp','exception'])
def test_failed_instruments_retain_partial_groups(tmp_path,fault):
    code,result,folder=execute(tmp_path,fault=fault)
    assert code==1 and result['status'] in ('instrument_failed','execution_failed')
    assert result['outcome']=='not_qualified' and result['groups'][5]['model_calls']==0
    assert result['model_calls']>0
    assert_artifacts(folder,result)


def test_finite_trust_exhaustion_is_unresolved(tmp_path):
    inputs=fixtures();inputs['trust_radius']=1e-9
    code,result,folder=execute(tmp_path,inputs=inputs)
    assert code==0 and result['outcome']=='root_search_unresolved'
    assert result['corrector']['reason']=='trust_ball_exhausted'
    assert result['groups'][5]['not_run_reason']=='root_search_unresolved'
    assert_artifacts(folder,result)


def test_preflight_does_not_load_or_call(tmp_path):
    code,result,folder=execute(tmp_path,preflight=True)
    assert code==0 and result['status']=='preflight_passed'
    assert not result['model_loaded'] and result['model_calls']==0
    assert_artifacts(folder,result)


@pytest.mark.parametrize('limit',['time','memory','raw'])
def test_terminal_resource_breach_cannot_pass(tmp_path,monkeypatch,limit):
    def protocol(backend,payload,rec,summary):
        summary.update(status='completed',outcome='root_search_unresolved')
        if limit=='time':rec.started-=m.MAX_SECONDS+1
        if limit=='memory':monkeypatch.setattr(m.instrument,'rss_bytes',lambda:9*2**30)
        if limit=='raw':monkeypatch.setattr(m,'RAW_CAP',rec.size()+1)
    code,result,folder=execute(tmp_path,protocol=protocol)
    assert code==1 and result['status']=='incomplete' and not result['resource_limits_passed']
    assert_artifacts(folder,result)


def test_cap_includes_manifest(tmp_path,monkeypatch):
    original=m.Recorder.json
    def save(rec,name,value):
        original(rec,name,value)
        if name=='manifest.json':monkeypatch.setattr(m,'RAW_CAP',rec.size()-1)
    monkeypatch.setattr(m.Recorder,'json',save)
    code,result,folder=execute(tmp_path,preflight=True)
    assert code==1 and result['status']=='incomplete'
    assert result['terminal_resources']['current_raw_bytes']>m.RAW_CAP
    assert_artifacts(folder,result)


def test_precision_is_fresh_and_complete():
    assert m.fresh_precision(OBS)
    assert not m.fresh_precision({})
    for group,values in OBS.items():
        for key in values:
            changed=copy.deepcopy(OBS);del changed[group][key]
            assert not m.fresh_precision(changed)
            changed[group][key]='torch.float32'
            assert not m.fresh_precision(changed)


def test_derived_nonfinite_metric_is_failure():
    trace=np.zeros((13,m.DIM),dtype=np.float64)
    trace[1::2]=1e300
    with np.errstate(over='ignore'),pytest.raises(ValueError,match='nonfinite cycle metrics'):
        m.cycle_checks(trace)


def test_cycle_qualification_rejects_lower_period():
    assert not m.cycle_checks(np.zeros((13,m.DIM)))['passed']


def test_wrong_hash_rejected_before_loading(tmp_path):
    path=tmp_path/'input.json';path.write_text('{}')
    with pytest.raises(ValueError,match='input hash differs'):m.checked(path,'0'*64)


def test_archive_input_rejects_symlink(tmp_path):
    path=tmp_path/'input';path.write_bytes(b'bound')
    link=tmp_path/'link';link.symlink_to(path)
    with pytest.raises(ValueError,match='symlink'):m.binding(link)
