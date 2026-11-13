"""Model-free fixed-call, cache transition and compact evidence integration."""
import copy
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_public_batch_writes as pilot
sys.path.pop(0)
from test_public_command_memory import refs

OBS={'norm':{k:'torch.float64' for k in ('input','reduction','weight','output')},
     'rotary':{k:'torch.float64' for k in ('input','inverse_frequency','angles','cos','sin')}}
ENDPOINT={'alpha':0.07299785589927693,'log10_temperature':-0.15395151820007266}
HOLD={'alpha':0.07293697809562831,'log10_temperature':-0.15395151820007266}
ZERO_ENDPOINT={'alpha':0.07293697809562831,'log10_temperature':-0.15350}


def historical(args,cycles,plan=None):
    requested=1 if plan is None else plan.requested_bit
    values=[]
    for i in range(2):
        previous=cycles[i,0].copy(); stages=[]
        for count in (120,240):
            states=[previous.copy()]
            for _ in range(count):
                previous=previous.copy();previous[0]=(previous[0]+1)%6;previous[1]=.1 if requested else 0.
                states.append(previous)
            stages.append({'states':np.stack(states),'token_ids':np.ones(count,dtype=np.int64)})
        values.append(stages)
    return values,{'source_commit':'synthetic','controls':[
        {'neutral_readout':{'qualified_label':'td' if requested else 'ad'}} for _ in range(2)]}


def caches(value=0.,batch_size=1):
    # Small synthetic cache contracts keep orchestration tests cheap. Separate
    # tests exercise the actual architecture-sized cache predicate unchanged.
    return {'key':np.full((batch_size,1,2,1),value,dtype=np.float64),
            'length':np.asarray(37,dtype=np.int64)}


def fake_valid(cache,batch_size=1,length=None):
    return (set(cache)=={'key','length'} and cache['key'].shape==(batch_size,1,2,1)
        and cache['key'].dtype==np.float64 and np.isfinite(cache['key']).all()
        and cache['length'].dtype==np.int64 and cache['length'].shape==() and int(cache['length'])==37)


class FakeBackend:
    fail_at=None; fault=None; time_trip=None
    def __init__(self,args,recorder,budget):
        assert (recorder.directory/'historical-inputs.json').is_file()
        assert (recorder.directory/'stage-00-initial.npy').is_file()
        self.recorder,self.budget=recorder,budget;self.observations=copy.deepcopy(OBS)
        self.cache=caches();self.controls=None;self.returned_call=None;self.end_inspections=0
    def prepare(self,controls):
        self.controls=controls.copy();self.budget.claim('native64_prefix')
        if self.budget.calls==self.fail_at: raise RuntimeError('injected prefix failure')
        self.cache=caches(controls['alpha']);self.observations=copy.deepcopy(OBS)
    def cache_arrays(self):
        if self.fault=='cache-inspect' and self.budget.calls>=8: raise RuntimeError('cache inspection failed')
        if self.fault=='cache-after' and self.returned_call==8: raise RuntimeError('post-forward cache inspection failed')
        if self.fault=='cache-final' and self.budget.calls==123:
            self.end_inspections+=1
            if self.end_inspections>=3: raise RuntimeError('final cache inspection failed')
        return {k:v.copy() for k,v in self.cache.items()}
    def forward(self,state,cached):
        if self.budget.calls==self.fail_at: raise RuntimeError('injected forward failure')
        y=state.copy();y[:,0]=(y[:,0]+1)%6
        if self.controls==ENDPOINT: y[:,1]=.1
        elif self.controls==ZERO_ENDPOINT: y[:,1]=0.
        elif self.controls!=HOLD: raise AssertionError('control transition wrong')
        p=np.zeros((2,4),dtype=np.float64);p[:,1]=1.
        extra={'input_after':state.copy(),'observed_output_dtype':'torch.float64',
               'observed_probability_dtype':'torch.float64','input_exact':True}
        if cached:
            extra['used_cache_before']={'key':np.repeat(self.cache['key'],2,axis=0)}
            extra['used_cache_length_after']=38
        self.observations=copy.deepcopy(OBS)
        if self.fault=='finite-diagnostic' and not cached: y[:,4]+=.01
        if self.fault=='finite-probability' and not cached: p[:,1]=.75;p[:,2]=.25
        if self.fault=='wrong': y[:,1]=0.
        if self.fault=='unresolved': y[:,2]+=.01
        if self.fault=='tokens': p[:,1]=0.;p[:,2]=1.
        if self.budget.calls==8:
            if self.fault=='nonfinite': y[0,3]=np.nan
            if self.fault=='dtype': y=y.astype(np.float32)
            if self.fault=='shape': y=y[:,:-1]
            if self.fault=='precision': self.observations['norm']['reduction']='torch.float32'
            if self.fault=='input': extra['input_after'][0,3]=-0.
            if self.fault=='cache': self.cache['key'][0,0,0,0]=-0.
            if self.fault=='expanded': extra['used_cache_before']['key'][1,0,0,0]+=.01
            if self.fault=='length': extra['used_cache_length_after']=39
            if self.fault=='missing-extra': del extra['input_after']
        if self.time_trip and self.budget.calls==self.time_trip[0]: self.time_trip[1][0]=600.
        self.returned_call=self.budget.calls
        return y,p,.01,extra


@pytest.fixture(autouse=True)
def fast_fake_environment(monkeypatch):
    monkeypatch.setattr(pilot.memory.os,'fsync',lambda fd:None)
    utc=pilot.reference.utc_now();monkeypatch.setattr(pilot.reference,'utc_now',lambda:utc)


def run(tmp_path,monkeypatch,backend=FakeBackend,requested_bit=None,**kwargs):
    monkeypatch.setattr(pilot.batch,'VOCAB',4)
    monkeypatch.setattr(pilot.batch,'valid_cache',fake_valid)
    args=SimpleNamespace(output_dir=tmp_path/'run')
    if requested_bit is not None: args.requested_bit=requested_bit
    return pilot.run(args,backend_factory=backend,
        qualifier=lambda args:(refs(),{}),historical_qualifier=historical,rss=lambda:0,
        install_signals=False,**kwargs)


def inventory(path,result):
    assert json.loads((path/'result.json').read_text())==result
    assert len(result['calls'])==366 and len(result['rows'])==2 and len(result['stages'])==2
    assert {x['call'] for x in result['calls']}==set(range(1,367))
    for i,row in enumerate(result['rows']): assert json.loads((path/f'row-{i:02d}.json').read_text())==row
    manifest=json.loads((path/'manifest.json').read_text())
    assert {x['path'] for x in manifest['files']}=={p.name for p in path.iterdir()}-{'manifest.json'}
    for item in manifest['files']:
        p=path/item['path'];data=p.read_bytes()
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
    for stage in result['stages']:
        if stage['trajectory']:
            report=json.loads((path/stage['trajectory']['path']).read_text())
            assert report['blocks']==stage['blocks'] and report['initial']==stage['initial']
            states,ids,times=pilot.reconstruct(path,stage)
            assert len(states)==stage['completed_steps']+1 and len(ids)==len(times)==stage['completed_steps']


def test_fixed_design_counts_and_native_envelope():
    d=pilot.derive_design()
    assert d['schedule']==dict(batch_size=2,batch_updates=360,row_updates=720,prefix_calls=2,
        comparison_calls=4,total_calls=366,endpoint_steps=120,hold_steps=240,reset_count=0)
    assert d['artifact_envelope']['estimated_uncompressed_envelope_bytes']==28627392
    assert d['artifact_envelope']['failed_return_reserved_bytes']==2473984
    assert 28627392+2473984+16384<30*1024**2
    assert [x['call'] for x in pilot.call_plan() if x['category']=='prefix']==[1,124]
    assert [x['call'] for x in pilot.call_plan() if x['category']=='comparison']==[2,3,125,126]
    assert [s.steps for s in pilot.actions()[0].stages]==[120,240]
    assert pilot.batch.CONTROLS==HOLD and pilot.memory.MAX_CALLS==1464
    assert pilot.historical_source.OLD_TRIALS==(2,1)
    assert d['limits']['source_upload_bytes']==8*1024**2


def test_native_cache_shape_and_signed_zero_gate():
    shape=(1,2,37,64)
    values={f'layer_{i:02d}_{kind}':np.ones(shape,dtype=np.float64) for i in range(24) for kind in ('key','value')}
    values['length']=np.asarray(37,dtype=np.int64)
    assert pilot.batch.valid_cache(values)
    changed={k:v.copy() for k,v in values.items()};changed['layer_00_key']=changed['layer_00_key'].astype(np.float32)
    assert not pilot.batch.valid_cache(changed)
    one={'x':np.asarray([0.])};two={'x':np.asarray([-0.])}
    assert np.array_equal(one['x'],two['x']) and not pilot.same_bytes(one,two)


def test_stage_adapter_passes_original_vector_without_global_change(monkeypatch):
    seen=[]; vector=object();loop=object();obs={}
    def controls(actual_loop,actual_vector,setting):
        seen.append((actual_loop,actual_vector,setting.copy()));obs.update(copy.deepcopy(OBS))
    monkeypatch.setattr(pilot.memory.writes,'set_controls',controls)
    fake=SimpleNamespace(loop=loop,vector=vector,observations=obs)
    pilot.Backend.prepare(fake,ENDPOINT);pilot.Backend.prepare(fake,HOLD)
    assert seen==[(loop,vector,ENDPOINT),(loop,vector,HOLD)] and pilot.batch.CONTROLS==HOLD


def test_complete_run_reconstructs_once_and_carries_exact_boundary(tmp_path,monkeypatch):
    result=run(tmp_path,monkeypatch);path=tmp_path/'run'
    assert result['status']=='completed' and result['numerical_equivalence_passed']
    assert result['model_calls']==366 and result['completed_row_updates']==720 and result['reset_count']==0
    assert result['call_counts']==dict(prefix=2,comparison=4,update=360)
    assert all(s['status']=='returned' for s in result['calls'])
    assert [r['readout']['qualified_label'] for r in result['rows']]==['td','td']
    a,b=result['stages'];left,ids,_=pilot.reconstruct(path,a);right,jds,_=pilot.reconstruct(path,b)
    assert left.shape==(121,2,896) and right.shape==(241,2,896)
    assert left[-1].tobytes()==right[0].tobytes()
    assert np.concatenate((left,right[1:])).shape==(361,2,896)
    assert len(a['blocks'])==10 and len(b['blocks'])==20
    assert not list(path.glob('*final*.npz'))
    for s in result['stages']:
        assert s['attempted_steps']==s['returned_steps']==s['completed_steps']==s['steps']
        assert s['final_scalar_cache_unchanged'] and s['cache_comparison']['passed']
        assert s['historical_comparison']['passed']
    with np.load(path/a['scalar_prefix_cache']['path']) as p,np.load(path/b['scalar_prefix_cache']['path']) as q:
        assert p['key'].tobytes()!=q['key'].tobytes()
    for row in result['rows']:
        assert row['dispatch_count']==1 and row['historical_comparison']['compared_states']==361
        assert row['historical_comparison']['compared_updates']==360
    inventory(path,result)


@pytest.mark.parametrize('fault',['finite-diagnostic','finite-probability','wrong','unresolved','tokens'])
def test_finite_scientific_disagreement_completes_fixed_schedule(tmp_path,monkeypatch,fault):
    result=run(tmp_path,monkeypatch,type('Finite',(FakeBackend,),{'fault':fault}))
    assert result['status']=='completed' and result['model_calls']==366
    assert not result['numerical_equivalence_passed']
    if fault=='wrong': assert [r['readout']['qualified_label'] for r in result['rows']]==['ad','ad']
    if fault=='unresolved': assert all(r['readout']['qualified_label'] is None for r in result['rows'])
    inventory(tmp_path/'run',result)


@pytest.mark.parametrize('call',[1,2,3,4,8,123,124,125,126,127,366])
def test_failure_all_slots_and_pending_blocks_retained(tmp_path,monkeypatch,call):
    result=run(tmp_path,monkeypatch,type('Failure',(FakeBackend,),{'fail_at':call}))
    assert result['status']=='failed' and result['model_calls']==call
    assert result['calls'][call-1]['status']=='failed'
    assert all(r['status']=='not_started' for r in result['calls'][call:])
    assert sum(s['completed_steps'] for s in result['stages'])==sum(
        x.get('state_carried',False) for x in result['calls'])
    inventory(tmp_path/'run',result)


@pytest.mark.parametrize('fault',['nonfinite','dtype','shape','precision','input','cache','expanded','length','cache-inspect',
                                'cache-after','missing-extra'])
def test_structural_errors_abort_without_admitting_bad_state(tmp_path,monkeypatch,fault):
    result=run(tmp_path,monkeypatch,type('Fault',(FakeBackend,),{'fault':fault}))
    assert result['status']=='failed' and result['model_calls']==8
    first=result['stages'][0]
    assert first['completed_steps']==4 and len(first['blocks'])==1
    assert result['calls'][7]['status']=='failed' and first['failed_outputs']
    assert result['stages'][1]['status']=='not_started'
    if fault=='cache-inspect': assert first['final_cache_error']=='RuntimeError: cache inspection failed'
    if fault=='missing-extra':
        with np.load(tmp_path/'run'/first['failed_outputs'][0]['path']) as raw:
            assert raw['output'].shape==(2,896) and raw['probabilities'].shape==(2,4)
    assert result['calls'][7]['forward_attempted']==(fault!='cache-inspect')
    inventory(tmp_path/'run',result)


def test_final_cache_inspection_failure_cannot_skip_completed_archive(tmp_path,monkeypatch):
    result=run(tmp_path,monkeypatch,type('FinalCache',(FakeBackend,),{'fault':'cache-final'}))
    assert result['status']=='failed' and result['model_calls']==123
    stage=result['stages'][0]
    assert stage['completed_steps']==120 and len(stage['blocks'])==10 and stage['trajectory']
    assert 'final cache inspection failed' in stage['final_cache_error']
    assert result['stages'][1]['status']=='not_started'
    inventory(tmp_path/'run',result)


@pytest.mark.parametrize('call',[8,366])
def test_timeout_after_return_preserves_mutation_before_resource_check(tmp_path,monkeypatch,call):
    now=[0.]
    result=run(tmp_path,monkeypatch,type('Timeout',(FakeBackend,),{'time_trip':(call,now)}),clock=lambda:now[0])
    assert result['status']=='incomplete' and result['model_calls']==call
    assert result['calls'][call-1]['state_carried'] and result['calls'][call-1]['status']=='incomplete'
    assert result['completed_batch_updates']==(5 if call==8 else 360)
    assert not result['resource_limits_passed']
    inventory(tmp_path/'run',result)


def test_next_call_raw_reserve_failure_flushes_pending_updates(tmp_path,monkeypatch):
    class Fill(FakeBackend):
        def forward(self,state,cached):
            value=super().forward(state,cached)
            if self.budget.calls==8: monkeypatch.setattr(pilot,'WORKING_OUTPUT',self.recorder.size()+10000)
            return value
    result=run(tmp_path,monkeypatch,Fill)
    assert result['model_calls']==8 and result['completed_batch_updates']==5 and result['status']=='failed'
    assert result['calls'][8]['status']=='not_started'
    inventory(tmp_path/'run',result)


@pytest.mark.parametrize('kind',['qualifier','history','backend'])
def test_preparation_failure_retains_all_planned_rows_and_calls(tmp_path,kind):
    def bad(*args,**kwargs): raise ValueError('injected preparation failure')
    result=pilot.run(SimpleNamespace(output_dir=tmp_path/'run'),
        qualifier=bad if kind=='qualifier' else lambda args:(refs(),{}),
        historical_qualifier=bad if kind=='history' else historical,
        backend_factory=bad if kind=='backend' else FakeBackend,rss=lambda:0,install_signals=False)
    assert result['model_calls']==0 and result['status']=='failed'
    assert all(c['status']=='not_started' for c in result['calls'])
    inventory(tmp_path/'run',result)


def test_reconstructor_rejects_hash_valid_gapped_or_wrong_dtype_block(tmp_path,monkeypatch):
    result=run(tmp_path,monkeypatch);stage=copy.deepcopy(result['stages'][0]);path=tmp_path/'run'
    block=path/stage['blocks'][0]['path']
    with np.load(block) as source: arrays={k:source[k].copy() for k in source.files}
    arrays['first_state_index']=np.asarray(2,dtype=np.int64)
    with block.open('wb') as f: np.savez(f,**arrays)
    stage['blocks'][0]=pilot.memory.binding(block)
    with pytest.raises(ValueError,match='block contract'): pilot.reconstruct(path,stage)
    arrays['first_state_index']=np.asarray(1,dtype=np.int64);arrays['token_ids']=arrays['token_ids'].astype(np.int32)
    with block.open('wb') as f: np.savez(f,**arrays)
    stage['blocks'][0]=pilot.memory.binding(block)
    with pytest.raises(ValueError,match='block contract'): pilot.reconstruct(path,stage)


def test_historical_actual_four_trajectory_contract_zero_forward():
    directory=Path('/data2/shrimp-memory-data/audits/p6-writes/20261025T222347Z-writes/run')
    refpath=Path('/data2/shrimp-memory-data/audits/p6-recovery/20261025T221011Z-recovery/reference-cycles.npz')
    if not directory.exists(): pytest.skip('optional local returned evidence absent')
    meta=json.loads((directory/'metadata.json').read_text())
    args=SimpleNamespace(historical_run=directory,**{k:meta['arguments'][k] for k in
        ('references_sha256','reference_manifest_sha256','recovery_results_sha256')})
    with np.load(refpath) as raw: cycles=raw['states']
    histories,provenance=pilot.qualify_historical(args,cycles)
    assert len(provenance['bindings'])==6 and [c['trial_id'] for c in provenance['controls']]==[2,3]
    assert [c['neutral_readout']['qualified_label'] for c in provenance['controls']]==['td','td']
    assert all([s['states'].shape for s in row]==[(121,896),(241,896)] for row in histories)


def test_budget_rejects_extra_or_reordered_calls_and_wrong_controls(tmp_path):
    recorder=pilot.Recorder(tmp_path/'run',rss=lambda:0);budget=pilot.Budget(recorder,pilot.call_plan())
    with pytest.raises(RuntimeError): budget.claim('outside')
    with pytest.raises(ValueError): budget.begin(pilot.stage_plan()[1])
    bad=pilot.stage_plan()[0];bad['steps']=60
    with pytest.raises(ValueError): budget.begin(bad)
    budget.begin(pilot.stage_plan()[0]);budget.claim('prefix')
    with pytest.raises(RuntimeError): budget.claim('prefix-again')
    with pytest.raises(RuntimeError): budget.claim('update-too-early','update')
    budget.calls=366
    with pytest.raises(RuntimeError): budget.claim('extra','update')
    recorder.stream.close()


@pytest.mark.parametrize('change,match',[
    (lambda m:m.update(threads=2),'map inputs'),
    (lambda m:m['vectors'].update(sha256='a'*64),'model input'),
    (lambda m:m['arguments'].update(references_sha256='b'*64),'prerequisite'),
    (lambda m:m['source_hashes'].update(**{'validate_qwen_map.py':'c'*64}),'mathematical source'),
])
def test_rehashed_incompatible_historical_inputs_do_not_pass(tmp_path,change,match):
    expected=dict(references_sha256='d'*64,reference_manifest_sha256='e'*64,recovery_results_sha256='f'*64)
    meta=dict(source_commit='f0c757ed2c7142bd2dd5cb78a796f69f7f49f4c4',
        model=pilot.reference.MODEL,revision=pilot.reference.REVISION,prefix_ids=pilot.reference.PREFIX_IDS,
        hold=pilot.memory.writes.HOLD,candidate_endpoints=pilot.memory.writes.ENDPOINTS,threads=1,
        weights={'sha256':pilot.reference.WEIGHTS_SHA256},config={'sha256':pilot.memory.hold_test.CONFIG_SHA256},
        vectors={'sha256':pilot.reference.VECTOR_SHA256},arguments=expected.copy(),
        source_hashes={Path(m.__file__).name:pilot.reference.sha256(Path(m.__file__))
                      for m in (pilot.reference,pilot.memory.hold_test,pilot.memory.recovery)})
    change(meta);raw=tmp_path/'raw';raw.mkdir()
    (raw/'metadata.json').write_text(json.dumps(meta));(raw/'results.json').write_text(json.dumps({'status':'passed'}))
    receipt=dict(independent_audit_status='audit_passed',experiment_source_commit=meta['source_commit'],
        bindings=[{**pilot.memory.binding(raw/name),'path':'synthetic/run/'+name}
                  for name in ('metadata.json','results.json')])
    receipt_path=tmp_path/'receipt.json';receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match=match):
        pilot.qualify_historical(SimpleNamespace(historical_run=raw,**expected),refs(),receipt_path)


def test_cli_eight_paths_three_hashes_and_no_scope_override(monkeypatch):
    observed=[];monkeypatch.setattr(pilot,'run',lambda args:observed.append(vars(args)) or {'status':'completed'})
    names=('cache-dir','vectors','references','reference-manifest','source-hold-results',
           'recovery-results','historical-run','output-dir')
    hashes=('references-sha256','reference-manifest-sha256','recovery-results-sha256')
    argv=[item for name in names for item in ('--'+name,'/tmp/'+name)]
    argv += [item for name in hashes for item in ('--'+name,'a'*64)]
    assert pilot.main(argv)==0 and len(observed[0])==12 and observed[0]['requested_bit']==1
    assert all(isinstance(observed[0][name.replace('-','_')],Path) for name in names)
    with pytest.raises(SystemExit): pilot.main(argv+['--steps','60'])


def test_protocol_binds_exact_limits_and_count():
    text=(Path(__file__).resolve().parents[1]/'docs/public-batch-writes.md').read_text()
    assert '28,627,392 bytes' in text and '2,473,984 bytes' in text
    assert '366 calls' in text and '30 MiB' in text and '32 MiB total raw' in text


@pytest.mark.parametrize('invalid',[True,False,-1,2,'0',0.,None])
def test_requested_bit_requires_an_explicit_binary_integer(invalid):
    with pytest.raises(ValueError): pilot.WritePlan(invalid)


def test_zero_plan_is_immutable_and_propagated_without_globals():
    zero=pilot.WritePlan(0)
    with pytest.raises(FrozenInstanceError): zero.requested_bit=1
    with pytest.raises(ValueError): pilot.validate_plan(SimpleNamespace(requested_bit=0))
    assert zero.messages==('Write bit 0 to register R0.',)*2 and zero.historical_trials==(0,1)
    assert [r['historical_trial_id'] for r in pilot.row_plan(zero)]==[0,1]
    assert [r['message'] for r in pilot.row_plan(zero)]==list(zero.messages)
    assert all(a.bit==0 and a.attractor_label=='ad' for a in pilot.actions(zero))
    assert [s['controls'] for s in pilot.stage_plan(zero)]==[ZERO_ENDPOINT,HOLD]
    design=pilot.derive_design(zero)
    assert design['requested_bit']==0 and design['historical_trial_ids']==[0,1]
    assert design['schedule']==pilot.derive_design()['schedule']
    assert pilot.call_plan(zero)==pilot.call_plan()  # Control lives in the stage; calls are identical.
    assert pilot.actions()==pilot.actions(pilot.WritePlan(1))
    assert pilot.stage_plan()==pilot.stage_plan(pilot.WritePlan(1))
    assert pilot.row_plan()==pilot.row_plan(pilot.WritePlan(1))
    assert pilot.MESSAGES==('Write bit 1 to register R0.',)*2 and pilot.OLD_TRIALS==(2,3)
    assert pilot.batch.CONTROLS==HOLD and pilot.historical_source.OLD_TRIALS==(2,1)


def test_temperature_only_control_change_still_rebuilds_prefix():
    class Loop:
        def __init__(self): self.calls=[]
        def prepare(self): self.calls.append((self.steering.copy(),self.temperature))
    loop=Loop();vector=np.linspace(-1.,1.,896,dtype=np.float64)
    pilot.memory.writes.set_controls(loop,vector,ZERO_ENDPOINT)
    pilot.memory.writes.set_controls(loop,vector,HOLD)
    assert len(loop.calls)==2
    assert loop.calls[0][0].tobytes()==loop.calls[1][0].tobytes()
    assert loop.calls[0][1]==10.**(-.15350) and loop.calls[1][1]==10.**(-.15395151820007266)
    assert loop.calls[0][1]!=loop.calls[1][1]


def test_zero_complete_run_binds_public0_history0_1_and_two_rebuilds(tmp_path,monkeypatch):
    prepared=[]
    class RecordPrepare(FakeBackend):
        def prepare(self,controls):
            super().prepare(controls);prepared.append((self.budget.calls,controls.copy()))
    result=run(tmp_path,monkeypatch,RecordPrepare,requested_bit=0);path=tmp_path/'run'
    assert result['status']=='completed' and result['numerical_equivalence_passed'] and result['requested_bit']==0
    assert result['model_calls']==366 and result['completed_row_updates']==720
    assert prepared==[(1,ZERO_ENDPOINT),(124,HOLD)]
    assert [r['readout']['qualified_label'] for r in result['rows']]==['ad','ad']
    assert [r['historical_trial_id'] for r in result['rows']]==[0,1]
    assert all(r['message']=='Write bit 0 to register R0.' and r['action']['bit']==0 for r in result['rows'])
    assert [s['controls'] for s in result['stages']]==[ZERO_ENDPOINT,HOLD]
    assert all([r['historical_trial_id'] for r in s['historical_comparison']['rows']]==[0,1] for s in result['stages'])
    design=json.loads((path/'design.json').read_text())
    assert design['requested_bit']==0 and design['historical_trial_ids']==[0,1]
    assert {b['path'] for b in result['source']['files']} >= {'public-batch-zero.md','public-batch-writes.json'}
    a,_,_=pilot.reconstruct(path,result['stages'][0]);b,_,_=pilot.reconstruct(path,result['stages'][1])
    assert a[0].tobytes()==refs()[:,0].tobytes() and a[-1].tobytes()==b[0].tobytes()
    inventory(path,result)


@pytest.mark.parametrize('call',[8,124,127,366])
def test_zero_failures_preserve_original_plan_and_partial_state(tmp_path,monkeypatch,call):
    result=run(tmp_path,monkeypatch,type('FailZero',(FakeBackend,),{'fail_at':call}),requested_bit=0)
    assert result['status']=='failed' and result['model_calls']==call and result['requested_bit']==0
    assert all(r['action']['bit']==0 for r in result['rows'])
    assert [r['historical_trial_id'] for r in result['rows']]==[0,1]
    inventory(tmp_path/'run',result)


def test_zero_finite_failure_still_completes_and_readout_is_neutral(tmp_path,monkeypatch):
    class WrongZero(FakeBackend):
        def forward(self,state,cached):
            y,p,seconds,extra=super().forward(state,cached);y[:,1]=.1
            return y,p,seconds,extra
    result=run(tmp_path,monkeypatch,WrongZero,requested_bit=0)
    assert result['status']=='completed' and result['model_calls']==366
    assert not result['numerical_equivalence_passed']
    assert [r['readout']['qualified_label'] for r in result['rows']]==['td','td']
    inventory(tmp_path/'run',result)


def test_zero_actual_historical_file_contract_and_plan_mismatch():
    directory=Path('/data2/shrimp-memory-data/audits/p6-writes/20261025T222347Z-writes/run')
    refpath=Path('/data2/shrimp-memory-data/audits/p6-recovery/20261025T221011Z-recovery/reference-cycles.npz')
    if not directory.exists(): pytest.skip('optional local returned evidence absent')
    meta=json.loads((directory/'metadata.json').read_text())
    args=SimpleNamespace(historical_run=directory,requested_bit=0,**{k:meta['arguments'][k] for k in
        ('references_sha256','reference_manifest_sha256','recovery_results_sha256')})
    with np.load(refpath) as raw: cycles=raw['states']
    histories,provenance=pilot.qualify_historical(args,cycles,plan=pilot.WritePlan(0))
    assert provenance['requested_bit']==0 and provenance['historical_trial_ids']==[0,1]
    assert provenance['messages']==['Write bit 0 to register R0.']*2
    assert [r['neutral_readout']['qualified_label'] for r in provenance['controls']]==['ad','ad']
    for i,row in enumerate(histories):
        assert row[0]['states'][0].tobytes()==cycles[i,0].tobytes()
        assert row[0]['states'][-1].tobytes()==row[1]['states'][0].tobytes()
    with pytest.raises(ValueError,match='plan differs'): pilot.qualify_historical(args,cycles,plan=pilot.WritePlan(1))


def test_zero_budget_rejects_write1_controls(tmp_path):
    recorder=pilot.Recorder(tmp_path/'run',rss=lambda:0);zero=pilot.WritePlan(0)
    budget=pilot.Budget(recorder,pilot.call_plan(zero),zero)
    with pytest.raises(ValueError,match='stage differs'): budget.begin(pilot.stage_plan()[0])
    budget.begin(pilot.stage_plan(zero)[0]);budget.claim('zero-prefix')
    assert budget.calls==1 and budget.plan.requested_bit==0
    recorder.stream.close()


def test_zero_cli_is_explicit_and_other_bits_rejected(monkeypatch):
    observed=[];monkeypatch.setattr(pilot,'run',lambda args:observed.append(vars(args)) or {'status':'completed'})
    names=('cache-dir','vectors','references','reference-manifest','source-hold-results',
           'recovery-results','historical-run','output-dir')
    hashes=('references-sha256','reference-manifest-sha256','recovery-results-sha256')
    argv=[item for name in names for item in ('--'+name,'/tmp/'+name)]
    argv += [item for name in hashes for item in ('--'+name,'a'*64)]
    assert pilot.main(argv+['--requested-bit','0'])==0 and observed[0]['requested_bit']==0
    with pytest.raises(SystemExit): pilot.main(argv+['--requested-bit','2'])
