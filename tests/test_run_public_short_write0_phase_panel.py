"""Model-free control, literal-carry, failure-retention and serialization fixtures."""
import ast
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import run_public_short_write0_phase_panel as runner


def actual_args(output):
    base = Path('/data2/shrimp-memory-data/audits/p6-writes/20261025T222347Z-writes')
    return SimpleNamespace(prospective_panel=Path('/data2/shrimp-memory-data/preparations/public-short-write0-phase-panel/20261029T153845Z/prospective-phase-panel.npz'),
        trajectory_h0=base/'run/trial-00-endpoint-final-120.npz',
        trajectory=base/'run/trial-01-endpoint-final-120.npz', fixed_inputs=base/'run/fixed-trial-inputs.npz',
        historical_metadata=base/'run/metadata.json', historical_plan=base/'run/trial-plan.json',
        historical_audit=base/'complete-audit.json', historical_manifest=base/'final-manifest.json',
        cache_dir=Path('/tmp/unused-model'), vectors=Path('/tmp/unused-vectors'), output_dir=Path(output),
        preflight_only=False)



def synthetic_payload():
    refs=np.zeros((2,6,896),dtype=np.float64); refs[1,:,0]=10.
    for i in range(6): refs[:,i,i+1]=.1
    states=np.repeat(refs[:,0][None],121,axis=0)
    states[:,:,20]=np.arange(121)[:,None]/1000
    return runner.derive_inputs(refs,states,np.zeros((120,2),dtype=np.int64),np.stack([refs[:,p] for p in runner.PHASES]))



def synthetic_source(args):
    return dict(source_commit='synthetic',files=[],working_tree_status=[],synthetic=True)



def synthetic_model(args):
    return dict(synthetic=True,no_model_loaded=True)



class FakeBackend:
    """Scripted outputs with actual production array geometry, never a model."""
    def __init__(self,args,budget,recorder,*,payload=None,bits=None,failure=None,defect=None,history_offset=0.):
        self.budget,self.recorder=budget,recorder
        self.payload=synthetic_payload() if payload is None else payload
        self.bits=[0,0,1,1]*4 if bits is None else bits
        self.failure,self.defect,self.history_offset=failure,defect,history_offset
        self.observations=self.good_observations(); self.cache=None; self.expanded=None
        self.selected_group=-1; self.reads=0

    @staticmethod
    def good_observations():
        return {group:{k:'torch.float64' for k in keys} for group,keys in runner.PRECISION_KEYS.items()}

    def runtime_info(self):
        return dict(patch=dict(norms=[f'model.layers.{i}.{k}' for i in range(24)
            for k in ('input_layernorm','post_attention_layernorm')]+['model.norm'],rotaries=['model.rotary_emb']),
            torch_threads=1,interop_threads=1,thread_env={k:'1' for k in runner.THREAD_KEYS},
            startup_thread_env={k:'1' for k in runner.THREAD_KEYS},mode='native64')

    def select_group(self,group):
        assert group==self.selected_group+1
        self.selected_group=group; self.reads=0

    def maybe_fail(self):
        if self.failure and self.budget.calls==self.failure[0]: raise self.failure[1]('injected offline failure')

    def probability(self,tokens):
        p=np.zeros((len(tokens),runner.VOCAB),dtype=np.float64)
        for i,token in enumerate(tokens): p[i,int(token)]=1.
        return p

    def prepare(self):
        self.budget.claim('native64_prefix'); self.maybe_fail(); self.observations=self.good_observations()
        self.cache={'length':np.asarray(37,dtype=np.int64)}
        for i in range(24):
            for kind in ('key','value'):
                self.cache[f'layer_{i:02d}_{kind}']=np.full(runner.batch.CACHE_SHAPE,
                    (i+1)/100+self.selected_group,dtype=np.float64)
        self.expanded={k:np.repeat(v,2,axis=0) for k,v in self.cache.items() if k!='length'}
        ids=np.asarray([runner.instrument.PREFIX_IDS],dtype=np.int64)
        return np.zeros((1,896),dtype=np.float64),self.probability([0]),.01,dict(input_after=ids.copy(),
            input_exact=True,observed_output_dtype='torch.float64',observed_probability_dtype='torch.float64')

    def cache_arrays(self):
        self.reads+=1
        if self.defect==('precall_cache',self.budget.calls) and self.reads==2:
            self.cache['layer_00_key'].flat[0]+=1
        return {k:v.copy() for k,v in self.cache.items()}

    def forward(self,x,cached):
        self.maybe_fail(); self.observations=self.good_observations()
        row=self.budget.slots[self.budget.calls-1]; group=row['group_id']
        assert group==self.selected_group
        lanes=list(range(2)) if len(x)==2 else [int(row['label'][-1])]
        step=row['step'] if row['category']=='update' else 1
        if group%4==0:
            y=self.payload['historical_states'][step,lanes].copy(); y[:,30]+=self.history_offset
            top=self.payload['historical_token_ids'][step-1,lanes]
        elif group%4==2:
            y=x+.001; top=np.zeros(len(x),dtype=np.int64)
        else:
            bits=[self.bits[4*(group//4)+((group%4)//2)*2+lane] for lane in lanes]
            y=np.stack([self.payload['references'][bit or 0,step%6] for bit in bits])
            for j,bit in enumerate(bits):
                if bit is None: y[j,25]+=.01*(step+1)
            top=np.zeros(len(x),dtype=np.int64)
        extra=dict(input_after=x.copy(),input_exact=True,observed_output_dtype='torch.float64',
            observed_probability_dtype='torch.float64')
        if cached:
            extra.update(used_cache_before={k:v.copy() for k,v in self.expanded.items()},
                used_cache_length_after=np.asarray(38,dtype=np.int64))
        p=self.probability(top)
        if self.defect and self.budget.calls==self.defect[1]:
            mode=self.defect[0]
            if mode=='precision': self.observations['norm']['reduction']='torch.float32'
            if mode=='nonfinite': y[0,0]=np.nan
            if mode=='shape': y=y[:,:-1]
            if mode=='input': extra['input_after'][0,0]+=1
            if mode=='cache': self.cache['layer_00_key'].flat[0]+=1
            if mode=='expanded': extra['used_cache_before']['layer_00_key'].flat[0]+=1
            if mode=='comparison': y[0,0]+=1e-6
            if mode=='probabilities': p[0,0]+=1e-4
            if mode=='token': p[:]=0.; p[:,1]=1.
        return y.copy(),p,.01,extra



def run_fixture(path,*,payload=None,bits=None,failure=None,defect=None,history_offset=0.,preflight=False):
    payload=synthetic_payload() if payload is None else payload
    args=actual_args(path); args.preflight_only=preflight
    def factory(a,b,r): return FakeBackend(a,b,r,payload=payload,bits=bits,failure=failure,defect=defect,history_offset=history_offset)
    return runner.run(args,backend_factory=factory,input_qualifier=lambda a:(payload,dict(synthetic=True,bindings={})),
        model_qualifier=synthetic_model,source_factory=synthetic_source)


@pytest.fixture(autouse=True)
def small_arrays(request,monkeypatch):
    if request.node.name != 'test_maximum_actual_serialization':
        monkeypatch.setattr(runner,'VOCAB',7)
        monkeypatch.setattr(runner.batch,'CACHE_SHAPE',(1,2,37,1))


def test_schedule_controls_and_case_boundaries_are_fixed():
    slots=runner.call_plan(); assert len(slots)==2624
    assert [sum(s['group_id']==g for s in slots) for g in range(16)]==[41,245,125,245]*4
    assert [s['call'] for s in slots if s['category']=='prefix']==[656*i+x for i in range(4) for x in (1,42,287,412)]
    assert sum(s['category']=='update' for s in slots)==2544
    assert sum(s['batch_size'] for s in slots)==5200
    for g in range(16):
        rows=[s for s in slots if s['group_id']==g]
        assert rows[-1]['step']==(36,240,120,240)[g%4]
        assert all((s['case_index'],s['stage_index'],s['reference_phase'])==(g//4,g%4,2+g//4) for s in rows)
        assert all(s['controls']==runner.CONTROLS[runner.CONTROL_NAMES[g]] for s in rows)
        assert [s['category'] for s in rows[:6]]==['prefix']+['comparison']*4+['update']
    d=runner.derive_design()
    assert d['reference_phases']==[2,3,4,5] and d['schedule']['cases']==4
    assert d['schedule']['groups']==16 and d['readout']['requested_bits']==[0,0,1,1]*4
    assert d['within_case_carry_boundaries']==[[4*i+j,4*i+j+1] for i in range(4) for j in range(3)]
    assert d['between_case_reset_group_ids']==[4,8,12]
    assert d['readout']['closure_pairs']==[[i,i-6] for i in range(235,241)]
    assert d['cross_phase_history']['group_ids']==[0,4,8,12]
    assert not d['cross_phase_history']['gates_chain'] and d['literal_phase_carry']
    assert d['limits']['raw_bytes']==9*1024**3 and d['limits']['finalization_reserve_bytes']==128*1024**2
    args=actual_args('/tmp/unused'); cli=[]
    for k,v in vars(args).items():
        if k!='preflight_only': cli.extend(['--'+k.replace('_','-'),str(v)])
    assert runner.parser().parse_args(cli).prospective_panel==args.prospective_panel
    for pair in (['--alpha','.1'],['--steps','24'],['--seed','3'],['--reference-phase','2'],['--phases','2,3']):
        with pytest.raises(SystemExit): runner.parser().parse_args(cli+pair)


def test_one_named_backend_accepts_all_sixteen_controls_without_reconstruction():
    backend=runner.QwenBackend.__new__(runner.QwenBackend)
    backend.vector=np.linspace(-1,1,896,dtype=np.float64)
    hook=object(); backend.loop=SimpleNamespace(handle=hook,cache=object(),steering=None,temperature=None)
    backend.selected_group=-1
    for group in range(16):
        cache=backend.loop.cache; backend.select_group(group)
        controls=runner.CONTROLS[runner.CONTROL_NAMES[group]]
        assert np.array_equal(backend.loop.steering,controls['alpha']*backend.vector.reshape(1,1,-1))
        assert backend.loop.temperature==10.**controls['log10_temperature']
        assert backend.loop.handle is hook and backend.loop.cache is cache
    for wrong in (15,16,0):
        with pytest.raises(runner.InstrumentFailure): backend.select_group(wrong)


def test_neutral_tail_and_outcome_precedence():
    refs=synthetic_payload()['references']; tail=np.concatenate((refs[1,-1:],refs[1],refs[1]))
    tail[0,50]=1000.
    result=runner.readout(tail,refs)
    assert result['decoded_bit']==1 and len(result['lag_six_phase_l2_residuals'])==6
    assert all(len(r['by_shift'])==6 for r in result['reference_distances'])
    assert runner.readout(tail,np.stack((refs[1],refs[1])))['decoded_bit'] is None
    assert runner.readout(np.repeat(refs[1,0][None],13,axis=0),refs)['decoded_bit'] is None
    bits=[0,0,1,1]*4
    assert runner.panel_outcome(bits)=='panel_qualified'
    wrong=bits.copy();wrong[7]=0
    assert runner.panel_outcome(wrong)=='panel_wrong'
    wrong[12]=None
    assert runner.panel_outcome(wrong)=='panel_unresolved'
    assert runner.chain_outcome([0,0,1,1])=='chain_qualified'
    assert runner.chain_outcome([1,0,1,1])=='chain_wrong'
    assert runner.chain_outcome([1,None,1,1])=='chain_unresolved'
    for bad in (bits[:-1],bits+[1],[True]+bits[1:],['0']+bits[1:]):
        with pytest.raises(runner.InstrumentFailure):runner.panel_outcome(bad)


def test_all_cases_continue_after_finite_wrong_and_unresolved(tmp_path,monkeypatch):
    bits=[0,0,1,1,1,0,1,1,None,0,1,1,0,0,1,1]
    constructed=[]; original=FakeBackend.__init__
    def counted(self,*args,**kwargs):
        constructed.append(self); original(self,*args,**kwargs)
    monkeypatch.setattr(FakeBackend,'__init__',counted)
    path=tmp_path/'run'; result=run_fixture(path,bits=bits,history_offset=.125,defect=('token',6))
    assert len(constructed)==1 and result['model_calls']==2624
    assert result['status']=='completed' and result['outcome']=='panel_unresolved' and result['resource_limits_passed']
    assert [c['outcome'] for c in result['cases']]==['chain_qualified','chain_wrong','chain_unresolved','chain_qualified']
    assert [g['model_calls'] for g in result['groups']]==[41,245,125,245]*4
    assert all(g['closed'] and g['status']=='completed' for g in result['groups'])
    assert [r['readout']['decoded_bit'] for r in result['rows']]==bits
    payload=synthetic_payload(); carried=None; owned=[]
    for g in range(16):
        case,stage=divmod(g,4); phase=case+2
        if stage==0: carried=payload['initial_states'][case]
        with np.load(path/f'group-{g:02d}-states.npz') as z:
            assert z['states'].shape==((37,241,121,241)[stage],2,896)
            assert runner.same_bytes(z['states'][0],carried)
            assert all(z[k].dtype==np.int64 and z[k].shape==() and int(z[k])==v
                       for k,v in dict(case_index=case,stage_index=stage,reference_phase=phase).items())
            carried=z['states'][-1].copy()
            for name,value in runner.CONTROLS[runner.CONTROL_NAMES[g]].items():
                assert runner.same_bytes(z[name],np.asarray(value)) and int(z[name+'_bits'])==int(np.asarray(value).view(np.uint64))
        marker=json.loads((path/f'group-{g:02d}.json').read_text()); assert marker==result['groups'][g]
        assert (marker['case_index'],marker['stage_index'],marker['reference_phase'])==(case,stage,phase)
        summary=json.loads((path/f'group-{g:02d}-summary.json').read_text())
        assert summary['predecessor_group_id']==(g-1 if stage else None)
        for b in marker['artifacts']:
            assert runner.binding(path/b['path'],flat=True)==b; owned.append(b['path'])
    assert len(owned)==len(set(owned))
    for i,row in enumerate(result['rows']):
        case,local=divmod(i,4); group=4*case+1+2*(local//2)
        assert (row['group_id'],row['lane'],row['requested_bit'])==(group,local%2,local//2)
        assert (row['case_index'],row['stage_index'],row['reference_phase'],row['initial_reference_phase'])==(case,group%4,case+2,case+2)
    for case,c in enumerate(result['cases']):
        assert c['status']=='completed' and c['reference_phase']==case+2
        assert c['row_ids']==c['group_ids']==list(range(4*case,4*case+4))
        diagnostic=c['cross_phase_history']; assert diagnostic==result['cross_phase_history'][case]
        assert diagnostic['case_index']==case and diagnostic['current_reference_phase']==case+2
        assert not diagnostic['same_input'] and not diagnostic['tolerance_applicable'] and not diagnostic['gates_chain']
        assert diagnostic['comparison_kind']=='different_initial_phases' and diagnostic['historical_reference_phase']==0
        assert diagnostic['maximum_coordinate_gap']>=.125 and not diagnostic['state_bytes_equal']
        assert not any('tolerance' in k or 'passed' in k or 'replay' in k for k in diagnostic if k!='tolerance_applicable')
        if case==0:
            assert [lane['token_mismatch_count'] for lane in diagnostic['lanes']]==[1,1]
        with np.load(path/f'case-{case:02d}-cross-phase-history.npz') as z:
            assert int(z['case_index'])==case and int(z['current_reference_phase'])==case+2
            assert z['coordinate_absolute_gaps'].shape==(37,2,896) and z['token_equal'].shape==(36,2)
            assert not z['state_bytes_equal'][0].any()
            assert np.array_equal(z['per_state_coordinate_maxima'],z['coordinate_absolute_gaps'].max(axis=2))
    events=[json.loads(s) for s in (path/'events.jsonl').read_text().splitlines()]
    assert sum(e['event']=='model_call_started' for e in events)==2624
    assert sum(e['event']=='model_call_completed' for e in events)==2624
    assert [e['control_name'] for e in events if e['event']=='control_selected']==['E0','H','E1','H']*4
    starts=[e for e in events if e['event']=='case_started']
    assert [e['reference_phase'] for e in starts]==[2,3,4,5]
    assert [e['reset_between_cases'] for e in starts]==[False,True,True,True]
    for i,e in enumerate(starts):
        assert e['initial_state_sha256']==runner.array_hash(payload['initial_states'][i])
        assert e['previous_terminal_state_sha256']==(None if i==0 else result['cases'][i-1]['terminal_state_sha256'])
    assert {p.name for p in path.glob('*-tail.npz')}=={f'group-{g:02d}-tail.npz' for g in range(16) if g%4 in (1,3)}
    for index in (999,1000,2624):
        slot=result['calls'][index-1]; prefix=f'call-{index:03d}-{slot["label"]}'
        assert (path/(prefix+'.json')).is_file() and (path/(prefix+'.npz')).is_file()
    manifest=json.loads((path/'manifest.json').read_text())
    assert {b['path'] for b in manifest['files']}=={p.name for p in path.iterdir()}-{'manifest.json'}
    assert result['terminal_resources']['final_raw_bytes']==sum(p.stat().st_size for p in path.iterdir())


@pytest.mark.parametrize('exception,status',[(RuntimeError,'execution_failed'),(OSError,'execution_failed'),
    (runner.ResourceLimit,'incomplete'),(TimeoutError,'incomplete'),(MemoryError,'incomplete'),(ValueError,'instrument_failed')])
def test_failed_attempt_keeps_inputs_and_unrun_groups(tmp_path,exception,status):
    path=tmp_path/'run'; result=run_fixture(path,failure=(6,exception))
    assert result['status']==status and result['model_calls']==6
    assert result['cross_phase_history']==[None]*4
    assert [g['model_calls'] for g in result['groups']]==[6]+[0]*15
    assert [g['status'] for g in result['groups']]==[status]+['not_run']*15
    assert [c['status'] for c in result['cases']]==[status]+['not_run']*3
    assert len(result['rows'])==16 and len(result['calls'])==2624
    row=json.loads((path/'call-006-g00-update-001.json').read_text())
    assert row['status']=='failed' and row['exception_category']==status
    with np.load(path/'call-006-g00-update-001.npz') as z: assert 'input' in z and 'output' not in z


@pytest.mark.parametrize('mode',['precision','nonfinite','shape','input','cache','expanded','probabilities'])
def test_returned_defect_preserves_raw_and_admission_distinction(tmp_path,mode):
    path=tmp_path/'run'; result=run_fixture(path,defect=(mode,6))
    assert result['status']=='instrument_failed' and result['model_calls']==6
    with np.load(path/'call-006-g00-update-001.npz') as z: assert 'probabilities' in z and 'output' in z
    summary=json.loads((path/'group-00-summary.json').read_text())
    assert summary['completed_updates']==0 and summary['returned_updates']==(0 if mode=='shape' else 1)


def test_preinvocation_cache_failure_is_retained_without_attempt(tmp_path):
    path=tmp_path/'run'; result=run_fixture(path,defect=('precall_cache',1))
    assert result['status']=='instrument_failed' and result['model_calls']==1
    row=json.loads((path/'blocked-call-002-g00-compare-cached.json').read_text())
    assert row['status']=='blocked_before_invocation' and row['model_calls']==0
    with np.load(path/'blocked-call-002-g00-compare-cached.npz') as z: assert 'cache_before_layer_00_key' in z
    assert result['calls'][1]['blocked_evidence']=='blocked-call-002-g00-compare-cached.json'


def test_next_case_prefix_failure_preserves_completed_case(tmp_path):
    path=tmp_path/'run'; result=run_fixture(path,failure=(657,OSError))
    assert result['status']=='execution_failed' and result['model_calls']==657
    assert [c['status'] for c in result['cases']]==['completed','execution_failed','not_run','not_run']
    assert result['cases'][0]['outcome']=='chain_qualified'
    assert [r['status'] for r in result['rows'][:4]]==['completed']*4
    with np.load(path/'group-04-states.npz') as z:
        assert z['states'].shape==(1,2,896)
        assert runner.same_bytes(z['states'][0],synthetic_payload()['initial_states'][1])
    for group in result['groups'][:4]:
        for b in group['artifacts']: assert runner.binding(path/b['path'],flat=True)==b


def test_late_failure_preserves_every_previously_closed_group(tmp_path):
    path=tmp_path/'run'; result=run_fixture(path,failure=(2623,OSError))
    assert result['status']=='execution_failed' and result['model_calls']==2623
    assert [g['model_calls'] for g in result['groups']]==[41,245,125,245]*3+[41,245,125,244]
    assert [c['status'] for c in result['cases']]==['completed']*3+['execution_failed']
    assert [r['status'] for r in result['rows']]==['completed']*14+['execution_failed']*2
    assert result['cases'][3]['terminal_state_sha256'] is None
    for group in result['groups'][:15]:
        assert group['status']=='completed'
        for b in group['artifacts']: assert runner.binding(path/b['path'],flat=True)==b
    with np.load(path/'group-15-states.npz') as z: assert z['states'].shape==(239,2,896)
    assert (path/'call-2623-g15-update-239.npz').is_file()


def test_comparison_failure_does_not_advance_state(tmp_path):
    path=tmp_path/'run'; result=run_fixture(path,defect=('comparison',3))
    assert result['status']=='instrument_failed' and result['model_calls']==5
    with np.load(path/'group-00-states.npz') as z: assert z['states'].shape==(1,2,896)


def test_prefix_failure_and_input_failure_are_explicit(tmp_path):
    path=tmp_path/'prefix'; result=run_fixture(path,failure=(1,OSError))
    assert result['status']=='execution_failed' and result['model_calls']==1
    with np.load(path/'call-001-g00-prefix.npz') as z: assert 'input' in z and 'probabilities' not in z
    args=actual_args(tmp_path/'input'); args.trajectory_h0=tmp_path/'bad.npz'; args.trajectory_h0.write_bytes(b'bad')
    result=runner.run(args,backend_factory=lambda *a:pytest.fail('model'),model_qualifier=synthetic_model,source_factory=synthetic_source)
    assert result['status']=='instrument_failed' and result['model_calls']==0 and not result['model_loaded']
    assert all(g['closed'] and g['status']=='not_run' for g in result['groups'])


def test_preflight_never_constructs_model_and_all_groups_close(tmp_path):
    result=run_fixture(tmp_path/'run',preflight=True)
    assert result['status']=='preflight_passed' and result['outcome']=='not_run_preflight'
    assert result['model_calls']==0 and not result['model_loaded'] and result['resource_limits_passed']
    assert len(result['groups'])==16 and all(g['closed'] and g['status']=='not_run' for g in result['groups'])
    assert all(c['status']=='not_run' and c['cross_phase_history'] is None for c in result['cases'])


def test_raw_admission_preserves_finalization_and_thread_gate(tmp_path,monkeypatch):
    normal=runner.RAW_CAP; monkeypatch.setattr(runner,'RAW_CAP',runner.FINAL_RESERVE+1024*1024)
    result=run_fixture(tmp_path/'cap')
    assert result['status']=='incomplete' and result['model_calls']==0
    assert (tmp_path/'cap/result.json').is_file() and (tmp_path/'cap/manifest.json').is_file()
    monkeypatch.setattr(runner,'RAW_CAP',normal)
    class Bad(FakeBackend):
        def runtime_info(self):
            row=super().runtime_info(); row['startup_thread_env']['OPENBLAS_NUM_THREADS']='2'; return row
    result=runner.run(actual_args(tmp_path/'threads'),backend_factory=Bad,
        input_qualifier=lambda a:(synthetic_payload(),dict(bindings={})),model_qualifier=synthetic_model,source_factory=synthetic_source)
    assert result['status']=='instrument_failed' and result['model_calls']==0


def test_actual_eight_pins_and_literal_phase_pairs(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,'VOCAB',151936)
    payload,proof=runner.qualify_inputs(actual_args(tmp_path/'unused'))
    assert len(proof['bindings'])==8 and proof['reference_phases']==[2,3,4,5]
    assert proof['historical_reference_phase']==0 and not proof['same_historical_input']
    assert payload['reference_phases'].dtype==np.int64
    assert runner.same_bytes(payload['historical_states'][0],payload['references'][:,0])
    for case,phase in enumerate((2,3,4,5)):
        assert runner.same_bytes(payload['initial_states'][case],payload['references'][:,phase])
        assert not runner.same_bytes(payload['initial_states'][case],payload['historical_states'][0])


@pytest.mark.parametrize('defect',['phases','indices','pair','shape','dtype','extra'])
def test_prospective_artifact_exact_contract(tmp_path,defect):
    refs=synthetic_payload()['references']
    arrays=dict(initial_states=np.stack([refs[:,p] for p in (2,3,4,5)]),reference_indices=np.arange(2,dtype=np.int64),
                reference_phases=np.asarray([2,3,4,5],dtype=np.int64))
    if defect=='phases': arrays['reference_phases']=np.asarray([2,3,5,4],dtype=np.int64)
    if defect=='indices': arrays['reference_indices']=np.array([1,0],dtype=np.int64)
    if defect=='pair': arrays['initial_states'][3]=refs[:,0]
    if defect=='shape': arrays['initial_states']=arrays['initial_states'][:3]
    if defect=='dtype': arrays['initial_states']=arrays['initial_states'].astype(np.float32)
    if defect=='extra': arrays['hidden_selection']=np.asarray(0,dtype=np.int64)
    path=tmp_path/'panel.npz'; np.savez(path,**arrays)
    with pytest.raises(runner.InstrumentFailure): runner.load_prospective_panel(path,refs)


def test_initial_and_historical_phase_contracts_are_independent():
    p=synthetic_payload(); refs=p['references']; states=p['historical_states']; tokens=p['historical_token_ids']
    pairs=p['initial_states'].copy(); pairs[2]=refs[:,1]
    with pytest.raises(runner.InstrumentFailure): runner.derive_inputs(refs,states,tokens,pairs)
    wrong=states.copy();wrong[0]=refs[:,2]
    with pytest.raises(runner.InstrumentFailure): runner.derive_inputs(refs,wrong,tokens,p['initial_states'])


def test_changed_prospective_pin_is_rejected_before_model(tmp_path):
    args=actual_args(tmp_path/'run'); args.prospective_panel=tmp_path/'wrong.npz'; args.prospective_panel.write_bytes(b'changed')
    result=runner.run(args,backend_factory=lambda *a:pytest.fail('model'),source_factory=synthetic_source,model_qualifier=synthetic_model)
    assert result['status']=='instrument_failed' and result['model_calls']==0 and not result['model_loaded']
    assert all(g['closed'] and g['status']=='not_run' for g in result['groups'])


def test_proven_numerical_and_file_primitives_unchanged_from_phase1():
    def nodes(path):
        return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    old=nodes(runner.ROOT/'tools/run_public_short_write0_phase1.py'); new=nodes(Path(runner.__file__))
    dump=lambda n:ast.dump(n,include_attributes=False)
    for name in ('InstrumentFailure','ResourceLimit','require','finite','same_bytes','array_hash','cache_hash','binding',
                 'checked_norm','binding_index','require_historical_binding','qualify_model','readout','chain_outcome',
                 'failure_status','Budget','precision_ok','validate_readiness','evaluate','compare','parser','main'):
        assert dump(old[name])==dump(new[name]),name
    for cls,methods in [('Recorder',('size','admit','event','register','json','save')),
                        ('QwenBackend',('__init__','prepare','runtime_info'))]:
        before={n.name:n for n in old[cls].body if isinstance(n,ast.FunctionDef)}
        after={n.name:n for n in new[cls].body if isinstance(n,ast.FunctionDef)}
        for name in methods: assert dump(before[name])==dump(after[name]),(cls,name)


def test_source_inventory_binds_panel_and_held_sources(tmp_path):
    source=runner.source_metadata(actual_args(tmp_path/'unused')); records={r['path']:r for r in source['files']}
    expected={'tools/run_public_short_write0_phase_panel.py','tests/test_run_public_short_write0_phase_panel.py',
        'tools/audit_public_short_write0_phase_panel.py','tests/test_audit_public_short_write0_phase_panel.py',
        'tools/run_public_short_write0_phase1.py','tests/test_run_public_short_write0_phase1.py',
        'tools/run_public_short_write0_chain.py','tests/test_run_public_short_write0_chain.py',
        'tools/run_public_nominal_handoffs.py','tools/run_public_state_intervention.py','tools/benchmark_qwen_batch.py',
        'tools/validate_qwen_map.py','tools/check_p6_writes.py','tools/check_p6_hold.py','tools/check_p6_recovery.py',
        'docs/public-short-write0-phase-panel.md','pyproject.toml','uv.lock'}
    assert set(records)==expected and len(source['source_commit'])==40
    assert isinstance(source['working_tree_status'],list)
    for name in expected: assert records[name]['sha256']==runner.instrument.sha256(runner.ROOT/name)


def test_maximum_actual_serialization(tmp_path):
    """One retained full-vocabulary 2624-call fixture; scripted outputs, never a model."""
    requested=os.environ.get('SHRIMP_PHASE_PANEL_FULL_FIXTURE')
    if not requested: pytest.skip('Full-size retained fixture is run once by the preparation harness')
    path=Path(requested)
    assert path.is_relative_to('/data2/shrimp-memory-data/preparations/public-short-write0-phase-panel')
    payload,_=runner.qualify_inputs(actual_args(path)); result=run_fixture(path,payload=payload)
    assert result['status']=='completed' and result['outcome']=='panel_qualified'
    assert result['model_calls']==2624 and result['resource_limits_passed']
    assert len(list(path.glob('call-*.npz')))==2624
    probabilities=0; cache_bytes=0; cache_arrays=0; maximum_call_file=0
    for file in sorted(path.glob('call-*.npz')):
        maximum_call_file=max(maximum_call_file,file.stat().st_size)
        with np.load(file,allow_pickle=False) as z:
            assert z['probabilities'].dtype==np.float64 and z['probabilities'].shape[1]==151936
            probabilities+=z['probabilities'].nbytes
    for file in sorted(path.glob('group-*-*prefix-cache.npz')):
        with np.load(file,allow_pickle=False) as z:
            cache_bytes+=sum(z[k].nbytes for k in z.files); cache_arrays+=len(z.files)
    assert probabilities==5200*151936*8==6320537600
    assert len(list(path.glob('group-*-*prefix-cache.npz')))==32 and cache_arrays==32*49
    actual=sum(p.stat().st_size for p in path.iterdir() if p.is_file())
    assert result['terminal_resources']['final_raw_bytes']==actual and actual+runner.FINAL_RESERVE<runner.RAW_CAP
    receipt=dict(synthetic=True,model_calls=2624,files=len(list(path.iterdir())),probability_vectors=5200,
        probability_bytes=probabilities,cache_array_bytes=cache_bytes,cache_arrays=cache_arrays,cache_artifacts=32,
        maximum_call_file_bytes=maximum_call_file,actual_raw_bytes=actual,
        raw_plus_final_reserve_bytes=actual+runner.FINAL_RESERVE,raw_cap_bytes=runner.RAW_CAP,
        peak_rss_bytes=result['peak_rss_bytes'],elapsed_seconds=result['elapsed_seconds'],
        threads={k:os.environ[k] for k in runner.THREAD_KEYS},
        measured_scope='Actual full-size serialization of four sequential batch2 cases with scripted outputs; no model evidence')
    (path.parent/(path.name+'-serialization.json')).write_text(json.dumps(receipt,indent=2)+'\n')



