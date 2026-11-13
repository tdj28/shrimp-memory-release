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
import run_public_carried72_handoff_phase_panel as runner


def actual_args(output):
    base = Path('/data2/shrimp-memory-data/audits/p6-writes/20261025T222347Z-writes')
    return SimpleNamespace(short_prefix=Path('/data2/shrimp-memory-data/preparations/public-canonical-write0-phase-panel/20261029T172047Z/accepted-short-prefix.npz'),
        short_prefix_provenance=Path('/data2/shrimp-memory-data/preparations/public-canonical-write0-phase-panel/20261029T172047Z/accepted-short-prefix-provenance.json'),
        prospective_panel=Path('/data2/shrimp-memory-data/preparations/public-short-write0-phase-panel/20261029T153845Z/prospective-phase-panel.npz'),
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
    pairs=np.stack([refs[:,p] for p in runner.PHASES])
    short=np.repeat(pairs[:,None],37,axis=1);short[:,:,:,20]+=np.arange(37)[None,:,None]/1000
    return runner.derive_inputs(refs,states,np.zeros((120,2),dtype=np.int64),pairs,short,np.zeros((3,36,2),dtype=np.int64))



def synthetic_source(args):
    return dict(source_commit='synthetic',files=[],working_tree_status=[],synthetic=True)



def synthetic_model(args):
    return dict(synthetic=True,no_model_loaded=True)



class FakeBackend:
    """Scripted outputs with actual production array geometry, never a model."""
    def __init__(self,args,budget,recorder,*,payload=None,bits=None,failure=None,defect=None,history_offset=0.):
        self.budget,self.recorder=budget,recorder
        self.payload=synthetic_payload() if payload is None else payload
        self.bits=[0,0,1,1]*3 if bits is None else bits
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
            if step<=36:
                y=self.payload['short_prefix_states'][group//4,step,lanes].copy();y[:,30]+=self.history_offset
                top=self.payload['short_prefix_token_ids'][group//4,step-1,lanes]
            else:
                y=x+.001;top=np.zeros(len(x),dtype=np.int64)
        elif group%4==2:
            y=x+.001;top=np.zeros(len(x),dtype=np.int64)
        else:
            bits=[self.bits[4*(group//4)+2*((group%4)//2)+lane] for lane in lanes]
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
    if request.node.name!='test_maximum_actual_serialization':
        monkeypatch.setattr(runner,'VOCAB',7)
        monkeypatch.setattr(runner.batch,'CACHE_SHAPE',(1,2,37,1))


def test_fixed_1788_call_schedule_controls_identity_and_cli():
    slots=runner.call_plan();assert len(slots)==1788
    assert [sum(s['group_id']==g for s in slots) for g in range(12)]==[77,149,125,245]*3
    assert [s['call'] for s in slots if s['category']=='prefix']==[596*c+o for c in range(3) for o in (1,78,227,352)]
    assert sum(s['category']=='update' for s in slots)==1728
    assert sum(s['batch_size'] for s in slots)==3540
    for s in slots:
        g=s['group_id'];case,stage=divmod(g,4)
        assert (s['case_index'],s['stage_index'],s['reference_phase'],s['packet_index'])==(case,stage,case+3,case+1)
        assert s['controls']==runner.CONTROLS[('E0','H','E1','H')[stage]]
        assert s['controls_hex']=={k:v.hex() for k,v in s['controls'].items()}
        assert s['controls_binary64']=={k:f'{int(np.asarray(v).view(np.uint64)):016x}' for k,v in s['controls'].items()}
    d=runner.derive_design();assert d['schedule']['total_calls']==1788
    assert d['within_case_carry_boundaries']==[[4*c+j,4*c+j+1] for c in range(3) for j in range(3)]
    assert d['between_case_reset_group_ids']==[4,8]
    assert d['readout']['requested_bits']==[0,0,1,1]*3
    assert d['prefix_replay']['group_ids']==[0,4,8] and d['prefix_replay']['source_group_ids']==[4,8,12]
    for g in (1,3,5,7,9,11):
        end=144 if g%4==1 else 240
        assert d['readout']['retained_state_indices_by_group'][str(g)]==[end-12,end]
        assert d['readout']['closure_pairs_by_group'][str(g)]==[[i,i-6] for i in range(end-5,end+1)]
    assert d['limits']['raw_bytes']==9*1024**3 and d['limits']['finalization_reserve_bytes']==128*1024**2
    assert d['limits']['model_calls']==1788 and d['limits']['internal_seconds']==3600
    args=actual_args('/tmp/unused');cli=[]
    for key,value in vars(args).items():
        if key!='preflight_only':cli.extend(['--'+key.replace('_','-'),str(value)])
    parsed=runner.parser().parse_args(cli);assert parsed.short_prefix==args.short_prefix
    for pair in (['--alpha','.1'],['--steps','144'],['--seed','3'],['--reference-phase','2'],['--hold','240']):
        with pytest.raises(SystemExit):runner.parser().parse_args(cli+pair)


def test_single_backend_selects_all_fixed_controls_without_replacing_hook():
    b=runner.QwenBackend.__new__(runner.QwenBackend);b.vector=np.linspace(-1,1,896,dtype=np.float64)
    hook=object();cache=object();b.loop=SimpleNamespace(handle=hook,cache=cache,steering=None,temperature=None);b.selected_group=-1
    for g in range(12):
        b.select_group(g);c=runner.CONTROLS[runner.CONTROL_NAMES[g]]
        assert np.array_equal(b.loop.steering,c['alpha']*b.vector.reshape(1,1,-1))
        assert b.loop.temperature==10.**c['log10_temperature'] and b.loop.handle is hook and b.loop.cache is cache
    with pytest.raises(runner.InstrumentFailure):b.select_group(12)


def test_six_pair_decoder_and_four_bit_neutral_outcomes():
    refs=synthetic_payload()['references'];tail=np.concatenate((refs[1,-1:],refs[1],refs[1]));tail[0,50]=1000.
    r=runner.readout(tail,refs);assert r['decoded_bit']==1 and len(r['lag_six_phase_l2_residuals'])==6
    assert all(len(d['by_shift'])==6 for d in r['reference_distances'])
    assert runner.readout(tail,np.stack((refs[1],refs[1])))['decoded_bit'] is None
    assert runner.readout(np.repeat(refs[1,0][None],13,axis=0),refs)['decoded_bit'] is None
    assert runner.case_outcome([0,0,1,1])=='case_qualified'
    assert runner.case_outcome([0,1,1,1])=='case_wrong'
    assert runner.case_outcome([0,None,1,1])=='case_unresolved'
    assert runner.panel_outcome([0,0,1,1]*3)=='panel_qualified'
    assert runner.panel_outcome([0,1,1,1]+[0,0,1,1]*2)=='panel_wrong'
    assert runner.panel_outcome([0,None,1,1]+[0,0,1,1]*2)=='panel_unresolved'
    for bad in ([0]*11,[True]+[0]*11,['0']+[0]*11):
        with pytest.raises(runner.InstrumentFailure):runner.panel_outcome(bad)


def test_complete_mixed_word_preserves_every_carry_reset_row_and_diagnostic(tmp_path,monkeypatch):
    bits=[0,0,1,1, 0,1,1,1, None,0,1,1];constructed=[];original=FakeBackend.__init__
    def counted(self,*a,**kw):constructed.append(self);original(self,*a,**kw)
    monkeypatch.setattr(FakeBackend,'__init__',counted)
    path=tmp_path/'run';r=run_fixture(path,bits=bits,history_offset=.125,defect=('token',6))
    assert len(constructed)==1 and r['status']=='completed' and r['model_calls']==1788
    assert r['outcome']=='panel_unresolved' and r['resource_limits_passed'] and r['prefix_replay_passed'] is False
    assert [c['outcome'] for c in r['cases']]==['case_qualified','case_wrong','case_unresolved']
    assert [row['readout']['decoded_bit'] for row in r['rows']]==bits
    assert [g['model_calls'] for g in r['groups']]==[77,149,125,245]*3
    payload=synthetic_payload();owned=[]
    for g in range(12):
        case,stage=divmod(g,4);steps=(72,144,120,240)[stage]
        if stage==0:carried=payload['initial_states'][case]
        with np.load(path/f'group-{g:02d}-states.npz') as z:
            assert z['states'].shape==(steps+1,2,896) and runner.same_bytes(z['states'][0],carried)
            assert all(z[k].dtype==np.int64 and z[k].shape==() and int(z[k])==v for k,v in
                dict(case_index=case,stage_index=stage,reference_phase=case+3,packet_index=case+1).items())
            carried=z['states'][-1].copy();trajectory=z['states'].copy()
        summary=json.loads((path/f'group-{g:02d}-summary.json').read_text())
        assert summary['predecessor_group_id']==(g-1 if stage else None)
        assert summary['initial_state_sha256']==runner.array_hash(trajectory[0])
        assert summary['terminal_state_sha256']==runner.array_hash(trajectory[-1])
        if stage in (1,3):
            with np.load(path/f'group-{g:02d}-tail.npz') as z:
                assert np.array_equal(z['state_indices'],np.arange(steps-12,steps+1))
                assert runner.same_bytes(z['states'],trajectory[steps-12:steps+1])
            for lane in range(2):
                expected=np.linalg.norm(trajectory[steps-5:steps+1,lane]-trajectory[steps-11:steps-5,lane],axis=1)
                assert np.array_equal(summary['readouts'][lane]['lag_six_phase_l2_residuals'],expected)
        marker=json.loads((path/f'group-{g:02d}.json').read_text());assert marker==r['groups'][g] and marker['closed']
        for b in marker['artifacts']:
            assert runner.binding(path/b['path'],flat=True)==b;owned.append(b['path'])
    assert len(owned)==len(set(owned))
    for i,row in enumerate(r['rows']):
        case,within=divmod(i,4);write,lane=divmod(within,2)
        assert (row['group_id'],row['case_index'],row['lane'],row['requested_bit'],row['reference_phase'])==(4*case+2*write+1,case,lane,write,case+3)
    for case,c in enumerate(r['cases']):
        assert c['group_ids']==c['row_ids']==list(range(4*case,4*case+4))
        d=c['prefix_replay'];assert d==r['prefix_replay'][case] and d['same_input'] and not d['gates_readout']
        assert not d['passed'] and d['maximum_coordinate_gap']>=.125 and d['state_count']==37 and d['update_count']==36
        if case==0:assert [v['token_mismatch_count'] for v in d['lanes']]==[1,1]
    events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
    assert sum(e['event']=='model_call_started' for e in events)==1788
    assert sum(e['event']=='model_call_completed' for e in events)==1788
    assert [e['control_name'] for e in events if e['event']=='control_selected']==['E0','H','E1','H']*3
    assert [e['reset_between_cases'] for e in events if e['event']=='case_started']==[False,True,True]
    assert {p.name for p in path.glob('*-tail.npz')}=={f'group-{g:02d}-tail.npz' for g in (1,3,5,7,9,11)}
    manifest=json.loads((path/'manifest.json').read_text())
    assert {b['path'] for b in manifest['files']}=={p.name for p in path.iterdir()}-{'manifest.json'}
    assert r['terminal_resources']['final_raw_bytes']==sum(p.stat().st_size for p in path.iterdir())


@pytest.mark.parametrize('exception,status',[(RuntimeError,'execution_failed'),(OSError,'execution_failed'),
    (runner.ResourceLimit,'incomplete'),(TimeoutError,'incomplete'),(MemoryError,'incomplete'),(ValueError,'instrument_failed')])
def test_early_failure_keeps_attempt_and_all_planned_rows(tmp_path,exception,status):
    p=tmp_path/'run';r=run_fixture(p,failure=(6,exception))
    assert r['status']==status and r['model_calls']==6 and r['prefix_replay']==[None]*3
    assert [g['model_calls'] for g in r['groups']]==[6]+[0]*11
    assert [g['status'] for g in r['groups']]==[status]+['not_run']*11
    assert len(r['rows'])==12 and len(r['calls'])==1788 and all(g['closed'] for g in r['groups'])
    row=json.loads((p/'call-006-g00-update-001.json').read_text());assert row['status']=='failed' and row['exception_category']==status
    with np.load(p/'call-006-g00-update-001.npz') as z:assert 'input' in z and 'output' not in z


@pytest.mark.parametrize('mode',['precision','nonfinite','shape','input','cache','expanded','probabilities'])
def test_returned_instrument_defect_is_retained(tmp_path,mode):
    p=tmp_path/'run';r=run_fixture(p,defect=(mode,6))
    assert r['status']=='instrument_failed' and r['model_calls']==6
    with np.load(p/'call-006-g00-update-001.npz') as z:assert 'probabilities' in z and 'output' in z
    row=json.loads((p/'group-00-summary.json').read_text())
    assert row['completed_updates']==0 and row['returned_updates']==(0 if mode=='shape' else 1)


def test_preinvocation_failure_retains_zero_attempt_slot(tmp_path):
    p=tmp_path/'run';r=run_fixture(p,defect=('precall_cache',1))
    assert r['status']=='instrument_failed' and r['model_calls']==1
    row=json.loads((p/'blocked-call-002-g00-compare-cached.json').read_text());assert row['status']=='blocked_before_invocation'
    with np.load(p/'blocked-call-002-g00-compare-cached.npz') as z:assert 'cache_before_layer_00_key' in z
    assert r['calls'][1]['blocked_evidence']=='blocked-call-002-g00-compare-cached.json'


def test_carried_write1_prefix_failure_keeps_h144_readout(tmp_path):
    p=tmp_path/'run';r=run_fixture(p,failure=(227,OSError))
    assert r['status']=='execution_failed' and r['model_calls']==227
    assert [row['status'] for row in r['rows'][:4]]==['completed','completed','not_run','not_run']
    assert [g['model_calls'] for g in r['groups']]==[77,149,1]+[0]*9
    with np.load(p/'group-01-states.npz') as z:carried=z['states'][-1].copy()
    with np.load(p/'group-02-states.npz') as z:assert runner.same_bytes(z['states'][0],carried)


def test_next_case_prefix_failure_keeps_complete_prior_word(tmp_path):
    p=tmp_path/'run';r=run_fixture(p,failure=(597,OSError))
    assert r['status']=='execution_failed' and r['model_calls']==597
    assert [c['status'] for c in r['cases']]==['completed','execution_failed','not_run']
    assert r['cases'][0]['outcome']=='case_qualified'
    with np.load(p/'group-04-states.npz') as z:assert runner.same_bytes(z['states'][0],synthetic_payload()['initial_states'][1])


def test_late_failure_retains_first_hold_and_partial_final_hold(tmp_path):
    p=tmp_path/'run';r=run_fixture(p,failure=(1787,OSError))
    assert r['status']=='execution_failed' and r['model_calls']==1787
    assert [g['model_calls'] for g in r['groups']]==[77,149,125,245]*2+[77,149,125,244]
    assert [row['status'] for row in r['rows']]==['completed']*10+['execution_failed']*2
    assert r['prefix_replay_passed'] is None
    for g in r['groups'][:11]:
        assert g['status']=='completed'
        for b in g['artifacts']:assert runner.binding(p/b['path'],flat=True)==b
    with np.load(p/'group-11-states.npz') as z:assert z['states'].shape==(239,2,896)
    assert (p/'call-1787-g11-update-239.npz').is_file()


def test_comparison_failure_prevents_trajectory_updates(tmp_path):
    p=tmp_path/'run';r=run_fixture(p,defect=('comparison',3))
    assert r['status']=='instrument_failed' and r['model_calls']==5
    with np.load(p/'group-00-states.npz') as z:assert z['states'].shape==(1,2,896)


def test_actual_ten_pins_literal_inputs_and_zero_model_preflight(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'VOCAB',151936)
    args=actual_args(tmp_path/'run');payload,proof=runner.qualify_inputs(args)
    assert len(proof['bindings'])==10 and proof['same_phase_short_prefix_input']
    assert payload['initial_states'].shape==(3,2,896)
    assert np.array_equal(payload['reference_phases'],[3,4,5]) and np.array_equal(payload['packet_indices'],[1,2,3])
    with np.load(args.short_prefix) as z:
        assert runner.same_bytes(payload['short_prefix_states'],z['states'][[1,2,3]])
        assert runner.same_bytes(payload['short_prefix_token_ids'],z['token_ids'][[1,2,3]])
    for case,phase in enumerate((3,4,5)):assert runner.same_bytes(payload['initial_states'][case],payload['references'][:,phase])
    args.preflight_only=True
    r=runner.run(args,backend_factory=lambda *a:pytest.fail('backend forbidden'),model_qualifier=synthetic_model,source_factory=synthetic_source)
    assert r['status']=='preflight_passed' and r['model_calls']==0 and not r['model_loaded'] and r['resource_limits_passed']
    assert len(r['groups'])==12 and len(r['rows'])==12 and all(g['closed'] and g['status']=='not_run' for g in r['groups'])


@pytest.mark.parametrize('key',['short_prefix','short_prefix_provenance','prospective_panel'])
def test_changed_pin_fails_before_backend(tmp_path,key):
    a=actual_args(tmp_path/'run');p=tmp_path/'changed';p.write_bytes(b'changed');setattr(a,key,p)
    r=runner.run(a,backend_factory=lambda *a:pytest.fail('backend forbidden'),model_qualifier=synthetic_model,source_factory=synthetic_source)
    assert r['status']=='instrument_failed' and r['model_calls']==0 and not r['model_loaded']
    assert all(g['closed'] and g['status']=='not_run' for g in r['groups'])


def test_resource_and_startup_thread_failures_are_not_u(tmp_path,monkeypatch):
    normal=runner.RAW_CAP;monkeypatch.setattr(runner,'RAW_CAP',runner.FINAL_RESERVE+1024*1024)
    r=run_fixture(tmp_path/'cap');assert r['status']=='incomplete' and r['model_calls']==0
    monkeypatch.setattr(runner,'RAW_CAP',normal)
    class Bad(FakeBackend):
        def runtime_info(self):
            d=super().runtime_info();d['startup_thread_env']['OPENBLAS_NUM_THREADS']='2';return d
    r=runner.run(actual_args(tmp_path/'threads'),backend_factory=Bad,input_qualifier=lambda a:(synthetic_payload(),{}),
        model_qualifier=synthetic_model,source_factory=synthetic_source)
    assert r['status']=='instrument_failed' and r['model_calls']==0


def test_numerical_storage_and_input_primitives_match_held_intermediate():
    def nodes(p):return {n.name:n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    old=nodes(runner.ROOT/'tools/run_public_intermediate_write0_phase_panel.py');new=nodes(Path(runner.__file__))
    dump=lambda n:ast.dump(n,include_attributes=False)
    for name in ('InstrumentFailure','ResourceLimit','require','finite','same_bytes','array_hash','cache_hash','binding',
        'checked_norm','load_prospective_panel','derive_inputs','load_short_prefix','binding_index','require_historical_binding',
        'qualify_model','readout','failure_status','Budget','precision_ok','validate_readiness','evaluate','compare',
        'parser','main','QwenBackend','Recorder','call_plan','prefix_replay'):
        assert dump(old[name])==dump(new[name]),name


def test_current_source_inventory_is_complete(tmp_path):
    s=runner.source_metadata(actual_args(tmp_path/'unused'));names={r['path'] for r in s['files']}
    required={'tools/run_public_carried72_handoff_phase_panel.py','tests/test_public_carried72_handoff_phase_panel.py',
        'tools/audit_public_carried72_handoff_phase_panel.py','tests/test_public_carried72_handoff_phase_panel_audit.py',
        'docs/public-carried72-handoff-phase-panel.md','tools/run_public_intermediate_write0_phase_panel.py',
        'tools/run_public_short_write0_phase_panel.py','tools/benchmark_qwen_batch.py','tools/validate_qwen_map.py','pyproject.toml','uv.lock'}
    assert required<=names and len(s['source_commit'])==40 and isinstance(s['working_tree_status'],list)
    for b in s['files']:assert b['sha256']==runner.instrument.sha256(runner.ROOT/b['path'])


def test_maximum_actual_serialization(tmp_path):
    requested=os.environ.get('SHRIMP_CARRIED72_PANEL_FULL_FIXTURE')
    if not requested:pytest.skip('Full-size retained fixture runs once by preparation harness')
    path=Path(requested);assert path.is_relative_to('/data2/shrimp-memory-data/preparations/public-carried72-handoff-phase-panel')
    payload,_=runner.qualify_inputs(actual_args(path));r=run_fixture(path,payload=payload)
    assert r['status']=='completed' and r['outcome']=='panel_qualified' and r['prefix_replay_passed']
    assert r['model_calls']==1788 and r['resource_limits_passed'] and len(list(path.glob('call-*.npz')))==1788
    prob=0;cache_bytes=0;cache_arrays=0;maxcall=0
    for p in sorted(path.glob('call-*.npz')):
        maxcall=max(maxcall,p.stat().st_size)
        with np.load(p,allow_pickle=False) as z:
            assert z['probabilities'].dtype==np.float64 and z['probabilities'].shape[1]==151936;prob+=z['probabilities'].nbytes
    for p in path.glob('group-*-*prefix-cache.npz'):
        with np.load(p,allow_pickle=False) as z:cache_bytes+=sum(z[k].nbytes for k in z.files);cache_arrays+=len(z.files)
    assert prob==3540*151936*8==4302827520 and cache_arrays==24*49
    assert len(list(path.glob('group-*-*prefix-cache.npz')))==24
    actual=sum(p.stat().st_size for p in path.iterdir() if p.is_file())
    assert r['terminal_resources']['final_raw_bytes']==actual and actual+runner.FINAL_RESERVE<runner.RAW_CAP
    receipt=dict(synthetic=True,model_calls=1788,files=len(list(path.iterdir())),probability_vectors=3540,probability_bytes=prob,
        cache_array_bytes=cache_bytes,cache_arrays=cache_arrays,cache_artifacts=24,maximum_call_file_bytes=maxcall,
        actual_raw_bytes=actual,raw_plus_final_reserve_bytes=actual+runner.FINAL_RESERVE,raw_cap_bytes=runner.RAW_CAP,
        peak_rss_bytes=r['peak_rss_bytes'],elapsed_seconds=r['elapsed_seconds'],threads={k:os.environ[k] for k in runner.THREAD_KEYS},
        measured_scope='Full-size three-case batch2 carried-word serialization with scripted outputs; no model evidence')
    (path.parent/(path.name+'-serialization.json')).write_text(json.dumps(receipt,indent=2)+'\n')


