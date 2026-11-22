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
import run_public_canonical_write0_phase_panel as runner


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


ARCHIVE = '/data2/shrimp-memory-data/'


def pinned_args(output):
    """actual_args(), but skip the test when a pinned historical file is not at its original path."""
    args = actual_args(output)
    missing = [p for p in vars(args).values() if isinstance(p, Path) and str(p).startswith(ARCHIVE) and not p.is_file()]
    if missing:
        pytest.skip(f'pinned historical artifact not available: {missing[0]}')
    return args


def synthetic_payload():
    refs=np.zeros((2,6,896),dtype=np.float64); refs[1,:,0]=10.
    for i in range(6): refs[:,i,i+1]=.1
    states=np.repeat(refs[:,0][None],121,axis=0)
    states[:,:,20]=np.arange(121)[:,None]/1000
    pairs=np.stack([refs[:,p] for p in runner.PHASES])
    short=np.repeat(pairs[:,None],37,axis=1);short[:,:,:,20]+=np.arange(37)[None,:,None]/1000
    return runner.derive_inputs(refs,states,np.zeros((120,2),dtype=np.int64),pairs,short,np.zeros((4,36,2),dtype=np.int64))



def synthetic_source(args):
    return dict(source_commit='synthetic',files=[],working_tree_status=[],synthetic=True)



def synthetic_model(args):
    return dict(synthetic=True,no_model_loaded=True)



class FakeBackend:
    """Scripted outputs with actual production array geometry, never a model."""
    def __init__(self,args,budget,recorder,*,payload=None,bits=None,failure=None,defect=None,history_offset=0.):
        self.budget,self.recorder=budget,recorder
        self.payload=synthetic_payload() if payload is None else payload
        self.bits=[0]*8 if bits is None else bits
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
        if group%2==0:
            if step<=36:
                y=self.payload['short_prefix_states'][group//2,step,lanes].copy();y[:,30]+=self.history_offset
                top=self.payload['short_prefix_token_ids'][group//2,step-1,lanes]
            else:
                y=x+.001;top=np.zeros(len(x),dtype=np.int64)
        else:
            bits=[self.bits[2*(group//2)+lane] for lane in lanes]
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


def test_fixed_1480_call_ledger_controls_and_cli():
    slots=runner.call_plan();assert len(slots)==1480
    assert [sum(s['group_id']==g for s in slots) for g in range(8)]==[125,245]*4
    assert [s['call'] for s in slots if s['category']=='prefix']==[370*i+x for i in range(4) for x in (1,126)]
    assert sum(s['category']=='update' for s in slots)==1440
    assert sum(s['batch_size'] for s in slots)==2936
    for s in slots:
        g=s['group_id'];assert (s['case_index'],s['stage_index'],s['reference_phase'])==(g//2,g%2,2+g//2)
        assert s['controls']==runner.CONTROLS[runner.CONTROL_NAMES[g]]
    d=runner.derive_design()
    assert d['within_case_carry_boundaries']==[[0,1],[2,3],[4,5],[6,7]]
    assert d['between_case_reset_group_ids']==[2,4,6]
    assert d['readout']['requested_bits']==[0]*8
    assert d['readout']['closure_pairs']==[[i,i-6] for i in range(235,241)]
    assert d['prefix_replay']['group_ids']==[0,2,4,6] and d['prefix_replay']['source_group_ids']==[0,4,8,12]
    assert d['prefix_replay']['same_input'] and not d['prefix_replay']['gates_readout']
    assert d['limits']['internal_seconds']==3600 and d['limits']['return_seconds']==7200
    assert d['limits']['raw_bytes']==9*1024**3 and d['limits']['transfer_reservation_bytes']==32*1024**3
    args=actual_args('/tmp/unused');cli=[]
    for k,v in vars(args).items():
        if k!='preflight_only':cli.extend(['--'+k.replace('_','-'),str(v)])
    assert runner.parser().parse_args(cli).short_prefix==args.short_prefix
    for pair in (['--alpha','.1'],['--steps','36'],['--seed','3'],['--reference-phase','2'],['--phases','2,3']):
        with pytest.raises(SystemExit):runner.parser().parse_args(cli+pair)


def test_one_backend_fixed_eight_stage_selection():
    backend=runner.QwenBackend.__new__(runner.QwenBackend);backend.vector=np.linspace(-1,1,896,dtype=np.float64)
    hook=object();backend.loop=SimpleNamespace(handle=hook,cache=object(),steering=None,temperature=None);backend.selected_group=-1
    for g in range(8):
        cache=backend.loop.cache;backend.select_group(g);c=runner.CONTROLS[runner.CONTROL_NAMES[g]]
        assert np.array_equal(backend.loop.steering,c['alpha']*backend.vector.reshape(1,1,-1))
        assert backend.loop.temperature==10.**c['log10_temperature']
        assert backend.loop.handle is hook and backend.loop.cache is cache
    for wrong in (7,8,0):
        with pytest.raises(runner.InstrumentFailure):backend.select_group(wrong)


def test_neutral_six_pairs_and_all_alignment_readout():
    refs=synthetic_payload()['references'];tail=np.concatenate((refs[1,-1:],refs[1],refs[1]));tail[0,50]=1000.
    r=runner.readout(tail,refs);assert r['decoded_bit']==1 and len(r['lag_six_phase_l2_residuals'])==6
    assert all(len(d['by_shift'])==6 for d in r['reference_distances'])
    assert runner.readout(tail,np.stack((refs[1],refs[1])))['decoded_bit'] is None
    assert runner.readout(np.repeat(refs[1,0][None],13,axis=0),refs)['decoded_bit'] is None
    assert runner.case_outcome([0,0])=='case_qualified' and runner.case_outcome([0,1])=='case_wrong'
    assert runner.case_outcome([None,1])=='case_unresolved'
    assert runner.panel_outcome([0]*8)=='panel_qualified' and runner.panel_outcome([1]+[0]*7)=='panel_wrong'
    assert runner.panel_outcome([1,None]+[0]*6)=='panel_unresolved'
    for bad in ([0]*7,[True]+[0]*7,['0']+[0]*7):
        with pytest.raises(runner.InstrumentFailure):runner.panel_outcome(bad)


def test_complete_mixed_case_plan_preserves_carry_resets_and_diagnostic_failures(tmp_path,monkeypatch):
    bits=[0,0,1,0,None,0,0,0];constructed=[];original=FakeBackend.__init__
    def counted(self,*a,**kw):constructed.append(self);original(self,*a,**kw)
    monkeypatch.setattr(FakeBackend,'__init__',counted)
    path=tmp_path/'run';r=run_fixture(path,bits=bits,history_offset=.125,defect=('token',6))
    assert len(constructed)==1 and r['model_calls']==1480 and r['status']=='completed'
    assert r['outcome']=='panel_unresolved' and r['resource_limits_passed'] and r['prefix_replay_passed'] is False
    assert [c['outcome'] for c in r['cases']]==['case_qualified','case_wrong','case_unresolved','case_qualified']
    assert [x['readout']['decoded_bit'] for x in r['rows']]==bits
    assert [g['model_calls'] for g in r['groups']]==[125,245]*4
    payload=synthetic_payload();owned=[]
    for g in range(8):
        case,stage=divmod(g,2)
        if stage==0:carried=payload['initial_states'][case]
        with np.load(path/f'group-{g:02d}-states.npz') as z:
            assert z['states'].shape==((121,241)[stage],2,896) and runner.same_bytes(z['states'][0],carried)
            assert all(z[k].dtype==np.int64 and z[k].shape==() and int(z[k])==v for k,v in
                dict(case_index=case,stage_index=stage,reference_phase=case+2).items())
            carried=z['states'][-1].copy()
        marker=json.loads((path/f'group-{g:02d}.json').read_text());assert marker==r['groups'][g] and marker['closed']
        summary=json.loads((path/f'group-{g:02d}-summary.json').read_text());assert summary['predecessor_group_id']==(g-1 if stage else None)
        for b in marker['artifacts']:
            assert runner.binding(path/b['path'],flat=True)==b;owned.append(b['path'])
    assert len(owned)==len(set(owned))
    for i,row in enumerate(r['rows']):
        case,lane=divmod(i,2)
        assert (row['group_id'],row['case_index'],row['lane'],row['requested_bit'],row['reference_phase'])==(2*case+1,case,lane,0,case+2)
    for case,c in enumerate(r['cases']):
        d=c['prefix_replay'];assert d==r['prefix_replay'][case] and d['same_input'] and not d['gates_readout']
        assert c['group_ids']==c['row_ids']==[2*case,2*case+1]
        assert d['coordinate_tolerance']==1e-10 and d['maximum_coordinate_gap']>=.125 and not d['passed']
        assert d['reference_phase']==case+2 and d['state_count']==37 and d['update_count']==36
        if case==0:assert [l['token_mismatch_count'] for l in d['lanes']]==[1,1]
        with np.load(path/f'case-{case:02d}-prefix-replay.npz') as z:
            assert z['coordinate_absolute_gaps'].shape==(37,2,896) and z['token_equal'].shape==(36,2)
            assert z['state_bytes_equal'][0].all()
            assert np.array_equal(z['per_state_coordinate_maxima'],z['coordinate_absolute_gaps'].max(axis=2))
            assert int(z['reference_phase'])==case+2
    events=[json.loads(x) for x in (path/'events.jsonl').read_text().splitlines()]
    assert sum(e['event']=='model_call_started' for e in events)==1480
    assert sum(e['event']=='model_call_completed' for e in events)==1480
    assert [e['control_name'] for e in events if e['event']=='control_selected']==['E0','H']*4
    assert [e['reset_between_cases'] for e in events if e['event']=='case_started']==[False,True,True,True]
    assert {p.name for p in path.glob('*-tail.npz')}=={f'group-{g:02d}-tail.npz' for g in (1,3,5,7)}
    for index in (999,1000,1480):
        s=r['calls'][index-1];assert (path/f'call-{index:03d}-{s["label"]}.npz').is_file()
    manifest=json.loads((path/'manifest.json').read_text())
    assert {b['path'] for b in manifest['files']}=={p.name for p in path.iterdir()}-{'manifest.json'}
    assert r['terminal_resources']['final_raw_bytes']==sum(p.stat().st_size for p in path.iterdir())


@pytest.mark.parametrize('exception,status',[(RuntimeError,'execution_failed'),(OSError,'execution_failed'),
    (runner.ResourceLimit,'incomplete'),(TimeoutError,'incomplete'),(MemoryError,'incomplete'),(ValueError,'instrument_failed')])
def test_failed_attempt_keeps_input_and_unrun_groups(tmp_path,exception,status):
    p=tmp_path/'run';r=run_fixture(p,failure=(6,exception))
    assert r['status']==status and r['model_calls']==6 and r['prefix_replay']==[None]*4
    assert [g['model_calls'] for g in r['groups']]==[6]+[0]*7
    assert [g['status'] for g in r['groups']]==[status]+['not_run']*7
    assert [c['status'] for c in r['cases']]==[status]+['not_run']*3
    assert len(r['rows'])==8 and len(r['calls'])==1480
    j=json.loads((p/'call-006-g00-update-001.json').read_text());assert j['status']=='failed' and j['exception_category']==status
    with np.load(p/'call-006-g00-update-001.npz') as z:assert 'input' in z and 'output' not in z


@pytest.mark.parametrize('mode',['precision','nonfinite','shape','input','cache','expanded','probabilities'])
def test_returned_defect_preserves_raw_and_admission(tmp_path,mode):
    p=tmp_path/'run';r=run_fixture(p,defect=(mode,6));assert r['status']=='instrument_failed' and r['model_calls']==6
    with np.load(p/'call-006-g00-update-001.npz') as z:assert 'probabilities' in z and 'output' in z
    j=json.loads((p/'group-00-summary.json').read_text());assert j['completed_updates']==0 and j['returned_updates']==(0 if mode=='shape' else 1)


def test_preinvocation_cache_defect_is_not_an_attempt(tmp_path):
    p=tmp_path/'run';r=run_fixture(p,defect=('precall_cache',1));assert r['status']=='instrument_failed' and r['model_calls']==1
    j=json.loads((p/'blocked-call-002-g00-compare-cached.json').read_text());assert j['status']=='blocked_before_invocation'
    with np.load(p/'blocked-call-002-g00-compare-cached.npz') as z:assert 'cache_before_layer_00_key' in z
    assert r['calls'][1]['blocked_evidence']=='blocked-call-002-g00-compare-cached.json'


def test_next_case_prefix_failure_preserves_prior_case(tmp_path):
    p=tmp_path/'run';r=run_fixture(p,failure=(371,OSError));assert r['status']=='execution_failed' and r['model_calls']==371
    assert [c['status'] for c in r['cases']]==['completed','execution_failed','not_run','not_run']
    assert r['cases'][0]['outcome']=='case_qualified' and [x['status'] for x in r['rows'][:2]]==['completed']*2
    with np.load(p/'group-02-states.npz') as z:assert runner.same_bytes(z['states'][0],synthetic_payload()['initial_states'][1])
    for g in r['groups'][:2]:
        for b in g['artifacts']:assert runner.binding(p/b['path'],flat=True)==b


def test_late_failure_keeps_all_closed_evidence(tmp_path):
    p=tmp_path/'run';r=run_fixture(p,failure=(1479,OSError));assert r['status']=='execution_failed' and r['model_calls']==1479
    assert [g['model_calls'] for g in r['groups']]==[125,245]*3+[125,244]
    assert [c['status'] for c in r['cases']]==['completed']*3+['execution_failed']
    assert [x['status'] for x in r['rows']]==['completed']*6+['execution_failed']*2
    assert r['prefix_replay_passed'] is None
    for g in r['groups'][:7]:
        assert g['status']=='completed'
        for b in g['artifacts']:assert runner.binding(p/b['path'],flat=True)==b
    with np.load(p/'group-07-states.npz') as z:assert z['states'].shape==(239,2,896)
    assert (p/'call-1479-g07-update-239.npz').is_file()


def test_comparison_failure_prevents_updates(tmp_path):
    p=tmp_path/'run';r=run_fixture(p,defect=('comparison',3));assert r['model_calls']==5 and r['status']=='instrument_failed'
    with np.load(p/'group-00-states.npz') as z:assert z['states'].shape==(1,2,896)


def test_prefix_failure_is_retained(tmp_path):
    p=tmp_path/'run';r=run_fixture(p,failure=(1,OSError));assert r['status']=='execution_failed' and r['model_calls']==1
    with np.load(p/'call-001-g00-prefix.npz') as z:assert 'input' in z and 'probabilities' not in z


def test_preflight_has_no_backend_and_closes_every_group(tmp_path):
    r=run_fixture(tmp_path/'run',preflight=True)
    assert r['status']=='preflight_passed' and r['model_calls']==0 and not r['model_loaded'] and r['resource_limits_passed']
    assert len(r['groups'])==8 and all(g['closed'] and g['status']=='not_run' for g in r['groups'])
    assert all(c['status']=='not_run' and c['prefix_replay'] is None for c in r['cases'])


def test_raw_cap_and_startup_thread_failure_preserved(tmp_path,monkeypatch):
    normal=runner.RAW_CAP;monkeypatch.setattr(runner,'RAW_CAP',runner.FINAL_RESERVE+1024*1024)
    r=run_fixture(tmp_path/'cap');assert r['status']=='incomplete' and r['model_calls']==0
    assert (tmp_path/'cap/result.json').is_file();monkeypatch.setattr(runner,'RAW_CAP',normal)
    class Bad(FakeBackend):
        def runtime_info(self):
            d=super().runtime_info();d['startup_thread_env']['OPENBLAS_NUM_THREADS']='2';return d
    r=runner.run(actual_args(tmp_path/'threads'),backend_factory=Bad,input_qualifier=lambda a:(synthetic_payload(),{}),
        model_qualifier=synthetic_model,source_factory=synthetic_source)
    assert r['status']=='instrument_failed' and r['model_calls']==0


def test_actual_ten_input_pins_and_literal_packet(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,'VOCAB',151936);p,proof=runner.qualify_inputs(pinned_args(tmp_path/'unused'))
    assert len(proof['bindings'])==10 and proof['same_phase_short_prefix_input']
    assert proof['historical_reference_phase']==0 and not proof['same_historical_input']
    for case,phase in enumerate((2,3,4,5)):
        assert runner.same_bytes(p['initial_states'][case],p['references'][:,phase])
        assert runner.same_bytes(p['short_prefix_states'][case,0],p['initial_states'][case])
    assert p['short_prefix_states'].shape==(4,37,2,896) and p['short_prefix_token_ids'].shape==(4,36,2)


@pytest.mark.parametrize('key',['short_prefix','short_prefix_provenance','prospective_panel','trajectory_h0'])
def test_changed_input_pin_stops_before_model(tmp_path,key):
    a=actual_args(tmp_path/'run');p=tmp_path/'changed';p.write_bytes(b'changed');setattr(a,key,p)
    r=runner.run(a,backend_factory=lambda *a:pytest.fail('backend'),source_factory=synthetic_source,model_qualifier=synthetic_model)
    assert r['status']=='instrument_failed' and r['model_calls']==0 and not r['model_loaded']
    assert all(g['closed'] and g['status']=='not_run' for g in r['groups'])


@pytest.mark.parametrize('defect',['phase','index','initial','tokens','shape','extra','provenance','conflict'])
def test_rehashed_prefix_packet_contract_rejects_semantic_tampering(tmp_path,defect,monkeypatch):
    monkeypatch.setattr(runner,'VOCAB',151936);a=pinned_args(tmp_path/'unused')
    payload,_=runner.qualify_inputs(a)
    with np.load(a.short_prefix) as z:arrays={k:z[k].copy() for k in z.files}
    proof=json.loads(a.short_prefix_provenance.read_text())
    if defect=='phase':arrays['reference_phases'][0]=1
    if defect=='index':arrays['update_indices'][0]=0
    if defect=='initial':arrays['states'][2,0,1,0]+=1
    if defect=='tokens':arrays['token_ids'][0,0,0]=151936
    if defect=='shape':arrays['states']=arrays['states'][:3]
    if defect=='extra':arrays['hidden_selection']=np.array(1,dtype=np.int64)
    if defect=='provenance':proof['accepted_audit_synthetic']=True
    if defect=='conflict':
        b=proof['source_prefix_files'][0].copy();b['sha256']='0'*64;proof['source_prefix_files'].append(b)
    p=tmp_path/'packet.npz';np.savez(p,**arrays)
    proof['packet']=runner.binding(p)
    for k,v in arrays.items():
        proof['arrays'][k]=dict(dtype=str(v.dtype),shape=list(v.shape),array_sha256=runner.array_hash(v),
            raw_c_sha256=runner.hashlib.sha256(v.tobytes(order='C')).hexdigest())
    q=tmp_path/'proof.json';q.write_text(json.dumps(proof))
    with pytest.raises(runner.InstrumentFailure):runner.load_short_prefix(p,q,payload['references'],payload['initial_states'])


def test_prefix_diagnostic_tolerance_and_exact_tokens_are_independent(tmp_path):
    payload=synthetic_payload();states=payload['short_prefix_states'][0].copy();tokens=payload['short_prefix_token_ids'][0].copy()
    recorder=runner.Recorder(tmp_path/'evidence')
    try:
        d=runner.prefix_replay(states,tokens,payload,recorder,0);assert d['passed'] and d['state_bytes_equal']
        states[1,0,40]=5e-11
        d=runner.prefix_replay(states,tokens,payload,recorder,1)
        assert not d['coordinate_replay_passed']  # Wrong case is never silently phase-aligned.
    finally:recorder.stream.close()
    recorder=runner.Recorder(tmp_path/'token')
    try:
        tokens[0,0]=1;d=runner.prefix_replay(payload['short_prefix_states'][0],tokens,payload,recorder,0)
        assert d['coordinate_replay_passed'] and not d['token_replay_passed'] and not d['passed']
    finally:recorder.stream.close()
    recorder=runner.Recorder(tmp_path/'small')
    try:
        d=runner.prefix_replay(states,payload['short_prefix_token_ids'][0],payload,recorder,0)
        assert d['passed'] and not d['state_bytes_equal'] and not d['gates_readout']
    finally:recorder.stream.close()


def test_proven_numerical_and_storage_primitives_unchanged():
    def nodes(p):return {n.name:n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    old=nodes(runner.ROOT/'tools/run_public_short_write0_phase_panel.py');new=nodes(Path(runner.__file__))
    dump=lambda n:ast.dump(n,include_attributes=False)
    for name in ('InstrumentFailure','ResourceLimit','require','finite','same_bytes','array_hash','cache_hash','binding',
        'checked_norm','binding_index','require_historical_binding','qualify_model','readout','failure_status','Budget',
        'precision_ok','validate_readiness','evaluate','compare','parser','main','QwenBackend','Recorder','call_plan'):
        assert dump(old[name])==dump(new[name]),name


def test_source_inventory_includes_current_checker_protocol_and_held_sources(tmp_path):
    s=runner.source_metadata(actual_args(tmp_path/'unused'));names={r['path'] for r in s['files']}
    required={'tools/run_public_canonical_write0_phase_panel.py','tests/test_public_canonical_write0_phase_panel.py',
        'tools/audit_public_canonical_write0_phase_panel.py','tests/test_audit_public_canonical_write0_phase_panel.py',
        'docs/public-canonical-write0-phase-panel.md','tools/run_public_short_write0_phase_panel.py',
        'tools/benchmark_qwen_batch.py','tools/validate_qwen_map.py','pyproject.toml','uv.lock'}
    assert required<=names and len(s['source_commit'])==40 and isinstance(s['working_tree_status'],list)
    for b in s['files']:assert b['sha256']==runner.instrument.sha256(runner.ROOT/b['path'])


def test_maximum_actual_serialization(tmp_path):
    requested=os.environ.get('SHRIMP_CANONICAL_PANEL_FULL_FIXTURE')
    if not requested:pytest.skip('Full-size retained fixture runs once by preparation harness')
    path=Path(requested);assert path.is_relative_to('/data2/shrimp-memory-data/preparations/public-canonical-write0-phase-panel')
    payload,_=runner.qualify_inputs(actual_args(path));r=run_fixture(path,payload=payload)
    assert r['status']=='completed' and r['outcome']=='panel_qualified' and r['prefix_replay_passed']
    assert r['model_calls']==1480 and r['resource_limits_passed'] and len(list(path.glob('call-*.npz')))==1480
    prob=0;cache_bytes=0;cache_arrays=0;maxcall=0
    for p in sorted(path.glob('call-*.npz')):
        maxcall=max(maxcall,p.stat().st_size)
        with np.load(p,allow_pickle=False) as z:
            assert z['probabilities'].dtype==np.float64 and z['probabilities'].shape[1]==151936;prob+=z['probabilities'].nbytes
    for p in path.glob('group-*-*prefix-cache.npz'):
        with np.load(p,allow_pickle=False) as z:cache_bytes+=sum(z[k].nbytes for k in z.files);cache_arrays+=len(z.files)
    assert prob==2936*151936*8==3568672768 and cache_arrays==16*49
    assert len(list(path.glob('group-*-*prefix-cache.npz')))==16
    actual=sum(p.stat().st_size for p in path.iterdir() if p.is_file())
    assert r['terminal_resources']['final_raw_bytes']==actual and actual+runner.FINAL_RESERVE<runner.RAW_CAP
    receipt=dict(synthetic=True,model_calls=1480,files=len(list(path.iterdir())),probability_vectors=2936,probability_bytes=prob,
        cache_array_bytes=cache_bytes,cache_arrays=cache_arrays,cache_artifacts=16,maximum_call_file_bytes=maxcall,
        actual_raw_bytes=actual,raw_plus_final_reserve_bytes=actual+runner.FINAL_RESERVE,raw_cap_bytes=runner.RAW_CAP,
        peak_rss_bytes=r['peak_rss_bytes'],elapsed_seconds=r['elapsed_seconds'],threads={k:os.environ[k] for k in runner.THREAD_KEYS},
        measured_scope='Full-size four-case batch2 serialization with scripted outputs; no model evidence')
    (path.parent/(path.name+'-serialization.json')).write_text(json.dumps(receipt,indent=2)+'\n')


