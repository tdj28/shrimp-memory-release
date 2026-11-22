"""Independent saved-array tests for the intermediate72 WRITE0 phase panel."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('intermediate_write0_phase_panel_auditor',ROOT/'tools/audit_public_intermediate_write0_phase_panel.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def references():
    refs=np.zeros((2,6,896),dtype=np.float64)
    angle=np.arange(6)*2*np.pi/6
    refs[:,:,0]=.2*np.cos(angle)
    refs[:,:,1]=.2*np.sin(angle)
    refs[0,:,2]=-.01
    refs[1,:,2]=.01
    return refs


def history():
    refs=references()
    states=np.stack([refs[:,i%6] for i in range(121)])
    states[:,:,8]+=np.arange(121)[:,None]/1000
    return refs,states,np.arange(240,dtype=np.int64).reshape(120,2)


def short_history(refs=None):
    refs=references() if refs is None else refs
    short=np.stack([np.repeat(refs[:,phase][None],37,axis=0) for phase in (2,3,4,5)])
    short[:,:,:,8]+=np.arange(37)[None,:,None]/1000
    return short,np.zeros((4,36,2),dtype=np.int64)

def packet_arrays():
    states,tokens=short_history()
    return dict(states=states,token_ids=tokens,reference_phases=np.asarray([2,3,4,5],dtype=np.int64),
        reference_indices=np.arange(2,dtype=np.int64),state_indices=np.arange(37,dtype=np.int64),
        update_indices=np.arange(1,37,dtype=np.int64))


def payload():
    refs,states,tokens=history()
    return audit.extract_chain_inputs(refs,states,tokens,*[x[list(audit.PACKET_INDICES)] for x in short_history(refs)])


def test_literal_inputs_include_all_fixed_prefixes_without_mutation():
    p=payload()
    assert p['initial_states'].shape==(3,2,896)
    assert p['short_prefix_states'].shape==(3,37,2,896)
    assert p['short_prefix_token_ids'].shape==(3,36,2)
    for case,phase in enumerate((3,4,5)):
        assert p['initial_states'][case].tobytes()==p['references'][:,phase].tobytes()
        assert p['short_prefix_states'][case,0].tobytes()==p['initial_states'][case].tobytes()
    assert p['packet_indices'].tolist()==[1,2,3]
    assert p['packet_reference_phases'].tolist()==[2,3,4,5]


@pytest.mark.parametrize('packet_index',range(4))
def test_all_source_packet_entries_are_checked_before_fixed_subset(packet_index):
    arrays=prospective_arrays()
    arrays['initial_states'][packet_index,0,8]=.01
    with pytest.raises(audit.AuditError):
        audit.qualify_prospective_panel(arrays,references())
    arrays=packet_arrays()
    arrays['states'][packet_index,0,0,8]=.01
    with pytest.raises(audit.AuditError):
        audit.qualify_short_prefix(arrays,references())


@pytest.mark.parametrize('mode',['phase','lane','states','tokens','token_range','index','extra','signed_zero'])
def test_short_packet_rejects_neighbor_phases_and_corrupt_arrays(mode):
    arrays=packet_arrays()
    if mode=='phase':arrays['reference_phases'][0]=1
    elif mode=='lane':arrays['states']=arrays['states'][:,:,::-1].copy()
    elif mode=='states':arrays['states']=arrays['states'].astype(np.float32)
    elif mode=='tokens':arrays['token_ids']=arrays['token_ids'].astype(np.int32)
    elif mode=='token_range':arrays['token_ids'][0,0,0]=audit.VOCAB
    elif mode=='index':arrays['update_indices'][0]=0
    elif mode=='extra':arrays['best_phase']=np.asarray(2)
    else:arrays['states'][0,0,0,20]=-0.
    with pytest.raises(audit.AuditError):audit.qualify_short_prefix(arrays,references())


@pytest.mark.parametrize('bits,expected', [([0,0],'case_qualified'),([1,0],'case_wrong'),([0,1],'case_wrong'),
    ([1,1],'case_wrong'),([None,0],'case_unresolved'),([1,None],'case_unresolved')])
def test_case_classification_is_neutral_and_retains_all_failures(bits,expected):
    assert audit.case_outcome(bits)==expected


@pytest.mark.parametrize('case',range(3))
def test_same_phase_replay_requires_both_coordinate_and_exact_token_gates(case):
    p=payload();states=p['short_prefix_states'][case].copy();tokens=p['short_prefix_token_ids'][case].copy()
    _,good=audit.replay_diagnostic(states,tokens,p,case)
    assert good['passed'] and good['same_input'] and not good['gates_readout']
    states[17,0,20]=5e-11;states[36,1,20]=-0.;tokens[8,1]=1
    arrays,wrong=audit.replay_diagnostic(states,tokens,p,case)
    assert wrong['coordinate_replay_passed'] and not wrong['token_replay_passed'] and not wrong['passed']
    assert not arrays['state_bytes_equal'][36,1] and arrays['per_state_coordinate_maxima'][36,1]==0
    assert wrong['lanes'][1]['token_mismatch_count']==1
    states[17,0,20]=2e-10;tokens[8,1]=0
    _,wrong=audit.replay_diagnostic(states,tokens,p,case)
    assert not wrong['coordinate_replay_passed'] and wrong['token_replay_passed'] and not wrong['passed']


@pytest.mark.parametrize('bad',['nan','overflow','wrong_shape','bad_tokens','different_initial','bad_case'])
def test_prefix_replay_invalid_values_are_not_scientific_nulls(bad):
    p=payload();states=p['short_prefix_states'][0].copy();tokens=p['short_prefix_token_ids'][0].copy();case=0
    if bad=='nan':states[2,1,0]=np.nan
    elif bad=='overflow':states[2,1,0]=1e308
    elif bad=='wrong_shape':states=states[:-1]
    elif bad=='bad_tokens':tokens=tokens.astype(np.float64)
    elif bad=='different_initial':states[0]=p['references'][:,0]
    else:case=True
    with pytest.raises(audit.AuditError):audit.replay_diagnostic(states,tokens,p,case)

def tail(bit=0,shift=0):
    refs=references()
    return np.stack([refs[bit,(i+shift)%6] for i in range(13)])




@pytest.mark.parametrize('bad',['phase0','phase1','nonfinite','float32','shape','token_dtype','token_range'])
def test_invalid_literal_inputs_fail_closed(bad):
    refs,states,tokens=history()
    if bad=='phase0':states[0,0]=refs[0,1]
    if bad=='phase1':states[0,1]=refs[1,1]
    if bad=='nonfinite':states[12,1,8]=np.nan
    if bad=='float32':states=states.astype(np.float32)
    if bad=='shape':states=states[:-1]
    if bad=='token_dtype':tokens=tokens.astype(np.float64)
    if bad=='token_range':tokens[0,0]=audit.VOCAB
    with pytest.raises(audit.AuditError):audit.extract_chain_inputs(refs,states,tokens,*[x[list(audit.PACKET_INDICES)] for x in short_history(refs)])


def prospective_arrays():
    return dict(initial_states=np.stack([references()[:,i] for i in (2,3,4,5)]),reference_indices=np.array([0,1],dtype=np.int64),
                reference_phases=np.asarray([2,3,4,5],dtype=np.int64))


def test_prospective_panel_is_exact_immutable_phases():
    arrays=prospective_arrays();before={k:v.tobytes() for k,v in arrays.items()}
    pair=audit.qualify_prospective_panel(arrays,references());pair[0,0]+=1
    assert before=={k:v.tobytes() for k,v in arrays.items()}


@pytest.mark.parametrize('bad',['phase0','phase2','phase_dtype','phase_shape','swapped','indices_dtype','float32','nan','extra','signed_zero'])
def test_prospective_pair_rejects_reselection_and_metadata_corruption(bad):
    arrays=prospective_arrays()
    if bad=='phase0':arrays['initial_states']=references()[:,0].copy()
    elif bad=='phase2':arrays['reference_phases'][0]=1
    elif bad=='phase_dtype':arrays['reference_phases']=np.asarray([2,3,4,5],dtype=np.float64)
    elif bad=='phase_shape':arrays['reference_phases']=np.asarray(2,dtype=np.int64)
    elif bad=='swapped':arrays['initial_states']=arrays['initial_states'][::-1].copy()
    elif bad=='indices_dtype':arrays['reference_indices']=arrays['reference_indices'].astype(np.int32)
    elif bad=='float32':arrays['initial_states']=arrays['initial_states'].astype(np.float32)
    elif bad=='nan':arrays['initial_states'][0,1,4]=np.nan
    elif bad=='extra':arrays['selected_phase']=np.asarray(1)
    elif bad=='signed_zero':arrays['initial_states'][1,0,8]=-0.
    with pytest.raises(audit.AuditError):audit.qualify_prospective_panel(arrays,references())


@pytest.mark.parametrize('bit',[0,1])
@pytest.mark.parametrize('shift',range(6))
def test_every_phase_decodes_neutrally_with_all_reference_shifts(bit,shift):
    result=audit.readout(tail(bit,shift),references())
    assert result['decoded_bit']==bit
    assert result['cycle_gate_passed']
    assert result['reference_matches']==[bit]
    assert len(result['lag_six_phase_l2_residuals'])==6
    assert len(result['proper_divisors'])==3
    assert all(len(row['by_shift'])==6 for row in result['reference_distances'])
    assert result['reference_distances'][bit]['distance']==0


def test_retained_thirteenth_state_is_context_not_a_seventh_gate():
    states=tail();states[0]+=100
    result=audit.readout(states,references())
    assert result['decoded_bit']==0
    assert result['maximum_l2_residual']==0


def test_all_six_pairs_gate_including_earliest_and_latest():
    for index in (1,6,7,12):
        states=tail();states[index,8]+=.001
        result=audit.readout(states,references())
        assert not result['cycle_gate_passed'] and result['decoded_bit'] is None


def test_ambiguous_reference_match_is_not_a_label():
    refs=references();refs[1]=refs[0]
    result=audit.readout(tail(),refs)
    assert result['reference_matches']==[0,1] and result['decoded_bit'] is None


def test_period_divisor_failure_keeps_distances_without_qualification():
    refs=references();states=np.repeat(refs[0,:1],13,axis=0)
    result=audit.readout(states,refs)
    assert result['maximum_l2_residual']==0
    assert not result['cycle_gate_passed'] and result['decoded_bit'] is None
    assert all(not row['excluded'] for row in result['proper_divisors'])


@pytest.mark.parametrize('bad',['float32','nan','shape','overflow'])
def test_invalid_readout_does_not_become_scientific_null(bad):
    states=tail()
    if bad=='float32':states=states.astype(np.float32)
    if bad=='nan':states[3,4]=np.nan
    if bad=='shape':states=states[1:]
    if bad=='overflow':states[3,4]=1e308
    with pytest.raises(audit.AuditError):audit.readout(states,references())








@pytest.mark.parametrize('text',['{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}'])
def test_json_rejects_duplicate_or_nonfinite_values(text):
    with pytest.raises(audit.AuditError):audit.strict_json(text)


def test_array_hash_includes_dtype_and_shape():
    x=np.zeros((2,3),dtype=np.float64)
    assert audit.array_hash(x)!=audit.array_hash(x.reshape(6))
    assert audit.array_hash(x)!=audit.array_hash(x.view(np.int64))


@pytest.fixture(scope='module')
def producer_fixture():
    """Use actual producer files, but never its classification in the checker."""
    spec=importlib.util.spec_from_file_location('intermediate_write0_phase_panel_fixture',ROOT/'tests/test_public_intermediate_write0_phase_panel.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    old_vocab,old_shape=module.runner.VOCAB,module.runner.batch.CACHE_SHAPE
    module.runner.VOCAB=7;module.runner.batch.CACHE_SHAPE=(1,2,37,1)
    yield module
    module.runner.VOCAB=old_vocab;module.runner.batch.CACHE_SHAPE=old_shape


@pytest.fixture(scope='module')
def completed(tmp_path_factory,producer_fixture):
    prior=os.environ.get('INTERMEDIATE_PANEL_AUDITOR_BASELINE')
    if prior:
        directory=Path(prior).resolve()
        assert load_json(directory/'result.json')['model_calls']==966
        return directory
    directory=tmp_path_factory.mktemp('completed')/'run'
    producer_fixture.run_fixture(directory,bits=[0]*6)
    return directory


def write_json(path,value):
    if path.exists():path.unlink()  # Never mutate another fixture through a hard link.
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def bind(path):
    return dict(path=path.name,bytes=path.stat().st_size,sha256=audit.digest(path.read_bytes()))


def load_json(path):
    return json.loads(path.read_text())


def replace_npz(path,change):
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    change(arrays);path.unlink();np.savez(path,**arrays)


def reseal(path,*,sync_diagnostic=True):
    """Refresh all ordinary hashes after tampering, leaving scientific claims alone.

    This makes semantic tests reach independent arithmetic/decision checks rather
    than stop at an intentionally stale byte hash.
    """
    result=load_json(path/'result.json')
    for row in result['calls'][:result['model_calls']]:
        stem=f'call-{row["call"]:03d}-{row["label"]}'
        saved=load_json(path/(stem+'.json'));npz=path/(stem+'.npz')
        saved['artifact']=bind(npz)
        with np.load(npz,allow_pickle=False) as z:saved['array_sha256']={k:audit.array_hash(z[k]) for k in z.files}
        write_json(path/(stem+'.json'),saved);row.clear();row.update(saved)
    if (path/'literal-inputs.npz').exists():
        preflight=load_json(path/'input-preflight.json');preflight['literal_inputs']=bind(path/'literal-inputs.npz')
        with np.load(path/'literal-inputs.npz',allow_pickle=False) as z:preflight['literal_array_sha256']={k:audit.array_hash(z[k]) for k in z.files}
        write_json(path/'input-preflight.json',preflight)
        result['literal_inputs']=preflight['literal_inputs'];result['literal_array_sha256']=preflight['literal_array_sha256']
    events=[load for line in (path/'events.jsonl').read_text().splitlines() if (load:=json.loads(line))]
    for event in events:
        if event['event']=='model_call_completed':event['artifact']=result['calls'][event['call']-1]['artifact']
    (path/'events.jsonl').unlink();(path/'events.jsonl').write_text(''.join(json.dumps(e,allow_nan=False)+'\n' for e in events))
    for gid in range(6):
        summary_path=path/f'group-{gid:02d}-summary.json';summary=load_json(summary_path)
        summary['calls']=[r for r in result['calls'][:result['model_calls']] if r['group_id']==gid]
        for key in ('trajectory','tail'):
            if summary[key] is not None:summary[key]=bind(path/summary[key]['path'])
        if summary['prefix_replay'] is not None:
            summary['prefix_replay']['artifact']=bind(path/summary['prefix_replay']['artifact']['path'])
            if gid%2==0 and sync_diagnostic:
                result['prefix_replay'][gid//2]=summary['prefix_replay']
                result['cases'][gid//2]['prefix_replay']=summary['prefix_replay']
        write_json(summary_path,summary)
        marker_path=path/f'group-{gid:02d}.json';marker=load_json(marker_path)
        marker['artifacts']=[bind(path/r['path']) for r in marker['artifacts']]
        write_json(marker_path,marker);result['groups'][gid]=marker
    others=[p for p in path.iterdir() if p.is_file() and p.name not in ('result.json','manifest.json')]
    result['raw_bytes_before_result_manifest']=sum(p.stat().st_size for p in others)
    manifest=dict(profile=audit.PROFILE,files=[])
    for _ in range(8):
        write_json(path/'result.json',result)
        manifest['files']=sorted([bind(p) for p in others]+[bind(path/'result.json')],key=lambda x:x['path'])
        write_json(path/'manifest.json',manifest)
        size=sum(p.stat().st_size for p in path.iterdir() if p.is_file())
        if result['terminal_resources']['final_raw_bytes']==size:return
        result['terminal_resources']['final_raw_bytes']=size
    raise AssertionError('test reseal size did not settle')


def clone(source,destination):
    shutil.copytree(source,destination,copy_function=os.link)
    return destination


def test_complete_actual_file_contract_and_every_row(completed):
    report=audit.audit(completed,synthetic=True)
    assert report['status']=='audit_passed' and report['model_calls']==966
    assert report['synthetic'] and report['outcome']=='panel_qualified'
    assert [r['readout']['decoded_bit'] for r in report['rows']]==[0]*6
    assert [r['model_calls'] for r in report['groups']]==[77,245]*3
    assert len(report['bindings'])==len(list(completed.iterdir()))
    assert [x['attempted_calls'] for x in report['call_costs'].values()]==[6,24,216,720]
    assert report['prefix_replay'][0]['same_input'] is True
    assert report['prefix_replay_passed'] is True
    assert [row['initial_reference_phase'] for row in report['rows']]==[phase for phase in (3,4,5) for _ in range(2)]
    assert [r['write_outcome'] for r in report['rows']]==['correct']*6
    for previous,current in zip(report['groups'],report['groups'][1:]):
        if current['stage_index']:assert previous['terminal_state_sha256']==current['initial_state_sha256']
        else:assert current['predecessor_group_id'] is None


@pytest.mark.parametrize('call,error,category',[(1,OSError,'execution_failed'),(6,OSError,'execution_failed'),
    (78,TimeoutError,'incomplete'),(323,MemoryError,'incomplete')])

def test_partial_failure_preserves_every_attempt_and_unrun_row(tmp_path,producer_fixture,call,error,category):
    directory=tmp_path/'run';producer_fixture.run_fixture(directory,failure=(call,error))
    report=audit.audit(directory,synthetic=True)
    assert report['model_calls']==call and report['scientific_status']==category
    assert len(report['rows'])==6 and len(report['groups'])==6
    assert report['outcome']=='not_qualified' and report['partial_evidence']


@pytest.mark.parametrize('defect',['precision','nonfinite','input','cache','expanded','probabilities'])
def test_retained_instrument_failure_is_auditable_without_becoming_scientific_null(tmp_path,producer_fixture,defect):
    directory=tmp_path/'run';producer_fixture.run_fixture(directory,defect=(defect,6))
    report=audit.audit(directory,synthetic=True)
    assert report['model_calls']==6 and report['scientific_status']=='instrument_failed'
    assert report['instrument_failures'] and report['rows'][0]['readout'] is None


def test_comparison_failure_stops_before_carry(tmp_path,producer_fixture):
    directory=tmp_path/'run';producer_fixture.run_fixture(directory,defect=('comparison',2))
    report=audit.audit(directory,synthetic=True)
    assert report['model_calls']==5 and report['groups'][0]['returned_updates']==0
    assert report['scientific_status']=='instrument_failed'


def test_blocked_cache_preserves_no_claimed_invocation(tmp_path,producer_fixture):
    directory=tmp_path/'run';producer_fixture.run_fixture(directory,defect=('precall_cache',1))
    report=audit.audit(directory,synthetic=True)
    assert report['model_calls']==1 and report['partial_evidence'][0]['status']=='blocked_before_invocation'


def test_preflight_zero_calls_and_no_model(tmp_path,producer_fixture):
    directory=tmp_path/'run';producer_fixture.run_fixture(directory,preflight=True)
    report=audit.audit(directory,synthetic=True)
    assert report['scientific_status']=='preflight_passed' and report['model_calls']==0
    assert [g['status'] for g in report['groups']]==['not_run']*6


def test_actual_ten_pinned_inputs_and_raw_prefix_extraction(tmp_path,producer_fixture,monkeypatch):
    """Zero forwards; independent qualification includes148 accepted raw files."""
    monkeypatch.setattr(producer_fixture.runner,'VOCAB',audit.VOCAB)
    directory=tmp_path/'run';args=producer_fixture.pinned_args(directory);args.preflight_only=True
    producer_fixture.runner.run(args,input_qualifier=producer_fixture.runner.qualify_inputs,
        model_qualifier=producer_fixture.synthetic_model,source_factory=producer_fixture.synthetic_source)
    result=load_json(directory/'result.json')
    assert result['status']=='preflight_passed' and result['model_calls']==0 and not result['model_loaded']
    checker=audit.Audit(directory,synthetic=True)
    with np.load(directory/'literal-inputs.npz',allow_pickle=False) as z:checker.literal={k:z[k] for k in z.files}
    checker.pinned_inputs(result['input_provenance'])
    assert len(checker.bindings)==163
    assert all(g['status']=='not_run' and g['model_calls']==0 for g in result['groups'])


@pytest.mark.parametrize('mutation',['source_profile','status','synthetic','source_commit','packet_hash','array_hash',
    'source_groups','phase_indices','control','checks'])
def test_pinned_prefix_proof_semantics_reject_corruption(mutation):
    base=Path('/data2/shrimp-memory-data/preparations/public-canonical-write0-phase-panel/20261029T172047Z')
    if not all((base/n).is_file() for n in ('accepted-short-prefix-provenance.json','accepted-short-prefix.npz')):pytest.skip(f'pinned historical artifact not available: {base}')
    proof=load_json(base/'accepted-short-prefix-provenance.json')
    with np.load(base/'accepted-short-prefix.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    audit.qualify_prefix_proof(proof,arrays)
    if mutation=='source_profile':proof['source_profile']=audit.PROFILE
    elif mutation=='status':proof['status']='pending'
    elif mutation=='synthetic':proof['accepted_audit_synthetic']=True
    elif mutation=='source_commit':proof['source_commit']='0'*40
    elif mutation=='packet_hash':proof['packet']['sha256']='0'*64
    elif mutation=='array_hash':proof['arrays']['states']['array_sha256']='0'*64
    elif mutation=='source_groups':proof['source_group_ids']=[0,2,4,6]
    elif mutation=='phase_indices':proof['reference_phases']=[1,2,3,4]
    elif mutation=='control':proof['controls']['log10_temperature']=audit.HOLD['log10_temperature']
    else:proof['checks']['every_raw_output']=False
    with pytest.raises(audit.AuditError):audit.qualify_prefix_proof(proof,arrays)








def test_input_qualification_failure_retains_four_unrun_rows(tmp_path,producer_fixture):
    directory=tmp_path/'run';args=producer_fixture.actual_args(directory)
    def fail(args):raise ValueError('injected wrong historical pin')
    producer_fixture.runner.run(args,input_qualifier=fail,source_factory=producer_fixture.synthetic_source)
    report=audit.audit(directory,synthetic=True)
    assert report['scientific_status']=='instrument_failed' and report['model_calls']==0
    assert len(report['rows'])==6 and all(r['status']=='not_run' for r in report['rows'])


def test_resource_stop_before_model_claim_keeps_exact_caps(tmp_path,producer_fixture,monkeypatch):
    directory=tmp_path/'run'
    def fail(budget):raise producer_fixture.runner.ResourceLimit('injected pre-invocation resource gate')
    monkeypatch.setattr(producer_fixture.runner.Budget,'check',fail)
    producer_fixture.run_fixture(directory)
    report=audit.audit(directory,synthetic=True)
    assert report['model_calls']==0 and report['scientific_status']=='incomplete'
    assert all(r['status']=='not_run' for r in report['rows'])


def test_failed_native64_readiness_is_retained_before_prefix(tmp_path,producer_fixture):
    directory=tmp_path/'run';args=producer_fixture.actual_args(directory)
    class Bad(producer_fixture.FakeBackend):
        def runtime_info(self):
            info=super().runtime_info();info['interop_threads']=2;return info
    producer_fixture.runner.run(args,backend_factory=Bad,
        input_qualifier=lambda a:(producer_fixture.synthetic_payload(),dict(synthetic=True,bindings={})),
        model_qualifier=producer_fixture.synthetic_model,source_factory=producer_fixture.synthetic_source)
    report=audit.audit(directory,synthetic=True)
    assert report['model_calls']==0 and report['scientific_status']=='instrument_failed'
    assert report['instrument_failures']==['backend readiness']


@pytest.mark.parametrize('mutation,match',[
    ('state_carry','exact state carry'),('wrong_controls','exact stage control'),('control_bits','exact control bits'),('probabilities','probability gate'),
    ('tokens','argmax token'),('precision','precision gate'),('source_indices','unmodified chain input'),
    ('literal_carry','unmodified chain input'),('literal_phase','unmodified chain input'),('historical_phase','unmodified chain input'),
    ('unexplained_call','unexplained'),('tail_phase','chronological states'),
    ('readout','row readouts'),('requested_bit','requested_bit'),('count','attempt sum'),('resource','terminal resource'),('unrun','terminal status')])
def test_semantic_tampering_fails_after_hashes_are_refreshed(completed,tmp_path,mutation,match):
    directory=clone(completed,tmp_path/'run')
    call=directory/'call-006-g00-update-001.npz'
    if mutation=='state_carry':replace_npz(call,lambda a:a['input'].__setitem__((0,0),a['input'][0,0]+1e-6))
    elif mutation=='wrong_controls':replace_npz(call,lambda a:a['alpha'].__setitem__((),.1))
    elif mutation=='control_bits':replace_npz(call,lambda a:a['alpha_bits'].__setitem__((),0))
    elif mutation=='probabilities':replace_npz(call,lambda a:a['probabilities'].__setitem__((0,0),.5))
    elif mutation=='tokens':replace_npz(call,lambda a:a['token_ids'].__setitem__(0,1))
    elif mutation=='precision':
        path=call.with_suffix('.json');row=load_json(path);row['dtype_observations']['norm']['reduction']='torch.float32';write_json(path,row)
    elif mutation=='source_indices':replace_npz(directory/'literal-inputs.npz',lambda a:a['source_reference_indices'].__setitem__(0,1))
    elif mutation=='literal_carry':replace_npz(directory/'literal-inputs.npz',lambda a:a['initial_states'].__setitem__((1,0,0),a['initial_states'][1,0,0]+1e-3))
    elif mutation=='literal_phase':replace_npz(directory/'literal-inputs.npz',lambda a:a['reference_phases'].__setitem__(0,0))
    elif mutation=='historical_phase':replace_npz(directory/'literal-inputs.npz',lambda a:a['historical_reference_phase'].__setitem__((),1))
    elif mutation=='unexplained_call':(directory/'call-967-g07-update-241.json').write_text('{}\n')
    elif mutation=='tail_phase':replace_npz(directory/'group-01-tail.npz',lambda a:a.__setitem__('states',np.roll(a['states'],1,axis=0)))
    elif mutation=='readout':
        path=directory/'group-01-summary.json';row=load_json(path);row['readouts'][0]['decoded_bit']=1;write_json(path,row)
    elif mutation=='requested_bit':
        path=directory/'result.json';row=load_json(path);row['rows'][0]['requested_bit']=1;write_json(path,row)
    elif mutation=='count':
        path=directory/'result.json';row=load_json(path);row['model_calls']-=1;write_json(path,row)
    elif mutation=='resource':
        path=directory/'result.json';row=load_json(path);row['terminal_resources']['calls_passed']=False;write_json(path,row)
    elif mutation=='unrun':
        path=call.with_suffix('.json');row=load_json(path);row['status']='not_run';write_json(path,row)
    reseal(directory)
    with pytest.raises(audit.AuditError,match=match):audit.audit(directory,synthetic=True)


@pytest.mark.parametrize('field',['coordinate_absolute_gaps','per_state_coordinate_maxima','per_state_l2_differences',
    'state_bytes_equal','token_equal','state_indices','reference_phase','summary_gate','summary_same_input',
    'summary_tolerance','summary_coordinate','summary_token','summary_passed','summary_kind','aggregate'])
def test_each_prefix_replay_component_and_aggregate_is_independently_checked(completed,tmp_path,field):
    directory=clone(completed,tmp_path/'run')
    if field=='aggregate':
        path=directory/'result.json';row=load_json(path);row['prefix_replay_passed']=False;write_json(path,row)
    elif field.startswith('summary_'):
        path=directory/'group-00-summary.json';row=load_json(path);diag=row['prefix_replay']
        key,value={'summary_gate':('gates_readout',True),'summary_same_input':('same_input',False),
            'summary_tolerance':('coordinate_tolerance',1e-5),'summary_coordinate':('coordinate_replay_passed',False),
            'summary_token':('token_replay_passed',False),'summary_passed':('passed',False),
            'summary_kind':('comparison_kind','different_initial_phases')}[field]
        diag[key]=value;write_json(path,row)
    else:
        def change(arrays):
            if arrays[field].dtype==bool:arrays[field].flat[0]=not arrays[field].flat[0]
            else:arrays[field].flat[0]+=1
        replace_npz(directory/'case-00-prefix-replay.npz',change)
    reseal(directory)
    with pytest.raises(audit.AuditError):audit.audit(directory,synthetic=True)


def test_completed_mixed_outcomes_and_failed_replay_do_not_stop_fixed_plan(tmp_path,producer_fixture):
    prior=os.environ.get('INTERMEDIATE_PANEL_AUDITOR_MIXED')
    directory=Path(prior) if prior else tmp_path/'run'
    if not prior:producer_fixture.run_fixture(directory,bits=[0,0,1,0,None,0],history_offset=.125,defect=('token',6))
    report=audit.audit(directory,synthetic=True)
    assert report['model_calls']==966 and report['outcome']=='panel_unresolved'
    assert [c['outcome'] for c in report['cases']]==['case_qualified','case_wrong','case_unresolved']
    assert all(g['status']=='completed' for g in report['groups'])
    assert report['prefix_replay_passed'] is False
    assert all(not d['passed'] and d['same_input'] and not d['gates_readout'] for d in report['prefix_replay'])
    assert [x['token_mismatch_count'] for x in report['prefix_replay'][0]['lanes']]==[1,1]






@pytest.mark.parametrize('group',[1,3,5])
def test_self_consistent_reference_reset_at_boundary_is_rejected(completed,tmp_path,group):
    directory=clone(completed,tmp_path/'run')
    with np.load(directory/'literal-inputs.npz',allow_pickle=False) as z:reset=z['initial_states'][group//2].copy()
    start=audit.GROUP_OFFSETS[group]
    for local,label in [(1,'compare-cached'),(2,'compare-full'),(3,'compare-scalar-0'),(4,'compare-scalar-1'),(5,'update-001')]:
        value=reset[local-3:local-2] if local in (3,4) else reset
        def change(arrays,value=value):arrays['input']=value.copy();arrays['input_after']=value.copy()
        replace_npz(directory/f'call-{start+local+1:03d}-g{group:02d}-{label}.npz',change)
    replace_npz(directory/f'group-{group:02d}-states.npz',lambda a:a['states'].__setitem__(0,reset))
    p=directory/f'group-{group:02d}-summary.json';value=load_json(p);value['initial_state_sha256']=audit.array_hash(reset);write_json(p,value)
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    for event in events:
        if event['event']=='control_selected' and event['group_id']==group:event['initial_state_sha256']=audit.array_hash(reset)
    (directory/'events.jsonl').unlink();(directory/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    reseal(directory)
    with pytest.raises(audit.AuditError,match='literal carried state'):audit.audit(directory,synthetic=True)


def test_identical_repeated_provenance_is_deduplicated_conflict_rejected(tmp_path):
    checker=audit.Audit(tmp_path,synthetic=True)
    a=dict(path='x',bytes=4,sha256='a'*64)
    assert checker.index([a,dict(a,path='./x')],'test')=={'x':(4,'a'*64)}
    with pytest.raises(audit.AuditError,match='conflicting'):
        checker.index([a,dict(a,path='./x',sha256='b'*64)],'test')


def test_corrupted_file_fails_original_hash(completed,tmp_path):
    directory=clone(completed,tmp_path/'run');path=directory/'call-007-g00-update-002.npz'
    raw=bytearray(path.read_bytes());raw[len(raw)//2]^=1;path.unlink();path.write_bytes(raw)
    with pytest.raises(audit.AuditError,match='hash/size'):audit.audit(directory,synthetic=True)


def test_checker_has_no_producer_analysis_imports():
    import ast
    tree=ast.parse((ROOT/'tools/audit_public_intermediate_write0_phase_panel.py').read_text())
    modules=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):modules.extend(x.name for x in node.names)
        if isinstance(node,ast.ImportFrom):modules.append(node.module or '')
    assert not any(x.startswith(('run_public','check_p6','benchmark_qwen','validate_qwen')) for x in modules)


def tiny_transport(directory):
    """Transport-only bytes, deliberately no assertion of numerical evidence."""
    import tarfile
    run=directory/'run';run.mkdir(parents=True)
    groups=[]
    for gid in range(6):
        path=run/f'group-{gid:02d}-summary.json';write_json(path,dict(synthetic=True,group_id=gid))
        group=dict(artifacts=[bind(path)])
        write_json(run/f'group-{gid:02d}.json',group);groups.append(group)
    write_json(run/'result.json',dict(groups=groups))
    files=[dict(bind(p),path='run/'+p.name) for p in sorted(run.iterdir())]
    archive=directory/'final-artifacts.tar'
    with tarfile.open(archive,'w') as tar:
        for row in files:tar.add(directory/row['path'],arcname=row['path'],recursive=False)
    manifest=dict(files=files,transferred_files=files,already_local_files=0,archive=bind(archive))
    write_json(directory/'final-manifest.json',manifest);sha=bind(directory/'final-manifest.json')['sha256']
    write_json(directory/'offline-fixture.json',dict(synthetic=True,no_model_calls=True))
    write_json(directory/'local-return-verification.json',dict(status='verified',files=len(files),manifest_sha256=sha,archive=manifest['archive']))
    write_json(directory/'transfer-closeout.json',dict(state='complete',local_return_verified=True,verified_manifest_entries=len(files),manifest_sha256=sha))
    write_json(directory/'completion.json',dict(transfer_reservation_closed=True))
    write_json(directory/'maintenance-closeout.json',dict(active=False,return_manifest_sha256=sha))
    write_json(directory/'transfer-account.json',dict(reserved_bytes=32*2**30,overhead_reserve=32*2**20,attempts=[dict(bytes=8*2**20)]))
    checker=audit.Audit(run,synthetic=True);checker.result=dict(groups=groups)
    return checker


@pytest.mark.parametrize('mode',['valid','duplicate','conflict','missing','archive','closure','reservation','production_process'])
def test_final_transport_binding_and_failure_contract(tmp_path,mode):
    checker=tiny_transport(tmp_path/'job');job=checker.root.parent
    if mode in ('duplicate','conflict'):
        manifest=load_json(job/'final-manifest.json');row=dict(manifest['files'][0]);row['path']='./'+row['path']
        if mode=='conflict':row['sha256']='0'*64
        manifest['files'].append(row);write_json(job/'final-manifest.json',manifest)
        sha=bind(job/'final-manifest.json')['sha256']
        for name in ('local-return-verification.json','transfer-closeout.json','maintenance-closeout.json'):
            value=load_json(job/name);key='return_manifest_sha256' if name=='maintenance-closeout.json' else 'manifest_sha256';value[key]=sha;write_json(job/name,value)
    elif mode=='missing':(checker.root/'group-01-summary.json').unlink()
    elif mode=='archive':(job/'final-artifacts.tar').write_bytes(b'corrupt archive')
    elif mode=='closure':
        value=load_json(job/'transfer-closeout.json');value['state']='open';write_json(job/'transfer-closeout.json',value)
    elif mode=='reservation':
        value=load_json(job/'transfer-account.json');value['attempts'][0]['bytes']=33*2**30;write_json(job/'transfer-account.json',value)
    elif mode=='production_process':checker.synthetic=False
    if mode in ('valid','duplicate'):
        checker.transport(synthetic_transport=True)
        assert checker.return_status=='synthetic_transport_verified' and checker.return_count==13
    else:
        with pytest.raises(audit.AuditError):checker.transport(synthetic_transport=mode!='production_process')


@pytest.mark.parametrize('row',range(6))
def test_each_neutral_panel_row_can_make_wrong_or_unresolved_without_selection(row):
    bits=[0]*6
    assert audit.panel_outcome(bits)=='panel_qualified'
    bits[row]=1-bits[row]
    assert audit.panel_outcome(bits)=='panel_wrong'
    bits[(row+1)%6]=None
    assert audit.panel_outcome(bits)=='panel_unresolved'


def test_fixed_schedule_accounts_every_phase_case_group_and_call():
    plan=audit.expected_calls()
    assert len(plan)==966 and sum(r['batch_size'] for r in plan)==1914
    assert [sum(r['group_id']==g for r in plan) for g in range(6)]==[77,245]*3
    assert len([r for r in plan if r['category']=='update'])==936
    for row in plan:
        assert row['case_index']==row['group_id']//2
        assert row['stage_index']==row['group_id']%2
        assert row['reference_phase']==3+row['case_index']
    for group in audit.HOLD_GROUPS:
        for lane in range(2):
            row=audit.result_row(group,lane,'not_run')
            i=row['row_id']
            assert row['case_index']==i//2 and row['stage_index']==1
            assert row['lane']==i%2 and row['reference_phase']==3+i//2






@pytest.mark.parametrize('mutation',[
    'raw_case','raw_stage','raw_phase','raw_phase_dtype','raw_packet','call_phase','call_packet','group_case','group_packet','row_phase','row_packet','case_outcome',
    'case_phase','case_initial_hash','case_terminal_hash','case_group_ids','case_row_ids',
    'top_diagnostic','case_diagnostic','case_reset_event','case_previous_terminal','missing_case_start','fake_case_finish'])
def test_panel_metadata_and_boundary_evidence_cannot_be_spoofed(completed,tmp_path,mutation):
    directory=clone(completed,tmp_path/'run')
    if mutation.startswith('raw_'):
        def change(a):
            key={'raw_case':'case_index','raw_stage':'stage_index','raw_phase':'reference_phase','raw_phase_dtype':'reference_phase','raw_packet':'packet_index'}[mutation]
            if mutation=='raw_phase_dtype':a[key]=a[key].astype(np.float64)
            else:a[key][()]+=1
        replace_npz(directory/'call-006-g00-update-001.npz',change)
    elif mutation=='call_phase':
        path=directory/'call-006-g00-update-001.json';row=load_json(path);row['reference_phase']=2;write_json(path,row)
    elif mutation=='call_packet':
        path=directory/'call-006-g00-update-001.json';row=load_json(path);row['packet_index']=0;write_json(path,row)
    elif mutation=='group_packet':
        for name in ('group-00.json','group-00-summary.json'):
            path=directory/name;row=load_json(path);row['packet_index']=0;write_json(path,row)
    elif mutation=='group_case':
        for name in ('group-00.json','group-00-summary.json'):
            path=directory/name;row=load_json(path);row['case_index']=1;write_json(path,row)
    elif mutation=='row_phase':
        path=directory/'result.json';row=load_json(path);row['rows'][0]['reference_phase']=2;write_json(path,row)
    elif mutation=='row_packet':
        path=directory/'result.json';row=load_json(path);row['rows'][0]['packet_index']=0;write_json(path,row)
    elif mutation.startswith(('case_reset','case_previous','missing_case','fake_case')):
        events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        if mutation=='missing_case_start':events=[e for e in events if not(e['event']=='case_started' and e['case_index']==1)]
        elif mutation=='fake_case_finish':
            event=next(e for e in events if e['event']=='case_finished');events.insert(events.index(event),dict(event))
        else:
            event=next(e for e in events if e['event']=='case_started' and e['case_index']==1)
            if mutation=='case_reset_event':event['reset_between_cases']=False
            else:event['previous_terminal_state_sha256']='0'*64
        (directory/'events.jsonl').unlink();(directory/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    else:
        path=directory/'result.json';row=load_json(path);case=row['cases'][1]
        if mutation=='case_outcome':case['outcome']='case_wrong'
        elif mutation=='case_phase':case['reference_phase']=2
        elif mutation=='case_initial_hash':case['initial_state_sha256']='0'*64
        elif mutation=='case_terminal_hash':case['terminal_state_sha256']='0'*64
        elif mutation=='case_group_ids':case['group_ids']=[0,1]
        elif mutation=='case_row_ids':case['row_ids']=[0,1]
        elif mutation=='top_diagnostic':row['prefix_replay'][1]=row['prefix_replay'][0]
        elif mutation=='case_diagnostic':case['prefix_replay']=row['cases'][0]['prefix_replay']
        write_json(path,row)
    reseal(directory,sync_diagnostic=mutation not in ('top_diagnostic','case_diagnostic'))
    with pytest.raises(audit.AuditError):audit.audit(directory,synthetic=True)


@pytest.mark.parametrize('group',[2,4])
def test_self_consistent_carry_across_case_boundary_is_rejected(completed,tmp_path,group):
    directory=clone(completed,tmp_path/'run')
    with np.load(directory/f'group-{group-1:02d}-states.npz',allow_pickle=False) as z:wrong=z['states'][-1].copy()
    start=audit.GROUP_OFFSETS[group]
    for local,label in [(1,'compare-cached'),(2,'compare-full'),(3,'compare-scalar-0'),(4,'compare-scalar-1'),(5,'update-001')]:
        value=wrong[local-3:local-2] if local in (3,4) else wrong
        replace_npz(directory/f'call-{start+local+1:03d}-g{group:02d}-{label}.npz',
                    lambda a,value=value:a.update(input=value.copy(),input_after=value.copy()))
    replace_npz(directory/f'group-{group:02d}-states.npz',lambda a:a['states'].__setitem__(0,wrong))
    path=directory/f'group-{group:02d}-summary.json';row=load_json(path);row['initial_state_sha256']=audit.array_hash(wrong);write_json(path,row)
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    for event in events:
        if event['event']=='control_selected' and event['group_id']==group:event['initial_state_sha256']=audit.array_hash(wrong)
    (directory/'events.jsonl').unlink();(directory/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    reseal(directory)
    with pytest.raises(audit.AuditError,match='literal carried state'):audit.audit(directory,synthetic=True)






def test_loaded_numpy_thread_pool_matches_external_one_thread_contract(tmp_path):
    import ctypes
    assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
    # Query the library NumPy actually mapped, without changing its settings.
    np.dot(np.ones((4,4)),np.ones((4,4)))
    mapped=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                   if 'openblas' in line.lower() and '/' in line})
    assert mapped,'the pinned Linux NumPy OpenBLAS runtime must be observable'
    records=[]
    for name in mapped:
        lib=ctypes.CDLL(name)
        symbols=('scipy_openblas_get_num_threads64_','openblas_get_num_threads64_','openblas_get_num_threads')
        symbol=next((s for s in symbols if hasattr(lib,s)),None)
        assert symbol is not None,'loaded OpenBLAS thread count must be queryable'
        function=getattr(lib,symbol);function.argtypes=[];function.restype=ctypes.c_int
        count=function();assert count==1
        records.append(dict(path=name,symbol=symbol,num_threads=count))
    write_json(tmp_path/'loaded-thread-pools.json',dict(external_threads={k:os.environ[k] for k in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},loaded_pools=records))
