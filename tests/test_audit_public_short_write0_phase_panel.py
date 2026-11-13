"""Model-free independent fixed remaining-phase panel and distinct-history contract tests."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('short_write0_phase_panel_auditor',ROOT/'tools/audit_public_short_write0_phase_panel.py')
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


def tail(bit=0,shift=0):
    refs=references()
    return np.stack([refs[bit,(i+shift)%6] for i in range(13)])


def test_literal_inputs_preserve_both_source_phases_and_complete_histories():
    refs,states,tokens=history();before=(refs.tobytes(),states.tobytes(),tokens.tobytes())
    result=audit.extract_chain_inputs(refs,states,tokens)
    assert (refs.tobytes(),states.tobytes(),tokens.tobytes())==before
    assert np.array_equal(result['source_reference_indices'],np.array([0,1],dtype=np.int64))
    assert result['initial_states'].shape==(4,2,896)
    assert result['initial_states'].tobytes()==np.stack([refs[:,i] for i in (2,3,4,5)]).tobytes()
    assert result['reference_phases'].dtype==np.int64 and result['reference_phases'].shape==(4,) and result['reference_phases'].tolist()==[2,3,4,5]
    assert result['historical_reference_phase'].dtype==np.int64 and result['historical_reference_phase']==0
    assert result['historical_states'].tobytes()==states.tobytes()
    assert result['historical_token_ids'].tobytes()==tokens.tobytes()
    result['initial_states'][0,0]+=1
    assert (refs.tobytes(),states.tobytes(),tokens.tobytes())==before


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
    with pytest.raises(audit.AuditError):audit.extract_chain_inputs(refs,states,tokens)


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


@pytest.mark.parametrize('bits,expected',[
    ([0,0,1,1],'chain_qualified'),([1,0,1,1],'chain_wrong'),([0,0,1,0],'chain_wrong'),
    ([1,1,0,0],'chain_wrong'),([None,0,1,1],'chain_unresolved'),([1,0,None,1],'chain_unresolved'),
    ([0,0,1,None],'chain_unresolved')])
def test_neutral_chain_outcome_preserves_wrong_and_unresolved(bits,expected):
    assert audit.chain_outcome(bits)==expected


def test_cross_phase_diagnostic_preserves_all_states_tokens_signed_zero_without_a_gate():
    refs,states,tokens=history();payload=audit.extract_chain_inputs(refs,states,tokens)
    fresh=states[:37].copy();fresh[0]=refs[:,2];ft=tokens[:36].copy();fresh[17,0,9]=2e-9;fresh[36,1,10]=-0.;ft[8,1]+=1
    arrays,result=audit.cross_phase_diagnostic(fresh,ft,payload,0)
    assert arrays['per_state_coordinate_maxima'].shape==(37,2)
    assert arrays['token_equal'].shape==(36,2) and not arrays['token_equal'][8,1]
    assert not arrays['state_bytes_equal'][36,1] and arrays['per_state_coordinate_maxima'][36,1]==0
    assert arrays['per_state_coordinate_maxima'][17,0]==2e-9 and result['lanes'][1]['token_mismatch_count']==1
    assert result['same_input'] is False and result['gates_chain'] is False and result['tolerance_applicable'] is False
    assert result['current_reference_phase']==2 and result['historical_reference_phase']==0
    assert arrays['current_reference_phase'].dtype==np.int64 and arrays['current_reference_phase']==2
    assert arrays['historical_reference_phase'].dtype==np.int64 and arrays['historical_reference_phase']==0
    assert not {'coordinate_tolerance','coordinate_replay_passed'}&set(result)
    assert all(not {'coordinate_tolerance','coordinate_replay_passed'}&set(row) for row in result['lanes'])


@pytest.mark.parametrize('bad',['nan','overflow','wrong_shape','bad_tokens','phase0_initial'])
def test_cross_phase_diagnostic_rejects_invalid_or_nonfinite_derived_data(bad):
    refs,states,tokens=history();payload=audit.extract_chain_inputs(refs,states,tokens);fresh=states[:37].copy();ft=tokens[:36].copy()
    fresh[0]=refs[:,2]
    if bad=='nan':fresh[2,1,0]=np.nan
    if bad=='overflow':fresh[2,1,0]=1e308
    if bad=='wrong_shape':fresh=fresh[:-1]
    if bad=='bad_tokens':ft=ft.astype(np.float64)
    if bad=='phase0_initial':fresh[0]=refs[:,0]
    with pytest.raises(audit.AuditError):audit.cross_phase_diagnostic(fresh,ft,payload,0)


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
    spec=importlib.util.spec_from_file_location('short_write0_phase_panel_fixture',ROOT/'tests/test_run_public_short_write0_phase_panel.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    old_vocab,old_shape=module.runner.VOCAB,module.runner.batch.CACHE_SHAPE
    module.runner.VOCAB=7;module.runner.batch.CACHE_SHAPE=(1,2,37,1)
    yield module
    module.runner.VOCAB=old_vocab;module.runner.batch.CACHE_SHAPE=old_shape


@pytest.fixture(scope='module')
def completed(tmp_path_factory,producer_fixture):
    prior=os.environ.get('PANEL_AUDITOR_BASELINE')
    if prior:
        directory=Path(prior).resolve()
        assert load_json(directory/'result.json')['model_calls']==2624
        return directory
    directory=tmp_path_factory.mktemp('completed')/'run'
    producer_fixture.run_fixture(directory,bits=[0,0,1,1]*4)
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
    for gid in range(16):
        summary_path=path/f'group-{gid:02d}-summary.json';summary=load_json(summary_path)
        summary['calls']=[r for r in result['calls'][:result['model_calls']] if r['group_id']==gid]
        for key in ('trajectory','tail'):
            if summary[key] is not None:summary[key]=bind(path/summary[key]['path'])
        if summary['cross_phase_history'] is not None:
            summary['cross_phase_history']['artifact']=bind(path/summary['cross_phase_history']['artifact']['path'])
            if gid%4==0 and sync_diagnostic:
                result['cross_phase_history'][gid//4]=summary['cross_phase_history']
                result['cases'][gid//4]['cross_phase_history']=summary['cross_phase_history']
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
    assert report['status']=='audit_passed' and report['model_calls']==2624
    assert report['synthetic'] and report['outcome']=='panel_qualified'
    assert [r['readout']['decoded_bit'] for r in report['rows']]==[0,0,1,1]*4
    assert [r['model_calls'] for r in report['groups']]==[41,245,125,245]*4
    assert len(report['bindings'])==len(list(completed.iterdir()))
    assert [x['attempted_calls'] for x in report['call_costs'].values()]==[16,64,624,1920]
    assert report['cross_phase_history'][0]['same_input'] is False
    assert report['cross_phase_history'][0]['tolerance_applicable'] is False
    assert [row['initial_reference_phase'] for row in report['rows']]==[phase for phase in (2,3,4,5) for _ in range(4)]
    assert [r['write_outcome'] for r in report['rows']]==['correct']*16
    for previous,current in zip(report['groups'],report['groups'][1:]):
        if current['stage_index']:assert previous['terminal_state_sha256']==current['initial_state_sha256']
        else:assert current['predecessor_group_id'] is None


@pytest.mark.parametrize('call,error,category',[(1,OSError,'execution_failed'),(6,OSError,'execution_failed'),
    (42,TimeoutError,'incomplete'),(657,MemoryError,'incomplete')])

def test_partial_failure_preserves_every_attempt_and_unrun_row(tmp_path,producer_fixture,call,error,category):
    directory=tmp_path/'run';producer_fixture.run_fixture(directory,failure=(call,error))
    report=audit.audit(directory,synthetic=True)
    assert report['model_calls']==call and report['scientific_status']==category
    assert len(report['rows'])==16 and len(report['groups'])==16
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
    assert [g['status'] for g in report['groups']]==['not_run']*16


def actual_historical_inputs():
    """Private immutable prerequisites, independently loaded when available."""
    base=Path('/data2/shrimp-memory-data/audits/p6-writes/20261025T222347Z-writes')
    names=dict(trajectory_h0='run/trial-00-endpoint-final-120.npz',trajectory='run/trial-01-endpoint-final-120.npz',fixed_inputs='run/fixed-trial-inputs.npz',
        historical_metadata='run/metadata.json',historical_plan='run/trial-plan.json',
        historical_audit='complete-audit.json',historical_manifest='final-manifest.json')
    if not all((base/name).is_file() for name in names.values()):
        pytest.skip('immutable private historical inputs unavailable')
    records={key:dict(path=str(base/name),bytes=(base/name).stat().st_size,
        sha256=audit.digest((base/name).read_bytes())) for key,name in names.items()}
    prospective=Path('/data2/shrimp-memory-data/preparations/public-short-write0-phase-panel/20261029T153845Z/prospective-phase-panel.npz')
    if not prospective.is_file():pytest.skip('immutable prospective pair unavailable')
    records['prospective_panel']=dict(path=str(prospective),bytes=prospective.stat().st_size,sha256=audit.digest(prospective.read_bytes()))
    with np.load(base/names['fixed_inputs'],allow_pickle=False) as z:refs=z['references']
    states=[];tokens=[]
    for key in ('trajectory_h0','trajectory'):
        with np.load(base/names[key],allow_pickle=False) as z:states.append(z['states']);tokens.append(z['token_ids'])
    states=np.stack(states,axis=1);tokens=np.stack(tokens,axis=1)
    literal=audit.extract_chain_inputs(refs,states,tokens)
    inputs=dict(bindings=records,historical_source_commit='f0c757ed2c7142bd2dd5cb78a796f69f7f49f4c4',
        historical_trial_ids=[0,1],historical_endpoint_controls=audit.ENDPOINT,
        historical_states_array_sha256=audit.array_hash(states),historical_tokens_array_sha256=audit.array_hash(tokens),
        references_array_sha256=audit.array_hash(refs),prospective_panel_array_sha256=audit.array_hash(literal['initial_states']),
        reference_phases=[2,3,4,5],historical_reference_phase=0,same_historical_input=False)
    return inputs,literal,states


def test_actual_pinned_input_preflight_has_zero_calls_and_independent_qualification(tmp_path,producer_fixture,monkeypatch):
    inputs,literal,_=actual_historical_inputs()
    # Historical token IDs use the original vocabulary; no probabilities or model are evaluated.
    monkeypatch.setattr(producer_fixture.runner,'VOCAB',audit.VOCAB)
    directory=tmp_path/'run';args=producer_fixture.actual_args(directory);args.preflight_only=True
    producer_fixture.runner.run(args,input_qualifier=producer_fixture.runner.qualify_inputs,
        model_qualifier=producer_fixture.synthetic_model,source_factory=producer_fixture.synthetic_source)
    result=load_json(directory/'result.json')
    assert result['status']=='preflight_passed' and result['model_calls']==0 and not result['model_loaded']
    checker=audit.Audit(directory,synthetic=True);checker.literal=literal
    checker.pinned_inputs(inputs)
    with np.load(directory/'literal-inputs.npz',allow_pickle=False) as z:
        assert set(z.files)==set(literal)
        assert all(audit.same_bytes(z[key],value) for key,value in literal.items())
    assert len(checker.bindings)==8
    assert all(g['status']=='not_run' and g['model_calls']==0 for g in result['groups'])


@pytest.mark.parametrize('mutation',['swap_lanes','adjacent_state','phase_h1','false_pin','historical_token',
    'prospective_pin','prospective_hash','same_input_claim','literal_phase','historical_phase'])
def test_actual_pins_reject_relabeling_neighbor_phase_or_wrong_binding(tmp_path,mutation):
    inputs,literal,states=actual_historical_inputs()
    if mutation=='swap_lanes':literal['historical_states']=literal['historical_states'][:,::-1].copy()
    elif mutation=='adjacent_state':literal['historical_states'][36]=states[35].copy()
    elif mutation=='phase_h1':literal['initial_states'][0,1]=literal['references'][1,0].copy()
    elif mutation=='false_pin':inputs['bindings']['trajectory']['sha256']='0'*64
    elif mutation=='historical_token':literal['historical_token_ids'][35,0]+=1
    elif mutation=='prospective_pin':inputs['bindings']['prospective_panel']['sha256']='0'*64
    elif mutation=='prospective_hash':inputs['prospective_panel_array_sha256']='0'*64
    elif mutation=='same_input_claim':inputs['same_historical_input']=True
    elif mutation=='literal_phase':literal['reference_phases'][0]=0
    elif mutation=='historical_phase':literal['historical_reference_phase'][()]=1
    checker=audit.Audit(tmp_path,synthetic=True);checker.literal=literal
    with pytest.raises(audit.AuditError):checker.pinned_inputs(inputs)


def test_input_qualification_failure_retains_four_unrun_rows(tmp_path,producer_fixture):
    directory=tmp_path/'run';args=producer_fixture.actual_args(directory)
    def fail(args):raise ValueError('injected wrong historical pin')
    producer_fixture.runner.run(args,input_qualifier=fail,source_factory=producer_fixture.synthetic_source)
    report=audit.audit(directory,synthetic=True)
    assert report['scientific_status']=='instrument_failed' and report['model_calls']==0
    assert len(report['rows'])==16 and all(r['status']=='not_run' for r in report['rows'])


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
    elif mutation=='unexplained_call':(directory/'call-2625-g15-update-241.json').write_text('{}\n')
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
    'state_bytes_equal','token_equal','state_indices','current_reference_phase','historical_reference_phase','summary_gate',
    'summary_same_input','summary_tolerance_applicable','summary_comparison_kind','added_tolerance','added_pass','added_lane_pass'])
def test_each_cross_phase_component_and_no_replay_claim_are_independently_checked(completed,tmp_path,field):
    directory=clone(completed,tmp_path/'run')
    if field.startswith(('summary_','added_')):
        path=directory/'group-00-summary.json';row=load_json(path);diag=row['cross_phase_history']
        if field=='summary_gate':diag['gates_chain']=True
        elif field=='summary_same_input':diag['same_input']=True
        elif field=='summary_tolerance_applicable':diag['tolerance_applicable']=True
        elif field=='summary_comparison_kind':diag['comparison_kind']='same_input_replay'
        elif field=='added_tolerance':diag['coordinate_tolerance']=1e-10
        elif field=='added_pass':diag['coordinate_replay_passed']=True
        elif field=='added_lane_pass':diag['lanes'][0]['coordinate_replay_passed']=True
        write_json(path,row)
    else:
        def change(arrays):
            if arrays[field].dtype==bool:arrays[field].flat[0]=not arrays[field].flat[0]
            else:arrays[field].flat[0]+=1
        replace_npz(directory/'case-00-cross-phase-history.npz',change)
    reseal(directory)
    with pytest.raises(audit.AuditError,match='cross-phase'):audit.audit(directory,synthetic=True)


@pytest.mark.parametrize('where',['result','summary','design'])
def test_legacy_same_input_replay_claim_is_rejected(completed,tmp_path,where):
    directory=clone(completed,tmp_path/'run')
    bogus=dict(coordinate_tolerance=1e-10,coordinate_replay_passed=True)
    if where=='summary':
        path=directory/'group-00-summary.json';row=load_json(path);row['prefix_replay']=bogus;write_json(path,row)
    else:
        path=directory/'result.json';row=load_json(path)
        if where=='result':row['prefix_replay']=bogus
        else:
            design=load_json(directory/'design.json');design['prefix_replay']=bogus
            write_json(directory/'design.json',design);row['design']=design
        write_json(path,row)
    reseal(directory)
    with pytest.raises(audit.AuditError,match='same-input replay'):audit.audit(directory,synthetic=True)


@pytest.mark.parametrize('group',[1,7,14])
def test_self_consistent_reference_reset_at_boundary_is_rejected(completed,tmp_path,group):
    directory=clone(completed,tmp_path/'run')
    with np.load(directory/'literal-inputs.npz',allow_pickle=False) as z:reset=z['initial_states'][group//4].copy()
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
    tree=ast.parse((ROOT/'tools/audit_public_short_write0_phase_panel.py').read_text())
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
    for gid in range(16):
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
        assert checker.return_status=='synthetic_transport_verified' and checker.return_count==33
    else:
        with pytest.raises(audit.AuditError):checker.transport(synthetic_transport=mode!='production_process')


@pytest.mark.parametrize('row',range(16))
def test_each_neutral_panel_row_can_make_wrong_or_unresolved_without_selection(row):
    bits=[0,0,1,1]*4
    assert audit.panel_outcome(bits)=='panel_qualified'
    bits[row]=1-bits[row]
    assert audit.panel_outcome(bits)=='panel_wrong'
    bits[(row+1)%16]=None
    assert audit.panel_outcome(bits)=='panel_unresolved'


def test_fixed_schedule_accounts_every_phase_case_group_and_call():
    plan=audit.expected_calls()
    assert len(plan)==2624 and sum(r['batch_size'] for r in plan)==5200
    assert [sum(r['group_id']==g for r in plan) for g in range(16)]==[41,245,125,245]*4
    assert len([r for r in plan if r['category']=='update'])==2544
    for row in plan:
        assert row['case_index']==row['group_id']//4
        assert row['stage_index']==row['group_id']%4
        assert row['reference_phase']==2+row['case_index']
    for group in audit.HOLD_GROUPS:
        for lane in range(2):
            row=audit.result_row(group,lane,'not_run')
            i=row['row_id']
            assert row['case_index']==i//4 and row['stage_index']==1+2*((i%4)//2)
            assert row['lane']==i%2 and row['reference_phase']==2+i//4


@pytest.mark.parametrize('case',range(4))
def test_each_cross_phase_history_retains_declared_case_and_phase(case):
    refs,states,tokens=history();payload=audit.extract_chain_inputs(refs,states,tokens)
    fresh=states[:37].copy();fresh[0]=payload['initial_states'][case]
    arrays,metric=audit.cross_phase_diagnostic(fresh,tokens[:36],payload,case)
    assert arrays['case_index'].dtype==np.int64 and arrays['case_index'].shape==() and arrays['case_index']==case
    assert metric['case_index']==case and metric['current_reference_phase']==case+2
    assert metric['same_input'] is False and metric['gates_chain'] is False


@pytest.mark.parametrize('case',[-1,4,True,1.])
def test_cross_phase_diagnostic_cannot_select_another_case(case):
    refs,states,tokens=history();payload=audit.extract_chain_inputs(refs,states,tokens)
    with pytest.raises(audit.AuditError):audit.cross_phase_diagnostic(states[:37],tokens[:36],payload,case)


@pytest.mark.parametrize('mutation',[
    'raw_case','raw_stage','raw_phase','raw_phase_dtype','call_phase','group_case','row_phase','case_outcome',
    'case_phase','case_initial_hash','case_terminal_hash','case_group_ids','case_row_ids',
    'top_diagnostic','case_diagnostic','case_reset_event','case_previous_terminal','missing_case_start','fake_case_finish'])
def test_panel_metadata_and_boundary_evidence_cannot_be_spoofed(completed,tmp_path,mutation):
    directory=clone(completed,tmp_path/'run')
    if mutation.startswith('raw_'):
        def change(a):
            key={'raw_case':'case_index','raw_stage':'stage_index','raw_phase':'reference_phase','raw_phase_dtype':'reference_phase'}[mutation]
            if mutation=='raw_phase_dtype':a[key]=a[key].astype(np.float64)
            else:a[key][()]+=1
        replace_npz(directory/'call-006-g00-update-001.npz',change)
    elif mutation=='call_phase':
        path=directory/'call-006-g00-update-001.json';row=load_json(path);row['reference_phase']=3;write_json(path,row)
    elif mutation=='group_case':
        for name in ('group-00.json','group-00-summary.json'):
            path=directory/name;row=load_json(path);row['case_index']=1;write_json(path,row)
    elif mutation=='row_phase':
        path=directory/'result.json';row=load_json(path);row['rows'][0]['reference_phase']=3;write_json(path,row)
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
        if mutation=='case_outcome':case['outcome']='chain_wrong'
        elif mutation=='case_phase':case['reference_phase']=2
        elif mutation=='case_initial_hash':case['initial_state_sha256']='0'*64
        elif mutation=='case_terminal_hash':case['terminal_state_sha256']='0'*64
        elif mutation=='case_group_ids':case['group_ids']=[0,1,2,3]
        elif mutation=='case_row_ids':case['row_ids']=[0,1,2,3]
        elif mutation=='top_diagnostic':row['cross_phase_history'][1]=row['cross_phase_history'][0]
        elif mutation=='case_diagnostic':case['cross_phase_history']=row['cases'][0]['cross_phase_history']
        write_json(path,row)
    reseal(directory,sync_diagnostic=mutation not in ('top_diagnostic','case_diagnostic'))
    with pytest.raises(audit.AuditError):audit.audit(directory,synthetic=True)


@pytest.mark.parametrize('group',[4,8,12])
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


def test_existing_mixed_panel_retains_all_finite_outcomes_large_descriptive_gaps_and_token_mismatch(tmp_path,producer_fixture):
    prior=os.environ.get('PANEL_AUDITOR_MIXED')
    if prior:
        directory=Path(prior).resolve()
        assert (directory/'result.json').is_file()
    else:
        directory=tmp_path/'run'
        producer_fixture.run_fixture(directory,bits=[0,0,1,1,1,0,1,1,None,0,1,1,0,0,1,1],history_offset=.125,defect=('token',6))
    report=audit.audit(directory,synthetic=True)
    assert report['model_calls']==2624 and report['outcome']=='panel_unresolved'
    assert [c['outcome'] for c in report['cases']]==['chain_qualified','chain_wrong','chain_unresolved','chain_qualified']
    assert all(g['status']=='completed' for g in report['groups'])
    assert all(d['maximum_coordinate_gap']>=.125 and d['same_input'] is False and d['gates_chain'] is False
               and d['tolerance_applicable'] is False for d in report['cross_phase_history'])
    assert [r['token_mismatch_count'] for r in report['cross_phase_history'][0]['lanes']]==[1,1]


@pytest.mark.parametrize('outcome',['panel_wrong','panel_unresolved'])
def test_saved_array_variant_keeps_original_finite_panel_expectations(completed,tmp_path,producer_fixture,outcome):
    """Retain both original finite expectations without redoing2624 scripted calls.

    This is explicitly manufactured test evidence. One H lane is replaced by a known period-six reference or a drifting
    trace, with every input/output, subsequent literal carry, tail, claim and
    event updated. No model or forward backend is evaluated.
    """
    group,lane=(15,1) if outcome=='panel_wrong' else (5,0)
    case=group//4;row_start=4*case+2*((group%4)//2)
    directory=clone(completed,tmp_path/'run')
    with np.load(directory/'literal-inputs.npz',allow_pickle=False) as z:refs=z['references'].copy()
    with np.load(directory/f'group-{group:02d}-states.npz',allow_pickle=False) as z:states=z['states'].copy()
    for step in range(1,241):
        states[step,lane]=refs[0,(step-1)%6]
        if outcome=='panel_unresolved':states[step,lane,30]+=.01*step
    for step in range(1,241):
        number=audit.GROUP_OFFSETS[group]+5+step
        replace_npz(directory/f'call-{number:03d}-g{group:02d}-update-{step:03d}.npz',
            lambda a,step=step:a.update(input=states[step-1].copy(),input_after=states[step-1].copy(),output=states[step].copy()))
    replace_npz(directory/f'group-{group:02d}-states.npz',lambda a:a.update(states=states))
    replace_npz(directory/f'group-{group:02d}-tail.npz',lambda a:a.update(states=states[228:241]))
    reads=[producer_fixture.runner.readout(states[228:241,lane],refs) for lane in range(2)]
    assert [x['decoded_bit'] for x in reads]==([1,0] if outcome=='panel_wrong' else [None,0])
    terminal=audit.array_hash(states[-1])
    path=directory/f'group-{group:02d}-summary.json';summary=load_json(path)
    summary.update(readouts=reads,terminal_state_sha256=terminal);write_json(path,summary)
    path=directory/'result.json';result=load_json(path)
    for index,read in enumerate(reads):
        result['rows'][row_start+index].update(readout=read,write_outcome='unresolved' if read['decoded_bit'] is None else 'correct' if read['decoded_bit']==result['rows'][row_start+index]['requested_bit'] else 'wrong')
    result.update(outcome=outcome);result['cases'][case]['outcome']=outcome.replace('panel_','chain_')
    if group%4==3:result['cases'][case]['terminal_state_sha256']=terminal
    else:
        following=group+1;start=audit.GROUP_OFFSETS[following]
        for local,label in [(1,'compare-cached'),(2,'compare-full'),(3,'compare-scalar-0'),(4,'compare-scalar-1'),(5,'update-001')]:
            value=states[-1][local-3:local-2] if local in (3,4) else states[-1]
            replace_npz(directory/f'call-{start+local+1:03d}-g{following:02d}-{label}.npz',lambda a,value=value:a.update(input=value.copy(),input_after=value.copy()))
        replace_npz(directory/f'group-{following:02d}-states.npz',lambda a:a['states'].__setitem__(0,states[-1]))
        next_path=directory/f'group-{following:02d}-summary.json';next_summary=load_json(next_path)
        next_summary['initial_state_sha256']=terminal;write_json(next_path,next_summary)
    source=load_json(directory/'source.json')
    source['saved_array_fixture_variant']=dict(no_model_or_backend_calls=True,changed_group=group,changed_lane=lane,
        expected_outcome=outcome,baseline_manifest=dict(bind(completed/'manifest.json'),path=str(completed/'manifest.json')))
    write_json(directory/'source.json',source);result['source']=source;write_json(path,result)
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    for event in events:
        if event['event']=='case_finished' and event['case_index']==case:
            event['outcome']=outcome.replace('panel_','chain_')
            if group%4==3:event['terminal_state_sha256']=terminal
        if outcome=='panel_unresolved' and event['event']=='control_selected' and event['group_id']==group+1:event['initial_state_sha256']=terminal
        if event['event']=='run_finished':event['outcome']=outcome
    (directory/'events.jsonl').unlink();(directory/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    reseal(directory)
    report=audit.audit(directory,synthetic=True)
    assert report['model_calls']==2624 and report['scientific_status']=='completed' and report['outcome']==outcome
    assert all(g['status']=='completed' for g in report['groups'])
    bits=[0,0,1,1]*4;bits[row_start+lane]=0 if outcome=='panel_wrong' else None
    assert [r['readout']['decoded_bit'] for r in report['rows']]==bits
    expected=['chain_qualified']*4;expected[case]=outcome.replace('panel_','chain_')
    assert [c['outcome'] for c in report['cases']]==expected


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
