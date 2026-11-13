"""Model-free tests for the independent nominal-handoff saved-file checker."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('nominal_handoffs_auditor',ROOT/'tools/audit_public_nominal_handoffs.py')
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
    states=np.stack([refs[1,i%6] for i in range(121)])
    states[12,3]+=.012;states[36,4]+=.036;states[60,5]+=.060
    return refs,states


def tail(bit=0,shift=0):
    refs=references()
    return np.stack([refs[bit,(i+shift)%6] for i in range(13)])


def test_literal_handoffs_preserve_original_indices_phase_bytes_and_inputs():
    refs,states=history();before=(refs.tobytes(),states.tobytes())
    result=audit.extract_handoffs(refs,states)
    assert (refs.tobytes(),states.tobytes())==before
    assert np.array_equal(result['historical_state_indices'],np.array([[0,60],[12,36]],dtype=np.int64))
    assert result['initial_states'].shape==(2,2,896)
    for key,index in [('h1',0),('x12',12),('x36',36),('x60',60)]:
        assert result[key].tobytes()==states[index].tobytes()
    assert result['initial_states'].tobytes()==states[[[0,60],[12,36]]].tobytes()
    assert not np.array_equal(result['x12'],result['x36'])
    result['initial_states'][0,0,0]+=1
    assert (refs.tobytes(),states.tobytes())==before


@pytest.mark.parametrize('bad',['phase','nonfinite','float32','shape'])
def test_invalid_literal_inputs_fail_closed(bad):
    refs,states=history()
    if bad=='phase':states[0]=refs[1,1]
    if bad=='nonfinite':states[12,8]=np.nan
    if bad=='float32':states=states.astype(np.float32)
    if bad=='shape':states=states[:-1]
    with pytest.raises(audit.AuditError):audit.extract_handoffs(refs,states)


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
    ([1,0,0,0],'both_early_handoffs_h0'),
    ([1,0,1,1],'both_early_handoffs_h1'),
    ([1,0,0,1],'mixed_early_handoffs'),
    ([1,0,1,0],'mixed_early_handoffs'),
    ([1,0,None,0],'early_handoff_unresolved'),
    ([1,0,1,None],'early_handoff_unresolved'),
    ([None,0,None,None],'control_unqualified'),
    ([1,1,0,0],'control_unqualified')])
def test_panel_classification_preserves_controls_and_unresolved(bits,expected):
    assert audit.panel_outcome(bits)==expected


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
    spec=importlib.util.spec_from_file_location('nominal_handoffs_fixture',ROOT/'tests/test_run_public_nominal_handoffs.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    old_vocab,old_shape=module.runner.VOCAB,module.runner.batch.CACHE_SHAPE
    module.runner.VOCAB=7;module.runner.batch.CACHE_SHAPE=(1,2,37,1)
    yield module
    module.runner.VOCAB=old_vocab;module.runner.batch.CACHE_SHAPE=old_shape


@pytest.fixture(scope='module')
def completed(tmp_path_factory,producer_fixture):
    directory=tmp_path_factory.mktemp('completed')/'run'
    producer_fixture.run_fixture(directory,bits=[1,0,1,1])
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


def reseal(path):
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
    for gid in range(2):
        summary_path=path/f'group-{gid:02d}-summary.json';summary=load_json(summary_path)
        summary['calls']=[r for r in result['calls'][:result['model_calls']] if r['group_id']==gid]
        for key in ('trajectory','tail'):
            if summary[key] is not None:summary[key]=bind(path/summary[key]['path'])
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
    assert report['status']=='audit_passed' and report['model_calls']==490
    assert report['synthetic'] and report['outcome']=='both_early_handoffs_h1'
    assert [r['readout']['decoded_bit'] for r in report['rows']]==[1,0,1,1]
    assert [r['model_calls'] for r in report['groups']]==[245]*2
    assert len(report['bindings'])==len(list(completed.iterdir()))


@pytest.mark.parametrize('bits,outcome',[
    ([1,0,0,0],'both_early_handoffs_h0'),
    ([1,0,0,1],'mixed_early_handoffs'),
    ([1,0,1,0],'mixed_early_handoffs'),
    ([1,0,1,None],'early_handoff_unresolved'),
    ([None,0,0,0],'control_unqualified')])
def test_every_finite_outcome_completes_all_groups(tmp_path,producer_fixture,bits,outcome):
    directory=tmp_path/'run';producer_fixture.run_fixture(directory,bits=bits)
    report=audit.audit(directory,synthetic=True)
    assert report['outcome']==outcome and report['model_calls']==490
    assert all(g['status']=='completed' for g in report['groups'])


@pytest.mark.parametrize('call,error,category',[(1,OSError,'execution_failed'),(6,OSError,'execution_failed'),
    (246,TimeoutError,'incomplete'),(489,MemoryError,'incomplete'),(490,OSError,'execution_failed')])
def test_partial_failure_preserves_every_attempt_and_unrun_row(tmp_path,producer_fixture,call,error,category):
    directory=tmp_path/'run';producer_fixture.run_fixture(directory,failure=(call,error))
    report=audit.audit(directory,synthetic=True)
    assert report['model_calls']==call and report['scientific_status']==category
    assert len(report['rows'])==4 and len(report['groups'])==2
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
    assert [g['status'] for g in report['groups']]==['not_run']*2


def actual_historical_inputs():
    """Private immutable prerequisites, independently loaded when available."""
    base=Path('/data2/shrimp-memory-data/audits/p6-writes/20261025T222347Z-writes')
    names=dict(trajectory='run/trial-01-endpoint-final-120.npz',fixed_inputs='run/fixed-trial-inputs.npz',
        historical_metadata='run/metadata.json',historical_plan='run/trial-plan.json',
        historical_audit='complete-audit.json',historical_manifest='final-manifest.json')
    if not all((base/name).is_file() for name in names.values()):
        pytest.skip('immutable private historical inputs unavailable')
    records={key:dict(path=str(base/name),bytes=(base/name).stat().st_size,
        sha256=audit.digest((base/name).read_bytes())) for key,name in names.items()}
    with np.load(base/names['fixed_inputs'],allow_pickle=False) as z:refs=z['references']
    with np.load(base/names['trajectory'],allow_pickle=False) as z:states=z['states']
    return dict(bindings=records),audit.extract_handoffs(refs,states),states


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
    assert len(checker.bindings)==6
    assert all(g['status']=='not_run' and g['model_calls']==0 for g in result['groups'])


@pytest.mark.parametrize('mutation',['swap12and36','adjacent12','phase_h1','false_pin'])
def test_actual_pins_reject_relabeling_neighbor_phase_or_wrong_binding(tmp_path,mutation):
    inputs,literal,states=actual_historical_inputs()
    if mutation=='swap12and36':literal['x12'],literal['x36']=literal['x36'],literal['x12']
    elif mutation=='adjacent12':literal['x12']=states[11].copy()
    elif mutation=='phase_h1':literal['h1']=literal['references'][1,1].copy()
    elif mutation=='false_pin':inputs['bindings']['trajectory']['sha256']='0'*64
    literal['initial_states']=np.stack((literal['h1'],literal['x60'],literal['x12'],literal['x36'])).reshape(2,2,896)
    checker=audit.Audit(tmp_path,synthetic=True);checker.literal=literal
    with pytest.raises(audit.AuditError):checker.pinned_inputs(inputs)


def test_input_qualification_failure_retains_four_unrun_rows(tmp_path,producer_fixture):
    directory=tmp_path/'run';args=producer_fixture.actual_args(directory)
    def fail(args):raise ValueError('injected wrong historical pin')
    producer_fixture.runner.run(args,input_qualifier=fail,source_factory=producer_fixture.synthetic_source)
    report=audit.audit(directory,synthetic=True)
    assert report['scientific_status']=='instrument_failed' and report['model_calls']==0
    assert len(report['rows'])==4 and all(r['status']=='not_run' for r in report['rows'])


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
    ('state_carry','exact state carry'),('wrong_controls','exact H'),('probabilities','probability gate'),
    ('tokens','argmax token'),('precision','precision gate'),('source_indices','original trajectory row indices'),
    ('literal_carry','unmodified state/phase carry'),('unexplained_call','unexplained'),('tail_phase','chronological states'),
    ('readout','row readouts'),('count','attempt sum'),('resource','terminal resource'),('unrun','terminal status')])
def test_semantic_tampering_fails_after_hashes_are_refreshed(completed,tmp_path,mutation,match):
    directory=clone(completed,tmp_path/'run')
    call=directory/'call-006-g00-update-001.npz'
    if mutation=='state_carry':replace_npz(call,lambda a:a['input'].__setitem__((0,0),a['input'][0,0]+1e-6))
    elif mutation=='wrong_controls':replace_npz(call,lambda a:a['alpha'].__setitem__((),.1))
    elif mutation=='probabilities':replace_npz(call,lambda a:a['probabilities'].__setitem__((0,0),.5))
    elif mutation=='tokens':replace_npz(call,lambda a:a['token_ids'].__setitem__(0,1))
    elif mutation=='precision':
        path=call.with_suffix('.json');row=load_json(path);row['dtype_observations']['norm']['reduction']='torch.float32';write_json(path,row)
    elif mutation=='source_indices':replace_npz(directory/'literal-inputs.npz',lambda a:a['historical_state_indices'].__setitem__((1,0),13))
    elif mutation=='literal_carry':replace_npz(directory/'literal-inputs.npz',lambda a:a['initial_states'].__setitem__((1,0,0),a['initial_states'][1,0,0]+1e-3))
    elif mutation=='unexplained_call':(directory/'call-491-g01-update-241.json').write_text('{}\n')
    elif mutation=='tail_phase':replace_npz(directory/'group-00-tail.npz',lambda a:a.__setitem__('states',np.roll(a['states'],1,axis=0)))
    elif mutation=='readout':
        path=directory/'group-00-summary.json';row=load_json(path);row['readouts'][0]['decoded_bit']=0;write_json(path,row)
    elif mutation=='count':
        path=directory/'result.json';row=load_json(path);row['model_calls']-=1;write_json(path,row)
    elif mutation=='resource':
        path=directory/'result.json';row=load_json(path);row['terminal_resources']['calls_passed']=False;write_json(path,row)
    elif mutation=='unrun':
        path=call.with_suffix('.json');row=load_json(path);row['status']='not_run';write_json(path,row)
    reseal(directory)
    with pytest.raises(audit.AuditError,match=match):audit.audit(directory,synthetic=True)


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
    tree=ast.parse((ROOT/'tools/audit_public_nominal_handoffs.py').read_text())
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
    for gid in range(2):
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
    write_json(directory/'transfer-account.json',dict(reserved_bytes=8*2**30,overhead_reserve=32*2**20,attempts=[dict(bytes=8*2**20)]))
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
        value=load_json(job/'transfer-account.json');value['attempts'][0]['bytes']=9*2**30;write_json(job/'transfer-account.json',value)
    elif mode=='production_process':checker.synthetic=False
    if mode in ('valid','duplicate'):
        checker.transport(synthetic_transport=True)
        assert checker.return_status=='synthetic_transport_verified' and checker.return_count==5
    else:
        with pytest.raises(audit.AuditError):checker.transport(synthetic_transport=mode!='production_process')
