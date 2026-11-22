"""Changed-contract tests for the independent carried72 four-stage checker.

Synthetic forwards only; reuse closed complete and mixed fixtures when supplied.
Fresh tiny fixture generation stubs fsync only in this test process. Production
source, the full vocabulary serializer and actual return durability are unchanged.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('carried72_handoff_phase_panel_auditor',ROOT/'tools/audit_public_carried72_handoff_phase_panel.py')
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

def tail(bit=0,shift=0):
    refs=references()
    return np.stack([refs[bit,(i+shift)%6] for i in range(13)])

def prospective_arrays():
    return dict(initial_states=np.stack([references()[:,i] for i in (2,3,4,5)]),reference_indices=np.array([0,1],dtype=np.int64),
                reference_phases=np.asarray([2,3,4,5],dtype=np.int64))

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

def test_array_hash_includes_dtype_and_shape():
    x=np.zeros((2,3),dtype=np.float64)
    assert audit.array_hash(x)!=audit.array_hash(x.reshape(6))
    assert audit.array_hash(x)!=audit.array_hash(x.view(np.int64))

@pytest.fixture(scope='module')
def producer_fixture():
    """Use actual producer files, but never its classification in the checker."""
    spec=importlib.util.spec_from_file_location('carried72_handoff_phase_panel_fixture',ROOT/'tests/test_public_carried72_handoff_phase_panel.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    old_vocab,old_shape,old_fsync=module.runner.VOCAB,module.runner.batch.CACHE_SHAPE,module.runner.os.fsync
    module.runner.VOCAB=7;module.runner.batch.CACHE_SHAPE=(1,2,37,1)
    module.runner.os.fsync=lambda descriptor:None
    yield module
    module.runner.VOCAB=old_vocab;module.runner.batch.CACHE_SHAPE=old_shape;module.runner.os.fsync=old_fsync

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
    for gid in range(12):
        summary_path=path/f'group-{gid:02d}-summary.json';summary=load_json(summary_path)
        summary['calls']=[r for r in result['calls'][:result['model_calls']] if r['group_id']==gid]
        for key in ('trajectory','tail'):
            if summary[key] is not None:summary[key]=bind(path/summary[key]['path'])
        if summary['prefix_replay'] is not None:
            summary['prefix_replay']['artifact']=bind(path/summary['prefix_replay']['artifact']['path'])
            if gid%4==0 and sync_diagnostic:
                result['prefix_replay'][gid//4]=summary['prefix_replay']
                result['cases'][gid//4]['prefix_replay']=summary['prefix_replay']
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

def test_checker_has_no_producer_analysis_imports():
    import ast
    tree=ast.parse((ROOT/'tools/audit_public_carried72_handoff_phase_panel.py').read_text())
    modules=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):modules.extend(x.name for x in node.names)
        if isinstance(node,ast.ImportFrom):modules.append(node.module or '')
    assert not any(x.startswith(('run_public','check_p6','benchmark_qwen','validate_qwen')) for x in modules)

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

@pytest.fixture(scope='module')
def completed(tmp_path_factory,producer_fixture):
    prior=os.environ.get('CARRIED72_AUDITOR_COMPLETE')
    if prior:
        directory=Path(prior).resolve()
        assert load_json(directory/'result.json')['model_calls']==1788
        return directory
    directory=tmp_path_factory.mktemp('completed')/'run'
    producer_fixture.run_fixture(directory)
    assert load_json(directory/'result.json')['model_calls']==1788
    return directory


def test_exact_schedule_row_identity_and_three_controls():
    plan=audit.expected_calls()
    assert len(plan)==1788 and sum(r['batch_size'] for r in plan)==3540
    assert [sum(r['group_id']==g for r in plan) for g in range(12)]==[77,149,125,245]*3
    assert sum(r['category']=='update' for r in plan)==1728
    assert [r['call'] for r in plan if r['category']=='prefix']==[1,78,227,352,597,674,823,948,1193,1270,1419,1544]
    for row in plan:
        case,stage=divmod(row['group_id'],4)
        assert (row['case_index'],row['stage_index'],row['reference_phase'],row['packet_index'])==(case,stage,case+3,case+1)
        assert row['control_name']==('E0','H','E1','H')[stage]
        assert row['controls']==audit.CONTROLS[row['control_name']]
    for g in audit.HOLD_GROUPS:
        for lane in range(2):
            row=audit.result_row(g,lane,'not_run')
            case,stage=divmod(g,4);desired=stage//2
            assert row['row_id']==case*4+2*desired+lane
            assert row['source_reference_index']==lane and row['requested_bit']==desired
            assert row['name']==f'phase{case+3}-write{desired}-from-h{lane}'


@pytest.mark.parametrize('row',range(12))
def test_every_neutral_row_retains_wrong_and_unresolved(row):
    bits=[0,0,1,1]*3
    assert audit.panel_outcome(bits)=='panel_qualified'
    bits[row]=1-bits[row]
    assert audit.panel_outcome(bits)=='panel_wrong'
    bits[(row+1)%12]=None
    assert audit.panel_outcome(bits)=='panel_unresolved'


def test_complete_actual_file_contract_h144_h240_and_all_nine_carries(completed):
    report=audit.audit(completed,synthetic=True)
    assert report['status']=='audit_passed' and report['model_calls']==1788
    assert report['outcome']=='panel_qualified' and report['prefix_replay_passed'] is True
    assert [r['readout']['decoded_bit'] for r in report['rows']]==[0,0,1,1]*3
    assert [g['model_calls'] for g in report['groups']]==[77,149,125,245]*3
    assert [x['attempted_calls'] for x in report['call_costs'].values()]==[12,48,576,1152]
    assert len(report['bindings'])==len(list(completed.iterdir()))
    carried=0
    for g in report['groups']:
        if g['stage_index']:
            assert g['predecessor_group_id']==g['group_id']-1
            assert report['groups'][g['group_id']-1]['terminal_state_sha256']==g['initial_state_sha256']
            carried+=1
        else:assert g['predecessor_group_id'] is None
    assert carried==9
    for gid in (1,3,5,7,9,11):
        end=144 if gid%4==1 else 240
        with np.load(completed/f'group-{gid:02d}-tail.npz',allow_pickle=False) as z:
            assert z['state_indices'].tolist()==list(range(end-12,end+1))
        assert report['groups'][gid]['completed_updates']==end


def test_mixed_wrong_unresolved_and_replay_mismatch_continue_all_stages(tmp_path,producer_fixture):
    prior=os.environ.get('CARRIED72_AUDITOR_MIXED')
    directory=Path(prior).resolve() if prior else tmp_path/'run'
    bits=[0,0,1,1,0,1,1,1,None,0,1,1]
    if not prior:producer_fixture.run_fixture(directory,bits=bits,history_offset=.125,defect=('token',6))
    report=audit.audit(directory,synthetic=True)
    assert report['model_calls']==1788 and report['outcome']=='panel_unresolved'
    assert [c['outcome'] for c in report['cases']]==['case_qualified','case_wrong','case_unresolved']
    assert [r['readout']['decoded_bit'] for r in report['rows']]==bits
    assert all(g['status']=='completed' for g in report['groups'])
    assert report['prefix_replay_passed'] is False
    assert all(not d['passed'] and not d['gates_readout'] for d in report['prefix_replay'])


@pytest.mark.parametrize('call,error,category',[(6,OSError,'execution_failed'),(227,TimeoutError,'incomplete'),(597,MemoryError,'incomplete')])
def test_partial_stage_and_case_failures_preserve_twelve_rows(tmp_path,producer_fixture,call,error,category):
    prior=os.environ.get('CARRIED72_AUDITOR_EARLY') if call==6 and error is OSError else None
    directory=Path(prior).resolve() if prior else tmp_path/'run'
    if not prior:producer_fixture.run_fixture(directory,failure=(call,error))
    report=audit.audit(directory,synthetic=True)
    assert report['model_calls']==call and report['scientific_status']==category
    assert len(report['rows'])==12 and len(report['groups'])==12
    assert report['outcome']=='not_qualified' and report['partial_evidence']
    assert all(g['model_calls']==0 for g in report['groups'] if g['group_id']>next(g['group_id'] for g in report['groups'] if g['status']==category))


@pytest.mark.parametrize('defect',['precision','precall_cache'])
def test_instrument_failures_and_blocked_calls_are_not_scientific_u(tmp_path,producer_fixture,defect):
    directory=tmp_path/'run'
    producer_fixture.run_fixture(directory,defect=(defect,1 if defect=='precall_cache' else 6))
    report=audit.audit(directory,synthetic=True)
    assert report['scientific_status']=='instrument_failed'
    assert report['model_calls']==(1 if defect=='precall_cache' else 6)
    assert all(r['readout'] is None for r in report['rows'])


def test_preflight_closes_all_twelve_groups_without_loading_model(tmp_path,producer_fixture):
    directory=tmp_path/'run';producer_fixture.run_fixture(directory,preflight=True)
    report=audit.audit(directory,synthetic=True)
    assert report['scientific_status']=='preflight_passed' and report['model_calls']==0
    assert [g['status'] for g in report['groups']]==['not_run']*12


def change_group_initial(directory,group,wrong):
    start=audit.GROUP_OFFSETS[group]
    for local,label in [(1,'compare-cached'),(2,'compare-full'),(3,'compare-scalar-0'),(4,'compare-scalar-1'),(5,'update-001')]:
        value=wrong[local-3:local-2] if local in (3,4) else wrong
        replace_npz(directory/f'call-{start+local+1:03d}-g{group:02d}-{label}.npz',
            lambda a,value=value:a.update(input=value.copy(),input_after=value.copy()))
    replace_npz(directory/f'group-{group:02d}-states.npz',lambda a:a['states'].__setitem__(0,wrong))
    path=directory/f'group-{group:02d}-summary.json';row=load_json(path)
    row['initial_state_sha256']=audit.array_hash(wrong);write_json(path,row)
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    for event in events:
        if event['event']=='control_selected' and event['group_id']==group:event['initial_state_sha256']=audit.array_hash(wrong)
    path=directory/'events.jsonl';path.unlink();path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    reseal(directory)


@pytest.mark.parametrize('group',[1,2,3,5,6,7,9,10,11])
def test_each_carried_boundary_rejects_self_consistent_reset(completed,tmp_path,group):
    directory=clone(completed,tmp_path/'run')
    with np.load(directory/'literal-inputs.npz',allow_pickle=False) as z:wrong=z['initial_states'][group//4].copy()
    change_group_initial(directory,group,wrong)
    with pytest.raises(audit.AuditError,match='literal carried state'):audit.audit(directory,synthetic=True)


@pytest.mark.parametrize('group',[4,8])
def test_case_boundaries_require_literal_reset_not_prior_terminal(completed,tmp_path,group):
    directory=clone(completed,tmp_path/'run')
    with np.load(directory/f'group-{group-1:02d}-states.npz',allow_pickle=False) as z:wrong=z['states'][-1].copy()
    change_group_initial(directory,group,wrong)
    with pytest.raises(audit.AuditError,match='literal carried state'):audit.audit(directory,synthetic=True)


@pytest.mark.parametrize('group',[1,3,5,7,9,11])
def test_each_tail_has_exact_duration_specific_indices(completed,tmp_path,group):
    directory=clone(completed,tmp_path/'run')
    replace_npz(directory/f'group-{group:02d}-tail.npz',lambda a:a['state_indices'].__iadd__(1))
    reseal(directory)
    with pytest.raises(audit.AuditError,match='tail indices'):audit.audit(directory,synthetic=True)


@pytest.mark.parametrize('mutation',['stage','packet','control','requested','case_groups','tail_window','closure_pairs','aggregate','raw_token'])
def test_new_mapping_and_gates_reject_hash_refreshed_corruption(completed,tmp_path,mutation):
    directory=clone(completed,tmp_path/'run')
    if mutation in ('stage','packet','control','raw_token'):
        path=directory/'call-232-g02-update-001.npz'
        def change(a):
            if mutation=='stage':a['stage_index'][()]=0
            elif mutation=='packet':a['packet_index'][()]=2
            elif mutation=='control':a['alpha'][()]=audit.ENDPOINT['alpha']
            else:a['token_ids'][0]=1
        replace_npz(path,change)
    elif mutation in ('requested','case_groups','aggregate'):
        path=directory/'result.json';value=load_json(path)
        if mutation=='requested':value['rows'][2]['requested_bit']=0
        elif mutation=='case_groups':value['cases'][1]['group_ids']=[2,3]
        else:value['prefix_replay_passed']=False
        write_json(path,value)
    else:
        path=directory/'design.json';value=load_json(path)
        if mutation=='tail_window':value['readout']['retained_state_indices_by_group']['1']=[228,240]
        else:value['readout']['closure_pairs_by_group']['1'][0]=[138,132]
        write_json(path,value)
    reseal(directory)
    with pytest.raises(audit.AuditError):audit.audit(directory,synthetic=True)


@pytest.mark.parametrize('case',range(3))
def test_each_same_phase_prefix_stays_bound_to_original_packet_index(completed,tmp_path,case):
    directory=clone(completed,tmp_path/'run')
    replace_npz(directory/f'case-{case:02d}-prefix-replay.npz',lambda a:a['packet_index'].__setitem__((),0))
    reseal(directory)
    with pytest.raises(audit.AuditError):audit.audit(directory,synthetic=True)


def test_production_cli_has_no_synthetic_bypass():
    import ast
    tree=ast.parse((ROOT/'tools/audit_public_carried72_handoff_phase_panel.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    flags=[arg.value for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='add_argument' for arg in n.args if isinstance(arg,ast.Constant)]
    assert flags==['--run-dir','--source-root','--input-root','--output-dir']
