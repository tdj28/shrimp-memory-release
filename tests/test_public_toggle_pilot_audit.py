"""Focused independent contracts for the fixed compact public pilots."""
import sys
import json
import shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
import pytest
import audit_public_toggle_pilot as auditor

DATA=Path(__file__).resolve().parents[2]/'shrimp-memory-data'
TOGGLE_FIXTURES=DATA/'preparations/public-toggle-pilot/20261029T205811Z/producer-validation/test-temp'
PATH_FIXTURE=DATA/'preparations/public-path-comparison/20261029T210407Z-root-validation/first-serializer-fixture/test_actual_serializer_and_car0/run'
PAIR_FIXTURES=DATA/'preparations/public-core-pair/20261029T210128Z/validation-20261029T210746Z/test-work'


@pytest.fixture
def tiny(monkeypatch):
    """Only synthetic geometry/provenance adapters; never exposed by CLI.

    Reuses immutable producer outputs. No model, producer execution, or complete
    word regeneration. Native cache and provenance stay mandatory in production.
    """
    monkeypatch.setattr(auditor.primitive,'VOCAB',4)
    def cache(values,batch,length):
        auditor.require(set(values)=={'key','length'},'synthetic cache keys')
        auditor.primitive.valid_matrix(values['key'],(batch,1,2,1),'synthetic cache')
        auditor.require(values['length'].dtype==np.int64 and values['length'].shape==() and int(values['length'])==length,'synthetic cache length')
        return auditor.primitive.cache_digest(values)
    monkeypatch.setattr(auditor.primitive,'cache_arrays',cache)
    def provenance(self):
        self.check(self.synthetic is True,'test-only provenance requires synthetic')
        self.record(self.run/'source.json')
        if not (self.run/'reference-cycles.npz').exists():
            self.check(self.result['model_calls']==0,'no work before fixture inputs')
            self.refs=None;return
        self.refs=auditor.primitive.npz(self.run/'reference-cycles.npz')['states']
        auditor.primitive.valid_matrix(self.refs,(2,6,896),'synthetic references')
        self.record(self.run/'reference-cycles.npz');self.initial=self.refs[:,0].copy()
        self.check(auditor.same(np.load(self.run/'reference-starts.npy'),self.initial),'synthetic literal pair')
        self.record(self.run/'reference-starts.npy')
    monkeypatch.setattr(auditor.Evidence,'provenance',provenance)


def inspect_fixture(path,profile):
    if not path.is_dir():pytest.skip('retained compact serializer is not installed')
    return auditor.audit(path,profile,synthetic=True,with_transport=False)


@pytest.mark.parametrize('profile,groups,stages,calls,updates',[
    (auditor.TOGGLE,11,22,3258,3192),(auditor.PATH,3,9,2907,2880)])
def test_fixed_schedule_counts(profile,groups,stages,calls,updates):
    g,s,c=auditor.schedule(profile)
    assert (len(g),len(s),len(c))==(groups,stages,calls)
    assert sum(x['steps'] for x in s)==updates
    assert [x['call'] for x in c]==list(range(1,calls+1))


@pytest.mark.parametrize('bits,want',[
    ([1,0],'toggle'),([0,1],'identity'),([0,0],'reset0'),([1,1],'reset1'),
    ([None,0],'unresolved'),([1,None],'unresolved'),([None,None],'unresolved')])
def test_simultaneous_mapping_keeps_original_lanes(bits,want):
    assert auditor.classify_pair(bits)==want


@pytest.mark.parametrize('bad',[[True,0],[1,False],[0],[2,0],(1,0),[1.,0]])
def test_invalid_pair_does_not_become_toggle(bad):
    with pytest.raises(ValueError):auditor.classify_pair(bad)


def test_path_matched_controls_and_dwells():
    groups,stages,_=auditor.schedule(auditor.PATH)
    assert [[stages[i]['steps'] for i in g['stage_ids']] for g in groups]==[[240,240],[960,240],[240]*5]
    assert stages[0]['controls']==stages[2]['controls']==stages[7]['controls']==auditor.controls(.78125)
    assert all(stages[g['stage_ids'][-1]]['controls']==auditor.controls(.75) for g in groups)


def test_phase_alignment_not_same_index_only():
    cycle=np.zeros((6,896));cycle[np.arange(6),np.arange(6)]=1.
    shifted=np.roll(cycle,2,axis=0)
    out=auditor.separation(cycle,shifted)
    assert out['minimum_over_shifts_of_maximum_l2_distance']==0
    assert not out['passed'] and len(out['all_shifts'])==6
    assert all(len(row['phase_l2_distances'])==6 for row in out['all_shifts'])


@pytest.mark.parametrize('distance,status',[(1e-8,'same_within_tolerance'),(5e-5,'intermediate'),(.1,'distinct')])
def test_path_intermediate_band(distance,status):
    assert auditor.pair_status({'both_primitive_six':True,'pair_separation':{'minimum_over_shifts_of_maximum_l2_distance':distance}})==status
    assert auditor.pair_status({'both_primitive_six':False})=='unresolved'


def test_actual_p_input_status_and_terminal_first_rotation():
    base=DATA/'audits/public-root-continuation/20261029T110830Z-public-root-continuation'
    if not base.is_dir():pytest.skip('pinned P input packet not installed')
    paths={key:base/name for key,name in [('cycle_packet','cycle-packet.json'),('cycle_states','cycle-states.npz'),('source_audit','source-audit.json')]}
    refs,records=auditor.p_references(paths)
    assert json.loads(paths['source_audit'].read_text())['status']=='passed'
    with np.load(paths['cycle_states']) as z:
        assert auditor.same(refs[:,0],z['source_terminal_states'])
        assert auditor.same(refs[:,1:].transpose(1,0,2),z['source_cycle_states'][:-1])
    assert {k:r['sha256'] for k,r in records.items()}==auditor.P_PINS


def test_actual_complete_toggle_file_contract(tiny):
    report=inspect_fixture(TOGGLE_FIXTURES/'test_complete_compact_contract0/complete',auditor.TOGGLE)
    assert report['status']=='audit_passed' and report['model_calls']==3258
    assert [g['mapping'] for g in report['groups']]==['identity','reset0','reset1','toggle','identity','reset0','reset1','unresolved','identity','reset0','reset1']
    assert report['baseline_qualified'] and report['toggle_candidate_group_ids']==[3]
    assert len(report['rows'])==22


def test_actual_complete_path_file_contract(tiny):
    report=inspect_fixture(PATH_FIXTURE,auditor.PATH)
    assert report['status']=='audit_passed' and report['model_calls']==2907
    assert len(report['rows'])==6 and len(report['stages'])==9
    assert all(g['mapping']=='not_applicable' for g in report['groups'])
    assert [r['lane'] for r in report['rows']]==[0,1]*3


@pytest.mark.parametrize('directory,status,calls',[
    ('test_failure_category_and_unru0/failed','execution_failed',8),
    ('test_failure_category_and_unru1/failed','instrument_failed',8),
    ('test_failure_category_and_unru2/failed','incomplete',8),
    ('test_failure_category_and_unru3/failed','incomplete',8),
    ('test_failure_category_and_unru4/failed','incomplete',8),
    ('test_returned_invalid_output_i0/failed','instrument_failed',8),
    ('test_returned_invalid_output_i1/failed','instrument_failed',8),
    ('test_returned_invalid_output_i2/failed','instrument_failed',8),
    ('test_comparison_disagreement_s0/mismatch','instrument_failed',3),
    ('test_preflight_closes_every_sl0/preflight','preflight_passed',0),
    ('test_thread_startup_not_silent0/threads','instrument_failed',0)])
def test_retained_failures_and_unrun_rows(tiny,directory,status,calls):
    report=inspect_fixture(TOGGLE_FIXTURES/directory,auditor.TOGGLE)
    assert report['scientific_status']==status and report['model_calls']==calls
    assert len(report['rows'])==22 and all(row['decoded_bit'] is None for row in report['rows'])
    assert all(g['mapping']=='unavailable' for g in report['groups'])


@pytest.mark.parametrize('corruption',['row','carry','planned_control','probability_hash','missing_marker','scope'])
def test_rebound_corruption_rejected(tiny,tmp_path,corruption):
    original=TOGGLE_FIXTURES/'test_failure_category_and_unru0/failed'
    if not original.is_dir():pytest.skip('retained serializer not installed')
    copy=tmp_path/'copy';shutil.copytree(original,copy)
    result=json.loads((copy/'result.json').read_text())
    if corruption=='row':result['rows'][1]['source_reference_index']=0
    elif corruption=='carry':result['stages'][0]['initial']['array_sha256']='0'*64
    elif corruption=='planned_control':result['stages'][0]['controls']['alpha']+=.01
    elif corruption=='probability_hash':result['calls'][3]['output_sha256']='0'*64
    elif corruption=='missing_marker':(copy/'group-01.json').unlink()
    else:result['calls'][0]['group_id']=1
    (copy/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest=json.loads((copy/'manifest.json').read_text())
    manifest['files']=[dict(auditor.binding(p),path=p.name) for p in sorted(copy.iterdir()) if p.is_file() and p.name!='manifest.json']
    (copy/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with pytest.raises((ValueError,FileNotFoundError)):inspect_fixture(copy,auditor.TOGGLE)


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def add_test_closeout(local,ident):
    """Metadata-only closure of an already copied synthetic operator return."""
    manifest=json.loads((local/'final-manifest.json').read_text());mb=auditor.binding(local/'final-manifest.json')
    save(local/'transfer-closeout.json',dict(id=ident,state='complete',local_return_verified=True,
        reserved_bytes=auditor.TRANSFER_CAP,manifest_sha256=mb['sha256'],verified_manifest_entries=len(manifest['files'])))
    code=int((local/'exit-status.txt').read_text())
    complete=dict(status='returned_pending_scientific_audit' if code==0 else 'failed_run_evidence_returned',
        exit_status=code,files=len(manifest['files']),transfer_reservation_closed=True)
    save(local/'completion.json',complete)
    return manifest,complete


@pytest.mark.parametrize('index',range(6))
def test_actual_operator_archive_contracts(index,tmp_path):
    original=PAIR_FIXTURES/f'test_actual_incremental_and_co{index}/local'
    if not original.is_dir():pytest.skip('retained operator transport fixture not installed')
    local=tmp_path/'local';shutil.copytree(original,local)
    groups=[json.loads(p.read_text()) for p in sorted((local/'run').glob('group-*.json'))]
    add_test_closeout(local,local.name)
    e=auditor.Evidence(local/'run',groups[0]['profile'],synthetic=True)
    e.result=dict(groups=groups,status='execution_failed' if int((local/'exit-status.txt').read_text()) else 'completed')
    e.transport()
    assert e.return_status=='synthetic_transport_verified' and e.return_entries>len(groups)


@pytest.fixture
def paired_metadata(tmp_path):
    pairroot=tmp_path/'pair';pairroot.mkdir();jobs=[];proofs=[];completions=[]
    for index,profile in enumerate(auditor.PROFILES):
        original=PAIR_FIXTURES/f'test_actual_incremental_and_co{index}/local'
        if not original.is_dir():pytest.skip('retained operator fixture not installed')
        ident='synthetic-'+profile;local=tmp_path/ident;shutil.copytree(original,local)
        manifest,completion=add_test_closeout(local,ident);completions.append(completion)
        jobs.append(dict(name=ident,profile=profile,local=str(local),remote='/synthetic/remote/'+ident,
            checkout='/synthetic/checkout/'+ident,head='a'*40,slot=index))
        account=json.loads((local/'transfer-account.json').read_text())
        names=['local-return-verification.json','transfer-closeout.json','final-manifest.json','terminal-process-check.json','finished-utc.txt','exit-status.txt','transfer-account.json']
        proofs.append(dict(id=ident,profile=profile,local=str(local),manifest_sha256=auditor.binding(local/'final-manifest.json')['sha256'],files=len(manifest['files']),exit_status=completion['exit_status'],
            no_owned_numerical_processes=True,transfer_reservation_closed=True,charged_attempt_bytes=sum(x['bytes'] for x in account['attempts']),bindings=[auditor.binding(local/n) for n in names]))
    pair=dict(profile='public-core-pair',pair_id='synthetic-pair',jobs=jobs)
    save(pairroot/'pair.json',pair);pb=auditor.binding(pairroot/'pair.json')
    closure=dict(profile='public-core-pair',status='cleared',active=False,validation_id=pair['pair_id'],pair_manifest=pb,jobs=proofs,
        gate=dict(active=False,validation_id=pair['pair_id'],return_manifest_sha256_by_job={p['id']:p['manifest_sha256'] for p in proofs}),all_human_inputs_untouched=True)
    save(pairroot/'maintenance-closeout.json',closure)
    for j,c in zip(jobs,completions):
        local=Path(j['local']);save(local/'pair-context.json',pb);save(local/'maintenance-closeout.json',closure)
        c['maintenance']=dict(status='cleared',validation_id=pair['pair_id'],pair_manifest=pb,pair_closeout=auditor.binding(pairroot/'maintenance-closeout.json'))
        save(local/'completion.json',c)
    return jobs,completions,closure


def inspect_pair(jobs,completions,index=0):
    local=Path(jobs[index]['local']);manifest=json.loads((local/'final-manifest.json').read_text())
    e=auditor.Evidence(local/'run',jobs[index]['profile'],'a'*40,synthetic=True)
    e.result={'status':'completed'}
    e.paired_closure(auditor.binding(local/'final-manifest.json'),len(manifest['files']),completions[index])
    return e


def test_both_actual_return_proofs_bound_before_pair_release(paired_metadata):
    jobs,completions,_=paired_metadata
    for index in (0,1):
        e=inspect_pair(jobs,completions,index)
        assert e.pair_return['both_reservations_closed'] and len(e.pair_return['manifest_sha256_by_job'])==2
        assert sum(Path(p).name=='terminal-process-check.json' for p in e.bindings)==2


@pytest.mark.parametrize('defect,match',[
    ('other_process','pair proof identity'),('other_reservation','pair proof identity'),
    ('other_manifest','pair manifest identity'),('missing_second','both pair proofs'),
    ('gate_own_only','shared gate binds both'),('premature_completion','completion after shared'),
    ('wrong_other_source','paired frozen source'),('wrong_charge','paired transfer reserves')])
def test_pair_closure_rejects_incomplete_or_inconsistent_second_job(paired_metadata,defect,match):
    jobs,completions,closure=paired_metadata;local=Path(jobs[0]['local'])
    if defect=='other_process':closure['jobs'][1]['no_owned_numerical_processes']=False
    elif defect=='other_reservation':closure['jobs'][1]['transfer_reservation_closed']=False
    elif defect=='other_manifest':closure['jobs'][1]['manifest_sha256']='0'*64
    elif defect=='missing_second':closure['jobs'].pop()
    elif defect=='gate_own_only':closure['gate']['return_manifest_sha256_by_job'].pop(jobs[1]['name'])
    elif defect=='premature_completion':completions[0]['status']='return_verified/awaiting_pair_maintenance'
    elif defect=='wrong_charge':closure['jobs'][1]['charged_attempt_bytes']+=1
    else:
        ref=json.loads((local/'pair-context.json').read_text());p=Path(ref['path']);v=json.loads(p.read_text());v['jobs'][1]['head']='b'*40;save(p,v)
        new=auditor.binding(p)
        for job in jobs:save(Path(job['local'])/'pair-context.json',new)
    save(local/'maintenance-closeout.json',closure)
    with pytest.raises(ValueError,match=match):inspect_pair(jobs,completions)
