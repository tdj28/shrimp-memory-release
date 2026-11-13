"""Model-free causality, uncertainty, raw binding and failure inventory controls."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import analyze_causal_geometry as m
sys.path.pop(0)


def refs():
    a=np.zeros((2,6,896),dtype=np.float64)
    a[:,:,0]=np.arange(6);a[1,:,1]=1.
    return a


def orbit(label,steps=120,phase=0):
    return refs()[label,(np.arange(steps+1)+phase)%6].copy()


def dummy_prior(states):
    r=refs();obs=m.observe(states,r);final=m.gates.recompute(states[1:],r,0)
    return dict(outcome=final,horizon_diagnostics=dict(grid=[dict(steps=c['steps'],qualified_label=c['strict_label'],
        maximum_closure_residual=c['maximum_closure_residual'],reference_matches=c['reference_matches']) for c in obs['checkpoints']]),
        shadow=dict(decision=obs['strict_decision']),proposed_policy=dict(decision=obs['strict_decision']))


def test_fixed_plan_separates_horizons_and_no_execution_claim():
    p=m.plan();assert sum(c['rows'] for c in p['cohorts'])==66
    assert [c['updates'] for c in p['cohorts']]==[600]*5+[120]
    assert p['checkpoint_interval']==12 and p['consecutive_labels']==3
    assert p['limits']['threads']==1 and p['limits']['total_output_bytes']==8*1024**2
    assert p['limits']['working_output_bytes']==5*1024**2
    assert 'No achieved' in p['scope'] and 'already seen' in p['adaptation']


def test_streak_ties_and_later_changes_do_not_rewrite_first_decision():
    s=m.Streak()
    for step,label in [(12,'ad'),(24,'ad'),(36,None),(48,'td'),(60,'td')]:s.push(step,label)
    assert s.decision is None
    s.push(72,'td');assert s.decision=={'steps':72,'label':'td'}
    for step in (84,96,108):s.push(step,'ad')
    assert s.decision=={'steps':72,'label':'td'}


@pytest.mark.parametrize('step,label',[(0,'ad'),(11,'ad'),(12,'other')])
def test_bad_checkpoint_is_loud(step,label):
    with pytest.raises(ValueError):m.Streak().push(step,label)


def test_checkpoint_order_and_invalid_labels_cannot_advance_the_streak():
    s=m.Streak()
    with pytest.raises(ValueError):s.push(24,'ad')
    with pytest.raises(ValueError):s.push(12,'bad')
    s.push(12,'ad')
    with pytest.raises(ValueError):s.push(12,'ad')
    with pytest.raises(ValueError):s.push(36,'ad')
    s.push(24,'ad');s.push(36,'ad')
    assert s.decision==dict(steps=36,label='ad')


def test_exact_tie_erases_but_small_unequal_margin_does_not():
    x=refs()[0,0].copy();x[1]=.5
    assert m.geometry(x,refs())[0] is None
    x[1]=np.nextafter(.5,0.);label,d=m.geometry(x,refs())
    assert label=='ad' and d[0]<d[1]


@pytest.mark.parametrize('bad', ['dtype','shape','nonfinite'])
def test_geometry_invalid_inputs(bad):
    x=refs()[0,0].copy()
    if bad=='dtype':x=x.astype(np.float32)
    if bad=='shape':x=x[:-1]
    if bad=='nonfinite':x[3]=np.nan
    with pytest.raises(ValueError):m.geometry(x,refs())


def test_changed_future_never_changes_first_commitment():
    a=orbit(0);b=a.copy();b[37:]=orbit(1)[37:]
    first=m.observe(a,refs());second=m.observe(b,refs())
    assert first['checkpoints'][:3]==second['checkpoints'][:3]
    assert first['geometry_decision']==second['geometry_decision']=={'steps':36,'label':'ad'}
    assert first['strict_decision']==second['strict_decision']=={'steps':36,'label':'ad'}
    final=m.qualified(m.gates.recompute(b[1:],refs(),0));assert final=='td'
    score=m.score_decision(second['geometry_decision'],final,second['checkpoints'])
    assert score['outcome']=='wrong' and score['later_geometry_opposite_steps'][0]==48
    assert score['later_opposite_qualified_steps'][0]==48


def test_initial_state_excluded_from_checkpoint12_and_streak():
    a=orbit(0);b=a.copy();b[0]=999.
    assert m.observe(a,refs())==m.observe(b,refs())
    assert m.observe(a,refs())['checkpoints'][0]['steps']==12
    assert m.observe(a,refs())['geometry_decision']['steps']==36


def test_unqualified_strict_checkpoint_resets_only_strict_streak():
    a=orbit(0);a[30,4]=.01
    result=m.observe(a,refs())
    assert result['geometry_decision']==dict(steps=36,label='ad')
    assert result['checkpoints'][2]['strict_label'] is None
    assert result['strict_decision']==dict(steps=72,label='ad')


def test_unknown_final_retains_contradictions_without_becoming_wrong():
    a=orbit(0);a[37:]=orbit(1)[37:];a[-1,4]=.01
    obs=m.observe(a,refs());final=m.qualified(m.gates.recompute(a[1:],refs(),0));assert final is None
    score=m.score_decision(obs['geometry_decision'],final,obs['checkpoints'])
    assert score['outcome']=='unknown_final' and score['final_unknown']
    assert score['later_geometry_opposite_steps'] and score['later_opposite_qualified_steps']
    assert score['later_unqualified_steps']==[120]


def test_never_three_consistent_labels_means_no_decision():
    a=orbit(0)
    for block in range(10):a[block*12+1:(block+1)*12+1]=orbit(block%2,12)[1:]
    obs=m.observe(a,refs());assert obs['geometry_decision'] is None
    score=m.score_decision(None,None,obs['checkpoints'])
    assert score['outcome']=='no_decision' and score['final_unknown']


def test_period_one_and_ambiguous_reference_fail_strict_gate():
    a=np.repeat(refs()[0,:1],121,axis=0);obs=m.observe(a,refs())
    assert obs['geometry_decision']==dict(steps=36,label='ad') and obs['strict_decision'] is None
    duplicate=refs();duplicate[1]=duplicate[0]
    obs=m.observe(orbit(0),duplicate)
    assert obs['geometry_decision'] is None and obs['strict_decision'] is None


def test_saved_strict_decision_and_checkpoint_tampering_rejected():
    a=orbit(0);obs=m.observe(a,refs());prior=dummy_prior(a)
    m.validate_saved_strict(obs,prior)
    prior['shadow']['decision']=dict(steps=48,label='ad')
    with pytest.raises(ValueError,match='decision'):m.validate_saved_strict(obs,prior)
    prior=dummy_prior(a);prior['horizon_diagnostics']['grid'][0]['qualified_label']=None
    with pytest.raises(ValueError,match='checkpoint'):m.validate_saved_strict(obs,prior)


def test_saved_final_gate_tampering_rejected():
    a=m.gates.recompute(orbit(0)[1:],refs(),0);b=copy.deepcopy(a);m.validate_final(a,b)
    b['proper_divisors'][0]['excluded']=False
    with pytest.raises(ValueError,match='divisor'):m.validate_final(a,b)


def test_summary_qualified_and_unknown_denominators_separate():
    rows=[]
    for final,outcome in [('ad','correct'),('ad','wrong'),('ad','no_decision'),(None,'unknown_final'),(None,'no_decision')]:
        score=dict(outcome=outcome,final_unknown=final is None,later_geometry_opposite_steps=[],
            later_opposite_qualified_steps=[],later_unqualified_steps=[])
        rows.append(dict(status='completed',final_label=final,geometry_score=score,strict_score=score))
    s=m.summarize(rows);assert s['qualified_finals']==3 and s['unknown_finals']==2
    assert s['geometry']['correct_fraction_all_qualified']==1/3
    assert s['geometry']['correct_fraction_accepted_qualified']==.5
    assert s['geometry']['accepted_unknown']==1 and s['geometry']['no_decision_unknown']==1
    rows[0]['status']='failed'
    with pytest.raises(ValueError):m.summarize(rows)


def test_evidence_hash_symlink_missing_and_conflicting_contract(tmp_path):
    p=tmp_path/'data.json';p.write_text('{}');b=m.binding(p);e=m.Evidence(tmp_path)
    assert e.add(b)==p;e.read(p);p.write_text('[]')
    with pytest.raises(ValueError,match='hash'):e.read(p)
    with pytest.raises(ValueError,match='unbound'):e.read(tmp_path/'other')
    p.write_text('{}');q=tmp_path/'link';q.symlink_to(p)
    with pytest.raises(ValueError,match='symlink'):e.add(dict(b,path=str(q)))
    with pytest.raises(ValueError,match='conflicting'):e.add(dict(b,bytes=9))


def write_row(path,states):
    n=len(states)-1;np.savez(path,states=states,token_ids=np.ones(n,dtype=np.int64),step_seconds=np.ones(n,dtype=np.float64),
        first_state_index=np.array(0,dtype=np.int64),alpha=np.array(.07293697809562831),log10_temperature=np.array(-.15395151820007266))


def test_actual_recovery_filename_reaches_analysis_and_unknown_is_preserved(tmp_path):
    a=orbit(0);a[-1,4]=.01;p=tmp_path/'trial-00-final-120.npz';write_row(p,a)
    item=dict(row_id=0,path=p,prior=dummy_prior(a));result=m.analyze_row(item,120,refs())
    assert result['final_label'] is None and result['geometry_score']['outcome']=='unknown_final'
    assert result['trajectory']['path'].endswith('trial-00-final-120.npz')


def test_actual_primary_filename_and_saved_grid_contract(tmp_path):
    a=orbit(0,600);p=tmp_path/'trial-00-readback-final-600.npz';write_row(p,a)
    item=dict(row_id=0,path=p,prior=dummy_prior(a));result=m.analyze_row(item,600,refs())
    assert result['strict_decision']==dict(steps=36,label='ad')
    assert len(result['checkpoints'])==50 and result['final_label']=='ad'
    assert 'classification' not in result['final_qualification']


def test_saved_strict_gate_failure_preserves_complete_observation(tmp_path):
    a=orbit(0,600);p=tmp_path/'trial-00-readback-final-600.npz';write_row(p,a)
    prior=dummy_prior(a);prior['shadow']['decision']=dict(steps=48,label='ad')
    with pytest.raises(ValueError,match='decision') as exc:
        m.analyze_row(dict(row_id=0,path=p,prior=prior),600,refs())
    partial=exc.value.partial_row
    assert partial['status']=='failed' and len(partial['partial_observation']['checkpoints'])==50
    assert partial['partial_observation']['strict_decision']==dict(steps=36,label='ad')


def synthetic_row(i):
    c=[dict(steps=12,geometry_label='ad',strict_label='ad')]
    score=m.score_decision(None,'ad',c)
    return dict(row_id=i,status='completed',final_label='ad',geometry_score=score,strict_score=score)


def fake_loader(name,count,steps,root,evidence):return refs(),[dict(row_id=i) for i in range(count)]


def fake_run(tmp_path,**kwargs):
    return m.run(tmp_path,tmp_path,tmp_path/'out',loader=fake_loader,
        row_analyzer=kwargs.pop('row_analyzer',lambda item,*rest:synthetic_row(item['row_id'])),
        clock=kwargs.pop('clock',lambda:0.),rss_reader=kwargs.pop('rss_reader',lambda:0),admit=lambda:{'synthetic':True},
        freeze=lambda root:{'synthetic':True},install_signal=False,**kwargs)


def test_complete66_slot_filename_manifest_contract(tmp_path):
    result=fake_run(tmp_path);out=tmp_path/'out'
    assert result['status']=='completed' and sum(len(c['rows']) for c in result['cohorts'])==66
    assert len(list(out.glob('*-row-*.json')))==66
    assert json.loads((out/'result.json').read_text())==result
    manifest=json.loads((out/'manifest.json').read_text())
    assert {Path(x['path']).name for x in manifest['files']}=={p.name for p in out.iterdir()}-{'manifest.json'}
    for x in manifest['files']:assert m.binding(x['path'])==x


def test_failure_after_saved_row_preserves_every_unstarted_slot(tmp_path):
    def fail(item,*rest):
        if item['row_id']==1:raise ValueError('injected row error')
        return synthetic_row(0)
    r=fake_run(tmp_path,row_analyzer=fail)
    assert r['status']=='failed' and r['cohorts'][0]['rows'][0]['status']=='completed'
    assert r['cohorts'][0]['rows'][1]['status']=='failed'
    assert r['cohorts'][0]['rows'][2]['status']=='not_started'
    assert sum(len(c['rows']) for c in r['cohorts'])==66
    assert len(list((tmp_path/'out').glob('*-row-*.json')))==1


def test_boundary_timeout_preserves_earlier_shards(tmp_path):
    current=[0.]
    def analyze(item,*rest):
        if item['row_id']==1:current[0]=120.
        return synthetic_row(item['row_id'])
    r=fake_run(tmp_path,row_analyzer=analyze,clock=lambda:current[0])
    assert r['status']=='failed' and 'TimeoutError' in r['error']
    assert r['cohorts'][0]['rows'][0]['status']=='completed'
    assert r['cohorts'][0]['rows'][1]['status']=='completed'


def test_memory_limit_preserves_all_planned_rows(tmp_path):
    r=fake_run(tmp_path,rss_reader=lambda:m.MAX_RSS)
    assert r['status']=='failed' and 'MemoryError' in r['error']
    assert sum(len(c['rows']) for c in r['cohorts'])==66


def test_writer_exclusive_and_working_cap_leaves_finalization_room(tmp_path,monkeypatch):
    w=m.Writer(tmp_path/'out');monkeypatch.setattr(m,'WORKING_OUTPUT',8)
    with pytest.raises(ValueError,match='cap'):w.json('too-large.json',{'abcdefgh':'x'})
    w.json('result.json',{'status':'failed'},final=True)
    with pytest.raises(FileExistsError):m.Writer(tmp_path/'out')


def test_timeout_midtrajectory_retains_observed_prefix(tmp_path):
    a=orbit(0);p=tmp_path/'trial-00-final-120.npz';write_row(p,a);calls=[0]
    def check():
        calls[0]+=1
        if calls[0]==30:raise TimeoutError('partial stop')
    with pytest.raises(TimeoutError) as exc:
        m.analyze_row(dict(row_id=0,path=p,prior=dummy_prior(a)),120,refs(),check)
    partial=exc.value.partial_row
    assert partial['status']=='failed' and partial['partial_observation']['consumed_updates']==29
    assert [c['steps'] for c in partial['partial_observation']['checkpoints']]==[12,24]
    assert partial['partial_observation']['geometry_decision'] is None


def test_partial_row_shard_and_all_future_slots_retained(tmp_path):
    def partial(item,*rest):
        exc=TimeoutError('partial');exc.partial_row=dict(row_id=0,status='failed',partial_observation={'consumed_updates':29})
        raise exc
    r=fake_run(tmp_path,row_analyzer=partial);slot=r['cohorts'][0]['rows'][0]
    assert slot['status']=='failed'
    saved=json.loads(Path(slot['artifact']['path']).read_text())
    assert saved['partial_observation']['consumed_updates']==29
    assert r['cohorts'][0]['rows'][1]['status']=='not_started'
    assert sum(len(c['rows']) for c in r['cohorts'])==66


def make_cohort(tmp_path,monkeypatch,steps=600):
    name='fresh-readout' if steps==600 else 'p6-recovery';root=tmp_path/'repo';data=tmp_path/'data';base=data/'audits'/name
    (root/'docs/receipts').mkdir(parents=True);(base/'run').mkdir(parents=True)
    rp=base/'reference-cycles.npz';np.savez(rp,states=refs());ref=m.binding(rp);monkeypatch.setattr(m,'REFERENCE_SHA',ref['sha256'])
    rows=[];bindings=[ref]
    for i in range(2):
        filename=f'trial-{i:02d}-readback-final-600.npz' if steps==600 else f'trial-{i:02d}-final-120.npz'
        p=base/'run'/filename;write_row(p,orbit(i,steps));b=m.binding(p);bindings.append(b)
        row=dict(packet_index=i,status='verified') if steps==600 else dict(trial_id=i,status='verified',integrity_passed=True,producer_agreement=True,bindings=[b])
        rows.append(row)
    if steps==600:
        audit=dict(status='audit_passed',verified_rows=2,rows=rows,bindings=bindings)
        receipt=dict(status='audit_passed',verified_rows=2)
    else:
        context=base/'run/results.json';context.write_text('{}')
        audit=dict(status='complete_audit_passed',complete_audit_passed=True,rows=rows,reference_binding=ref,
            input_context=dict(bindings=[]),final_results=dict(binding=m.binding(context)))
        receipt=dict(independent_audit_status='complete_audit_passed')
    ap=base/'complete-audit.json';ap.write_text(json.dumps({'audit':audit}));receipt['bindings']=[m.binding(ap)]
    rec=root/'docs/receipts'/f'{name}.json';rec.write_text(json.dumps(receipt))
    return name,root,data,base,rec,ap


@pytest.mark.parametrize('steps',[600,120])
def test_actual_loader_schema_and_filename_contract(tmp_path,monkeypatch,steps):
    name,root,data,base,rec,ap=make_cohort(tmp_path,monkeypatch,steps)
    e=m.Evidence(data);r,rows=m.load_cohort(name,2,steps,root,e)
    assert np.array_equal(r,refs()) and [x['row_id'] for x in rows]==[0,1]
    assert all(x['path'].exists() for x in rows)
    assert str(ap) in e.items and str(rows[0]['path']) in e.items


@pytest.mark.parametrize('fault',['missing','reordered','unverified','changed'])
def test_loader_rejects_bad_inventory_even_with_rehashed_audit(tmp_path,monkeypatch,fault):
    name,root,data,base,rec,ap=make_cohort(tmp_path,monkeypatch)
    d=json.loads(ap.read_text())
    if fault=='missing':d['audit']['rows'].pop()
    if fault=='reordered':d['audit']['rows'].reverse()
    if fault=='unverified':d['audit']['rows'][0]['status']='unverified'
    if fault=='changed':(base/'run/trial-00-readback-final-600.npz').write_bytes(b'changed')
    ap.write_text(json.dumps(d));r=json.loads(rec.read_text());r['bindings']=[m.binding(ap)];rec.write_text(json.dumps(r))
    with pytest.raises(ValueError):m.load_cohort(name,2,600,root,m.Evidence(data))


@pytest.mark.parametrize('reference_fault',['identical_three','distinct_path','conflicting_bytes'])
def test_repeated_reference_inventory_contract(tmp_path,monkeypatch,reference_fault):
    name,root,data,base,rec,ap=make_cohort(tmp_path,monkeypatch)
    audit=json.loads(ap.read_text());reference=copy.deepcopy(audit['audit']['bindings'][0])
    audit['audit']['bindings'].extend([copy.deepcopy(reference),copy.deepcopy(reference)])
    if reference_fault=='distinct_path':
        other=base/'distinct-reference.npz';other.write_bytes(Path(reference['path']).read_bytes())
        audit['audit']['bindings'][-1]=m.binding(other)
    elif reference_fault=='conflicting_bytes':audit['audit']['bindings'][-1]['bytes']+=1
    ap.write_text(json.dumps(audit));receipt=json.loads(rec.read_text())
    receipt['bindings']=[m.binding(ap)];rec.write_text(json.dumps(receipt))
    evidence=m.Evidence(data)
    if reference_fault=='identical_three':
        actual,rows=m.load_cohort(name,2,600,root,evidence)
        assert np.array_equal(actual,refs()) and len(rows)==2
        assert sum(x['sha256']==reference['sha256'] for x in evidence.items.values())==1
    else:
        with pytest.raises(ValueError,match='ambiguous reference|conflicting evidence'):
            m.load_cohort(name,2,600,root,evidence)


def test_source_freeze_covers_six_receipts_and_rejects_changed_receipt(tmp_path,monkeypatch):
    relative=['tools/analyze_causal_geometry.py','tools/audit_p6_recovery.py','tools/streaming_readout.py',
        'docs/causal-geometry-readout.md','tests/test_causal_geometry.py','pyproject.toml','uv.lock']
    relative += ['docs/receipts/'+name+'.json' for name in ('fresh-readout','transverse-readout',
        'transverse-intermediate','transverse-amplitude','transverse-direction2','p6-recovery')]
    snapshot={name:('frozen '+name).encode() for name in relative}
    for name,blob in snapshot.items():
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob)
    monkeypatch.setattr(m,'__file__',str(tmp_path/relative[0]))
    monkeypatch.setattr(m.gates,'__file__',str(tmp_path/relative[1]))
    monkeypatch.setattr(m.streaming_readout,'__file__',str(tmp_path/relative[2]))
    calls=[]
    def git(command,**kwargs):
        calls.append(command)
        if command[-2:]==['rev-parse','HEAD']:return 'a'*40+'\n'
        assert command[:4]==['git','-C',str(tmp_path),'show']
        commit,name=command[-1].split(':',1);assert commit=='a'*40
        return snapshot[name]
    monkeypatch.setattr(m.subprocess,'check_output',git)
    result=m.source_bindings(tmp_path)
    assert {item['relative_path'] for item in result['files']}==set(relative)
    assert len(result['files'])==13 and len(calls)==14
    (tmp_path/'docs/receipts/p6-recovery.json').write_text('changed but syntactically irrelevant')
    with pytest.raises(ValueError,match='not frozen: docs/receipts/p6-recovery.json'):
        m.source_bindings(tmp_path)


def test_receipt_cannot_change_between_source_freeze_and_loader(tmp_path,monkeypatch):
    name,root,data,base,rec,ap=make_cohort(tmp_path,monkeypatch)
    evidence=m.Evidence(data);evidence.add(m.binding(rec))
    receipt=json.loads(rec.read_text());receipt['unrelated_field']='new bytes after source freeze'
    rec.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='conflicting evidence binding'):
        m.load_cohort(name,2,600,root,evidence)


def test_run_registers_frozen_receipt_before_loader_access(tmp_path):
    receipt=tmp_path/'frozen-receipt.json';receipt.write_text('{}');frozen=m.binding(receipt)
    def mutate(name,count,steps,root,evidence):
        assert str(receipt) in evidence.items
        receipt.write_text('[]');evidence.add(m.binding(receipt))
    result=m.run(tmp_path,tmp_path,tmp_path/'out',loader=mutate,row_analyzer=lambda *args:None,
        clock=lambda:0.,rss_reader=lambda:0,admit=lambda:{'synthetic':True},
        freeze=lambda root:{'files':[frozen]},install_signal=False)
    assert result['status']=='failed' and 'conflicting evidence binding' in result['error']
    assert all(row['status']=='not_started' for c in result['cohorts'] for row in c['rows'])
