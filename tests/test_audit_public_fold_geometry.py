"""Model-free contract and corruption checks; no numerical model is loaded."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import audit_public_fold_geometry as checker
import public_fold_map as fm
import public_pseudoarclength as palc
import run_public_fold_geometry as producer

PRECISION={'norm':{k:'torch.float64' for k in ('input','reduction','weight','output')},
           'rotary':{k:'torch.float64' for k in ('input','inverse_frequency','angles','cos','sin')}}


def binding(path):
    p=Path(path).resolve()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=checker.sha(p))


def packet(base):
    base.mkdir()
    states=np.zeros((120,896),dtype=np.float64)
    points=[dict(theta=[-.1536,.7293697809562831]) for _ in range(120)]
    curve=dict(args=dict(alpha_scale=10.),points=points)
    (base/'curve.json').write_text(json.dumps(curve))
    np.save(base/'states.npy',states)
    cycles=np.zeros((6,2,896),dtype=np.float64)
    for phase in range(6):cycles[phase,:,phase]=.01
    cycles[:,1,10]=.02
    np.savez(base/'p.npz',source_cycle_states=cycles,source_terminal_states=cycles[-1])
    np.savez(base/'h.npz',states=cycles.transpose(1,0,2))
    (base/'packet.json').write_text('{}')
    np.savez(base/'inputs.npz',fold_states=states[[21,22]],fold_alphas=np.array([.07293697809562831]*2),
             fold_logtemps=np.array([-.1536]*2),p_terminal=cycles[-1],
             p_cycles=np.concatenate((cycles[-1:],cycles[:-1])),h_cycles=cycles.transpose(1,0,2),
             state_scale=np.array(np.linalg.norm(cycles[-1,1]-cycles[-1,0])))
    sources=dict(fold_curve=binding(base/'curve.json'),fold_states=binding(base/'states.npy'),
                 p_packet=binding(base/'packet.json'),p_states=binding(base/'p.npz'),h_states=binding(base/'h.npz'))
    doc=dict(schema=1,sources=sources,arrays=binding(base/'inputs.npz'),parent_indices=[21,22],
             p_phase_order=[480,475,476,477,478,479],model_calls=0)
    (base/'inputs.json').write_text(json.dumps(doc))
    return base/'inputs.json'


class ConstantFakeMap:
    """An intentionally non-period-six instrument, using actual recorder files."""
    def __init__(self,args,rec,counts):
        self.rec=rec;self.counts=counts;self.alpha=fm.ALPHA
        counts.claim('native64_prefix')
        rec.event('prefix_ready',mode='native64',call=counts.calls,seconds=0.,length=37)
        rec.event('alpha_selected',alpha=self.alpha)
    def cycle(self,x,temp,*,label,steps=6,**kwargs):
        for _ in range(steps):self.counts.claim('fake')
        states=np.repeat(np.asarray(x)[None],steps+1,axis=0)
        top=np.repeat(np.array([np.arange(5),np.linspace(.4,.1,5)])[None],steps,axis=0)
        name=self.rec.arrays(label,states=states,tangents=np.array([]),alpha=np.array(self.alpha),
                             logtemp=np.array(temp),dlogtemp=np.array(0.),top5=top)
        self.rec.event('cycle',file=name,label=label,steps_completed=steps,differentiated=False,cached=True,
                       seconds=0.,failed=None,precision=[PRECISION]*steps)
        return states,None


@pytest.fixture(scope='module')
def actual_failed_run(tmp_path_factory):
    base=tmp_path_factory.mktemp('fold-audit-actual-failure')
    pin=packet(base/'packet'); run=base/'run'
    args=SimpleNamespace(input_packet=pin,input_packet_sha256=checker.sha(pin),vectors=base/'unused.npz',
                         cache_dir=base/'unused-cache',output_dir=run,preflight_only=False,pilot_only=False)
    old=fm.Map;fm.Map=ConstantFakeMap
    # This is an explicitly synthetic source-contract fixture while the owned
    # producer/checker edits are uncommitted. All real source bytes are still
    # recorded; only its local Git cleanliness admission is stubbed here.
    real_check=subprocess.check_output
    def source_check(command,*a,**kw):
        if command[:3]==['git','status','--porcelain']:return '' if kw.get('text') else b''
        return real_check(command,*a,**kw)
    try:
        with patch.object(producer.subprocess,'check_output',side_effect=source_check):exitcode=producer.run(args)
    finally:fm.Map=old
    assert exitcode==1
    return dict(run=run,pin=pin,commit=json.loads((run/'source.json').read_text())['commit'],root=Path(producer.ROOT))


def test_actual_runner_failure_retains_category_without_pulse_admission(actual_failed_run):
    f=actual_failed_run
    report=checker.audit(f['run'],f['root'],f['commit'],f['pin'],checker.sha(f['pin']),synthetic=True)
    assert report['status']=='audit_passed'
    assert report['scientific_status']=='instrument_failed'
    assert report['counts']['ordinary']==13 and report['counts']['model_invocations']==13
    assert not report['pulse_eligible'] and report['closed_verified_return'] is False


def test_real_primitive_gate_uses_all_six_phases_and_divisors():
    trace=np.zeros((13,896)); trace[:,0]=np.arange(13)%6
    assert checker.cycle_metrics(trace)['passed']
    trace[10,1]=2e-10
    assert not checker.cycle_metrics(trace)['passed']
    trace[:]=0
    assert not checker.cycle_metrics(trace)['passed']


@pytest.mark.parametrize('name',['../x','a/../b','/x','a//b','./x',''])
def test_artifact_paths_fail_closed(name):
    with pytest.raises(checker.AuditError):checker.safe_name(name)


def test_typed_array_envelope_requires_actual_values():
    value=palc.serializable(np.array([1.,2.]))
    np.testing.assert_array_equal(checker.decode(value),[1.,2.])
    value['values'][1]=3.
    with pytest.raises(checker.AuditError,match='typed array'):checker.decode(value)


def test_manifest_rejects_late_raw_artifact(actual_failed_run,tmp_path):
    import shutil
    dest=tmp_path/'run';shutil.copytree(actual_failed_run['run'],dest)
    (dest/'unlisted.json').write_text('{}')
    with pytest.raises(checker.AuditError,match='complete flat inventory'):checker.Audit(dest,synthetic=True).inventory()


def test_signed_derivative_orders_do_not_invent_zero_precision():
    rows=[dict(h=2.**-k,relative_error=1e-4/4**k) for k in range(5)]
    assert checker.derivative_gate(rows)['passed']
    rows[-1]['relative_error']=0
    gate=checker.derivative_gate(rows)
    assert gate['orders'][-1]['order'] is None
    for row in rows:row['relative_error']=0
    assert not checker.derivative_gate(rows)['passed']


class RecordedToyMap:
    """Synthetic Phi action, saved as six-step records solely for algebra tests."""
    def __init__(self,rec):
        self.rec=rec;self.counts=fm.Counts(rec);self.alpha=fm.ALPHA;self.scale=.02
        rec.event('alpha_selected',alpha=self.alpha)
    def cycle(self,x,temp,*,v=None,dlogtemp=0.,label,steps=6):
        assert steps==6
        self.last_cycle_file=None
        u=x/self.scale;eta=(temp-fm.H)/fm.DELTA
        r=-.5*u;r[0]=u[0]**2+eta
        last=x+self.scale*r
        states=np.stack([x+(last-x)*i/6 for i in range(7)])
        if v is None:
            tangents=np.array([])
            for _ in range(6):self.counts.claim('toy')
        else:
            self.counts.claim_kind('jvp_cycles');self.counts.calls+=6
            j=.5*v;j[0]=(1+2*u[0])*v[0]+self.scale*dlogtemp/fm.DELTA
            tangents=np.stack([v+(j-v)*i/6 for i in range(7)])
        top=np.repeat(np.array([np.arange(5),np.linspace(.4,.1,5)])[None],6,axis=0)
        name=self.rec.arrays(label,states=states,tangents=tangents,alpha=np.array(self.alpha),logtemp=np.array(temp),
                             dlogtemp=np.array(dlogtemp),top5=top)
        self.last_cycle_file=name
        self.rec.event('cycle',file=name,label=label,steps_completed=6,differentiated=v is not None,cached=True,
                       seconds=0.,failed=None,precision=[PRECISION]*6)
        return states,None if v is None else tangents


@pytest.fixture(scope='module')
def recorded_extended_solver(tmp_path_factory):
    base=tmp_path_factory.mktemp('fold-audit-extended')
    rec=fm.Record(base/'run');model=RecordedToyMap(rec);scaled=fm.Scaled(model,model.scale)
    c=np.zeros(896);c[0]=1.
    # Production initializes v from an explicitly normalized critical vector.
    # A retained first fixture deliberately used .9 and exhausted this tiny
    # unpreconditioned Krylov subspace; that unresolved counterexample is kept.
    seed=np.zeros(1793);seed[0]=-.02;seed[896]=-.001;seed[897]=1.
    rec.arrays('interpolated-seed',z=seed,c=c,weight=np.array(.5))
    f=fm.Fold(scaled,c)
    result=palc.solve_system(f.residual,f.action,seed,config=palc.NewtonConfig(residual_tolerance=1e-8),
        residual_blocks={'orbit':(0,896,1e-10/model.scale),'null':(896,1792,1e-8),'normalization':(1792,1793,1e-10)},
        event=lambda event:rec.event('fold-solver',stage='primary',solver_event=event))
    rec.json('primary-solver.json',palc.serializable(result));rec.stream.close()
    assert result['status']=='converged'
    a=checker.Audit(base/'run',synthetic=True);a.scale=model.scale;a.normalization=c;a.events()
    return a,result


def test_actual_extended_solver_and_raw_action_contract(recorded_extended_solver):
    a,result=recorded_extended_solver
    a.solver('fold-solver:primary',result)
    assert len(result['calls'])>3


@pytest.mark.parametrize('part',['orbit','null','normalization','fresh_linear','gmres_info','damping'])
def test_solver_corruptions_cannot_pass(recorded_extended_solver,part):
    a,original=recorded_extended_solver
    result=copy.deepcopy(original)
    if part in ('orbit','null','normalization'):
        index={'orbit':0,'null':896,'normalization':1792}[part]
        result['residual'][index]+=1e-3
    elif part=='fresh_linear':result['iterations'][0]['linear']['fresh_action'][0]+=.1
    elif part=='gmres_info':result['iterations'][0]['linear']['gmres_info']=1
    else:result['iterations'][0]['trials'][0]['damping']=.75
    with pytest.raises(checker.AuditError):a.solver('fold-solver:primary',result)


def test_saved_callback_cannot_be_rebound_to_another_state(recorded_extended_solver):
    original,result=recorded_extended_solver;a=copy.deepcopy(original)
    called=next(q for q in a.solver_calls['fold-solver:primary'].values() if q['cycles'])
    called['cycles'][0]['z']['states'][0,3]+=1e-5
    with pytest.raises(checker.AuditError,match='initial state'):a.solver('fold-solver:primary',result)


def test_hermite_uses_state_as_well_as_temperature(tmp_path):
    result=palc.continue_branch(lambda y:np.array([y[0]**2+y[1]]),
        lambda y,d:np.array([2*y[0]*d[0]+d[1]]),np.array([-.2,-.04]),np.array([0.,1.]),
        config=palc.PACConfig(step_sizes=(.075,.0375),max_accepted_steps=6),stop_on_turn=True)
    a=checker.Audit(tmp_path,synthetic=True);row=a.turn(result);expected=producer.turn_estimate(result)
    a.equal(expected,row,'independent Hermite formula')
    assert abs(row['scaled_state_estimate'][0])<1e-4
    changed=copy.deepcopy(result)
    changed['points'][-1]['tangent'][-1]=0
    with pytest.raises(checker.AuditError,match='orientation'):a.turn(changed)


def test_normalization_gate_is_not_orbit_gate(tmp_path):
    a=checker.Audit(tmp_path,synthetic=True)
    config=dict(residual_tolerance=1e-8)
    r=np.zeros(1793);r[-1]=2e-10
    result=a.gates(r,{'orbit':(0,896,1e-10),'null':(896,1792,1e-8),'normalization':(1792,1793,1e-10)},config)
    assert result['full_passed'] and result['blocks']['orbit']['passed'] and not result['passed']


def test_all_cyclic_native_comparisons_are_retained():
    a=np.arange(6*896,dtype=np.float64).reshape(6,896)
    b=np.roll(a,-2,axis=0)
    distances=checker.cyclic_distances(a,b)
    assert distances.shape==(6,) and distances.min()==0 and distances[0]>0


def test_counters_reject_unreported_extra_action(actual_failed_run):
    f=actual_failed_run;a=checker.Audit(f['run'],synthetic=True)
    a.result=a.read('result.json');a.events()
    a.counters()
    a.result['counts']['jvp_cycles']=2
    with pytest.raises(checker.AuditError,match='JVP count'):a.counters()


@pytest.fixture(scope='module')
def recorded_branch(tmp_path_factory):
    base=tmp_path_factory.mktemp('fold-audit-branch')
    rec=fm.Record(base/'run');model=RecordedToyMap(rec);scaled=fm.Scaled(model,model.scale)
    start=np.zeros(897);start[0]=-.02;start[-1]=-.0004
    border=np.zeros(897);border[-1]=1.
    config=palc.PACConfig(equation_tolerance=1e-10/model.scale,arc_tolerance=1e-10,residual_tolerance=1e-8,
                         step_sizes=(.008,.004,.002),max_attempted_steps=16,max_accepted_steps=12,max_rejected_steps=4)
    result=palc.continue_branch(scaled.residual,scaled.action,start,border,config=config,stop_on_turn=True,
                               event=lambda event:rec.event('branch-solver',solver_event=event))
    rec.json('p1-branch.json',palc.serializable(result));rec.stream.close()
    assert result['status']=='completed' and result['reason']=='qualified_tangent_turn'
    a=checker.Audit(base/'run',synthetic=True);a.scale=model.scale;a.events()
    return a,result


def test_actual_continuation_has_fresh_bordered_actions_and_arc_gates(recorded_branch):
    a,result=recorded_branch;a.solver('branch-solver',result)
    assert len(result['points'])>2 and len(result['turn_brackets'])==1
    a.equal(producer.turn_estimate(result),a.turn(result),'actual branch independent Hermite')


@pytest.mark.parametrize('part',['arc','predictor','tangent','correction','orientation','turn','counts'])
def test_branch_corruptions_fail_closed(recorded_branch,part):
    a,original=recorded_branch;result=copy.deepcopy(original);attempt=result['attempts'][0]
    if part=='arc':attempt['corrector']['residual'][-1]=2e-10
    elif part=='predictor':attempt['predictor'][0]+=.001
    elif part=='tangent':attempt['tangent_update']['normalized_action'][0]=1e-4
    elif part=='correction':attempt['correction_distance_passed']=False
    elif part=='orientation':attempt['tangent_update']['orientation']=-1.
    elif part=='turn':result['turn_brackets'][0]['point_indices']=[0,1]
    else:result['accepted_steps']+=1
    with pytest.raises(checker.AuditError):a.solver('branch-solver',result)


class RecordedDiagonalMap(RecordedToyMap):
    def cycle(self,x,temp,*,v=None,dlogtemp=0.,label,steps=6):
        assert v is not None and dlogtemp==0 and steps==6
        diagonal=np.tile(np.array([1.,.9,.8,.7,.6,.5,.4,.3]),112)
        self.counts.claim_kind('jvp_cycles');self.counts.calls+=6
        states=np.repeat(np.asarray(x)[None],7,axis=0)
        j=diagonal*v;tangents=np.stack([v+(j-v)*i/6 for i in range(7)])
        top=np.repeat(np.array([np.arange(5),np.linspace(.4,.1,5)])[None],6,axis=0)
        name=self.rec.arrays(label,states=states,tangents=tangents,alpha=np.array(self.alpha),
                             logtemp=np.array(temp),dlogtemp=np.array(dlogtemp),top5=top)
        self.rec.event('cycle',file=name,label=label,steps_completed=6,differentiated=True,cached=True,
                       seconds=0.,failed=None,precision=[PRECISION]*6)
        return states,tangents


@pytest.fixture(scope='module')
def recorded_spectrum(tmp_path_factory):
    base=tmp_path_factory.mktemp('fold-audit-spectrum')
    rec=fm.Record(base/'run');model=RecordedDiagonalMap(rec);scaled=fm.Scaled(model,model.scale)
    y=np.zeros(897);y[-1]=.75
    vector,result=producer.spectrum(scaled,y,'target-fold',seed=1)
    rec.stream.close();a=checker.Audit(base/'run',synthetic=True);a.scale=model.scale;a.events();a.spectra={}
    return a,y,vector,result


def test_actual_arnoldi_file_contract_and_fresh_full_space_pairs(recorded_spectrum):
    a,y,vector,result=recorded_spectrum
    actual,row=a.spectral('target-fold',y,checker.ALPHA,1)
    np.testing.assert_allclose(actual,vector,atol=2e-14,rtol=0)
    a.equal(result,row,'actual spectral output')
    assert row['basis_passed'] and len(row['pairs'])==3 and all(r['qualified'] for r in row['pairs'])


@pytest.mark.parametrize('part',['basis','hessenberg','matvec','fresh','control','missing_fresh'])
def test_spectral_corruptions_fail_without_projected_acceptance(recorded_spectrum,tmp_path,part):
    import shutil
    original,y,_,_=recorded_spectrum;dest=tmp_path/'run';shutil.copytree(original.run,dest)
    if part in ('basis','hessenberg','matvec'):
        path=next(dest.glob('*-target-fold-arnoldi.npz'))
        with np.load(path) as z:arrays={k:z[k].copy() for k in z.files}
        key={'basis':'q','hessenberg':'h','matvec':'matvecs'}[part];arrays[key][0,0]+=.001
        np.savez(path,**arrays)
    elif part=='fresh':
        path=sorted(dest.glob('*-target-fold-eigenpair.npz'))[0]
        with np.load(path) as z:arrays={k:z[k].copy() for k in z.files}
        arrays['action'][0]+=.001;np.savez(path,**arrays)
    elif part=='control':
        path=next(dest.glob('*-target-fold-action.npz'))
        with np.load(path) as z:arrays={k:z[k].copy() for k in z.files}
        arrays['logtemp']+=1e-4;np.savez(path,**arrays)
    else:sorted(dest.glob('*-target-fold-eigenpair.npz'))[-1].unlink()
    a=checker.Audit(dest,synthetic=True);a.scale=original.scale;a.events();a.spectra={}
    with pytest.raises(checker.AuditError):a.spectral('target-fold',y,checker.ALPHA,1)


def transport_fixture(base,source,context=None):
    """Actual flat/archive/closure fields, with no external transport actions."""
    import shutil
    import tarfile
    job=base/'job';job.mkdir();shutil.copytree(source,job/'run')
    def write(name,value):
        (job/name).write_text(json.dumps(value))
    (job/'exit-status.txt').write_text('1\n')
    (job/'started-utc.txt').write_text('2026-10-30T00:00:00Z\n')
    if context is not None:
        for p in Path(context).iterdir():
            if p.is_file():shutil.copy2(p,job/p.name)
    write('terminal-process-check.json',dict(no_owned_numerical_processes=True,remaining_numerical_processes=[]))
    rows=[{**binding(p),'path':str(p.relative_to(job))} for p in sorted(job.rglob('*')) if p.is_file()]
    with tarfile.open(job/'final-artifacts.tar','w:') as archive:
        for row in rows:archive.add(job/row['path'],arcname=row['path'],recursive=False)
    ar={**binding(job/'final-artifacts.tar'),'path':'final-artifacts.tar'}
    write('final-manifest.json',dict(files=rows,transferred_files=rows,already_local_files=0,archive=ar))
    mh=checker.sha(job/'final-manifest.json')
    charged=ar['bytes']+(job/'final-manifest.json').stat().st_size
    write('local-return-verification.json',dict(status='verified',manifest_sha256=mh,files=len(rows)))
    account=dict(reserved_bytes=12*2**30,overhead_reserve=32*2**20,attempts=[dict(bytes=charged)])
    if context is not None:
        doc=json.loads((job/'case-context.json').read_text())
        account.update(reserved_bytes=doc['remaining_limits']['transfer_bytes'],case_context_sha256=checker.sha(job/'case-context.json'),
                       case_prior_charged_transfer_bytes=doc['prior_usage']['transfer_bytes'])
    write('transfer-account.json',account)
    write('transfer-closeout.json',dict(id=job.name,state='complete',local_return_verified=True,manifest_sha256=mh,
        verified_manifest_entries=len(rows),reserved_bytes=account['reserved_bytes'],charged_transfer_bytes=charged,
        transfer_account_sha256=checker.sha(job/'transfer-account.json'),terminal_process_sha256=checker.sha(job/'terminal-process-check.json')))
    if context is not None:
        closed=json.loads((job/'transfer-closeout.json').read_text())
        closed.update(case_context_sha256=account['case_context_sha256'],case_prior_charged_transfer_bytes=account['case_prior_charged_transfer_bytes'],
                      case_charged_transfer_bytes=account['case_prior_charged_transfer_bytes']+charged)
        write('transfer-closeout.json',closed)
    write('maintenance-closeout.json',dict(active=False,validation_id=job.name,return_manifest_sha256=mh))
    write('completion.json',dict(status='failed_run_evidence_returned',exit_status=1,files=len(rows),
        transfer_reservation_closed=True,maintenance=dict(status='cleared',validation_id=job.name)))
    return job


def test_failed_run_complete_return_is_valid_but_not_scientific_success(actual_failed_run,tmp_path):
    job=transport_fixture(tmp_path,actual_failed_run['run'])
    result=checker.Audit(job/'run').transport()
    assert result['closed_verified_return'] and result['exit_status']==1


@pytest.mark.parametrize('part',['active','reservation','account','missing','archive','completion'])
def test_complete_return_rejects_unclosed_or_changed_evidence(actual_failed_run,tmp_path,part):
    job=transport_fixture(tmp_path,actual_failed_run['run'])
    if part=='missing':(job/'terminal-process-check.json').unlink()
    elif part=='archive':
        with (job/'final-artifacts.tar').open('ab') as stream:stream.write(b'changed')
    else:
        name={'active':'maintenance-closeout.json','reservation':'transfer-closeout.json','account':'transfer-account.json','completion':'completion.json'}[part]
        value=json.loads((job/name).read_text())
        if part=='active':value['active']=True
        elif part=='reservation':value['state']='active'
        elif part=='account':value['attempts'][0]['bytes']+=1
        else:value['status']='returned_pending_scientific_audit'
        (job/name).write_text(json.dumps(value))
    with pytest.raises(checker.AuditError):checker.Audit(job/'run').transport()


@pytest.mark.parametrize('fail_at',[1,2,5])
def test_partial_solver_callback_exception_keeps_attempted_inputs(tmp_path,fail_at):
    rec=fm.Record(tmp_path/'run');model=RecordedToyMap(rec);scaled=fm.Scaled(model,model.scale)
    c=np.zeros(896);c[0]=1.;seed=np.zeros(1793);seed[0]=-.02;seed[896]=-.001;seed[897]=1.
    rec.arrays('interpolated-seed',z=seed,c=c,weight=np.array(.5));fold=fm.Fold(scaled,c)
    attempted=[]
    def invoke(function,*args):
        attempted.append(tuple(x.copy() for x in args))
        if len(attempted)==fail_at:raise OSError('synthetic instrument interruption')
        return function(*args)
    with pytest.raises(palc.SolverFailure) as exc:
        palc.solve_system(lambda y:invoke(fold.residual,y),lambda y,d:invoke(fold.action,y,d),seed,
            config=palc.NewtonConfig(residual_tolerance=1e-8),
            residual_blocks={'orbit':(0,896,1e-10/model.scale),'null':(896,1792,1e-8),'normalization':(1792,1793,1e-10)},
            event=lambda event:rec.event('fold-solver',stage='primary',solver_event=event))
    rec.stream.close();result=exc.value.result
    a=checker.Audit(tmp_path/'run',synthetic=True);a.scale=model.scale;a.normalization=c;a.events()
    a.solver('fold-solver:primary',result)
    assert result['status']=='failed' and len(result['calls'])==fail_at
    np.testing.assert_array_equal(result['calls'][-1]['y'],attempted[-1][0])


def test_failed_nonfinite_tangent_preserves_last_finite_primal(tmp_path):
    rec=fm.Record(tmp_path/'run');rec.event('alpha_selected',alpha=checker.ALPHA)
    x=np.zeros(896);v=np.ones(896);received=np.full(896,np.nan)
    rec.arrays('mixed-jvp-failed-return',received_state=x,received_probabilities=np.ones(8)/8,received_tangent=received)
    name=rec.arrays('mixed-jvp',states=np.stack([x,x]),tangents=v[None],alpha=np.array(checker.ALPHA),
                    logtemp=np.array(checker.P),dlogtemp=np.array(0.),top5=np.array([[np.arange(5),np.ones(5)/8]]))
    rec.event('cycle',file=name,label='mixed-jvp',steps_completed=1,differentiated=True,cached=True,seconds=0.,
              failed=dict(type='ValueError',message='finite float64 shape required'),precision=[PRECISION])
    rec.stream.close();a=checker.Audit(tmp_path/'run',synthetic=True);a.events()
    assert len(a.cycles)==1 and a.cycles[0]['event']['failed'] is not None


def test_source_inventory_cannot_omit_a_runtime_dependency(actual_failed_run,tmp_path):
    import shutil
    f=actual_failed_run;dest=tmp_path/'run';shutil.copytree(f['run'],dest)
    doc=json.loads((dest/'source.json').read_text());doc['files']=doc['files'][:-1]
    (dest/'source.json').write_text(json.dumps(doc))
    a=checker.Audit(dest,synthetic=True)
    with pytest.raises(checker.AuditError,match='source inventory'):
        a.provenance(f['root'],f['commit'],f['pin'],checker.sha(f['pin']))


def test_alpha_selection_is_literal_and_fixed_after_bridge(actual_failed_run):
    f=actual_failed_run;a=checker.Audit(f['run'],synthetic=True)
    a.provenance(f['root'],f['commit'],f['pin'],checker.sha(f['pin']));a.events();a.backend()
    a.alphas.append(dict(alpha=checker.ALPHA+.001))
    with pytest.raises(checker.AuditError,match='alpha selection'):a.backend()


@pytest.fixture
def actual_norm_reduction_callback(tmp_path):
    path=Path('/data2/shrimp-memory-data/preparations/public-e01/auditor-fixtures/first-production-callback-norm-v1.npz')
    if not path.exists():pytest.skip('external, hash-pinned returned-callback regression fixture is absent')
    assert checker.sha(path)=='ac8d61873be444058092f63abb15b14db309c8a8bd446807022991681d0ff8be'
    with np.load(path,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
    a=checker.Audit(tmp_path,synthetic=True);a.scale=float(values['scale']);a.normalization=values['normalization']
    labels=['mixed-jvp','fold-fd-plus','fold-fd-minus','fold-vector'];cycles=[]
    for i,label in enumerate(labels):
        z={k.removeprefix(f'cycle_{i}_'):v for k,v in values.items() if k.startswith(f'cycle_{i}_')}
        cycles.append(dict(name=f'{90+i:05d}-{label}.npz',z=z,event=dict(label=label,differentiated=True,failed=None)))
    call=dict(kind='action',y=values['y'],direction=values['direction'])
    return a,call,cycles,values['output']


def test_actual_failed_callback_replays_saved_offsets_and_action(actual_norm_reduction_callback):
    a,call,cycles,output=actual_norm_reduction_callback
    actual=a.callback_value(call,cycles,'fold-solver:primary')
    # This is the existing callback-output agreement gate; no solver or
    # scientific residual threshold is changed by the provenance regression.
    a.close(actual,output,'returned callback output',rtol=5e-12,atol=2e-13)
    np.testing.assert_array_equal(actual[896:1792],output[896:1792])
    assert all(abs(r['norm_ulp_offset'])<=8 and r['saved_signed_inputs_and_temperatures_exact']
               for r in a.derived_offset_reductions)


@pytest.mark.parametrize('part',['plus_state','minus_state','temperature','literal_base'])
def test_actual_offset_reduction_allowance_rejects_changed_raw_inputs(actual_norm_reduction_callback,part):
    a,call,cycles,_=actual_norm_reduction_callback
    if part=='plus_state':cycles[1]['z']['states'][0,191]+=1e-8
    elif part=='minus_state':cycles[2]['z']['states'][0,191]+=1e-8
    elif part=='temperature':cycles[1]['z']['logtemp']+=1e-9
    else:cycles[0]['z']['states'][0,191]+=1e-20
    with pytest.raises(checker.AuditError):a.callback_value(call,cycles,'fold-solver:primary')


def test_offset_reconstruction_requires_one_common_bounded_reduction(tmp_path):
    a=checker.Audit(tmp_path,synthetic=True);a.scale=1.
    base=np.zeros(897);direction=np.zeros(897);direction[0]=1.;h=.001
    def offset(sign,reduction):
        x,t=a.unpack(base+sign*h*direction/reduction)
        return dict(z=dict(states=x[None],logtemp=np.array(t)),event=dict(label='derived'))
    near=float(np.nextafter(1.,np.inf));far=1.
    for _ in range(9):far=float(np.nextafter(far,np.inf))
    with pytest.raises(checker.AuditError,match='common norm'):
        a.offset_norm(base,direction,h,[(1,offset(1,near)),(-1,offset(-1,1.))])
    with pytest.raises(checker.AuditError,match='eight ULPs'):
        a.offset_norm(base,direction,h,[(1,offset(1,far)),(-1,offset(-1,far))])


@pytest.fixture(scope='module')
def recorded_aligned_solver(tmp_path_factory):
    base=tmp_path_factory.mktemp('fold-audit-amended-solver')
    rec=fm.Record(base/'run');model=RecordedToyMap(rec);scaled=fm.Scaled(model,model.scale)
    c=np.zeros(896);c[0]=1.;seed=np.zeros(1793);seed[0]=-.02;seed[896]=-.001;seed[897]=1.
    rec.arrays('interpolated-seed',z=seed,c=c,weight=np.array(.5));fold=fm.Fold(scaled,c)
    result=palc.solve_system(lambda z:producer.fold_solver_output(fold.residual(z)),
        lambda z,d:producer.fold_solver_output(fold.action(z,d)),seed,
        config=palc.NewtonConfig(residual_tolerance=1e-8),
        residual_blocks={'orbit':(0,896,1e-10/model.scale),'normalization':(896,897,1e-10),'null':(897,1793,1e-8)},
        event=lambda e:rec.event('fold-solver',stage='primary',solver_event=e))
    rec.json('primary-solver.json',palc.serializable(result));rec.stream.close()
    assert result['status']=='converged'
    a=checker.Audit(base/'run',synthetic=True);a.amended=True;a.scale=model.scale;a.normalization=c
    a.result=dict(status='completed',counts=model.counts.report());a.events()
    return a,result


def test_actual_amended_solver_preserves_all_equations_and_recorded_offsets(recorded_aligned_solver):
    a,result=copy.deepcopy(recorded_aligned_solver)
    a.solver('fold-solver:primary',result);a.offset_evidence()
    last=next(c for c in reversed(result['calls']) if c['kind']=='residual' and c['status']=='completed')
    cycles=a.solver_calls['fold-solver:primary'][last['id']]['cycles']
    physical=a.callback_value(last,cycles,'fold-solver:primary',solver_rows=False)
    np.testing.assert_array_equal(last['output'],np.r_[physical[:896],physical[-1:],physical[896:-1]])
    assert np.linalg.norm(physical)<=1e-8
    assert result['config']['gmres_restart']==10 and result['config']['max_linear_actions']==12
    assert a.recorded_offset_reductions and not a.derived_offset_reductions


@pytest.mark.parametrize('part',['block_order','old_output_order','norm_gate','null_gate','fresh_info'])
def test_amended_solver_rejects_changed_equations_or_gates(recorded_aligned_solver,part):
    a,result=copy.deepcopy(recorded_aligned_solver)
    if part=='block_order':result['residual_blocks']['normalization']=[1792,1793,1e-10]
    elif part=='old_output_order':
        c=next(c for c in result['calls'] if c['status']=='completed')
        c['output']=np.r_[c['output'][:896],c['output'][897:],c['output'][896:897]]
    elif part=='norm_gate':result['residual'][896]=2e-10
    elif part=='null_gate':result['residual'][897]=2e-8
    else:result['iterations'][0]['linear']['gmres_info']=1
    with pytest.raises(checker.AuditError):a.solver('fold-solver:primary',result)


@pytest.mark.parametrize('part',['missing','scalar','hex','step','base_hash','direction_hash','plus_link','status'])
def test_recorded_scalar_is_not_reconstructed_or_silently_repaired(recorded_aligned_solver,tmp_path,part):
    a,result=copy.deepcopy(recorded_aligned_solver)
    call=next(c for c in result['calls'] if c['kind']=='action' and np.linalg.norm(c['direction'][:897])>0)
    cycles=a.solver_calls['fold-solver:primary'][call['id']]['cycles'];event=a.offset_events[cycles[0]['name']]
    if part=='missing':del a.offset_events[cycles[0]['name']]
    elif part=='scalar':event['norm']*=1.0001;event['norm_hex']=event['norm'].hex()
    elif part=='hex':
        import shutil
        shutil.copytree(a.run,tmp_path/'run')
        log=tmp_path/'run'/'events.jsonl';rows=[json.loads(line) for line in log.read_text().splitlines()]
        for row in rows:
            if row['event']=='fold_offset_reduction':row['norm_hex']='0x1.0p+0';break
        log.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        with pytest.raises(checker.AuditError,match='norm encoding'):checker.Audit(tmp_path/'run',synthetic=True).events()
        return
    elif part=='step':event['h']*=2
    elif part=='base_hash':event['z_sha256']='0'*64
    elif part=='direction_hash':event['dy_sha256']='0'*64
    elif part=='plus_link':event['plus_file']=event['minus_file']
    else:event['status']='incomplete'
    with pytest.raises(checker.AuditError):a.callback_value(call,cycles,'fold-solver:primary')


def test_recorded_scalar_uses_actual_one_ulp_value_without_inference(actual_norm_reduction_callback):
    a,call,cycles,expected=actual_norm_reduction_callback;a.amended=True
    base=call['y'][:897];d=call['direction'];dy=d[:897]
    value=0.047147200427405465
    a.offset_events[cycles[0]['name']]=dict(augmented_action=8,h=.001,norm=value,norm_hex=value.hex(),
        y_sha256=checker.array_hash(base),dy_sha256=checker.array_hash(dy),z_sha256=checker.array_hash(call['y']),
        d_sha256=checker.array_hash(d),top_file=cycles[0]['name'],plus_file=cycles[1]['name'],
        minus_file=cycles[2]['name'],vector_file=cycles[3]['name'],status='completed')
    output=a.callback_value(call,cycles,'fold-solver:primary')
    a.close(output,np.r_[expected[:896],expected[-1:],expected[896:-1]],'actual input exact recorded scalar',rtol=5e-12,atol=2e-13)
    assert not a.derived_offset_reductions and a.recorded_offset_reductions[0]['recorded_direction_norm']==value


@pytest.fixture(scope='module')
def actual_case_preflight(tmp_path_factory):
    import shutil
    original=Path('/tmp/e01-continuation-ops/context')
    if not (original/'case-context.json').exists():pytest.skip('external actual amendment context is absent')
    base=tmp_path_factory.mktemp('fold-audit-case-contract');context=base/'context'
    shutil.copytree(original,context)
    doc=json.loads((context/'case-context.json').read_text())
    assert doc['prior_audit']['sha256']==checker.FIRST_AUDIT_SHA
    args=producer.parser().parse_args(['--input-packet',str(context/'inputs.json'),
        '--input-packet-sha256',checker.sha(context/'inputs.json'),'--case-context',str(context/'case-context.json'),
        '--case-context-sha256',checker.sha(context/'case-context.json'),'--cache-dir',str(base/'unused'),
        '--vectors',str(base/'unused.npz'),'--output-dir',str(base/'run'),'--preflight-only'])
    real_check=subprocess.check_output
    def source_check(command,*a,**kw):
        if command[:3]==['git','status','--porcelain']:return '' if kw.get('text') else b''
        return real_check(command,*a,**kw)
    with patch.object(producer.subprocess,'check_output',side_effect=source_check), \
         patch.object(fm.time,'time',return_value=doc['geometry_deadline_unix']-30000), \
         patch.object(fm,'Map',side_effect=AssertionError('no model in preflight')):
        assert producer.run(args)==0
    return dict(base=base,args=args,commit=json.loads((args.output_dir/'source.json').read_text())['commit'])


def case_reader(f,base=None):
    args=f['args'];a=checker.Audit(args.output_dir,synthetic=True);a.result=a.read('result.json')
    a.provenance(producer.ROOT,f['commit'],args.input_packet,args.input_packet_sha256)
    if base is None:context=args.case_context
    else:
        import shutil
        shutil.copytree(args.case_context.parent,base/'context');context=base/'context'/'case-context.json'
    return a,context


def test_actual_context_zero_model_preflight_and_binary_bindings(actual_case_preflight):
    f=actual_case_preflight;args=f['args']
    report=checker.audit(args.output_dir,producer.ROOT,f['commit'],args.input_packet,args.input_packet_sha256,
        synthetic=True,case_context=args.case_context,case_context_sha256=args.case_context_sha256)
    assert report['status']=='audit_passed' and report['scientific_status']=='preflight_passed'
    assert report['counts']['ordinary']==0 and report['case_counts']['ordinary']==139
    assert report['attempt_index']==1 and not report['pulse_eligible']
    assert any(r['path'].endswith('inputs.npz') for r in report['bindings'])


@pytest.mark.parametrize('part',['prior_count','raw_debit','remaining','deadline','repeat','packet','order','missing_caller'])
def test_case_context_rejects_fresh_allowance_or_changed_inputs(actual_case_preflight,tmp_path,part):
    a,path=case_reader(actual_case_preflight,tmp_path);doc=json.loads(path.read_text())
    if part=='prior_count':doc['prior_counts']['ordinary']-=1
    elif part=='raw_debit':doc['prior_usage']['raw_bytes']-=1
    elif part=='remaining':doc['remaining_limits']['ordinary']=checker.CAPS['ordinary']
    elif part=='deadline':doc['geometry_deadline_unix']+=1
    elif part=='repeat':doc['attempt_index']=2
    elif part=='packet':doc['input_arrays']['sha256']='0'*64
    elif part=='order':a.result['fold_equation_order']=['orbit','null','normalization']
    path.write_text(json.dumps(doc));pin=checker.sha(path)
    a.source['arguments']['case_context_sha256']=pin
    with pytest.raises(checker.AuditError):a.case_context(None if part=='missing_caller' else path,pin)


@pytest.mark.parametrize('part',['ordinary','jvp_cycles','augmented','elapsed_seconds','finished_unix','raw_limit'])
def test_case_counter_and_original_deadline_gates(actual_case_preflight,part):
    a,path=case_reader(actual_case_preflight);a.case_context(path,checker.sha(path));a.events()
    if part=='finished_unix':a.result['finished_unix']=a.case['geometry_deadline_unix']+1
    elif part=='raw_limit':
        a.result['raw_limit_bytes']+=1
        with pytest.raises(checker.AuditError):a.case_context(path,checker.sha(path))
        return
    else:a.result['case_counts'][part]+=1
    with pytest.raises(checker.AuditError):a.counters()


@pytest.mark.parametrize('failed_label',['mixed-jvp','fold-fd-plus','fold-fd-minus','fold-vector'])
def test_amended_partial_action_keeps_inputs_scalar_and_signed_raw(tmp_path,failed_label):
    rec=fm.Record(tmp_path/'run');model=RecordedToyMap(rec);scaled=fm.Scaled(model,model.scale)
    c=np.zeros(896);c[0]=1.;seed=np.zeros(1793);seed[0]=-.02;seed[896]=-.001;seed[897]=1.
    direction=np.zeros_like(seed);direction[0]=.2;direction[896]=.01;direction[900]=.3
    original=model.cycle
    def cycle(*args,**kwargs):
        model.last_cycle_file=None
        if kwargs['label']==failed_label:raise OSError('synthetic callback interruption')
        return original(*args,**kwargs)
    model.cycle=cycle
    with pytest.raises(OSError):fm.Fold(scaled,c).action(seed,direction)
    rec.stream.close();a=checker.Audit(tmp_path/'run',synthetic=True)
    a.amended=True;a.scale=model.scale;a.normalization=c
    a.result=dict(status='execution_failed',counts=model.counts.report());a.events();a.offset_evidence()
    a.failed_callback_inputs(dict(kind='action',y=seed,direction=direction),a.cycles,'fold-solver:primary')
    assert len(list(a.run.glob('*-fold-action-input.npz')))==1
    if failed_label!='mixed-jvp':assert a.recorded_offset_reductions and not a.derived_offset_reductions


def test_amended_return_uses_remaining_shared_allowance(actual_case_preflight,tmp_path):
    a,path=case_reader(actual_case_preflight);a.case_context(path,checker.sha(path))
    job=transport_fixture(tmp_path,a.run,path.parent);a.run=job/'run';a.synthetic=False
    result=a.transport()
    assert result['closed_verified_return'] and result['exit_status']==1
    assert json.loads((job/'transfer-account.json').read_text())['reserved_bytes']==12853548598


@pytest.mark.parametrize('part',['full_reserve','prior_debit','cumulative_charge','archive_allowance','case_start'])
def test_amended_return_rejects_reused_full_allowance_or_missing_context(actual_case_preflight,tmp_path,part):
    a,path=case_reader(actual_case_preflight);a.case_context(path,checker.sha(path))
    job=transport_fixture(tmp_path,a.run,path.parent);a.run=job/'run';a.synthetic=False
    if part=='case_start':(job/'case-started-utc.txt').write_text('2026-10-30T04:00:00Z\n')
    elif part=='archive_allowance':a.case['remaining']['archive_bytes']=1
    else:
        closed=json.loads((job/'transfer-closeout.json').read_text())
        if part=='full_reserve':closed['reserved_bytes']=12*2**30
        elif part=='prior_debit':closed['case_prior_charged_transfer_bytes']=0
        else:closed['case_charged_transfer_bytes']=closed['charged_transfer_bytes']
        (job/'transfer-closeout.json').write_text(json.dumps(closed))
    with pytest.raises(checker.AuditError):a.transport()
