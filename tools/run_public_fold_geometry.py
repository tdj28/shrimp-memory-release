#!/usr/bin/env python3
"""E01: exact-alpha numerical fold localization with an explicit branch link.

Exploratory geometry only. Conditional pulse experiments require its returned,
audited native states; this runner never launches another process or a trial.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
import numpy as np
import public_fold_map as fm
import public_spectral as spectral

ROOT=Path(__file__).resolve().parents[1]
PROFILE='public-e01'
AMENDMENT='orthogonal-row-order-1'
FOLD_SOLVER_PERMUTATION=np.r_[np.arange(896),1792,np.arange(896,1792)]
class NumericalUnresolved(RuntimeError):pass

def _binding(path):
    path=Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=fm.instrument.sha256(path))

def _same_bytes(left,right):
    return left['bytes']==right['bytes'] and left['sha256']==right['sha256']

def case_context(args,*,now=None):
    """Qualify one accepted failed first attempt; no renewed case allowance.

    Relocated bindings may change paths only. Original source, result, UTC start,
    packet and arrays remain byte-identical to their accepted audit bindings.
    """
    path=getattr(args,'case_context',None);pin=getattr(args,'case_context_sha256',None)
    if path is None and pin is None:return None
    if path is None or pin is None:raise ValueError('case context path and hash are required together')
    path=Path(path);binding=_binding(path)
    if binding['sha256']!=pin:raise ValueError('case context hash differs')
    doc=json.loads(path.read_text())
    if (doc.get('profile')!='public-e01-case-context' or doc.get('amendment')!=AMENDMENT
        or doc.get('attempt_index')!=1):raise ValueError('fixed first amendment context required')
    required=('prior_audit','prior_result','prior_source','started_utc','input_packet','input_arrays')
    verified={}
    for key,value in doc.items():
        if key not in required and not (key.startswith('prior_') and isinstance(value,dict) and 'path' in value):continue
        if not isinstance(value,dict) or set(('path','bytes','sha256'))-value.keys():raise ValueError('invalid context binding '+key)
        target=Path(value['path']);target=target if target.is_absolute() else path.parent/target
        actual=_binding(target)
        if not _same_bytes(value,actual):raise ValueError('context binding differs: '+key)
        verified[key]=actual
    if set(required)-verified.keys():raise ValueError('missing prior context binding')
    audit=json.loads(Path(verified['prior_audit']['path']).read_text())
    for key,value in dict(status='audit_passed',integrity_status='passed',profile=PROFILE,synthetic=False,
        scientific_status='completed',outcome='primary_fold_unresolved',pulse_eligible=False,
        return_status='manifest_verified',closed_verified_return=True).items():
        if audit.get(key)!=value:raise ValueError('prior audit precondition '+key)
    if audit.get('pulse_admission',{}).get('eligible') is not False:raise ValueError('prior pulse admission must be false')
    aliases={'prior_result':'result','prior_source':'source','started_utc':'started_utc',
             'input_packet':'input_packet','input_arrays':'input_arrays'}
    for key,original in aliases.items():
        if not _same_bytes(verified[key],audit[original]):raise ValueError('prior audit binding differs: '+key)
    for key,original in doc.get('original_bindings',{}).items():
        original_key=aliases.get(key,key.removeprefix('prior_'))
        if original_key in audit and audit[original_key] is not None and original!=audit[original_key]:
            raise ValueError('original accepted binding differs: '+key)
        if key in verified and not _same_bytes(original,verified[key]):raise ValueError('relocated binding differs: '+key)
    previous=json.loads(Path(verified['prior_result']['path']).read_text())
    source=json.loads(Path(verified['prior_source']['path']).read_text())
    commit=previous.get('source_commit','')
    if (not re.fullmatch('[0-9a-f]{40}',commit) or commit!=audit['source_commit'] or commit!=source['commit']
        or previous.get('profile')!=PROFILE or previous.get('status')!='completed'
        or previous.get('outcome')!='primary_fold_unresolved' or previous.get('attempt_index',0)!=0
        or previous.get('case_context') is not None):raise ValueError('prior first-attempt identity differs')
    if previous['counts']!=audit['counts']:raise ValueError('prior count binding differs')
    prior=previous['counts']
    if 'prior_counts' in doc and doc['prior_counts']!=prior:raise ValueError('context prior counts differ')
    if 'source_commit' in doc and doc['source_commit']!=commit:raise ValueError('context prior source differs')
    for key in ('ordinary','jvp_cycles','augmented','model_invocations'):
        if type(prior[key]) is not int or prior[key]<0:raise ValueError('invalid prior count '+key)
    for key in ('elapsed_seconds','peak_rss_bytes'):
        if not math.isfinite(prior[key]) or prior[key]<0:raise ValueError('invalid prior resource '+key)
    if previous.get('caps')!=fm.CAPS:raise ValueError('original case limits differ')
    if not _same_bytes(_binding(args.input_packet),verified['input_packet']):raise ValueError('case packet changed')
    if not _same_bytes(_binding(Path(args.input_packet).parent/'inputs.npz'),verified['input_arrays']):raise ValueError('case arrays changed')
    text=Path(verified['started_utc']['path']).read_text().strip()
    start=datetime.fromisoformat(text.replace('Z','+00:00'))
    if start.tzinfo is None or start.utcoffset().total_seconds()!=0:raise ValueError('case start must be explicit UTC')
    started=start.timestamp();geometry_deadline=started+fm.CAPS['seconds'];campaign_deadline=started+72000
    if text!=audit['campaign_started_utc'] or campaign_deadline!=audit['campaign_deadline_unix']:
        raise ValueError('original case deadline differs')
    for key,expected in (('case_started_utc',text),('geometry_deadline_unix',geometry_deadline),('campaign_deadline_unix',campaign_deadline)):
        if key in doc and doc[key]!=expected:raise ValueError('context deadline differs: '+key)
    current=time.time() if now is None else float(now)
    if not math.isfinite(current) or current<started or current>=geometry_deadline:
        raise TimeoutError('original geometry deadline expired or clock predates case')
    limits=doc['case_limits'];usage=doc['prior_usage'];remaining=doc['remaining_limits']
    for key in ('ordinary','jvp_cycles','augmented','raw_bytes'):
        if limits.get(key)!=fm.CAPS[key]:raise ValueError('case limit changed: '+key)
        if type(usage.get(key)) is not int or usage[key]<0 or remaining.get(key)!=limits[key]-usage[key] or remaining[key]<=0:
            raise ValueError('remaining case limit differs: '+key)
        if key!='raw_bytes' and usage[key]!=prior[key]:raise ValueError('prior usage differs: '+key)
    if usage['raw_bytes']<audit['raw_inventory']['raw_bytes']:raise ValueError('prior raw debit below accepted inventory')
    if usage.get('model_invocations')!=prior['model_invocations']:raise ValueError('prior invocation debit differs')
    for key,cap in (('package_bytes',3*2**30),('archive_bytes',3*2**30),('transfer_bytes',12*2**30)):
        if (limits.get(key)!=cap or type(usage.get(key)) is not int or usage[key]<0
            or remaining.get(key)!=cap-usage[key] or remaining[key]<=0):raise ValueError('remaining transport limit differs: '+key)
    if remaining['raw_bytes']<=16*2**20:raise fm.LimitError('remaining raw allowance cannot retain finalization')
    if prior['elapsed_seconds']>=fm.CAPS['seconds']:raise TimeoutError('original geometry numerical time exhausted')
    return dict(binding=binding,document=doc,verified_bindings=verified,prior_counts=prior,
        prior_source_commit=commit,geometry_deadline_unix=geometry_deadline,campaign_deadline_unix=campaign_deadline,
        raw_limit=remaining['raw_bytes'],prior_raw_bytes=usage['raw_bytes'])

def fold_solver_output(value):
    """Orthogonal row permutation only; physical Fold methods stay unchanged."""
    return fm.finite(value,(1793,))[FOLD_SOLVER_PERMUTATION]

def inputs(args):
    if fm.instrument.sha256(args.input_packet)!=args.input_packet_sha256:
        raise ValueError('input packet hash differs')
    doc=json.loads(args.input_packet.read_text());p=args.input_packet.parent/'inputs.npz'
    if p.stat().st_size!=doc['arrays']['bytes'] or fm.instrument.sha256(p)!=doc['arrays']['sha256']:
        raise ValueError('input arrays hash differs')
    with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    for k,shape in dict(fold_states=(2,896),fold_alphas=(2,),fold_logtemps=(2,),
                        p_terminal=(2,896),p_cycles=(6,2,896),h_cycles=(2,6,896),state_scale=()).items():
        fm.finite(a[k],shape)
    if not np.array_equal(a['p_cycles'][0],a['p_terminal']):raise ValueError('P phase order')
    if float(a['state_scale'])!=float(np.linalg.norm(a['p_terminal'][1]-a['p_terminal'][0])):
        raise ValueError('literal terminal scale')
    return a,doc

def cycle_gate(trace,tol=1e-10):
    trace=np.asarray(trace)
    if trace.shape!=(13,896) or not np.isfinite(trace).all():raise ValueError('13-state validation trace')
    closure=np.linalg.norm(trace[6:12]-trace[:6],axis=1)
    gaps={str(k):float(np.max(np.linalg.norm(trace[k:k+6]-trace[:6],axis=1))) for k in (1,2,3)}
    return dict(passed=bool(np.max(closure)<=tol and min(gaps.values())>1e-4),
                closure_l2=closure.tolist(),closure_max=float(np.max(closure)),proper_divisor_gaps=gaps)

def align_cycle(cycle,reference):
    distances=[float(np.max(np.linalg.norm(np.roll(cycle,-i,axis=0)-reference,axis=1))) for i in range(6)]
    i=int(np.argmin(distances));return np.roll(cycle,-i,axis=0).copy(),dict(shift=i,distances=distances)

def turn_estimate(result):
    """Local cubic-Hermite temperature maximum; numerical estimate, no bound."""
    if result['status']!='completed' or result['reason']!='qualified_tangent_turn' or len(result['turn_brackets'])!=1:
        raise ValueError('qualified single turn required')
    ids=result['turn_brackets'][0]['point_indices'];a,b=[result['points'][i] for i in ids]
    f0,f1=float(a['y'][-1]),float(b['y'][-1]);length=float(np.linalg.norm(b['y']-a['y']))
    m0,m1=float(a['tangent'][-1])*length,float(b['tangent'][-1])*length
    if not m0>0>m1:raise NumericalUnresolved('temperature-maximum orientation unresolved')
    coeff=np.array([2*f0-2*f1+m0+m1,-3*f0+3*f1-2*m0-m1,m0,f0])
    roots=np.roots(np.polyder(coeff));eligible=[float(r.real) for r in roots if abs(r.imag)<1e-10 and 0<r.real<1 and np.polyval(np.polyder(coeff,2),r.real)<0]
    if len(eligible)!=1:raise NumericalUnresolved('ambiguous local temperature maximum')
    eta=float(np.polyval(coeff,eligible[0]))
    q=eligible[0]
    yestimate=((2*q**3-3*q**2+1)*a['y']+(q**3-2*q**2+q)*length*a['tangent']
               +(-2*q**3+3*q**2)*b['y']+(q**3-q**2)*length*b['tangent'])
    return dict(log10_temperature=fm.H+fm.DELTA*eta,parameter=eligible[0],
        chord_length=length,coefficients=coeff.tolist(),point_indices=ids,scaled_state_estimate=yestimate[:-1].tolist(),
        scope='Local cubic-Hermite interpolation, refinement spread is not an enclosure.')

def spectrum(s,y,label,seed=0):
    rec=s.model.rec
    a=spectral.arnoldi(lambda v:s.av(y,v,label+'-action'),np.random.default_rng(seed).normal(size=896),8)
    r=spectral.ritz_at(a)
    file=rec.arrays(label+'-arnoldi',**{k:v for k,v in a.items() if isinstance(v,np.ndarray)},
                    values=r['values'],vectors=r['vectors'],ritz_residuals=r['residual_norms'])
    order=np.argsort(-abs(r['values']))[:3];pairs=[]
    for i in order:
        v=r['vectors'][:,i];lam=r['values'][i]
        av=s.av(y,v.real,label+'-fresh-real').astype(complex)
        if np.linalg.norm(v.imag)>1e-14:av+=1j*s.av(y,v.imag,label+'-fresh-imag')
        residual=float(np.linalg.norm(av-lam*v)/max(np.linalg.norm(av),abs(lam)*np.linalg.norm(v),1e-14))
        rec.arrays(label+'-eigenpair',vector=v,action=av,eigenvalue=np.array(lam))
        pairs.append(dict(index=int(i),value=[float(lam.real),float(lam.imag)],modulus=float(abs(lam)),
                          normalized_residual=residual,qualified=residual<=1e-6))
    chosen=int(np.argmin(abs(r['values']-1)))
    v=r['vectors'][:,chosen]
    if np.linalg.norm(v.imag)>1e-8:raise NumericalUnresolved('bounded critical mode remains complex')
    v=v.real/np.linalg.norm(v.real)
    row=dict(basis_passed=a['checks_passed'],pairs=pairs,arnoldi=file,selected_index=chosen,
        selected_value=[float(r['values'][chosen].real),float(r['values'][chosen].imag)],
        observed_other_distance_from_one=float(min(abs(np.delete(r['values'],chosen)-1))),
        scope='Eight-dimensional numerical mode diagnostic, not a proof of spectral coverage.')
    rec.json(label+'-spectrum.json',row)
    return v,row

def derivative_pilot(s,y):
    rec=s.model.rec;x,t=s.unpack(y)
    rng=np.random.default_rng(74);dx=rng.normal(size=896);dx/=np.linalg.norm(dx)
    # Unit state perturbation with a comparably scaled temperature component.
    d=np.r_[dx/s.scale,1/s.scale]
    tr,j=s.model.cycle(x,t,v=dx,dlogtemp=fm.DELTA/s.scale,label='pilot-mixed')
    primal,_=s.model.cycle(x,t,label='pilot-primal')
    repeat,_=s.model.cycle(x,t,label='pilot-repeat')
    gaps=[float(np.max(abs(tr-primal))),float(np.max(abs(primal-repeat)))]
    for k in (0,3):
        full,_=s.model.cycle(primal[k],t,label='pilot-full',steps=1,cached=False)
        gaps.append(float(np.max(abs(full[-1]-primal[k+1]))))
    rows=[]
    for h in (4e-5,2e-5,1e-5,5e-6,2.5e-6):
        xp,tp=s.unpack(y+h*d);xm,tm=s.unpack(y-h*d)
        plus,_=s.model.cycle(xp,tp,label='pilot-fd-plus');minus,_=s.model.cycle(xm,tm,label='pilot-fd-minus')
        fd=(plus[-1]-minus[-1])/(2*h)
        rows.append(dict(h=h,relative_error=float(np.linalg.norm(fd-j[-1])/max(np.linalg.norm(j[-1]),1e-14))))
    check=fm.instrument.convergence_report(rows)
    u=np.r_[dx/s.scale,0.];v=np.r_[np.zeros(896),1/s.scale]
    au=s.action(y,u);av=s.action(y,v);mix=s.action(y,.6*u-.8*v)
    linear=float(np.linalg.norm(mix-(.6*au-.8*av))/max(np.linalg.norm(mix),1e-14))
    row=dict(passed=bool(max(gaps)<=1e-10 and gaps[1]==0. and check['passed'] and linear<=1e-10),
             primal_cache_gaps=gaps,fd_rows=rows,fd_gate=check,linearity_relative=linear)
    rec.json('derivative-pilot.json',row)
    return row

def fold_gate(s,z,c,label):
    y=z[:897];v=z[897:];x,t=s.unpack(y);f=fm.Fold(s,c)
    residual=f.residual(z);trace,_=s.model.cycle(x,t,label=label+'-orbit',steps=12)
    cg=cycle_gate(trace);norm=float(np.linalg.norm(v))
    row=dict(orbit_l2=float(np.linalg.norm(residual[:896])*s.scale),
        null_l2=float(np.linalg.norm(residual[896:1792])),normalization=abs(float(residual[-1])),
        vector_norm=norm,log10_temperature=t,alpha=s.model.alpha,cycle=cg)
    row['passed']=bool(cg['passed'] and row['orbit_l2']<=1e-10 and row['null_l2']<=1e-7
        and row['normalization']<=1e-10 and 1e-4<norm<1e4)
    s.model.rec.json(label+'-gate.json',row)
    s.model.rec.arrays(label+'-candidate',z=z,normalization_covector=c,cycle=trace[:6])
    return row,trace

def geometry(model,a,rec,summary,pilot_only=False):
    import public_pseudoarclength as palc
    scale=float(a['state_scale']);s=fm.Scaled(model,scale)
    # Literal native P references first; no transient-generated substitute.
    for k in (0,1):
        tr,_=model.cycle(a['p_terminal'][k],fm.P,label=f'p{k}-anchor',steps=12)
        gate=cycle_gate(tr);rec.json(f'p{k}-anchor.json',gate)
        if not gate['passed']:raise fm.InstrumentError('current P anchor gate')
    row21=None;fold_cycles=[];alignments=[]
    for k in (0,1):
        model.set_alpha(float(a['fold_alphas'][k]))
        tr,_=model.cycle(a['fold_states'][k],float(a['fold_logtemps'][k]),label=f'parent-{21+k}',steps=12)
        gate=cycle_gate(tr)
        cyc,alignment=align_cycle(tr[:6],a['p_cycles'][:,1]);fold_cycles.append(cyc);alignments.append(alignment)
        rec.json(f'parent-{21+k}-bridge.json',dict(gate=gate,alignment=alignment))
        if not gate['passed']:raise fm.InstrumentError('parent current-map bridge')
    model.set_alpha(float(a['fold_alphas'][0]));y21=s.pack(fold_cycles[0][0],float(a['fold_logtemps'][0]))
    pilot=derivative_pilot(s,y21);summary['pilot']=pilot
    if not pilot['passed']:raise fm.InstrumentError('mixed derivative pilot gate')
    c,spec=spectrum(s,y21,'parent21',seed=0)
    z21=np.r_[y21,c]
    gate,_=fold_gate(s,z21,c,'parent21')
    if not gate['passed'] or not spec['basis_passed'] or not spec['pairs'][0]['qualified']:
        raise NumericalUnresolved('saved parent fold/mode positive control did not qualify')
    summary['pilot_counts']=model.counts.report()
    if time.monotonic()-rec.started>1800:raise TimeoutError('feasibility pilot exceeded 1800 seconds')
    rec.checkpoint('feasibility-pilot',dict(counts=model.counts.report(),derivative=pilot,parent_fold=gate))
    if pilot_only:
        summary.update(status='completed',outcome='pilot_qualified');return
    model.counts.check()
    signal.setitimer(signal.ITIMER_REAL,max(.001,model.counts.remaining_seconds()))
    model.set_alpha(fm.ALPHA)
    w=(fm.ALPHA-a['fold_alphas'][0])/(a['fold_alphas'][1]-a['fold_alphas'][0])
    xseed=(1-w)*fold_cycles[0][0]+w*fold_cycles[1][0]
    tseed=float((1-w)*a['fold_logtemps'][0]+w*a['fold_logtemps'][1])
    zseed=np.r_[s.pack(xseed,tseed),c]
    rec.arrays('interpolated-seed',z=zseed,c=c,weight=np.array(w))
    solutions=[];gates=[]
    for name,h in [('primary',1e-3),('refined',5e-4)]:
        f=fm.Fold(s,c,h=h)
        # Finite-difference extended actions are checked at the common seed.
        rng=np.random.default_rng(75);d=rng.normal(size=1793);d/=np.linalg.norm(d)
        ah=f.action(zseed,d);half=fm.Fold(s,c,h=h/2).action(zseed,d)
        relative=float(np.linalg.norm(ah-half)/max(np.linalg.norm(half),1e-14))
        full_fd=(f.residual(zseed+1e-4*d)-f.residual(zseed-1e-4*d))/(2e-4)
        full_gap=float(np.linalg.norm(ah-full_fd)/max(np.linalg.norm(full_fd),1e-14))
        d2=rng.normal(size=1793);d2/=np.linalg.norm(d2)
        a2=f.action(zseed,d2);am=f.action(zseed,.6*d-.8*d2)
        linear=float(np.linalg.norm(am-(.6*ah-.8*a2))/max(np.linalg.norm(am),1e-14))
        passed=max(relative,full_gap,linear)<=1e-4
        rec.arrays(name+'-augmented-check',direction=d,second_direction=d2,action=ah,
                   half_action=half,full_residual_fd=full_fd,second_action=a2,mixed_action=am)
        rec.json(name+'-augmented-instrument.json',dict(fd_step=h,halving_relative=relative,
            full_residual_fd_relative=full_gap,linearity_relative=linear,passed=passed))
        if not passed:raise fm.InstrumentError('augmented action independent consistency gate')
        config=palc.NewtonConfig(max_newton_iterations=6,gmres_restart=10,
                                max_linear_actions=12,linear_relative_tolerance=1e-3,residual_tolerance=1e-8)
        aligned=summary.get('attempt_index',0)==1
        residual=(lambda z:fold_solver_output(f.residual(z))) if aligned else f.residual
        action=(lambda z,d:fold_solver_output(f.action(z,d))) if aligned else f.action
        blocks=({'orbit':(0,896,1e-10/scale),'normalization':(896,897,1e-10),'null':(897,1793,1e-8)}
                if aligned else {'orbit':(0,896,1e-10/scale),'null':(896,1792,1e-8),'normalization':(1792,1793,1e-10)})
        result=palc.solve_system(residual,action,zseed.copy(),config=config,residual_blocks=blocks,
            event=lambda event:rec.event('fold-solver',stage=name,solver_event=event))
        # API is deliberately checked by integration tests before a model run.
        rec.json(name+'-solver.json',palc.serializable(result))
        if result['status']!='converged':
            summary.update(status='completed',outcome=name+'_fold_unresolved');return
        z=result['y'];gate,tr=fold_gate(s,z,c,name);gates.append(gate);solutions.append(z)
        rec.checkpoint(name+'-fold',dict(gate=gate,counts=model.counts.report()))
        if not gate['passed']:
            summary.update(status='completed',outcome=name+'_fold_gate_failed');return
    fold=solutions[1];foldy=fold[:897];foldx,foldt=s.unpack(foldy)
    spread=abs(gates[0]['log10_temperature']-foldt)
    v,spec=spectrum(s,foldy,'target-fold',seed=1)
    summary['fold']=dict(primary=gates[0],refined=gates[1],refinement_spread=spread,spectrum=spec)
    spectrum_ok=(spec['basis_passed'] and all(q['qualified'] for q in spec['pairs'])
        and abs(complex(*spec['selected_value'])-1)<=1e-6
        and all(q['modulus']<1-1e-6 for q in spec['pairs'] if q['index']!=spec['selected_index'])
        and spec['observed_other_distance_from_one']>=.01)
    if spread>1e-6 or not spectrum_ok:
        summary.update(status='completed',outcome='fold_refinement_or_mode_unresolved');return
    # Public connected branch from original P1. Its literal phase is never relabelled.
    p1=s.pack(a['p_terminal'][1],fm.P)
    direction=np.zeros(897);direction[-1]=1.
    ds=float(np.linalg.norm(foldy-p1)/5.)
    cfg=palc.PACConfig(equation_tolerance=1e-10/scale,arc_tolerance=1e-10,residual_tolerance=1e-8,
        step_sizes=(ds,ds/2,ds/4),max_attempted_steps=16,max_accepted_steps=12,max_rejected_steps=4)
    result=palc.continue_branch(s.residual,s.action,p1,direction,config=cfg,stop_on_turn=True,
        event=lambda event:rec.event('branch-solver',solver_event=event))
    rec.json('p1-branch.json',palc.serializable(result))
    # The branch helper's complete-record contract is tested before launch.
    summary['branch']=palc.serializable(result)
    if result['status']!='completed' or result['reason']!='qualified_tangent_turn':
        summary.update(status='completed',outcome='p1_branch_link_unresolved');return
    coarse=turn_estimate(result)
    left=result['points'][result['turn_brackets'][0]['point_indices'][0]]
    accepted=[q for q in result['attempts'] if q['status']=='accepted']
    half=accepted[-1]['step_size']/2
    cfg2=palc.PACConfig(equation_tolerance=1e-10/scale,arc_tolerance=1e-10,residual_tolerance=1e-8,
        step_sizes=(half,half/2,half/4),max_attempted_steps=8,max_accepted_steps=4,max_rejected_steps=4)
    refinement=palc.continue_branch(s.residual,s.action,left['y'],left['tangent'],config=cfg2,stop_on_turn=True,
        event=lambda event:rec.event('branch-refinement',solver_event=event))
    rec.json('p1-branch-refinement.json',palc.serializable(refinement))
    if refinement['status']!='completed' or refinement['reason']!='qualified_tangent_turn':
        summary.update(status='completed',outcome='p1_turn_refinement_unresolved');return
    finer=turn_estimate(refinement)
    temperatures=[q['log10_temperature'] for q in (*gates,coarse,finer)]
    branch_spread=max(temperatures)-min(temperatures)
    estimates=[np.array(q['scaled_state_estimate'])*scale for q in (coarse,finer)]
    estimates.extend(s.unpack(q[:897])[0] for q in solutions)
    state_spread=max(float(np.linalg.norm(left-right)) for i,left in enumerate(estimates) for right in estimates[i+1:])
    summary['branch_link']=dict(passed=branch_spread<=1e-6 and state_spread<=1e-6,
        coarse=coarse,refined=finer,temperature_spread=branch_spread,state_spread_l2=state_spread)
    if not summary['branch_link']['passed']:
        summary.update(status='completed',outcome='fold_branch_location_disagreement');return
    sides=[]
    for k,index in enumerate(refinement['turn_brackets'][0]['point_indices']):
        point=refinement['points'][index];vv,ss=spectrum(s,point['y'],f'branch-side-{k}',seed=k)
        critical=complex(*ss['selected_value'])
        qualified=[q for q in ss['pairs'] if q['index']==ss['selected_index'] and q['qualified']]
        sidepassed=(ss['basis_passed'] and len(qualified)==1 and abs(critical.imag)<=1e-8
            and (abs(critical)<1-1e-5 if k==0 else critical.real>1+1e-5)
            and abs(float(vv@c))>.5 and all(q['qualified'] and q['modulus']<1-1e-6
                for q in ss['pairs'] if q['index']!=ss['selected_index']))
        sides.append(dict(passed=sidepassed,spectrum=ss,critical_alignment=abs(float(vv@c))))
    summary['branch_sides']=sides
    if not all(q['passed'] for q in sides):
        summary.update(status='completed',outcome='branch_stability_change_unresolved');return
    spread=max(spread,branch_spread)
    margin=max(1e-6,5*spread)
    order=fm.P+margin<foldt<fm.N-margin
    summary['fold']['comparison_margin']=margin
    summary['fold']['P_below_N_above']=order
    if not order:
        summary.update(status='completed',outcome='fold_margin_unresolved');return
    B=(fm.P+foldt)/2
    summary['B_log10_temperature']=B
    # Fixed-parameter native roots are controls here, after branch localization;
    # failed root searches beyond the fold are never used as nonexistence tests.
    native=[]
    for history,temp in [(0,B),(1,B),(0,fm.N)]:
        label=f'native-{history}-'+('B' if temp==B else 'N')
        eta=(temp-fm.H)/fm.DELTA
        start=a['p_terminal'][history]/scale
        if history==0 and temp==fm.N:start=native[0]['u'].copy()
        root=palc.solve_system(lambda u:s.residual(np.r_[u,eta]),
            lambda u,d:s.action(np.r_[u,eta],np.r_[d,0.]),start,
            config=palc.NewtonConfig(residual_tolerance=1e-10/scale),
            event=lambda event:rec.event('native-root',stage=label,solver_event=event))
        rec.json(label+'-solver.json',palc.serializable(root))
        if root['status']!='converged':
            summary.update(status='completed',outcome=label+'_unresolved');return
        u=root['y'];tr,_=model.cycle(u*scale,temp,label=label+'-orbit',steps=12)
        gate=cycle_gate(tr);native.append(dict(u=u,cycle=tr[:6],gate=gate,temp=temp,history=history))
        rec.json(label+'-gate.json',gate)
        if not gate['passed']:
            summary.update(status='completed',outcome=label+'_gate_failed');return
    _,comparison=align_cycle(native[0]['cycle'],native[1]['cycle'])
    separation=min(comparison['distances'])
    rec.arrays('native-controls',B_cycles=np.stack([q['cycle'] for q in native[:2]]),
        N_survivor_cycle=native[2]['cycle'],B_logtemp=np.array(B),fold_z=fold)
    summary['native_controls']=dict(B_cycle_separation=separation,all_primitive_six=True,
        scope='Native periodic roots; attraction beyond the tested mode diagnostics is not certified.')
    summary.update(status='completed',outcome='geometry_qualified_return_for_audit' if separation>1e-4 else 'native_B_distinction_unresolved')
    rec.checkpoint('geometry',dict(counts=model.counts.report(),outcome=summary['outcome']))

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('input_packet','vectors','cache_dir','output_dir'):p.add_argument('--'+k.replace('_','-'),type=Path,required=True)
    p.add_argument('--input-packet-sha256',required=True)
    p.add_argument('--case-context',type=Path)
    p.add_argument('--case-context-sha256')
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--pilot-only',action='store_true')
    return p

def run(args):
    rec=fm.Record(args.output_dir);counts=fm.Counts(rec)
    summary=dict(profile=PROFILE,status='execution_failed',outcome='not_run',evidence_class='adapted exploratory',
        evidence_rung=0,caps=fm.CAPS,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        attempt_index=0,case_context=None,fold_equation_order=['orbit','null','normalization'],started_unix=time.time())
    old={}
    try:
        a,doc=inputs(args)
        context=case_context(args)
        if context is not None:
            rec.raw_limit=context['raw_limit'];rec.check(0)
            counts=fm.Counts(rec,prior=context['prior_counts'],geometry_deadline_unix=context['geometry_deadline_unix'],
                             campaign_deadline_unix=context['campaign_deadline_unix'])
            summary.update(attempt_index=1,amendment=AMENDMENT,case_context=context['binding'],
                prior_source_commit=context['prior_source_commit'],geometry_deadline_unix=context['geometry_deadline_unix'],
                campaign_deadline_unix=context['campaign_deadline_unix'],prior_raw_bytes=context['prior_raw_bytes'],
                raw_limit_bytes=context['raw_limit'],fold_equation_order=['orbit','normalization','null'])
            rec.json('qualified-case-context.json',context)
        counts.check();rec.json('qualified-inputs.json',doc)
        files=['tools/run_public_fold_geometry.py','tools/public_fold_map.py','tools/public_pseudoarclength.py',
            'tools/public_spectral.py','tools/validate_qwen_map.py','tools/run_public_periodic_corrector.py',
            'tools/audit_public_fold_geometry.py','tests/test_public_fold_geometry.py','tests/test_audit_public_fold_geometry.py',
            'docs/public-e01-fold-protocol.md','pyproject.toml','uv.lock']
        dirty=subprocess.check_output(['git','status','--porcelain','--',*files],cwd=ROOT,text=True).splitlines()
        rec.json('source.json',dict(commit=summary['source_commit'],working_tree_status=dirty,
            files=[dict(path=p,bytes=(ROOT/p).stat().st_size,sha256=fm.instrument.sha256(ROOT/p)) for p in files],
            arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}))
        if dirty and not args.preflight_only:raise ValueError('numerical source must be committed and clean')
        if args.preflight_only:summary.update(status='preflight_passed',outcome='not_run_preflight')
        else:
            def stop(signum,frame):raise TimeoutError('geometry wall deadline or signal')
            for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):
                old[sig]=signal.getsignal(sig);signal.signal(sig,stop)
            signal.setitimer(signal.ITIMER_REAL,min(1800,counts.remaining_seconds()))
            model=fm.Map(args,rec,counts);geometry(model,a,rec,summary,args.pilot_only)
        counts.check()
    except BaseException as e:
        root=e
        while root.__cause__ is not None:root=root.__cause__
        kind='completed' if isinstance(root,NumericalUnresolved) else ('instrument_failed' if isinstance(root,fm.InstrumentError) else ('incomplete' if isinstance(root,(TimeoutError,MemoryError,fm.LimitError,KeyboardInterrupt)) else 'execution_failed'))
        summary.update(status=kind,error=f'{type(e).__name__}: {e}',root_error=f'{type(root).__name__}: {root}',traceback=traceback.format_exc())
        if isinstance(root,NumericalUnresolved):summary['outcome']='finite_geometry_unresolved'
        if hasattr(e,'result'):
            import public_pseudoarclength as palc
            rec.json('failed-solver.json',palc.serializable(e.result))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,h in old.items():signal.signal(sig,h)
        rec.finalizing=True
        summary['counts']=counts.report();summary['case_counts']=counts.case_report(summary['counts'])
        summary['finished_unix']=time.time();summary['utc']=fm.instrument.utc_now()
        rec.json('result.json',summary);rec.stream.close()
        files=[dict(path=p.name,bytes=p.stat().st_size,sha256=fm.instrument.sha256(p)) for p in sorted(rec.directory.iterdir()) if p.is_file()]
        actual=sum(r['bytes'] for r in files)
        if actual>rec.raw_limit:raise fm.LimitError('actual remaining raw bytes exceed cap')
        rec.json('manifest.json',dict(files=files,actual_bytes_before_manifest=actual))
    print(json.dumps(dict(status=summary['status'],outcome=summary['outcome'],counts=summary['counts'])))
    return 0 if summary['status'] in ('completed','preflight_passed') else 1

if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
