#!/usr/bin/env python3
"""Audit every shadow checkpoint and compare all states to an earlier replay."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import audit_p6_readback as context
from audit_readout_horizons import curve
from evaluate_readout_policy import proposal
record=context.common.independent.file_record


def check_shadow(base,index,shadow,states,refs,definition,bindings):
    if shadow['status']!='completed' or shadow['completed_steps']!=600 or shadow['definition']!=definition:
        raise ValueError('incomplete shadow or changed definition')
    timings=np.asarray(shadow['observer_seconds_per_update'])
    if timings.shape!=(600,) or not np.isfinite(timings).all() or np.any(timings<0):raise ValueError('invalid observer timings')
    if not np.isfinite(shadow['total_observer_seconds']) or shadow['total_observer_seconds']<0:raise ValueError('invalid observer timing total')
    if abs(sum(shadow['observer_seconds_per_update'])-shadow['total_observer_seconds'])>1e-12:raise ValueError('timing total mismatch')
    grid=curve(states,np.zeros(600),refs)['grid']
    expected=[{k:r[k] for k in ('steps','qualified_label','maximum_closure_residual','reference_matches')} for r in grid]
    if shadow['checkpoints']!=expected or shadow['decision']!=proposal(grid)['decision']:raise ValueError('observer differs from offline gates')
    names=[f'trial-{index:02d}-shadow-block-{n:03d}.json' for n in range(12,601,12)]+[f'trial-{index:02d}-shadow-final-600.json']
    artifacts=shadow['artifacts']
    if [r['path'] for r in artifacts]!=names:raise ValueError('shadow artifact contract')
    for i,r in enumerate(artifacts):
        p=base/r['path'];bindings.append(record(p,r));saved=json.loads(p.read_text());end=min((i+1)*12,600)
        if saved['completed_steps']!=end or saved['artifacts']!=artifacts[:i] or saved['definition']!=definition:
            raise ValueError('sidecar progression differs')
        if saved['checkpoints']!=expected[:end//12] or saved['observer_seconds_per_update']!=shadow['observer_seconds_per_update'][:end]:raise ValueError('sidecar prefix differs')
        if saved['decision']!=proposal(grid[:end//12])['decision']:raise ValueError('sidecar decision differs')
        if saved['total_observer_seconds']!=sum(shadow['observer_seconds_per_update'][:end]):raise ValueError('sidecar timing total differs')
        if saved['status']!=('completed' if i==50 else 'running'):raise ValueError('sidecar status differs')
    return {'observer_seconds':shadow['total_observer_seconds'],'decision':shadow['decision'],'checkpoint_count':len(expected)}


def load_audit(path):
    a=json.loads(path.read_text())['audit'];bindings=[record(path)]
    if a['status']!='audit_passed' or len(a['rows'])!=8:raise ValueError('require complete eight-row audit')
    for b in a['bindings']:bindings.append(record(b['path'],b))
    results=[Path(b['path']) for b in a['bindings'] if Path(b['path']).name=='results.json']
    if len(results)!=1:raise ValueError('ambiguous result binding')
    return a,results[0].parent,bindings


def audit(current,baseline,references):
    a,base,bindings=load_audit(current);old,prior,oldbindings=load_audit(baseline);bindings+=oldbindings
    bindings.append(record(references,{'sha256':context.REF_HASH}))
    with np.load(references,allow_pickle=False) as d:refs=d['states']
    meta=json.loads((base/'metadata.json').read_text());result=json.loads((base/'results.json').read_text())
    definition=meta['shadow_readout']
    if definition['references']['sha256']!=context.REF_HASH:raise ValueError('shadow reference mismatch')
    rows=[]
    for i in range(8):
        row={'packet_index':i};rows.append(row)
        try:
            name=f'trial-{i:02d}-readback-final-600.npz'
            if not all(any(Path(b['path'])==folder/name for b in bound) for folder,bound in ((base,a['bindings']),(prior,old['bindings']))):raise ValueError('unbound final trajectory')
            with np.load(base/name,allow_pickle=False) as x,np.load(prior/name,allow_pickle=False) as y:
                states=x['states'];previous=y['states']
                if states.shape!=(601,896) or previous.shape!=states.shape:raise ValueError('trajectory shape mismatch')
                row.update(check_shadow(base,i,result['rows'][i]['shadow_readout'],states,refs,definition,bindings))
                row.update(maximum_coordinate_gap=float(np.max(np.abs(states-previous))),tokens_equal=bool(np.array_equal(x['token_ids'],y['token_ids'])),states_exactly_equal=bool(np.array_equal(states,previous)))
            row.update(current_final_label=a['rows'][i]['qualified_state'],baseline_final_label=old['rows'][i]['qualified_state'],final_labels_equal=a['rows'][i]['qualified_state']==old['rows'][i]['qualified_state'])
            row['status']='verified'
        except Exception as e:row.update(status='failed',error=str(e))
    return {'status':'audit_passed' if all(r['status']=='verified' for r in rows) else 'audit_failed','rows':rows,'bindings':bindings,'all_trajectories_exactly_equal':all(r.get('states_exactly_equal',False) and r.get('tokens_equal',False) and r.get('final_labels_equal',False) for r in rows),'scope':'Shadow/reference-access replay; timer excludes logging and report bookkeeping; no achieved early-stop speedup.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();
    for k in ('current','baseline','references','output'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();r=audit(args.current,args.baseline,args.references)
    with args.output.open('x') as f:json.dump({'utc':subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),'audit':r},f,indent=2);f.write('\n')
    print(json.dumps({'status':r['status'],'all_trajectories_exactly_equal':r['all_trajectories_exactly_equal']}))
    raise SystemExit(r['status']!='audit_passed')
