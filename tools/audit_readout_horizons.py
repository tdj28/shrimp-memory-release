#!/usr/bin/env python3
"""Retrospective fixed-grid readout diagnostics; never an online stopping proof."""
import argparse
import json
from pathlib import Path
import subprocess
import audit_p6_readback as readback
common=readback.common
np=common.np


def curve(states,seconds,refs):
    if states.ndim!=2 or states.shape[1]!=896 or seconds.shape!=(len(states)-1,) or (len(states)-1)%12:
        raise ValueError('require full trajectory with initial state and twelve-step grid')
    rows=[]
    for end in range(12,len(states),12):
        measured=common.independent.recompute(states[1:end+1],refs,0)
        matches=measured['reference_matches']
        label=matches[0] if measured['cycle_gate_passed'] and len(matches)==1 else None
        rows.append({'steps':end,'qualified_label':label,'maximum_closure_residual':measured['maximum_l2_residual'],
                     'reference_matches':matches,'cumulative_saved_forward_seconds':float(seconds[:end].sum())})
    first=next((r for r in rows if r['qualified_label'] is not None),None)
    stable=next((r for i,r in enumerate(rows) if r['qualified_label'] is not None and all(s['qualified_label']==r['qualified_label'] for s in rows[i:])),None)
    later=[] if first is None else [r for r in rows if r['steps']>first['steps']]
    return {'grid':rows,'first_qualified_steps':None if first is None else first['steps'],
            'first_qualified_label':None if first is None else first['qualified_label'],
            'later_unresolved_after_first':any(r['qualified_label'] is None for r in later),
            'later_different_qualified_label':any(r['qualified_label'] not in (None,first['qualified_label']) for r in later) if first else False,
            'retrospective_stable_grid_steps':None if stable is None else stable['steps'],
            'final_qualified_label':rows[-1]['qualified_label'],'full_steps':len(states)-1}


def analyze(audit_path,refs):
    record=common.independent.file_record;document=json.loads(audit_path.read_text());a=document['audit']
    if a['status']!='audit_passed':raise ValueError('require passing independent audit')
    bindings=[record(audit_path)]
    for b in a['bindings']:bindings.append(record(b['path'],b))
    results=[Path(b['path']) for b in a['bindings'] if Path(b['path']).name=='results.json']
    if len(results)!=1:raise ValueError('require one current-run results binding')
    finals=[b for b in a['bindings'] if Path(b['path']).parent==results[0].parent and Path(b['path']).name.startswith('trial-') and ('-hold-final-' in Path(b['path']).name or '-readback-final-' in Path(b['path']).name)]
    if len(finals)!=len(a['rows']) or len({Path(b['path']).name for b in finals})!=len(finals):raise ValueError('require exactly one hold/readback trajectory per audited row')
    rows=[]
    for b in sorted(finals,key=lambda b:Path(b['path']).name):
        with np.load(b['path'],allow_pickle=False) as d:
            common.check_controls(d,common.HOLD);states=d['states'];seconds=d['step_seconds']
            if states.dtype!=np.float64 or not np.isfinite(states).all() or not np.isfinite(seconds).all() or np.any(seconds<0):raise ValueError('invalid saved arrays')
        rows.append({'trajectory':b,**curve(states,seconds,refs)})
    return {'audit':record(audit_path),'rows':rows,'bindings':bindings}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,action='append',required=True);p.add_argument('--references',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    reference=common.independent.file_record(a.references,{'sha256':readback.REF_HASH})
    with np.load(a.references,allow_pickle=False) as data:refs=data['states']
    runs=[analyze(p,refs) for p in a.audit]
    output={'utc':subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'references':reference,'runs':runs,'scope':'exploratory retrospective twelve-step grid; no guarantee between grid points or beyond saved horizon; online stopping and decoder cost unvalidated'}
    with a.output.open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps([{'audit':r['audit']['path'],'rows':[{k:v for k,v in row.items() if k not in ('grid','trajectory')} for row in r['rows']]} for r in runs]))

if __name__=='__main__':main()
