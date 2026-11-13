#!/usr/bin/env python3
"""Evaluate a fixed three-checkpoint readout proposal on hash-bound saved grids."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def proposal(grid):
    if not grid or [r['steps'] for r in grid]!=list(range(12,12*(len(grid)+1),12)):
        raise ValueError('require contiguous twelve-update grid')
    streak=0; previous=None; decision=None
    for row in grid:
        label=row['qualified_label']
        if label not in (None,'ad','td'):raise ValueError('invalid label')
        streak=streak+1 if label is not None and label==previous else (1 if label else 0)
        previous=label
        if streak==3 and decision is None:decision={'steps':row['steps'],'label':label}
    final=grid[-1]['qualified_label']
    after=[] if decision is None else [r for r in grid if r['steps']>decision['steps']]
    return {'decision':decision,'final_label':final,
            'disagrees_with_qualified_final':decision is not None and final is not None and decision['label']!=final,
            'final_unresolved_after_decision':decision is not None and final is None,
            'later_loss':any(r['qualified_label'] is None for r in after),
            'later_label_change':any(r['qualified_label'] not in (None,decision['label']) for r in after) if decision else False,
            'remaining_updates_after_decision':None if decision is None else grid[-1]['steps']-decision['steps']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();raw=args.analysis.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.sha256:raise ValueError('analysis hash mismatch')
    data=json.loads(raw)
    for run in data['runs']:
        for b in run['bindings']:
            path=Path(b['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=b['sha256'] or path.stat().st_size!=b['bytes']:raise ValueError('bound raw file changed')
    result={'utc':subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'analysis':{'path':str(args.analysis),'sha256':args.sha256},
            'runs':[{'audit':r['audit'],'rows':[proposal(x['grid']) for x in r['rows']]} for r in data['runs']],
            'scope':'Offline replay of fixed three-consecutive-checkpoint proposal; online decoder cost and reliability beyond saved trajectories unmeasured.'}
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')

if __name__=='__main__':main()
