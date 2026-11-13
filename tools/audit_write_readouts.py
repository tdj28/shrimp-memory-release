#!/usr/bin/env python3
"""Exploratory visible-token collisions in independently audited writes.

Uses saved argmax diagnostics, not ordinary-language conversations. A collision
only limits this specific readout; it says nothing about a stronger observer.
"""
import argparse
import json
from pathlib import Path
import subprocess
import audit_p6_writes as auditor


def canonical_word(tokens):
    word = tuple(int(x) for x in tokens)
    if not word or any(x < 0 for x in word): raise ValueError('valid nonempty token word required')
    return min(word[i:] + word[:i] for i in range(len(word)))


def collision_report(rows, field):
    groups = {}
    for row in rows:
        key = tuple(row[field])
        group = groups.setdefault(key, {'tokens':list(key), 'states':set(), 'rows':[]})
        group['states'].add(row['decoded_state'])
        group['rows'].append({'run':row['run'],'trial_id':row['trial_id']})
    return [{'tokens':g['tokens'],'states':sorted(g['states']),'rows':g['rows'],
             'cross_state_collision':len(g['states'])>1} for g in groups.values()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); rows=[]; bindings=[]; excluded=[]
    for path in args.audit:
        audit=json.loads(path.read_text())['audit']
        if audit['status']!='audit_passed': raise ValueError('passing audit required')
        bindings.append(auditor.independent.file_record(path))
        by_name={Path(b['path']).name:b for b in audit['bindings']}
        for row in audit['rows']:
            matches=row['outcome']['reference_matches']
            if row['status']!='verified' or not row['outcome']['cycle_gate_passed'] or len(matches)!=1:
                excluded.append({'run':str(path),'trial_id':row['trial_id'],'reason':'unqualified state readout'})
                continue
            record={'run':str(path),'trial_id':row['trial_id'],'decoded_state':matches[0]}
            for stage,steps in (('endpoint',120),('hold',240)):
                binding=by_name[f"trial-{row['trial_id']:02d}-{stage}-final-{steps:03d}.npz"]
                bindings.append(auditor.independent.file_record(binding['path'],binding))
                with auditor.np.load(binding['path'],allow_pickle=False) as data:
                    tokens=data['token_ids']
                    if tokens.shape!=(steps,): raise ValueError('incomplete token sequence')
                    record[stage+'_last_six']=list(canonical_word(tokens[-6:]))
            rows.append(record)
    root=Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root): parser.error('save analysis outside repository')
    result={'utc':subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
      'source_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
      'source':auditor.independent.file_record(__file__),'bindings':bindings,'rows':rows,'excluded':excluded,
      'endpoint_groups':collision_report(rows,'endpoint_last_six'),'hold_groups':collision_report(rows,'hold_last_six'),
      'scope':'Exploratory phase-invariant six-token argmax readout. No general secrecy, classifier accuracy or natural-language channel claim.'}
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'output':str(args.output),'qualified_rows':len(rows),'excluded_rows':len(excluded),
                     'endpoint_groups':len(result['endpoint_groups']),'hold_groups':len(result['hold_groups'])}))


if __name__=='__main__': main()
