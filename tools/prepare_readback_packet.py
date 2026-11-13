#!/usr/bin/env python3
"""Export eight audited terminal vectors; keep evaluation labels out of packet."""
import argparse
import json
from pathlib import Path
import subprocess
import audit_p6_writes as auditor
np = auditor.np


def packet_order():
    return np.random.Generator(np.random.PCG64(20261026)).permutation(8)


def prepare(audit_path, output, decimal_places=None):
    if decimal_places not in (None, 4, 5):
        raise ValueError('only exact or fixed four/five-decimal transport is specified')
    audit = json.loads(audit_path.read_text())['audit']
    if audit['status'] != 'audit_passed' or audit.get('protocol') != 'sequence' or not audit['all_writes_correct']:
        raise ValueError('require complete qualified sequence audit')
    if len(audit['rows']) != 8: raise ValueError('require eight source writes')
    bindings = {Path(r['path']).name:r for r in audit['bindings']}
    states, labels, sources = [], [], []
    for i,row in enumerate(audit['rows']):
        if row['trial_id'] != i or row['status'] != 'verified' or len(row['outcome']['reference_matches']) != 1:
            raise ValueError('unqualified source row')
        record = bindings[f'trial-{i:02d}-hold-final-240.npz']
        sources.append(auditor.independent.file_record(record['path'],record))
        with np.load(record['path'],allow_pickle=False) as data:
            if data['states'].shape != (241,896) or data['states'].dtype != np.float64:
                raise ValueError('invalid source trajectory')
            states.append(data['states'][-1].copy())
        labels.append(row['outcome']['reference_matches'][0])
    order = packet_order()
    output.mkdir(parents=True,exist_ok=False)
    packet = output/'packet.npz'
    transmitted = np.stack(states)[order]
    if decimal_places is not None:
        transmitted = np.round(transmitted, decimals=decimal_places)
    with packet.open('xb') as stream: np.savez_compressed(stream,states=transmitted)
    key = {'packet':auditor.independent.file_record(packet),'source_audit':auditor.independent.file_record(audit_path),
           'sources':sources,'permutation_seed':20261026, 'decimal_places':decimal_places,
           'rows':[{'packet_index':i,'source_trial':int(j),'expected_state':labels[j]} for i,j in enumerate(order)],
           'scope':'explicit float64 state-vector transport; labels and source order withheld from receiver inputs, not an OS security boundary'}
    (output/'evaluation-key.json').write_text(json.dumps(key,indent=2)+'\n')
    return key


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--decimal-places',type=int,choices=[4,5])
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root):p.error('save packet outside repository')
    key=prepare(args.audit,args.output,args.decimal_places)
    (args.output/'provenance.json').write_text(json.dumps({'utc':subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
        'source_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        'script':auditor.independent.file_record(__file__)},indent=2)+'\n')
    print(json.dumps({'packet':key['packet'],'evaluation_key':str(args.output/'evaluation-key.json')}))


if __name__=='__main__':main()
