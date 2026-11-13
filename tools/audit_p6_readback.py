#!/usr/bin/env python3
"""Independently audit explicit vector readback and a static reference baseline."""
import argparse
import json
from pathlib import Path
import subprocess
import audit_p6_writes as common
import prepare_readback_packet as packet_tool
np = common.np
REF_HASH = 'd96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0'
INPUT_HASHES = {
    'weights':'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe',
    'vectors':'b630bb7bef06bde87f8e6a7ca5e8672f83d24bc0585bf19048e9e494a4357f41',
    'config':'18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45'}


def static_decode(state, refs):
    distances = np.linalg.norm(refs-state, axis=2).min(axis=1)
    matches = [label for label,d in zip(common.LABELS,distances) if d<=1e-6]
    order = np.argsort(distances)
    margin = float(distances[order[1]]-distances[order[0]])
    return {'distances':dict(zip(common.LABELS,map(float,distances))),
            'decoded':matches[0] if len(matches)==1 else None,
            'nearest_label':common.LABELS[int(order[0])] if margin>0 else None,
            'nearest_distance_margin':margin,
            'nearest_rule':'closest reference phase without distance rejection; exact ties unresolved'}


def audit(base, packet, key_path, references, reference_hash=REF_HASH, steps=120):
    if steps not in (120,600):raise ValueError("unsupported readback horizon")
    expected_calls,expected_seconds={120:(977,600),600:(4817,2700)}[steps]
    record = common.independent.file_record
    key = json.loads(key_path.read_text())
    decimals = key.get('decimal_places')
    if decimals not in (None, 4, 5):raise ValueError('unsupported transport precision')
    bindings = [record(key_path),record(packet,key['packet']),record(references,{'sha256':reference_hash})]
    order = packet_tool.packet_order().tolist()
    if [r['packet_index'] for r in key['rows']] != list(range(8)) or [r['source_trial'] for r in key['rows']] != order:
        raise ValueError('evaluation key permutation differs from design')
    with np.load(packet,allow_pickle=False) as a:
        if set(a.files)!={'states'}:raise ValueError('packet contains non-state fields')
        states=a['states']
    if states.shape!=(8,896) or states.dtype!=np.float64 or not np.isfinite(states).all():raise ValueError('invalid packet states')
    bindings.append(record(key['source_audit']['path'],key['source_audit']))
    sender=json.loads(Path(key['source_audit']['path']).read_text())['audit']
    if sender['status']!='audit_passed' or sender['protocol']!='sequence' or not sender['all_writes_correct']:
        raise ValueError('source byte not qualified')
    transport_errors=[]
    for i,r in enumerate(key['rows']):
        j=r['source_trial']; source=key['sources'][j]
        bindings.append(record(source['path'],source))
        bound=next(b for b in sender['bindings'] if Path(b['path']).name==f'trial-{j:02d}-hold-final-240.npz')
        if source['sha256']!=bound['sha256'] or sender['rows'][j]['outcome']['reference_matches']!=[r['expected_state']]:
            raise ValueError('sender source or label mismatch')
        with np.load(source['path'],allow_pickle=False) as a:
            original = a['states'][-1]
            transmitted = original if decimals is None else np.round(original, decimals=decimals)
            if not np.array_equal(transmitted,states[i]):raise ValueError('packet differs from prescribed sender transform')
            transport_errors.append({'packet_index':i,'maximum_coordinate_error':float(np.max(np.abs(original-states[i]))),
                                     'l2_error':float(np.linalg.norm(original-states[i]))})
    with np.load(references,allow_pickle=False) as a:refs=a['states']
    if refs.shape!=(2,6,896) or refs.dtype!=np.float64 or not np.isfinite(refs).all():raise ValueError('invalid references')
    bindings += [record(base/'results.json'),record(base/'metadata.json')]
    metadata=json.loads((base/'metadata.json').read_text())
    result=json.loads((base/'results.json').read_text())
    if any(metadata.get(k,{}).get('sha256')!=v for k,v in INPUT_HASHES.items()):
        raise ValueError('receiver model inputs differ from pinned hashes')
    observations=result.get('dtype_observations',{})
    for group,keys in {'rotary':('input','inverse_frequency','angles','cos','sin'),
                       'norm':('input','reduction','weight','output')}.items():
        if any(observations.get(group,{}).get(k)!='torch.float64' for k in keys):
            raise ValueError('missing float64 operation observations')
    if metadata['packet']['sha256']!=bindings[1]['sha256'] or metadata['controls']!=common.HOLD:
        raise ValueError('receiver packet or controls mismatch')
    if (metadata['max_calls'],metadata['max_seconds'],metadata['max_rss_gib'])!=(expected_calls,expected_seconds,8):raise ValueError('receiver limits changed')
    rows=[]
    for i,expected in enumerate(key['rows']):
        row={'packet_index':i,'source_trial':expected['source_trial'],'expected':expected['expected_state']}
        rows.append(row)
        try:
            raw=result['rows'][i];stage=raw['stage']
            if raw['packet_index']!=i or stage['name']!='readback' or stage['steps']!=steps or stage['controls']!=common.HOLD:
                raise ValueError('receiver schedule mismatch')
            check,trajectory=common.audit_stage(base,i,stage,bindings,expected_steps=steps)
            if check['status']!='verified':raise ValueError('incomplete receiver trajectory')
            if not np.array_equal(trajectory['states'][0],states[i]):raise ValueError('receiver start differs from packet')
            outcome=common.classify(trajectory['states'][1:],refs,expected['expected_state'])
            row.update(status='verified',stage=check,outcome=outcome,static_baseline=static_decode(states[i],refs))
        except Exception as error:row.update(status='failed',error=str(error))
    passed=len(result['rows'])==8 and result['status']=='completed' and result['model_calls']==expected_calls and all(r['status']=='verified' for r in rows)
    counts={k:sum(r.get('outcome',{}).get('classification')==k for r in rows) for k in ('correct','wrong','unresolved')}
    def bit(row,static=False,nearest=False):
        label=row.get('static_baseline',{}).get('nearest_label' if nearest else 'decoded') if static else (
            row['outcome']['reference_matches'][0] if row.get('outcome',{}).get('cycle_gate_passed') and len(row['outcome']['reference_matches'])==1 else None)
        return {'ad':'0','td':'1',None:'?'}[label]
    ordered=sorted(rows,key=lambda r:r['source_trial'])
    return {'status':'audit_passed' if passed else 'audit_failed','rows':rows,'bindings':bindings,
            'transport_decimal_places':decimals,'readback_steps':steps,
            'transport_errors':transport_errors,
            'outcome_counts':counts,'decoded_source_order':''.join(bit(r) for r in ordered),
            'static_decoded_source_order':''.join(bit(r,True) for r in ordered),
            'nearest_reference_decoded_source_order':''.join(bit(r,True,True) for r in ordered),
            'scope':'explicit vector transport; no language channel or secrecy test',
            'producer_source_commit':metadata['source_commit']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','packet','key','references','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,choices=(120,600),default=120)
    a=p.parse_args();result=audit(a.base,a.packet,a.key,a.references,steps=a.steps)
    a.output.write_text(json.dumps({'utc':subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),'audit':result},indent=2)+'\n')
    print(json.dumps({'status':result['status'],'counts':result['outcome_counts'],'decoded':result['decoded_source_order'],'static':result['static_decoded_source_order']}))
    return int(result['status']!='audit_passed')

if __name__=='__main__':raise SystemExit(main())
