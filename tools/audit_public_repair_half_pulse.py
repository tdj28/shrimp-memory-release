#!/usr/bin/env python3
"""Independent saved-output audit of opposite-bit writes from repaired states.

No producer/backend import or model replay. This rechecks returned bytes,
compact trajectories, explicit controls, cache evidence, and neutral readout.
The array formulas and old-format scalar checks follow the earlier independently
reviewed public-batch audit, now committed in one portable checker.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
import traceback

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent/'shrimp-memory-data'
OLD_SOURCE = 'f0c757ed2c7142bd2dd5cb78a796f69f7f49f4c4'
OLD_BASE = DATA/'audits/p6-writes/20261025T222347Z-writes'
DIM, VOCAB = 896, 151936
LABELS = ['ad', 'td']
ZERO, ONE = 'Write bit 0 to register R0.', 'Write bit 1 to register R0.'
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
HOLD = {'alpha': 0.07293697809562831, 'log10_temperature': -0.15395151820007266}
ENDPOINTS = [{'alpha': 0.07293697809562831, 'log10_temperature': -0.15350},
             {'alpha': 0.07299785589927693, 'log10_temperature': -0.15395151820007266}]
GROUPS = ('repair36-handoff36','repair36-handoff240','repair240-handoff36','repair240-handoff240')
AGES = (36,240,36,240)
PROFILE = 'public-repair-half-pulse'
REFERENCE_HASH = 'd96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0'
INPUT_HASHES = {'weights':'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe',
 'config':'18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45',
 'vectors':'b630bb7bef06bde87f8e6a7ca5e8672f83d24bc0585bf19048e9e494a4357f41'}
PRECISION = {'norm':{k:'torch.float64' for k in ('input','reduction','weight','output')},
 'rotary':{k:'torch.float64' for k in ('input','inverse_frequency','angles','cos','sin')}}
SOURCES = ['tools/run_public_control_continuation.py','tools/run_public_control_tolerance.py','tools/run_public_batch_writes.py','tools/run_public_command_memory.py',
 'tools/transparent_command_contract.py','tools/check_p6_writes.py','tools/check_p6_recovery.py',
 'tools/check_p6_hold.py','tools/validate_qwen_map.py','tools/benchmark_qwen_batch.py',
 'tools/run_public_half_pulse.py','tools/validate_qwen_batch_trajectory.py',
 'tools/run_public_repair_half_pulse.py','docs/public-repair-half-pulse.md','tests/test_public_repair_half_pulse.py',
 'docs/receipts/p6-writes.json','docs/receipts/p6-write-path-design.json',
 'docs/receipts/public-control-tolerance.json','docs/receipts/public-control-continuation.json',
 'tools/analyze_causal_geometry.py','tools/streaming_readout.py','tools/audit_p6_recovery.py','pyproject.toml','uv.lock']
LIMITS = dict(model_calls=1224, internal_seconds=1800, external_seconds=2100,
 termination_grace_seconds=30,threads=1,peak_rss_bytes=8*1024**3,external_vm_bytes=16*1024**3,
 working_raw_bytes=248*1024**2,finalization_reserve_bytes=8*1024**2,total_raw_bytes=256*1024**2,
 archive_bytes=352*1024**2,transfer_reservation_bytes=448*1024**2,source_upload_bytes=8*1024**2)

def require(ok,label):
    if not bool(ok):raise ValueError(label)

def finite_nonnegative(x):return type(x) in (int,float) and math.isfinite(x) and x>=0

def sha(b):return hashlib.sha256(b).hexdigest()

def binding(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'missing/symlink '+str(path));raw=path.read_bytes()
    return dict(path=str(path.resolve()),bytes=len(raw),sha256=sha(raw))

def checked(base,rec):
    path=Path(rec['path']);require(not path.is_absolute() and '..' not in path.parts,'unsafe relative binding')
    path=base/path;require(path.resolve().is_relative_to(base.resolve()),'binding escapes root')
    current=path
    while current!=base:require(not current.is_symlink(),'symlink component');current=current.parent
    got=binding(path);require(type(rec['bytes']) is int and all(got[k]==rec[k] for k in ('bytes','sha256')),'binding mismatch '+str(path));return path

def resolve_external(rec):
    s=rec['path']
    for remote,local in [('/home/ubuntu/shrimp-memory-data/',DATA),('/home/ubuntu/shrimp-memory/',ROOT)]:
        if s.startswith(remote):return local/s[len(remote):]
    return Path(s)

def check_external(rec,report):
    path=resolve_external(rec)
    suffix='/docs/receipts/p6-write-path-design.json'
    if not path.exists() and rec['path'].startswith('/home/ubuntu/shrimp-memory-data/checkouts/') and rec['path'].endswith(suffix):
        relative=suffix[1:];raw=subprocess.check_output(['git','-C',str(ROOT),'show',report['audited_source_commit']+':'+relative])
        require(len(raw)==rec['bytes'] and sha(raw)==rec['sha256'],'isolated checkout design differs from frozen Git')
        path=ROOT/relative
    got=binding(path);require(all(got[k]==rec[k] for k in ('bytes','sha256')),'external binding differs '+str(path));report['external_bindings'].append(got);return path

def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def scalar(a,kind=float):
    require(a.shape==(),'not scalar');return kind(a.item())

def action(bit):
    return dict(operation='write',register='R0',bit=bit,attractor_label=('ad','td')[bit],stages=[dict(name=name,steps=steps,**controls) for name,steps,controls in [('endpoint',120,ENDPOINTS[bit]),('hold',240,HOLD)]])

def typed_action_equal(got,bit):
    return (type(got) is dict and type(got.get('bit')) is int and got==action(bit)
        and type(got.get('stages')) is list and all(type(s.get('steps')) is int
        and type(s.get('alpha')) is float and type(s.get('log10_temperature')) is float for s in got['stages']))

def trajectory(z,controls):
    require(set(z)=={'states','first_state_index','token_ids','step_seconds','alpha','log10_temperature'},'trajectory field inventory')
    a=z['states'];require(a.ndim==2 and a.shape[1]==896 and a.dtype==np.float64 and len(a)>=1,'trajectory shape/dtype')
    n=len(a)-1;require(z['token_ids'].dtype==np.int64 and z['token_ids'].shape==(n,),'token shape/dtype')
    require(z['step_seconds'].dtype==np.float64 and z['step_seconds'].shape==(n,) and np.isfinite(z['step_seconds']).all() and (z['step_seconds']>=0).all(),'update times')
    require(z['first_state_index'].dtype.kind in 'iu' and scalar(z['first_state_index'],int)>=0,'first index')
    for key,value in controls.items():require(z[key].dtype==np.float64 and scalar(z[key])==value,'raw control '+key)
    return a

def neutral_match(saved,recomputed):
    gate=saved['cycle_gate'];require(gate['passed']==recomputed['cycle_gate_passed'],'cycle gate boolean')
    require(gate['closure_tolerance']==1e-8 and gate['divisor_separation']==1e-4 and saved['match_tolerance']==1e-6,'readout thresholds')
    require(np.allclose(gate['lag_six_phase_l2_residuals'],recomputed['lag_six_phase_l2_residuals'],rtol=1e-12,atol=1e-18),'lag-six residuals')
    require(math.isclose(gate['maximum_l2_residual'],recomputed['maximum_l2_residual'],rel_tol=1e-12,abs_tol=1e-18),'max residual')
    require(len(gate['proper_divisors'])==3,'divisor count')
    for old,new in zip(gate['proper_divisors'],recomputed['proper_divisors']):
        require(old['divisor']==new['divisor'] and old['excluded']==new['excluded'],'proper-divisor gate')
        require(np.allclose(old['phase_l2_gaps'],new['phase_l2_gaps'],rtol=1e-12,atol=1e-18) and math.isclose(old['maximum_l2_gap'],new['maximum_l2_gap'],rel_tol=1e-12,abs_tol=1e-18),'proper-divisor distances')
    require(len(saved['reference_distances'])==2,'reference count')
    for old,new in zip(saved['reference_distances'],recomputed['reference_distances']):
        require(old['label']==new['label'] and old['best_shift']==new['phase'] and math.isclose(old['distance'],new['distance'],rel_tol=1e-12,abs_tol=1e-18),'reference distance/phase')
    label=None if recomputed['decoded_bit'] is None else ('ad','td')[recomputed['decoded_bit']]
    require(saved['reference_matches']==recomputed['reference_matches'] and saved['qualified_label']==label,'reference classification')


def verify_return(base,report,commit):
    outer=base.parent;mf=json.loads((outer/'final-manifest.json').read_text());mb=binding(outer/'final-manifest.json');report['bindings'].append(mb);names=set()
    for rec in mf['files']:
        require(rec['path'] not in names,'duplicate return file');names.add(rec['path']);report['bindings'].append(binding(checked(outer,rec)))
    report['bindings'].append(binding(checked(outer,mf['archive'])))
    require(mf['archive']['bytes']<=LIMITS['archive_bytes'],'returned archive byte cap')
    for name in ('local-return-verification.json','transfer-closeout.json','completion.json'):report['bindings'].append(binding(outer/name))
    v=json.loads((outer/'local-return-verification.json').read_text());c=json.loads((outer/'transfer-closeout.json').read_text())
    require(v['status']=='verified' and v['archive']==mf['archive'] and v['manifest_sha256']==mb['sha256'],'local return verification')
    require(c['state']=='complete' and c['local_return_verified'] is True and c['verified_manifest_entries']==len(names) and c['manifest_sha256']==mb['sha256'],'transfer closeout')
    require((outer/'source-commit.txt').read_text().strip()==commit,'source return')
    report.update(verified_return_inventory_files=len(names),transfer_reservation_closed=True,transfer_closeout=c)

def same_bytes(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes(order='C')==b.tobytes(order='C')

def array_digest(x):
 header=str(x.dtype).encode()+b'\0'+json.dumps(list(x.shape),separators=(',',':')).encode()+b'\0'
 return hashlib.sha256(header+x.tobytes(order='C')).hexdigest()

def cache_digest(values):
 return hashlib.sha256(json.dumps({k:array_digest(v) for k,v in values.items()},sort_keys=True,separators=(',',':')).encode()).hexdigest()

def valid_matrix(x,shape,label):
 require(x.dtype==np.float64 and x.shape==shape and np.isfinite(x).all(),'finite float64 '+label)

def cache_arrays(values,batch,length):
 keys={f'layer_{i:02d}_{kind}' for i in range(24) for kind in ('key','value')}
 require(set(values)==keys|{'length'},'cache field inventory')
 require(values['length'].dtype==np.int64 and values['length'].shape==() and int(values['length'])==length,'cache length')
 for k in keys:valid_matrix(values[k],(batch,2,length,64),'cache '+k)
 return cache_digest(values)

def cache_expansion(scalar,expanded,length):
 scalar_hash=cache_arrays(scalar,1,length);expanded_hash=cache_arrays(expanded,2,length)
 for name,value in scalar.items():
  expected=value if name=='length' else np.repeat(value,2,axis=0)
  require(same_bytes(expanded[name],expected),'actual expanded cache differs from two exact scalar copies')
 return dict(scalar_sha256=scalar_hash,expanded_sha256=expanded_hash)

def reconstruct(initial,blocks,controls):
 """Blocks contain post-update states only, with no duplicated boundaries."""
 valid_matrix(initial,(2,DIM),'batch initial');states=[initial[None]];tokens=[];times=[];last=0
 for index,b in enumerate(blocks):
  require(set(b)=={'states','token_ids','step_seconds','first_state_index','last_state_index','alpha','log10_temperature'},'block field inventory')
  for key in ('first_state_index','last_state_index'):require(b[key].dtype==np.int64 and b[key].shape==(),'block index dtype/shape')
  first,end=int(b['first_state_index']),int(b['last_state_index']);n=end-first+1
  require(first==last+1 and 1<=n<=12,'noncontiguous or oversized post-update block')
  require(n==12 or index==len(blocks)-1,'partial block must be last')
  valid_matrix(b['states'],(n,2,DIM),'block states');valid_matrix(b['step_seconds'],(n,),'block times')
  require((b['step_seconds']>0).all(),'nonpositive update duration')
  t=b['token_ids'];require(t.dtype==np.int64 and t.shape==(n,2) and (t>=0).all() and (t<VOCAB).all(),'block tokens')
  for key,want in controls.items():require(b[key].shape==() and b[key].dtype==np.float64 and float(b[key])==want,'block controls')
  states.append(b['states']);tokens.append(t);times.append(b['step_seconds']);last=end
 return dict(states=np.concatenate(states),token_ids=np.concatenate(tokens) if tokens else np.empty((0,2),dtype=np.int64),step_seconds=np.concatenate(times) if times else np.empty(0,dtype=np.float64),updates=last)


def diagnostic_comparison(cached,full):
 for arrays in (cached,full):
  valid_matrix(arrays['input'],(2,DIM),'diagnostic input');valid_matrix(arrays['output'],(2,DIM),'diagnostic output');valid_matrix(arrays['probabilities'],(2,VOCAB),'diagnostic probabilities')
  p=arrays['probabilities'];require((p>=0).all() and (p<=1).all() and np.all(np.abs(p.sum(axis=1)-1)<=1e-12),'diagnostic probability simplex')
 require(same_bytes(cached['input'],full['input']),'diagnostic inputs differ')
 coordinate=float(np.abs(cached['output']-full['output']).max());probability=float(np.abs(cached['probabilities']-full['probabilities']).max());tokens=np.array_equal(cached['probabilities'].argmax(axis=1),full['probabilities'].argmax(axis=1))
 return dict(maximum_coordinate_gap=coordinate,maximum_probability_gap=probability,top_tokens_equal=bool(tokens),tolerance=1e-12,passed=coordinate<=1e-12 and probability<=1e-12 and bool(tokens))


def same_state_bytes(x,y):
    return x.dtype==y.dtype==np.dtype('float64') and x.shape==y.shape and x.tobytes(order='C')==y.tobytes(order='C')

def read_cycle(states,refs):
    if states.ndim!=2 or states.shape[0]<12 or states.shape[1]!=896 or states.dtype!=np.float64 or refs.shape!=(2,6,896) or refs.dtype!=np.float64 or not np.isfinite(states).all() or not np.isfinite(refs).all():
        raise ValueError('finite float64 trajectories and reference cycles required')
    cycle=states[-6:];previous=states[-12:-6]
    residuals=np.linalg.norm(cycle-previous,axis=1)
    divisors=[]
    for d in (1,2,3):
        gaps=np.linalg.norm(cycle-np.concatenate((cycle[d:],cycle[:d]),axis=0),axis=1)
        divisors.append(dict(divisor=d,phase_l2_gaps=gaps.tolist(),maximum_l2_gap=float(gaps.max()),excluded=bool(gaps.max()>1e-4)))
    distances=[]
    for index,ref in enumerate(refs):
        shift_distances=[]
        for shift in range(6):
            rotated=np.concatenate((ref[shift:],ref[:shift]),axis=0)
            shift_distances.append(float(np.linalg.norm(cycle-rotated,axis=1).max()))
        distances.append(dict(label=LABELS[index],distance=min(shift_distances),phase=int(np.argmin(shift_distances)),shift_distances=shift_distances))
    gate=bool(residuals.max()<=1e-8 and all(d['excluded'] for d in divisors))
    matches=[d['label'] for d in distances if d['distance']<=1e-6]
    decoded=LABELS.index(matches[0]) if gate and len(matches)==1 else None
    return dict(lag_six_phase_l2_residuals=residuals.tolist(),maximum_l2_residual=float(residuals.max()),proper_divisors=divisors,cycle_gate_passed=gate,reference_distances=distances,reference_matches=matches,decoded_bit=decoded)


def observe_prefix(states,refs):
    """Independent causal geometry/strict observation; state zero is excluded."""
    require(states.ndim==2 and states.shape[1:]==(DIM,) and len(states)>=13
        and (len(states)-1)%12==0 and states.dtype==np.float64
        and np.isfinite(states).all(),'complete finite hold observation prefix')
    valid_matrix(refs,(2,6,DIM),'observation references')
    previous={'geometry':None,'strict':None};counts={'geometry':0,'strict':0}
    decisions={'geometry':None,'strict':None};checkpoints=[]
    for step in range(12,len(states),12):
        distances=np.linalg.norm(refs-states[step],axis=2).min(axis=1)
        geometry=None if distances[0]==distances[1] else LABELS[int(distances[1]<distances[0])]
        strict=read_cycle(states[step-11:step+1],refs)
        label=None if strict['decoded_bit'] is None else LABELS[strict['decoded_bit']]
        for kind,value in [('geometry',geometry),('strict',label)]:
            counts[kind]=counts[kind]+1 if value is not None and value==previous[kind] else (1 if value else 0)
            previous[kind]=value
            if counts[kind]==3 and decisions[kind] is None:decisions[kind]=dict(steps=step,label=value)
        checkpoints.append(dict(steps=step,geometry_label=geometry,geometry_label_distances=distances.tolist(),
            geometry_absolute_margin=float(abs(distances[0]-distances[1])),strict_label=label,
            maximum_closure_residual=strict['maximum_l2_residual'],reference_matches=strict['reference_matches']))
    return dict(geometry_decision=decisions['geometry'],strict_decision=decisions['strict'],checkpoints=checkpoints)


def equal_observation(saved,actual):
    """Categorical fields are exact; recomputed floating distances have stated slack."""
    require(set(saved)==set(actual),'observer field inventory')
    require(saved['geometry_decision']==actual['geometry_decision']
        and saved['strict_decision']==actual['strict_decision'],'observer first decisions')
    require(len(saved['checkpoints'])==len(actual['checkpoints']),'observer checkpoint count')
    numerical={'geometry_label_distances','geometry_absolute_margin','maximum_closure_residual'}
    for old,new in zip(saved['checkpoints'],actual['checkpoints'],strict=True):
        require(set(old)==set(new),'observer checkpoint fields')
        require(all(old[k]==new[k] for k in set(new)-numerical),'observer checkpoint labels/grid')
        for key in numerical:
            require(np.allclose(old[key],new[key],rtol=1e-12,atol=1e-18),'observer distances '+key)


class Plan:
 __slots__=('bit','messages','endpoint','profile','groups','ages','source_index','source_label','phases')
 def __init__(self,bit):
  require(type(bit) is int and bit in (0,1),'explicit requested bit must be integer zero or one')
  self.bit=bit;self.profile=PROFILE;self.groups=GROUPS
  self.ages=AGES
  self.messages=((ZERO,ZERO) if bit==0 else (ONE,ONE))
  self.endpoint=ENDPOINTS[bit].copy()
  self.source_index=bit;self.source_label=LABELS[self.source_index];self.phases=(0,1)



def batch_state(base,rec):
 value=np.load(checked(base,rec),allow_pickle=False);valid_matrix(value,(2,896),'saved state')
 require(rec['dtype']=='float64' and rec['shape']==[2,896] and rec['array_sha256']==sha(value.tobytes()),'state metadata/hash');return value

def walk_bindings(v,report):
 if isinstance(v,dict):
  if {'path','bytes','sha256'}<=set(v):check_external(v,report)
  for k,x in v.items():
   if k!='manifest':walk_bindings(x,report)
 elif isinstance(v,list):
  for x in v:walk_bindings(x,report)

def references(base,report):
 if not (base/'qualified-inputs.json').exists():return None
 q=json.loads((base/'qualified-inputs.json').read_text());walk_bindings(q,report)
 raw=npz(check_external(q['references']['references'],report))['states']
 require(raw.shape==(2,6,896) and raw.dtype==np.float64 and np.isfinite(raw).all(),'reference arrays')
 for key in ('source_trajectories','control_trajectories'):
  records=q['references'][key] if key=='source_trajectories' else q['recovery_prerequisite'][key]
  require(len(records)==2,'prerequisite trajectory count')
  for i,rec in enumerate(records):
   arr=trajectory(npz(check_external(rec,report)),HOLD);require(arr.shape==(121,896) and read_cycle(arr[1:],raw)['decoded_bit']==i,'prerequisite readout')
   if key=='source_trajectories':require(arr[-6:].tobytes()==raw[i].tobytes(),'reference extraction')
   else:require(same_state_bytes(arr[0],raw[i,0]),'recovery start')
 return raw





def diagnostic_record(base,slot,initial):
 require(slot['artifact']['path']==f'call-{slot["call"]:03d}-diagnostic.npz','diagnostic filename')
 z=npz(checked(base,slot['artifact']));require(set(z)=={'input','input_after','output','probabilities'},'complete diagnostic arrays')
 for k,shape in [('input',(2,896)),('input_after',(2,896)),('output',(2,896)),('probabilities',(2,VOCAB))]:valid_matrix(z[k],shape,'diagnostic '+k)
 require(same_bytes(z['input'],initial) and same_bytes(z['input'],z['input_after']),'diagnostic input mutation')
 for k,field in [('input','input_sha256'),('output','output_sha256'),('probabilities','probabilities_sha256')]:require(array_digest(z[k])==slot[field],'diagnostic array fingerprint')
 require(slot['top_tokens']==z['probabilities'].argmax(axis=1).tolist(),'diagnostic argmax tokens');return z

def audit_cache_stage(base,stage,slots,initial,prefix_length):
 sid=stage['stage_id'];report={};prefix=None;expanded=None
 if stage.get('scalar_prefix_cache'):
  require(stage['scalar_prefix_cache']['path']==f'stage-{sid:02d}-scalar-prefix.npz','scalar prefix filename');prefix=npz(checked(base,stage['scalar_prefix_cache']));report['scalar_prefix_sha256']=cache_arrays(prefix,1,prefix_length)
 if stage.get('expanded_prefix_cache'):
  require(prefix is not None and stage['expanded_prefix_cache']['path']==f'stage-{sid:02d}-expanded-prefix.npz','expanded prefix filename');expanded=npz(checked(base,stage['expanded_prefix_cache']))
  report['expanded_sha256']=cache_digest(expanded)
  try:cache_expansion(prefix,expanded,prefix_length);report['expanded_snapshot_valid']=True
  except ValueError as error:report.update(expanded_snapshot_valid=False,expanded_snapshot_error=str(error))
  if stage['status']=='completed':require(report['expanded_snapshot_valid'],'completed stage has invalid expanded snapshot')
 if stage.get('final_scalar_cache_sha256') is not None:
  # Final cache is represented by a source-bound observation hash; the initial
  # scalar and actual first expanded tensors are independently rehashed here.
  unchanged=stage['final_scalar_cache_sha256']==report.get('scalar_prefix_sha256')
  require(stage['final_scalar_cache_unchanged']==unchanged,'final cache declaration differs')
  if stage['status']=='completed':require(unchanged,'completed stage has mutated final cache')
  report['final_scalar_cache_hash_unchanged']=unchanged
 comps=[x for x in slots if x['category']=='comparison']
 if stage['cache_comparison'] is not None:
  require(len(comps)==2 and all(x['status']=='returned' for x in comps),'diagnostic comparison lacks both returns')
  left,right=[diagnostic_record(base,x,initial) for x in comps];q=diagnostic_comparison(left,right)
  expected=dict(passed=q['passed'],maximum_coordinate_gap=q['maximum_coordinate_gap'],maximum_probability_gap=q['maximum_probability_gap'],outputs_exact=bool(np.array_equal(left['output'],right['output'])),probabilities_exact=bool(np.array_equal(left['probabilities'],right['probabilities'])),top_tokens_equal=q['top_tokens_equal'],tolerance=1e-12)
  require(stage['cache_comparison']==expected,'diagnostic equivalence result differs');report['diagnostic']=q
 if stage['status']=='completed':require(prefix is not None and expanded is not None and stage['cache_comparison'] is not None,'completed stage missing cache evidence')
 return report,prefix,expanded

def inspect_call(base,slot,arrays,cache,prefix_length):
 """Saved data validates emitted state fingerprints, not unretained probabilities."""
 if slot['status']=='not_started':return None
 n=slot['call'];category=slot['category'];sid=slot['stage_id']
 result=dict(call=n,stage_id=sid,category=category,status=slot['status'],step=slot['step'])
 if category=='prefix':
  if slot['status']=='returned':require(slot['dtype_observations']==PRECISION,'prefix effective precision')
  return result
 initial=arrays['states'][0] if slot['step']==0 else arrays['states'][min(slot['step']-1,arrays['updates'])]
 if slot.get('input_sha256') is not None:require(slot['input_sha256']==array_digest(initial),'per-call input differs from reconstructed carried state')
 returned=slot.get('output_sha256') is not None;carried=slot.get('state_carried',False)
 require(type(slot.get('forward_attempted')) is bool,'explicit map-forward attempt marker')
 if slot['status']=='returned':
  require(returned and slot['dtype_observations']==PRECISION and slot['output_dtype']==slot['probability_dtype']=='torch.float64','returned precision observations')
  require(all(slot[k] is True for k in ('cache_unchanged','input_unchanged','precision_passed','outputs_valid','expanded_cache_valid')),'returned structural gates')
  require(slot['cache_before_sha256']==slot['cache_after_sha256']==cache['scalar_prefix_sha256'],'per-call scalar prefix fingerprint')
  require(slot['cache_length_before']==slot['cache_length_after']==prefix_length,'per-call scalar prefix lengths')
  if slot['cached']:require(slot['expanded_cache_sha256']==cache['expanded_sha256'] and slot['used_cache_length_after']==prefix_length+1,'per-call expanded cache fingerprint')
  else:require('expanded_cache_sha256' not in slot and 'used_cache_length_after' not in slot,'full-prefix call claims expanded cache')
  require(finite_nonnegative(slot['seconds']) and slot['seconds']>0 and len(slot['probabilities_sha256'])==64,'call time/probability hash')
  if category=='update':require(carried,'returned update not carried')
 if carried:
  require(category=='update' and arrays['updates']>=slot['step'],'carried update absent from blocks')
  x=arrays['states'][slot['step']];tokens=arrays['token_ids'][slot['step']-1]
  require(slot['output_sha256']==array_digest(x) and slot['top_tokens']==tokens.tolist(),'per-update output/tokens differ from blocks')
  require(math.isclose(slot['seconds'],arrays['step_seconds'][slot['step']-1],rel_tol=0,abs_tol=0),'per-update timing differs from block')
 if slot.get('artifact') is not None:
  suffix='diagnostic' if category=='comparison' else 'failed';require(slot['artifact']['path']==f'call-{n:03d}-{suffix}.npz','call raw artifact filename')
  z=npz(checked(base,slot['artifact']));require('input' in z and same_bytes(z['input'],initial),'failed/diagnostic original input')
  returned=returned or ('output' in z and 'probabilities' in z)
  for k,field in [('input','input_sha256'),('output','output_sha256'),('probabilities','probabilities_sha256')]:
   if k in z and field in slot:require(array_digest(z[k])==slot[field],'failed/diagnostic raw fingerprint '+k)
  if slot['status']=='returned':require(category=='comparison','unexpected full update artifact')
  result['raw_artifact']=slot['artifact'];result['saved_fields']=sorted(z)
 result.update(forward_return_recorded=returned,state_carried=carried)
 return result


def controls_for(bit, group, local_stage):
    Plan(bit)
    require(type(group) is int and 0 <= group < 4 and local_stage in (0,1), 'control index')
    return (HOLD if local_stage == 1 else ENDPOINTS[bit]).copy()


def group_metadata(group):
    require(type(group) is int and 0 <= group < 4, 'group identity')
    age=AGES[group]
    return dict(hold_age=age,age_index=group,historical_hold_updates=(36,240)[group//2]+age,
        source_failed_hold_age=(36,240)[group//2],historical_total_updates=240+(36,240)[group//2]+age,total_path_updates=540+(36,240)[group//2]+age)


def profile_payload(record,profile,label):
    """The continuation identity is explicit on every scheduled object."""
    require(record.get('profile')==profile==PROFILE,label+' profile')
    return {key:value for key,value in record.items() if key!='profile'}


def expected_calls():
    names=GROUPS
    out=[]
    for group in range(4):
        for local,(stage,n) in enumerate((('intervention',60),('hold',240))):
            for category,step,cached in [('prefix',0,None),('comparison',0,True),('comparison',0,False)]+[
                    ('update',i,True) for i in range(1,n+1)]:
                out.append(dict(call=len(out)+1,group_id=group,group=names[group],stage_id=2*group+local,
                    stage=stage,category=category,step=step,cached=cached))
    return out


# Fixed input identities were prepared from complete independent phase audits.
SOURCE_PHASE_COMMIT = '5cbb3ec5800a7eadb24cbcf1fbe6f23f23b68d9b'
PREPARATION = DATA/'preparations/public-repair-handoff/20261026T210031Z'
PACKET_HASHES = ('6ae4b79fae84ee12b01f7c66d2498ef0dcb4ff96ad85fb219eb43192a7b52a28',
                 '8e834c61fac5b0590f583ffb123374192cb271b6ce06ae0c2eb30f82d3476962')
SOURCE_AUDIT_HASHES = ('b74f4ac56f01d0773aed1927fded173ec65da68e717a8312cd37a998664fad49',
                       '99c8b90861afa5213bac3e1f604001b37b1d6d7bfbace161813b0ce7cb460f53')
HISTORY_SCOPE = 'not_applicable_new_repair_half_pulse'
OBSERVER_HASHES = {'tools/analyze_causal_geometry.py':'7c9d7495d60dc15a650d978b193e682c31c6faa85ff890a184e1b4a850cd0dfd',
    'tools/streaming_readout.py':'15101e9d809e92a85aabf44dea6b3eb09ffeb5909a071f43191d45ac8685cd46',
    'tools/audit_p6_recovery.py':'dff1be03ec6cacbe00d4093a9162725a48b3778f42d20680e2767593f88835b5'}


def old_endpoint(bit):
    value=ENDPOINTS[bit].copy();value['log10_temperature' if bit==0 else 'alpha']-=1e-4 if bit==0 else 1e-5
    return value


def load_hold_packet(bit,refs,report,packet_path=None):
    """Independently reconstruct the four exact handoff batches; no producer imports."""
    Plan(bit);sourcebit=1-bit
    path=Path(packet_path) if packet_path is not None else PREPARATION/f'bit{bit}'/'packet.json'
    rec=binding(path);require(rec['sha256']==PACKET_HASHES[bit],'fixed hold packet hash');report['external_bindings'].append(rec)
    packet=json.loads(path.read_text())
    require(packet['status']=='literal_saved_states_verified_no_model_replay' and packet['profile']=='public-repair-handoff-inputs'
        and packet['next_write_bit']==bit and packet['source_repaired_bit']==sourcebit and packet['source_commit']==SOURCE_PHASE_COMMIT,'packet identity')
    proofpath=check_external(packet['source_audit'],report);proof=json.loads(proofpath.read_text())
    require(packet['source_audit']['sha256']==SOURCE_AUDIT_HASHES[bit] and proof['status']=='passed' and proof['producer_status']=='completed'
        and proof['profile']=='public-hold-repair' and proof['requested_bit']==sourcebit and proof['audited_source_commit']==SOURCE_PHASE_COMMIT
        and proof['all_rows_preserved'] and proof['transfer_reservation_closed'],'source audit identity')
    unique={}
    for item in proof['bindings']+proof['external_bindings']:
        key=str(resolve_external(item).resolve());pair=(item['bytes'],item['sha256'])
        if key in unique:require(unique[key]==pair,'conflicting source binding')
        else:check_external(item,report);unique[key]=pair
    require(len(proof['rows'])==8,'source all rows')
    for i,row in enumerate(proof['rows']):
        require(row['row_id']==i and row['origin_reference_index']==bit and row['origin_reference_phase']==i%2
            and row['numerical_diagnostics_passed'] and row['validated_outcome']==('passed' if i//2 in (1,3) else 'failed'),'fixed prior source row')
    rawpath=check_external(packet['source_result'],report);raw=json.loads(rawpath.read_text());base=rawpath.parent
    require(raw['profile']=='public-hold-repair' and raw['requested_bit']==sourcebit and raw['source']['source_commit']==SOURCE_PHASE_COMMIT,'source raw identity')
    for item in packet['source_files']:check_external(item,report)
    artifacts=[];names=[];expected=set()
    for sid,n in ((2,120),(3,240),(6,120),(7,240)):
        expected|={f'stage-{sid:02d}-initial.npy',f'stage-{sid:02d}-trajectory.json'}
        expected|={f'stage-{sid:02d}-block-{k:03d}.npz' for k in range(12,n+1,12)}
    for item in packet['source_artifacts']:
        p=check_external(item,report);require(p.parent==base,'artifact outside original run');names.append(p.name);artifacts.append(binding(p))
    require(len(names)==len(set(names)) and set(names)==expected,'exact source inventory')
    selected=[]
    for ix,gid in enumerate((1,3)):
        arrays={}
        for sid,n in ((2*gid,120),(2*gid+1,240)):
            stage=raw['stages'][sid];controls=ENDPOINTS[sourcebit] if sid%2==0 else HOLD
            require(stage['stage_id']==sid and stage['group_id']==gid and stage['status']=='completed'
                and stage['steps']==stage['completed_steps']==stage['attempted_steps']==stage['returned_steps']==n
                and stage['controls']==controls and stage['cache_comparison']['passed'],'source stage')
            initial=batch_state(base,stage['initial'])
            if sid%2:require(same_bytes(initial,arrays[sid-1]['states'][-1]),'source carried state')
            blocks=[]
            for end,item in zip(range(12,n+1,12),stage['blocks'],strict=True):
                require(item['path']==f'stage-{sid:02d}-block-{end:03d}.npz','source block ordering');blocks.append(npz(checked(base,item)))
            value=reconstruct(initial,blocks,controls);require(value['updates']==n,'source update count');arrays[sid]=value
            saved=json.loads(checked(base,stage['trajectory']).read_text())
            require(saved==dict(stage_id=sid,initial=stage['initial'],blocks=stage['blocks'],completed_steps=n,state_shape=[n+1,2,DIM],status='completed',controls=controls,format='initial plus nonoverlapping post-update blocks'),'source trajectory descriptor')
        a,h=arrays[2*gid]['states'],arrays[2*gid+1]['states'];history=packet['source_histories'][ix]
        require(history==dict(source_group_id=gid,intervention_states_sha256=array_digest(a),hold_states_sha256=array_digest(h),endpoint_to_hold_bitwise_equal=True),'source history digest')
        selected.extend([h[36],h[240]])
    selected=np.stack(selected);zpath=check_external(packet['states_npz'],report);saved=npz(zpath)
    require(set(saved)=={'states','source_hold_ages','handoff_ages','origin_reference_phases'},'input NPZ schema')
    valid_matrix(saved['states'],(4,2,DIM),'handoff states');require(same_bytes(saved['states'],selected),'literal handoff state substitution')
    for key,values in [('source_hold_ages',[36,36,240,240]),('handoff_ages',list(AGES)),('origin_reference_phases',[0,1])]:
        require(saved[key].dtype==np.int64 and saved[key].shape==(len(values),) and saved[key].tolist()==values,'input order/dtype')
    require(packet['all_states_array_sha256']==array_digest(selected) and len(packet['slots'])==4 and packet['source_producer_total_calls']==raw['model_calls']==1464,'packet count/hash')
    for g,slot in enumerate(packet['slots']):
        oldage=(36,240)[g//2];age=AGES[g]
        expected_slot=dict(group_id=g,name=GROUPS[g],source_repaired_bit=sourcebit,next_write_bit=bit,source_group_id=1+2*(g//2),source_stage_id=3+4*(g//2),source_hold_age=oldage,handoff_age=age,origin_reference_index=bit,origin_reference_phases=[0,1],historical_perturbed_endpoint_updates=120,historical_failed_hold_updates=oldage,historical_repair_endpoint_updates=120,historical_repaired_hold_updates=age,historical_path_updates=240+oldage+age,new_path_updates=360,total_path_updates=600+oldage+age,batch_array_sha256=array_digest(selected[g]),row_array_sha256=[array_digest(x) for x in selected[g]])
        require(slot==expected_slot,'packet slot identity/history')
    return selected,dict(packet_binding=rec,packet=packet,source_audit_binding=binding(proofpath),source_result_binding=binding(rawpath),source_artifact_bindings=artifacts,states_binding=binding(zpath),all_age_initial_sha256=array_digest(selected))


def hold_repair_inputs(base,result,refs,report,plan):
    path=base/'hold-inputs.json'
    if not path.exists():return None
    require(refs is not None,'handoff inputs before references')
    initial,expected=load_hold_packet(plan.bit,refs,report);saved=json.loads(path.read_text())
    require(set(saved)==set(expected)|{'scope'} and saved['packet']==expected['packet'] and saved['all_age_initial_sha256']==expected['all_age_initial_sha256'],'input packet structure/hash')
    args=result['source']['arguments']
    for key,arg in [('packet_binding','hold_packet'),('states_binding','hold_states'),('source_audit_binding','source_audit')]:
        got,want=saved[key],expected[key]
        require(set(got)=={'path','bytes','sha256'} and got['path']==args[arg] and all(got[k]==want[k] for k in ('bytes','sha256')),'copied input binding')
    require(saved['source_result_binding']==dict(path=str(Path(args['source_run'])/'result.json'),bytes=expected['source_result_binding']['bytes'],sha256=expected['source_result_binding']['sha256']),'source result copy')
    require(args['hold_packet_sha256']==PACKET_HASHES[plan.bit] and len(saved['source_artifact_bindings'])==len(expected['source_artifact_bindings']),'packet argument/inventory')
    for got,want in zip(saved['source_artifact_bindings'],expected['source_artifact_bindings'],strict=True):
        require(got==dict(path=str(Path(args['source_run'])/Path(want['path']).name),bytes=want['bytes'],sha256=want['sha256']),'copied source artifact')
    report['hold_inputs']=saved;report['source_history_cost']=dict(source_model_calls=expected['packet']['source_producer_total_calls'],histories=expected['packet']['slots'])
    return initial


def source_design(base,result,commit,report,plan):
    src=json.loads((base/'source.json').read_text())
    require(src==result['source'] and src['source_commit']==commit,'source identity')
    require(src['arguments']['requested_bit']==str(plan.bit),'source argument bit')
    require(src['arguments'].get('profile')==plan.profile,'source argument profile')
    require(result.get('profile')==plan.profile,'result profile')
    require(len(src['files'])==len(SOURCES),'source inventory')
    for rec,path in zip(src['files'],SOURCES,strict=True):
        require(rec['path']==Path(path).name,'source basename order')
        raw=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+path])
        require(len(raw)==rec['bytes'] and sha(raw)==rec['sha256'],'Git source mismatch '+path)
    report['git_source_bindings_verified']=SOURCES
    tree=ast.parse(subprocess.check_output(['git','-C',str(ROOT),'show',commit+':tools/validate_qwen_map.py'],text=True))
    prefix=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='PREFIX_IDS' for t in n.targets))
    design=json.loads((base/'design.json').read_text())
    require(design.get('profile')==plan.profile,'design profile')
    require(type(design['requested_bit']) is int and design['requested_bit']==plan.bit,'design bit')
    require(design['public_messages']==list(plan.messages),'public messages')
    require(type(design['origin_reference_index']) is int and design['origin_reference_index']==plan.source_index
        and design['origin_reference_label']==plan.source_label and design['origin_reference_phases']==[0,1]
        and all(type(v) is int for v in design['origin_reference_phases']) and design['source_stage_ids']==[3,3,7,7]
        and design['source_endpoint_stage_ids']==[2,2,6,6] and design['hold_ages']==list(AGES)
        and design['source_result_row_ids']==[[2,3],[2,3],[6,7],[6,7]] and design['historical_endpoint_updates']==240
        and design['hold_packet_sha256']==PACKET_HASHES[plan.bit] and design['source_phase_commit']==SOURCE_PHASE_COMMIT,
        'design endpoint origin and historical path')
    require(design['limits']==LIMITS,'frozen resource limits')
    require(design['schedule']==dict(groups=4,batch_size=2,rows=8,stages=8,intervention_steps=60,
        hold_steps=240,batch_updates=1200,row_updates=2400,prefix_calls=8,comparison_calls=16,
        total_calls=1224,hold_state_initializations=4,between_group_resets=3,reference_initializations=0,
        historical_endpoint_updates_per_row=240,historical_hold_ages_per_group=[72,276,276,480],
        new_updates_per_row=300,total_path_updates_per_group=[612,816,816,1020]),'schedule counts')
    require(design['map']==dict(model=MODEL,revision=REVISION,prefix_ids=prefix,hold=HOLD,
        endpoints=dict(ad=ENDPOINTS[0],td=ENDPOINTS[1])),'map identity')
    require(design['diagnostic_tolerance']==1e-12 and 'nominal_trajectory_tolerance' not in design,'diagnostic tolerance/no scalar equivalence')
    envelope=design['artifact_envelope']
    envelope_fields=dict(original_group_bytes=28627392,groups=4,
        estimated_uncompressed_envelope_bytes=4*28627392,input_metadata_allowance_bytes=4*1024**2,
        failed_return_reserved_bytes=2473984,event_allowance_bytes=16384,
        total_working_requirement_bytes=4*28627392+4*1024**2+2473984+16384)
    require(all(envelope.get(k)==v for k,v in envelope_fields.items())
        and envelope['total_working_requirement_bytes']<LIMITS['working_raw_bytes'],'raw envelope')
    require(len(design['realized_stages'])==8,'realized stage count')
    for sid,stage in enumerate(design['realized_stages']):
        group,local=divmod(sid,2)
        require(profile_payload(stage,plan.profile,'realized stage design')==dict(stage_id=sid,group_id=group,
            name=('intervention','hold')[local],steps=(60,240)[local],age_index=group,hold_age=AGES[group],
            controls=controls_for(plan.bit,group,local),nominal_controls=(plan.endpoint,HOLD)[local]),'realized stage design')
    require(len(design['groups'])==4,'design group count')
    for gid,group in enumerate(design['groups']):
        profile_payload(group,plan.profile,'design group')
        require(group['group_id']==gid and group['name']==plan.groups[gid]
            and group['age_index']==gid and group['hold_age']==AGES[gid]
            and group['intervention']=='write-opposite-bit'
            and group['intervention_controls']==controls_for(plan.bit,gid,0),'design group identity')
    if (base/'model-inputs.json').exists():
        inputs=json.loads((base/'model-inputs.json').read_text())
        require(set(inputs)==set(INPUT_HASHES),'model input inventory')
        for key,digest in INPUT_HASHES.items():
            require(inputs[key]['sha256']==digest and type(inputs[key]['bytes']) is int and inputs[key]['bytes']>0,'model hash '+key)
        report['remote_model_hash_declarations']=inputs
    return design


def incremental_failure(base,report):
    """An interrupted shard is evidence only after a separately checked full return."""
    path=base.parent/'incremental-return-failure.json'
    if not path.exists():return None
    record=binding(path);value=json.loads(path.read_text())
    require(type(value.get('error')) is str and value['error']
        and value.get('final_return_still_required') is True,'invalid incremental failure record')
    report['bindings'].append(record)
    report['incremental_return_failure']=dict(binding=record,record=value)
    return value if report.get('transfer_reservation_closed') is True else None


def verify_group_shard(base,directory,group,report):
    """Rehash tar members in place; never extract or execute collector archives."""
    names={p.name for p in directory.iterdir()}
    require(names=={'archive.tar','manifest.json','verified.json'},'incomplete group shard inventory')
    manifest_path=directory/'manifest.json';manifest=json.loads(manifest_path.read_text())
    require(set(manifest)=={'group_id','files','archive'} and type(manifest['group_id']) is int
        and manifest['group_id']==group,'group shard manifest schema')
    require(type(manifest['files']) is list and set(manifest['archive'])=={'bytes','sha256'},'group shard inventory schema')
    archive=checked(directory,dict(path='archive.tar',**manifest['archive']))
    require(archive.stat().st_size<=40*1024**2,'group shard archive cap')
    marker_path=base/f'group-{group:02d}.json';marker=json.loads(marker_path.read_text())
    require(type(marker['group_id']) is int and marker['group_id']==group
        and marker['status']=='completed' and type(marker['artifacts']) is list,'shard completed group marker')
    marker_binding=binding(marker_path);marker_binding['path']=marker_path.name
    require(manifest['files']==marker['artifacts']+[marker_binding],'shard inventory differs from group marker')
    expected={}
    for rec in manifest['files']:
        name=rec['path']
        require(type(name) is str and Path(name).name==name and name not in ('','.','..')
            and name not in expected and set(rec)=={'path','bytes','sha256'},'unsafe/duplicate shard filename')
        require(type(rec['bytes']) is int and rec['bytes']>=0,'shard file size')
        report['bindings'].append(binding(checked(base,rec)));expected[name]=rec
    require(sum(r['bytes'] for r in expected.values())<=40*1024**2,'group shard member cap')
    with tarfile.open(archive,'r:') as stream:
        members=stream.getmembers()
        require(len(members)==len(expected) and len({m.name for m in members})==len(members)
            and {m.name for m in members}==set(expected),'group shard tar inventory')
        for member in members:
            rec=expected[member.name]
            require(member.isfile() and member.size==rec['bytes'],'group shard member type/size')
            digest=hashlib.sha256();count=0
            with stream.extractfile(member) as payload:
                for chunk in iter(lambda:payload.read(1024**2),b''):
                    count+=len(chunk);digest.update(chunk)
            require(count==rec['bytes'] and digest.hexdigest()==rec['sha256'],'group shard member hash')
    verified=json.loads((directory/'verified.json').read_text())
    require(verified==dict(group_id=group,files=len(expected),manifest_sha256=binding(manifest_path)['sha256']),
        'group shard verified receipt differs')
    bindings=[binding(directory/name) for name in sorted(names)];report['bindings'].extend(bindings)
    return dict(group_id=group,status='verified',files=len(expected),bindings=bindings,
        archive_bytes=archive.stat().st_size,marker=marker_binding)


def return_shards(base,report,failure):
    directory=base/'return-shards'
    if not directory.exists():return []
    require(directory.is_dir() and not directory.is_symlink(),'invalid return shard directory')
    summaries=[]
    for child in sorted(directory.iterdir()):
        require(not child.is_symlink(),'return operational symlink')
        if child.name=='transfer.lock':
            require(child.is_file() and child.stat().st_size==0,'invalid transfer lock')
            report['bindings'].append(binding(child));continue
        require(child.name in {f'group-{i:02d}' for i in range(4)} and child.is_dir(),
            'unknown return operational files')
        files=list(child.iterdir());allowed={'archive.tar','manifest.json','verified.json'}
        require(all(p.name in allowed and p.is_file() and not p.is_symlink() for p in files),
            'unknown group shard files')
        require(all(p.stat().st_size<=40*1024**2 for p in files),'partial group shard cap')
        group=int(child.name[-2:])
        try:summary=verify_group_shard(base,child,group,report)
        except (ValueError,KeyError,TypeError,FileNotFoundError,tarfile.TarError,json.JSONDecodeError) as error:
            # A claimed verified shard must pass, even if another delivery failed.
            require(failure is not None and not (child/'verified.json').exists(),
                'unverified group shard without documented final-return recovery: '+str(error))
            marker=base/f'group-{group:02d}.json';value=json.loads(marker.read_text())
            require(value['group_id']==group and value['status']=='completed','partial shard lacks final completed marker')
            bindings=[binding(p) for p in sorted(files)];report['bindings'].extend(bindings)
            summary=dict(group_id=group,status='retained_incomplete',error=str(error),bindings=bindings,
                used_as_scientific_evidence=False,full_final_return_verified=True)
        summaries.append(summary)
    report['incremental_group_shards']=summaries
    return summaries


def producer_manifest(base,report):
    records=json.loads((base/'manifest.json').read_text())['files'];names=set()
    report['bindings'].append(binding(base/'manifest.json'))
    for rec in records:
        require(rec['path'] not in names,'duplicate producer manifest member')
        names.add(rec['path']);report['bindings'].append(binding(checked(base,rec)))
    failure=incremental_failure(base,report)
    actual={p.name for p in base.iterdir() if p.is_file() and p.name!='manifest.json'}
    extras=actual-names;partials=[];inventory={r['path']:r for r in records}
    for name in sorted(extras):
        target=name.removeprefix('.partial-')
        require(failure is not None and name.startswith('.partial-') and target in inventory,
            'producer file inventory')
        path=base/name;require(path.stat().st_size<=inventory[target]['bytes'],'partial extracted file cap')
        rec=binding(path);report['bindings'].append(rec);partials.append(rec)
    require(names<=actual,'producer file inventory')
    report['retained_partial_extractions']=partials
    for path in base.iterdir():
        require(not path.is_symlink(),'producer symlink')
        if path.is_dir():require(path.name=='return-shards','unknown producer subdirectory')
    return_shards(base,report,failure)
    report['verified_producer_manifest_files']=len(names)
    return names


def stage_arrays(base,stage,initial,plan):
    sid=stage['stage_id'];group,local=divmod(sid,2)
    profile_payload(stage,plan.profile,'stage')
    name=('intervention','hold')[local];steps=(60,240)[local];controls=controls_for(plan.bit,group,local)
    require(stage['group_id']==group and stage['group']==plan.groups[group] and stage['local_stage_id']==local
        and stage['name']==name and stage['steps']==steps and stage['controls']==controls
        and stage['nominal_controls']==(plan.endpoint,HOLD)[local]
        and stage['age_index']==group and stage['hold_age']==AGES[group],'stage schedule/realized controls')
    scope='not_applicable_new_repair_half_pulse'
    require(stage['historical_comparison_scope']==scope,'historical scope')
    require(stage['historical_comparison'] is None,'hold repair compared with nominal history')
    require(stage['status'] in ('not_started','completed','failed','incomplete'),'stage status')
    if stage['initial'] is None:
        require(stage['completed_steps']==0 and not stage['blocks'] and stage['trajectory'] is None,'missing initial hides updates')
        return None,dict(stage_id=sid,status=stage['status'],completed_updates=0)
    require(initial is not None and stage['initial']['path']==f'stage-{sid:02d}-initial.npy','stage initial filename/input')
    x=batch_state(base,stage['initial']);require(same_bytes(x,initial),'reset/carry initial bytes differ')
    if stage['status']=='not_started':
        require(stage['completed_steps']==stage['attempted_steps']==stage['returned_steps']==0
            and not stage['blocks'] and stage['trajectory'] is None,'unstarted stage contains work')
        return reconstruct(x,[],controls),dict(stage_id=sid,status='not_started',completed_updates=0,blocks=0,update_seconds=0.)
    blocks=[];total=0
    for rec in stage['blocks']:
        arrays=npz(checked(base,rec));total+=len(arrays['states'])
        require(rec['path']==f'stage-{sid:02d}-block-{total:03d}.npz','block filename')
        blocks.append(arrays)
    arrays=reconstruct(x,blocks,controls);n=arrays['updates']
    require(n==stage['completed_steps']<=steps and n<=stage['returned_steps']<=stage['attempted_steps']<=min(steps,n+1),'stage update counts')
    require(stage['trajectory'] is not None and stage['trajectory']['path']==f'stage-{sid:02d}-trajectory.json','trajectory JSON')
    saved=json.loads(checked(base,stage['trajectory']).read_text())
    require(saved==dict(stage_id=sid,initial=stage['initial'],blocks=stage['blocks'],completed_steps=n,
        state_shape=[n+1,2,896],status=stage['status'],controls=controls,
        format='initial plus nonoverlapping post-update blocks'),'trajectory manifest contract')
    if stage['status']=='completed':require(n==steps==stage['attempted_steps']==stage['returned_steps'],'completed stage counts')
    return arrays,dict(stage_id=sid,group_id=group,name=name,status=stage['status'],completed_updates=n,
        attempted_updates=stage['attempted_steps'],returned_updates=stage['returned_steps'],blocks=len(blocks),
        update_seconds=float(arrays['step_seconds'].sum()),initial=stage['initial'],trajectory=stage['trajectory'])


def call_inventory(result,events):
    profile=PROFILE
    slots=result['calls'];expected=expected_calls();require(len(slots)==1224,'full 1224 call slots')
    active=[];stopped=False
    for slot,want in zip(slots,expected,strict=True):
        profile_payload(slot,profile,'call slot')
        require(all(type(slot[k]) is type(v) and slot[k]==v for k,v in want.items()),'fixed call identity')
        require(slot['status'] in ('not_started','started','returned','failed','incomplete'),'call status')
        if slot['status']=='not_started':stopped=True
        else:require(not stopped,'call after unstarted slot');active.append(slot)
    require(len(active)==result['model_calls']<=1224,'admitted calls')
    counts=dict(prefix=0,comparison=0,update=0);counts.update(Counter(x['category'] for x in active))
    require(counts==result['call_counts'],'call category totals')
    started=[e for e in events if e['event']=='model_call_started']
    finished=[e for e in events if e['event']=='model_call_completed']
    require(len(started)==len(active),'start event inventory')
    for event,slot in zip(started,active,strict=True):
        profile_payload(event,profile,'call started event')
        require(event['status']=='started' and all(event[k]==slot[k] for k in expected[slot['call']-1]),'start event')
    require(len({e['call'] for e in finished})==len(finished),'duplicate completed event')
    for event in finished:
        profile_payload(event,profile,'call completed event')
        require(1<=event['call']<=len(active),'completed event outside admitted slots')
        require(all(event.get(k)==v for k,v in slots[event['call']-1].items()),'completed call event differs')
    if result['status']=='completed':
        require(len(active)==len(finished)==1224 and all(s['status']=='returned' for s in active),'completed run call coverage')
    return active,counts


def audit(base,commit,report,requested):
    base=Path(base);plan=Plan(requested);report['profile']=PROFILE
    require(base.name=='run','use actual returned run directory')
    verify_return(base,report,commit);producer_manifest(base,report)
    result=json.loads((base/'result.json').read_text());report['producer_status']=result['status']
    require(result.get('profile')==plan.profile,'result profile')
    require(type(result['requested_bit']) is int and result['requested_bit']==plan.bit,'result bit')
    require(result['status'] in ('completed','failed','incomplete'),'result status')
    design=source_design(base,result,commit,report,plan)
    refs=references(base,report);initial=hold_repair_inputs(base,result,refs,report,plan)
    if refs is not None:
        q=json.loads((base/'qualified-inputs.json').read_text())
        require(q['references']['references']['sha256']==REFERENCE_HASH,'reference identity')
    events=[json.loads(line) for line in (base/'events.jsonl').read_text().splitlines()]
    require(events and all(finite_nonnegative(e['elapsed_seconds']) and type(e['peak_rss_bytes']) is int
        and e['peak_rss_bytes']>=0 for e in events),'event resources')
    require(all(a['elapsed_seconds']<=b['elapsed_seconds'] for a,b in zip(events,events[1:])),'event time order')
    active,counts=call_inventory(result,events)
    final_events=[e for e in events if e['event']=='finished']
    require(len(final_events)==1 and final_events[0] is events[-1]
        and final_events[0]['status']==result['status'] and final_events[0]['model_calls']==result['model_calls'],
        'final event status/count')
    require(not (base/'phase-inputs.json').exists() and not (base/'historical-inputs.json').exists() and not (base/'continuation-inputs.json').exists(),'reference phase/scalar history input on hold repair')
    rows,stages,groups=result['rows'],result['stages'],result['groups']
    require((len(rows),len(stages),len(groups))==(8,8,4),'all row/stage/group slots')
    planned=json.loads((base/'planned-rows.json').read_text())
    require(len(planned)==8,'planned row inventory')
    frozen=json.loads((base/'frozen-rows.json').read_text()) if (base/'frozen-rows.json').exists() else None
    if frozen is not None:require(len(frozen)==8,'frozen row inventory')
    for i,row in enumerate(rows):
        group,start=divmod(i,2)
        require(not any(k in row for k in ('source_reference_index','source_reference_label','source_phase')),'derived hold state mislabeled as current reference phase')
        profile_payload(row,plan.profile,'row')
        profile_payload(planned[i],plan.profile,'planned row')
        fixed=dict(row_id=i,group_id=group,group=plan.groups[group],batch_index=start,message=plan.messages[start],
            requested_bit=plan.bit,origin_reference_index=plan.source_index,origin_reference_label=plan.source_label,origin_reference_phase=start,source_result_row_id=2+4*(group//2)+start,source_stage_id=3+4*(group//2),historical_endpoint_updates=240,source_repaired_bit=1-plan.bit,**group_metadata(group))
        realized=[dict(stage_id=2*group+j,name=('intervention','hold')[j],steps=(60,240)[j],
            controls=controls_for(plan.bit,group,j)) for j in range(2)]
        require(all(type(row[k]) is type(v) and row[k]==v and planned[i][k]==v for k,v in fixed.items()),'row identity')
        scope='not_applicable_new_repair_half_pulse'
        require(row['historical_comparison_scope']==planned[i]['historical_comparison_scope']==scope,'row historical scope')
        require(row['historical_comparison'] is None,'hold-repair row claims historical equivalence')
        require([profile_payload(s,plan.profile,'row realized stage') for s in row['realized_stages']]
            ==[profile_payload(s,plan.profile,'planned row realized stage') for s in planned[i]['realized_stages']]
            ==realized,'row realized schedule')
        require(row==json.loads((base/f'row-{i:02d}.json').read_text()),'row shard differs')
        require(planned[i]['status']==planned[i]['parser_outcome']=='not_started'
            and planned[i]['public_action'] is None and planned[i]['dispatch_count']==0
            and planned[i]['completed_updates']==0 and planned[i]['readout'] is None
            and planned[i]['outcome']=='not_evaluated','planned outcomes present')
        if row['parser_outcome']=='accepted':
            require(typed_action_equal(row['public_action'],plan.bit) and row['dispatch_count']==1,'nominal typed action')
        else:require(row['parser_outcome']=='not_started' and row['public_action'] is None and row['dispatch_count']==0,'parser failure record')
        if frozen is not None:
            f=frozen[i]
            profile_payload(f,plan.profile,'frozen row')
            require(all(f[k]==row[k] for k in fixed)
                and [profile_payload(s,plan.profile,'frozen realized stage') for s in f['realized_stages']]==realized
                and f['historical_comparison_scope']==scope and f['historical_comparison'] is None
                and f['public_action']==action(plan.bit) and f['dispatch_count']==1
                and f['parser_outcome']=='accepted' and f['status']=='not_started'
                and f['completed_updates']==0 and f['readout'] is None and f['outcome']=='not_evaluated','frozen public input')
    for name,expected in [('planned-calls.json',expected_calls()),('planned-stages.json',stages),('planned-groups.json',groups)]:
        saved=json.loads((base/name).read_text());require(len(saved)==len(expected),'planned inventory '+name)
        if name=='planned-calls.json':
            require(all(profile_payload(s,plan.profile,'planned call')==dict(**e,status='not_started')
                for s,e in zip(saved,expected,strict=True)),'planned calls contain outcomes')
        elif name=='planned-stages.json':
            for sid,s in enumerate(saved):
                profile_payload(s,plan.profile,'planned stage')
                require(s['stage_id']==sid and s['controls']==controls_for(plan.bit,sid//2,sid%2)
                    and s['group_id']==sid//2 and s['group']==GROUPS[sid//2] and s['local_stage_id']==sid%2
                    and s['age_index']==sid//2 and s['hold_age']==AGES[sid//2]
                    and s['name']==('intervention','hold')[sid%2] and s['steps']==(60,240)[sid%2]
                    and s['nominal_controls']==(plan.endpoint,HOLD)[sid%2]
                    and s['historical_comparison'] is None and s['historical_comparison_scope']==HISTORY_SCOPE
                    and s['status']=='not_started' and s['completed_steps']==s['attempted_steps']==s['returned_steps']==0
                    and s['initial'] is None and s['trajectory'] is None and s['blocks']==[],'planned stage outcomes')
        else:require(saved==design['groups'] and all(g['status']=='not_started' and g['initial_state_sha256'] is None
            and g['terminal_state_sha256'] is None for g in saved),'planned group outcomes')
    if active:require(initial is not None,'calls without hold-input packet')
    freeze=[e for e in events if e['event']=='inputs_frozen']
    if freeze:
        require(len(freeze)==1 and frozen is not None and initial is not None
            and freeze[0]['initial_sha256']==array_digest(initial) and freeze[0]['source_phase_commit']==SOURCE_PHASE_COMMIT
            and freeze[0]['historical_endpoint_updates']==240 and freeze[0]['hold_ages']==list(AGES),'frozen input binding')
        if active:require(events.index(freeze[0])<next(i for i,e in enumerate(events) if e['event']=='model_call_started'),'late input freeze')
    if active:require(freeze and result['existing_history_cost']==report['source_history_cost'],'calls before freeze or history cost omitted')
    group_starts=[e for e in events if e['event']=='group_started']
    require(result['hold_state_initializations']==len(group_starts)<=4
        and result['between_group_resets']==max(0,len(group_starts)-1),'reset count')
    for i,event in enumerate(group_starts):
        require(event['group_id']==i and event['age_index']==i and event['hold_age']==AGES[i]
            and event['initial_sha256']==array_digest(initial[i]),'literal age reset event')
        group_calls=[e for e in events if e['event']=='model_call_started' and e['group_id']==i]
        if group_calls:require(events.index(event)<events.index(group_calls[0]),'reset after group call')
    arrays={};stage_reports=[];call_reports=[];stopped=False
    for sid,stage in enumerate(stages):
        require(type(stage['stage_id']) is int and stage['stage_id']==sid,'stage index')
        if stopped:require(stage['status']=='not_started','stage after structural failure')
        previous=initial[sid//2] if initial is not None and sid%2==0 else arrays[sid-1]['states'][-1] if sid-1 in arrays else None
        arr,summary=stage_arrays(base,stage,previous,plan)
        calls=[x for x in active if x['stage_id']==sid]
        if calls:require(sid//2<len(group_starts),'calls without group initialization')
        if arr is not None:
            arrays[sid]=arr
            cache,_,_=audit_cache_stage(base,stage,calls,arr['states'][0],len(design['map']['prefix_ids']))
            summary['cache']=cache
            checked_calls=[inspect_call(base,c,arr,cache,len(design['map']['prefix_ids'])) for c in calls]
            call_reports.extend(checked_calls)
            updates=[c for c in calls if c['category']=='update']
            require(stage['attempted_steps']==len(updates)
                and stage['returned_steps']==sum(bool(c.get('forward_return_recorded')) for c in checked_calls if c['category']=='update')
                and stage['completed_steps']==sum(bool(c.get('state_carried')) for c in updates),'admitted/returned/carried accounting')
            require(stage['failed_outputs']==[c['artifact'] for c in updates if c.get('artifact') is not None],'failed output inventory')
            require(stage['historical_comparison'] is None,'hold repair has scalar history comparison')
        else:require(not calls and stage['attempted_steps']==stage['returned_steps']==0,'calls without initial')
        if stage['status']=='completed':require(arr is not None,'completed stage evidence')
        else:stopped=True
        stage_reports.append(summary)
    block_events=[e for e in events if e['event']=='block_saved']
    blocks=[(s['stage_id'],b) for s in stages for b in s['blocks']]
    require(len(block_events)==len(blocks),'block event coverage')
    for event,(sid,b) in zip(block_events,blocks,strict=True):
        require(event['stage_id']==sid and event['artifact']==b
            and event['last_state_index']==int(npz(checked(base,b))['last_state_index']),'block event')
    stage_events=[e for e in events if e['event']=='stage_finished']
    attempted_stages=[s for s in stages if s['status']!='not_started']
    require(len(stage_events)==len(attempted_stages),'stage-finalization events')
    for event,stage in zip(stage_events,attempted_stages,strict=True):
        require(all(event[k]==stage[k] for k in ('stage_id','group_id','status','completed_steps')),'stage event fields')
    row_reports=[]
    for i,row in enumerate(rows):
        group,start=divmod(i,2);sid=group*2+1
        n=sum(stages[j]['completed_steps'] for j in (sid-1,sid))
        require(row['completed_updates']==n and row['status'] in ('not_started','failed','incomplete','completed'),'row progress')
        out=dict(row_id=i,group_id=group,group=plan.groups[group],batch_index=start,requested_bit=plan.bit,
            origin_reference_index=plan.source_index,origin_reference_label=plan.source_label,origin_reference_phase=start,source_result_row_id=2+4*(group//2)+start,source_stage_id=3+4*(group//2),historical_endpoint_updates=240,source_repaired_bit=1-plan.bit,**group_metadata(group),
            historical_comparison_scope=row['historical_comparison_scope'],
            status=row['status'],completed_updates=n,outcome='not_evaluated',qualified_label=None,numerical_diagnostics_passed=None,validated_outcome='not_evaluated')
        if row['readout'] is not None:
            require(sid in arrays and stages[sid]['status']=='completed','readout before full hold')
            read=read_cycle(arrays[sid]['states'][1:,start],refs);neutral_match(row['readout'],read)
            out.update(readout=read,qualified_label=None if read['decoded_bit'] is None else LABELS[read['decoded_bit']],
                outcome='unresolved' if read['decoded_bit'] is None else 'passed' if read['decoded_bit']==plan.bit else 'failed')
            diagnostics=all(stages[j]['cache_comparison']['passed'] for j in (sid-1,sid))
            require(row['numerical_diagnostics_passed']==diagnostics,'row diagnostic validity')
            out['numerical_diagnostics_passed']=diagnostics
            out['validated_outcome']=out['outcome'] if diagnostics else 'diagnostic_invalid'
        require(row['outcome']==out['outcome'],'row outcome differs from neutral gate')
        require(row['historical_comparison'] is None,'hold-repair row has scalar history comparison')
        if row['status']=='completed':
            require(n==300 and 'readout' in out,'completed row evidence')
            require(json.loads((base/f'row-{i:02d}-completed.json').read_text())==row,'completed row checkpoint differs')
        row_reports.append(out)
    group_reports=[]
    for gid,group in enumerate(groups):
        profile_payload(group,plan.profile,'group')
        fixed=dict(group_id=gid,name=plan.groups[gid],age_index=gid,hold_age=AGES[gid],
            intervention='write-opposite-bit',intervention_controls=controls_for(plan.bit,gid,0),
            row_ids=[2*gid,2*gid+1],stage_ids=[2*gid,2*gid+1])
        require(all(group[k]==v for k,v in fixed.items()),'group identity')
        if gid<len(group_starts):require(group['initial_state_sha256']==array_digest(initial[gid]),'group reset digest')
        else:require(group['status']=='not_started' and group['initial_state_sha256'] is None,'unstarted group has reset')
        if 'diagnostics_passed' in group:
            require(group['diagnostics_passed']==all(stages[j]['cache_comparison']['passed'] for j in group['stage_ids']), 'group diagnostic aggregate')
        if group['status']=='completed':
            require(all(rows[i]['status']=='completed' for i in group['row_ids']),'completed group rows')
            require(group['terminal_state_sha256']==array_digest(arrays[2*gid+1]['states'][-1]),'group terminal state')
            marker=json.loads((base/f'group-{gid:02d}.json').read_text())
            require(marker==group,'group completion marker differs')
            artifacts=marker['artifacts'];require([b['path'] for b in artifacts]==sorted(b['path'] for b in artifacts)
                and len({b['path'] for b in artifacts})==len(artifacts),'group artifact order/uniqueness')
            for rec in artifacts:report['bindings'].append(binding(checked(base,rec)))
            expected_names={p.name for p in base.iterdir() if p.is_file() and (
                p.name.startswith((f'stage-{2*gid:02d}-',f'stage-{2*gid+1:02d}-'))
                or p.name in (f'row-{2*gid:02d}-completed.json',f'row-{2*gid+1:02d}-completed.json')
                or (p.name.startswith('call-') and any(c.get('artifact',{}).get('path')==p.name for c in active if c['group_id']==gid)))}
            require({b['path'] for b in artifacts}==expected_names,'closed group artifact inventory')
        group_reports.append(dict(**fixed,status=group['status'],outcomes=[row_reports[i]['outcome'] for i in group['row_ids']],
            initial_state_sha256=group['initial_state_sha256'],terminal_state_sha256=group['terminal_state_sha256']))
    finishes=[e for e in events if e['event']=='group_finished']
    require([e['group_id'] for e in finishes]==[g['group_id'] for g in groups if g['status']=='completed'],'group completion events')
    require(result['historical_scalar_equivalence'] is None and result['historical_comparison_scope']==HISTORY_SCOPE
        and result['reference_initializations']==0 and result['historical_endpoint_updates_per_row']==240,
        'hold repair cannot assert scalar equivalence/reference reset or omit historical exposure')
    if group_starts:
        gid=len(group_starts)-1
        current=initial[gid]
        for sid in (2*gid,2*gid+1):
            if sid in arrays:current=arrays[sid]['states'][-1]
    else:
        current=initial[0] if freeze else None
        # Assignment precedes the initial artifact write; a failed write can
        # retain its digest without a frozen event or reconstructable array.
        if not freeze and frozen is not None and result['terminal_state_sha256'] is not None:
            require(result['status']!='completed' and not active and initial is not None,
                'unbacked initial-state observation on successful work')
            current=initial[0]
            report['initial_assignment_observed_without_saved_artifact']=True
    require(result['terminal_state_sha256']==(array_digest(current) if current is not None else None),'final actual state digest')
    updates=sum(s['completed_steps'] for s in stages)
    require(result['completed_batch_updates']==updates and result['completed_row_updates']==2*updates,'aggregate updates')
    outcomes={k:sum(r['outcome']==k for r in row_reports) for k in ('passed','failed','unresolved','not_evaluated')}
    require(result['outcome_counts']==outcomes,'all eight outcome counts')
    require(finite_nonnegative(result['elapsed_seconds']) and type(result['peak_rss_bytes']) is int
        and result['peak_rss_bytes']>=0,'resource observations')
    raw_bytes=sum(p.stat().st_size for p in base.iterdir() if p.is_file())
    require(raw_bytes<=LIMITS['total_raw_bytes'],'raw output cap')
    if result['status']=='completed':
        require(all(s['status']=='completed' for s in stages) and all(g['status']=='completed' for g in groups)
            and result['resource_limits_passed'] and result['elapsed_seconds']<1800
            and result['peak_rss_bytes']<8*1024**3,'completed execution/resource gates')
    else:require(result.get('error'),'incomplete execution lacks error')
    report.update(requested_bit=plan.bit,rows=row_reports,groups=group_reports,stages=stage_reports,
        call_summaries=call_reports,outcome_counts=outcomes,
        validated_outcome_counts={k:sum(r['validated_outcome']==k for r in row_reports)
            for k in ('passed','failed','unresolved','diagnostic_invalid','not_evaluated')},
        diagnostic_invalid_row_ids=[r['row_id'] for r in row_reports if r['numerical_diagnostics_passed'] is False],
        historical_comparison_scope='not_applicable_new_repair_half_pulse',
        calls=dict(admitted_model_calls=len(active),category_counts=counts,completed_batch_updates=updates,
            completed_row_updates=2*updates,nonprefix_forward_attempts=sum(c.get('forward_attempted',False) for c in active)),
        timings=dict(total_seconds=result['elapsed_seconds'],cold_initialization_seconds=result.get('cold_initialization_seconds'),
            update_seconds=sum(s.get('update_seconds',0) for s in stage_reports),peak_rss_bytes=result['peak_rss_bytes']),
        raw_bytes=raw_bytes,all_rows_preserved=True,retained_producer_error=result.get('error'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--commit')
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--requested-bit',type=int,choices=(0,1),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if not args.preflight_only and (args.run is None or args.commit is None):parser.error('--run and --commit are required for a result audit')
    if args.preflight_only and (args.run is not None or args.commit is not None):parser.error('input preflight is separate from a result audit')
    report=dict(schema_version=1,utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
        auditor_source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        audited_source_commit=args.commit,status='failed',bindings=[],external_bindings=[],auditor_source=binding(__file__),
        limitations=['Saved-data arithmetic, provenance and state-control audit; no transformer replay or independent model normalization proof.',
            'Per-update probabilities and cache tensors are represented by source-bound hashes/instrumentation; only retained arrays are recomputed.',
            'Four fixed opposite-bit handoffs from repaired early/late states and two origin histories per bit; no all-phase reliability, continuous tolerance, ECC or language-interface claim.'])
    try:
        if args.preflight_only:
            reference=DATA/'audits/p6-recovery/20261025T221011Z-recovery/reference-cycles.npz'
            bound=binding(reference);require(bound['sha256']==REFERENCE_HASH,'preflight reference hash');report['external_bindings'].append(bound)
            refs=npz(reference)['states'];initial,packet=load_hold_packet(args.requested_bit,refs,report)
            report.update(mode='input_preflight',requested_bit=args.requested_bit,profile=PROFILE,hold_inputs=packet,
                model_calls=0,model_loaded=False,initial_shape=list(initial.shape),initial_dtype=str(initial.dtype),
                all_age_initial_sha256=array_digest(initial))
        else:audit(args.run,args.commit,report,args.requested_bit)
        report['status']='passed'
    except Exception as error:
        report.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','outcome_counts','error') if k in report}))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
