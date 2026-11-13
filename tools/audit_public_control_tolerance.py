#!/usr/bin/env python3
"""Independent saved-output audit of public controller perturbations.

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
GROUPS = ('nominal', 'alpha-minus', 'alpha-plus', 'log10T-minus', 'log10T-plus')
DELTAS = ((0.,0.), (-1e-5,0.), (1e-5,0.), (0.,-1e-4), (0.,1e-4))
PROFILE_AXIS = 'axis-stencil'
PROFILE_REFINEMENT = 'negative-axis-refinement'
PROFILES = (PROFILE_AXIS, PROFILE_REFINEMENT)
FRACTIONS = (0., .25, .5, .75, 1.)
REFINEMENT_GROUPS = ('fraction-0', 'fraction-025', 'fraction-050', 'fraction-075', 'fraction-100')
REFERENCE_HASH = 'd96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0'
INPUT_HASHES = {'weights':'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe',
 'config':'18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45',
 'vectors':'b630bb7bef06bde87f8e6a7ca5e8672f83d24bc0585bf19048e9e494a4357f41'}
PRECISION = {'norm':{k:'torch.float64' for k in ('input','reduction','weight','output')},
 'rotary':{k:'torch.float64' for k in ('input','inverse_frequency','angles','cos','sin')}}
SOURCES = ['tools/run_public_batch_writes.py','tools/run_public_command_memory.py',
 'tools/transparent_command_contract.py','tools/check_p6_writes.py','tools/check_p6_recovery.py',
 'tools/check_p6_hold.py','tools/validate_qwen_map.py','tools/benchmark_qwen_batch.py',
 'tools/run_public_half_pulse.py','tools/validate_qwen_batch_trajectory.py',
 'tools/run_public_control_tolerance.py','docs/public-control-tolerance.md',
 'docs/receipts/p6-writes.json','docs/receipts/p6-write-path-design.json','pyproject.toml','uv.lock']
LIMITS = dict(model_calls=1830, internal_seconds=1800, external_seconds=2100,
 termination_grace_seconds=30,threads=1,peak_rss_bytes=8*1024**3,external_vm_bytes=16*1024**3,
 working_raw_bytes=152*1024**2,finalization_reserve_bytes=8*1024**2,total_raw_bytes=160*1024**2,
 archive_bytes=224*1024**2,transfer_reservation_bytes=320*1024**2,source_upload_bytes=8*1024**2)

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

def audit_scalar_stage(base,row,s,bit,index,endpoint_steps=60,hold_steps=240):
    require(type(endpoint_steps) is int and endpoint_steps in (60,120),'undeclared endpoint duration')
    require(type(hold_steps) is int and hold_steps in (240,241),'undeclared hold duration')
    name=('endpoint','hold')[index];steps=(endpoint_steps,hold_steps)[index];controls=(ENDPOINTS[bit],HOLD)[index];mid=row['message_id']
    require(s['name']==name and s['steps']==steps and s['controls']==controls,'typed stage schedule')
    require(s['status'] in ('completed','failed','incomplete','not_started'),'stage status')
    if s['status']=='not_started':
        require(s['completed_steps']==0 and not s.get('trajectory') and not s.get('checkpoints'),'unstarted stage contains work');return None,dict(stage=name,status=s['status'])
    z=npz(checked(base,s['trajectory']));a=trajectory(z,controls);n=len(a)-1
    require(s['trajectory']['path']==f'trial-{mid:02d}-{name}-final-{n:03d}.npz' and scalar(z['first_state_index'],int)==0,'final filename/index')
    require(0<=n<=steps and s['completed_steps']==n and n<=s['attempted_steps']<=min(steps,n+1),'completed attempted updates')
    require(np.isfinite(a).all() and (z['token_ids']>=0).all() and (z['token_ids']<151936).all(),'validated trajectory contains invalid state/token')
    failed_outputs=[]
    for rec in s.get('failed_outputs',[]):
        bad=npz(checked(base,rec));require(set(bad)=={'output','probabilities'},'failed output fields')
        require(rec['path']==f'invalid-message-{mid:02d}-{name}-{s["attempted_steps"]:03d}.npz','failed output filename')
        invalid=(bad['output'].shape!=(896,) or bad['output'].dtype!=np.float64 or not np.isfinite(bad['output']).all() or bad['probabilities'].ndim!=1 or bad['probabilities'].dtype!=np.float64 or not np.isfinite(bad['probabilities']).all() or len(bad['probabilities'])<2)
        require(invalid and s['status']!='completed' and s['attempted_steps']==n+1,'failed artifact does not explain aborted update')
        failed_outputs.append(dict(binding=rec,output_shape=list(bad['output'].shape),output_dtype=str(bad['output'].dtype),output_finite=bool(np.isfinite(bad['output']).all()),probabilities_shape=list(bad['probabilities'].shape),probabilities_dtype=str(bad['probabilities'].dtype),probabilities_finite=bool(np.isfinite(bad['probabilities']).all())))
    require(len(failed_outputs)<=1,'multiple failed outputs in one aborted stage')
    if s['status']=='completed':require(n==steps==s['completed_steps'] and np.isfinite(a).all(),'completed stage incomplete/nonfinite')
    blocks=s.get('checkpoints',[]);require(blocks and blocks[-1]==s['trajectory'],'final checkpoint not last')
    stop=0
    for rec in blocks[:-1]:
        b=npz(checked(base,rec));ba=trajectory(b,controls);start=scalar(b['first_state_index'],int);end=start+len(ba)-1
        expected_length=1 if name=='hold' and hold_steps==241 and start==240 else 12
        require(start==stop and end-start==expected_length and end<=n and rec['path']==f'trial-{mid:02d}-{name}-block-{end:03d}.npz','block inventory/order')
        for field in ('states','token_ids','step_seconds'):
            expected=z[field][start:end+1] if field=='states' else z[field][start:end]
            require(b[field].dtype==expected.dtype and b[field].tobytes()==expected.tobytes(),'block disagrees with final '+field)
        stop=end
    if s['status']=='completed':require(stop==steps,'missing full block')
    comp=s.get('cache_comparison');gaps=None
    if comp is not None:
        require(len(comp['artifacts'])==2,'cache artifact count');arrays=[]
        for suffix,rec in zip(('cached','full'),comp['artifacts']):
            require(rec['path']==f'trial-{mid:02d}-{name}-{suffix}.npz','cache filename');c=npz(checked(base,rec));arrays.append(c)
            require(set(c)=={'input','output','probabilities','alpha','log10_temperature'},'cache field inventory')
            require(same_state_bytes(c['input'],a[0]),'cache input differs from initial state')
            for field,shape in [('input',(896,)),('output',(896,)),('probabilities',(151936,))]:require(c[field].shape==shape and c[field].dtype==np.float64 and np.isfinite(c[field]).all(),'cache finite dtype/shape')
            pr=c['probabilities'];require((pr>=0).all() and (pr<=1).all() and abs(float(pr.sum())-1)<=1e-12,'cache probability simplex')
            for k,v in controls.items():require(c[k].dtype==np.float64 and scalar(c[k])==v,'cache controls')
        gap=float(np.max(np.abs(arrays[0]['output']-arrays[1]['output'])))
        pgap=float(np.max(np.abs(arrays[0]['probabilities']-arrays[1]['probabilities'])))
        require(comp['maximum_coordinate_gap']==gap and comp['tolerance']==1e-10 and comp['input_unchanged'] is True and comp['passed']==(gap<=1e-10),'saved cache comparison')
        first=None if n==0 else float(np.max(np.abs(arrays[0]['output']-a[1])))
        if s['status']=='completed':require(comp['passed'] and first<=1e-10,'cache/first-update gate')
        gaps=dict(maximum_coordinate_gap=gap,maximum_probability_gap=pgap,cached_first_update_gap=first)
    elif s['status']=='completed':raise ValueError('completed stage lacks cache comparison')
    return a,dict(stage=name,status=s['status'],attempted_updates=s['attempted_steps'],completed_updates=n,requested_updates=steps,blocks=len(blocks)-1,cache=gaps,failed_outputs=failed_outputs,update_seconds=float(z['step_seconds'].sum()),finite_states=bool(np.isfinite(a).all()))

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

def trajectory_comparison(actual,tokens,historical,historical_tokens):
 valid_matrix(actual,(len(actual),DIM),'current row');valid_matrix(historical,(len(historical),DIM),'historical row')
 require(len(actual)<=len(historical) and tokens.shape==(len(actual)-1,) and tokens.dtype==np.int64 and historical_tokens.shape==(len(historical)-1,) and historical_tokens.dtype==np.int64,'comparison horizons')
 old=historical[:len(actual)];difference=actual-old;coordinates=np.max(np.abs(difference),axis=1);l2=np.linalg.norm(difference,axis=1);mismatch=tokens!=historical_tokens[:len(tokens)]
 return dict(compared_states=len(actual),compared_updates=len(tokens),initial_bytes_equal=same_bytes(actual[0],historical[0]),states_exact=same_bytes(actual,old),maximum_coordinate_gap=float(coordinates.max()),maximum_l2_gap=float(l2.max()),per_state_coordinate_gaps=coordinates.tolist(),per_state_l2_gaps=l2.tolist(),coordinate_exceedance_indices=np.flatnonzero(coordinates>1e-10).tolist(),token_mismatch_update_indices=(np.flatnonzero(mismatch)+1).tolist(),token_mismatch_count=int(mismatch.sum()),coordinate_tolerance=1e-10,coordinate_passed=bool((coordinates<=1e-10).all()),tokens_equal=not bool(mismatch.any()),passed=bool((coordinates<=1e-10).all() and not mismatch.any()))

def diagnostic_comparison(cached,full):
 for arrays in (cached,full):
  valid_matrix(arrays['input'],(2,DIM),'diagnostic input');valid_matrix(arrays['output'],(2,DIM),'diagnostic output');valid_matrix(arrays['probabilities'],(2,VOCAB),'diagnostic probabilities')
  p=arrays['probabilities'];require((p>=0).all() and (p<=1).all() and np.all(np.abs(p.sum(axis=1)-1)<=1e-12),'diagnostic probability simplex')
 require(same_bytes(cached['input'],full['input']),'diagnostic inputs differ')
 coordinate=float(np.abs(cached['output']-full['output']).max());probability=float(np.abs(cached['probabilities']-full['probabilities']).max());tokens=np.array_equal(cached['probabilities'].argmax(axis=1),full['probabilities'].argmax(axis=1))
 return dict(maximum_coordinate_gap=coordinate,maximum_probability_gap=probability,top_tokens_equal=bool(tokens),tolerance=1e-12,passed=coordinate<=1e-12 and probability<=1e-12 and bool(tokens))

def requested_bit(text):
    if type(text) is not str:return None
    return 0 if text==ZERO else 1 if text==ONE else None

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

def delivery(text,readout):
    wanted=requested_bit(text)
    if wanted is None:return 'rejected'
    decoded=readout['decoded_bit']
    return 'unresolved' if decoded is None else 'correct' if decoded==wanted else 'wrong'

class Plan:
 __slots__=('bit','messages','old_ids','endpoint','profile','groups','deltas','fractions')
 def __init__(self,bit,profile=PROFILE_AXIS):
  require(type(bit) is int and bit in (0,1),'explicit requested bit must be integer zero or one')
  require(type(profile) is str and profile in PROFILES,'unknown control profile')
  self.bit=bit
  self.profile=profile
  self.groups=GROUPS if profile==PROFILE_AXIS else REFINEMENT_GROUPS
  self.fractions=None if profile==PROFILE_AXIS else FRACTIONS
  self.deltas=DELTAS if profile==PROFILE_AXIS else tuple(
      (0.,-fraction*1e-4) if bit==0 else (-fraction*1e-5,0.) for fraction in FRACTIONS)
  self.messages=('Write bit 0 to register R0.','Write bit 0 to register R0.') if bit==0 else ('Write bit 1 to register R0.','Write bit 1 to register R0.')
  self.old_ids=(0,1) if bit==0 else (2,3)
  self.endpoint=ENDPOINTS[bit].copy()

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

def historical_binding_inventory(h,plan):
 expected=['results.json','metadata.json']+[f'trial-{trial:02d}-{stage}-final-{steps}.npz' for trial in plan.old_ids for stage,steps in [('endpoint',120),('hold',240)]]
 require(h['receipt']['path']=='p6-writes.json' and [x['path'] for x in h['bindings']]==expected,'historical basename binding inventory')
 require(len(h['controls'])==2 and [[x['path'] for x in row['trajectories']] for row in h['controls']]==[expected[2:4],expected[4:6]],'historical selected trajectory inventory')

def historical(base,refs,commit,report,plan):
 if not (base/'historical-inputs.json').exists():return None
 h=json.loads((base/'historical-inputs.json').read_text());require(type(h.get('requested_bit')) is int and h['requested_bit']==plan.bit and h['messages']==list(plan.messages) and h['historical_trial_ids']==list(plan.old_ids),'historical explicit plan');require(h['source_commit']==OLD_SOURCE and len(h['controls'])==2,'historical identity');historical_binding_inventory(h,plan)
 receipt_bytes=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':docs/receipts/p6-writes.json'])
 require(len(receipt_bytes)==h['receipt']['bytes'] and sha(receipt_bytes)==h['receipt']['sha256'],'frozen historical receipt')
 receipt=json.loads(receipt_bytes);require(receipt['experiment_source_commit']==OLD_SOURCE and receipt['independent_audit_status']=='audit_passed','historical receipt source/audit')
 for item in receipt['bindings']:report['external_bindings'].append(binding(checked(DATA,item)))
 mf=json.loads((OLD_BASE/'final-manifest.json').read_text());seen=set()
 for item in mf['files']:
  require(item['path'] not in seen,'historical inventory duplicate');seen.add(item['path']);report['external_bindings'].append(binding(checked(OLD_BASE,item)))
 report['external_bindings'].append(binding(checked(OLD_BASE,mf['archive'])))
 for item in h['bindings']:report['external_bindings'].append(binding(checked(OLD_BASE/'run',item)))
 meta=json.loads((OLD_BASE/'run/metadata.json').read_text());old=json.loads((OLD_BASE/'run/results.json').read_text());src=json.loads((base/'source.json').read_text())
 require(meta['source_commit']==OLD_SOURCE and old['status']=='passed' and len(old['rows'])==4,'historical completion')
 require(meta['model']==MODEL and meta['revision']==REVISION and meta['threads']==1 and meta['hold']==HOLD and meta['candidate_endpoints']==dict(ad=ENDPOINTS[0],td=ENDPOINTS[1]),'historical map')
 require(meta['prefix_ids']==json.loads((base/'design.json').read_text())['map']['prefix_ids'],'historical prefix')
 for k,hsh in INPUT_HASHES.items():require(meta[k]['sha256']==hsh,'historical model hash')
 for k in ('references_sha256','reference_manifest_sha256','recovery_results_sha256'):require(meta['arguments'][k]==src['arguments'][k],'historical prerequisite hash')
 for name,hsh in meta['source_hashes'].items():
  oldbytes=subprocess.check_output(['git','-C',str(ROOT),'show',OLD_SOURCE+':tools/'+name]);require(sha(oldbytes)==hsh,'historical frozen source '+name)
  if name!='check_p6_writes.py':require(sha(subprocess.check_output(['git','-C',str(ROOT),'show',commit+':tools/'+name]))==hsh,'unchanged mathematical source '+name)
 histories=[];controls=[]
 for i,old_id in enumerate(plan.old_ids):
  row=old['rows'][old_id];require(row['trial_id']==old_id and row['source_index']==i and row['source_label']==('ad','td')[i] and row['phase']==0 and row['status']=='completed' and row['completed_steps']==360,'historical selected trial')
  previous=refs[i,0];arrays=[];summaries=[]
  for j,s in enumerate(row['stages']):
   adapted=dict(s,attempted_steps=s['completed_steps']);arr,summary=audit_scalar_stage(OLD_BASE/'run',dict(message_id=old_id),adapted,plan.bit,j,120)
   require(same_state_bytes(arr[0],previous),'historical stage continuity');previous=arr[-1];arrays.append(arr);summaries.append(summary)
  require(len(arrays)==2,'historical stages');read=read_cycle(arrays[1][1:],refs)
  neutral={k:v for k,v in row['outcome'].items() if k not in ('classification','desired_label')};neutral['qualified_label']=None if read['decoded_bit'] is None else ('ad','td')[read['decoded_bit']];neutral_match(neutral,read)
  require(row['outcome']['classification']==delivery(plan.messages[i],read),'historical outcome')
  saved=h['controls'][i];require(saved['trial_id']==old_id and saved['source_index']==i and saved['model_calls']==366,'historical control record');neutral_match(saved['neutral_readout'],read)
  for item,s in zip(saved['trajectories'],row['stages'],strict=True):require(checked(OLD_BASE/'run',item)==checked(OLD_BASE/'run',s['trajectory']),'historical trajectory record')
  histories.append([dict(states=x,token_ids=npz(checked(OLD_BASE/'run',stage['trajectory']))['token_ids']) for x,stage in zip(arrays,row['stages'],strict=True)]);controls.append(dict(trial_id=old_id,stages=summaries,readout=read,model_calls=366))
 report['historical_controls']=controls;report['historical_manifest_files']=len(seen)
 return histories

def compare_stage(saved,arrays,histories,sid,plan):
 require(len(saved['rows'])==2,'two historical comparison rows');summaries=[]
 for i,item in enumerate(saved['rows']):
  old=histories[i][sid];q=trajectory_comparison(arrays['states'][:,i],arrays['token_ids'][:,i],old['states'],old['token_ids'])
  expected=dict(row_id=i,historical_trial_id=plan.old_ids[i],compared_states=q['compared_states'],compared_updates=q['compared_updates'],maximum_coordinate_gap=q['maximum_coordinate_gap'],coordinate_tolerance=1e-10,token_mismatch_update_indices=q['token_mismatch_update_indices'],token_mismatch_count=q['token_mismatch_count'],state_bytes_equal=q['states_exact'],token_bytes_equal=q['tokens_equal'],passed=q['passed'])
  require(item==expected,'saved historical trajectory comparison differs');summaries.append(dict(row_id=i,historical_trial_id=plan.old_ids[i],**q))
 require(saved['passed']==all(x['passed'] for x in summaries),'historical aggregate');return summaries

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


def controls_for(bit, group, local_stage, profile=PROFILE_AXIS):
    plan=Plan(bit,profile)
    require(type(group) is int and 0 <= group < 5 and local_stage in (0,1), 'control index')
    if local_stage == 1:
        return HOLD.copy()
    return {key: ENDPOINTS[bit][key]+delta for key,delta in zip(
        ('alpha','log10_temperature'), plan.deltas[group], strict=True)}


def profile_payload(record,profile,label):
    """Missing tags have exactly one legacy meaning; never infer refinement."""
    require(record.get('profile',PROFILE_AXIS)==profile,label+' profile')
    return {key:value for key,value in record.items() if key!='profile'}


def expected_calls(profile=PROFILE_AXIS):
    names=Plan(0,profile).groups
    out=[]
    for group in range(5):
        for local,(stage,n) in enumerate((('endpoint',120),('hold',240))):
            for category,step,cached in [('prefix',0,None),('comparison',0,True),('comparison',0,False)]+[
                    ('update',i,True) for i in range(1,n+1)]:
                out.append(dict(call=len(out)+1,group_id=group,group=names[group],stage_id=2*group+local,
                    stage=stage,category=category,step=step,cached=cached))
    return out


def source_design(base,result,commit,report,plan):
    src=json.loads((base/'source.json').read_text())
    require(src==result['source'] and src['source_commit']==commit,'source identity')
    require(src['arguments']['requested_bit']==str(plan.bit),'source argument bit')
    require(src['arguments'].get('profile',PROFILE_AXIS)==plan.profile,'source argument profile')
    require(result.get('profile',PROFILE_AXIS)==plan.profile,'result profile')
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
    require(design.get('profile',PROFILE_AXIS)==plan.profile,'design profile')
    require(type(design['requested_bit']) is int and design['requested_bit']==plan.bit,'design bit')
    require(design['public_messages']==list(plan.messages),'public messages')
    require(design['limits']==LIMITS,'frozen resource limits')
    require(design['schedule']==dict(groups=5,batch_size=2,rows=10,stages=10,endpoint_steps=120,
        hold_steps=240,batch_updates=1800,row_updates=3600,prefix_calls=10,comparison_calls=20,
        total_calls=1830,reference_initializations=5,between_group_resets=4),'schedule counts')
    require(design['map']==dict(model=MODEL,revision=REVISION,prefix_ids=prefix,hold=HOLD,
        endpoints=dict(ad=ENDPOINTS[0],td=ENDPOINTS[1])),'map identity')
    require(design['diagnostic_tolerance']==1e-12 and design['nominal_trajectory_tolerance']==1e-10,'tolerances')
    envelope=design['artifact_envelope']
    require(envelope['estimated_uncompressed_envelope_bytes']==5*28627392
        and envelope['failed_return_reserved_bytes']==2473984
        and 5*28627392+2473984+16384<LIMITS['working_raw_bytes'],'raw envelope')
    require(len(design['realized_stages'])==10,'realized stage count')
    for sid,stage in enumerate(design['realized_stages']):
        group,local=divmod(sid,2)
        require(profile_payload(stage,plan.profile,'realized stage design')==dict(stage_id=sid,group_id=group,name=('endpoint','hold')[local],steps=(120,240)[local],
            controls=controls_for(plan.bit,group,local,plan.profile),nominal_controls=(plan.endpoint,HOLD)[local]),'realized stage design')
    require(len(design['groups'])==5,'design group count')
    for gid,group in enumerate(design['groups']):
        profile_payload(group,plan.profile,'design group')
        require(group['group_id']==gid and group['name']==plan.groups[gid]
            and group['endpoint_delta']==dict(zip(('alpha','log10_temperature'),plan.deltas[gid],strict=True)),
            'design group identity')
        if plan.fractions is not None:
            require(type(group.get('fraction')) is float and group['fraction']==plan.fractions[gid],
                'design group fraction')
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
        require(child.name in {f'group-{i:02d}' for i in range(5)} and child.is_dir(),
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
    name=('endpoint','hold')[local];steps=(120,240)[local];controls=controls_for(plan.bit,group,local,plan.profile)
    require(stage['group_id']==group and stage['group']==plan.groups[group] and stage['local_stage_id']==local
        and stage['name']==name and stage['steps']==steps and stage['controls']==controls
        and stage['nominal_controls']==(plan.endpoint,HOLD)[local],'stage schedule/realized controls')
    scope='nominal_only' if group==0 else 'not_applicable_perturbed_control'
    require(stage['historical_comparison_scope']==scope,'historical scope')
    if group:require(stage['historical_comparison'] is None,'perturbed group compared with nominal history')
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


def call_inventory(result,events,profile=PROFILE_AXIS):
    slots=result['calls'];expected=expected_calls(profile);require(len(slots)==1830,'full 1830 call slots')
    active=[];stopped=False
    for slot,want in zip(slots,expected,strict=True):
        profile_payload(slot,profile,'call slot')
        require(all(type(slot[k]) is type(v) and slot[k]==v for k,v in want.items()),'fixed call identity')
        require(slot['status'] in ('not_started','started','returned','failed','incomplete'),'call status')
        if slot['status']=='not_started':stopped=True
        else:require(not stopped,'call after unstarted slot');active.append(slot)
    require(len(active)==result['model_calls']<=1830,'admitted calls')
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
        require(len(active)==len(finished)==1830 and all(s['status']=='returned' for s in active),'completed run call coverage')
    return active,counts


def audit(base,commit,report,requested,profile=PROFILE_AXIS):
    base=Path(base);plan=Plan(requested,profile);report['profile']=profile
    require(base.name=='run','use actual returned run directory')
    verify_return(base,report,commit);producer_manifest(base,report)
    result=json.loads((base/'result.json').read_text());report['producer_status']=result['status']
    require(result.get('profile',PROFILE_AXIS)==plan.profile,'result profile')
    require(type(result['requested_bit']) is int and result['requested_bit']==plan.bit,'result bit')
    require(result['status'] in ('completed','failed','incomplete'),'result status')
    design=source_design(base,result,commit,report,plan)
    refs=references(base,report);histories=historical(base,refs,commit,report,plan)
    if refs is not None:
        q=json.loads((base/'qualified-inputs.json').read_text())
        require(q['references']['references']['sha256']==REFERENCE_HASH,'reference identity')
    events=[json.loads(line) for line in (base/'events.jsonl').read_text().splitlines()]
    require(events and all(finite_nonnegative(e['elapsed_seconds']) and type(e['peak_rss_bytes']) is int
        and e['peak_rss_bytes']>=0 for e in events),'event resources')
    require(all(a['elapsed_seconds']<=b['elapsed_seconds'] for a,b in zip(events,events[1:])),'event time order')
    active,counts=call_inventory(result,events,plan.profile)
    final_events=[e for e in events if e['event']=='finished']
    require(len(final_events)==1 and final_events[0] is events[-1]
        and final_events[0]['status']==result['status'] and final_events[0]['model_calls']==result['model_calls'],
        'final event status/count')
    rows,stages,groups=result['rows'],result['stages'],result['groups']
    require((len(rows),len(stages),len(groups))==(10,10,5),'all row/stage/group slots')
    planned=json.loads((base/'planned-rows.json').read_text())
    require(len(planned)==10,'planned row inventory')
    frozen=json.loads((base/'frozen-rows.json').read_text()) if (base/'frozen-rows.json').exists() else None
    if frozen is not None:require(len(frozen)==10,'frozen row inventory')
    for i,row in enumerate(rows):
        group,start=divmod(i,2)
        profile_payload(row,plan.profile,'row')
        profile_payload(planned[i],plan.profile,'planned row')
        fixed=dict(row_id=i,group_id=group,group=plan.groups[group],batch_index=start,message=plan.messages[start],
            requested_bit=plan.bit,source_reference_index=start,source_reference_label=LABELS[start],source_phase=0)
        realized=[dict(stage_id=2*group+j,name=('endpoint','hold')[j],steps=(120,240)[j],
            controls=controls_for(plan.bit,group,j,plan.profile)) for j in range(2)]
        require(all(type(row[k]) is type(v) and row[k]==v and planned[i][k]==v for k,v in fixed.items()),'row identity')
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
                and f['public_action']==action(plan.bit) and f['dispatch_count']==1
                and f['parser_outcome']=='accepted' and f['status']=='not_started'
                and f['completed_updates']==0 and f['readout'] is None and f['outcome']=='not_evaluated','frozen public input')
    for name,expected in [('planned-calls.json',expected_calls(plan.profile)),('planned-stages.json',stages),('planned-groups.json',groups)]:
        saved=json.loads((base/name).read_text());require(len(saved)==len(expected),'planned inventory '+name)
        if name=='planned-calls.json':
            require(all(profile_payload(s,plan.profile,'planned call')==dict(**e,status='not_started')
                for s,e in zip(saved,expected,strict=True)),'planned calls contain outcomes')
        elif name=='planned-stages.json':
            for sid,s in enumerate(saved):
                profile_payload(s,plan.profile,'planned stage')
                require(s['stage_id']==sid and s['controls']==controls_for(plan.bit,sid//2,sid%2,plan.profile)
                    and s['status']=='not_started' and s['completed_steps']==s['attempted_steps']==s['returned_steps']==0
                    and s['initial'] is None and s['trajectory'] is None and s['blocks']==[],'planned stage outcomes')
        else:require(saved==design['groups'] and all(g['status']=='not_started' and g['initial_state_sha256'] is None
            and g['terminal_state_sha256'] is None for g in saved),'planned group outcomes')
    initial=refs[:,0] if refs is not None else None
    freeze=[e for e in events if e['event']=='inputs_frozen']
    if freeze:
        require(len(freeze)==1 and frozen is not None and histories is not None and initial is not None
            and freeze[0]['initial_sha256']==array_digest(initial) and freeze[0]['historical_source']==OLD_SOURCE,'frozen input binding')
        if active:require(events.index(freeze[0])<next(i for i,e in enumerate(events) if e['event']=='model_call_started'),'late input freeze')
    if active:require(freeze,'calls before freeze')
    group_starts=[e for e in events if e['event']=='group_started']
    require(result['reference_initializations']==len(group_starts)<=5
        and result['between_group_resets']==max(0,len(group_starts)-1),'reset count')
    for i,event in enumerate(group_starts):
        require(event['group_id']==i and event['initial_sha256']==array_digest(initial),'literal reset event')
        group_calls=[e for e in events if e['event']=='model_call_started' and e['group_id']==i]
        if group_calls:require(events.index(event)<events.index(group_calls[0]),'reset after group call')
    arrays={};stage_reports=[];call_reports=[];comparisons={};stopped=False
    for sid,stage in enumerate(stages):
        require(type(stage['stage_id']) is int and stage['stage_id']==sid,'stage index')
        if stopped:require(stage['status']=='not_started','stage after structural failure')
        previous=initial if sid%2==0 else arrays[sid-1]['states'][-1] if sid-1 in arrays else None
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
            if stage['historical_comparison'] is not None:
                require(sid<2 and histories is not None,'nominal-only comparison')
                comparisons[sid]=compare_stage(stage['historical_comparison'],arr,histories,sid,plan)
                summary['historical_comparison']=comparisons[sid]
        else:require(not calls and stage['attempted_steps']==stage['returned_steps']==0,'calls without initial')
        if stage['status']=='completed':require(arr is not None and (sid>=2 or sid in comparisons),'completed stage evidence')
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
        if row['historical_comparison'] is not None:
            require(group==0 and len(comparisons)==2 and 'readout' in out,'nominal row comparison prerequisites')
            q=[comparisons[j][start] for j in (0,1)];oldlabel=LABELS[report['historical_controls'][start]['readout']['decoded_bit']]
            expected=dict(compared_states=361,compared_updates=360,maximum_coordinate_gap=max(x['maximum_coordinate_gap'] for x in q),
                token_mismatch_count=sum(x['token_mismatch_count'] for x in q),state_bytes_equal=all(x['states_exact'] for x in q),
                token_bytes_equal=all(x['tokens_equal'] for x in q),final_labels_equal=out['qualified_label']==oldlabel,
                historical_qualified_label=oldlabel,passed=all(x['passed'] for x in q) and out['qualified_label']==oldlabel)
            require(row['historical_comparison']==expected,'nominal row comparison aggregate');out['historical_comparison']=expected
        if row['status']=='completed':
            require(n==360 and 'readout' in out and (group!=0 or row['historical_comparison'] is not None),'completed row evidence')
            require(json.loads((base/f'row-{i:02d}-completed.json').read_text())==row,'completed row checkpoint differs')
        row_reports.append(out)
    group_reports=[]
    for gid,group in enumerate(groups):
        profile_payload(group,plan.profile,'group')
        fixed=dict(group_id=gid,name=plan.groups[gid],endpoint_delta=dict(zip(('alpha','log10_temperature'),plan.deltas[gid],strict=True)),
            row_ids=[2*gid,2*gid+1],stage_ids=[2*gid,2*gid+1])
        if plan.fractions is not None:
            fixed['fraction']=plan.fractions[gid]
            require(type(group.get('fraction')) is float,'group fraction type')
        require(all(group[k]==v for k,v in fixed.items()),'group identity')
        if gid<len(group_starts):require(group['initial_state_sha256']==array_digest(initial),'group reset digest')
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
    nominal=groups[0]
    nominal_equivalence=(nominal['diagnostics_passed'] and all(rows[i]['historical_comparison']['passed'] for i in (0,1))) if all(rows[i]['status']=='completed' for i in (0,1)) else None
    require(result['nominal_equivalence_passed']==nominal_equivalence,'nominal equivalence aggregate')
    if group_starts:
        current=initial
        gid=len(group_starts)-1
        for sid in (2*gid,2*gid+1):
            if sid in arrays:current=arrays[sid]['states'][-1]
    else:
        current=initial if freeze else None
        # Assignment precedes the initial artifact write; a failed write can
        # retain its digest without a frozen event or reconstructable array.
        if not freeze and frozen is not None and result['terminal_state_sha256'] is not None:
            require(result['status']!='completed' and not active and initial is not None,
                'unbacked initial-state observation on successful work')
            current=initial
            report['initial_assignment_observed_without_saved_artifact']=True
    require(result['terminal_state_sha256']==(array_digest(current) if current is not None else None),'final actual state digest')
    updates=sum(s['completed_steps'] for s in stages)
    require(result['completed_batch_updates']==updates and result['completed_row_updates']==2*updates,'aggregate updates')
    outcomes={k:sum(r['outcome']==k for r in row_reports) for k in ('passed','failed','unresolved','not_evaluated')}
    require(result['outcome_counts']==outcomes,'all ten outcome counts')
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
        nominal_equivalence_passed=nominal_equivalence,
        calls=dict(admitted_model_calls=len(active),category_counts=counts,completed_batch_updates=updates,
            completed_row_updates=2*updates,nonprefix_forward_attempts=sum(c.get('forward_attempted',False) for c in active)),
        timings=dict(total_seconds=result['elapsed_seconds'],cold_initialization_seconds=result.get('cold_initialization_seconds'),
            update_seconds=sum(s.get('update_seconds',0) for s in stage_reports),peak_rss_bytes=result['peak_rss_bytes']),
        raw_bytes=raw_bytes,all_rows_preserved=True,retained_producer_error=result.get('error'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--requested-bit',type=int,choices=(0,1),required=True)
    parser.add_argument('--profile',choices=PROFILES,default=PROFILE_AXIS)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    report=dict(schema_version=1,utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
        auditor_source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        audited_source_commit=args.commit,status='failed',bindings=[],external_bindings=[],auditor_source=binding(__file__),
        limitations=['Saved-data arithmetic, provenance and state-control audit; no transformer replay or independent model normalization proof.',
            'Per-update probabilities and cache tensors are represented by source-bound hashes/instrumentation; only retained arrays are recomputed.',
            'Five discrete endpoint settings and two phase-zero starts per public bit; not a continuous tolerance region, language interface or reliability estimate.'])
    try:
        audit(args.run,args.commit,report,args.requested_bit,args.profile);report['status']='passed'
    except Exception as error:
        report.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','outcome_counts','error') if k in report}))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
