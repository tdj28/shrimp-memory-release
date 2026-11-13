#!/usr/bin/env python3
"""Independent saved-array audit of two earlier nominal WRITE0 handoffs.

This checker never imports producer classification code or evaluates a model.
Literal saved states preserve original trajectory indices and phase. This
finite panel does not establish first acquisition, monotonicity, a threshold,
general reliability, branch loss, text reachability or causal hysteresis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tarfile

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np

PROFILE = 'public-nominal-handoffs'
DIM, PERIOD, STEPS, MAX_CALLS, VOCAB = 896, 6, 240, 490, 151936
HOLD = dict(alpha=0.07293697809562831, log10_temperature=-0.15395151820007266)
ENDPOINT = dict(alpha=0.07293697809562831, log10_temperature=-0.1535)
GROUPS = ('source-and-replay', 'early-handoffs')
ROW_NAMES = ('source-h1','replay-x60','early-x12','early-x36')
SOURCE_INDICES = np.array([[0,60],[12,36]],dtype=np.int64)
PINS = {
    'trajectory': (833183, 'c8e4d37d64765734d3b23b3fb3b38e22337d8af1a003cc7bc235cccc3ea51f9a'),
    'fixed_inputs': (96827, '0ad3846d11ff65d7616b5fc6d125f6b30c16ea56ed490da3bfcd27f510586b44'),
    'historical_metadata': (7518, 'efa7bcc502a6ccbd27789d1deee714d58836e870e14a3f0544f9473e2536807a'),
    'historical_plan': (3655, '2762d850ffedaca20c29110f494c3209100c6287cbdfd22fd839a60f0e92c409'),
    'historical_audit': (77854, '257f03396a80c139cab31049fd759d12f610a0566932ccb4e0ebd33d8617fefe'),
    'historical_manifest': (54152, '4536fb39b68e9ae9e1c9e3e251b36b481caefa8889fcf32f297b3097aec79be3'),
}
MODEL_PINS = dict(
    weights='fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe',
    config='18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45',
    vectors='b630bb7bef06bde87f8e6a7ca5e8672f83d24bc0585bf19048e9e494a4357f41',
    tokenizer='c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    tokenizer_config='5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583')
PREFIX = [151644,8948,198,2610,525,458,15235,7404,448,2441,15235,13,1446,614,
          4583,11290,13,31733,1910,311,22729,8820,498,1366,13,151645,198,151644,
          872,198,9707,0,151645,198,151644,77091,198]
PRECISION = dict(norm={k:'torch.float64' for k in ('input','reduction','weight','output')},
                 rotary={k:'torch.float64' for k in ('input','inverse_frequency','angles','cos','sin')})
EPS = np.finfo(np.float64).eps


class AuditError(ValueError):
    pass


def require(value, message):
    if not bool(value):
        raise AuditError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def array_hash(value):
    return digest(str(value.dtype).encode()+b'\0'+
                  json.dumps(list(value.shape),separators=(',',':')).encode()+b'\0'+value.tobytes(order='C'))


def raw_hash(value):
    return digest(value.tobytes(order='C'))


def same_bytes(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order='C') == b.tobytes(order='C')


def finite(value, shape, label):
    require(isinstance(value,np.ndarray) and value.dtype == np.float64 and value.shape == shape
            and np.isfinite(value).all(), label+' finite float64 shape')


def finite_norm(value, axis=None):
    with np.errstate(over='ignore', invalid='ignore'):
        result=np.linalg.norm(value,axis=axis)
    require(np.isfinite(result).all(), 'nonfinite derived norm')
    return result


def strict_json(text):
    def pairs(items):
        output={}
        for key,value in items:
            require(key not in output,'duplicate JSON key: '+key)
            output[key]=value
        return output
    def bad(value):
        raise AuditError('nonstandard JSON numeric value: '+value)
    return json.loads(text,object_pairs_hook=pairs,parse_constant=bad)


def extract_handoffs(references, states):
    """Extract the original literal state bytes; do not rotate or normalize."""
    finite(references,(2,PERIOD,DIM),'reference cycles')
    finite(states,(121,DIM),'historical trajectory')
    h1=references[1,0].copy()
    require(same_bytes(states[0],h1),'historical initial phase is not literal H1 phase0')
    return dict(references=references.copy(),h1=h1,x12=states[12].copy(),x36=states[36].copy(),
        x60=states[60].copy(),initial_states=states[SOURCE_INDICES].copy(),
        historical_state_indices=SOURCE_INDICES.copy())


def readout(tail, references):
    """Last13 retained; six comparisons [-6:] versus [-12:-6], not seven."""
    finite(tail,(13,DIM),'readout tail')
    finite(references,(2,6,DIM),'readout references')
    cycle=tail[-6:]
    residual=finite_norm(cycle-tail[-12:-6],axis=1)
    divisors=[]
    for shift in (1,2,3):
        gaps=finite_norm(cycle-np.concatenate((cycle[shift:],cycle[:shift])),axis=1)
        divisors.append(dict(divisor=shift,phase_l2_gaps=gaps.tolist(),maximum_l2_gap=float(gaps.max()),excluded=bool(gaps.max()>1e-4)))
    distances=[]
    for bit,reference in enumerate(references):
        values=[float(finite_norm(cycle-np.concatenate((reference[k:],reference[:k])),axis=1).max()) for k in range(6)]
        distances.append(dict(bit=bit,distance=min(values),best_shift=int(np.argmin(values)),by_shift=values,
                              tied_best_shifts=[i for i,x in enumerate(values) if x==min(values)]))
    matches=[x['bit'] for x in distances if x['distance']<=1e-6]
    gate=bool(residual.max()<=1e-8 and all(x['excluded'] for x in divisors))
    return dict(lag_six_phase_l2_residuals=residual.tolist(),maximum_l2_residual=float(residual.max()),
                proper_divisors=divisors,cycle_gate_passed=gate,reference_distances=distances,
                reference_matches=matches,decoded_bit=matches[0] if gate and len(matches)==1 else None)


def panel_outcome(bits):
    require(len(bits)==4 and all(x is None or type(x) is int and x in (0,1) for x in bits),'four neutral row classifications')
    if not (bits[0]==1 and bits[1]==0):return 'control_unqualified'
    if None in bits[2:]:return 'early_handoff_unresolved'
    if bits[2:]==[0,0]:return 'both_early_handoffs_h0'
    if bits[2:]==[1,1]:return 'both_early_handoffs_h1'
    return 'mixed_early_handoffs'


def cache_hash(values):
    return digest(json.dumps({k:array_hash(v) for k,v in values.items()},sort_keys=True,separators=(',',':')).encode())


def validate_cache(values, batch, head_dim=64):
    keys={f'layer_{i:02d}_{kind}' for i in range(24) for kind in ('key','value')}
    require(set(values)==keys|{'length'},'cache field inventory')
    require(values['length'].shape==() and values['length'].dtype==np.int64 and int(values['length'])==len(PREFIX),'cache length')
    for key in keys:
        finite(values[key],(batch,2,len(PREFIX),head_dim),'cache '+key)
    return cache_hash(values)


def expected_calls():
    output=[]
    for group in range(2):
        planned=[('prefix',0,None,1,'prefix'),('comparison',0,True,2,'compare-cached'),
                 ('comparison',0,False,2,'compare-full'),('comparison',0,False,1,'compare-scalar-0'),
                 ('comparison',0,False,1,'compare-scalar-1')]
        planned += [('update',i,True,2,f'update-{i:03d}') for i in range(1,241)]
        for category,step,cached,width,label in planned:
            output.append(dict(call=len(output)+1,group_id=group,category=category,step=step,cached=cached,
                               batch_size=width,label=f'g{group:02d}-{label}',controls=HOLD,
                               controls_hex={k:v.hex() for k,v in HOLD.items()}))
    return output


class Audit:
    def __init__(self,directory,*,source_root=None,input_root=None,synthetic=False):
        self.root=Path(directory).resolve()
        self.source_root=None if source_root is None else Path(source_root).resolve()
        self.input_root=None if input_root is None else Path(input_root).resolve()
        self.synthetic=synthetic
        self.assertions=0
        self.bindings={}
        self.partial=[]
        self.rows=[]
        self.group_reports=[]
        self.instrument_failures=[]

    def check(self,ok,label):
        self.assertions+=1
        require(ok,label)

    def same(self,a,b,label):
        self.check(same_bytes(a,b),label)

    def close(self,a,b,label,*,factor=128,scale=None):
        a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
        self.check(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),label+' finite shape')
        size=max(float(finite_norm(a)),float(finite_norm(b)),1e-300) if scale is None else max(float(scale),1e-300)
        self.check(float(finite_norm(a-b))<=factor*EPS*size,label+' arithmetic replay')

    def metrics(self,expected,saved,label):
        if isinstance(expected,dict):
            self.check(isinstance(saved,dict) and set(expected)<=set(saved),label+' fields')
            for key,value in expected.items():self.metrics(value,saved[key],label+'.'+key)
        elif isinstance(expected,list):
            self.check(isinstance(saved,list) and len(expected)==len(saved),label+' length')
            for i,(a,b) in enumerate(zip(expected,saved)):self.metrics(a,b,label+'.'+str(i))
        elif type(expected) is float:
            self.check(type(saved) in (int,float) and math.isfinite(saved),label+' numeric type')
            self.close(expected,saved,label)
        else:self.check(type(expected) is type(saved) and expected==saved,label+' exact identity')

    def file(self,name,root=None):
        base=self.root if root is None else Path(root).resolve()
        self.check(isinstance(name,str) and not Path(name).is_absolute() and '..' not in Path(name).parts,'safe relative artifact path')
        normalized=os.path.normpath(name)
        self.check(normalized not in ('.','..'),'nonempty artifact path')
        path=base/normalized
        self.check(path.is_file() and path.resolve().is_relative_to(base),'missing or escaped artifact '+name)
        cursor=path
        while cursor!=base:
            self.check(not cursor.is_symlink(),'symlink artifact path');cursor=cursor.parent
        return path

    def record(self,path,expected=None):
        path=Path(path)
        self.check(path.is_file() and not path.is_symlink(),'regular evidence file')
        h=hashlib.sha256();size=0
        with path.open('rb') as stream:
            while block:=stream.read(1024*1024):h.update(block);size+=len(block)
        observed=dict(path=str(path.resolve()),bytes=size,sha256=h.hexdigest())
        if expected is not None:
            self.check(type(expected['bytes']) is int and size==expected['bytes'] and observed['sha256']==expected['sha256'],'file hash/size '+str(path))
        prior=self.bindings.get(observed['path'])
        self.check(prior is None or prior==observed,'conflicting repeated provenance '+str(path))
        self.bindings[observed['path']]=observed
        return observed

    def bound(self,row,root=None):
        path=self.file(row['path'],root)
        self.record(path,row)
        return path

    def json(self,name,root=None):
        return strict_json(self.file(name,root).read_text())

    def arrays(self,name):
        with np.load(self.file(name),allow_pickle=False) as z:
            self.check(len(z.files)==len(set(z.files)),'no duplicate NPZ keys')
            return {k:z[k] for k in z.files}

    def index(self,rows,label):
        result={}
        for row in rows:
            name=os.path.normpath(row['path']);pair=(row['bytes'],row['sha256'])
            self.check(name not in result or result[name]==pair,label+' conflicting record')
            result[name]=pair
        return result

    def controls(self,values,label):
        for key,value in HOLD.items():
            a=values[key]
            self.check(a.shape==() and a.dtype==np.float64 and float(a)==value,label+' exact H '+key)

    def inventory(self):
        self.result=self.json('result.json');r=self.result
        self.check(r['profile']==PROFILE and r['status'] in ('completed','preflight_passed','instrument_failed','execution_failed','incomplete'),'terminal profile/status')
        manifest=self.json('manifest.json');self.check(manifest['profile']==PROFILE,'manifest profile')
        index=self.index(manifest['files'],'producer manifest')
        for row in manifest['files']:self.bound(row)
        self.record(self.file('manifest.json'))
        actual={p.name for p in self.root.iterdir() if p.is_file()}
        self.check(actual==set(index)|{'manifest.json'},'manifest covers every run file exactly')
        self.check(not any(p.is_symlink() or p.is_dir() and p.name!='return-shards' or p.name.endswith('.pending') for p in self.root.iterdir()),'no symlink, unfinished or unexpected directory')
        self.file_groups={};self.summaries=[]
        self.check(len(r['groups'])==2,'both group markers retained')
        for gid,name in enumerate(GROUPS):
            marker=self.json(f'group-{gid:02d}.json');summary=self.json(f'group-{gid:02d}-summary.json')
            self.check(marker==r['groups'][gid] and marker['closed'] is True and marker['profile']==PROFILE and marker['group_id']==gid and marker['name']==name,'immutable group marker identity')
            self.check(summary['profile']==PROFILE and summary['group_id']==gid and summary['name']==name and summary['status']==marker['status'] and summary['model_calls']==marker['model_calls'],'group summary identity')
            gi=self.index(marker['artifacts'],'immutable group')
            self.check(f'group-{gid:02d}-summary.json' in gi,'immutable group summary present')
            for row in marker['artifacts']:
                path=self.bound(row);key=path.name
                self.check(index.get(key)==(row['bytes'],row['sha256']),'group member matches final producer inventory')
                self.check(key not in self.file_groups or self.file_groups[key]==gid,'immutable groups do not overlap')
                self.check(key not in ('result.json','manifest.json','events.jsonl') and not re.fullmatch('group-[0-9]{2}.json',key),'immutable group excludes mutable/global marker')
                self.file_groups[key]=gid
            self.check(type(marker['model_calls']) is int and 0<=marker['model_calls']<=245,'per-group call cap')
            self.summaries.append(summary)
        calls=r['model_calls'];self.check(type(calls) is int and 0<=calls<=490,'total invocation cap')
        self.check(sum(g['model_calls'] for g in r['groups'])==calls,'group attempt sum')
        self.design=self.json('design.json');d=self.design
        self.check(d==r['design'] and d['profile']==PROFILE,'fixed design summary')
        expected=dict(group_names=list(GROUPS),row_names=list(ROW_NAMES),controls=HOLD,
            controls_hex={k:v.hex() for k,v in HOLD.items()},source_indices=[[0,60],[12,36]],literal_phase_carry=True,
            schedule=dict(groups=2,calls_per_group=245,prefix_calls=2,comparison_calls=8,batch_updates=480,lane_updates=960,total_calls=490,steps_per_group=240,batch_size=2),
            readout=dict(retained_state_indices=[228,240],closure_pairs=[[i,i-6] for i in range(235,241)],closure_tolerance=1e-8,proper_divisors=[1,2,3],divisor_tolerance=1e-4,reference_match_tolerance=1e-6,all_cyclic_shifts=list(range(6))),comparison_tolerance=1e-10,
            limits=dict(model_calls=490,internal_seconds=2400,external_seconds=2460,termination_grace_seconds=30,outer_seconds=2640,return_seconds=4800,threads=1,peak_rss_bytes=8*2**30,external_vm_bytes=16*2**30,raw_bytes=2304*2**20,working_raw_bytes=2272*2**20,finalization_reserve_bytes=32*2**20,package_bytes=2560*2**20,archive_bytes=3072*2**20,upload_bytes=8*2**20,transfer_reservation_bytes=8*2**30))
        for key,value in expected.items():self.check(d[key]==value,'frozen design '+key)
        self.actual_bytes=sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())
        self.check(type(r['raw_bytes_before_result_manifest']) is int and 0<=r['raw_bytes_before_result_manifest']<=self.actual_bytes,'raw size accounting')
        self.check(type(r['elapsed_seconds']) in (int,float) and math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds']>=0 and type(r['peak_rss_bytes']) is int and r['peak_rss_bytes']>=0,'finite measured resources')
        resources=r['terminal_resources']
        measured=dict(model_calls=calls,elapsed_seconds=r['elapsed_seconds'],peak_rss_bytes=r['peak_rss_bytes'],
            final_raw_bytes=self.actual_bytes,calls_passed=calls<=490,wall_time_passed=r['elapsed_seconds']<2400,
            rss_passed=r['peak_rss_bytes']<8*2**30,raw_passed=self.actual_bytes<=2304*2**20,
            raw_size_scope='All files including immutable result.json and manifest.json; predicted byte count verified after write.')
        self.metrics(measured,resources,'terminal resource accounting including final files')
        passed=all(measured[k] for k in ('calls_passed','wall_time_passed','rss_passed','raw_passed'))
        self.check(type(r['resource_limits_passed']) is bool and r['resource_limits_passed']==passed,'independent terminal resource decision')
        self.check(passed or r['status']=='incomplete','resource failure cannot be a scientific completion')
        self.check(r['raw_bytes_before_result_manifest']==self.actual_bytes-self.file('result.json').stat().st_size-self.file('manifest.json').stat().st_size,'exact pre-final serialization bytes')
        if r['status'] in ('completed','preflight_passed'):
            self.check(self.actual_bytes<=2304*2**20 and r['elapsed_seconds']<2400 and r['peak_rss_bytes']<8*2**30,'successful terminal resource bounds')

    def call_ledger(self):
        self.plan=expected_calls();slots=self.result['calls'];n=self.result['model_calls']
        self.check(len(slots)==490,'all490 called or unrun schedule slots')
        for i,(want,row) in enumerate(zip(self.plan,slots)):
            for key,value in want.items():self.check(type(row[key]) is type(value) and row[key]==value,'call schedule '+str(i+1)+' '+key)
            if i<n:self.check(row['status'] in ('returned','failed'),'attempt has explicit terminal status')
            else:self.check(row['status']=='not_run','unattempted schedule slot is explicit')
        failed=[s for s in slots[:n] if s['status']=='failed']
        self.check(not failed or len(failed)==1 and failed[0]['call']==n,'first failed invocation stops subsequent attempts')
        self.events=[strict_json(line) for line in self.file('events.jsonl').read_text().splitlines() if line]
        elapsed=-1.
        for event in self.events:
            value=event['elapsed_seconds'];self.check(type(value) in (int,float) and math.isfinite(value) and value>=elapsed,'monotone finite event clock');elapsed=value
        starts=[e for e in self.events if e['event']=='model_call_started']
        ends=[e for e in self.events if e['event']=='model_call_completed']
        self.check([x['call'] for x in starts]==list(range(1,n+1)),'every attempted model invocation starts once')
        self.check([x['call'] for x in ends]==list(range(1,n+1)),'every attempted invocation has retained completion or failure')
        for want,event in zip(self.plan,starts):
            for key,value in want.items():self.check(event[key]==value,'attempt ledger fixed schedule '+key)
            self.check(event['status']=='started','start event precedes mutable return')
        last_finished=[e for e in self.events if e['event']=='run_finished']
        self.check(len(last_finished)==1 and last_finished[0]['status']==self.result['status'] and last_finished[0]['outcome']==self.result['outcome'] and last_finished[0]['model_calls']==n,'terminal event matches result')
        group_events=[e for e in self.events if e['event']=='group_closed']
        self.check([e['group_id'] for e in group_events]==[0,1],'all groups close once in fixed order')
        for gid,summary in enumerate(self.summaries):
            expected=[r for r in slots[:n] if r['group_id']==gid]
            self.check(summary['calls']==expected and summary['model_calls']==len(expected),'group preserves exact attempted slots')
            self.check(group_events[gid]['status']==summary['status'] and group_events[gid]['model_calls']==len(expected),'group closure event matches immutable summary')
        expected_files={'design.json','source.json','events.jsonl','result.json','manifest.json'}
        expected_files.update(k for k in ('literal-inputs.npz','input-preflight.json','backend-readiness.json','failure.json') if (self.root/k).exists())
        blocked_slots=[s for s in slots if 'blocked_evidence' in s]
        self.check(len(blocked_slots)<=1 and all(s['call']==n+1 for s in blocked_slots),'one immediately next blocked invocation at most')
        for gid,summary in enumerate(self.summaries):
            own={f'group-{gid:02d}-summary.json'}
            for suffix in ('prefix-cache.npz','expanded-prefix-cache.npz'):
                name=f'group-{gid:02d}-{suffix}'
                if (self.root/name).exists():own.add(name)
            for key in ('trajectory','tail'):
                if summary[key] is not None:own.add(summary[key]['path'])
            for row in slots[:n]:
                if row['group_id']==gid:
                    stem=f'call-{row["call"]:03d}-{row["label"]}'
                    own.update((stem+'.json',stem+'.npz'))
            for row in blocked_slots:
                if row['group_id']==gid:own.update((row['blocked_evidence'],row['blocked_evidence'][:-5]+'.npz'))
            self.check(own=={name for name,owner in self.file_groups.items() if owner==gid},'immutable group has exactly its evaluated/partial evidence')
            expected_files.update(own|{f'group-{gid:02d}.json'})
        self.check(expected_files=={p.name for p in self.root.iterdir() if p.is_file()},'no unexplained numerical or evidence files')
        self.check(len([e for e in self.events if e['event']=='model_call_blocked'])==len(blocked_slots),'all blocked invocations have explicit uncalled slots')
        self.call_ends={x['call']:x for x in ends}

    def readiness(self):
        if not self.result['model_loaded']:
            self.check(self.result['model_calls']==0,'unloaded model has zero invocations')
            return
        row=self.json('backend-readiness.json')
        if 'readiness' in self.result:self.check(row==self.result['readiness'],'saved native64 readiness summary')
        else:self.check(self.result['status']=='instrument_failed' and self.result['model_calls']==0,'missing readiness result stops before prefix')
        norms={f'model.layers.{i}.{name}' for i in range(24) for name in ('input_layernorm','post_attention_layernorm')}|{'model.norm'}
        actual=row.get('patch',{}).get('norms',[])
        passed=(len(actual)==49 and set(actual)==norms and row['patch'].get('rotaries')==['model.rotary_emb'] and row.get('torch_threads')==row.get('interop_threads')==1 and row.get('mode')=='native64' and all(row.get('thread_env',{}).get(k)==row.get('startup_thread_env',{}).get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')))
        self.check(row['passed']==passed,'independent native64/thread readiness decision')
        if not passed:
            self.check(self.result['status']=='instrument_failed' and self.result['model_calls']==0,'failed readiness stops before prefix')
            self.instrument_failures.append('backend readiness')

    def provenance(self):
        source=self.json('source.json')
        self.check(source==self.result['source'],'source summary unchanged')
        self.source=source
        if self.synthetic:
            self.check(source.get('fixture') is True or source.get('synthetic') is True,'explicit synthetic source')
        else:
            commit=source['source_commit']
            self.check(isinstance(commit,str) and re.fullmatch('[0-9a-f]{40}',commit) and not source['working_tree_status'],'clean frozen source commit')
            self.check(self.source_root is not None,'source root required for production')
            sources=self.index(source['files'],'source')
            for name,(size,sha) in sources.items():
                self.check(not Path(name).is_absolute() and '..' not in Path(name).parts,'safe frozen source path')
                raw=subprocess.check_output(['git','-C',str(self.source_root),'show',commit+':'+name])
                self.check(len(raw)==size and digest(raw)==sha,'frozen source member '+name)
            self.check({'tools/run_public_nominal_handoffs.py','tools/audit_public_nominal_handoffs.py',
                        'tests/test_run_public_nominal_handoffs.py','tests/test_audit_public_nominal_handoffs.py',
                        'tools/run_public_state_intervention.py',
                        'tools/benchmark_qwen_batch.py','tools/validate_qwen_map.py',
                        'tools/check_p6_writes.py','tools/check_p6_hold.py','tools/check_p6_recovery.py',
                        'docs/public-nominal-handoffs.md','pyproject.toml','uv.lock'}<=set(sources),'frozen dependency and checker inventory')
            own=Path(__file__).read_bytes()
            self.check(sources['tools/audit_public_nominal_handoffs.py']==(len(own),digest(own)),'executing exact frozen checker source')
        if not (self.root/'literal-inputs.npz').exists():
            self.check(self.result['model_calls']==0 and self.result['status'] not in ('completed','preflight_passed'),'literal inputs required before any model invocation')
            self.partial.append('Input qualification failed before literal states were saved.')
            return
        self.literal=self.arrays('literal-inputs.npz')
        p=self.literal
        fields={'references','h1','x12','x36','x60','initial_states','historical_state_indices'}
        self.check(set(p)==fields,'literal handoff field inventory')
        finite(p['references'],(2,6,DIM),'literal references')
        for key in ('h1','x12','x36','x60'):finite(p[key],(DIM,),'literal '+key)
        finite(p['initial_states'],(2,2,DIM),'literal group initial states')
        self.same(p['historical_state_indices'],SOURCE_INDICES,'exact original trajectory row indices')
        self.same(p['h1'],p['references'][1,0],'literal H1 phase0')
        initial=np.stack((p['h1'],p['x60'],p['x12'],p['x36'])).reshape(2,2,DIM)
        self.same(p['initial_states'],initial,'literal unmodified state/phase carry')
        preflight=self.json('input-preflight.json');inputs=preflight['inputs']
        self.check(preflight['profile']==PROFILE and preflight['status']=='preflight_passed' and preflight['model_calls']==0 and preflight['model_loaded'] is False,'zero-call input qualification record')
        self.check(inputs==self.result['input_provenance'] and preflight['model']==self.result['model'],'input/model metadata summary')
        self.check(preflight['literal_inputs']==self.result['literal_inputs'],'literal artifact binding')
        self.bound(preflight['literal_inputs'])
        hashes={k:array_hash(v) for k,v in p.items()}
        self.check(hashes==preflight['literal_array_sha256']==self.result['literal_array_sha256'],'all literal array fingerprints')
        if self.synthetic:
            self.check(inputs.get('fixture') is True or inputs.get('synthetic') is True,'explicit synthetic prerequisites')
        else:
            self.pinned_inputs(inputs)
            model=preflight['model']
            self.check(set(model)==set(MODEL_PINS),'complete pinned model metadata')
            for key,pin in MODEL_PINS.items():self.check(model[key]['sha256']==pin,'model payload pin '+key)

    def pinned_inputs(self,inputs):
        records=inputs['bindings']
        self.check(isinstance(records,dict) and set(records)==set(PINS),'six exact historical input bindings')
        paths={}
        for key,(size,sha) in PINS.items():
            row=records[key]
            path=Path(row['path']) if self.input_root is None else self.input_root/Path(row['path']).name
            self.check(row['bytes']==size and row['sha256']==sha,'historical declared pin '+key)
            self.record(path,row);paths[key]=path
        with np.load(paths['fixed_inputs'],allow_pickle=False) as z:
            self.check(set(z.files)=={'references','initial_states'},'historical fixed-input fields')
            finite(z['initial_states'],(4,DIM),'historical initial states')
            self.same(self.literal['references'],z['references'],'original reference cycles')
            self.same(z['initial_states'][1],self.literal['h1'],'original WRITE0 source index1')
        with np.load(paths['trajectory'],allow_pickle=False) as z:
            states=z['states'];finite(states,(121,DIM),'historical endpoint trajectory')
            self.check(int(z['first_state_index'])==0 and z['first_state_index'].shape==(),'historical state-zero indexing')
            self.check(z['token_ids'].shape==(120,) and z['token_ids'].dtype==np.int64,'historical endpoint token shape')
            finite(z['step_seconds'],(120,),'historical endpoint times')
            self.check((z['step_seconds']>=0).all(),'historical endpoint nonnegative time')
            self.same(states[0],self.literal['h1'],'historical trajectory literal H1 phase0')
            for step in (12,36,60):self.same(states[step],self.literal[f'x{step}'],'literal endpoint state'+str(step))
            self.same(states[SOURCE_INDICES],self.literal['initial_states'],'all four exact source slices without phase rotation')
            for key,value in ENDPOINT.items():self.check(z[key].dtype==np.float64 and z[key].shape==() and float(z[key])==value,'historical endpoint control '+key)
        meta=strict_json(paths['historical_metadata'].read_text());plan=strict_json(paths['historical_plan'].read_text())
        oldaudit=strict_json(paths['historical_audit'].read_text());manifest=strict_json(paths['historical_manifest'].read_text())
        self.check(meta['model']=='Qwen/Qwen2.5-0.5B-Instruct' and meta['revision']=='7ae557604adf67be50417f59c2c2f167def9a775' and meta['prefix_ids']==PREFIX,'historical model/prefix identity')
        self.check(meta['hold']==plan['hold']==HOLD and meta['candidate_endpoints']['ad']==ENDPOINT,'historical H/E0 controls')
        trial=plan['trials'][1]
        self.check(trial['trial_id']==1 and trial['source_index']==1 and trial['phase']==0 and trial['desired_label']=='ad','historical trial identity')
        self.check(trial['stages']==[dict(name='endpoint',controls=ENDPOINT,steps=120,status='not_started',completed_steps=0),dict(name='hold',controls=HOLD,steps=240,status='not_started',completed_steps=0)],'historical120/240 schedule')
        self.check(oldaudit['audit']['status']=='audit_passed' and oldaudit['audit']['all_four_cases_verified'] is True and oldaudit['audit']['all_four_writes_correct'] is True,'complete original independent audit')
        index=self.index(manifest['files'],'historical manifest')
        for key,name in [('trajectory','run/trial-01-endpoint-final-120.npz'),('fixed_inputs','run/fixed-trial-inputs.npz'),('historical_metadata','run/metadata.json'),('historical_plan','run/trial-plan.json')]:
            self.check(index.get(name)==PINS[key],'historical manifest member '+key)

    def cache_valid(self,values,batch):
        width=values.get('layer_00_key',np.empty((0,))).shape
        width=width[-1] if self.synthetic and len(width)==4 else 64
        try:validate_cache(values,batch,width);return True
        except AuditError:return False

    def call_arrays(self,row,initial,prefix,expanded):
        name=f'call-{row["call"]:03d}-{row["label"]}'
        saved=self.json(name+'.json')
        self.check(saved==row,'immutable per-attempt JSON matches terminal ledger')
        self.check(row['artifact']['path']==name+'.npz','actual call filename')
        self.bound(row['artifact']);a=self.arrays(name+'.npz')
        self.check(self.file_groups.get(name+'.json')==row['group_id'] and self.file_groups.get(name+'.npz')==row['group_id'],'call evidence assigned to its immutable group')
        self.check(row['array_sha256']=={k:array_hash(v) for k,v in a.items()},'every raw call array fingerprint')
        self.controls(a,'call '+str(row['call']))
        self.same(a['input'],initial,'literal call input and exact state carry')
        end=self.call_ends[row['call']]
        self.check(end['group_id']==row['group_id'] and end['label']==row['label'] and end['status']==row['status'] and end['artifact']==row['artifact'] and end['exception_category']==row.get('exception_category'),'completed invocation event binds raw evidence')
        if row['status']=='failed':
            self.check(row['exception_category']==self.result['status'] and row['exception_category'] in ('instrument_failed','execution_failed','incomplete'),'failed invocation retains execution category')
            error=row['error'].split(':',1)[0]
            category=('incomplete' if error in ('ResourceLimit','TimeoutError','MemoryError','KeyboardInterrupt') else 'instrument_failed' if error in ('InstrumentFailure','ValueError','FloatingPointError') else 'execution_failed')
            self.check(category==row['exception_category'],'independent exception taxonomy')
        returned={'output','probabilities','input_after'}<=set(a)
        if not returned:
            self.check(row['status']=='failed' and 'output' not in a and 'probabilities' not in a,'partial backend exception cannot claim a returned map')
            self.partial.append(dict(call=row['call'],status='no_returned_map',saved_fields=sorted(a)))
            return a,False
        width=row['batch_size'];y,p=a['output'],a['probabilities']
        if not hasattr(self,'vocab'):
            self.vocab=p.shape[1] if self.synthetic and p.ndim==2 else VOCAB
            self.check(type(self.vocab) is int and self.vocab>=2,'probability vocabulary size')
        valid=(y.dtype==p.dtype==np.float64 and y.shape==(width,DIM) and p.shape==(width,self.vocab)
               and np.isfinite(y).all() and np.isfinite(p).all() and np.all(p>=0)
               and np.allclose(p.sum(axis=1),1,rtol=0,atol=1e-12))
        self.check(row['outputs_valid']==bool(valid),'raw output/probability gate reconstructed')
        input_unchanged=same_bytes(a['input'],a['input_after'])
        self.check(not row['input_unchanged'] or input_unchanged,'observed input-unchanged flag requires exact raw input preservation')
        precision=(row['dtype_observations']==PRECISION and row['output_dtype']==row['probability_dtype']=='torch.float64')
        self.check(row['precision_observations_passed']==precision,'fresh per-call precision gate')
        token=p.argmax(axis=1).astype(np.int64) if p.ndim==2 and p.shape[1] else np.empty(0,dtype=np.int64)
        self.same(a['token_ids'],token,'saved full-probability argmax token IDs')
        self.check(type(row['seconds']) in (int,float) and math.isfinite(row['seconds']) and row['seconds']>0,'positive finite measured invocation time')
        if row['category']=='prefix':
            self.check(row['cache_before_sha256'] is None,'prefix has no earlier model cache')
            self.check(prefix is not None and expanded is not None,'returned prefix retains both cache snapshots')
            for key,filename in [('prefix_cache_artifact',f'group-{row["group_id"]:02d}-prefix-cache.npz'),('expanded_cache_artifact',f'group-{row["group_id"]:02d}-expanded-prefix-cache.npz')]:
                self.check(row[key]['path']==filename,'prefix cache filename');self.bound(row[key])
        else:
            self.check(prefix is not None and expanded is not None,'map follows retained prefix')
            self.check(row['cache_before_sha256']==cache_hash(prefix),'map uses the unchanged original prefix')
        ph,eh=cache_hash(prefix),cache_hash(expanded)
        self.check(row['prefix_cache_sha256']==ph and row['expanded_cache_sha256']==eh,'canonical saved cache hashes')
        after={k[len('cache_after_'):]:v for k,v in a.items() if k.startswith('cache_after_')}
        if after:
            self.check(row['cache_after_sha256']==cache_hash(after),'retained anomalous cache hash')
            unchanged=self.cache_valid(after,1) and cache_hash(after)==ph
        else:
            self.check(row['cache_after_sha256']==ph,'changed cache cannot omit anomalous arrays')
            unchanged=self.cache_valid(prefix,1)
        self.check(row['cache_unchanged']==unchanged,'cache preservation gate reconstructed')
        used={k[len('used_cache_'):]:v for k,v in a.items() if k.startswith('used_cache_')}
        if row['cached']:
            if used:
                self.check(row['used_cache_sha256']==cache_hash(used),'retained anomalous expanded cache hash')
                expanded_valid=self.cache_valid(used,2) and cache_hash(used)==eh and row['used_cache_length_after']==38
            else:
                self.check(row['used_cache_sha256']==eh,'used expanded cache fingerprint')
                expanded_valid=self.cache_valid(expanded,2) and row['used_cache_length_after']==38
        else:
            self.check(row['used_cache_sha256'] is None and row['used_cache_length_after'] is None,'full-prefix call does not claim a cached map')
            expanded_valid=True
        self.check(row['expanded_cache_valid']==expanded_valid,'actual expanded cache gate reconstructed')
        normal={'input','input_after','output','probabilities','token_ids','alpha','log10_temperature'}
        self.check(set(a)==normal|{k for k in a if k.startswith(('cache_before_','cache_after_','used_cache_'))},'raw returned field inventory')
        passed=bool(valid and input_unchanged and precision and unchanged and expanded_valid and row['input_unchanged'])
        if not passed:
            self.check(row['status']=='failed' and row['exception_category']=='instrument_failed','instrument defect aborts before further map calls')
            self.instrument_failures.append(dict(call=row['call'],outputs_valid=bool(valid),input_unchanged=input_unchanged,precision=precision,cache_unchanged=unchanged,expanded_cache_valid=expanded_valid))
        if row['status']=='returned':self.check(passed,'returned call passes every instrument gate')
        return a,passed

    def blocked_attempt(self,row,initial,prefix):
        name=row['blocked_evidence'];self.check(name==f'blocked-call-{row["call"]:03d}-{row["label"]}.json','blocked invocation filename')
        blocked=self.json(name);a=self.arrays(name[:-5]+'.npz')
        self.check(blocked['status']=='blocked_before_invocation' and blocked['model_calls']==0 and blocked['call']==row['call'],'blocked forward is not a model invocation')
        self.check(blocked['exception_category']==self.result['status'] and row['status']=='not_run','blocked cause and uncalled ledger')
        self.check(blocked['array_sha256']=={k:array_hash(v) for k,v in a.items()},'all blocked evidence arrays hashed')
        self.bound(blocked['artifact']);self.same(a['input'],initial,'blocked exact input');self.controls(a,'blocked call')
        for filename in (name,blocked['artifact']['path']):self.check(self.file_groups.get(filename)==row['group_id'],'blocked evidence belongs to active group')
        before={k[len('cache_before_'):]:v for k,v in a.items() if k.startswith('cache_before_')}
        if before:
            self.check(prefix is not None and (not self.cache_valid(before,1) or cache_hash(before)!=cache_hash(prefix)),'blocked prefix-cache defect independently observed')
            self.check(blocked['cache_before_sha256']==cache_hash(before),'blocked raw cache fingerprint')
            self.check(blocked['exception_category']=='instrument_failed','pre-invocation cache defect taxonomy')
        else:
            self.check(blocked['exception_category'] in ('incomplete','execution_failed'),'non-cache blocked attempt has operational/resource cause')
        events=[e for e in self.events if e['event']=='model_call_blocked' and e['call']==row['call']]
        self.check(len(events)==1 and events[0]['artifact']==blocked['artifact'] and events[0]['exception_category']==blocked['exception_category'],'zero-call blocked event binds evidence')
        self.partial.append(dict(call=row['call'],status='blocked_before_invocation',category=blocked['exception_category'],saved_fields=sorted(a)))

    def protocol(self):
        slots=self.result['calls'];n=self.result['model_calls'];self.check(hasattr(self,'literal') or n==0,'literal inputs before execution')
        if not hasattr(self,'literal'):
            for gid,summary in enumerate(self.summaries):
                self.check(summary['status']=='not_run' and summary['model_calls']==0 and summary['comparison'] is None and summary['readouts']==[] and summary['trajectory'] is None and summary['tail'] is None,'unqualified inputs leave every group unrun')
                self.rows.extend(dict(row_id=2*gid+lane,group_id=gid,lane=lane,name=ROW_NAMES[2*gid+lane],status='not_run',readout=None) for lane in range(2))
                self.group_reports.append(dict(group_id=gid,name=GROUPS[gid],status='not_run',model_calls=0,returned_updates=0,completed_updates=0,comparison=None))
            self.metrics(self.rows,self.result['rows'],'input-failed four unrun rows')
            self.check(self.result['outcome']=='not_qualified','input failure is not a scientific outcome')
            return
        for gid,summary in enumerate(self.summaries):
            selected=slots[245*gid:min(245*(gid+1),n)]
            initial=self.literal['initial_states'][gid];states=[initial.copy()];completed=0
            prefix=expanded=None
            cp=self.root/f'group-{gid:02d}-prefix-cache.npz';ep=self.root/f'group-{gid:02d}-expanded-prefix-cache.npz'
            if cp.exists():prefix=self.arrays(cp.name)
            if ep.exists():expanded=self.arrays(ep.name)
            if prefix is not None and expanded is not None:
                expected={k:v.copy() if k=='length' else np.repeat(v,2,axis=0) for k,v in prefix.items()}
                self.check(set(expected)==set(expanded) and all(same_bytes(expected[k],expanded[k]) for k in expected),'literal two-copy prefix expansion')
            comparisons=[]
            for local,row in enumerate(selected):
                if local==0:x=np.asarray([PREFIX],dtype=np.int64)
                elif local in (1,2):x=initial
                elif local in (3,4):x=initial[local-3:local-2]
                else:x=states[-1]
                a,valid=self.call_arrays(row,x,prefix,expanded)
                if local in (1,2,3,4) and row['status']=='returned':comparisons.append((a['output'],a['probabilities']))
                if local>=5:
                    if 'output' in a and isinstance(a['output'],np.ndarray) and a['output'].shape==(2,DIM):states.append(a['output'].copy())
                    if row['status']=='returned':completed+=1
            next_index=245*gid+len(selected)
            if next_index<len(slots) and slots[next_index]['group_id']==gid and 'blocked_evidence' in slots[next_index]:
                local=next_index-245*gid
                x=np.asarray([PREFIX],dtype=np.int64) if local==0 else initial if local in (1,2) else initial[local-3:local-2] if local in (3,4) else states[-1]
                self.blocked_attempt(slots[next_index],x,prefix)
            comparison=None
            if len(comparisons)==4:
                stacked=tuple(np.concatenate((comparisons[2][i],comparisons[3][i]),axis=0) for i in range(2))
                comparison={}
                for name,left,right in [('cached_vs_full',comparisons[0],comparisons[1]),('cached_vs_scalars',comparisons[0],stacked),('full_vs_scalars',comparisons[1],stacked)]:
                    gap=float(np.abs(left[0]-right[0]).max());pgap=float(np.abs(left[1]-right[1]).max())
                    self.check(math.isfinite(gap) and math.isfinite(pgap),'finite comparison differences')
                    comparison[name]=dict(passed=max(gap,pgap)<=1e-10,maximum_coordinate_gap=gap,maximum_probability_gap=pgap,tolerance=1e-10)
                self.metrics(comparison,summary['comparison'],'all three comparison gates')
                events=[e for e in self.events if e['event']=='group_comparison' and e['group_id']==gid]
                self.check(len(events)==1 and events[0]['comparison']==summary['comparison'],'comparison event bound to summary')
                if not all(x['passed'] for x in comparison.values()):
                    self.check(summary['status']=='instrument_failed' and len(selected)==5 and completed==0,'failed comparison stops before state carry')
                    self.instrument_failures.append(dict(group_id=gid,comparison=comparison))
            else:self.check(summary['comparison'] is None,'partial comparison cannot qualify')
            self.check(summary['completed_updates']==completed and summary['returned_updates']==len(states)-1,'all returned versus completed updates retained')
            trace=np.stack(states)
            if summary['trajectory'] is not None:
                self.check(summary['trajectory']['path']==f'group-{gid:02d}-states.npz','trajectory filename');self.bound(summary['trajectory'])
                saved=self.arrays(summary['trajectory']['path']);self.controls(saved,'trajectory')
                self.same(saved['states'],trace,'complete or partial exact returned-state chain')
                self.same(saved['state_indices'],np.arange(len(states),dtype=np.int64),'literal state indices')
                self.check(set(saved)=={'states','state_indices','alpha','log10_temperature'},'trajectory field inventory')
            else:self.check(not selected,'invoked group must retain at least its literal initial state')
            computed=[]
            if summary['tail'] is not None:
                self.check(completed==240 and summary['tail']['path']==f'group-{gid:02d}-tail.npz','tail only after all240 successful updates');self.bound(summary['tail'])
                tail=self.arrays(summary['tail']['path']);self.controls(tail,'tail')
                self.same(tail['states'],trace[228:241],'retained last13 exact chronological states')
                self.same(tail['state_indices'],np.arange(228,241,dtype=np.int64),'tail indices228..240')
                try:computed=[readout(tail['states'][:,lane],self.literal['references']) for lane in range(2)]
                except AuditError:
                    self.check(summary['status']=='instrument_failed' and summary['readouts']==[],'nonfinite derived readout aborts instrument')
                    self.instrument_failures.append(dict(group_id=gid,readout='nonfinite derived norm'))
            if summary['status']=='completed':
                self.check(len(selected)==245 and completed==240 and len(computed)==2 and comparison is not None and all(x['passed'] for x in comparison.values()),'completed group has exact full protocol')
                self.metrics(computed,summary['readouts'],'both neutral row readouts')
                for lane,read in enumerate(computed):self.rows.append(dict(row_id=2*gid+lane,group_id=gid,lane=lane,name=ROW_NAMES[2*gid+lane],status='completed',readout=read))
            else:
                self.check(summary['readouts']==[],'failed or unrun group does not publish qualified readout')
                self.rows.extend(dict(row_id=2*gid+lane,group_id=gid,lane=lane,name=ROW_NAMES[2*gid+lane],status=summary['status'],readout=None) for lane in range(2))
            self.group_reports.append(dict(group_id=gid,name=GROUPS[gid],status=summary['status'],model_calls=len(selected),returned_updates=len(states)-1,completed_updates=completed,comparison=comparison))
        self.check(len(self.result['rows'])==4,'all four result rows retained')
        for rebuilt,saved in zip(self.rows,self.result['rows']):self.metrics(rebuilt,saved,'result row '+str(rebuilt['row_id']))
        if self.result['status']=='completed':
            self.check(n==490 and all(r['status']=='completed' for r in self.rows),'finite scientific outcome executes all fixed groups')
            self.check(self.result['outcome']==panel_outcome([x['readout']['decoded_bit'] for x in self.rows]),'neutral panel classification')
        elif self.result['status']=='preflight_passed':
            self.check(n==0 and self.result['model_loaded'] is False and self.result['outcome']=='not_run_preflight','preflight has no numerical observations')
        else:self.check(self.result['outcome']=='not_qualified','execution failure is not a scientific null')

    def transport(self, *, synthetic_transport=False):
        # The standalone rehearsal verifies real copied bytes but cannot attest
        # termination of a model process that it never launched. Production CLI
        # has no bypass for the required process-closure receipt.
        self.check(not synthetic_transport or self.synthetic,'transport rehearsal must be explicitly synthetic')
        if self.synthetic and not synthetic_transport:
            self.return_status='synthetic_not_applicable';self.return_count=0
            return
        job=self.root.parent
        if synthetic_transport:
            offline=self.json('offline-fixture.json',job);self.record(job/'offline-fixture.json')
            self.check(offline['synthetic'] is True and offline['no_model_calls'] is True,'explicit offline transport rehearsal')
        manifest=self.json('final-manifest.json',job);mb=self.record(job/'final-manifest.json')
        index=self.index(manifest['files'],'final transport')
        for row in manifest['files']:self.bound(row,job)
        expected_run={'run/'+p.name for p in self.root.iterdir() if p.is_file()}
        self.check(expected_run<=set(index),'final transport inventory covers every numerical file')
        self.check(sum(v[0] for v in index.values())<=2560*2**20,'complete package byte cap')
        def archive(path,record,expected):
            self.record(path,record)
            self.check(record['bytes']<=3072*2**20,'archive byte cap')
            names=set()
            with tarfile.open(path,'r:*') as tar:
                for member in tar:
                    self.check(member.isfile() and member.name in expected and member.name not in names,'safe unique declared archive member')
                    self.check(member.size==expected[member.name][0],'archive member byte count')
                    h=hashlib.sha256();stream=tar.extractfile(member)
                    while block:=stream.read(1024*1024):h.update(block)
                    self.check(h.hexdigest()==expected[member.name][1],'archive member hash')
                    names.add(member.name)
            self.check(names==set(expected),'complete archive inventory')
        transferred=self.index(manifest['transferred_files'],'transferred subset')
        self.check(all(index.get(k)==v for k,v in transferred.items()),'transferred records match final inventory')
        archive(self.file(manifest['archive']['path'],job),manifest['archive'],transferred)
        already=manifest['already_local_files']
        self.check(type(already) is int and already==len(set(index)-set(transferred)),'already-local count is the verified inventory complement')
        returned_bytes=mb['bytes']+manifest['archive']['bytes']
        for gid in range(2):
            folder=self.root/'return-shards'/f'group-{gid:02d}'
            if not folder.exists():continue
            sm=self.json('manifest.json',folder);sb=self.record(folder/'manifest.json')
            self.check(sm['group_id']==gid,'return shard group identity')
            expected=self.index(sm['files'],'return shard')
            wanted=self.index(self.result['groups'][gid]['artifacts'],'closed group')
            marker=self.record(self.file(f'group-{gid:02d}.json'))
            wanted[f'group-{gid:02d}.json']=(marker['bytes'],marker['sha256'])
            self.check(expected==wanted,'shard contains exactly immutable group and marker')
            returned_bytes+=sb['bytes']
            if not (folder/'verified.json').exists():
                # A failed shard is retained, never silently discarded or
                # misrepresented as a verified incremental return. Final files
                # have independently passed the complete manifest above.
                partial_archive=self.record(folder/'archive.tar') if (folder/'archive.tar').exists() else None
                if partial_archive is not None:returned_bytes+=partial_archive['bytes']
                self.partial.append(dict(group_id=gid,status='unverified_shard_retained',manifest=sb,archive=partial_archive))
                continue
            archive(folder/'archive.tar',sm['archive'],expected)
            returned_bytes+=sm['archive']['bytes']
            verified=self.json('verified.json',folder);self.record(folder/'verified.json')
            self.check(verified['manifest_sha256']==sb['sha256'] and verified['files']==len(expected),'incremental verification binding')
        records={}
        names=['local-return-verification.json','transfer-closeout.json','completion.json',
               'transfer-account.json','maintenance-closeout.json']
        if not synthetic_transport:names.append('terminal-process-check.json')
        for name in names:
            records[name]=self.json(name,job);self.record(job/name)
        v,c=records['local-return-verification.json'],records['transfer-closeout.json']
        self.check(v['status']=='verified' and v['files']==len(index) and v['manifest_sha256']==mb['sha256'] and v['archive']==manifest['archive'],'verified complete return')
        self.check(c['state']=='complete' and c['local_return_verified'] is True and c['verified_manifest_entries']==len(index) and c['manifest_sha256']==mb['sha256'],'reservation closes after full verification')
        self.check(records['completion.json']['transfer_reservation_closed'] is True,'completion closed transfer')
        if not synthetic_transport:
            terminal=records['terminal-process-check.json']
            self.check(terminal['no_owned_numerical_processes'] is True and terminal['remaining_numerical_processes']==[],'no remaining owned numerical processes')
        maintenance=records['maintenance-closeout.json']
        self.check(maintenance['active'] is False and maintenance['return_manifest_sha256']==mb['sha256'],'maintenance released against exact return inventory')
        account=records['transfer-account.json']
        self.check(account['reserved_bytes']==8*2**30 and all(type(r['bytes']) is int and r['bytes']>=0 for r in account['attempts']),'fixed transfer reservation and charged attempts')
        self.check(type(account['overhead_reserve']) is int and account['overhead_reserve']>=0 and sum(r['bytes'] for r in account['attempts'])+account['overhead_reserve']<=account['reserved_bytes'],'transfer accounting within reservation')
        self.check(sum(r['bytes'] for r in account['attempts'])>=returned_bytes,'transfer account includes at least every retained returned archive/manifest byte')
        self.return_status='synthetic_transport_verified' if synthetic_transport else 'verified';self.return_count=len(index)



    def run(self):
        self.inventory();self.provenance();self.call_ledger();self.readiness();self.protocol();self.transport()
        return dict(status='audit_passed',integrity_status='passed',profile=PROFILE,synthetic=self.synthetic,
            scientific_status=self.result['status'],outcome=self.result['outcome'],model_calls=self.result['model_calls'],
            audited_source_commit=self.source.get('source_commit'),assertion_count=self.assertions,
            actual_raw_bytes=self.actual_bytes,return_status=self.return_status,
            returned_manifest_entries=self.return_count,transfer_reservation_closed=not self.synthetic,
            bindings=list(self.bindings.values()),groups=self.group_reports,rows=self.rows,
            instrument_failures=self.instrument_failures,partial_evidence=self.partial,
            scope='Independent saved-array consistency of two earlier literal handoffs and two controls at fixed H. No model replay or new endpoint preparation; no first-acquisition, monotonicity, threshold, basin-volume, branch-loss, text-reachability, general-reliability or causal-hysteresis inference.')


def audit(directory,*,source_root=None,input_root=None,synthetic=False):
    return Audit(directory,source_root=source_root,input_root=input_root,synthetic=synthetic).run()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--input-root',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv);args.output_dir.mkdir(parents=True,exist_ok=False)
    try:
        report=audit(args.run_dir,source_root=args.source_root,input_root=args.input_root);code=0
    except Exception as error:
        report=dict(status='audit_failed',integrity_status='failed',profile=PROFILE,error_type=type(error).__name__,error=str(error));code=1
    report['utc']=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
    report['auditor_sha256']=digest(Path(__file__).read_bytes())
    (args.output_dir/'independent-review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('bindings','rows','partial_evidence','instrument_failures')},indent=2))
    return code


if __name__=='__main__':
    raise SystemExit(main())
