#!/usr/bin/env python3
"""Independent saved-array audit of the fixed carried72 handoff phase panel.

No producer classification imports or model evaluation. Each case carries its
actual E0(72), H144, E1(120), H240 states through nine within-case
boundaries, with literal resets only between the three cases.
Same-phase first36-update replay is a separate diagnostic, never a replacement
state or a neutral decoding gate. No minimum-duration or general reliability
claim follows from this finite numerical observation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tarfile

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np

PROFILE = 'public-carried72-handoff-phase-panel'
DIM, PERIOD, STEPS, MAX_CALLS, VOCAB = 896, 6, 240, 1788, 151936
HOLD = dict(alpha=0.07293697809562831, log10_temperature=-0.15395151820007266)
ENDPOINT = dict(alpha=0.07293697809562831, log10_temperature=-0.1535)
WRITE1 = dict(alpha=0.07299785589927693,log10_temperature=-0.15395151820007266)
CONTROLS = dict(E0=ENDPOINT,H=HOLD,E1=WRITE1)
PACKET_PHASES = (2,3,4,5)
PACKET_INDICES = (1,2,3)
PHASES = (3,4,5)
CASE_COUNT, GROUP_COUNT = 3, 12
STAGE_NAMES = ('intermediate-write0','hold-after-write0','carried-write1','hold-after-write1')
GROUPS = tuple(f'phase{phase}-{stage}' for phase in PHASES for stage in STAGE_NAMES)
CONTROL_NAMES = ('E0','H','E1','H') * CASE_COUNT
GROUP_STEPS = (72,144,120,240) * CASE_COUNT
GROUP_CAPS = (77,149,125,245) * CASE_COUNT
GROUP_OFFSETS = tuple(sum(GROUP_CAPS[:i]) for i in range(GROUP_COUNT+1))
HOLD_GROUPS = tuple(i for i in range(GROUP_COUNT) if i % 2 == 1)
ROW_BASE_NAMES = ('write0-from-h0','write0-from-h1','write1-from-h0','write1-from-h1')
ROW_NAMES = tuple(f'phase{phase}-{name}' for phase in PHASES for name in ROW_BASE_NAMES)
SOURCE_INDICES = np.array([0,1],dtype=np.int64)
PINS = {
    'short_prefix': (2126228, '810b396212c88bad19225d363e53ac2e3327cb315f8a0e003de4fe3bea68db72'),
    'short_prefix_provenance': (46218, 'b22f602afa95a0b321c432bb966cf89502da3b2f42f5463bd65d6177c35d1d4b'),
    'prospective_panel': (55637, '513ebc3988dd50502324986dbe2b2647b0a78e12b056031b6986505373e197a3'),
    'trajectory_h0': (833281, 'dce4af2b24d392d08589a7e747e832f1a690ca11172d544e7f7d5878e5045f67'),
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


def extract_chain_inputs(references, states, tokens, short_states, short_tokens):
    """Preserve the fixed three phase pairs and complete old phase-zero histories."""
    finite(references,(2,PERIOD,DIM),'reference cycles')
    finite(states,(121,2,DIM),'historical trajectories')
    require(tokens.dtype==np.int64 and tokens.shape==(120,2) and np.all((tokens>=0)&(tokens<VOCAB)),
            'historical token IDs')
    initial=np.stack([references[:,phase] for phase in PHASES])
    require(same_bytes(states[0],references[:,0]),'historical initial states are not literal reference phase0')
    finite(short_states,(CASE_COUNT,37,2,DIM),'accepted same-phase short states')
    require(short_tokens.dtype==np.int64 and short_tokens.shape==(CASE_COUNT,36,2) and
            np.all((short_tokens>=0)&(short_tokens<VOCAB)), 'accepted short token IDs')
    for case in range(CASE_COUNT):
        require(same_bytes(short_states[case,0],initial[case]),'short prefix literal phase input')
    return dict(references=references.copy(),initial_states=initial,historical_states=states.copy(),
                historical_token_ids=tokens.copy(),source_reference_indices=SOURCE_INDICES.copy(),
                short_prefix_states=short_states.copy(),short_prefix_token_ids=short_tokens.copy(),
                reference_phases=np.asarray(PHASES,dtype=np.int64),
                packet_indices=np.asarray(PACKET_INDICES,dtype=np.int64),packet_reference_phases=np.asarray(PACKET_PHASES,dtype=np.int64),
                historical_reference_phase=np.asarray(0,dtype=np.int64))


def qualify_prospective_panel(arrays,references):
    """Require all preselected pairs in order, without rotation or relabeling."""
    require(set(arrays)=={'initial_states','reference_indices','reference_phases'},'prospective panel field inventory')
    finite(references,(2,PERIOD,DIM),'prospective reference cycles')
    finite(arrays['initial_states'],(4,2,DIM),'prospective initial pairs')
    require(same_bytes(arrays['reference_indices'],SOURCE_INDICES),'prospective source indices')
    require(same_bytes(arrays['reference_phases'],np.asarray(PACKET_PHASES,dtype=np.int64)),'prospective fixed phases2..5')
    require(same_bytes(arrays['initial_states'],np.stack([references[:,phase] for phase in PACKET_PHASES])),
            'prospective panel literal reference phases2..5')
    return arrays['initial_states'].copy()


def qualify_short_prefix(arrays,references):
    """Validate all four source prefixes without rotating or selecting phases."""
    require(set(arrays)=={'states','token_ids','reference_phases','reference_indices','state_indices','update_indices'},
            'short prefix packet field inventory')
    finite(references,(2,6,DIM),'short prefix references')
    finite(arrays['states'],(4,37,2,DIM),'short prefix states')
    tokens=arrays['token_ids']
    require(tokens.dtype==np.int64 and tokens.shape==(4,36,2) and np.all((tokens>=0)&(tokens<VOCAB)),
            'short prefix token shape/range')
    for key,value in dict(reference_phases=np.asarray(PACKET_PHASES,dtype=np.int64),reference_indices=SOURCE_INDICES,
                          state_indices=np.arange(37,dtype=np.int64),update_indices=np.arange(1,37,dtype=np.int64)).items():
        require(same_bytes(arrays[key],value),'short prefix exact '+key)
    for case,phase in enumerate(PACKET_PHASES):
        require(same_bytes(arrays['states'][case,0],references[:,phase]),'short prefix same-phase literal input')
    return {key:value.copy() for key,value in arrays.items()}


def qualify_prefix_proof(proof,arrays):
    """Proof metadata and array fingerprints; file/chain checks are separate."""
    expected=dict(schema_version=1,status='source_arrays_verified',kind='accepted_same_phase_short_e0_prefix',
        model_calls=0,source_profile='public-short-write0-phase-panel',
        source_commit='3d4ec07967e4919409fd4f52e09a636405f13a8d',accepted_audit_status='audit_passed',
        accepted_audit_synthetic=False,source_group_ids=[0,4,8,12],reference_phases=list(PACKET_PHASES),
        reference_indices=[0,1],controls=ENDPOINT)
    for key,value in expected.items():
        require(type(proof[key]) is type(value) and proof[key]==value,'accepted prefix proof '+key)
    checks={'source_file_hashes','exact_initial_pairs','every_raw_input_and_input_after','every_raw_output',
            'all_token_ids_and_probability_argmax','packet_roundtrip_byte_equality'}
    require(set(proof['checks'])==checks and all(proof['checks'][k] is True for k in checks),'all declared prefix extraction checks')
    require((proof['packet']['bytes'],proof['packet']['sha256'])==PINS['short_prefix'],'proof packet pin')
    require(set(proof['arrays'])==set(arrays),'proof array inventory')
    for key,value in arrays.items():
        expected=dict(dtype=str(value.dtype),shape=list(value.shape),array_sha256=array_hash(value),raw_c_sha256=raw_hash(value))
        require(proof['arrays'][key]==expected,'prefix proof array '+key)


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


def case_outcome(bits):
    require(len(bits)==4 and all(x is None or type(x) is int and x in (0,1) for x in bits),'four neutral row classifications')
    if None in bits:return 'case_unresolved'
    return 'case_qualified' if bits==[0,0,1,1] else 'case_wrong'


def replay_diagnostic(states,tokens,payload,case_index):
    """Measure same-phase replay separately from all neutral readout decisions."""
    require(type(case_index) is int and 0<=case_index<CASE_COUNT,'fixed diagnostic case index')
    finite(states,(37,2,DIM),'fresh endpoint states')
    require(tokens.dtype==np.int64 and tokens.shape==(36,2),'fresh endpoint token shape')
    require(same_bytes(states[0],payload['initial_states'][case_index]),'same-phase prefix fresh literal initial state')
    old=payload['short_prefix_states'][case_index];old_tokens=payload['short_prefix_token_ids'][case_index]
    finite(old,(37,2,DIM),'accepted same-phase prefix')
    require(old_tokens.dtype==np.int64 and old_tokens.shape==(36,2),'accepted same-phase tokens')
    require(same_bytes(states[0],old[0]),'same input replay requires literal identical starts')
    differences=np.stack([np.stack([states[i,lane]-old[i,lane] for lane in range(2)]) for i in range(37)])
    absolute=np.abs(differences)
    require(np.isfinite(absolute).all(),'nonfinite prefix differences')
    maxima=absolute.max(axis=2)
    l2=np.asarray([[float(finite_norm(differences[i,lane])) for lane in range(2)] for i in range(37)])
    byte_equal=np.asarray([[same_bytes(states[i,lane],old[i,lane]) for lane in range(2)] for i in range(37)],dtype=bool)
    token_equal=np.equal(tokens,old_tokens)
    arrays=dict(coordinate_absolute_gaps=absolute,per_state_coordinate_maxima=maxima,
        per_state_l2_differences=l2,state_bytes_equal=byte_equal,token_equal=token_equal,
        state_indices=np.arange(37,dtype=np.int64),update_indices=np.arange(1,37,dtype=np.int64),
        case_index=np.asarray(case_index,dtype=np.int64),reference_phase=np.asarray(PHASES[case_index],dtype=np.int64),packet_index=np.asarray(PACKET_INDICES[case_index],dtype=np.int64))
    lanes=[]
    for lane in range(2):
        lanes.append(dict(lane=lane,source_reference_index=lane,maximum_coordinate_gap=float(np.max(maxima[:,lane])),
            maximum_l2_difference=float(np.max(l2[:,lane])),state_bytes_equal=bool(np.all(byte_equal[:,lane])),
            token_bytes_equal=same_bytes(tokens[:,lane],old_tokens[:,lane]),token_mismatch_count=int(np.count_nonzero(~token_equal[:,lane])),
            coordinate_replay_passed=bool(np.max(maxima[:,lane])<=1e-10),token_replay_passed=bool(np.all(token_equal[:,lane]))))
    metrics=dict(status='completed',state_count=37,update_count=36,lanes=lanes,
        maximum_coordinate_gap=float(np.max(maxima)),maximum_l2_difference=float(np.max(l2)),state_bytes_equal=same_bytes(states,old),
        token_bytes_equal=same_bytes(tokens,old_tokens),fresh_states_array_sha256=array_hash(states),
        historical_states_array_sha256=array_hash(old),fresh_tokens_array_sha256=array_hash(tokens),
        historical_tokens_array_sha256=array_hash(old_tokens),same_input=True,
        comparison_kind='accepted_same_phase_e0_prefix',case_index=case_index,reference_phase=PHASES[case_index],packet_index=PACKET_INDICES[case_index],
        coordinate_tolerance=1e-10,coordinate_replay_passed=bool(np.max(maxima)<=1e-10),
        token_replay_passed=bool(np.all(token_equal)),passed=bool(np.max(maxima)<=1e-10 and np.all(token_equal)),gates_readout=False)
    return arrays,metrics


def panel_outcome(bits):
    require(len(bits)==4*CASE_COUNT,'twelve neutral row classifications')
    cases=[case_outcome(bits[i:i+4]) for i in range(0,4*CASE_COUNT,4)]
    if 'case_unresolved' in cases:return 'panel_unresolved'
    return 'panel_qualified' if all(x=='case_qualified' for x in cases) else 'panel_wrong'


def result_row(group,lane,status,readout=None):
    case,stage=divmod(group,4);desired=stage//2;row_id=case*4+2*desired+lane
    return dict(row_id=row_id,group_id=group,case_index=case,stage_index=stage,
        reference_phase=PHASES[case],packet_index=PACKET_INDICES[case],lane=lane,source_reference_index=lane,
        initial_reference_phase=PHASES[case],requested_bit=desired,name=ROW_NAMES[row_id],status=status,readout=readout,
        write_outcome=None if readout is None else 'unresolved' if readout['decoded_bit'] is None
            else 'correct' if readout['decoded_bit']==desired else 'wrong')


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
    for group in range(GROUP_COUNT):
        planned=[('prefix',0,None,1,'prefix'),('comparison',0,True,2,'compare-cached'),
                 ('comparison',0,False,2,'compare-full'),('comparison',0,False,1,'compare-scalar-0'),
                 ('comparison',0,False,1,'compare-scalar-1')]
        planned += [('update',i,True,2,f'update-{i:03d}') for i in range(1,GROUP_STEPS[group]+1)]
        for category,step,cached,width,label in planned:
            output.append(dict(call=len(output)+1,group_id=group,case_index=group//4,stage_index=group%4,reference_phase=PHASES[group//4],packet_index=PACKET_INDICES[group//4],category=category,step=step,cached=cached,
                               batch_size=width,label=f'g{group:02d}-{label}',control_name=CONTROL_NAMES[group],controls=CONTROLS[CONTROL_NAMES[group]],
                               controls_hex={k:v.hex() for k,v in CONTROLS[CONTROL_NAMES[group]].items()},
                               controls_binary64={k:float_bits(v) for k,v in CONTROLS[CONTROL_NAMES[group]].items()}))
    return output


def float_bits(value):
    return struct.pack('>d',value).hex()


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
        self.cases=[]

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

    def controls(self,values,label,group):
        for key,value in dict(case_index=group//4,stage_index=group%4,reference_phase=PHASES[group//4],packet_index=PACKET_INDICES[group//4]).items():
            self.same(values[key],np.asarray(value,dtype=np.int64),label+' exact '+key)
        self.same(values['control_name'],np.asarray(CONTROL_NAMES[group]),label+' exact named control')
        for key,value in CONTROLS[CONTROL_NAMES[group]].items():
            a=values[key]
            self.same(a,np.asarray(value,dtype=np.float64),label+' exact stage control '+key)
            self.same(values[key+'_bits'],np.asarray(int(float_bits(value),16),dtype=np.uint64),label+' exact control bits '+key)

    def inventory(self):
        self.result=self.json('result.json');r=self.result
        self.check(r['profile']==PROFILE and r['status'] in ('completed','preflight_passed','instrument_failed','execution_failed','incomplete'),'terminal profile/status')
        self.check('cross_phase_history' not in r,'canonical result cannot substitute a different-input history diagnostic')
        manifest=self.json('manifest.json');self.check(manifest['profile']==PROFILE,'manifest profile')
        index=self.index(manifest['files'],'producer manifest')
        for row in manifest['files']:self.bound(row)
        self.record(self.file('manifest.json'))
        actual={p.name for p in self.root.iterdir() if p.is_file()}
        self.check(actual==set(index)|{'manifest.json'},'manifest covers every run file exactly')
        self.check(not any(p.is_symlink() or p.is_dir() and p.name!='return-shards' or p.name.endswith('.pending') for p in self.root.iterdir()),'no symlink, unfinished or unexpected directory')
        self.file_groups={};self.summaries=[]
        self.check(len(r['groups'])==GROUP_COUNT,'all twelve group markers retained')
        for gid,name in enumerate(GROUPS):
            marker=self.json(f'group-{gid:02d}.json');summary=self.json(f'group-{gid:02d}-summary.json')
            self.check('cross_phase_history' not in summary,'canonical group cannot substitute different-input history')
            self.check(marker==r['groups'][gid] and marker['closed'] is True and marker['profile']==PROFILE and marker['group_id']==gid and marker['name']==name,'immutable group marker identity')
            self.check(summary['profile']==PROFILE and summary['group_id']==gid and summary['name']==name and summary['status']==marker['status'] and summary['model_calls']==marker['model_calls'],'group summary identity')
            for key,value in dict(case_index=gid//4,stage_index=gid%4,reference_phase=PHASES[gid//4],packet_index=PACKET_INDICES[gid//4]).items():
                self.check(type(marker[key]) is int and marker[key]==value and type(summary[key]) is int and summary[key]==value,'immutable group exact '+key)
            gi=self.index(marker['artifacts'],'immutable group')
            self.check(f'group-{gid:02d}-summary.json' in gi,'immutable group summary present')
            for row in marker['artifacts']:
                path=self.bound(row);key=path.name
                self.check(index.get(key)==(row['bytes'],row['sha256']),'group member matches final producer inventory')
                self.check(key not in self.file_groups or self.file_groups[key]==gid,'immutable groups do not overlap')
                self.check(key not in ('result.json','manifest.json','events.jsonl') and not re.fullmatch('group-[0-9]{2}.json',key),'immutable group excludes mutable/global marker')
                self.file_groups[key]=gid
            self.check(type(marker['model_calls']) is int and 0<=marker['model_calls']<=GROUP_CAPS[gid],'per-group call cap')
            self.summaries.append(summary)
        calls=r['model_calls'];self.check(type(calls) is int and 0<=calls<=MAX_CALLS,'total invocation cap')
        self.check(sum(g['model_calls'] for g in r['groups'])==calls,'group attempt sum')
        self.design=self.json('design.json');d=self.design
        self.check('cross_phase_history' not in d,'canonical design cannot substitute different-input history')
        self.check(d==r['design'] and d['profile']==PROFILE,'fixed design summary')
        expected=dict(group_names=list(GROUPS),row_names=list(ROW_NAMES),controls=CONTROLS,
            controls_hex={name:{k:v.hex() for k,v in c.items()} for name,c in CONTROLS.items()},
            controls_binary64={name:{k:float_bits(v) for k,v in c.items()} for name,c in CONTROLS.items()},
            source_reference_indices=[0,1],reference_phases=list(PHASES),packet_indices=list(PACKET_INDICES),packet_reference_phases=list(PACKET_PHASES),historical_reference_phase=0,literal_phase_carry=True,
            cases=[dict(case_index=i,reference_phase=phase,packet_index=PACKET_INDICES[i],group_ids=list(range(i*4,i*4+4)),row_ids=list(range(i*4,i*4+4)),initial_source=f'references[:,{phase}]') for i,phase in enumerate(PHASES)],
            within_case_carry_boundaries=[[i*4+j,i*4+j+1] for i in range(CASE_COUNT) for j in range(3)],
            between_case_reset_group_ids=[4,8],case_boundary_policy='Exact declared literal pair at each case start; no other reset.',
            schedule=dict(groups=12,cases=3,control_names=list(CONTROL_NAMES),calls_per_group=list(GROUP_CAPS),prefix_calls=12,comparison_calls=48,comparison_batch_calls=24,comparison_scalar_calls=24,batch_updates=1728,lane_updates=3456,total_calls=1788,steps_per_group=list(GROUP_STEPS),batch_size=2),
            readout=dict(group_ids=list(HOLD_GROUPS),
                retained_state_indices_by_group={str(g):[GROUP_STEPS[g]-12,GROUP_STEPS[g]] for g in HOLD_GROUPS},
                closure_pairs_by_group={str(g):[[i,i-6] for i in range(GROUP_STEPS[g]-5,GROUP_STEPS[g]+1)] for g in HOLD_GROUPS},
                closure_tolerance=1e-8,proper_divisors=[1,2,3],divisor_tolerance=1e-4,reference_match_tolerance=1e-6,all_cyclic_shifts=list(range(6)),requested_bits=[0,0,1,1]*3),
            prefix_replay=dict(group_ids=[0,4,8],source_group_ids=[4,8,12],packet_indices=list(PACKET_INDICES),state_indices=list(range(37)),update_indices=list(range(1,37)),
                reference_phases=list(PHASES),same_input=True,comparison_kind='accepted_same_phase_e0_prefix',
                coordinate_tolerance=1e-10,exact_tokens_required=True,gates_readout=False,mismatch_action='report_separately_continue_finite_plan'),comparison_tolerance=1e-10,
            limits=dict(model_calls=1788,internal_seconds=3600,external_seconds=3660,termination_grace_seconds=30,outer_seconds=3840,return_seconds=7200,threads=1,peak_rss_bytes=8*2**30,external_vm_bytes=16*2**30,raw_bytes=9*2**30,working_raw_bytes=9*2**30-128*2**20,finalization_reserve_bytes=128*2**20,package_bytes=10*2**30,archive_bytes=12*2**30,upload_bytes=8*2**20,transfer_reservation_bytes=32*2**30))
        for key,value in expected.items():self.check(d[key]==value,'frozen design '+key)
        self.actual_bytes=sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())
        self.check(type(r['raw_bytes_before_result_manifest']) is int and 0<=r['raw_bytes_before_result_manifest']<=self.actual_bytes,'raw size accounting')
        self.check(type(r['elapsed_seconds']) in (int,float) and math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds']>=0 and type(r['peak_rss_bytes']) is int and r['peak_rss_bytes']>=0,'finite measured resources')
        resources=r['terminal_resources']
        measured=dict(model_calls=calls,elapsed_seconds=r['elapsed_seconds'],peak_rss_bytes=r['peak_rss_bytes'],
            final_raw_bytes=self.actual_bytes,calls_passed=calls<=MAX_CALLS,wall_time_passed=r['elapsed_seconds']<3600,
            rss_passed=r['peak_rss_bytes']<8*2**30,raw_passed=self.actual_bytes<=9*2**30,
            raw_size_scope='All files including immutable result.json and manifest.json; predicted byte count verified after write.')
        self.metrics(measured,resources,'terminal resource accounting including final files')
        passed=all(measured[k] for k in ('calls_passed','wall_time_passed','rss_passed','raw_passed'))
        self.check(type(r['resource_limits_passed']) is bool and r['resource_limits_passed']==passed,'independent terminal resource decision')
        self.check(passed or r['status']=='incomplete','resource failure cannot be a scientific completion')
        self.check(r['raw_bytes_before_result_manifest']==self.actual_bytes-self.file('result.json').stat().st_size-self.file('manifest.json').stat().st_size,'exact pre-final serialization bytes')
        if r['status'] in ('completed','preflight_passed'):
            self.check(self.actual_bytes<=9*2**30 and r['elapsed_seconds']<3600 and r['peak_rss_bytes']<8*2**30,'successful terminal resource bounds')

    def call_ledger(self):
        self.plan=expected_calls();slots=self.result['calls'];n=self.result['model_calls']
        self.check(len(slots)==MAX_CALLS,'all1788 called or unrun schedule slots')
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
        self.check([e['group_id'] for e in group_events]==list(range(GROUP_COUNT)),'all groups close once in fixed order')
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
            if summary['prefix_replay'] is not None:own.add(summary['prefix_replay']['artifact']['path'])
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
            self.check({'tools/run_public_carried72_handoff_phase_panel.py','tools/audit_public_carried72_handoff_phase_panel.py',
                        'tests/test_public_carried72_handoff_phase_panel.py','tests/test_public_carried72_handoff_phase_panel_audit.py',
                        'tools/run_public_short_write0_chain.py','tests/test_run_public_short_write0_chain.py',
                        'tools/run_public_short_write0_phase_panel.py','tests/test_run_public_short_write0_phase_panel.py',
                        'tools/audit_public_short_write0_phase_panel.py','tests/test_audit_public_short_write0_phase_panel.py',
                        'tools/run_public_short_write0_phase1.py','tests/test_run_public_short_write0_phase1.py',
                        'tools/run_public_nominal_handoffs.py','tools/run_public_state_intervention.py',
                        'tools/benchmark_qwen_batch.py','tools/validate_qwen_map.py',
                        'tools/check_p6_writes.py','tools/check_p6_hold.py','tools/check_p6_recovery.py',
                        'tools/run_public_canonical_write0_phase_panel.py','tests/test_public_canonical_write0_phase_panel.py',
                        'tools/audit_public_canonical_write0_phase_panel.py','tests/test_audit_public_canonical_write0_phase_panel.py',
                        'tools/run_public_intermediate_write0_phase_panel.py','tests/test_public_intermediate_write0_phase_panel.py',
                        'tools/audit_public_intermediate_write0_phase_panel.py','tests/test_public_intermediate_write0_phase_panel_audit.py',
                        'docs/public-intermediate-write0-phase-panel.md','docs/public-canonical-write0-phase-panel.md','docs/public-short-write0-phase-panel.md','docs/public-carried72-handoff-phase-panel.md','pyproject.toml','uv.lock'}<=set(sources),'frozen dependency and checker inventory')
            own=Path(__file__).read_bytes()
            self.check(sources['tools/audit_public_carried72_handoff_phase_panel.py']==(len(own),digest(own)),'executing exact frozen checker source')
        if not (self.root/'literal-inputs.npz').exists():
            self.check(self.result['model_calls']==0 and self.result['status'] not in ('completed','preflight_passed'),'literal inputs required before any model invocation')
            self.partial.append('Input qualification failed before literal states were saved.')
            return
        self.literal=self.arrays('literal-inputs.npz')
        p=self.literal
        fields={'references','initial_states','historical_states','historical_token_ids','source_reference_indices',
                'reference_phases','historical_reference_phase','short_prefix_states','short_prefix_token_ids','packet_indices','packet_reference_phases'}
        self.check(set(p)==fields,'literal chain field inventory')
        finite(p['references'],(2,6,DIM),'literal references')
        rebuilt=extract_chain_inputs(p['references'],p['historical_states'],p['historical_token_ids'],p['short_prefix_states'],p['short_prefix_token_ids'])
        for key,value in rebuilt.items():self.same(p[key],value,'literal unmodified chain input '+key)
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
        self.check(isinstance(records,dict) and set(records)==set(PINS),'ten exact historical/prospective/prefix input bindings')
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
            self.same(z['initial_states'],self.literal['references'][[0,1,0,1],0],'all original phase-zero sources')
            self.same(np.stack([self.literal['references'][:,phase] for phase in PHASES]),self.literal['initial_states'],'all literal phase3..5 initial chain states')
        self.same(self.literal['reference_phases'],np.asarray(PHASES,dtype=np.int64),'literal fixed reference phases3..5')
        self.same(self.literal['historical_reference_phase'],np.asarray(0,dtype=np.int64),'literal historical phase0')
        with np.load(paths['prospective_panel'],allow_pickle=False) as z:
            self.check(len(z.files)==len(set(z.files)),'no duplicate prospective NPZ keys')
            pair=qualify_prospective_panel({k:z[k] for k in z.files},self.literal['references'])
            self.same(pair[list(PACKET_INDICES)],self.literal['initial_states'],'literal initial pair equals preselected input')
        for lane,key in enumerate(('trajectory_h0','trajectory')):
            with np.load(paths[key],allow_pickle=False) as z:
                states=z['states'];finite(states,(121,DIM),'historical endpoint trajectory')
                self.same(z['first_state_index'],np.asarray(0,dtype=np.int64),'historical state-zero indexing')
                tokens=z['token_ids']
                self.check(tokens.shape==(120,) and tokens.dtype==np.int64 and np.all((tokens>=0)&(tokens<VOCAB)),'historical endpoint token shape/range')
                finite(z['step_seconds'],(120,),'historical endpoint times')
                self.check((z['step_seconds']>0).all(),'historical endpoint positive time')
                self.same(states[0],self.literal['references'][lane,0],'historical trajectory literal reference phase0')
                self.same(states,self.literal['historical_states'][:,lane],'all121 exact historical states in lane'+str(lane))
                self.same(tokens,self.literal['historical_token_ids'][:,lane],'all120 historical tokens in lane'+str(lane))
                for name,value in ENDPOINT.items():self.same(z[name],np.asarray(value,dtype=np.float64),'historical endpoint control '+name)
        meta=strict_json(paths['historical_metadata'].read_text());plan=strict_json(paths['historical_plan'].read_text())
        oldaudit=strict_json(paths['historical_audit'].read_text());manifest=strict_json(paths['historical_manifest'].read_text())
        self.check(meta['model']=='Qwen/Qwen2.5-0.5B-Instruct' and meta['revision']=='7ae557604adf67be50417f59c2c2f167def9a775' and meta['prefix_ids']==PREFIX,'historical model/prefix identity')
        self.check(meta['source_commit']=='f0c757ed2c7142bd2dd5cb78a796f69f7f49f4c4','historical frozen source')
        self.check(meta['hold']==plan['hold']==HOLD and meta['candidate_endpoints']['ad']==ENDPOINT and meta['candidate_endpoints']['td']==WRITE1,'historical H/E0/E1 controls')
        for key in ('weights','config','vectors'):self.check(meta[key]['sha256']==MODEL_PINS[key],'historical model/vector pin '+key)
        for i,trial in enumerate(plan['trials']):
            self.check(i<4 and trial['trial_id']==i and trial['source_index']==i%2 and trial['source_label']==('ad','td')[i%2] and trial['phase']==0 and trial['desired_label']==('ad' if i<2 else 'td'),'historical trial identity')
            self.check(trial['stages']==[dict(name='endpoint',controls=ENDPOINT if i<2 else WRITE1,steps=120,status='not_started',completed_steps=0),dict(name='hold',controls=HOLD,steps=240,status='not_started',completed_steps=0)],'historical120/240 schedule')
        self.check(len(plan['trials'])==4,'all four original cases')
        self.check((plan['inputs']['bytes'],plan['inputs']['sha256'])==PINS['fixed_inputs'],'original plan fixed-input pin')
        self.check(oldaudit['audit']['status']=='audit_passed' and oldaudit['audit']['all_four_cases_verified'] is True and oldaudit['audit']['all_four_writes_correct'] is True,'complete original independent audit')
        index=self.index(manifest['files'],'historical manifest')
        oldbindings=self.index(oldaudit['audit']['bindings'],'historical audit')
        for key,name in [('trajectory_h0','run/trial-00-endpoint-final-120.npz'),('trajectory','run/trial-01-endpoint-final-120.npz'),('fixed_inputs','run/fixed-trial-inputs.npz'),('historical_metadata','run/metadata.json'),('historical_plan','run/trial-plan.json')]:
            self.check(index.get(name)==PINS[key],'historical manifest member '+key)
            matches=[value for path,value in oldbindings.items() if Path(path).name==Path(name).name]
            self.check(bool(matches) and all(value==PINS[key] for value in matches),'historical independent audit member '+key)
        self.check(inputs['historical_source_commit']==meta['source_commit'] and inputs['historical_trial_ids']==[0,1] and inputs['historical_endpoint_controls']==ENDPOINT,'declared historical source/trials/controls')
        for key,name in [('historical_states_array_sha256','historical_states'),('historical_tokens_array_sha256','historical_token_ids'),('references_array_sha256','references')]:
            self.check(inputs[key]==array_hash(self.literal[name]),'declared literal historical fingerprint '+key)
        self.check(inputs['prospective_panel_array_sha256']==array_hash(self.literal['initial_states']),
                   'declared prospective panel fingerprint')
        self.check(type(inputs['reference_phases']) is list and all(type(x) is int for x in inputs['reference_phases']) and inputs['reference_phases']==list(PHASES)
                   and type(inputs['historical_reference_phase']) is int and inputs['historical_reference_phase']==0
                   and inputs['same_historical_input'] is False,'declared different historical phase inputs')
        self.check(inputs['initial_pair_array_sha256']==[array_hash(x) for x in self.literal['initial_states']],
                   'every declared initial pair fingerprint')
        with np.load(paths['short_prefix'],allow_pickle=False) as z:
            self.check(len(z.files)==len(set(z.files)),'no duplicate short prefix NPZ keys')
            packet=qualify_short_prefix({key:z[key] for key in z.files},self.literal['references'])
        proof=strict_json(paths['short_prefix_provenance'].read_text());qualify_prefix_proof(proof,packet)
        self.same(packet['states'][list(PACKET_INDICES)],self.literal['short_prefix_states'],'literal accepted short prefix states')
        self.same(packet['token_ids'][list(PACKET_INDICES)],self.literal['short_prefix_token_ids'],'literal accepted short prefix tokens')
        self.check(inputs['short_prefix_states_array_sha256']==array_hash(packet['states'][list(PACKET_INDICES)]) and
                   inputs['short_prefix_tokens_array_sha256']==array_hash(packet['token_ids'][list(PACKET_INDICES)]) and
                   inputs['same_phase_short_prefix_input'] is True and inputs['short_prefix_source_commit']==proof['source_commit'],
                   'declared same-phase prefix provenance')
        self.metrics(list(PACKET_INDICES),inputs['packet_indices'],'declared selected packet indices')
        self.metrics(list(PACKET_PHASES),inputs['packet_reference_phases'],'declared original packet phases')
        self.short_prefix_sources(proof,packet)

    def short_prefix_sources(self,proof,packet):
        """Re-extract only the148 accepted source files, without rerunning an audit."""
        records=proof['source_bindings'];prefix_records=proof['source_prefix_files']
        combined=self.index(records+prefix_records,'accepted prefix provenance')
        self.check(len(self.index(records,'prefix source inventory'))==5 and
                   len(self.index(prefix_records,'prefix raw inventory'))==148 and len(combined)==153,
                   'exact accepted prefix source inventory')
        for name,(size,sha) in combined.items():
            path=Path(name);self.check(path.is_absolute(),'absolute accepted prefix source path')
            self.record(path,dict(bytes=size,sha256=sha))
        def one(suffix):
            matches=[Path(name) for name in self.index(records,'prefix source roles') if name.endswith(suffix)]
            self.check(len(matches)==1,'unambiguous accepted prefix source '+suffix)
            return matches[0]
        receipt_path=one('/docs/receipts/public-short-write0-phase-panel.json')
        audit_path=one('/independent-review.json');result_path=one('/run/result.json')
        run_manifest_path=one('/run/manifest.json');final_manifest_path=one('/final-manifest.json')
        receipt=strict_json(receipt_path.read_text());accepted=strict_json(audit_path.read_text())
        result=strict_json(result_path.read_text());run_index=self.index(strict_json(run_manifest_path.read_text())['files'],'prefix original run manifest')
        final_index=self.index(strict_json(final_manifest_path.read_text())['files'],'prefix original final manifest')
        audit_index=self.index(accepted['bindings'],'prefix accepted audit bindings')
        self.check(receipt['status']=='independently_audited' and receipt['scientific_status']=='completed' and
                   receipt['return_status']=='verified' and receipt['transfer_reservation_closed'] is True,
                   'accepted prefix public receipt is complete')
        for field,path in [('independent_audit',audit_path),('result',result_path)]:
            self.check(self.record(path)==receipt[field],'receipt binds selected prefix '+field)
        self.check(accepted['status']=='audit_passed' and accepted['integrity_status']=='passed' and
                   accepted['synthetic'] is False and accepted['profile']=='public-short-write0-phase-panel' and
                   accepted['scientific_status']==result['status']=='completed' and accepted['return_status']=='verified' and
                   accepted['transfer_reservation_closed'] is True and accepted['audited_source_commit']==proof['source_commit'],
                   'accepted prefix audit status/source')
        run=result_path.parent;wanted=set()
        for case,gid in enumerate((0,4,8,12)):
            group=result['groups'][gid]
            self.check(group['status']=='completed' and group['case_index']==case and group['reference_phase']==PACKET_PHASES[case],
                       'accepted source case identity')
            name=f'group-{gid:02d}-states.npz';path=run/name;wanted.add(str(path))
            with np.load(path,allow_pickle=False) as z:
                self.same(z['states'],packet['states'][case],'all37 accepted source states')
                self.same(z['state_indices'],np.arange(37,dtype=np.int64),'accepted prefix state indices')
            for step in range(1,37):
                number=656*case+5+step
                name=f'call-{number:03d}-g{gid:02d}-update-{step:03d}.npz';path=run/name;wanted.add(str(path))
                with np.load(path,allow_pickle=False) as z:
                    for key,value in (('input',packet['states'][case,step-1]),('input_after',packet['states'][case,step-1]),('output',packet['states'][case,step]),('token_ids',packet['token_ids'][case,step-1])):
                        self.same(z[key],value,'accepted raw prefix '+key)
                    probabilities=z['probabilities'];finite(probabilities,(2,VOCAB),'accepted source full probabilities')
                    self.check(np.all(probabilities>=0) and np.allclose(probabilities.sum(axis=1),1,rtol=0,atol=1e-12),
                               'accepted source probabilities normalized')
                    self.same(probabilities.argmax(axis=1).astype(np.int64),packet['token_ids'][case,step-1],
                              'accepted source exact probability argmax')
                    for key,value in dict(case_index=case,stage_index=0,reference_phase=PACKET_PHASES[case]).items():
                        self.same(z[key],np.asarray(value,dtype=np.int64),'accepted source phase/control '+key)
                    self.same(z['control_name'],np.asarray('E0'),'accepted source E0')
                    for key,value in ENDPOINT.items():
                        self.same(z[key],np.asarray(value,dtype=np.float64),'accepted source binary64 '+key)
                        self.same(z[key+'_bits'],np.asarray(int(float_bits(value),16),dtype=np.uint64),'accepted source control bits '+key)
        self.check(wanted==set(self.index(prefix_records,'prefix source membership')),'all and only148 selected source files')
        for name in wanted:
            pair=combined[name];base=Path(name).name
            self.check(audit_index.get(name)==run_index.get(base)==final_index.get('run/'+base)==pair,
                       'prefix source file present in accepted audit and both manifests')

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
        self.controls(a,'call '+str(row['call']),row['group_id'])
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
        normal={'case_index','stage_index','reference_phase','packet_index','input','input_after','output','probabilities','token_ids','alpha','log10_temperature','control_name','alpha_bits','log10_temperature_bits'}
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
        self.bound(blocked['artifact']);self.same(a['input'],initial,'blocked exact input');self.controls(a,'blocked call',row['group_id'])
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
        self.prefix_replay=[None]*CASE_COUNT
        if not hasattr(self,'literal'):
            for gid,summary in enumerate(self.summaries):
                self.check(summary['status']=='not_run' and summary['model_calls']==0 and summary['comparison'] is None and summary['readouts']==[] and summary['trajectory'] is None and summary['tail'] is None,'unqualified inputs leave every group unrun')
                if gid in HOLD_GROUPS:self.rows.extend(result_row(gid,lane,'not_run') for lane in range(2))
                self.group_reports.append(dict(group_id=gid,case_index=gid//4,stage_index=gid%4,reference_phase=PHASES[gid//4],packet_index=PACKET_INDICES[gid//4],name=GROUPS[gid],status='not_run',model_calls=0,returned_updates=0,completed_updates=0,comparison=None))
            self.metrics(self.rows,self.result['rows'],'input-failed twelve unrun rows')
            self.check(self.result['outcome']=='not_qualified','input failure is not a scientific outcome')
            self.case_summaries()
            return
        carried=None
        for gid,summary in enumerate(self.summaries):
            case,stage=divmod(gid,4)
            if stage==0:carried=self.literal['initial_states'][case].copy()
            start,end=GROUP_OFFSETS[gid:gid+2]
            selected=slots[start:min(end,n)] if start<n else []
            initial=carried.copy();states=[initial.copy()];tokens=[];completed=0
            self.check(summary['control_name']==CONTROL_NAMES[gid] and summary['controls']==CONTROLS[CONTROL_NAMES[gid]]
                and summary['requested_updates']==GROUP_STEPS[gid] and summary['predecessor_group_id']==(gid-1 if stage else None),'fixed stage identity and predecessor')
            control_events=[e for e in self.events if e['event']=='control_selected' and e['group_id']==gid]
            if selected:self.check(len(control_events)==1,'one control selection before invoked stage')
            else:self.check(len(control_events)<=1,'at most one uninvoked control selection')
            for event in control_events:
                for key,value in dict(case_index=case,stage_index=stage,reference_phase=PHASES[case],packet_index=PACKET_INDICES[case]).items():
                    self.check(type(event[key]) is int and event[key]==value,'control event exact '+key)
                self.check(event['control_name']==CONTROL_NAMES[gid] and event['controls']==CONTROLS[CONTROL_NAMES[gid]]
                    and event['controls_hex']=={k:v.hex() for k,v in CONTROLS[CONTROL_NAMES[gid]].items()}
                    and event['initial_state_sha256']==array_hash(initial) and event['predecessor_group_id']==(gid-1 if stage else None),'control event binds literal carried state and parameters')
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
                    if row['status']=='returned':completed+=1;tokens.append(a['token_ids'].copy())
            next_index=start+len(selected)
            if next_index<len(slots) and slots[next_index]['group_id']==gid and 'blocked_evidence' in slots[next_index]:
                local=next_index-start
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
                saved=self.arrays(summary['trajectory']['path']);self.controls(saved,'trajectory',gid)
                self.same(saved['states'],trace,'complete or partial exact returned-state chain')
                self.same(saved['state_indices'],np.arange(len(states),dtype=np.int64),'literal state indices')
                self.check(set(saved)=={'case_index','stage_index','reference_phase','packet_index','states','state_indices','alpha','log10_temperature','control_name','alpha_bits','log10_temperature_bits'},'trajectory field inventory')
                self.check(summary['initial_state_sha256']==array_hash(initial) and summary['terminal_state_sha256']==array_hash(trace[-1]),'exact initial and terminal carried-state fingerprints')
            else:
                self.check(not selected and summary['initial_state_sha256'] is None and summary['terminal_state_sha256'] is None,'uninvoked group has no fabricated state trace')
            if stage==0 and summary['prefix_replay'] is not None:
                self.check(completed==72 and len(states)==73,'same-phase prefix diagnostic is retained after the complete72-update endpoint')
                diagnostic_arrays,diagnostic=replay_diagnostic(trace[:37],np.stack(tokens[:36]),self.literal,case)
                actual=summary['prefix_replay'];self.check(actual['artifact']['path']==f'case-{case:02d}-prefix-replay.npz','same-phase prefix diagnostic filename')
                self.check(set(actual)==set(diagnostic)|{'artifact'},'same-phase prefix diagnostic exact fields')
                self.check(len(actual['lanes'])==2 and all(set(a)==set(b) for a,b in zip(actual['lanes'],diagnostic['lanes'])),
                           'same-phase prefix lane exact fields')
                self.bound(actual['artifact']);saved_diagnostic=self.arrays(f'case-{case:02d}-prefix-replay.npz')
                self.check(set(saved_diagnostic)==set(diagnostic_arrays),'complete same-phase prefix diagnostic field inventory')
                for key,value in diagnostic_arrays.items():
                    if key=='per_state_l2_differences':
                        finite(saved_diagnostic[key],value.shape,'saved same-phase prefix L2 differences')
                        self.metrics(value.tolist(),saved_diagnostic[key].tolist(),'each same-phase prefix L2 difference')
                    else:self.same(saved_diagnostic[key],value,'every same-phase prefix diagnostic '+key)
                self.metrics(diagnostic,actual,'independent same-phase prefix historical comparison')
                self.check(actual==self.result['cases'][case]['prefix_replay'],'same-phase prefix diagnostic case/immutable summary identity')
                self.prefix_replay[case]=diagnostic
            else:
                self.check(summary['prefix_replay'] is None,'historical diagnostic only at first endpoint')
                if stage==0:
                    self.check(summary['status']!='completed','completed first endpoint preserves diagnostic')
                    self.check(self.result['cases'][case]['prefix_replay'] is None,'partial case endpoint cannot claim full same-phase prefix comparison')
            computed=[]
            if summary['tail'] is not None:
                self.check(gid in HOLD_GROUPS and completed==GROUP_STEPS[gid] and summary['tail']['path']==f'group-{gid:02d}-tail.npz','tail only after all declared H updates');self.bound(summary['tail'])
                tail=self.arrays(summary['tail']['path']);self.controls(tail,'tail',gid)
                self.same(tail['states'],trace[GROUP_STEPS[gid]-12:GROUP_STEPS[gid]+1],'retained last13 exact chronological states')
                self.same(tail['state_indices'],np.arange(GROUP_STEPS[gid]-12,GROUP_STEPS[gid]+1,dtype=np.int64),'exact H144 or H240 tail indices')
                self.check(set(tail)=={'case_index','stage_index','reference_phase','packet_index','states','state_indices','alpha','log10_temperature','control_name','alpha_bits','log10_temperature_bits'},'tail field inventory')
                try:computed=[readout(tail['states'][:,lane],self.literal['references']) for lane in range(2)]
                except AuditError:
                    self.check(summary['status']=='instrument_failed' and summary['readouts']==[],'nonfinite derived readout aborts instrument')
                    self.instrument_failures.append(dict(group_id=gid,readout='nonfinite derived norm'))
            if summary['status']=='completed':
                self.check(len(selected)==GROUP_CAPS[gid] and completed==GROUP_STEPS[gid] and comparison is not None and all(x['passed'] for x in comparison.values()),'completed group has exact full protocol')
                if gid in HOLD_GROUPS:
                    self.check(len(computed)==2,'completed H stage has two neutral readouts')
                    self.metrics(computed,summary['readouts'],'both neutral row readouts')
                    self.rows.extend(result_row(gid,lane,'completed',read) for lane,read in enumerate(computed))
                else:self.check(summary['tail'] is None and summary['readouts']==[],'endpoint does not claim H qualification')
                carried=trace[-1].copy()
            else:
                self.check(summary['readouts']==[],'failed or unrun group does not publish qualified readout')
                if gid in HOLD_GROUPS:self.rows.extend(result_row(gid,lane,summary['status']) for lane in range(2))
                self.check(all(g['model_calls']==0 for g in self.summaries[gid+1:]),'failed stage cannot advance later stage')
            self.group_reports.append(dict(group_id=gid,case_index=case,stage_index=stage,reference_phase=PHASES[case],packet_index=PACKET_INDICES[case],name=GROUPS[gid],status=summary['status'],control_name=CONTROL_NAMES[gid],
                model_calls=len(selected),returned_updates=len(states)-1,completed_updates=completed,comparison=comparison,
                initial_state_sha256=summary['initial_state_sha256'],terminal_state_sha256=summary['terminal_state_sha256'],predecessor_group_id=summary['predecessor_group_id']))
        self.check(len(self.result['rows'])==4*CASE_COUNT,'all twelve result rows retained')
        for rebuilt,saved in zip(self.rows,self.result['rows']):self.metrics(rebuilt,saved,'result row '+str(rebuilt['row_id']))
        if self.result['status']=='completed':
            self.check(n==MAX_CALLS and all(r['status']=='completed' for r in self.rows),'finite scientific outcome executes all fixed groups')
            self.check(self.result['outcome']==panel_outcome([x['readout']['decoded_bit'] for x in self.rows]),'neutral chain classification independent of same-phase prefix diagnostic')
        elif self.result['status']=='preflight_passed':
            self.check(n==0 and self.result['model_loaded'] is False and self.result['outcome']=='not_run_preflight','preflight has no numerical observations')
        else:self.check(self.result['outcome']=='not_qualified','execution failure is not a scientific null')
        self.case_summaries()
        aggregate=all(d['passed'] for d in self.prefix_replay) if self.result['status']=='completed' else None
        self.check(type(self.result['prefix_replay_passed']) is type(aggregate) and self.result['prefix_replay_passed']==aggregate,
                   'separate aggregate replay requires all three completed cases')

    def case_summaries(self):
        self.check(len(self.result['cases'])==CASE_COUNT,'all three planned cases retained')
        if self.result['status']!='completed':
            self.check(self.result['prefix_replay_passed'] is None,'unfinished plan has no aggregate replay qualification')
        starts=[e for e in self.events if e['event']=='case_started']
        finishes=[e for e in self.events if e['event']=='case_finished']
        self.check([e['case_index'] for e in starts]==list(range(len(starts))) and len(starts)<=CASE_COUNT,'cases start once in fixed order')
        self.check([e['case_index'] for e in finishes]==list(range(len(finishes))) and len(finishes)<=len(starts),'cases finish once in fixed order')
        self.check(len(starts)-len(finishes) in (0,1),'no later case starts after unfinished case')
        self.check(len(self.result['prefix_replay'])==CASE_COUNT,'all three descriptive history slots retained')
        for case,phase in enumerate(PHASES):
            groups=self.summaries[case*4:case*4+4];rows=self.rows[case*4:case*4+4]
            completed=all(g['status']=='completed' for g in groups)
            started=case<len(starts)
            self.check(not any(g['model_calls'] for g in groups) or started,'case start precedes all its calls')
            status='completed' if completed else self.result['status'] if started else 'not_run'
            outcome=case_outcome([r['readout']['decoded_bit'] for r in rows]) if completed else 'not_qualified'
            initial=array_hash(self.literal['initial_states'][case]) if started else None
            terminal=groups[-1]['terminal_state_sha256'] if completed else None
            expected=dict(case_index=case,reference_phase=phase,packet_index=PACKET_INDICES[case],group_ids=list(range(case*4,case*4+4)),row_ids=list(range(case*4,case*4+4)),
                status=status,outcome=outcome,initial_state_sha256=initial,terminal_state_sha256=terminal,
                prefix_replay=self.prefix_replay[case])
            saved=self.result['cases'][case]
            self.check('cross_phase_history' not in saved,'case cannot substitute different-input history')
            self.metrics(expected,saved,'case status and literal phase '+str(case))
            self.check(saved['prefix_replay']==self.result['prefix_replay'][case],'all case/top history descriptions agree')
            if started:
                expected_start=dict(case_index=case,reference_phase=phase,packet_index=PACKET_INDICES[case],initial_state_sha256=initial,
                    state_origin='declared_literal_reference_pair',reset_between_cases=bool(case),
                    previous_terminal_state_sha256=self.summaries[case*4-1]['terminal_state_sha256'] if case else None)
                self.metrics(expected_start,starts[case],'case literal reset event')
                start_position=self.events.index(starts[case])
                own_starts=[self.events.index(e) for e in self.events if e['event']=='model_call_started' and e['case_index']==case]
                self.check(all(i>start_position for i in own_starts),'case reset event precedes invocation ledger')
                if case:self.check(self.events.index(finishes[case-1])<start_position,'prior case closes before reset')
            if completed:
                self.check(case<len(finishes),'completed case has immutable finish event')
                self.metrics(dict(case_index=case,reference_phase=phase,packet_index=PACKET_INDICES[case],status='completed',outcome=outcome,
                    terminal_state_sha256=terminal),finishes[case],'case terminal outcome event')
            else:self.check(case>=len(finishes),'unfinished case cannot claim a terminal science event')
            self.cases.append(expected)

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
        self.check(mb['bytes']<=4*2**20,'final transport manifest byte cap')
        index=self.index(manifest['files'],'final transport')
        for row in manifest['files']:self.bound(row,job)
        expected_run={'run/'+p.name for p in self.root.iterdir() if p.is_file()}
        self.check(expected_run<=set(index),'final transport inventory covers every numerical file')
        self.check(sum(v[0] for v in index.values())<=10*2**30,'complete package byte cap')
        def archive(path,record,expected):
            self.record(path,record)
            self.check(record['bytes']<=12*2**30,'archive byte cap')
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
        for gid in range(GROUP_COUNT):
            folder=self.root/'return-shards'/f'group-{gid:02d}'
            if not folder.exists():continue
            sm=self.json('manifest.json',folder);sb=self.record(folder/'manifest.json')
            self.check(sb['bytes']<=2**20,'incremental group manifest byte cap')
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
        self.check(account['reserved_bytes']==32*2**30 and all(type(r['bytes']) is int and r['bytes']>=0 for r in account['attempts']),'fixed transfer reservation and charged attempts')
        self.check(type(account['overhead_reserve']) is int and account['overhead_reserve']>=0 and sum(r['bytes'] for r in account['attempts'])+account['overhead_reserve']<=account['reserved_bytes'],'transfer accounting within reservation')
        self.check(sum(r['bytes'] for r in account['attempts'])>=returned_bytes,'transfer account includes at least every retained returned archive/manifest byte')
        self.return_status='synthetic_transport_verified' if synthetic_transport else 'verified';self.return_count=len(index)



    def run(self):
        self.inventory();self.provenance();self.call_ledger();self.readiness();self.protocol();self.transport()
        costs={name:dict(attempted_calls=0,returned_calls=0,failed_calls=0,observed_call_seconds=0.,calls_without_returned_timing=0)
               for name in ('prefix','comparisons','endpoint_updates','hold_updates')}
        for row in self.result['calls'][:self.result['model_calls']]:
            category='prefix' if row['category']=='prefix' else 'comparisons' if row['category']=='comparison' else 'hold_updates' if row['group_id'] in HOLD_GROUPS else 'endpoint_updates'
            measured=costs[category];measured['attempted_calls']+=1;measured['returned_calls']+=int(row['status']=='returned');measured['failed_calls']+=int(row['status']=='failed')
            if type(row.get('seconds')) in (int,float) and math.isfinite(row['seconds']):measured['observed_call_seconds']+=row['seconds']
            else:measured['calls_without_returned_timing']+=1
        return dict(status='audit_passed',integrity_status='passed',profile=PROFILE,synthetic=self.synthetic,
            scientific_status=self.result['status'],outcome=self.result['outcome'],model_calls=self.result['model_calls'],
            audited_source_commit=self.source.get('source_commit'),assertion_count=self.assertions,
            actual_raw_bytes=self.actual_bytes,return_status=self.return_status,
            returned_manifest_entries=self.return_count,transfer_reservation_closed=not self.synthetic,
            bindings=list(self.bindings.values()),groups=self.group_reports,rows=self.rows,cases=self.cases,
            prefix_replay=self.prefix_replay,prefix_replay_passed=self.result['prefix_replay_passed'],call_costs=costs,
            instrument_failures=self.instrument_failures,partial_evidence=self.partial,
            scope='Independent saved-array consistency of E0(72), H144, E1(120), H240 at three literal reference phases3..5 selected from original packet indices1..3, with nine carried boundaries and two literal resets. The first36 updates are compared with accepted same-phase short prefixes, separately from neutral readout qualification and never used as replacement states. No model replay, minimum-duration, monotonicity, attraction, general-reliability or causal-hysteresis inference.')


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
