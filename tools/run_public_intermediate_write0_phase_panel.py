#!/usr/bin/env python3
"""Fixed three-phase intermediate-duration WRITE0 panel with a same-phase prefix diagnostic; all returned calls are retained.

This runner has no tunable state, control, handoff index or iteration CLI.
The injected backend/qualifier arguments are for model-free orchestration tests.
Readout is descriptive finite numerical qualification, not an attractor proof.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback

THREAD_KEYS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
START_THREAD_ENV = {k: os.environ.get(k) for k in THREAD_KEYS}
for _key in THREAD_KEYS:
    os.environ[_key] = '1'
import numpy as np
import benchmark_qwen_batch as batch
import validate_qwen_map as instrument

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'public-intermediate-write0-phase-panel'
DIM, VOCAB, MAX_CALLS = 896, 151936, 966
PHASES = (3, 4, 5)
PACKET_PHASES = (2, 3, 4, 5)
PACKET_INDICES = (1, 2, 3)
CASE_COUNT, GROUP_COUNT = 3, 6
HOLD = dict(alpha=0.07293697809562831, log10_temperature=-0.15395151820007266)
ENDPOINT = dict(alpha=0.07293697809562831, log10_temperature=-0.1535)
WRITE1 = dict(alpha=0.07299785589927693, log10_temperature=-0.15395151820007266)
CONTROLS = dict(E0=ENDPOINT, H=HOLD)
STAGE_NAMES = ('intermediate-write0', 'hold-after-write0')
GROUP_NAMES = tuple(f'phase{phase}-{name}' for phase in PHASES for name in STAGE_NAMES)
CONTROL_NAMES = ('E0', 'H') * CASE_COUNT
GROUP_STEPS = (72, 240) * CASE_COUNT
GROUP_CAPS = (77, 245) * CASE_COUNT
CASE_ROW_NAMES = ('write0-from-h0', 'write0-from-h1')
ROW_NAMES = tuple(f'phase{phase}-{name}' for phase in PHASES for name in CASE_ROW_NAMES)
MAX_SECONDS, EXTERNAL_SECONDS, GRACE, OUTER_SECONDS, RETURN_SECONDS = 3600, 3660, 30, 3840, 7200
MAX_RSS, MAX_VM = 8 * 1024**3, 16 * 1024**3
RAW_CAP, FINAL_RESERVE = 9 * 1024**3, 128 * 1024**2
PACKAGE_CAP, ARCHIVE_CAP, UPLOAD_CAP, TRANSFER_CAP = 10*1024**3, 12*1024**3, 8*1024**2, 32*1024**3
GAP_TOLERANCE, CLOSURE_TOLERANCE, DIVISOR_TOLERANCE, MATCH_TOLERANCE = 1e-10, 1e-8, 1e-4, 1e-6
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
ORIGINAL_NAMES = dict(trajectory_h0='trial-00-endpoint-final-120.npz', trajectory='trial-01-endpoint-final-120.npz', fixed_inputs='fixed-trial-inputs.npz',
    historical_metadata='metadata.json', historical_plan='trial-plan.json',
    historical_audit='complete-audit.json', historical_manifest='final-manifest.json')
MODEL_PINS = dict(weights=instrument.WEIGHTS_SHA256, vectors=instrument.VECTOR_SHA256,
    config='18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45',
    tokenizer='c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    tokenizer_config='5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583')
PRECISION_KEYS = dict(norm=('input', 'reduction', 'weight', 'output'),
    rotary=('input', 'inverse_frequency', 'angles', 'cos', 'sin'))


class InstrumentFailure(ValueError):
    pass


class ResourceLimit(RuntimeError):
    pass


def require(value, message):
    if not bool(value):
        raise InstrumentFailure(message)


def finite(value, shape, label):
    require(isinstance(value, np.ndarray) and value.dtype == np.float64 and value.shape == shape
            and np.isfinite(value).all(), label + ': finite float64 shape required')


def same_bytes(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order='C') == b.tobytes(order='C')


def array_hash(value):
    header = str(value.dtype).encode() + b'\0' + json.dumps(list(value.shape), separators=(',', ':')).encode() + b'\0'
    return hashlib.sha256(header + value.tobytes(order='C')).hexdigest()


def cache_hash(arrays):
    return hashlib.sha256(json.dumps({k: array_hash(v) for k, v in arrays.items()},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def binding(path, flat=False):
    path = Path(path)
    require(path.is_file() and (not path.is_symlink() or not flat), 'missing/unsafe artifact ' + str(path))
    return dict(path=path.name if flat else str(path.resolve()), bytes=path.stat().st_size,
                sha256=instrument.sha256(path))


def checked_norm(value, axis=None):
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.linalg.norm(value, axis=axis)
    require(np.isfinite(result).all(), 'derived norm is nonfinite')
    return result


def load_prospective_panel(path, references):
    """Verify the complete frozen four-phase packet, then select fixed indices."""
    finite(references, (2,6,DIM), 'references for prospective panel')
    with np.load(path, allow_pickle=False) as z:
        require(set(z.files) == {'initial_states','reference_indices','reference_phases'}, 'prospective panel keys differ')
        require(z['reference_indices'].dtype == np.int64 and z['reference_indices'].shape == (2,)
                and same_bytes(z['reference_indices'], np.arange(2,dtype=np.int64)), 'prospective reference indices differ')
        require(z['reference_phases'].dtype == np.int64 and z['reference_phases'].shape == (len(PACKET_PHASES),)
                and same_bytes(z['reference_phases'], np.asarray(PACKET_PHASES,dtype=np.int64)), 'prospective reference phases differ')
        pairs = z['initial_states'].copy()
    finite(pairs, (len(PACKET_PHASES),2,DIM), 'prospective panel')
    for case, phase in enumerate(PACKET_PHASES):
        require(same_bytes(pairs[case], references[:,phase]), 'prospective pair differs from literal reference phase')
    require(tuple(PACKET_PHASES[i] for i in PACKET_INDICES)==PHASES, 'fixed packet selection differs')
    return pairs[list(PACKET_INDICES)].copy()


def derive_inputs(references, historical_states, historical_tokens, prospective_panel, short_states, short_tokens):
    """Three literal starts and immutable same-phase short prefixes; no reference substitution."""
    finite(references, (2,6,DIM), 'reference cycles')
    finite(historical_states, (121,2,DIM), 'historical phase-zero E0 trajectories')
    require(historical_tokens.dtype == np.int64 and historical_tokens.shape == (120,2)
            and np.all((0 <= historical_tokens) & (historical_tokens < VOCAB)), 'historical tokens invalid')
    finite(prospective_panel, (CASE_COUNT,2,DIM), 'prospective panel')
    finite(short_states, (CASE_COUNT,37,2,DIM), 'same-phase short prefix states')
    require(short_tokens.dtype == np.int64 and short_tokens.shape == (CASE_COUNT,36,2)
            and np.all((0 <= short_tokens) & (short_tokens < VOCAB)), 'short prefix tokens invalid')
    for case,phase in enumerate(PHASES):
        require(same_bytes(prospective_panel[case],references[:,phase]), 'initial pair is not the declared literal reference phase')
        require(same_bytes(short_states[case,0],prospective_panel[case]), 'short prefix starts from a different literal pair')
    require(same_bytes(historical_states[0],references[:,0]), 'historical initial pair is not literal reference phase zero')
    return dict(references=references.copy(),initial_states=prospective_panel.copy(),
        historical_states=historical_states.copy(),historical_token_ids=historical_tokens.copy(),
        short_prefix_states=short_states.copy(),short_prefix_token_ids=short_tokens.copy(),
        source_reference_indices=np.arange(2,dtype=np.int64),reference_phases=np.asarray(PHASES,dtype=np.int64),
        historical_reference_phase=np.asarray(0,dtype=np.int64),
        packet_indices=np.asarray(PACKET_INDICES,dtype=np.int64),packet_reference_phases=np.asarray(PACKET_PHASES,dtype=np.int64))


def load_short_prefix(path, provenance_path, references, prospective_panel):
    """Validate the fixed accepted packet and its pinned extraction proof, without remote access."""
    proof=json.loads(Path(provenance_path).read_text())
    require(proof['schema_version']==1 and proof['status']=='source_arrays_verified'
            and proof['kind']=='accepted_same_phase_short_e0_prefix'
            and proof['source_profile']=='public-short-write0-phase-panel'
            and proof['accepted_audit_status']=='audit_passed' and proof['accepted_audit_synthetic'] is False,
            'short prefix accepted provenance differs')
    require(proof['reference_phases']==list(PACKET_PHASES) and proof['reference_indices']==[0,1]
            and proof['source_group_ids']==[0,4,8,12] and proof['controls']==ENDPOINT
            and proof['model_calls']==0 and all(v is True for v in proof['checks'].values()),
            'short prefix source semantics differ')
    row=binding(path)
    require((row['bytes'],row['sha256'])==(proof['packet']['bytes'],proof['packet']['sha256']),
            'short prefix packet/provenance binding differs')
    binding_index(proof['source_bindings']+proof['source_prefix_files'])
    require(len(proof['source_prefix_files'])==148, 'short prefix source inventory differs')
    expected={'states','token_ids','reference_phases','reference_indices','state_indices','update_indices'}
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files)==expected and set(proof['arrays'])==expected, 'short prefix packet keys differ')
        arrays={k:z[k].copy() for k in z.files}
    for key,value in arrays.items():
        p=proof['arrays'][key]
        require(str(value.dtype)==p['dtype'] and list(value.shape)==p['shape']
                and array_hash(value)==p['array_sha256']
                and hashlib.sha256(value.tobytes(order='C')).hexdigest()==p['raw_c_sha256'],
                'short prefix array binding differs: '+key)
    for key,value in dict(reference_phases=np.asarray(PACKET_PHASES,dtype=np.int64),
            reference_indices=np.arange(2,dtype=np.int64),state_indices=np.arange(37,dtype=np.int64),
            update_indices=np.arange(1,37,dtype=np.int64)).items():
        require(same_bytes(arrays[key],value), 'short prefix index differs: '+key)
    finite(arrays['states'],(len(PACKET_PHASES),37,2,DIM),'short prefix states')
    require(arrays['token_ids'].dtype==np.int64 and arrays['token_ids'].shape==(len(PACKET_PHASES),36,2)
            and np.all((0<=arrays['token_ids']) & (arrays['token_ids']<VOCAB)), 'short prefix token array differs')
    for packet_index,phase in enumerate(PACKET_PHASES):
        require(same_bytes(arrays['states'][packet_index,0],references[:,phase]), 'short prefix literal initial differs')
    finite(prospective_panel,(CASE_COUNT,2,DIM),'selected prospective panel')
    for case,packet_index in enumerate(PACKET_INDICES):
        require(same_bytes(arrays['states'][packet_index,0],prospective_panel[case]), 'selected short prefix initial differs')
    selected={k:v.copy() for k,v in arrays.items()}
    for key in ('states','token_ids','reference_phases'):
        selected[key]=arrays[key][list(PACKET_INDICES)].copy()
    return selected,proof


def binding_index(rows):
    output = {}
    for row in rows:
        path = os.path.normpath(row['path'])
        value = (row['bytes'], row['sha256'])
        require(path not in output or output[path] == value, 'conflicting repeated provenance binding')
        output[path] = value
    return output


def require_historical_binding(index, name, pin):
    matches = [value for path, value in index.items() if Path(path).name == name]
    require(bool(matches) and all(value == pin for value in matches), 'historical binding differs: ' + name)


def qualify_inputs(args):
    bindings = {}
    for key, pin in PINS.items():
        row = binding(getattr(args, key))
        require((row['bytes'], row['sha256']) == pin, 'pinned input differs: ' + key)
        bindings[key] = row
    metadata = json.loads(args.historical_metadata.read_text())
    plan = json.loads(args.historical_plan.read_text())
    audit = json.loads(args.historical_audit.read_text())['audit']
    manifest = json.loads(args.historical_manifest.read_text())
    require(audit['status'] == 'audit_passed' and audit['all_four_cases_verified'], 'historical audit failed')
    audit_bindings, manifest_bindings = binding_index(audit['bindings']), binding_index(manifest['files'])
    for key in ('trajectory_h0', 'trajectory', 'fixed_inputs', 'historical_metadata', 'historical_plan'):
        require_historical_binding(audit_bindings, ORIGINAL_NAMES[key], PINS[key])
        require_historical_binding(manifest_bindings, ORIGINAL_NAMES[key], PINS[key])
    require(metadata['source_commit'] == 'f0c757ed2c7142bd2dd5cb78a796f69f7f49f4c4', 'historical source differs')
    require(metadata['model'] == instrument.MODEL and metadata['revision'] == instrument.REVISION
            and metadata['prefix_ids'] == instrument.PREFIX_IDS, 'historical map identity differs')
    require(metadata['hold'] == plan['hold'] == HOLD and metadata['candidate_endpoints']['ad'] == ENDPOINT
            and metadata['candidate_endpoints']['td'] == WRITE1, 'historical controls differ')
    require(all(metadata[k]['sha256'] == MODEL_PINS[k] for k in ('weights', 'config', 'vectors')),
            'historical model/vector pins differ')
    for i in range(4):
        trial = plan['trials'][i]; source = i % 2; desired = 'ad' if i < 2 else 'td'
        endpoint = ENDPOINT if i < 2 else WRITE1
        require((trial['trial_id'], trial['source_index'], trial['source_label'], trial['phase'], trial['desired_label'])
                == (i, source, ('ad', 'td')[source], 0, desired), 'historical trial semantics differ')
        require([(x['name'], x['steps'], x['controls']) for x in trial['stages']]
                == [('endpoint', 120, endpoint), ('hold', 240, HOLD)], 'historical stage plan differs')
    require((plan['inputs']['bytes'], plan['inputs']['sha256']) == PINS['fixed_inputs'], 'historical fixed input binding')
    states, tokens = [], []
    for key in ('trajectory_h0', 'trajectory'):
        with np.load(getattr(args, key), allow_pickle=False) as z:
            finite(z['states'], (121, DIM), 'historical states')
            states.append(z['states'].copy()); tokens.append(z['token_ids'].copy())
            require(z['first_state_index'].dtype == np.int64 and z['first_state_index'].shape == ()
                    and int(z['first_state_index']) == 0, 'historical state indices differ')
            for name, value in ENDPOINT.items():
                finite(z[name], (), 'historical ' + name)
                require(same_bytes(z[name], np.asarray(value)), 'historical binary64 control differs')
            finite(z['step_seconds'], (120,), 'historical times')
            require(np.all(z['step_seconds'] > 0), 'historical nonpositive time')
    with np.load(args.fixed_inputs, allow_pickle=False) as z:
        references, initials = z['references'].copy(), z['initial_states'].copy()
    finite(initials, (4, DIM), 'historical initial states')
    finite(references, (2, 6, DIM), 'historical references')
    require(same_bytes(initials, references[[0, 1, 0, 1], 0]), 'historical fixed phase starts differ')
    pair = load_prospective_panel(args.prospective_panel, references)
    short,short_proof = load_short_prefix(args.short_prefix,args.short_prefix_provenance,references,pair)
    payload = derive_inputs(references,np.stack(states,axis=1),np.stack(tokens,axis=1),pair,short['states'],short['token_ids'])
    return payload, dict(bindings=bindings, historical_source_commit=metadata['source_commit'],
        historical_trial_ids=[0, 1], historical_endpoint_controls=ENDPOINT,
        historical_states_array_sha256=array_hash(payload['historical_states']),
        historical_tokens_array_sha256=array_hash(payload['historical_token_ids']),
        references_array_sha256=array_hash(references), prospective_panel_array_sha256=array_hash(pair),
        initial_pair_array_sha256=[array_hash(p) for p in pair], reference_phases=list(PHASES), historical_reference_phase=0, same_historical_input=False,
        short_prefix_states_array_sha256=array_hash(short['states']),short_prefix_tokens_array_sha256=array_hash(short['token_ids']),
        packet_indices=list(PACKET_INDICES),packet_reference_phases=list(PACKET_PHASES),
        same_phase_short_prefix_input=True,short_prefix_source_commit=short_proof['source_commit'],
        scope='Fixed phases 3..5 selected from packet indices 1..3, each E0(72), H240 with exact carry and declared literal resets. Accepted same-phase short prefixes are diagnostic only.')


def qualify_model(args):
    snapshot = args.cache_dir / 'models--Qwen--Qwen2.5-0.5B-Instruct' / 'snapshots' / instrument.REVISION
    paths = dict(weights=snapshot/'model.safetensors', config=snapshot/'config.json',
                 tokenizer=snapshot/'tokenizer.json', tokenizer_config=snapshot/'tokenizer_config.json', vectors=args.vectors)
    return {k: instrument.require_hash(path, MODEL_PINS[k]) for k, path in paths.items()}


def source_metadata(args):
    names = ['tools/run_public_intermediate_write0_phase_panel.py', 'tests/test_public_intermediate_write0_phase_panel.py',
        'tools/audit_public_intermediate_write0_phase_panel.py', 'tests/test_public_intermediate_write0_phase_panel_audit.py',
        'docs/public-intermediate-write0-phase-panel.md',
        'tools/run_public_canonical_write0_phase_panel.py', 'tests/test_public_canonical_write0_phase_panel.py',
        'tools/audit_public_canonical_write0_phase_panel.py', 'tests/test_audit_public_canonical_write0_phase_panel.py',
        'tools/run_public_short_write0_phase_panel.py', 'tests/test_run_public_short_write0_phase_panel.py',
        'tools/audit_public_short_write0_phase_panel.py', 'tests/test_audit_public_short_write0_phase_panel.py',
        'tools/run_public_short_write0_phase1.py', 'tests/test_run_public_short_write0_phase1.py',
        'tools/run_public_short_write0_chain.py', 'tests/test_run_public_short_write0_chain.py',
        'tools/run_public_nominal_handoffs.py', 'tools/run_public_state_intervention.py',
        'tools/benchmark_qwen_batch.py', 'tools/validate_qwen_map.py', 'tools/check_p6_writes.py',
        'tools/check_p6_hold.py', 'tools/check_p6_recovery.py', 'docs/public-short-write0-phase-panel.md', 'docs/public-canonical-write0-phase-panel.md', 'pyproject.toml', 'uv.lock']
    return dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        files=[{**binding(ROOT/n), 'path': n} for n in names],
        working_tree_status=subprocess.check_output(['git', 'status', '--porcelain', '--', *names], cwd=ROOT, text=True).splitlines(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})



def group_identity(group):
    return dict(case_index=group//2,stage_index=group%2,reference_phase=PHASES[group//2],packet_index=PACKET_INDICES[group//2])


def panel_outcome(bits):
    require(len(bits)==2*CASE_COUNT and all(x is None or type(x) is int and x in (0,1) for x in bits), 'six neutral bits required')
    if any(x is None for x in bits): return 'panel_unresolved'
    return 'panel_qualified' if bits==[0]*(2*CASE_COUNT) else 'panel_wrong'


def case_records():
    return [dict(case_index=i,reference_phase=phase,packet_index=PACKET_INDICES[i],group_ids=[2*i,2*i+1],row_ids=[2*i,2*i+1],
        status='not_run',outcome='not_qualified',initial_state_sha256=None,terminal_state_sha256=None,
        prefix_replay=None) for i,phase in enumerate(PHASES)]


def readout_records():
    rows=[]
    for i,name in enumerate(ROW_NAMES):
        case,lane=divmod(i,2);group=2*case+1
        rows.append(dict(row_id=i,group_id=group,**group_identity(group),lane=lane,source_reference_index=lane,
            initial_reference_phase=PHASES[case],requested_bit=0,name=name,status='not_run',readout=None,write_outcome=None))
    return rows


def call_plan():
    slots = []
    for group, steps in enumerate(GROUP_STEPS):
        control_name = CONTROL_NAMES[group]; controls = CONTROLS[control_name]
        items = [('prefix', 0, None, 1, 'prefix'), ('comparison', 0, True, 2, 'compare-cached'),
                 ('comparison', 0, False, 2, 'compare-full'), ('comparison', 0, False, 1, 'compare-scalar-0'),
                 ('comparison', 0, False, 1, 'compare-scalar-1')]
        items += [('update', step, True, 2, f'update-{step:03d}') for step in range(1, steps+1)]
        for category, step, cached, width, label in items:
            slots.append(dict(call=len(slots)+1, group_id=group, **group_identity(group), category=category, step=step,
                cached=cached, batch_size=width, label=f'g{group:02d}-{label}', status='not_started',
                control_name=control_name, controls=controls.copy(), controls_hex={k: v.hex() for k, v in controls.items()},
                controls_binary64={k: f'{int(np.asarray(v).view(np.uint64)):016x}' for k, v in controls.items()}))
    require(len(slots) == MAX_CALLS and [sum(x['group_id']==i for x in slots) for i in range(GROUP_COUNT)] == list(GROUP_CAPS),
            'fixed schedule arithmetic differs')
    return slots


def derive_design():
    return dict(profile=PROFILE, schema_version=1, evidence='adapted exploratory', group_names=list(GROUP_NAMES),
        row_names=list(ROW_NAMES), controls=CONTROLS,
        controls_hex={name:{k:v.hex() for k,v in c.items()} for name,c in CONTROLS.items()},
        controls_binary64={name:{k:f'{int(np.asarray(v).view(np.uint64)):016x}' for k,v in c.items()} for name,c in CONTROLS.items()},
        source_reference_indices=[0,1], reference_phases=list(PHASES), packet_indices=list(PACKET_INDICES),
        packet_reference_phases=list(PACKET_PHASES), historical_reference_phase=0, literal_phase_carry=True,
        cases=[dict(case_index=i,reference_phase=p,packet_index=PACKET_INDICES[i],group_ids=list(range(2*i,2*i+2)),row_ids=list(range(2*i,2*i+2)),
                    initial_source=f'references[:,{p}]') for i,p in enumerate(PHASES)],
        within_case_carry_boundaries=[[2*i,2*i+1] for i in range(CASE_COUNT)],
        between_case_reset_group_ids=[2,4], case_boundary_policy='Exact declared literal pair at each case start; no other reset.',
        schedule=dict(cases=CASE_COUNT,groups=GROUP_COUNT,control_names=list(CONTROL_NAMES),calls_per_group=list(GROUP_CAPS),
            prefix_calls=6,comparison_calls=24,comparison_batch_calls=12,comparison_scalar_calls=12,
            batch_updates=936,lane_updates=1872,total_calls=MAX_CALLS,steps_per_group=list(GROUP_STEPS),batch_size=2),
        readout=dict(group_ids=[i for i in range(GROUP_COUNT) if i%2 == 1],retained_state_indices=[228,240],
            closure_pairs=[[i,i-6] for i in range(235,241)],closure_tolerance=CLOSURE_TOLERANCE,
            proper_divisors=[1,2,3],divisor_tolerance=DIVISOR_TOLERANCE,reference_match_tolerance=MATCH_TOLERANCE,
            all_cyclic_shifts=list(range(6)),requested_bits=[0]*(2*CASE_COUNT)),
        prefix_replay=dict(group_ids=[0,2,4],source_group_ids=[4,8,12],packet_indices=list(PACKET_INDICES),state_indices=list(range(37)),update_indices=list(range(1,37)),
            reference_phases=list(PHASES),same_input=True,comparison_kind='accepted_same_phase_e0_prefix',
            coordinate_tolerance=GAP_TOLERANCE,exact_tokens_required=True,gates_readout=False,mismatch_action='report_separately_continue_finite_plan'),
        comparison_tolerance=GAP_TOLERANCE,
        limits=dict(model_calls=MAX_CALLS,internal_seconds=MAX_SECONDS,external_seconds=EXTERNAL_SECONDS,
            termination_grace_seconds=GRACE,outer_seconds=OUTER_SECONDS,return_seconds=RETURN_SECONDS,
            threads=1,peak_rss_bytes=MAX_RSS,external_vm_bytes=MAX_VM,raw_bytes=RAW_CAP,
            working_raw_bytes=RAW_CAP-FINAL_RESERVE,finalization_reserve_bytes=FINAL_RESERVE,
            package_bytes=PACKAGE_CAP,archive_bytes=ARCHIVE_CAP,upload_bytes=UPLOAD_CAP,transfer_reservation_bytes=TRANSFER_CAP),
        prefix_output='Soft embedding from the actual prefix logits at the current control; recorded but never carried.',
        cache_storage='One original and one expanded cache per group; canonical hashes every call; anomalous caches retained.',
        scope='Both reference values at literal phases 3,4,5; one fixed 72-update WRITE0 per case. No off-cycle reliability, monotone duration response, minimum duration, speedup, attraction or causal-hysteresis claim.')


def readout(tail, references):
    finite(tail, (13, DIM), 'readout tail'); finite(references, (2, 6, DIM), 'readout references')
    cycle = tail[-6:]; residuals = checked_norm(cycle-tail[-12:-6], axis=1)
    divisors = []
    for shift in (1, 2, 3):
        gaps = checked_norm(cycle-np.roll(cycle, -shift, axis=0), axis=1)
        divisors.append(dict(divisor=shift, phase_l2_gaps=gaps.tolist(), maximum_l2_gap=float(gaps.max()),
                             excluded=bool(gaps.max() > DIVISOR_TOLERANCE)))
    distances = []
    for bit, reference in enumerate(references):
        values = [float(checked_norm(cycle-np.roll(reference, -shift, axis=0), axis=1).max()) for shift in range(6)]
        distances.append(dict(bit=bit, distance=min(values), best_shift=int(np.argmin(values)), by_shift=values,
                              tied_best_shifts=[i for i, v in enumerate(values) if v == min(values)]))
    matches = [d['bit'] for d in distances if d['distance'] <= MATCH_TOLERANCE]
    gate = bool(residuals.max() <= CLOSURE_TOLERANCE and all(d['excluded'] for d in divisors))
    return dict(lag_six_phase_l2_residuals=residuals.tolist(), maximum_l2_residual=float(residuals.max()),
        proper_divisors=divisors, cycle_gate_passed=gate, reference_distances=distances,
        reference_matches=matches, decoded_bit=matches[0] if gate and len(matches) == 1 else None)


def case_outcome(bits):
    require(len(bits)==2 and all(x is None or type(x) is int and x in (0,1) for x in bits), 'two neutral bits required')
    if any(x is None for x in bits): return 'case_unresolved'
    return 'case_qualified' if bits==[0,0] else 'case_wrong'


def prefix_replay(states,tokens,payload,recorder,case_index):
    """Same initial bytes/control; a separately reported diagnostic, never a state source."""
    finite(states,(37,2,DIM),'fresh endpoint prefix states')
    require(tokens.dtype==np.int64 and tokens.shape==(36,2),'fresh endpoint prefix tokens')
    historical=payload['short_prefix_states'][case_index];old_tokens=payload['short_prefix_token_ids'][case_index]
    absolute=np.abs(states-historical);maxima=absolute.max(axis=2);l2=checked_norm(states-historical,axis=2)
    exact=np.asarray([[same_bytes(states[i,lane],historical[i,lane]) for lane in range(2)] for i in range(37)])
    token_equal=tokens==old_tokens
    artifact=recorder.save(f'case-{case_index:02d}-prefix-replay.npz',coordinate_absolute_gaps=absolute,
        per_state_coordinate_maxima=maxima,per_state_l2_differences=l2,state_bytes_equal=exact,token_equal=token_equal,
        state_indices=np.arange(37,dtype=np.int64),update_indices=np.arange(1,37,dtype=np.int64),
        case_index=np.asarray(case_index,dtype=np.int64),reference_phase=np.asarray(PHASES[case_index],dtype=np.int64),
        packet_index=np.asarray(PACKET_INDICES[case_index],dtype=np.int64))
    lanes=[dict(lane=lane,source_reference_index=lane,maximum_coordinate_gap=float(maxima[:,lane].max()),
        maximum_l2_difference=float(l2[:,lane].max()),state_bytes_equal=bool(exact[:,lane].all()),
        token_bytes_equal=same_bytes(tokens[:,lane],old_tokens[:,lane]),token_mismatch_count=int((~token_equal[:,lane]).sum()),
        coordinate_replay_passed=bool(maxima[:,lane].max()<=GAP_TOLERANCE),token_replay_passed=bool(token_equal[:,lane].all())) for lane in range(2)]
    coordinate_passed=bool(maxima.max()<=GAP_TOLERANCE);token_passed=bool(token_equal.all())
    return dict(status='completed',state_count=37,update_count=36,artifact=artifact,lanes=lanes,
        maximum_coordinate_gap=float(maxima.max()),maximum_l2_difference=float(l2.max()),
        state_bytes_equal=same_bytes(states,historical),token_bytes_equal=same_bytes(tokens,old_tokens),
        fresh_states_array_sha256=array_hash(states),historical_states_array_sha256=array_hash(historical),
        fresh_tokens_array_sha256=array_hash(tokens),historical_tokens_array_sha256=array_hash(old_tokens),
        case_index=case_index,reference_phase=PHASES[case_index],packet_index=PACKET_INDICES[case_index],same_input=True,
        comparison_kind='accepted_same_phase_e0_prefix',coordinate_tolerance=GAP_TOLERANCE,
        coordinate_replay_passed=coordinate_passed,token_replay_passed=token_passed,
        passed=bool(coordinate_passed and token_passed),gates_readout=False)


def failure_status(error):
    if isinstance(error, (ResourceLimit, TimeoutError, MemoryError, KeyboardInterrupt)):
        return 'incomplete'
    if isinstance(error, (InstrumentFailure, ValueError, FloatingPointError)):
        return 'instrument_failed'
    return 'execution_failed'


class Recorder:
    def __init__(self, directory):
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic(); self.stream = (self.directory/'events.jsonl').open('x')
        self.active_group = None; self.files = {}; self.owned = {i: [] for i in range(GROUP_COUNT)}; self.finalizing = False
        self.groups = [dict(profile=PROFILE, group_id=i, **group_identity(i), name=name, closed=False, status='not_started',
                            model_calls=0, artifacts=[]) for i, name in enumerate(GROUP_NAMES)]

    def size(self):
        return sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file())

    def admit(self, extra):
        cap = RAW_CAP if self.finalizing else RAW_CAP-FINAL_RESERVE
        if self.size()+extra > cap:
            raise ResourceLimit('raw evidence byte cap')

    def event(self, event, **values):
        row = dict(event=event, elapsed_seconds=time.monotonic()-self.started, **values)
        raw = json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n'
        self.admit(len(raw.encode())); self.stream.write(raw); self.stream.flush(); os.fsync(self.stream.fileno())

    def register(self, path):
        row = binding(path, flat=True); self.files[path.name] = row
        if self.active_group is not None:
            self.owned[self.active_group].append(path.name)
        return row

    def json(self, name, value, atomic=False):
        require(Path(name).name == name, 'flat JSON name required')
        raw = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode(); self.admit(len(raw))
        path = self.directory/name; target = self.directory/(name+'.pending') if atomic else path
        require(not path.exists(), 'immutable artifact already exists: '+name)
        with target.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        if atomic:
            os.link(target, path); target.unlink()
            descriptor = os.open(self.directory, os.O_RDONLY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
        return self.register(path)

    def save(self, name, **arrays):
        require(Path(name).name == name, 'flat NPZ name required')
        arrays = {k: np.asarray(v) for k, v in arrays.items()}
        require(all(a.dtype.kind in 'biufUS' for a in arrays.values()), 'object/complex evidence forbidden')
        self.admit(sum(a.nbytes for a in arrays.values()) + 1024*len(arrays) + 65536)
        path = self.directory/name
        with path.open('xb') as stream:
            np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        return self.register(path)

    def close_group(self, group, summary, calls):
        require(not self.groups[group]['closed'], 'group already closed')
        self.active_group = group
        summary.update(profile=PROFILE, group_id=group, **group_identity(group), name=GROUP_NAMES[group],
                       model_calls=len(calls), calls=copy.deepcopy(calls))
        self.json(f'group-{group:02d}-summary.json', summary)
        marker = dict(profile=PROFILE, group_id=group, **group_identity(group), name=GROUP_NAMES[group], status=summary['status'],
            closed=True, model_calls=len(calls), artifacts=[self.files[n] for n in self.owned[group]])
        if 'not_run_reason' in summary: marker['not_run_reason'] = summary['not_run_reason']
        self.active_group = None
        self.json(f'group-{group:02d}.json', marker, atomic=True)
        self.groups[group] = marker
        self.event('group_closed', group_id=group, status=marker['status'], model_calls=len(calls))


class Budget:
    def __init__(self, recorder, slots):
        self.recorder, self.slots, self.calls = recorder, slots, 0
        self.max_rss = MAX_RSS; self.group = None

    def check(self):
        if time.monotonic()-self.recorder.started >= MAX_SECONDS: raise TimeoutError('internal wall-time cap')
        if instrument.rss_bytes() >= MAX_RSS: raise MemoryError('peak RSS cap')
        self.recorder.admit(0)

    def claim(self, label):
        self.check()
        require(self.calls < MAX_CALLS, 'fixed call budget exhausted')
        slot = self.slots[self.calls]
        require(slot['group_id'] == self.group and (label == slot['label'] or
                label == 'native64_prefix' and slot['category'] == 'prefix'), 'call schedule drift')
        # Admission precedes invocation and reserves a full returned two-lane call plus anomalous caches.
        self.recorder.admit(2*VOCAB*8 + 3*2*DIM*8 + 20*1024**2)
        self.calls += 1; slot['status'] = 'started'
        self.recorder.event('model_call_started', **copy.deepcopy(slot))
        return slot


class QwenBackend(batch.QwenBackend):
    """Existing native64 batch map, retaining the actual prefix return as well."""
    def __init__(self, args, budget, recorder):
        super().__init__(args, budget, recorder)
        # Keep the original pinned vector; never reconstruct it by dividing steering.
        instrument.require_hash(args.vectors, instrument.VECTOR_SHA256)
        with np.load(args.vectors, allow_pickle=False) as archive:
            vector = archive['philosophy'].copy()
        finite(vector, (DIM,), 'original steering vector')
        self.vector = self.torch.tensor(vector, dtype=self.torch.float64)
        self.selected_group = -1

    def select_group(self, group):
        require(type(group) is int and group == self.selected_group+1 and 0 <= group < GROUP_COUNT,
                'fixed control group order differs')
        controls = CONTROLS[CONTROL_NAMES[group]]
        self.loop.steering = controls['alpha'] * self.vector.reshape(1, 1, -1)
        self.loop.temperature = 10.0 ** controls['log10_temperature']
        self.selected_group = group

    def prepare(self):
        from transformers import DynamicCache
        from torch.nn.attention import SDPBackend, sdpa_kernel
        torch, loop = self.torch, self.loop
        loop.budget.claim('native64_prefix'); self.observations.clear(); loop.cache = DynamicCache()
        before = loop.prefix.detach().numpy().copy()
        started = time.perf_counter()
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            result = loop.model(input_ids=loop.prefix, past_key_values=loop.cache, use_cache=True, logits_to_keep=1)
            logits = result.logits[:, -1]
            p = torch.softmax(logits/loop.temperature, dim=-1); y = p @ loop.embedding
        seconds = time.perf_counter()-started
        return y.detach().numpy().copy(), p.detach().numpy().copy(), seconds, dict(
            input_after=loop.prefix.detach().numpy().copy(), input_exact=np.array_equal(before, loop.prefix.detach().numpy()),
            observed_output_dtype=str(y.dtype), observed_probability_dtype=str(p.dtype))

    def runtime_info(self):
        return dict(patch=copy.deepcopy(self.patch), torch_threads=self.torch.get_num_threads(),
            interop_threads=self.torch.get_num_interop_threads(), thread_env={k: os.environ.get(k) for k in THREAD_KEYS},
            startup_thread_env=START_THREAD_ENV.copy(), mode=self.loop.mode)


def precision_ok(observed):
    return all(observed.get(group, {}).get(k) == 'torch.float64' for group, keys in PRECISION_KEYS.items() for k in keys)


def validate_readiness(backend, recorder):
    row = backend.runtime_info()
    expected = {f'model.layers.{i}.{name}' for i in range(24) for name in ('input_layernorm', 'post_attention_layernorm')} | {'model.norm'}
    norms = row.get('patch', {}).get('norms', [])
    row['passed'] = bool(len(norms) == 49 and set(norms) == expected and
        row['patch'].get('rotaries') == ['model.rotary_emb'] and row.get('torch_threads') == row.get('interop_threads') == 1
        and all(row.get('thread_env', {}).get(k) == row.get('startup_thread_env', {}).get(k) == '1' for k in THREAD_KEYS)
        and row.get('mode') == 'native64')
    recorder.json('backend-readiness.json', row)
    require(row['passed'], 'native64 patch/thread readiness failed')
    return row


def raw_controls(group):
    controls = CONTROLS[CONTROL_NAMES[group]]
    return dict(control_name=np.asarray(CONTROL_NAMES[group]),
        **{k:np.asarray(v,dtype=np.int64) for k,v in group_identity(group).items()},
        **{k: np.asarray(v,dtype=np.float64) for k,v in controls.items()},
        **{k+'_bits': np.asarray(v,dtype=np.float64).view(np.uint64) for k,v in controls.items()})


def evaluate(backend, x, recorder, budget, prefix=None, expanded=None, on_return=None):
    """Retain raw results before any instrument/resource gate; failed attempts count."""
    slot = budget.slots[budget.calls]
    is_prefix = slot['category'] == 'prefix'
    arrays = dict(input=x.copy(), **raw_controls(slot['group_id']))
    before = None; row = slot; started_calls = budget.calls
    try:
        if is_prefix:
            y, p, seconds, extra = backend.prepare()  # prepare claims its own one model invocation.
        else:
            before = backend.cache_arrays()
            row['cache_before_sha256'] = cache_hash(before)
            require(batch.valid_cache(before) and cache_hash(before) == cache_hash(prefix), 'prefix cache changed before call')
            budget.claim(row['label'])
            y, p, seconds, extra = backend.forward(x.copy(), row['cached'])
        require(budget.calls == started_calls+1, 'backend prefix invocation was not recorded exactly once')
        arrays.update(output=y, probabilities=p, input_after=extra['input_after'])
        if on_return is not None and isinstance(y, np.ndarray) and y.shape == (2, DIM): on_return(y.copy())
        row.update(seconds=float(seconds), dtype_observations=copy.deepcopy(backend.observations),
            output_dtype=extra['observed_output_dtype'], probability_dtype=extra['observed_probability_dtype'])
        after = backend.cache_arrays(); row['cache_after_sha256'] = cache_hash(after)
        if is_prefix:
            prefix = after
            row['cache_before_sha256'] = None
            row['prefix_cache_artifact'] = recorder.save(f'group-{row["group_id"]:02d}-prefix-cache.npz', **prefix)
            expanded = {k: a.copy() if k == 'length' else np.repeat(a, 2, axis=0) for k, a in prefix.items()}
            row['expanded_cache_artifact'] = recorder.save(f'group-{row["group_id"]:02d}-expanded-prefix-cache.npz', **expanded)
        row['prefix_cache_sha256'] = cache_hash(prefix)
        row['expanded_cache_sha256'] = cache_hash(expanded)
        row['cache_unchanged'] = bool(batch.valid_cache(after) and cache_hash(after) == cache_hash(prefix))
        if not row['cache_unchanged']:
            arrays.update({'cache_after_'+k: v for k, v in after.items()})
        used_valid = True
        if not is_prefix and row['cached']:
            used = dict(extra.get('used_cache_before', {})); used['length'] = np.asarray(len(instrument.PREFIX_IDS), dtype=np.int64)
            row['used_cache_sha256'] = cache_hash(used)
            row['used_cache_length_after'] = int(extra.get('used_cache_length_after', -1))
            used_valid = (batch.valid_cache(used, batch_size=2) and cache_hash(used) == cache_hash(expanded)
                          and row['used_cache_length_after'] == len(instrument.PREFIX_IDS)+1)
            if not used_valid: arrays.update({'used_cache_'+k: v for k, v in used.items()})
        else:
            row['used_cache_sha256'] = None; row['used_cache_length_after'] = None
        width = row['batch_size']
        valid = bool(y.dtype == p.dtype == np.float64 and y.shape == (width, DIM) and p.shape == (width, VOCAB)
            and np.isfinite(y).all() and np.isfinite(p).all() and np.all(p >= 0)
            and np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12))
        arrays['token_ids'] = p.argmax(axis=1).astype(np.int64) if p.ndim == 2 and p.shape[1] else np.empty(0, dtype=np.int64)
        row.update(input_unchanged=bool(extra['input_exact'] and same_bytes(x, arrays['input_after'])),
            expanded_cache_valid=bool(used_valid), outputs_valid=valid,
            precision_observations_passed=bool(precision_ok(row['dtype_observations'])
                and row['output_dtype'] == row['probability_dtype'] == 'torch.float64'))
        require(all(row[k] for k in ('cache_unchanged', 'input_unchanged', 'expanded_cache_valid', 'outputs_valid',
                'precision_observations_passed')), 'input/cache/output/native64 precision gate failed')
        require(math.isfinite(seconds) and seconds > 0, 'invalid measured forward duration')
        budget.check(); row['status'] = 'returned'
    except BaseException as error:
        row.update(status='failed' if budget.calls > started_calls else 'not_started',
                   exception_category=failure_status(error), error=f'{type(error).__name__}: {error}',
                   dtype_observations=copy.deepcopy(backend.observations))
        if before is not None and prefix is not None and cache_hash(before) != cache_hash(prefix):
            arrays.update({'cache_before_'+k: v for k, v in before.items()})
        raise
    finally:
        if budget.calls > started_calls:
            name = f'call-{row["call"]:03d}-{row["label"]}'
            with batch.preserve_call_files():
                row['artifact'] = recorder.save(name+'.npz', **arrays)
                row['array_sha256'] = {k: array_hash(a) for k, a in arrays.items()}
                recorder.json(name+'.json', row)
                recorder.event('model_call_completed', call=row['call'], group_id=row['group_id'], label=row['label'],
                    status=row['status'], exception_category=row.get('exception_category'), artifact=row['artifact'])
        elif 'error' in row:
            # No attempted invocation: retain the obstruction outside the call ledger.
            name = f'blocked-call-{row["call"]:03d}-{row["label"]}'
            with batch.preserve_call_files():
                blocked = copy.deepcopy(row)
                blocked.update(status='blocked_before_invocation', model_calls=0,
                    artifact=recorder.save(name+'.npz', **arrays))
                blocked['array_sha256'] = {k: array_hash(a) for k, a in arrays.items()}
                recorder.json(name+'.json', blocked)
                recorder.event('model_call_blocked', call=row['call'], group_id=row['group_id'],
                    exception_category=blocked['exception_category'], artifact=blocked['artifact'])
                row['blocked_evidence'] = name+'.json'
    return (y, p), prefix, expanded


def compare(left, right):
    gap = float(np.max(np.abs(left[0]-right[0]))); pgap = float(np.max(np.abs(left[1]-right[1])))
    return dict(passed=bool(max(gap, pgap) <= GAP_TOLERANCE), maximum_coordinate_gap=gap,
                maximum_probability_gap=pgap, tolerance=GAP_TOLERANCE)


def execute_group(group, initial, backend, payload, recorder, budget):
    recorder.active_group = group; budget.group = group
    finite(initial, (2, DIM), 'literal carried initial pair')
    states = [initial.copy()]; tokens = []; steps = GROUP_STEPS[group]
    summary = dict(status='started', control_name=CONTROL_NAMES[group], controls=CONTROLS[CONTROL_NAMES[group]].copy(),
        requested_updates=steps, initial_state_sha256=array_hash(states[0]), terminal_state_sha256=None,
        predecessor_group_id=group-1 if group%2 else None, comparison=None, readouts=[],
        completed_updates=0, returned_updates=0, trajectory=None, tail=None, prefix_replay=None)
    try:
        backend.select_group(group)
        recorder.event('control_selected', group_id=group, **group_identity(group), control_name=CONTROL_NAMES[group],
            controls=summary['controls'], controls_hex={k:v.hex() for k,v in summary['controls'].items()},
            initial_state_sha256=summary['initial_state_sha256'], predecessor_group_id=summary['predecessor_group_id'])
        _, prefix, expanded = evaluate(backend, np.asarray([instrument.PREFIX_IDS], dtype=np.int64), recorder, budget)
        outputs = []
        for lane in (None, None, 0, 1):
            x = states[0] if lane is None else states[0][lane:lane+1]
            value, _, _ = evaluate(backend, x, recorder, budget, prefix, expanded); outputs.append(value)
        stacked = tuple(np.concatenate((outputs[2][i], outputs[3][i]), axis=0) for i in range(2))
        summary['comparison'] = dict(cached_vs_full=compare(outputs[0], outputs[1]),
            cached_vs_scalars=compare(outputs[0], stacked), full_vs_scalars=compare(outputs[1], stacked))
        recorder.event('group_comparison', group_id=group, comparison=summary['comparison'])
        require(all(row['passed'] for row in summary['comparison'].values()), 'batch/full/scalar comparison failed')
        del outputs, stacked
        for _ in range(steps):
            value, _, _ = evaluate(backend, states[-1], recorder, budget, prefix, expanded, on_return=states.append)
            tokens.append(value[1].argmax(axis=1).astype(np.int64))
            summary['completed_updates'] += 1
        if group%2 == 0:
            summary['prefix_replay'] = prefix_replay(np.stack(states[:37]),np.stack(tokens[:36]),payload,recorder,group//2)
        if group%2 == 1:
            tail = np.stack(states[-13:])
            summary['tail'] = recorder.save(f'group-{group:02d}-tail.npz', states=tail,
                state_indices=np.arange(228, 241, dtype=np.int64), **raw_controls(group))
            summary['readouts'] = [readout(tail[:, lane], payload['references']) for lane in range(2)]
        summary['status'] = 'completed'
    except BaseException as error:
        summary.update(status=failure_status(error), error=f'{type(error).__name__}: {error}')
        raise
    finally:
        was_final = recorder.finalizing; recorder.finalizing = True
        try:
            summary['returned_updates'] = len(states)-1
            summary['terminal_state_sha256'] = array_hash(states[-1])
            summary['trajectory'] = recorder.save(f'group-{group:02d}-states.npz', states=np.stack(states),
                state_indices=np.arange(len(states), dtype=np.int64), **raw_controls(group))
            calls = [row for row in budget.slots[:budget.calls] if row['group_id'] == group]
            recorder.close_group(group, summary, calls)
        finally:
            recorder.finalizing = was_final
    return summary, states[-1].copy()


def run(args, *, backend_factory=QwenBackend, input_qualifier=qualify_inputs,
        model_qualifier=qualify_model, source_factory=source_metadata):
    """One execution; injection never changes the production CLI or fixed schedule."""
    recorder = Recorder(args.output_dir); slots = call_plan(); budget = Budget(recorder, slots)
    result = dict(profile=PROFILE, status='started', outcome='not_qualified', model_loaded=False,
        design=derive_design(), source=None, model_calls=0, groups=recorder.groups, calls=slots,
        rows=readout_records(), cases=case_records(), prefix_replay=[None]*CASE_COUNT,prefix_replay_passed=None)
    try:
        recorder.json('design.json', result['design'])
        result['source'] = source_factory(args); recorder.json('source.json', result['source'])
        payload, provenance = input_qualifier(args)
        model = model_qualifier(args)
        result['input_provenance'] = provenance; result['model'] = model
        result['literal_inputs'] = recorder.save('literal-inputs.npz', **payload)
        result['literal_array_sha256'] = {k: array_hash(a) for k, a in payload.items()}
        recorder.json('input-preflight.json', dict(profile=PROFILE, status='preflight_passed', model_loaded=False,
            model_calls=0, inputs=provenance, model=model, literal_inputs=result['literal_inputs'],
            literal_array_sha256=result['literal_array_sha256']))
        if args.preflight_only:
            result.update(status='preflight_passed', outcome='not_run_preflight')
        else:
            budget.check(); backend = backend_factory(args, budget, recorder); result['model_loaded'] = True
            result['readiness'] = validate_readiness(backend, recorder)
            previous_terminal = None
            for case_index, phase in enumerate(PHASES):
                case = result['cases'][case_index]
                carried = payload['initial_states'][case_index].copy()
                finite(carried,(2,DIM),'declared case initial pair')
                require(same_bytes(carried,payload['references'][:,phase]), 'literal case start differs')
                case.update(status='started',initial_state_sha256=array_hash(carried))
                recorder.active_group = None
                recorder.event('case_started',case_index=case_index,reference_phase=phase,packet_index=PACKET_INDICES[case_index],
                    initial_state_sha256=case['initial_state_sha256'],state_origin='declared_literal_reference_pair',
                    reset_between_cases=bool(case_index),previous_terminal_state_sha256=previous_terminal)
                for stage_index in range(2):
                    group = 2*case_index+stage_index
                    summary, carried = execute_group(group,carried,backend,payload,recorder,budget)
                    if stage_index == 0:
                        case['prefix_replay'] = summary['prefix_replay']
                        result['prefix_replay'][case_index] = summary['prefix_replay']
                    if stage_index == 1:
                        for lane in range(2):
                            row = result['rows'][2*case_index+lane]
                            decoded = summary['readouts'][lane]['decoded_bit']
                            row.update(status='completed',readout=summary['readouts'][lane],
                                write_outcome='unresolved' if decoded is None else 'correct' if decoded == row['requested_bit'] else 'wrong')
                bits = [result['rows'][i]['readout']['decoded_bit'] for i in case['row_ids']]
                case.update(status='completed',outcome=case_outcome(bits),terminal_state_sha256=array_hash(carried))
                previous_terminal = case['terminal_state_sha256']
                recorder.event('case_finished',case_index=case_index,reference_phase=phase,packet_index=PACKET_INDICES[case_index],status=case['status'],
                    outcome=case['outcome'],terminal_state_sha256=previous_terminal)
            require(budget.calls == MAX_CALLS, 'completed fixed plan has wrong invocation count')
            result.update(status='completed',outcome=panel_outcome([r['readout']['decoded_bit'] for r in result['rows']]),
                prefix_replay_passed=all(d['passed'] for d in result['prefix_replay']))
    except BaseException as error:
        result.update(status=failure_status(error), outcome='not_qualified', error=f'{type(error).__name__}: {error}')
        recorder.finalizing = True
        recorder.json('failure.json', dict(status=result['status'], error=result['error'], traceback=traceback.format_exc()))
    finally:
        recorder.finalizing = True; recorder.active_group = None
        for slot in slots[budget.calls:]:
            slot.update(status='not_run', not_run_reason=result['status'])
        for group in range(GROUP_COUNT):
            if not recorder.groups[group]['closed']:
                calls = [row for row in slots[:budget.calls] if row['group_id'] == group]
                recorder.close_group(group, dict(status=result['status'] if calls else 'not_run',
                    not_run_reason=result.get('error', result['outcome']), completed_updates=0, returned_updates=0,
                    comparison=None, readouts=[], trajectory=None, tail=None, prefix_replay=None,
                    control_name=CONTROL_NAMES[group], controls=CONTROLS[CONTROL_NAMES[group]].copy(),
                    requested_updates=GROUP_STEPS[group], initial_state_sha256=None, terminal_state_sha256=None,
                    predecessor_group_id=group-1 if group%2 else None), calls)
        result.update(model_calls=budget.calls, groups=recorder.groups,
            elapsed_seconds=time.monotonic()-recorder.started, peak_rss_bytes=instrument.rss_bytes())
        for row in result['rows']:
            if row['status'] != 'completed':
                row['status'] = recorder.groups[row['group_id']]['status']
                row['not_run_reason'] = result.get('error', result['outcome'])
        for case in result['cases']:
            if case['status'] != 'completed':
                case['status'] = result['status'] if case['status'] == 'started' else 'not_run'
                case['outcome'] = 'not_qualified'
                case['not_run_reason'] = result.get('error',result['outcome'])
        if result['elapsed_seconds'] >= MAX_SECONDS or result['peak_rss_bytes'] >= MAX_RSS:
            result.update(status='incomplete', outcome='not_qualified', error='final measured resource cap')
        recorder.event('run_finished', status=result['status'], outcome=result['outcome'], model_calls=budget.calls)
        recorder.stream.close()
        files = [binding(p, flat=True) for p in sorted(recorder.directory.iterdir()) if p.is_file()]
        result['raw_bytes_before_result_manifest'] = recorder.size()
        result.update(elapsed_seconds=time.monotonic()-recorder.started, peak_rss_bytes=instrument.rss_bytes())
        resources = dict(model_calls=budget.calls, elapsed_seconds=result['elapsed_seconds'],
            peak_rss_bytes=result['peak_rss_bytes'], final_raw_bytes=0,
            calls_passed=budget.calls <= MAX_CALLS, wall_time_passed=result['elapsed_seconds'] < MAX_SECONDS,
            rss_passed=result['peak_rss_bytes'] < MAX_RSS, raw_passed=True,
            raw_size_scope='All files including immutable result.json and manifest.json; predicted byte count verified after write.')
        result['terminal_resources'] = resources
        # Resolve the self-referential byte-count digits before immutable publication.
        # Hashes have fixed length; no compressed-size assumption enters this calculation.
        for _ in range(12):
            result['resource_limits_passed'] = all(resources[k] for k in ('calls_passed','wall_time_passed','rss_passed','raw_passed'))
            if not result['resource_limits_passed']:
                result.update(status='incomplete', outcome='not_qualified', error='final measured resource cap')
            encoded = (json.dumps(result, indent=2, allow_nan=False)+'\n').encode()
            result_binding = dict(path='result.json', bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest())
            manifest = dict(profile=PROFILE, files=sorted(files+[result_binding], key=lambda x:x['path']))
            manifest_bytes = len((json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode())
            final_size = result['raw_bytes_before_result_manifest'] + len(encoded) + manifest_bytes
            raw_passed = final_size <= RAW_CAP
            if final_size == resources['final_raw_bytes'] and raw_passed == resources['raw_passed']:
                break
            resources.update(final_raw_bytes=final_size, raw_passed=raw_passed)
        else:
            raise ResourceLimit('final resource accounting did not settle')
        recorder.json('result.json', result)
        recorder.json('manifest.json', manifest)
        require(recorder.size() == resources['final_raw_bytes'], 'final serialization differs from exact resource accounting')
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in PINS:
        p.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True); p.add_argument('--vectors', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True); p.add_argument('--preflight-only', action='store_true')
    return p


def main():
    args = parser().parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (MAX_VM, MAX_VM))
    def stop(signum, frame):
        raise TimeoutError(f'bounded runner signal {signum}')
    signal.signal(signal.SIGALRM, stop); signal.signal(signal.SIGTERM, stop)
    signal.alarm(MAX_SECONDS)
    try:
        result = run(args)
    finally:
        signal.alarm(0)
    print(json.dumps(dict(status=result['status'], outcome=result['outcome'], model_calls=result['model_calls'])))
    return 0 if result['status'] in ('completed', 'preflight_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
