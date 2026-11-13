#!/usr/bin/env python3
"""Fixed literal nominal-pulse handoffs at H; all returned calls are retained.

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
PROFILE = 'public-nominal-handoffs'
DIM, VOCAB, STEPS, MAX_CALLS = 896, 151936, 240, 490
SOURCE_INDICES = ((0, 60), (12, 36))
HOLD = dict(alpha=0.07293697809562831, log10_temperature=-0.15395151820007266)
ENDPOINT = dict(alpha=0.07293697809562831, log10_temperature=-0.1535)
GROUP_NAMES = ('source-and-replay', 'early-handoffs')
ROW_NAMES = ('source-h1', 'replay-x60', 'early-x12', 'early-x36')
MAX_SECONDS, EXTERNAL_SECONDS, GRACE, OUTER_SECONDS, RETURN_SECONDS = 2400, 2460, 30, 2640, 4800
MAX_RSS, MAX_VM = 8 * 1024**3, 16 * 1024**3
RAW_CAP, FINAL_RESERVE = 2304 * 1024**2, 32 * 1024**2
PACKAGE_CAP, ARCHIVE_CAP, UPLOAD_CAP, TRANSFER_CAP = 2560*1024**2, 3072*1024**2, 8*1024**2, 8*1024**3
GAP_TOLERANCE, CLOSURE_TOLERANCE, DIVISOR_TOLERANCE, MATCH_TOLERANCE = 1e-10, 1e-8, 1e-4, 1e-6
PINS = {
    'trajectory': (833183, 'c8e4d37d64765734d3b23b3fb3b38e22337d8af1a003cc7bc235cccc3ea51f9a'),
    'fixed_inputs': (96827, '0ad3846d11ff65d7616b5fc6d125f6b30c16ea56ed490da3bfcd27f510586b44'),
    'historical_metadata': (7518, 'efa7bcc502a6ccbd27789d1deee714d58836e870e14a3f0544f9473e2536807a'),
    'historical_plan': (3655, '2762d850ffedaca20c29110f494c3209100c6287cbdfd22fd839a60f0e92c409'),
    'historical_audit': (77854, '257f03396a80c139cab31049fd759d12f610a0566932ccb4e0ebd33d8617fefe'),
    'historical_manifest': (54152, '4536fb39b68e9ae9e1c9e3e251b36b481caefa8889fcf32f297b3097aec79be3'),
}
ORIGINAL_NAMES = dict(trajectory='trial-01-endpoint-final-120.npz', fixed_inputs='fixed-trial-inputs.npz',
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


def derive_handoffs(references, states):
    """Select only the four pinned literal slices; preserve their actual phases."""
    finite(references, (2, 6, DIM), 'reference cycles')
    finite(states, (121, DIM), 'historical trajectory')
    h1 = references[1, 0].copy()
    require(same_bytes(states[0], h1), 'historical start is not literal H1 phase zero')
    x12, x36, x60 = (states[index].copy() for index in (12, 36, 60))
    initial = np.stack((h1, x60, x12, x36)).reshape(2, 2, DIM)
    return dict(references=references.copy(), h1=h1, x12=x12, x36=x36, x60=x60,
        initial_states=initial, historical_state_indices=np.asarray(SOURCE_INDICES, dtype=np.int64))


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
    for key in ('trajectory', 'fixed_inputs', 'historical_metadata', 'historical_plan'):
        require_historical_binding(audit_bindings, ORIGINAL_NAMES[key], PINS[key])
        require_historical_binding(manifest_bindings, ORIGINAL_NAMES[key], PINS[key])
    require(metadata['source_commit'] == 'f0c757ed2c7142bd2dd5cb78a796f69f7f49f4c4', 'historical source differs')
    require(metadata['model'] == instrument.MODEL and metadata['revision'] == instrument.REVISION
            and metadata['prefix_ids'] == instrument.PREFIX_IDS, 'historical map identity differs')
    require(metadata['hold'] == plan['hold'] == HOLD and metadata['candidate_endpoints']['ad'] == ENDPOINT,
            'historical controls differ')
    require(all(metadata[k]['sha256'] == MODEL_PINS[k] for k in ('weights', 'config', 'vectors')),
            'historical model/vector pins differ')
    trial = plan['trials'][1]
    require((trial['trial_id'], trial['source_index'], trial['source_label'], trial['phase'], trial['desired_label'])
            == (1, 1, 'td', 0, 'ad'), 'historical trial semantics differ')
    require([(x['name'], x['steps'], x['controls']) for x in trial['stages']]
            == [('endpoint', 120, ENDPOINT), ('hold', 240, HOLD)], 'historical stage plan differs')
    require((plan['inputs']['bytes'], plan['inputs']['sha256']) == PINS['fixed_inputs'], 'historical fixed input binding')
    with np.load(args.trajectory, allow_pickle=False) as z:
        states = z['states'].copy()
        require(z['first_state_index'].dtype == np.int64 and z['first_state_index'].shape == ()
                and int(z['first_state_index']) == 0, 'historical state indices differ')
        for key, value in ENDPOINT.items():
            finite(z[key], (), 'historical ' + key)
            require(same_bytes(z[key], np.asarray(value)), 'historical binary64 control differs')
        require(z['token_ids'].dtype == np.int64 and z['token_ids'].shape == (120,)
                and np.all((0 <= z['token_ids']) & (z['token_ids'] < VOCAB)), 'historical tokens invalid')
        finite(z['step_seconds'], (120,), 'historical times')
        require(np.all(z['step_seconds'] > 0), 'historical nonpositive time')
    with np.load(args.fixed_inputs, allow_pickle=False) as z:
        references, initials = z['references'].copy(), z['initial_states'].copy()
    finite(initials, (4, DIM), 'historical initial states')
    finite(references, (2, 6, DIM), 'historical references')
    require(same_bytes(initials, references[[0, 1, 0, 1], 0]), 'historical fixed phase starts differ')
    payload = derive_handoffs(references, states)
    return payload, dict(bindings=bindings, historical_source_commit=metadata['source_commit'],
        historical_trial_id=1, historical_endpoint_controls=ENDPOINT,
        historical_states_array_sha256=array_hash(states), references_array_sha256=array_hash(references),
        scope='Literal historical states 0/60/12/36; original endpoint preparation is prior cost, not rerun.')


def qualify_model(args):
    snapshot = args.cache_dir / 'models--Qwen--Qwen2.5-0.5B-Instruct' / 'snapshots' / instrument.REVISION
    paths = dict(weights=snapshot/'model.safetensors', config=snapshot/'config.json',
                 tokenizer=snapshot/'tokenizer.json', tokenizer_config=snapshot/'tokenizer_config.json', vectors=args.vectors)
    return {k: instrument.require_hash(path, MODEL_PINS[k]) for k, path in paths.items()}


def source_metadata(args):
    names = ['tools/run_public_nominal_handoffs.py', 'tests/test_run_public_nominal_handoffs.py',
        'tools/audit_public_nominal_handoffs.py', 'tests/test_audit_public_nominal_handoffs.py',
        'tools/run_public_state_intervention.py', 'tools/benchmark_qwen_batch.py', 'tools/validate_qwen_map.py', 'tools/check_p6_writes.py',
        'tools/check_p6_hold.py', 'tools/check_p6_recovery.py', 'docs/public-nominal-handoffs.md', 'pyproject.toml', 'uv.lock']
    return dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        files=[{**binding(ROOT/n), 'path': n} for n in names],
        working_tree_status=subprocess.check_output(['git', 'status', '--porcelain', '--', *names], cwd=ROOT, text=True).splitlines(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})


def call_plan():
    slots = []
    for group in range(2):
        items = [('prefix', 0, None, 1, 'prefix'), ('comparison', 0, True, 2, 'compare-cached'),
                 ('comparison', 0, False, 2, 'compare-full'), ('comparison', 0, False, 1, 'compare-scalar-0'),
                 ('comparison', 0, False, 1, 'compare-scalar-1')]
        items += [('update', step, True, 2, f'update-{step:03d}') for step in range(1, STEPS+1)]
        for category, step, cached, width, label in items:
            slots.append(dict(call=len(slots)+1, group_id=group, category=category, step=step,
                cached=cached, batch_size=width, label=f'g{group:02d}-{label}', status='not_started',
                controls=HOLD.copy(), controls_hex={k: v.hex() for k, v in HOLD.items()}))
    require(len(slots) == MAX_CALLS, 'fixed schedule arithmetic differs')
    return slots


def derive_design():
    return dict(profile=PROFILE, schema_version=1, evidence='adapted exploratory', group_names=list(GROUP_NAMES),
        row_names=list(ROW_NAMES), controls=HOLD, controls_hex={k: v.hex() for k, v in HOLD.items()},
        source_indices=[list(row) for row in SOURCE_INDICES], literal_phase_carry=True,
        schedule=dict(groups=2, calls_per_group=245, prefix_calls=2, comparison_calls=8,
            batch_updates=480, lane_updates=960, total_calls=MAX_CALLS, steps_per_group=STEPS, batch_size=2),
        readout=dict(retained_state_indices=[228, 240], closure_pairs=[[i, i-6] for i in range(235, 241)],
            closure_tolerance=CLOSURE_TOLERANCE, proper_divisors=[1, 2, 3], divisor_tolerance=DIVISOR_TOLERANCE,
            reference_match_tolerance=MATCH_TOLERANCE, all_cyclic_shifts=list(range(6))),
        comparison_tolerance=GAP_TOLERANCE,
        limits=dict(model_calls=MAX_CALLS, internal_seconds=MAX_SECONDS, external_seconds=EXTERNAL_SECONDS,
            termination_grace_seconds=GRACE, outer_seconds=OUTER_SECONDS, return_seconds=RETURN_SECONDS,
            threads=1, peak_rss_bytes=MAX_RSS, external_vm_bytes=MAX_VM, raw_bytes=RAW_CAP,
            working_raw_bytes=RAW_CAP-FINAL_RESERVE, finalization_reserve_bytes=FINAL_RESERVE,
            package_bytes=PACKAGE_CAP, archive_bytes=ARCHIVE_CAP, upload_bytes=UPLOAD_CAP,
            transfer_reservation_bytes=TRANSFER_CAP),
        prefix_output='Soft embedding from the actual prefix logits at H temperature; recorded but never carried.',
        cache_storage='One original and one expanded cache per group; canonical hashes every call; anomalous caches retained.',
        scope='Two fixed historical handoffs and fresh controls. No first-acquisition, monotonicity, basin-volume, branch-loss, text-reachability or causal-hysteresis claim.')


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


def panel_outcome(bits):
    require(len(bits) == 4 and all(x is None or type(x) is int and x in (0, 1) for x in bits), 'four neutral bits required')
    if not (bits[0] == 1 and bits[1] == 0):
        return 'control_unqualified'
    if any(x is None for x in bits[2:]):
        return 'early_handoff_unresolved'
    if bits[2:] == [0, 0]:
        return 'both_early_handoffs_h0'
    if bits[2:] == [1, 1]:
        return 'both_early_handoffs_h1'
    return 'mixed_early_handoffs'


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
        self.active_group = None; self.files = {}; self.owned = {i: [] for i in range(2)}; self.finalizing = False
        self.groups = [dict(profile=PROFILE, group_id=i, name=name, closed=False, status='not_started',
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
        summary.update(profile=PROFILE, group_id=group, name=GROUP_NAMES[group],
                       model_calls=len(calls), calls=copy.deepcopy(calls))
        self.json(f'group-{group:02d}-summary.json', summary)
        marker = dict(profile=PROFILE, group_id=group, name=GROUP_NAMES[group], status=summary['status'],
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


def raw_controls():
    return {k: np.asarray(v, dtype=np.float64) for k, v in HOLD.items()}


def evaluate(backend, x, recorder, budget, prefix=None, expanded=None, on_return=None):
    """Retain raw results before any instrument/resource gate; failed attempts count."""
    slot = budget.slots[budget.calls]
    is_prefix = slot['category'] == 'prefix'
    arrays = dict(input=x.copy(), **raw_controls())
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


def execute_group(group, backend, payload, recorder, budget):
    recorder.active_group = group; budget.group = group
    states = [payload['initial_states'][group].copy()]
    summary = dict(status='started', initial_state_sha256=array_hash(states[0]), comparison=None, readouts=[],
                   completed_updates=0, returned_updates=0, trajectory=None, tail=None)
    try:
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
        for _ in range(STEPS):
            evaluate(backend, states[-1], recorder, budget, prefix, expanded, on_return=states.append)
            summary['completed_updates'] += 1
        tail = np.stack(states[-13:])
        summary['tail'] = recorder.save(f'group-{group:02d}-tail.npz', states=tail,
            state_indices=np.arange(228, 241, dtype=np.int64), **raw_controls())
        summary['readouts'] = [readout(tail[:, lane], payload['references']) for lane in range(2)]
        summary['status'] = 'completed'
    except BaseException as error:
        summary.update(status=failure_status(error), error=f'{type(error).__name__}: {error}')
        raise
    finally:
        # A finalization reserve retains all returned states even after a working-limit stop.
        was_final = recorder.finalizing; recorder.finalizing = True
        try:
            summary['returned_updates'] = len(states)-1
            summary['trajectory'] = recorder.save(f'group-{group:02d}-states.npz', states=np.stack(states),
                state_indices=np.arange(len(states), dtype=np.int64), **raw_controls())
            calls = [row for row in budget.slots[:budget.calls] if row['group_id'] == group]
            recorder.close_group(group, summary, calls)
        finally:
            recorder.finalizing = was_final
    return summary


def run(args, *, backend_factory=QwenBackend, input_qualifier=qualify_inputs,
        model_qualifier=qualify_model, source_factory=source_metadata):
    """One execution; injection never changes the production CLI or fixed schedule."""
    recorder = Recorder(args.output_dir); slots = call_plan(); budget = Budget(recorder, slots)
    result = dict(profile=PROFILE, status='started', outcome='not_qualified', model_loaded=False,
        design=derive_design(), source=None, model_calls=0, groups=recorder.groups, calls=slots,
        rows=[dict(row_id=i, group_id=i//2, lane=i%2, name=name, status='not_run', readout=None) for i, name in enumerate(ROW_NAMES)])
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
            for group in range(2):
                summary = execute_group(group, backend, payload, recorder, budget)
                for lane in range(2):
                    result['rows'][group*2+lane].update(status='completed', readout=summary['readouts'][lane])
            require(budget.calls == MAX_CALLS, 'completed fixed plan has wrong invocation count')
            result.update(status='completed', outcome=panel_outcome([r['readout']['decoded_bit'] for r in result['rows']]))
    except BaseException as error:
        result.update(status=failure_status(error), outcome='not_qualified', error=f'{type(error).__name__}: {error}')
        recorder.finalizing = True
        recorder.json('failure.json', dict(status=result['status'], error=result['error'], traceback=traceback.format_exc()))
    finally:
        recorder.finalizing = True; recorder.active_group = None
        for slot in slots[budget.calls:]:
            slot.update(status='not_run', not_run_reason=result['status'])
        for group in range(2):
            if not recorder.groups[group]['closed']:
                calls = [row for row in slots[:budget.calls] if row['group_id'] == group]
                recorder.close_group(group, dict(status=result['status'] if calls else 'not_run',
                    not_run_reason=result.get('error', result['outcome']), completed_updates=0, returned_updates=0,
                    comparison=None, readouts=[], trajectory=None, tail=None), calls)
        result.update(model_calls=budget.calls, groups=recorder.groups,
            elapsed_seconds=time.monotonic()-recorder.started, peak_rss_bytes=instrument.rss_bytes())
        for row in result['rows']:
            if row['status'] != 'completed':
                row['status'] = recorder.groups[row['group_id']]['status']
                row['not_run_reason'] = result.get('error', result['outcome'])
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
