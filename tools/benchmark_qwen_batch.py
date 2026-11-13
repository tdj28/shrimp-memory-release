#!/usr/bin/env python3
"""Bounded two-state one-step batch benchmark; never a trajectory experiment."""
import argparse
import copy
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
import check_p6_writes as writes
import validate_qwen_map as reference

PACKET_SHA256 = '8fc36ac62413a099fd6dd562f1165dfad92528de3e7011a3a072ec4f93e7b8b1'
INDICES = (0, 2)
MAX_CALLS, MAX_SECONDS, MAX_RSS_GIB = 28, 180, 8
MAX_RAW_BYTES = 255 * 1024**2
GAP_TOLERANCE = 1e-12
CONTROLS = dict(writes.HOLD)
DIM, VOCAB, LAYERS = 896, 151936, 24
CACHE_SHAPE = (1, 2, len(reference.PREFIX_IDS), 64)


def schedule():
    rows = []
    def add(label, indices, cached=True, phase='diagnostic', repeat=None):
        rows.append({'call': len(rows)+2, 'label': label, 'packet_indices': list(indices),
                     'batch_size': len(indices), 'cached': cached, 'phase': phase, 'repeat': repeat,
                     'status': 'unstarted'})
    for index in INDICES:
        for cached in (True, False):
            add(f'diagnostic_scalar_{index}_{"cached" if cached else "full"}', (index,), cached)
    for cached in (True, False):
        add('diagnostic_batch_' + ('cached' if cached else 'full'), INDICES, cached)
    for index in INDICES:
        add(f'warmup_scalar_{index}', (index,), phase='warmup')
    add('warmup_batch', INDICES, phase='warmup')
    for repeat in range(6):
        groups = [('scalar_0', (0,)), ('scalar_2', (2,)), ('batch', INDICES)]
        if repeat % 2:
            groups = groups[2:] + groups[:2]
        for name, indices in groups:
            add(f'timed_{repeat:02d}_{name}', indices, phase='timed', repeat=repeat)
    return rows


def file_record(path):
    path = Path(path)
    return {'path': path.name, 'bytes': path.stat().st_size, 'sha256': reference.sha256(path)}


def save_arrays(path, arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    return file_record(path)


@contextmanager
def preserve_call_files():
    """Delay process signals until a returned call's immutable files are flushed."""
    watched = {signal.SIGALRM, signal.SIGTERM, signal.SIGINT}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def valid_cache(arrays, batch_size=1, length=None):
    length = len(reference.PREFIX_IDS) if length is None else length
    names = {f'layer_{i:02d}_{kind}' for i in range(LAYERS) for kind in ('key', 'value')}
    if set(arrays) != names | {'length'}:
        return False
    if arrays['length'].shape != () or arrays['length'].dtype != np.int64 or int(arrays['length']) != length:
        return False
    shape = (batch_size, CACHE_SHAPE[1], length, CACHE_SHAPE[3])
    return all(arrays[key].shape == shape and arrays[key].dtype == np.float64
               and np.isfinite(arrays[key]).all() for key in names)


def same_arrays(left, right):
    return set(left) == set(right) and all(np.array_equal(left[key], right[key]) for key in left)


def repeat_prefix_cache(cache, batch_size=2, repeat_tensor=None):
    """Copy layer containers and allocate repeated key/value tensors, never aliasing the source."""
    if batch_size not in (2, 4):
        raise ValueError('only fixed batch sizes two and four are supported')
    repeat_tensor = repeat_tensor or (lambda tensor: tensor.repeat(batch_size, 1, 1, 1))
    result = copy.copy(cache)
    result.layers = [copy.copy(layer) for layer in cache.layers]
    before = {}
    for i, layer in enumerate(result.layers):
        for attr, kind in (('keys', 'key'), ('values', 'value')):
            original = getattr(cache.layers[i], attr)
            copied = repeat_tensor(original)
            setattr(layer, attr, copied)
            # Keep tensor references, not an expensive copied audit snapshot, inside the timer.
            before[f'layer_{i:02d}_{kind}'] = copied
    return result, before


def compare(left, right):
    shape_ok = left[0].shape == right[0].shape and left[1].shape == right[1].shape
    finite = shape_ok and all(np.isfinite(a).all() for a in (*left, *right))
    gap = float(np.max(np.abs(left[0]-right[0]))) if finite else None
    pgap = float(np.max(np.abs(left[1]-right[1]))) if finite else None
    tokens = bool(np.array_equal(left[1].argmax(-1), right[1].argmax(-1))) if finite else False
    return {'passed': bool(finite and max(gap, pgap) <= GAP_TOLERANCE and tokens),
            'maximum_coordinate_gap': gap, 'maximum_probability_gap': pgap,
            'outputs_exact': bool(shape_ok and np.array_equal(left[0], right[0])),
            'probabilities_exact': bool(shape_ok and np.array_equal(left[1], right[1])),
            'top_tokens_equal': tokens, 'tolerance': GAP_TOLERANCE}


class QwenBackend:
    def __init__(self, args, budget, recorder):
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_grad_enabled(False)
        snapshot = args.cache_dir / 'models--Qwen--Qwen2.5-0.5B-Instruct' / 'snapshots' / reference.REVISION
        self.bindings = {'weights': reference.require_hash(snapshot/'model.safetensors', reference.WEIGHTS_SHA256),
                         'config': reference.require_hash(snapshot/'config.json', writes.hold_test.CONFIG_SHA256),
                         'vectors': reference.require_hash(args.vectors, reference.VECTOR_SHA256)}
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        ids = tokenizer.apply_chat_template(reference.MESSAGES, add_generation_prompt=True)
        if list(ids['input_ids'] if hasattr(ids, 'keys') else ids) != reference.PREFIX_IDS:
            raise ValueError('prefix token IDs differ')
        model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
            dtype=torch.float64, attn_implementation='sdpa').cpu().eval().requires_grad_(False)
        cfg = model.config
        if (cfg.model_type, cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size) != ('qwen2', DIM, LAYERS, VOCAB):
            raise ValueError('unexpected model architecture')
        if any(t.dtype != torch.float64 or t.device.type != 'cpu' for t in model.parameters()):
            raise ValueError('parameters must be CPU float64')
        with np.load(args.vectors, allow_pickle=False) as archive:
            vector = archive['philosophy']
        if vector.shape != (DIM,) or vector.dtype != np.float64 or not np.isfinite(vector).all():
            raise ValueError('invalid pinned steering vector')
        self.observations = {}
        self.patch = reference.patch_native_float64(model, self.observations)
        self.loop = reference.FeedbackMap(model, torch.tensor(vector, dtype=torch.float64),
            CONTROLS['alpha'], CONTROLS['log10_temperature'], budget, recorder)
        self.loop.mode = 'native64'

    def prepare(self):
        self.loop.prepare()

    def cache_arrays(self):
        cache = self.loop.cache
        arrays = {'length': np.asarray(cache.get_seq_length(), dtype=np.int64)}
        for i, layer in enumerate(cache.layers):
            for attr, kind in (('keys', 'key'), ('values', 'value')):
                arrays[f'layer_{i:02d}_{kind}'] = getattr(layer, attr).detach().numpy().copy()
        return arrays

    def forward(self, array, cached):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        torch, loop = self.torch, self.loop
        if len(array) not in (1, 2, 4):
            raise ValueError('only scalar, batch-two and batch-four inputs are supported')
        x = torch.tensor(array, dtype=torch.float64)
        before = x.detach().numpy().copy()
        used = {}
        # Require fresh observations from this call, rather than stale prefix/scalar observations.
        self.observations.clear()
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            started = time.perf_counter()
            if len(x) == 1:
                y, probabilities = loop.forward(x[0], cached=cached)
                y, probabilities = y.unsqueeze(0), probabilities.unsqueeze(0)
            else:
                if cached:
                    cache, used = repeat_prefix_cache(loop.cache, len(x))
                    result = loop.model(inputs_embeds=x[:, None, :], past_key_values=cache,
                                        use_cache=True, logits_to_keep=1)
                else:
                    prefix = loop.embedding[loop.prefix].expand(len(x), -1, -1)
                    result = loop.model(inputs_embeds=torch.cat((prefix, x[:, None, :]), dim=1),
                                        use_cache=False, logits_to_keep=1)
                logits = result.logits[:, -1]
                probabilities = torch.softmax(logits / loop.temperature, dim=-1)
                y = probabilities @ loop.embedding
            seconds = time.perf_counter()-started
        extras = {'input_after': x.detach().numpy().copy(),
                  'observed_output_dtype': str(y.dtype), 'observed_probability_dtype': str(probabilities.dtype),
                  'input_exact': np.array_equal(before, x.detach().numpy())}
        if used:
            extras['used_cache_before'] = {key: tensor.detach().numpy().copy() for key, tensor in used.items()}
            extras['used_cache_length_after'] = np.asarray(cache.get_seq_length(), dtype=np.int64)
        return y.detach().numpy().copy(), probabilities.detach().numpy().copy(), seconds, extras


def evaluate(backend, states, row, recorder, budget, prefix):
    arrays = {'input': states[np.asarray(row['packet_indices'])].copy()}
    before = backend.cache_arrays()
    arrays.update({'cache_before_' + key: value for key, value in before.items()})
    budget.claim(row['label'])
    if budget.calls != row['call']:
        raise ValueError('call schedule drifted')
    row['status'] = 'started'
    result = None
    try:
        y, p, seconds, extra = backend.forward(arrays['input'].copy(), row['cached'])
        result = (y, p)
        arrays.update(output=y, probabilities=p, input_after=extra['input_after'])
        if 'used_cache_before' in extra:
            arrays.update({'used_cache_before_' + key: value for key, value in extra['used_cache_before'].items()})
            arrays['used_cache_length_after'] = extra['used_cache_length_after']
        # Returned values are retained even if subsequent cache inspection raises.
        row.update(seconds=seconds, dtype_observations=copy.deepcopy(backend.observations),
                   output_dtype=extra['observed_output_dtype'], probability_dtype=extra['observed_probability_dtype'])
        after = backend.cache_arrays()
        arrays.update({'cache_after_' + key: value for key, value in after.items()})
        shape = (row['batch_size'], DIM)
        output_valid = (y.shape == shape and p.shape == (row['batch_size'], VOCAB)
                        and y.dtype == p.dtype == np.float64 and np.isfinite(y).all() and np.isfinite(p).all()
                        and np.all(p >= 0) and np.allclose(p.sum(-1), 1., rtol=0, atol=1e-12))
        used_valid = True
        if row['batch_size'] > 1 and row['cached']:
            expected = {key: np.repeat(value, row['batch_size'], axis=0) for key, value in prefix.items() if key != 'length'}
            used_valid = (same_arrays(extra.get('used_cache_before', {}), expected)
                          and int(extra['used_cache_length_after']) == len(reference.PREFIX_IDS)+1)
        row.update(cache_unchanged=valid_cache(before) and valid_cache(after) and same_arrays(before, prefix) and same_arrays(after, prefix),
                   input_unchanged=bool(extra['input_exact'] and np.array_equal(arrays['input'], arrays['input_after'])),
                   expanded_cache_valid=bool(used_valid), outputs_valid=bool(output_valid),
                   top_tokens=p.argmax(-1).tolist() if output_valid else None)
        observed = row['dtype_observations']
        precision = all(observed.get(group, {}).get(key) == 'torch.float64' for group, keys in {
            'norm': ('input', 'reduction', 'weight', 'output'),
            'rotary': ('input', 'inverse_frequency', 'angles', 'cos', 'sin')}.items() for key in keys)
        row['precision_observations_passed'] = bool(precision and row['output_dtype'] == row['probability_dtype'] == 'torch.float64')
        if not all(row[key] for key in ('cache_unchanged', 'input_unchanged', 'expanded_cache_valid', 'outputs_valid', 'precision_observations_passed')):
            raise ValueError('input/cache/output/observed precision gate failed')
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('invalid timed duration')
        if reference.rss_bytes() >= budget.max_rss or time.monotonic()-recorder.started >= MAX_SECONDS:
            raise TimeoutError('resource limit after returned call')
        row['status'] = 'returned'
    except BaseException as error:
        row.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        name = f'call-{row["call"]:03d}-{row["label"]}'
        with preserve_call_files():
            row['artifact'] = save_arrays(recorder.directory/(name+'.npz'), arrays)
            recorder.json(name+'.json', row)
            recorder.event('model_call_completed', call=row['call'], label=row['label'], status=row['status'], artifact=row['artifact'])
    return result


def comparisons(outputs, calls):
    get = lambda label: outputs[label]
    checks = {}
    for index in INDICES:
        checks[f'scalar_{index}_cached_full'] = compare(get(f'diagnostic_scalar_{index}_cached'), get(f'diagnostic_scalar_{index}_full'))
    scalar = tuple(np.concatenate([get(f'diagnostic_scalar_{index}_cached')[part] for index in INDICES]) for part in (0, 1))
    checks['batch_cached_full'] = compare(get('diagnostic_batch_cached'), get('diagnostic_batch_full'))
    checks['scalar_batch_cached'] = compare(scalar, get('diagnostic_batch_cached'))
    for label, output in outputs.items():
        indices = next(row['packet_indices'] for row in calls if row['label'] == label)
        baseline = tuple(np.concatenate([get(f'diagnostic_scalar_{index}_cached')[part] for index in indices]) for part in (0, 1))
        checks[label + '_scalar_baseline'] = compare(baseline, output)
    return checks


def run(args, backend_factory=QwenBackend, finalization_guard=None):
    recorder = reference.Recorder(args.output_dir)
    budget = reference.Budget(recorder, MAX_CALLS, MAX_SECONDS, MAX_RSS_GIB)
    calls = schedule()
    report = {'status': 'failed', 'evidence_status': 'exploratory numerical one-step benchmark',
              'calls': calls, 'comparisons': {}, 'pairs': [], 'timing_eligible': False, 'timing_summary': None,
              'timing_scope': 'Two scalar cached forwards summed versus one cached batch-of-two forward. Input tensor preparation, cache audit snapshots and file I/O excluded; normal cache copying, model, softmax and embedding feedback included.',
              'scope': 'Two previously observed states, one step each, six paired timings on one host. No trajectory, general precision, reliable channel, or workload-wide acceleration claim.'}
    backend = None
    outputs = {}
    prefix_status = 'unstarted'
    try:
        if args.packet_sha256 != PACKET_SHA256:
            raise ValueError('this design requires the pinned exact-readback packet')
        packet_binding = reference.require_hash(args.packet, PACKET_SHA256)
        with np.load(args.packet, allow_pickle=False) as archive:
            if set(archive.files) != {'states'}:
                raise ValueError('packet must contain only states')
            states = archive['states']
        if states.shape != (8, DIM) or states.dtype != np.float64 or not np.isfinite(states).all() or np.array_equal(states[0], states[2]):
            raise ValueError('require eight finite float64 states and distinct selected states')
        root = Path(__file__).resolve().parents[1]
        source = {'source_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                  'source_hashes': {Path(path).name: reference.sha256(path) for path in (__file__, reference.__file__, writes.__file__)}}
        recorder.json('inputs.json', {'packet': packet_binding, 'packet_indices': list(INDICES), 'prefix_ids': reference.PREFIX_IDS, **source})
        started = time.monotonic()
        backend = backend_factory(args, budget, recorder)
        report['backend_initialization_seconds'] = time.monotonic()-started
        recorder.json('metadata.json', {'utc': reference.utc_now(), **source, **backend.bindings,
            'model': reference.MODEL, 'revision': reference.REVISION, 'packet': packet_binding,
            'packet_indices': list(INDICES), 'prefix_ids': reference.PREFIX_IDS, 'controls': CONTROLS,
            'threads': 1, 'max_calls': MAX_CALLS, 'max_seconds': MAX_SECONDS, 'max_rss_gib': MAX_RSS_GIB,
            'max_raw_bytes': MAX_RAW_BYTES, 'gap_tolerance': GAP_TOLERANCE,
            'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}})
        try:
            prefix_status = 'started'
            backend.prepare()
            prefix_status = 'returned'
        except BaseException:
            prefix_status = 'failed'
            raise
        finally:
            if budget.calls:
                recorder.event('model_call_completed', call=1, label='native64_prefix', status=prefix_status)
        prefix = backend.cache_arrays()
        report['prefix_cache'] = save_arrays(recorder.directory/'prefix-cache.npz', prefix)
        if not valid_cache(prefix):
            raise ValueError('invalid scalar prefix cache')
        for row in calls:
            outputs[row['label']] = evaluate(backend, states, row, recorder, budget, prefix)
            if sum(path.stat().st_size for path in recorder.directory.iterdir() if path.is_file()) > MAX_RAW_BYTES:
                raise MemoryError('raw artifact budget exceeded')
        report['comparisons'] = comparisons(outputs, calls)
        for repeat in range(6):
            selected = [row for row in calls if row['repeat'] == repeat]
            scalars = [row for row in selected if row['batch_size'] == 1]
            batch = next(row for row in selected if row['batch_size'] == 2)
            scalar_output = tuple(np.concatenate([outputs[f'timed_{repeat:02d}_scalar_{index}'][part] for index in INDICES]) for part in (0, 1))
            pair = {'repeat': repeat, 'order': [row['label'] for row in selected],
                    'scalar_pair_seconds': sum(row['seconds'] for row in scalars), 'batch_seconds': batch['seconds'],
                    'comparison': compare(scalar_output, outputs[batch['label']]),
                    'call_numbers': [row['call'] for row in selected]}
            report['pairs'].append(pair)
            recorder.json(f'pair-{repeat:02d}.json', pair)
        report['numerical_gates_passed'] = all(check['passed'] for check in report['comparisons'].values()) and all(row['comparison']['passed'] for row in report['pairs'])
        report['status'] = 'completed' if report['numerical_gates_passed'] else 'numerical_gate_failed'
    except BaseException as error:
        report.update(status='incomplete' if isinstance(error, (TimeoutError, InterruptedError, MemoryError)) else 'failed',
                      error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    finally:
        if finalization_guard is not None:
            finalization_guard()
        elapsed, rss = time.monotonic()-recorder.started, reference.rss_bytes()
        raw_bytes = sum(path.stat().st_size for path in recorder.directory.iterdir() if path.is_file())
        report.update(model_calls=budget.calls, prefix_status=prefix_status, elapsed_seconds=elapsed,
                      peak_rss_bytes=rss, raw_bytes_before_results=raw_bytes,
                      dtype_observations={} if backend is None else backend.observations)
        report['resource_limits_passed'] = budget.calls <= MAX_CALLS and elapsed <= MAX_SECONDS and rss < budget.max_rss and raw_bytes < MAX_RAW_BYTES
        if report['status'] in ('completed', 'numerical_gate_failed') and (budget.calls != MAX_CALLS or not report['resource_limits_passed']):
            report.update(status='incomplete', error='final count/resource gate failed')
        report['timing_eligible'] = report['status'] == 'completed'
        if report['timing_eligible']:
            ratios = [row['scalar_pair_seconds']/row['batch_seconds'] for row in report['pairs']]
            report['timing_summary'] = {'scalar_pair_over_batch_ratios': ratios, 'median_ratio': float(np.median(ratios)),
                'minimum_ratio': min(ratios), 'maximum_ratio': max(ratios),
                'ratio_of_total_seconds': sum(row['scalar_pair_seconds'] for row in report['pairs'])/sum(row['batch_seconds'] for row in report['pairs'])}
        recorder.json('results.json', report)
        recorder.event('finished', status=report['status'])
        recorder.stream.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache-dir', 'vectors', 'packet', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--packet-sha256', required=True)
    args = parser.parse_args(argv)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    def stop(sig, frame):
        signal.setitimer(signal.ITIMER_REAL, 0)
        for handled in handlers:
            signal.signal(handled, signal.SIG_IGN)
        raise TimeoutError('internal deadline reached') if sig == signal.SIGALRM else InterruptedError(f'signal {sig}')
    def guard():
        signal.setitimer(signal.ITIMER_REAL, 0)
        for handled in handlers:
            signal.signal(handled, signal.SIG_IGN)
    try:
        for sig in handlers:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
        result = run(args, finalization_guard=guard)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, old in handlers.items():
            signal.signal(sig, old)
    print(json.dumps({'status': result['status'], 'model_calls': result['model_calls']}))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
