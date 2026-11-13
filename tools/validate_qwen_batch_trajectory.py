#!/usr/bin/env python3
"""Carry two fixed Qwen states for 600 batch updates; retain all outcomes."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import time
import traceback

import benchmark_qwen_batch as batch
import numpy as np

reference = batch.reference
PACKET_SHA256 = 'db143ff5bdc264c59303cc6a34d3695ac3ee0d09b74ccf9dc3a71fccafda396c'
INDICES = (0, 1)
STEPS, BLOCK = 600, 12
MAX_CALLS, MAX_SECONDS, MAX_RSS_GIB = 605, 900, 8
MAX_RAW_BYTES = 64 * 1024**2
DIAGNOSTIC_TOLERANCE, TRAJECTORY_TOLERANCE = 1e-12, 1e-10
CONTROLS = dict(batch.CONTROLS)


def array_hash(array):
    array = np.asarray(array)
    header = str(array.dtype).encode() + b'\0' + json.dumps(list(array.shape), separators=(',', ':')).encode() + b'\0'
    return hashlib.sha256(header + array.tobytes(order='C')).hexdigest()


def cache_hash(arrays):
    values = {key: array_hash(value) for key, value in arrays.items()}
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def schedule():
    rows = []
    def add(label, phase, state_index, cached=True):
        rows.append({'call': len(rows)+2, 'label': label, 'phase': phase, 'state_index': state_index,
                     'cached': cached, 'status': 'unstarted'})
    for cached in (True, False):
        add('diagnostic_initial_' + ('cached' if cached else 'full'), 'initial', 0, cached)
    for step in range(1, STEPS+1):
        add(f'batch_step_{step:03d}', 'update', step)
    for cached in (True, False):
        add('diagnostic_final_' + ('cached' if cached else 'full'), 'final', STEPS, cached)
    return rows


class TrajectoryArchive:
    def __init__(self, recorder, initial):
        self.recorder = recorder
        self.states, self.tokens, self.seconds = [initial.copy()], [], []
        self.blocks, self.final, self.saved = [], None, 0

    def append(self, y, p, seconds):
        # A malformed returned array is retained in the failed-call NPZ instead.
        if y.shape != (2, batch.DIM):
            return
        self.states.append(y.copy())
        self.tokens.append(p.argmax(-1).astype(np.int64) if p.shape == (2, batch.VOCAB) and np.isfinite(p).all()
                           else np.full(2, -1, dtype=np.int64))
        self.seconds.append(seconds)

    def save(self, final=False):
        end = len(self.states)-1
        first = 0 if final else self.saved+1
        if not final and (end == self.saved or end % BLOCK):
            return
        name = f'trajectory-{"final" if final else "block"}-{end:03d}.npz'
        token_start = 0 if final else self.saved
        arrays = {'states': np.stack(self.states[first:]),
                  'token_ids': np.asarray(self.tokens[token_start:], dtype=np.int64).reshape(-1, 2),
                  'step_seconds': np.asarray(self.seconds[token_start:], dtype=np.float64),
                  'first_state_index': np.asarray(first, dtype=np.int64),
                  'last_state_index': np.asarray(end, dtype=np.int64),
                  'packet_indices': np.asarray(INDICES, dtype=np.int64),
                  **{key: np.asarray(value, dtype=np.float64) for key, value in CONTROLS.items()}}
        record = batch.save_arrays(self.recorder.directory/name, arrays)
        if final:
            self.final = record
        else:
            self.blocks.append(record); self.saved = end
        self.recorder.event('trajectory_saved', final=final, first_state_index=first, last_state_index=end, artifact=record)


def precision(observations):
    return all(observations.get(group, {}).get(key) == 'torch.float64' for group, keys in {
        'norm': ('input', 'reduction', 'weight', 'output'),
        'rotary': ('input', 'inverse_frequency', 'angles', 'cos', 'sin')}.items() for key in keys)


def evaluate(backend, x, row, recorder, budget, prefix, expanded_hash, archive, report):
    original = x.copy()
    before = backend.cache_arrays()
    budget.claim(row['label'])
    if budget.calls != row['call']:
        raise ValueError('fixed model-call schedule drifted')
    row['status'] = 'started'
    arrays = {'input': original, **{key: np.asarray(value, dtype=np.float64) for key, value in CONTROLS.items()}}
    returned = None
    try:
        y, p, seconds, extra = backend.forward(original.copy(), row['cached'])
        arrays.update(output=y, probabilities=p, input_after=extra['input_after'])
        returned = (y, p)
        if row['phase'] == 'update':
            archive.append(y, p, seconds)
        row.update(seconds=seconds, input_sha256=array_hash(original), output_sha256=array_hash(y),
                   probability_sha256=array_hash(p), input_after_sha256=array_hash(extra['input_after']),
                   dtype_observations=copy.deepcopy(backend.observations),
                   output_dtype=extra['observed_output_dtype'], probability_dtype=extra['observed_probability_dtype'])
        if 'used_cache_before' in extra:
            expanded = {**extra['used_cache_before'], 'length': np.asarray(len(reference.PREFIX_IDS), dtype=np.int64)}
            row['expanded_cache_sha256'] = cache_hash(expanded)
            row['used_cache_length_after'] = int(extra['used_cache_length_after'])
            if row['label'] == 'diagnostic_initial_cached':
                with batch.preserve_call_files():
                    report['expanded_prefix_cache'] = batch.save_arrays(recorder.directory/'prefix-cache-expanded.npz', expanded)
        else:
            row.update(expanded_cache_sha256=None, used_cache_length_after=None)
        after = backend.cache_arrays()
        row.update(cache_before_sha256=cache_hash(before), cache_after_sha256=cache_hash(after),
                   cache_length_before=int(before['length']), cache_length_after=int(after['length']))
        valid = (y.shape == (2, batch.DIM) and p.shape == (2, batch.VOCAB)
                 and y.dtype == p.dtype == np.float64 and np.isfinite(y).all() and np.isfinite(p).all()
                 and np.all(p >= 0) and np.all(p <= 1) and np.allclose(p.sum(-1), 1., rtol=0, atol=DIAGNOSTIC_TOLERANCE))
        row.update(outputs_valid=bool(valid), top_tokens=p.argmax(-1).tolist() if valid else None,
                   probability_row_sums=p.sum(-1).tolist() if np.isfinite(p).all() else None,
                   input_unchanged=bool(extra['input_exact'] and np.array_equal(original, extra['input_after'])),
                   cache_unchanged=bool(batch.valid_cache(before) and batch.valid_cache(after)
                                        and batch.same_arrays(before, prefix) and batch.same_arrays(after, prefix)),
                   expanded_cache_valid=bool(not row['cached'] or (row['expanded_cache_sha256'] == expanded_hash
                                             and row['used_cache_length_after'] == len(reference.PREFIX_IDS)+1)),
                   precision_observations_passed=bool(precision(row['dtype_observations'])
                           and row['output_dtype'] == row['probability_dtype'] == 'torch.float64'))
        if not all(row[key] for key in ('outputs_valid', 'input_unchanged', 'cache_unchanged',
                                        'expanded_cache_valid', 'precision_observations_passed')):
            raise ValueError('input/cache/finite-output/native64 observation gate failed')
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('invalid forward duration')
        if reference.rss_bytes() >= budget.max_rss or time.monotonic()-recorder.started >= MAX_SECONDS:
            raise TimeoutError('resource cap after returned forward')
        row['status'] = 'returned'
    except BaseException as error:
        row.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        with batch.preserve_call_files():
            if row['phase'] != 'update' or row['status'] != 'returned':
                filename = row['label'] + '.npz' if row['phase'] != 'update' else f'failed-{row["label"]}.npz'
                row['artifact'] = batch.save_arrays(recorder.directory/filename, arrays)
            recorder.json(f'call-{row["call"]:03d}-{row["label"]}.json', row)
            recorder.event('model_call_completed', call=row['call'], label=row['label'], status=row['status'],
                           state_index=row['state_index'], artifact=row.get('artifact'))
            if row['phase'] == 'update':
                archive.save()
    return returned


def run(args, backend_factory=batch.QwenBackend, finalization_guard=None):
    recorder = reference.Recorder(args.output_dir)
    budget = reference.Budget(recorder, MAX_CALLS, MAX_SECONDS, MAX_RSS_GIB)
    report = {'status': 'failed', 'evidence_status': 'exploratory finite-horizon batch validation',
              'calls': schedule(), 'diagnostics': {}, 'prefix_status': 'unstarted',
              'trajectory': None, 'blocks': [], 'returned_updates': 0, 'qualified_updates': 0,
              'timing_scope': 'Each timed batch forward includes normal cache handling, model, softmax and feedback; excludes tensor preparation, dtype/cache audit copies and hashes, serialization and independent scalar comparison.',
              'scope': 'Two fixed states carried for 600 batch updates. No baseline outcomes or reference labels are receiver inputs; no early stopping. Separate local audit compares saved scalar trajectories and readout policy. No certified/global dynamics or end-to-end communication claim.'}
    backend, archive, prefix = None, None, None
    diagnostics = {}
    try:
        if args.packet_sha256 != PACKET_SHA256:
            raise ValueError('require the pinned untouched fresh-readout packet')
        packet_binding = reference.require_hash(args.packet, PACKET_SHA256)
        with np.load(args.packet, allow_pickle=False) as source:
            if set(source.files) != {'states'}:
                raise ValueError('packet must contain states only')
            states = source['states']
        if states.shape != (8, batch.DIM) or states.dtype != np.float64 or not np.isfinite(states).all():
            raise ValueError('invalid eight-state float64 packet')
        x = states[list(INDICES)].copy()
        archive = TrajectoryArchive(recorder, x)
        root = Path(__file__).resolve().parents[1]
        source = {'source_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                  'source_hashes': {Path(path).name: reference.sha256(path) for path in
                       (__file__, batch.__file__, reference.__file__, batch.writes.__file__)}}
        recorder.json('inputs.json', {'packet': packet_binding, 'packet_indices': list(INDICES),
            'prefix_ids': reference.PREFIX_IDS, **source})
        report['initial_states'] = batch.save_arrays(recorder.directory/'initial-states.npz', {'states': x})
        started = time.monotonic()
        backend = backend_factory(args, budget, recorder)
        report['backend_initialization_seconds'] = time.monotonic()-started
        recorder.json('metadata.json', {'utc': reference.utc_now(), **source, **backend.bindings,
            'model': reference.MODEL, 'revision': reference.REVISION, 'packet': packet_binding,
            'packet_indices': list(INDICES), 'prefix_ids': reference.PREFIX_IDS, 'controls': CONTROLS,
            'steps': STEPS, 'block_size': BLOCK, 'batch_size': 2, 'threads': 1,
            'max_calls': MAX_CALLS, 'max_seconds': MAX_SECONDS, 'max_rss_gib': MAX_RSS_GIB,
            'max_raw_bytes': MAX_RAW_BYTES, 'diagnostic_tolerance': DIAGNOSTIC_TOLERANCE,
            'trajectory_tolerance': TRAJECTORY_TOLERANCE,
            'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}})
        try:
            report['prefix_status'] = 'started'
            backend.prepare()
            report['prefix_status'] = 'returned'
        except BaseException:
            report['prefix_status'] = 'failed'
            raise
        finally:
            if budget.calls:
                recorder.event('model_call_completed', call=1, label='native64_prefix', status=report['prefix_status'])
        prefix = backend.cache_arrays()
        report['initial_prefix_cache'] = batch.save_arrays(recorder.directory/'prefix-cache-initial.npz', prefix)
        if not batch.valid_cache(prefix):
            raise ValueError('invalid scalar prefix cache')
        expanded = {key: np.repeat(value, 2, axis=0) if key != 'length' else value for key, value in prefix.items()}
        expanded_hash = cache_hash(expanded)
        report['scalar_prefix_sha256'], report['expanded_prefix_sha256'] = cache_hash(prefix), expanded_hash
        for row in report['calls']:
            output = evaluate(backend, x, row, recorder, budget, prefix, expanded_hash, archive, report)
            if row['phase'] == 'update':
                x = output[0].copy()
            else:
                diagnostics[row['label']] = output
                if not row['cached']:
                    name = row['phase']
                    report['diagnostics'][name] = batch.compare(diagnostics[f'diagnostic_{name}_cached'], output)
                    recorder.json(f'diagnostic-{name}-comparison.json', report['diagnostics'][name])
            if sum(path.stat().st_size for path in recorder.directory.iterdir() if path.is_file()) >= MAX_RAW_BYTES-16*1024**2:
                raise MemoryError('raw artifact budget reached, reserving partial final trajectory')
        report['diagnostic_gates_passed'] = all(item['passed'] for item in report['diagnostics'].values())
        report['status'] = 'completed' if report['diagnostic_gates_passed'] else 'numerical_gate_failed'
    except BaseException as error:
        report.update(status='incomplete' if isinstance(error, (TimeoutError, InterruptedError, MemoryError)) else 'failed',
                      error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    finally:
        if finalization_guard is not None:
            finalization_guard()
        try:
            with batch.preserve_call_files():
                if archive is not None:
                    archive.save(final=True)
                    report.update(trajectory=archive.final, blocks=archive.blocks, returned_updates=len(archive.states)-1)
                if prefix is not None:
                    final_cache = backend.cache_arrays()
                    report['final_prefix_cache'] = batch.save_arrays(recorder.directory/'prefix-cache-final.npz', final_cache)
                    report['final_cache_unchanged'] = batch.valid_cache(final_cache) and batch.same_arrays(prefix, final_cache)
                    if not report['final_cache_unchanged'] and report['status'] in ('completed', 'numerical_gate_failed'):
                        report.update(status='failed', error='final cache changed')
        except BaseException as error:
            report.update(status='failed', finalization_error=f'{type(error).__name__}: {error}')
        report['qualified_updates'] = sum(row['phase'] == 'update' and row['status'] == 'returned' for row in report['calls'])
        elapsed, rss = time.monotonic()-recorder.started, reference.rss_bytes()
        raw_bytes = sum(path.stat().st_size for path in recorder.directory.iterdir() if path.is_file())
        report.update(model_calls=budget.calls, elapsed_seconds=elapsed, peak_rss_bytes=rss,
                      raw_bytes_before_results=raw_bytes, dtype_observations={} if backend is None else backend.observations)
        report['resource_limits_passed'] = budget.calls <= MAX_CALLS and elapsed <= MAX_SECONDS and rss < budget.max_rss and raw_bytes < MAX_RAW_BYTES
        report['full_schedule_completed'] = (budget.calls == MAX_CALLS and report['returned_updates'] == STEPS
            and report['qualified_updates'] == STEPS and all(row['status'] == 'returned' for row in report['calls']))
        if report['status'] in ('completed', 'numerical_gate_failed') and (not report['resource_limits_passed'] or not report['full_schedule_completed']):
            report.update(status='incomplete', error='final schedule/resource gate failed')
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
    signals = (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)
    handlers = {sig: signal.getsignal(sig) for sig in signals}
    def guard():
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig in signals:
            signal.signal(sig, signal.SIG_IGN)
    def stop(sig, frame):
        guard()
        raise TimeoutError('internal deadline') if sig == signal.SIGALRM else InterruptedError(f'signal {sig}')
    try:
        for sig in signals:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
        result = run(args, finalization_guard=guard)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, old in handlers.items():
            signal.signal(sig, old)
    print(json.dumps({'status': result['status'], 'model_calls': result['model_calls'], 'returned_updates': result['returned_updates']}))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
