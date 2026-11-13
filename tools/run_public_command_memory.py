#!/usr/bin/env python3
"""One public, exact-command dispatch stream through the float64 P6 memory map.

Only the parser's typed action supplies numerical controls. There is no private
expected-bit input, semantic-model translator, hidden encoding, repair or retry.
Scientific correctness is evaluated separately from these execution records.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')

import numpy as np
import check_p6_writes as writes
import transparent_command_contract as contract

reference, recovery, hold_test = writes.reference, writes.recovery, writes.hold_test
MESSAGES = (
    'Write bit 0 to register R0.',
    'Write bit 2 to register R0.',
    'Write bit 1 to register R0.',
    'Write bit 0 to register R0. Then erase the latch.',
    'Write bit 1 to register R0.',
    'Do not write bit 0 to register R0.',
    'Write bit 0 to register R0.',
)
MAX_CALLS, MAX_SECONDS, EXTERNAL_SECONDS, GRACE = 1464, 840, 900, 30
MAX_RSS = 8 * 1024**3
WORKING_OUTPUT, FINAL_RESERVE = 64 * 1024**2, 8 * 1024**2
MAX_OUTPUT = WORKING_OUTPUT + FINAL_RESERVE
CHECKPOINT_EVERY = 12
PRECISION_SCOPE = ('Existing native float64 Qwen map and normalization/rotary patches; '
    'per-call dtype instrumentation and eight finite cache/full-prefix comparisons. '
    'No new derivative-convergence test or global precision proof.')


def array_hash(value):
    value = np.asarray(value)
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def binding(path):
    path = Path(path)
    return dict(path=path.name, bytes=path.stat().st_size, sha256=reference.sha256(path))


def message_plan():
    return [dict(message_id=i, message=text, status='not_started', parser_outcome='not_started',
        action=None, dispatch_count=0, call_start=None, call_end=None, model_calls=0,
        state_before=None, state_after=None, completed_steps=0, stages=[], readout=None,
        frontend_seconds=0.0, seconds=0.0) for i, text in enumerate(MESSAGES)]


def derive_design():
    """Derive the call envelope from parsed public actions, not a target array."""
    accepted, rejected, stages, updates = 0, 0, 0, 0
    for text in MESSAGES:
        try:
            action = contract.parse_command(text)
        except contract.CommandRejected:
            rejected += 1
            continue
        contract.verify_action(text, action)
        accepted += 1
        stages += len(action.stages)
        updates += sum(stage.steps for stage in action.stages)
    calls = updates + stages + 2*stages
    if (accepted, rejected, stages, updates, calls) != (4, 3, 8, 1440, MAX_CALLS):
        raise ValueError('public parser/schedule no longer has the declared fixed envelope')
    return dict(schema_version=1, evidence='adapted exploratory public-controller integration',
        messages=list(MESSAGES), initial_state=dict(reference_label='ad', reference_index=0, phase=0),
        map=dict(model=reference.MODEL, revision=reference.REVISION, prefix_ids=reference.PREFIX_IDS,
            hold=writes.HOLD, endpoints=writes.ENDPOINTS),
        schedule=dict(accepted_messages=accepted, rejected_messages=rejected, stages=stages,
            numerical_updates=updates, prefix_calls=stages, comparison_calls=2*stages, total_calls=calls),
        limits=dict(model_calls=MAX_CALLS, internal_seconds=MAX_SECONDS, external_seconds=EXTERNAL_SECONDS,
            termination_grace_seconds=GRACE, peak_rss_bytes=MAX_RSS, working_raw_bytes=WORKING_OUTPUT,
            finalization_reserve_bytes=FINAL_RESERVE, total_raw_bytes=MAX_OUTPUT, threads=1),
        cache_tolerance=writes.CACHE_TOLERANCE, match_tolerance=recovery.MATCH_TOLERANCE,
        checkpoint_every_steps=CHECKPOINT_EVERY, precision_scope=PRECISION_SCOPE,
        evaluation='Independent evaluator compares the original visible messages with saved controls and trajectories. '
                   'Producer readout is neutral orbit qualification, not an expected-bit oracle.',
        stopping='Execute fixed stream once, carrying actual finite state after wrong/unresolved readout. '
                 'Abort on execution/precision/cache/resource failure; no reset, retry or horizon extension.')


class Recorder:
    def __init__(self, directory, clock=time.monotonic, rss=reference.rss_bytes):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.clock, self.rss, self.started = clock, rss, clock()
        self.stream = (self.directory/'events.jsonl').open('x', encoding='utf-8')
        self.finalizing = False

    def size(self):
        return sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file())

    def admit(self, size, final=False):
        if self.size()+size > (MAX_OUTPUT if final or self.finalizing else WORKING_OUTPUT):
            raise RuntimeError('raw artifact limit reached')

    def blob(self, name, value, final=False):
        self.admit(len(value), final)
        path = self.directory/name
        with path.open('xb') as stream:
            stream.write(value); stream.flush(); os.fsync(stream.fileno())
        return binding(path)

    def json(self, name, value, final=False):
        data = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
        return self.blob(name, data, final)

    def event(self, event, _final=False, **details):
        data = json.dumps(dict(utc=reference.utc_now(), elapsed_seconds=self.clock()-self.started,
            peak_rss_bytes=self.rss(), event=event, **details), allow_nan=False)+'\n'
        self.admit(len(data.encode()), final=_final)
        self.stream.write(data); self.stream.flush(); os.fsync(self.stream.fileno())

    def npz(self, name, final=False, **arrays):
        stream = io.BytesIO(); np.savez_compressed(stream, **arrays)
        return self.blob(name, stream.getvalue(), final)

    def state(self, name, state, final=False):
        stream = io.BytesIO(); np.save(stream, state, allow_pickle=False)
        return {**self.blob(name, stream.getvalue(), final), 'array_sha256': array_hash(state),
                'shape': list(state.shape), 'dtype': str(state.dtype)}


class Budget:
    def __init__(self, recorder):
        self.recorder, self.calls = recorder, 0
        self.counts = dict(prefix=0, comparison=0, update=0)
        self.scope, self.stage_counts = None, None
        self.max_rss = MAX_RSS

    def check(self):
        if self.recorder.clock()-self.recorder.started >= MAX_SECONDS:
            raise TimeoutError('840-second internal deadline reached')
        if self.recorder.rss() >= MAX_RSS:
            raise MemoryError('8 GiB RSS cap reached')
        if self.recorder.size() > WORKING_OUTPUT:
            raise RuntimeError('working raw artifact limit reached')

    def begin(self, message_id, stage):
        self.scope = dict(message_id=message_id, stage=stage['name'], steps=stage['steps'])
        self.stage_counts = dict(prefix=0, comparison=0, update=0)

    def claim(self, label, category='prefix'):
        self.check()
        if self.scope is None or category not in self.counts:
            raise RuntimeError('model call outside a declared stage')
        per_stage = dict(prefix=1, comparison=2, update=self.scope['steps'])
        total = dict(prefix=8, comparison=16, update=1440)
        if (self.calls >= MAX_CALLS or self.stage_counts[category] >= per_stage[category]
                or self.counts[category] >= total[category]):
            raise RuntimeError('declared model-call limit reached')
        self.calls += 1; self.counts[category] += 1; self.stage_counts[category] += 1
        self.recorder.event('model_call_started', call=self.calls, category=category, label=label, **self.scope)


class StageArchive:
    """The established P6 block/final array contract, with bounded writes."""
    def __init__(self, recorder, message_id, stage, initial):
        self.recorder, self.message_id, self.stage = recorder, message_id, stage
        self.states, self.tokens, self.seconds, self.first = [initial.copy()], [], [], 0

    def save(self, final=False):
        completed = len(self.states)-1
        first = 0 if final else self.first
        kind = 'final' if final else 'block'
        name = f'trial-{self.message_id:02d}-{self.stage["name"]}-{kind}-{completed:03d}.npz'
        item = self.recorder.npz(name, final=final, states=np.stack(self.states[first:]),
            first_state_index=first, token_ids=np.asarray(self.tokens[first:], dtype=np.int64),
            step_seconds=np.asarray(self.seconds[first:], dtype=np.float64), **self.stage['controls'])
        self.stage.setdefault('checkpoints', []).append(item)
        if final: self.stage['trajectory'] = item
        self.first = completed
        self.recorder.event('checkpoint', message_id=self.message_id, stage=self.stage['name'],
                            completed_steps=completed, final=final, _final=final, **item)


def qualify_inputs(args):
    refs, provenance = recovery.qualify_references(args)
    recovery_provenance = writes.qualify_recovery(args, refs)
    root = Path(__file__).resolve().parents[1]
    design_path = root/'docs/receipts/p6-write-path-design.json'
    design = json.loads(design_path.read_text())
    if design['hold'] != writes.HOLD or design['candidate_endpoints'] != writes.ENDPOINTS:
        raise ValueError('committed P6 design differs from the reference implementation')
    return refs, dict(references=provenance, recovery_prerequisite=recovery_provenance,
        geometric_design=reference.require_hash(design_path, reference.sha256(design_path)))


class Backend:
    """Existing native-float64 P6 map; controls come only from parsed stages."""
    def __init__(self, args, recorder, budget):
        self.recorder, self.budget, self.observations = recorder, budget, {}
        snapshot = args.cache_dir/'models--Qwen--Qwen2.5-0.5B-Instruct'/'snapshots'/reference.REVISION
        inputs = dict(weights=reference.require_hash(snapshot/'model.safetensors', reference.WEIGHTS_SHA256),
            config=reference.require_hash(snapshot/'config.json', hold_test.CONFIG_SHA256),
            vectors=reference.require_hash(args.vectors, reference.VECTOR_SHA256))
        recorder.json('model-inputs.json', inputs)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_grad_enabled(False)
        recorder.event('environment', packages={name: importlib.metadata.version(name) for name in
            ('numpy', 'torch', 'transformers', 'safetensors')})
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        ids = tokenizer.apply_chat_template(reference.MESSAGES, add_generation_prompt=True)
        if list(ids['input_ids'] if hasattr(ids, 'keys') else ids) != reference.PREFIX_IDS:
            raise ValueError('tokenizer prefix differs from fixed map prefix')
        with np.load(args.vectors, allow_pickle=False) as archive: vector = archive['philosophy']
        if vector.shape != (896,) or not np.isfinite(vector).all(): raise ValueError('invalid steering vector')
        self.vector = torch.tensor(vector, dtype=torch.float64)
        budget.check(); recorder.event('model_loading')
        model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
            dtype=torch.float64, attn_implementation='sdpa').cpu().eval().requires_grad_(False)
        cfg = model.config
        if (cfg.model_type, cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size) != ('qwen2', 896, 24, 151936):
            raise ValueError('unexpected model architecture')
        if any(p.device.type != 'cpu' or p.dtype != torch.float64 for p in model.parameters()):
            raise ValueError('model parameters must all be CPU float64')
        patches = reference.patch_native_float64(model, self.observations)
        self.loop = reference.FeedbackMap(model, self.vector, writes.HOLD['alpha'],
            writes.HOLD['log10_temperature'], budget, recorder)
        self.loop.mode = 'native64'
        recorder.json('backend.json', dict(inputs=inputs, patches=patches, attention=cfg._attn_implementation,
            model=reference.MODEL, revision=reference.REVISION, prefix_ids=reference.PREFIX_IDS,
            precision_scope=PRECISION_SCOPE))
        budget.check()

    def precision(self):
        for name, keys in (('norm', ('input','reduction','weight','output')),
                           ('rotary', ('input','inverse_frequency','angles','cos','sin'))):
            if self.observations.get(name) != {key: 'torch.float64' for key in keys}:
                raise ValueError('missing or non-float64 effective precision observation: '+name)
        return copy.deepcopy(self.observations)

    def prepare(self, controls):
        writes.set_controls(self.loop, self.vector, controls)
        self.recorder.event('precision_observed', call=self.budget.calls, observations=self.precision())

    def forward(self, state, cached):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        x = self.torch.tensor(state, dtype=self.torch.float64)
        with self.torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            y, probabilities = self.loop.forward(x, cached=cached)
        self.recorder.event('precision_observed', call=self.budget.calls, observations=self.precision())
        return y.detach().numpy().copy(), probabilities.detach().numpy().copy()


def check_output(state, probabilities):
    if (state.shape != (896,) or state.dtype != np.float64 or not np.isfinite(state).all()
            or probabilities.ndim != 1 or probabilities.dtype != np.float64
            or not np.isfinite(probabilities).all() or len(probabilities) < 2):
        raise ValueError('invalid/non-float64 map output; available arrays preserved')


def cache_check(backend, state, row, stage, recorder, budget):
    before = state.copy(); outputs, artifacts = [], []
    for cached in (True, False):
        label = f'trial-{row["message_id"]:02d}-{stage["name"]}-'+('cached' if cached else 'full')
        budget.claim(label, 'comparison')
        y, probabilities = backend.forward(state.copy(), cached)
        artifacts.append(recorder.npz(label+'.npz', input=state, output=y, probabilities=probabilities,
                                      **stage['controls']))
        check_output(y, probabilities); budget.check(); outputs.append(y)
    gap = float(np.max(np.abs(outputs[0]-outputs[1])))
    unchanged = array_hash(state)==array_hash(before)
    result = dict(passed=gap <= writes.CACHE_TOLERANCE and unchanged,
        maximum_coordinate_gap=gap, tolerance=writes.CACHE_TOLERANCE,
        input_unchanged=unchanged, artifacts=artifacts)
    recorder.event('cache_comparison', message_id=row['message_id'], stage=stage['name'], **result)
    return result


def neutral_readout(trajectory, refs):
    gate = hold_test.cycle_gate(trajectory)
    distances = []
    for label, cycle in zip(recovery.LABELS, refs):
        item = hold_test.separation_gate(trajectory[-6:], cycle)
        distances.append(dict(label=label,
            distance=item['minimum_over_shifts_of_maximum_l2_distance'], best_shift=item['best_shift']))
    matches = [item['label'] for item in distances if item['distance'] <= recovery.MATCH_TOLERANCE]
    return dict(cycle_gate=gate, reference_distances=distances, reference_matches=matches,
        match_tolerance=recovery.MATCH_TOLERANCE,
        qualified_label=matches[0] if gate['passed'] and len(matches)==1 else None)


def execute_action(action, row, state, backend, refs, recorder, budget):
    row['dispatch_count'] += 1
    if row['dispatch_count'] != 1: raise RuntimeError('duplicate dispatcher invocation')
    row['stages'] = [dict(name=s.name, controls=dict(alpha=s.alpha, log10_temperature=s.log10_temperature),
        steps=s.steps, status='not_started', attempted_steps=0, completed_steps=0) for s in action.stages]
    for stage in row['stages']:
        archive = StageArchive(recorder, row['message_id'], stage, state)
        stage['status'] = 'running'; budget.begin(row['message_id'], stage)
        try:
            backend.prepare(stage['controls'])
            stage['cache_comparison'] = cache_check(backend, state, row, stage, recorder, budget)
            if not stage['cache_comparison']['passed']: raise ValueError('cache/full-prefix comparison failed')
            for step in range(1, stage['steps']+1):
                budget.claim(f'message_{row["message_id"]:02d}_{stage["name"]}_{step:03d}', 'update')
                stage['attempted_steps'] += 1
                started = recorder.clock()
                y, probabilities = backend.forward(state.copy(), True)
                elapsed = recorder.clock()-started
                try:
                    check_output(y, probabilities)
                except BaseException:
                    failed = recorder.npz(f'invalid-message-{row["message_id"]:02d}-{stage["name"]}-{step:03d}.npz',
                                          final=True, output=y, probabilities=probabilities)
                    stage.setdefault('failed_outputs', []).append(failed)
                    raise
                archive.states.append(y.copy()); archive.seconds.append(elapsed)
                archive.tokens.append(int(np.argmax(probabilities)))
                state[:] = y  # the only numerical state mutation
                stage['completed_steps'] = step; row['completed_steps'] += 1
                budget.check()
                if step % CHECKPOINT_EVERY == 0: archive.save()
            stage['status'] = 'completed'
        except BaseException as exc:
            stage.update(status=error_status(exc), error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            archive.save(final=True)
            recorder.event('stage_finished', message_id=row['message_id'], _final=True, **stage)
    row['readout'] = neutral_readout(np.stack(archive.states[1:]), refs)


def error_status(exc):
    return 'incomplete' if isinstance(exc, (TimeoutError, InterruptedError, MemoryError, KeyboardInterrupt)) else 'failed'


def execute_message(row, state, backend, refs, recorder, budget):
    row['status'] = 'running'; row['call_start'] = budget.calls
    started = recorder.clock()
    try:
        row['state_before'] = recorder.state(f'message-{row["message_id"]:02d}-before.npy', state)
        frontend = recorder.clock()
        try: action = contract.parse_command(row['message'])
        except contract.CommandRejected as exc:
            row.update(parser_outcome='rejected', rejection=str(exc))
            row['frontend_seconds'] = recorder.clock()-frontend
            if budget.calls != row['call_start'] or array_hash(state) != row['state_before']['array_sha256']:
                raise ValueError('rejected input performed a call or mutated state')
        else:
            contract.verify_action(row['message'], action)
            row.update(parser_outcome='accepted', action={**asdict(action), 'stages':[asdict(s) for s in action.stages]})
            def dispatch(parsed):
                row['frontend_seconds'] = recorder.clock()-frontend
                execute_action(parsed, row, state, backend, refs, recorder, budget)
            contract.dispatch_command(row['message'], dispatch, prepared_action=action)
        budget.check(); row['status'] = 'completed'
    except BaseException as exc:
        row.update(status=error_status(exc), error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        row['call_end'] = budget.calls; row['model_calls'] = budget.calls-row['call_start']
        row['seconds'] = recorder.clock()-started
        row['state_after'] = recorder.state(f'message-{row["message_id"]:02d}-after.npy', state, final=True)
        recorder.event('message_finished', _final=True, **row)


def digital_control():
    latch, rows = contract.DigitalLatch(0), []
    for i, text in enumerate(MESSAGES):
        start = time.monotonic(); before, calls = latch.state_bytes.hex(), latch.dispatch_count
        try: latch.execute(text); parsed = 'accepted'
        except contract.CommandRejected: parsed = 'rejected'
        rows.append(dict(message_id=i, message=text, parser_outcome=parsed, state_before=before,
            state_after=latch.state_bytes.hex(), dispatch_calls=latch.dispatch_count-calls,
            seconds=time.monotonic()-start))
    return dict(scope='Conventional finite exact-parser/digital-latch control, not a numerical or semantic oracle.', rows=rows)


def run(args, backend_factory=Backend, qualifier=qualify_inputs, clock=time.monotonic,
        rss=reference.rss_bytes, install_signals=True):
    recorder = Recorder(args.output_dir, clock, rss); budget = Budget(recorder)
    rows = message_plan(); backend = None; state = None
    result = dict(schema_version=1, status='failed', evidence='adapted exploratory public-controller integration',
        rows=rows, precision_scope=PRECISION_SCOPE, evaluation='Independent intended-bit evaluation is required.')
    old_handlers = {}
    if install_signals:
        def stop(signum, frame):
            if signum == signal.SIGALRM: raise TimeoutError('declared internal deadline reached')
            raise InterruptedError(f'received signal {signum}')
        old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
        for sig in old_handlers: signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    try:
        design = derive_design(); recorder.json('design.json', design)
        recorder.json('messages.json', list(MESSAGES)); recorder.json('planned-messages.json', rows)
        root = Path(__file__).resolve().parents[1]
        paths = [Path(__file__), Path(contract.__file__), Path(writes.__file__), Path(recovery.__file__),
                 Path(hold_test.__file__), Path(reference.__file__), root/'docs/public-command-memory.md',
                 root/'docs/receipts/p6-write-path-design.json', root/'pyproject.toml', root/'uv.lock']
        result['source'] = dict(utc=reference.utc_now(), source_commit=subprocess.check_output(
            ['git','-C',str(root),'rev-parse','HEAD'], text=True).strip(),
            arguments={k:str(v) for k,v in vars(args).items()}, files=[binding(p) for p in paths])
        recorder.json('source.json', result['source'])
        recorder.json('digital-control.json', digital_control())
        refs, provenance = qualifier(args)
        if refs.shape != (2,6,896) or refs.dtype != np.float64 or not np.isfinite(refs).all():
            raise ValueError('qualified references must be finite float64 (2,6,896)')
        recorder.json('qualified-inputs.json', provenance)
        recorder.npz('fixed-inputs.npz', references=refs, initial_state=refs[0,0])
        state = refs[0,0].copy()
        budget.check(); started = clock(); backend = backend_factory(args, recorder, budget)
        result['cold_initialization_seconds'] = clock()-started
        for row in rows:
            execute_message(row, state, backend, refs, recorder, budget)
        if budget.calls != MAX_CALLS or budget.counts != dict(prefix=8, comparison=16, update=1440):
            raise ValueError('completed stream has unexpected model-call accounting')
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status=error_status(exc), error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        if install_signals:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in old_handlers.items(): signal.signal(sig, handler)
        recorder.finalizing = True
        result.update(model_calls=budget.calls, call_counts=budget.counts,
            elapsed_seconds=clock()-recorder.started, peak_rss_bytes=rss(),
            precision_observations=getattr(backend, 'observations', {}),
            limits=dict(model_calls=MAX_CALLS, internal_seconds=MAX_SECONDS, peak_rss_bytes=MAX_RSS,
                        working_raw_bytes=WORKING_OUTPUT, total_raw_bytes=MAX_OUTPUT),
            accepted_messages=sum(r['parser_outcome']=='accepted' for r in rows),
            rejected_messages=sum(r['parser_outcome']=='rejected' for r in rows),
            completed_messages=sum(r['status']=='completed' for r in rows))
        for row in rows: recorder.json(f'message-{row["message_id"]:02d}.json', row, final=True)
        if state is not None: result['terminal_state'] = recorder.state('terminal-state.npy', state, final=True)
        result['resource_limits_passed'] = (budget.calls <= MAX_CALLS and result['elapsed_seconds'] < MAX_SECONDS
            and result['peak_rss_bytes'] < MAX_RSS and recorder.size() <= MAX_OUTPUT)
        recorder.json('result.json', result, final=True)
        recorder.event('finished', status=result['status'], model_calls=budget.calls)
        recorder.stream.close()
        recorder.json('manifest.json', dict(files=[binding(p) for p in sorted(recorder.directory.iterdir())
            if p.is_file() and p.name != 'manifest.json']), final=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache-dir','vectors','references','reference-manifest','source-hold-results','recovery-results','output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('references-sha256','reference-manifest-sha256','recovery-results-sha256'):
        p.add_argument('--'+name, required=True)
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    for digest in (args.references_sha256, args.reference_manifest_sha256, args.recovery_results_sha256):
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            p.error('input hashes must be lowercase SHA-256 digests')
    return 0 if run(args)['status']=='completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
