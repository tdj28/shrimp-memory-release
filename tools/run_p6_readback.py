#!/usr/bin/env python3
"""A fresh model process evolves eight explicit received vectors at common hold.

By default receives no expected bits, source trial order, write endpoints or
reference cycles; the local evaluator performs final decoding. Optional shadow
observation explicitly supplies reference cycles and records proposed readouts
without stopping or changing the model. This is numerical state transport,
not ordinary-language communication.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import check_p6_writes as writes
import validate_qwen_map as reference

STEPS, MAX_CALLS, MAX_SECONDS, MAX_RSS_GIB = 120, 977, 600, 8
CONTROLS = dict(writes.HOLD)
SHADOW_REFERENCE_SHA256 = 'd96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0'
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def load_shadow_references(path):
    """Require the pinned archive before importing the numerical observer."""
    if any(os.environ.get(name) != '1' for name in THREAD_VARIABLES):
        raise ValueError('configure one-thread environment before shadow imports')
    binding = reference.require_hash(path, SHADOW_REFERENCE_SHA256)
    import numpy as np
    import streaming_readout

    with np.load(path, allow_pickle=False) as archive:
        refs = archive['states']
    # Use the actual observer's validation without advancing its state.
    streaming_readout.StreamingReadout(refs)
    sources = {Path(p).name: reference.sha256(p)
               for p in (streaming_readout.__file__, streaming_readout.gates.__file__)}
    return refs, {'references': binding, 'source_hashes': sources,
                  'information_access': 'Reference cycles supplied explicitly; no expected packet labels.',
                  'scope': 'Shadow observation only; full fixed horizon remains unchanged.'}


class ShadowReadout:
    """Runner adapter whose report and checkpoint sidecars retain partial work."""
    def __init__(self, references, definition):
        if any(os.environ.get(name) != '1' for name in THREAD_VARIABLES):
            raise ValueError('configure one-thread environment before shadow imports')
        from streaming_readout import StreamingReadout

        self.observer = StreamingReadout(references)
        self.report = {'status': 'running', 'definition': definition, 'completed_steps': 0,
                       'checkpoints': [], 'decision': None, 'observer_seconds_per_update': [],
                       'total_observer_seconds': 0.0, 'artifacts': [],
                       'timing_scope': 'Only StreamingReadout.push; excludes loading, logging, model calls and report bookkeeping.'}

    def observe(self, state, completed_step):
        if completed_step != self.report['completed_steps'] + 1:
            raise ValueError('shadow requires consecutive post-update states, excluding the initial state')
        started = time.perf_counter()
        try:
            checkpoint = self.observer.push(state)
        except BaseException as error:
            self.report.update(status='failed', attempted_step=completed_step,
                               failed_observation_seconds=time.perf_counter()-started,
                               error=f'{type(error).__name__}: {error}')
            raise
        seconds = time.perf_counter()-started
        self.report['observer_seconds_per_update'].append(seconds)
        self.report['total_observer_seconds'] += seconds
        self.report['completed_steps'] = completed_step
        if checkpoint is not None:
            self.report['checkpoints'].append(checkpoint)
        self.report['decision'] = None if self.observer.decision is None else dict(self.observer.decision)

    def save(self, recorder, packet_index, *, final=False, stage_status=None):
        if final and self.report['status'] != 'failed':
            self.report['status'] = 'completed' if stage_status == 'completed' else 'incomplete'
        name = f"trial-{packet_index:02d}-shadow-{'final' if final else 'block'}-{self.report['completed_steps']:03d}.json"
        path = recorder.directory / name
        with path.open('x', encoding='utf-8') as stream:
            json.dump(self.report, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        binding = {'path': name, 'bytes': path.stat().st_size, 'sha256': reference.sha256(path)}
        self.report['artifacts'].append(binding)
        recorder.event('shadow_checkpoint', packet_index=packet_index, final=final,
                       completed_steps=self.report['completed_steps'], **binding)


def run_limits(steps):
    if steps not in (120, 600):
        raise ValueError('only fixed 120 or 600 update designs are supported')
    return 1+8*(2+steps), 600 if steps==120 else 2700


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache-dir", "vectors", "packet", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--steps", type=int, choices=[120,600], default=120)
    parser.add_argument("--shadow-references", type=Path,
                        help='Optional pinned reference cycles for shadow observation; never enables early stopping.')
    args = parser.parse_args(argv)
    steps = args.steps
    max_calls, max_seconds = run_limits(steps)
    recorder = reference.Recorder(args.output_dir)
    budget = reference.Budget(recorder, max_calls, max_seconds, MAX_RSS_GIB)
    controls = CONTROLS
    rows = []
    result = {"status":"failed", "evidence":"separate-process numerical state-vector readback", "rows":rows}
    observations = {}
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    def stop(sig, frame):
        raise TimeoutError("wall-time cap reached") if sig == signal.SIGALRM else InterruptedError(f"signal {sig}")
    for sig in handlers:
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, max_seconds)
    try:
        packet_binding = reference.require_hash(args.packet,args.packet_sha256)
        for key in THREAD_VARIABLES:
            os.environ[key] = "1"
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        shadow_refs, shadow_definition = (None, None) if args.shadow_references is None else load_shadow_references(args.shadow_references)
        import numpy as np
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        snapshot = args.cache_dir / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / reference.REVISION
        root = Path(__file__).resolve().parents[1]
        recorder.json("metadata.json", {"utc": reference.utc_now(), "source_commit": subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
            "source_hashes": {Path(p).name: reference.sha256(p) for p in (__file__, reference.__file__, writes.__file__)},
            "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
            "packet": packet_binding, "controls": controls,
            "weights": reference.require_hash(snapshot/'model.safetensors', reference.WEIGHTS_SHA256),
            "config": reference.require_hash(snapshot/'config.json', writes.hold_test.CONFIG_SHA256),
            "vectors": reference.require_hash(args.vectors, reference.VECTOR_SHA256),
            "max_calls": max_calls, "max_seconds": max_seconds, "max_rss_gib": MAX_RSS_GIB,
            **({'shadow_readout': shadow_definition} if shadow_definition is not None else {})})
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        ids = tokenizer.apply_chat_template(reference.MESSAGES, add_generation_prompt=True)
        if list(ids['input_ids'] if hasattr(ids, 'keys') else ids) != reference.PREFIX_IDS:
            raise ValueError("prefix mismatch")
        model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
            dtype=torch.float64, attn_implementation='sdpa').cpu().eval().requires_grad_(False)
        cfg = model.config
        if (cfg.model_type, cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size) != ('qwen2', 896, 24, 151936):
            raise ValueError('unexpected model architecture')
        if any(t.dtype != torch.float64 or t.device.type != 'cpu' for t in model.parameters()):
            raise ValueError("parameters not CPU float64")
        with np.load(args.vectors, allow_pickle=False) as data:
            vector = torch.tensor(data['philosophy'], dtype=torch.float64)
        reference.patch_native_float64(model, observations)
        loop = reference.FeedbackMap(model, vector, controls["alpha"], controls["log10_temperature"], budget, recorder)
        loop.mode = 'native64'
        loop.prepare()
        with np.load(args.packet,allow_pickle=False) as data:
            if set(data.files) != {'states'}:
                raise ValueError('receiver packet must contain only states')
            received = data['states']
        if received.shape != (8,896) or received.dtype != np.float64 or not np.isfinite(received).all():
            raise ValueError('require eight finite float64 vectors')
        for i,initial in enumerate(received):
            stage={'name':'readback','controls':dict(controls),'steps':steps,'completed_steps':0,'status':'running'}
            rows.append({'packet_index':i,'stage':stage})
            x=torch.tensor(initial,dtype=torch.float64)
            archive=writes.StageArchive(recorder,i,stage,initial)
            shadow = None if shadow_refs is None else ShadowReadout(shadow_refs, shadow_definition)
            if shadow is not None: rows[-1]['shadow_readout'] = shadow.report
            try:
                stage['cache_comparison']=writes.check_stage_cache(loop,x,recorder,budget,f'trial-{i:02d}-readback',controls)
                if not stage['cache_comparison']['passed']:raise ValueError('cache comparison failed')
                for step in range(1,steps+1):
                    budget.claim(f'packet_{i:02d}_readback_{step:03d}')
                    started=time.monotonic()
                    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
                        y,probabilities=loop.forward(x,cached=True)
                    archive.states.append(y.detach().numpy().copy())
                    archive.seconds.append(time.monotonic()-started)
                    archive.tokens.append(int(probabilities.argmax()) if bool(torch.isfinite(probabilities).all()) else -1)
                    stage['completed_steps']=step
                    if not np.isfinite(archive.states[-1]).all() or archive.tokens[-1]<0:
                        raise ValueError('non-finite output retained')
                    if reference.rss_bytes()>=budget.max_rss:raise MemoryError('RSS cap reached')
                    if shadow is not None: shadow.observe(archive.states[-1], step)
                    x=y
                    if step%12==0:
                        archive.save()
                        if shadow is not None: shadow.save(recorder, i)
                stage['status']='completed'
            finally:
                try:
                    archive.save(final=True)
                finally:
                    if shadow is not None: shadow.save(recorder, i, final=True, stage_status=stage['status'])
                recorder.event('packet_finished',**rows[-1])
        result['status']='completed'
    except BaseException as error:
        result.update(status='incomplete' if isinstance(error, (TimeoutError, InterruptedError, MemoryError)) else 'failed',
                      error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        if rows and rows[-1]['stage']['status']=='running':rows[-1]['stage']['status']=result['status']
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in handlers.items(): signal.signal(sig, handler)
        result.update(model_calls=budget.calls, elapsed_seconds=time.monotonic()-recorder.started,
                      peak_rss_bytes=reference.rss_bytes(), dtype_observations=observations)
        recorder.json('results.json', result)
        recorder.event('finished', status=result['status'])
        recorder.stream.close()
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
