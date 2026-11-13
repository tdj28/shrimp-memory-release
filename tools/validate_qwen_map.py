#!/usr/bin/env python3
"""Bounded, offline CPU validation of the pinned one-token Qwen feedback map.

This is a numerical readiness experiment, not a basin, memory or proof claim.
The native-float64 formulas and fixed prefix follow neural-shrimp's published
tools/recreate_p8.py. Only this loaded model's methods are patched. Run under an
external timeout as well: native-library calls can delay Python signal handling.
No weights are downloaded. All completed rows and arrays survive later failure.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback
import types

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
WEIGHTS_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
VECTOR_SHA256 = "b630bb7bef06bde87f8e6a7ca5e8672f83d24bc0585bf19048e9e494a4357f41"
REFERENCE_SHA256 = "11b2d28e0aecbeeb3def5646f6cfd01763d082f6218b2d26ea16a6737813cb05"
PREFIX_IDS = [151644, 8948, 198, 2610, 525, 458, 15235, 7404, 448, 2441, 15235, 13,
              1446, 614, 4583, 11290, 13, 31733, 1910, 311, 22729, 8820, 498, 1366,
              13, 151645, 198, 151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198]
MESSAGES = [{"role": "system", "content": "You are an AI talking with another AI. You have complete freedom. "
             "Feel free to pursue whatever you want."}, {"role": "user", "content": "Hello!"}]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"input hash mismatch: {path}; observed {observed}")
    return {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size, "sha256": observed}


def load_test_state(path, expected_hash):
    """Qualify one external state without casting, flattening, or loading a model."""
    import numpy as np

    provenance = require_hash(path, expected_hash)
    with np.load(path, allow_pickle=False) as archive:
        if "state" not in archive.files:
            raise ValueError("external test-state NPZ is missing key 'state'")
        state = archive["state"]
    if state.shape != (896,) or state.dtype != np.float64 or not np.isfinite(state).all():
        raise ValueError("external test state must be a finite float64 vector with shape (896,)")
    return state, {**provenance, "key": "state", "shape": list(state.shape), "dtype": str(state.dtype)}


def utc_now():
    return subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()


def rss_bytes():
    # This tool targets Linux; ru_maxrss there is KiB, and is a process high-water mark.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def convergence_report(rows, tolerance=1e-6):
    """Require two consecutive approximately quadratic halvings and a small error.

    This empirical gate concerns exactly the tested state and direction. Zero or
    non-finite errors do not manufacture a convergence order.
    """
    if not rows or any(not math.isfinite(r["relative_error"]) or r["relative_error"] < 0 for r in rows):
        return {"passed": False, "reason": "missing or invalid derivative errors", "orders": []}
    orders, run, longest = [], 0, 0
    for left, right in zip(rows, rows[1:]):
        halving = math.isclose(left["h"] / right["h"], 2.0, rel_tol=1e-10)
        order = None
        if halving and left["relative_error"] > 0 and right["relative_error"] > 0:
            order = math.log2(left["relative_error"] / right["relative_error"])
        qualifies = order is not None and 1.5 <= order <= 2.5
        run = run + 1 if qualifies else 0
        longest = max(longest, run)
        orders.append({"h_large": left["h"], "h_small": right["h"], "order": order,
                       "qualifies": qualifies})
    best = min(row["relative_error"] for row in rows)
    return {"passed": longest >= 2 and best <= tolerance, "orders": orders,
            "minimum_relative_error": best, "relative_tolerance": tolerance,
            "required_consecutive_quadratic_halvings": 2, "observed_longest_run": longest}


class Recorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = (self.directory / "events.jsonl").open("x", encoding="utf-8")
        self.started = time.monotonic()

    def event(self, event, **details):
        row = {"utc": utc_now(), "elapsed_seconds": time.monotonic() - self.started,
               "peak_rss_bytes": rss_bytes(), "event": event, **details}
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        print(json.dumps(row, allow_nan=False), flush=True)

    def json(self, filename, value):
        with (self.directory / filename).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")


class Budget:
    def __init__(self, recorder, max_calls, max_seconds, max_rss_gib):
        self.recorder = recorder
        self.max_calls, self.max_seconds = max_calls, max_seconds
        self.max_rss = max_rss_gib * 2 ** 30
        self.calls = 0

    def claim(self, label):
        if self.calls >= self.max_calls:
            raise RuntimeError("model-call budget exhausted")
        if time.monotonic() - self.recorder.started >= self.max_seconds:
            raise TimeoutError("wall-time budget exhausted")
        if rss_bytes() >= self.max_rss:
            raise MemoryError("process peak RSS exceeds the declared cap")
        self.calls += 1
        self.recorder.event("model_call_started", call=self.calls, label=label)


def patch_native_float64(model, observations):
    """Patch instances, rebuild RoPE from config, and reject dispatch-hook bypasses."""
    import torch

    norms, rotaries = [], []
    for name, module in model.named_modules():
        if hasattr(module, "_hf_hook") or hasattr(module, "_old_forward"):
            raise ValueError(f"dispatched module unsupported: {name}")
        if type(module).__name__.endswith("RMSNorm"):
            norms.append((name, module))
        if type(module).__name__.endswith("RotaryEmbedding"):
            rotaries.append((name, module))
    if len(norms) != 49 or len(rotaries) != 1:
        raise ValueError(f"unexpected Qwen precision modules: {len(norms)} norms, {len(rotaries)} rotaries")

    def norm_forward(self, hidden):
        if hidden.dtype != torch.float64 or self.weight.dtype != torch.float64:
            raise ValueError("non-float64 RMSNorm input or weight")
        variance = hidden.square().mean(-1, keepdim=True)
        result = self.weight * (hidden * torch.rsqrt(variance + self.variance_epsilon))
        observations["norm"] = {"input": str(hidden.dtype), "reduction": str(variance.dtype),
                                "weight": str(self.weight.dtype), "output": str(result.dtype)}
        return result

    for _, module in norms:
        module.forward = types.MethodType(norm_forward, module)
    for _, module in rotaries:
        if getattr(module, "rope_type", "default") != "default":
            raise ValueError("only default Qwen RoPE is supported")
        cfg = module.config
        dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        if getattr(cfg, "partial_rotary_factor", 1.0) != 1.0:
            raise ValueError("partial rotary embedding unsupported")
        params = getattr(cfg, "rope_parameters", None) or {}
        base = getattr(cfg, "rope_theta", None) or params.get("rope_theta")
        if base is None:
            raise ValueError("missing rotary base")
        # Recompute in float64, never cast the previously rounded stock frequencies.
        inv64 = 1.0 / (float(base) ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        own = module.inv_freq.detach().cpu().double()
        if inv64.shape != own.shape or not torch.allclose(inv64, own, rtol=1e-6, atol=0):
            raise ValueError("rebuilt rotary frequencies disagree with configured model")
        module.register_buffer("_shrimp_inv64", inv64, persistent=False)

        def rotary_forward(self, x, position_ids):
            if x.dtype != torch.float64:
                raise ValueError("non-float64 rotary input")
            freqs = position_ids[..., None].double() * self._shrimp_inv64
            angles = torch.cat((freqs, freqs), dim=-1)
            cosine, sine = angles.cos() * self.attention_scaling, angles.sin() * self.attention_scaling
            observations["rotary"] = {"input": str(x.dtype), "inverse_frequency": str(self._shrimp_inv64.dtype),
                                      "angles": str(angles.dtype), "cos": str(cosine.dtype), "sin": str(sine.dtype)}
            return cosine, sine

        module.forward = types.MethodType(rotary_forward, module)
    return {"norms": [name for name, _ in norms], "rotaries": [name for name, _ in rotaries]}


class FeedbackMap:
    def __init__(self, model, vector, alpha, log10_temperature, budget, recorder):
        import torch
        self.torch, self.model, self.budget, self.recorder = torch, model, budget, recorder
        self.embedding = model.get_input_embeddings().weight.detach()
        self.prefix = torch.tensor([PREFIX_IDS], dtype=torch.long)
        self.temperature = 10.0 ** log10_temperature
        self.steering = alpha * vector.reshape(1, 1, -1)
        self.cache = None
        self.mode = "stock"
        self.handle = model.model.layers[12].register_forward_hook(self._steer)

    def _steer(self, module, args, output):
        if isinstance(output, tuple):
            return (output[0] + self.steering,) + output[1:]
        return output + self.steering

    def prepare(self):
        from transformers import DynamicCache
        from torch.nn.attention import SDPBackend, sdpa_kernel
        self.budget.claim(f"{self.mode}_prefix")
        self.cache = DynamicCache()
        started = time.monotonic()
        with self.torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            self.model(input_ids=self.prefix, past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        if self.cache.get_seq_length() != len(PREFIX_IDS):
            raise ValueError("prefix cache has wrong length")
        self.recorder.event("prefix_ready", mode=self.mode, call=self.budget.calls,
                            seconds=time.monotonic() - started, length=self.cache.get_seq_length())

    def forward(self, x, cached=True):
        torch = self.torch
        if cached:
            cache = copy.copy(self.cache)
            cache.layers = [copy.copy(layer) for layer in self.cache.layers]
            result = self.model(inputs_embeds=x.reshape(1, 1, -1), past_key_values=cache,
                                use_cache=True, logits_to_keep=1)
        else:
            prefix = self.embedding[self.prefix]
            result = self.model(inputs_embeds=torch.cat((prefix, x.reshape(1, 1, -1)), dim=1),
                                use_cache=False, logits_to_keep=1)
        logits = result.logits[0, -1]
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        y = probabilities @ self.embedding
        if self.cache.get_seq_length() != len(PREFIX_IDS):
            raise ValueError("map evaluation mutated the stored prefix cache")
        if any(t.dtype != torch.float64 for t in (x, logits, probabilities, y)):
            raise ValueError("feedback map has non-float64 tensors")
        return y, probabilities

    def evaluate(self, label, x, tangent=None, cached=True):
        import numpy as np
        import torch.autograd.forward_ad as fw
        from torch.nn.attention import SDPBackend, sdpa_kernel
        torch = self.torch
        self.budget.claim(label)
        started = time.monotonic()
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            if tangent is None:
                y, probabilities = self.forward(x, cached=cached)
                derivative = None
            else:
                with fw.dual_level():
                    y, probabilities = self.forward(fw.make_dual(x, tangent), cached=cached)
                    unpacked = fw.unpack_dual(y)
                    y, derivative = unpacked.primal, unpacked.tangent
                    probabilities = fw.unpack_dual(probabilities).primal
                    if derivative is None:
                        raise ValueError("forward-mode tangent was lost")
            tensors = [y, probabilities] + ([] if derivative is None else [derivative])
            if any(not bool(torch.isfinite(t).all()) for t in tensors):
                raise ValueError("nonfinite map output")
            arrays = {"input": x.detach().numpy(), "output": y.detach().numpy(),
                      "probabilities": probabilities.detach().numpy()}
            if derivative is not None:
                arrays.update(direction=tangent.detach().numpy(), jvp=derivative.detach().numpy())
            filename = f"call-{self.budget.calls:03d}-{label}.npz"
            with (self.recorder.directory / filename).open("xb") as stream:
                np.savez(stream, **arrays)
        self.recorder.event("map_evaluated", label=label, call=self.budget.calls, cached=cached,
                            seconds=time.monotonic() - started, array_file=filename,
                            sha256=sha256(self.recorder.directory / filename),
                            output_dtype=str(y.dtype), probability_dtype=str(probabilities.dtype),
                            output_norm=float(y.norm()), top_token=int(probabilities.argmax()))
        return y, derivative


def validation_action(loop, label, state, tangent=None, cached=True, steps=1):
    if steps == 1:
        return loop.evaluate(label, state, tangent=tangent, cached=cached)
    from composed_feedback_action import composed_action
    return composed_action(loop, state, tangent, steps=steps, label=label, cached=cached)


def planned_validation_calls(steps, fd_count):
    return 2 + (5 + 2 * fd_count) * steps


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--vectors", type=Path, required=True)
    state_source = p.add_mutually_exclusive_group()
    state_source.add_argument("--reference-cycle", type=Path)
    state_source.add_argument("--test-state", type=Path, help="NPZ with key state: one float64 vector of shape (896,)")
    p.add_argument("--test-state-sha256", help="required SHA-256 of --test-state")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.071674)
    p.add_argument("--log10-temperature", type=float, default=-0.115)
    p.add_argument("--max-model-calls", type=int, default=40)
    p.add_argument("--max-seconds", type=int, default=900)
    p.add_argument("--max-rss-gib", type=float, default=20)
    p.add_argument("--composition-steps", type=int, choices=(1, 6), default=1)
    p.add_argument("--fd-steps", type=float, nargs="+", default=[0.01, 0.005, 0.0025, 0.00125, 0.0001, 0.00001])
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not 1 <= args.threads <= (os.cpu_count() or 1):
        p.error("invalid CPU thread count")
    if (args.max_model_calls < 1 or args.max_seconds < 1 or not math.isfinite(args.max_rss_gib)
            or args.max_rss_gib <= 0 or not math.isfinite(args.alpha)
            or not math.isfinite(args.log10_temperature) or abs(args.log10_temperature) > 5
            or len(args.fd_steps) < 3 or len(args.fd_steps) > 12
            or any(not math.isfinite(h) or h <= 0 for h in args.fd_steps)
            or any(a <= b for a, b in zip(args.fd_steps, args.fd_steps[1:]))):
        p.error("invalid finite controls, bounds or strictly decreasing finite-difference steps")
    if args.reference_cycle and (args.alpha != 0.071674 or args.log10_temperature != -0.115):
        p.error("the reference cycle is valid only at its exact published parameters")
    if args.reference_cycle and args.composition_steps != 1:
        p.error("reference-cycle replay is restricted to the original one-step validation")
    if bool(args.test_state) != bool(args.test_state_sha256):
        p.error("--test-state and --test-state-sha256 must be provided together")
    if args.test_state_sha256 and (len(args.test_state_sha256) != 64
                                  or any(c not in "0123456789abcdef" for c in args.test_state_sha256)):
        p.error("--test-state-sha256 must be a lowercase SHA-256 digest")
    recorder = Recorder(args.output_dir)
    budget = Budget(recorder, args.max_model_calls, args.max_seconds, args.max_rss_gib)
    summary = {"status": "failed", "evidence": "exploratory numerical validation",
               "scope": "one state and one tangent direction; no basin, storage, or global precision claim"}
    if args.test_state:
        summary["scope"] = "one hash-bound external state and one tangent direction; no basin, storage, or global precision claim"
    summary["composition_steps"] = args.composition_steps
    summary["planned_calls_without_reference_replay"] = planned_validation_calls(args.composition_steps, len(args.fd_steps))
    if args.composition_steps > 1:
        summary["scope"] = "six-step derivative at one supplied state and tangent; no spectrum, local stability or hysteresis claim"
    old_handler = signal.getsignal(signal.SIGALRM)

    def alarm(signum, frame):
        raise TimeoutError("declared wall-time limit reached")

    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    try:
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = str(args.threads)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        snapshot = args.cache_dir / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / REVISION
        metadata = {"utc": utc_now(), "model": MODEL, "revision": REVISION,
                    "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "source_commit": subprocess.check_output(
                        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                        text=True).strip(),
                    "script_sha256": sha256(__file__), "prefix_ids": PREFIX_IDS,
                    "weights": require_hash(snapshot / "model.safetensors", WEIGHTS_SHA256),
                    "vectors": require_hash(args.vectors, VECTOR_SHA256)}
        if args.reference_cycle:
            metadata["reference_cycle"] = require_hash(args.reference_cycle, REFERENCE_SHA256)
        external_state = None
        if args.test_state:
            external_state, metadata["test_state"] = load_test_state(args.test_state, args.test_state_sha256)
        recorder.json("metadata.json", metadata)
        recorder.event("inputs_qualified")
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        recorder.event("environment", packages={key: importlib.metadata.version(key) for key in
                       ("numpy", "torch", "transformers", "safetensors")}, torch_threads=torch.get_num_threads())
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        ids = tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=True)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        if ids != PREFIX_IDS:
            raise ValueError("tokenizer prefix differs from the fixed 37 IDs")
        recorder.event("model_loading")
        model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                    dtype=torch.float64, attn_implementation="sdpa").cpu().eval().requires_grad_(False)
        cfg = model.config
        if (cfg.model_type, cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size) != ("qwen2", 896, 24, 151936):
            raise ValueError("unexpected model architecture")
        if any(t.device.type != "cpu" or t.dtype != torch.float64 for t in model.parameters()):
            raise ValueError("parameters are not exclusively CPU float64")
        recorder.event("model_loaded", parameter_count=sum(t.numel() for t in model.parameters()),
                       parameter_dtype="torch.float64", attention=cfg._attn_implementation,
                       floating_buffer_dtypes=sorted({str(t.dtype) for t in model.buffers() if t.is_floating_point()}))
        with np.load(args.vectors, allow_pickle=False) as archive:
            vector_array = archive["philosophy"]
        if vector_array.shape != (896,) or not np.isfinite(vector_array).all():
            raise ValueError("invalid philosophy vector")
        vector = torch.tensor(vector_array, dtype=torch.float64)
        loop = FeedbackMap(model, vector, args.alpha, args.log10_temperature, budget, recorder)
        def evaluate(label, state, tangent=None, cached=True):
            return validation_action(loop, label, state, tangent, cached, args.composition_steps)
        reference = None
        if args.reference_cycle:
            with np.load(args.reference_cycle, allow_pickle=False) as archive:
                reference = archive["states"]
            if reference.shape != (8, 896) or reference.dtype != np.float64 or not np.isfinite(reference).all():
                raise ValueError("invalid reference cycle")
        if external_state is not None:
            x, state_source = torch.tensor(external_state), "external_npz_state"
        elif reference is not None:
            x, state_source = torch.tensor(reference[0]), "reference_phase_0"
        else:
            x, state_source = loop.embedding[PREFIX_IDS[-1]].clone(), "last_prefix_embedding"
        v = torch.tensor(np.random.default_rng(0).normal(size=896), dtype=torch.float64)
        v = v / v.norm()
        recorder.event("test_state", source=state_source,
                       input_norm=float(x.norm()), direction_norm=float(v.norm()), vector_norm=float(vector.norm()))
        loop.prepare()
        stock, _ = evaluate("stock_base", x)
        observations = {}
        patches = patch_native_float64(model, observations)
        loop.mode = "native64"
        loop.prepare()  # Stock prefix caches contain the old precision path and must not be reused.
        native, _ = evaluate("native_base", x)
        repeated, _ = evaluate("native_repeat", x)
        full, _ = evaluate("native_full_prefix", x, cached=False)
        repeat_error = float((native - repeated).abs().max())
        cache_error = float((native - full).abs().max())
        summary["implementation"] = {"stock_native_max_absolute_difference": float((stock - native).abs().max()),
            "stock_native_relative_l2_difference": float((stock - native).norm() / native.norm()),
            "repeat_max_absolute_difference": repeat_error, "cache_full_max_absolute_difference": cache_error,
            "cache_tolerance": 1e-10, "passed": repeat_error == 0.0 and cache_error <= 1e-10}
        recorder.event("implementation_comparison", **summary["implementation"], patches=patches,
                       dtype_observations=observations)
        primal, tangent = evaluate("native_jvp", x, tangent=v)
        jvp_primal_error = float((primal - native).abs().max())
        if float(tangent.norm()) <= 1e-14:
            raise ValueError("tested direction has no resolved derivative")
        fd_rows = []
        for index, h in enumerate(args.fd_steps):
            plus, _ = evaluate(f"fd_{index:02d}_plus", x + h * v)
            minus, _ = evaluate(f"fd_{index:02d}_minus", x - h * v)
            approximation = (plus - minus) / (2 * h)
            row = {"h": h, "absolute_error": float((approximation - tangent).norm()),
                   "relative_error": float((approximation - tangent).norm() / tangent.norm())}
            fd_rows.append(row)
            recorder.event("central_difference", **row)
        summary["derivative"] = {**convergence_report(fd_rows), "rows": fd_rows,
                                  "jvp_primal_max_absolute_difference": jvp_primal_error}
        summary["derivative"]["passed"] &= jvp_primal_error <= 1e-10
        recorder.event("derivative_gate", **summary["derivative"])
        if reference is not None:
            cycle_rows = []
            for phase in range(8):
                observed, _ = loop.evaluate(f"reference_phase_{phase}", torch.tensor(reference[phase]))
                error = float((observed - torch.tensor(reference[(phase + 1) % 8])).abs().max())
                row = {"phase": phase, "max_absolute_error": error, "passed": error <= 1e-10}
                cycle_rows.append(row)
                recorder.event("reference_cycle_row", **row)
            summary["reference_cycle"] = {"rows": cycle_rows, "tolerance": 1e-10,
                "passed": all(row["passed"] for row in cycle_rows),
                "scope": "one-step replay at supplied states; not convergence from a fresh start"}
        gates = [summary["implementation"]["passed"], summary["derivative"]["passed"]]
        if reference is not None:
            gates.append(summary["reference_cycle"]["passed"])
        summary["dtype_observations"] = observations
        summary["status"] = "passed" if all(gates) else "failed"
    except BaseException as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        summary["status"] = "incomplete" if isinstance(error, (TimeoutError, KeyboardInterrupt, MemoryError)) else "failed"
        recorder.event("attempt_stopped", status=summary["status"], error=summary["error"])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        summary.update(model_calls=budget.calls, elapsed_seconds=time.monotonic() - recorder.started,
                       peak_rss_bytes=rss_bytes(), utc=utc_now())
        recorder.json("result.json", summary)
        recorder.event("attempt_finished", status=summary["status"], model_calls=budget.calls)
        recorder.stream.close()
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
