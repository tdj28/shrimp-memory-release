#!/usr/bin/env python3
"""Bounded offline CPU replay of the published Qwen period-8 reference.

Adapted from neural-shrimp/tools/recreate_p8.py, SHA-256
efc7f0e74c7102ff59143f76631e0a7db496c21625ff815a8e614f1c0f15ffaa.
This is a reference replay, not full-map derivative qualification or kite validation.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import time

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PINS = {
    "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "vectors": "b630bb7bef06bde87f8e6a7ca5e8672f83d24bc0585bf19048e9e494a4357f41",
    "reference": "11b2d28e0aecbeeb3def5646f6cfd01763d082f6218b2d26ea16a6737813cb05",
}
PREFIX = [151644, 8948, 198, 2610, 525, 458, 15235, 7404, 448, 2441, 15235, 13, 1446, 614,
          4583, 11290, 13, 31733, 1910, 311, 22729, 8820, 498, 1366, 13, 151645, 198, 151644,
          872, 198, 9707, 0, 151645, 198, 151644, 77091, 198]
ALPHA, LOG10_T = 0.071674, -0.115


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def patch_float64(model, torch):
    def rms(self, h):
        return self.weight * (h * torch.rsqrt(h.square().mean(-1, keepdim=True) + self.variance_epsilon))
    norms = [m for m in model.modules() if type(m).__name__.endswith("RMSNorm")]
    for module in norms:
        require(not hasattr(module, "_old_forward"), "device-dispatched models are outside this CPU replay")
        module.forward = rms.__get__(module)
    cfg = model.config
    dim = cfg.hidden_size // cfg.num_attention_heads
    base = getattr(cfg, "rope_theta", None) or cfg.rope_parameters["rope_theta"]
    inv64 = 1. / (float(base) ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))

    def rotary(self, x, position_ids):
        require(x.dtype == torch.float64 and x.device.type == "cpu", "non-float64 CPU rotary input")
        phases = position_ids[..., None].to(torch.float64) * inv64
        phases = torch.cat((phases, phases), dim=-1)
        return phases.cos() * self.attention_scaling, phases.sin() * self.attention_scaling

    rotations = [m for m in model.modules() if type(m).__name__.endswith("RotaryEmbedding")]
    for module in rotations:
        require(getattr(module, "rope_type", "default") == "default", "unsupported rotary scaling")
        require(not hasattr(module, "_old_forward"), "device-dispatched rotary module")
        require(torch.allclose(inv64, module.inv_freq.to(torch.float64), rtol=1e-6, atol=0), "rotary frequency mismatch")
        module.forward = rotary.__get__(module)
    require(norms and rotations, "missing RMSNorm or rotary modules")
    require(model.config._attn_implementation == "sdpa", "SDPA attention required")
    return {"rmsnorm_modules": len(norms), "rotary_modules": len(rotations), "head_dim": dim, "rope_theta": base}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--threads", type=int, choices=[1], default=1)
    args = parser.parse_args()
    if not 9 <= args.steps <= 120:
        parser.error("--steps must be in 9..120; this replay cannot extend its own budget")
    try:
        args.output_dir.mkdir(parents=False, exist_ok=False)
    except OSError as error:
        parser.error(f"a fresh output directory with an existing parent is required: {error}")
    start = time.perf_counter()
    states, tokens, step_seconds = [], [], []
    np, x0 = None, None
    report = {"status": "running", "scope": "reference P8 replay only; not derivative or kite validation",
              "source_sha256": sha256(__file__), "reference_source": "neural-shrimp/tools/recreate_p8.py",
              "reference_source_sha256": "efc7f0e74c7102ff59143f76631e0a7db496c21625ff815a8e614f1c0f15ffaa",
              "revision": REVISION, "alpha": ALPHA, "log10_temperature": LOG10_T,
              "steps_requested": args.steps, "threads": 1, "prefix_ids": PREFIX,
              "cache": "disabled; full 38-position input at every step", "checkpoints": [],
              "python": platform.python_version(), "platform": platform.platform(), "gates": {}}

    def status(phase):
        report.update(phase=phase, steps_completed=len(states), elapsed_seconds=time.perf_counter() - start,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      updated_utc=subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip())
        report["step_seconds"] = step_seconds
        temporary = args.output_dir / "status.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output_dir / "status.json")
        event = {key: report[key] for key in ("phase", "status", "steps_completed", "elapsed_seconds", "peak_rss_bytes")}
        with (args.output_dir / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    def checkpoint(final=False):
        if np is None or not states:
            return
        name = "trajectory-final.npz" if final else f"trajectory-step-{len(states):04d}.npz"
        path = args.output_dir / name
        with path.open("xb") as stream:
            np.savez_compressed(stream, states=np.stack(states), token_ids=np.asarray(tokens, dtype=np.int64),
                                initial_state=x0, step_seconds=np.asarray(step_seconds), prefix_ids=np.asarray(PREFIX))
        report["checkpoints"].append({"path": name, "sha256": sha256(path), "bytes": path.stat().st_size})

    def interrupt(signum, frame):
        raise InterruptedError(f"received signal {signum}; completed checkpoints retained")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        status("checking_inputs")
        require(args.snapshot.name == REVISION, "snapshot directory must name the pinned revision")
        inputs = {"model.safetensors": args.snapshot / "model.safetensors", "config.json": args.snapshot / "config.json",
                  "vectors": args.vectors, "reference": args.reference}
        report["inputs"] = {}
        for name, path in inputs.items():
            actual = sha256(path)
            report["inputs"][name] = {"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size}
            require(actual == PINS[name], f"pinned input hash mismatch: {name}")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        report["packages"] = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "transformers", "tokenizers", "safetensors")}
        with np.load(args.vectors, allow_pickle=False) as saved:
            vector = saved["philosophy"].copy()
        with np.load(args.reference, allow_pickle=False) as saved:
            reference = saved["states"].copy()
        require(vector.shape == (896,) and np.isfinite(vector).all(), "invalid steering vector")
        require(reference.shape == (8, 896) and np.isfinite(reference).all(), "invalid reference cycle")
        status("loading_model")
        tok = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.snapshot, local_files_only=True, dtype=torch.float64,
                                                     attn_implementation="sdpa").to("cpu").eval()
        model.requires_grad_(False)
        report["precision_patches"] = patch_float64(model, torch)
        messages = [{"role": "system", "content": "You are an AI talking with another AI. You have complete freedom. Feel free to pursue whatever you want."},
                    {"role": "user", "content": "Hello!"}]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        require(ids == PREFIX, "tokenizer prefix differs from the reference")
        v = torch.tensor(vector, dtype=torch.float64)

        def steer(module, inp, out):
            return (out[0] + ALPHA * v,) + tuple(out[1:]) if isinstance(out, tuple) else out + ALPHA * v

        model.model.layers[12].register_forward_hook(steer)
        embedding = model.get_input_embeddings().weight
        require(tuple(embedding.shape) == (151936, 896), "wrong embedding dimensions")
        require(embedding.data_ptr() == model.get_output_embeddings().weight.data_ptr(), "embedding weights are not tied")
        require(all(p.dtype == torch.float64 and p.device.type == "cpu" for p in model.parameters()), "mixed parameter precision/device")
        prefix = embedding[torch.tensor(PREFIX)]
        x = embedding[PREFIX[-1]].clone()
        x0 = x.numpy().copy()
        status("iterating")
        for step in range(1, args.steps + 1):
            tick = time.perf_counter()
            logits = model(inputs_embeds=torch.cat([prefix, x[None]])[None], use_cache=False).logits[0, -1]
            require(logits.dtype == torch.float64 and bool(torch.isfinite(logits).all()), "invalid logits")
            probabilities = torch.softmax(logits / (10. ** LOG10_T), dim=-1)
            x = probabilities @ embedding
            require(x.dtype == torch.float64 and bool(torch.isfinite(x).all()), "invalid feedback state")
            states.append(x.numpy().copy())
            tokens.append(int(probabilities.argmax()))
            step_seconds.append(time.perf_counter() - tick)
            if step % 10 == 0:
                checkpoint()
                status("iterating")
        trajectory = np.stack(states)
        gaps = {str(q): float(np.linalg.norm(trajectory[-1] - trajectory[-1-q])) for q in (1, 2, 4, 8)}
        gap, phase = min((float(np.max(np.abs(np.roll(trajectory[-8:], -k, axis=0) - reference))), k) for k in range(8))
        report["closure_norms"] = gaps
        report["reference_max_absolute_gap"] = gap
        report["reference_phase_shift"] = phase
        report["last_token_ids"] = tokens[-8:]
        report["last_tokens"] = [tok.decode([token]) for token in tokens[-8:]]
        report["gates"] = {"q8_closure_at_most_1e-11": gaps["8"] <= 1e-11,
                           "proper_divisor_gaps_above_1e-3": all(gaps[str(q)] > 1e-3 for q in (1, 2, 4)),
                           "reference_max_gap_at_most_1e-10": gap <= 1e-10}
        report["status"] = "passed" if all(report["gates"].values()) else "failed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            checkpoint(final=True)
        except Exception as error:
            report["status"] = "failed"
            report["checkpoint_error"] = f"{type(error).__name__}: {error}"
        status("finished")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
