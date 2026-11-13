#!/usr/bin/env python3
"""Bounded common-hold replay of two stored period-six branch seeds.

Offline CPU float64, one thread, at most 120 cached steps per seed. This is an
exploratory numerical coexistence check, not a writable-bit, basin, stability
proof, or gate experiment. Run under an external 900-second timeout as well;
native library calls can delay Python signal handling. No model is downloaded.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import validate_qwen_map as reference

CONFIG_SHA256 = "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"
PERIOD = 6
CLOSURE_TOLERANCE = 1e-8
DIVISOR_SEPARATION = 1e-4
CYCLE_SEPARATION = 1e-3


def cycle_gate(trajectory):
    """Gate all six lag-six pairs, and explicitly reject proper divisors."""
    import numpy as np

    states = np.asarray(trajectory)
    if states.ndim != 2 or len(states) < 2 * PERIOD or not np.isfinite(states).all():
        return {"passed": False, "reason": "fewer than twelve states or invalid trajectory"}
    cycle = states[-PERIOD:]
    closure = np.linalg.norm(cycle - states[-2 * PERIOD:-PERIOD], axis=1)
    divisors = []
    for divisor in (1, 2, 3):
        gaps = np.linalg.norm(cycle - np.roll(cycle, -divisor, axis=0), axis=1)
        divisors.append({"divisor": divisor, "phase_l2_gaps": gaps.tolist(),
                         "maximum_l2_gap": float(gaps.max()),
                         "excluded": bool(gaps.max() > DIVISOR_SEPARATION)})
    return {"passed": bool(closure.max() <= CLOSURE_TOLERANCE
                           and all(row["excluded"] for row in divisors)),
            "lag_six_phase_l2_residuals": closure.tolist(),
            "maximum_l2_residual": float(closure.max()),
            "closure_tolerance": CLOSURE_TOLERANCE,
            "proper_divisors": divisors, "divisor_separation": DIVISOR_SEPARATION}


def separation_gate(left, right):
    """Compare complete cycles, minimizing the maximum phase error over shifts."""
    import numpy as np

    left, right = np.asarray(left), np.asarray(right)
    if (left.shape != right.shape or left.ndim != 2 or len(left) != PERIOD
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        return {"passed": False, "reason": "missing or invalid six-state cycles"}
    rows = []
    for shift in range(PERIOD):
        distances = np.linalg.norm(left - np.roll(right, -shift, axis=0), axis=1)
        rows.append({"shift": shift, "phase_l2_distances": distances.tolist(),
                     "maximum_l2_distance": float(distances.max())})
    best = min(rows, key=lambda row: row["maximum_l2_distance"])
    return {"passed": best["maximum_l2_distance"] > CYCLE_SEPARATION,
            "minimum_over_shifts_of_maximum_l2_distance": best["maximum_l2_distance"],
            "best_shift": best["shift"], "required_separation": CYCLE_SEPARATION,
            "all_shifts": rows}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--vectors", type=Path, required=True)
    p.add_argument("--seeds", type=Path, required=True)
    p.add_argument("--seeds-sha256", required=True)
    p.add_argument("--seed-manifest", type=Path, required=True)
    p.add_argument("--seed-manifest-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=120, help="steps per seed, at most 120")
    p.add_argument("--threads", type=int, choices=[1], default=1)
    p.add_argument("--max-seconds", type=int, default=840, help="leave time before external 900-second stop")
    p.add_argument("--max-rss-gib", type=float, default=20)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (not 12 <= args.steps <= 120 or not 1 <= args.max_seconds <= 840
            or not math.isfinite(args.max_rss_gib) or not 0 < args.max_rss_gib <= 24):
        p.error("require 12..120 steps, 1..840 seconds, and a finite RSS limit in (0,24] GiB")
    for value in (args.seeds_sha256, args.seed_manifest_sha256):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            p.error("input hashes must be lowercase SHA-256 digests")

    recorder = reference.Recorder(args.output_dir)
    budget = reference.Budget(recorder, 1 + 2 * args.steps, args.max_seconds, args.max_rss_gib)
    rows = [{"label": label, "status": "not_started", "completed_steps": 0} for label in ("ad", "td")]
    summary = {"status": "failed", "evidence": "exploratory finite-precision observation",
               "scope": "two supplied seeds at one common hold; no write, basin, or stability-proof claim",
               "rows": rows, "separation": {"passed": False, "reason": "not evaluated"}}
    trajectories, observations = {}, {}
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}

    def stop(signum, frame):
        if signum == signal.SIGALRM:
            raise TimeoutError("declared wall-time limit reached")
        raise InterruptedError(f"received signal {signum}")

    for sig in old_handlers:
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    try:
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "1"
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        snapshot = args.cache_dir / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / reference.REVISION
        metadata = {"utc": reference.utc_now(), "model": reference.MODEL, "revision": reference.REVISION,
                    "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "source_commit": subprocess.check_output(
                        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                        text=True).strip(),
                    "script_sha256": reference.sha256(__file__),
                    "helper_sha256": reference.sha256(reference.__file__), "prefix_ids": reference.PREFIX_IDS,
                    "weights": reference.require_hash(snapshot / "model.safetensors", reference.WEIGHTS_SHA256),
                    "config": reference.require_hash(snapshot / "config.json", CONFIG_SHA256),
                    "vectors": reference.require_hash(args.vectors, reference.VECTOR_SHA256),
                    "seeds": reference.require_hash(args.seeds, args.seeds_sha256),
                    "seed_manifest": reference.require_hash(args.seed_manifest, args.seed_manifest_sha256),
                    "gates": {"closure_l2_tolerance": CLOSURE_TOLERANCE,
                              "proper_divisor_l2_separation": DIVISOR_SEPARATION,
                              "cycle_l2_separation": CYCLE_SEPARATION},
                    "checkpoint_every_steps": 12}
        manifest = json.loads(args.seed_manifest.read_text())
        if (manifest["output"]["sha256"] != args.seeds_sha256
                or manifest["output"]["order"] != ["ad", "td"]
                or [seed["label"] for seed in manifest["seeds"]] != ["ad", "td"]):
            raise ValueError("seed manifest does not bind the supplied seeds in ad,td order")
        hold = manifest["hold"]
        alpha, log_temperature = hold["alpha"], hold["log10_temperature"]
        if not all(math.isfinite(v) for v in (alpha, log_temperature)) or abs(log_temperature) > 5:
            raise ValueError("invalid hold controls")
        metadata.update(hold=hold, input_provenance=manifest)
        recorder.json("metadata.json", metadata)
        recorder.event("inputs_qualified", hold=hold)

        import numpy as np
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        with np.load(args.seeds, allow_pickle=False) as archive:
            seeds = archive["states"]
        if seeds.shape != (2, 1, 896) or seeds.dtype != np.float64 or not np.isfinite(seeds).all():
            raise ValueError("seeds must be a finite float64 array with shape (2,1,896)")
        for index, seed in enumerate(manifest["seeds"]):
            digest = hashlib.sha256(seeds[index].astype("<f8").tobytes()).hexdigest()
            if digest != seed["state_raw_little_endian_float64_sha256"]:
                raise ValueError(f"raw seed hash mismatch for {seed['label']}")
        with np.load(args.vectors, allow_pickle=False) as archive:
            vector = archive["philosophy"]
        if vector.shape != (896,) or not np.isfinite(vector).all():
            raise ValueError("invalid philosophy steering vector")
        recorder.event("environment", packages={name: importlib.metadata.version(name) for name in
                       ("numpy", "torch", "transformers", "safetensors")}, torch_threads=torch.get_num_threads())
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        ids = tokenizer.apply_chat_template(reference.MESSAGES, add_generation_prompt=True)
        if list(ids["input_ids"] if hasattr(ids, "keys") else ids) != reference.PREFIX_IDS:
            raise ValueError("tokenizer prefix differs from the fixed 37 IDs")
        recorder.event("model_loading")
        model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                    dtype=torch.float64, attn_implementation="sdpa").cpu().eval().requires_grad_(False)
        cfg = model.config
        if (cfg.model_type, cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size) != ("qwen2", 896, 24, 151936):
            raise ValueError("unexpected model architecture")
        if any(t.device.type != "cpu" or t.dtype != torch.float64 for t in model.parameters()):
            raise ValueError("model parameters must all be CPU float64")
        patches = reference.patch_native_float64(model, observations)
        recorder.event("model_loaded", parameter_count=sum(t.numel() for t in model.parameters()),
                       attention=cfg._attn_implementation, patches=patches)
        loop = reference.FeedbackMap(model, torch.tensor(vector, dtype=torch.float64),
                                     alpha, log_temperature, budget, recorder)
        loop.mode = "native64"
        loop.prepare()

        for index, row in enumerate(rows):
            label = row["label"]
            state_list = [seeds[index, 0].copy()]
            token_ids, step_seconds, checkpoint_end = [], [], 0
            row.update(status="running", seed_provenance=manifest["seeds"][index])

            def save_checkpoint(final=False):
                nonlocal checkpoint_end
                completed = len(state_list) - 1
                filename = f"{label}-{'final' if final else 'block'}-{completed:03d}.npz"
                path = recorder.directory / filename
                with path.open("xb") as stream:
                    np.savez_compressed(stream, states=np.stack(state_list if final else state_list[checkpoint_end:]),
                                        first_state_index=0 if final else checkpoint_end,
                                        token_ids=np.asarray(token_ids if final else token_ids[checkpoint_end:], dtype=np.int64),
                                        step_seconds=np.asarray(step_seconds if final else step_seconds[checkpoint_end:]),
                                        alpha=alpha, log10_temperature=log_temperature)
                    stream.flush()
                    os.fsync(stream.fileno())
                record = {"path": filename, "bytes": path.stat().st_size, "sha256": reference.sha256(path)}
                row.setdefault("checkpoints", []).append(record)
                if final:
                    row["trajectory"] = record
                checkpoint_end = completed
                recorder.event("checkpoint", label=label, completed_steps=completed, final=final, **record)

            try:
                x = torch.tensor(state_list[0], dtype=torch.float64)
                recorder.event("seed_started", label=label, seed_source_alpha=manifest["seeds"][index]["source_alpha"])
                for step in range(1, args.steps + 1):
                    budget.claim(f"{label}_step_{step:03d}")
                    started = time.monotonic()
                    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                        y, probabilities = loop.forward(x, cached=True)
                    state_list.append(y.detach().cpu().numpy().copy())
                    step_seconds.append(time.monotonic() - started)
                    token_ids.append(int(probabilities.argmax()) if bool(torch.isfinite(probabilities).all()) else -1)
                    row["completed_steps"] = step
                    if not np.isfinite(state_list[-1]).all() or token_ids[-1] < 0:
                        raise ValueError("non-finite map output; attempted output preserved")
                    x = y
                    if reference.rss_bytes() >= budget.max_rss:
                        raise MemoryError("process peak RSS exceeds the declared cap")
                    if step % 12 == 0:
                        save_checkpoint()
                        recorder.event("progress", label=label, step=step, cycle_gate=cycle_gate(state_list[1:]))
                row["cycle_gate"] = cycle_gate(state_list[1:])
                row["status"] = "passed" if row["cycle_gate"]["passed"] else "failed"
                row["last_six_top_token_ids"] = token_ids[-PERIOD:]
                row["last_six_decoded_tokens"] = [tokenizer.decode([token]) for token in token_ids[-PERIOD:]]
            except BaseException as error:
                row.update(status="incomplete" if isinstance(error, (TimeoutError, InterruptedError, MemoryError)) else "failed",
                           error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                # A global resource/signal stop ends the run. Other seed failures
                # are preserved while still testing the independently planned row.
                if isinstance(error, (TimeoutError, InterruptedError, MemoryError, KeyboardInterrupt)):
                    raise
            finally:
                trajectories[label] = np.stack(state_list)
                save_checkpoint(final=True)
                recorder.event("seed_finished", **row)

        if all(row["status"] == "passed" for row in rows):
            summary["separation"] = separation_gate(trajectories["ad"][-PERIOD:], trajectories["td"][-PERIOD:])
        else:
            summary["separation"] = {"passed": False, "reason": "both primitive period-six gates must pass first"}
        summary["status"] = "passed" if (all(row["status"] == "passed" for row in rows)
                                           and summary["separation"]["passed"]) else "failed"
    except BaseException as error:
        summary.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                       status="incomplete" if isinstance(error, (TimeoutError, InterruptedError, MemoryError, KeyboardInterrupt))
                       else "failed")
        recorder.event("attempt_stopped", status=summary["status"], error=summary["error"])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        summary.update(dtype_observations=observations, model_calls=budget.calls,
                       elapsed_seconds=time.monotonic() - recorder.started, peak_rss_bytes=reference.rss_bytes())
        recorder.json("results.json", summary)
        recorder.event("finished", status=summary["status"], model_calls=budget.calls)
        recorder.stream.close()
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
