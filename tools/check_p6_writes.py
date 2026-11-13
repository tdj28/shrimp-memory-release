#!/usr/bin/env python3
"""Fixed isolated writes or six two-write round trips on the period-six pair.

Each trial starts at reference phase zero, evolves for 120 steps at its proposed
endpoint, then 240 at the original hold. This is exploratory: it does not prove
hysteresis, phase-independent writing, basin geometry, language control or logic.
Offline CPU native float64, one thread, no extensions or replacement trials.
The roundtrip protocol starts at all six ad phases and writes td then ad with
actual state continuity (4392 calls, 2400-second internal cap). Use an external
2460-second timeout for roundtrip, 900 seconds for the default isolated test.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import check_p6_hold as hold_test
import check_p6_recovery as recovery
import validate_qwen_map as reference

HOLD = dict(recovery.HOLD)
ENDPOINTS = {
    "ad": {"alpha": HOLD["alpha"], "log10_temperature": -0.15350},
    "td": {"alpha": 0.07299785589927693, "log10_temperature": HOLD["log10_temperature"]},
}
ENDPOINT_STEPS, HOLD_STEPS = 120, 240
MAX_CALLS, MAX_SECONDS, MAX_RSS_GIB = 1464, 840, 20
CACHE_TOLERANCE = 1e-10
ROUNDTRIP_CALLS, ROUNDTRIP_SECONDS = 4392, 2400
SEQUENCE_CALLS, SEQUENCE_SECONDS = 2928, 1600


def trial_plan():
    rows = []
    for desired in recovery.LABELS:
        for source in recovery.LABELS:
            rows.append({"trial_id": len(rows), "source_label": source,
                         "source_index": recovery.LABELS.index(source), "phase": 0,
                         "desired_label": desired, "status": "not_started", "completed_steps": 0,
                         "stages": [{"name": "endpoint", "controls": dict(ENDPOINTS[desired]),
                                     "steps": ENDPOINT_STEPS, "status": "not_started", "completed_steps": 0},
                                    {"name": "hold", "controls": dict(HOLD), "steps": HOLD_STEPS,
                                     "status": "not_started", "completed_steps": 0}]})
    return rows


def roundtrip_plan():
    """Six ad phases, each requesting td then ad without replacing the state."""
    rows = []
    for phase in range(6):
        for pulse, desired in enumerate(("td", "ad")):
            row = trial_plan()[2 if pulse == 0 else 1]
            row.update(trial_id=len(rows), phase=phase, chain_id=phase, pulse_index=pulse,
                       source_label="ad" if pulse == 0 else "carried",
                       source_index=0 if pulse == 0 else None,
                       predecessor_trial_id=None if pulse == 0 else len(rows) - 1)
            rows.append(row)
    return rows


def sequence_plan():
    """One continuous register, ad=0 and td=1, writing the requested byte."""
    rows = []
    for i, bit in enumerate("10110001"):
        row = trial_plan()[2 if bit == "1" else 0]
        row.update(trial_id=i, phase=0, chain_id=0, pulse_index=i,
                   source_label="ad" if i == 0 else "carried",
                   source_index=0 if i == 0 else None,
                   predecessor_trial_id=None if i == 0 else i - 1)
        rows.append(row)
    return rows


def initial_for_row(row, refs, carried):
    """Never substitute a reference when a predecessor is missing or unsuccessful."""
    predecessor = row.get("predecessor_trial_id")
    if predecessor is not None:
        if predecessor not in carried:
            raise ValueError("missing predecessor endpoint; cannot reset the state")
        return carried[predecessor].copy()
    return refs[row["source_index"], row["phase"]].copy()


def classify_write(trajectory, refs, desired):
    if desired not in recovery.LABELS:
        raise ValueError("desired label must be ad or td")
    result = recovery.classify_recovery(trajectory, refs, recovery.LABELS.index(desired))
    result["classification"] = {"returned": "correct", "switched": "wrong", "unresolved": "unresolved"}[
        result["classification"]]
    return {**result, "desired_label": desired}


def require_returning_controls(results):
    """The two controls must return; measured perturbation failures are allowed."""
    controls = [row for row in results.get("rows", []) if row.get("kind") == "control"]
    if (len(controls) != 2 or {row.get("source_label") for row in controls} != set(recovery.LABELS)
            or any(row.get("status") != "completed" or row.get("completed_steps") != recovery.STEPS
                   or row.get("outcome", {}).get("classification") != "returned" for row in controls)):
        raise ValueError("recovery results must contain both completed, returning unperturbed controls")
    return controls


def qualify_recovery(args, refs):
    import numpy as np

    provenance = {"results": reference.require_hash(args.recovery_results, args.recovery_results_sha256)}
    controls = require_returning_controls(json.loads(args.recovery_results.read_text()))
    metadata_path = args.recovery_results.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if (metadata["hold"] != HOLD or metadata["model"] != reference.MODEL
            or metadata["revision"] != reference.REVISION or metadata["prefix_ids"] != reference.PREFIX_IDS
            or metadata["weights"]["sha256"] != reference.WEIGHTS_SHA256
            or metadata["vectors"]["sha256"] != reference.VECTOR_SHA256
            or metadata["config"]["sha256"] != hold_test.CONFIG_SHA256
            or metadata["qualified_inputs"]["references"]["sha256"] != args.references_sha256):
        raise ValueError("recovery controls were not run with the same map and reference cycles")
    provenance["metadata"] = reference.require_hash(metadata_path, reference.sha256(metadata_path))
    provenance["control_trajectories"] = []
    for row in controls:
        record = row["trajectory"]
        relative = Path(record["path"])
        if relative.is_absolute() or relative.name != str(relative):
            raise ValueError("recovery control trajectory must be a colocated filename")
        path = args.recovery_results.parent / relative
        provenance["control_trajectories"].append(reference.require_hash(path, record["sha256"]))
        with np.load(path, allow_pickle=False) as archive:
            states = archive["states"]
            if float(archive["alpha"]) != HOLD["alpha"] or float(archive["log10_temperature"]) != HOLD["log10_temperature"]:
                raise ValueError("recovery control used different hold settings")
        source = recovery.LABELS.index(row["source_label"])
        if (states.shape != (recovery.STEPS + 1, 896) or states.dtype != np.float64
                or not np.array_equal(states[0], refs[source, 0])
                or recovery.classify_recovery(states[1:], refs, source)["classification"] != "returned"):
            raise ValueError("saved recovery control does not independently qualify as returned")
    return provenance


def set_controls(loop, vector, controls):
    """Update the existing hook's intervention and rebuild even for T-only changes."""
    loop.steering = controls["alpha"] * vector.reshape(1, 1, -1)
    loop.temperature = 10.0 ** controls["log10_temperature"]
    loop.prepare()


class StageArchive:
    def __init__(self, recorder, trial_id, stage, initial):
        self.recorder, self.trial_id, self.stage = recorder, trial_id, stage
        self.states, self.tokens, self.seconds, self.first = [initial.copy()], [], [], 0

    def save(self, final=False):
        import numpy as np

        completed = len(self.states) - 1
        first = 0 if final else self.first
        name = f"trial-{self.trial_id:02d}-{self.stage['name']}-{'final' if final else 'block'}-{completed:03d}.npz"
        path = self.recorder.directory / name
        with path.open("xb") as stream:
            np.savez_compressed(stream, states=np.stack(self.states[first:]), first_state_index=first,
                                token_ids=np.asarray(self.tokens[first:], dtype=np.int64),
                                step_seconds=np.asarray(self.seconds[first:]), **self.stage["controls"])
            stream.flush()
            os.fsync(stream.fileno())
        record = {"path": name, "bytes": path.stat().st_size, "sha256": reference.sha256(path)}
        self.stage.setdefault("checkpoints", []).append(record)
        if final:
            self.stage["trajectory"] = record
        self.first = completed
        self.recorder.event("checkpoint", trial_id=self.trial_id, stage=self.stage["name"],
                            completed_steps=completed, final=final, **record)


def check_stage_cache(loop, x, recorder, budget, label, controls):
    """Save both one-step outputs without advancing the stage's initial state."""
    import numpy as np
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    before = x.clone()
    outputs, artifacts = [], []
    for cached in (True, False):
        name = f"{label}-{'cached' if cached else 'full'}.npz"
        budget.claim(name)
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            y, probabilities = loop.forward(x, cached=cached)
        path = recorder.directory / name
        with path.open("xb") as stream:
            np.savez_compressed(stream, input=x.detach().numpy(), output=y.detach().numpy(),
                                probabilities=probabilities.detach().numpy(), **controls)
            stream.flush()
            os.fsync(stream.fileno())
        artifacts.append({"path": name, "bytes": path.stat().st_size, "sha256": reference.sha256(path)})
        recorder.event("cache_comparison_output", label=label, cached=cached, **artifacts[-1])
        if not bool(torch.isfinite(y).all()) or not bool(torch.isfinite(probabilities).all()):
            raise ValueError("non-finite comparison output; arrays preserved")
        outputs.append(y)
        if reference.rss_bytes() >= budget.max_rss:
            raise MemoryError("process peak RSS exceeds declared cap")
    gap = float((outputs[0] - outputs[1]).abs().max())
    unchanged = bool(torch.equal(x, before))
    result = {"passed": gap <= CACHE_TOLERANCE and unchanged, "maximum_coordinate_gap": gap,
              "tolerance": CACHE_TOLERANCE, "input_unchanged": unchanged, "artifacts": artifacts}
    recorder.event("cache_comparison", label=label, **result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", choices=("isolated", "roundtrip", "sequence"), default="isolated")
    for name in ("cache-dir", "vectors", "references", "reference-manifest", "source-hold-results",
                 "recovery-results", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("references-sha256", "reference-manifest-sha256", "recovery-results-sha256"):
        p.add_argument("--" + name, required=True)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    for digest in (args.references_sha256, args.reference_manifest_sha256, args.recovery_results_sha256):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            p.error("input hashes must be lowercase SHA-256 digests")
    recorder = reference.Recorder(args.output_dir)
    roundtrip = args.protocol == "roundtrip"
    max_calls, max_seconds = {"isolated": (MAX_CALLS, MAX_SECONDS),
                              "roundtrip": (ROUNDTRIP_CALLS, ROUNDTRIP_SECONDS),
                              "sequence": (SEQUENCE_CALLS, SEQUENCE_SECONDS)}[args.protocol]
    budget = reference.Budget(recorder, max_calls, max_seconds, MAX_RSS_GIB)
    rows, observations = {"isolated": trial_plan, "roundtrip": roundtrip_plan,
                          "sequence": sequence_plan}[args.protocol](), {}
    summary = {"status": "failed", "evidence": "exploratory finite-control-pulse observation",
               "protocol": args.protocol,
               "scope": ("one fixed 10110001 register sequence; not separate-instance communication or language encoding"
                         if args.protocol == "sequence" else "six ad-phase round trips, td then ad, with carried states; not arbitrary sequences, "
                         "general reliability, language control or logic" if roundtrip else
                         "four phase-zero trials; not proven hysteresis, general writing, language control or logic"),
               "rows": rows}
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}

    def stop(signum, frame):
        if signum == signal.SIGALRM:
            raise TimeoutError("declared wall-time limit reached")
        raise InterruptedError(f"received signal {signum}")

    for sig in old_handlers:
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, max_seconds)
    try:
        root = Path(__file__).resolve().parents[1]
        execution = {"utc": reference.utc_now(), "arguments": {k: str(v) for k, v in vars(args).items()},
                     "source_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                     "source_hashes": {Path(path).name: reference.sha256(path) for path in
                                       (__file__, reference.__file__, hold_test.__file__, recovery.__file__)},
                     "max_model_calls": max_calls, "max_seconds": max_seconds, "max_rss_gib": MAX_RSS_GIB,
                     "threads": 1, "checkpoint_every_steps": 12}
        recorder.json("attempt.json", execution)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "1"
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        refs, provenance = recovery.qualify_references(args)
        recovery_provenance = qualify_recovery(args, refs)
        design_path = root / "docs/receipts/p6-write-path-design.json"
        design = json.loads(design_path.read_text())
        if design["hold"] != HOLD or design["candidate_endpoints"] != ENDPOINTS:
            raise ValueError("committed geometric design and implemented controls disagree")
        import numpy as np

        roots = [row for row in rows if row.get("predecessor_trial_id") is None]
        initial_states = np.stack([initial_for_row(row, refs, {}) for row in roots])
        input_path = recorder.directory / "fixed-trial-inputs.npz"
        with input_path.open("xb") as stream:
            np.savez_compressed(stream, references=refs, initial_states=initial_states)
            stream.flush()
            os.fsync(stream.fileno())
        recorder.json("trial-plan.json", {"trials": rows, "hold": HOLD, "candidate_endpoints": ENDPOINTS,
                      "cache_tolerance": CACHE_TOLERANCE, "match_tolerance": recovery.MATCH_TOLERANCE,
                      "inputs": reference.require_hash(input_path, reference.sha256(input_path))})
        snapshot = args.cache_dir / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / reference.REVISION
        recorder.json("metadata.json", {**execution, "model": reference.MODEL, "revision": reference.REVISION,
                      "hold": HOLD, "candidate_endpoints": ENDPOINTS, "prefix_ids": reference.PREFIX_IDS,
                      "qualified_inputs": provenance, "recovery_prerequisite": recovery_provenance,
                      "geometric_design": reference.require_hash(design_path, reference.sha256(design_path)),
                      "weights": reference.require_hash(snapshot / "model.safetensors", reference.WEIGHTS_SHA256),
                      "config": reference.require_hash(snapshot / "config.json", hold_test.CONFIG_SHA256),
                      "vectors": reference.require_hash(args.vectors, reference.VECTOR_SHA256)})
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        recorder.event("environment", packages={name: importlib.metadata.version(name) for name in
                       ("numpy", "torch", "transformers", "safetensors")})
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        ids = tokenizer.apply_chat_template(reference.MESSAGES, add_generation_prompt=True)
        if list(ids["input_ids"] if hasattr(ids, "keys") else ids) != reference.PREFIX_IDS:
            raise ValueError("tokenizer prefix differs from the fixed IDs")
        with np.load(args.vectors, allow_pickle=False) as archive:
            vector = archive["philosophy"]
        if vector.shape != (896,) or not np.isfinite(vector).all():
            raise ValueError("invalid steering vector")
        vector = torch.tensor(vector, dtype=torch.float64)
        recorder.event("model_loading")
        model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                    dtype=torch.float64, attn_implementation="sdpa").cpu().eval().requires_grad_(False)
        cfg = model.config
        if (cfg.model_type, cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size) != ("qwen2", 896, 24, 151936):
            raise ValueError("unexpected model architecture")
        if any(t.device.type != "cpu" or t.dtype != torch.float64 for t in model.parameters()):
            raise ValueError("model parameters must all be CPU float64")
        patches = reference.patch_native_float64(model, observations)
        recorder.event("model_loaded", patches=patches, attention=cfg._attn_implementation)
        # One hook for the entire run: only its stored steering and temperature change.
        loop = reference.FeedbackMap(model, vector, HOLD["alpha"], HOLD["log10_temperature"], budget, recorder)
        loop.mode = "native64"
        carried = {}
        for row in rows:
            row["status"] = "running"
            x = torch.tensor(initial_for_row(row, refs, carried), dtype=torch.float64)
            try:
                for stage in row["stages"]:
                    archive = StageArchive(recorder, row["trial_id"], stage, x.detach().numpy())
                    stage["status"] = "running"
                    try:
                        set_controls(loop, vector, stage["controls"])
                        stage["cache_comparison"] = check_stage_cache(loop, x, recorder, budget,
                            f"trial-{row['trial_id']:02d}-{stage['name']}", stage["controls"])
                        if not stage["cache_comparison"]["passed"]:
                            raise ValueError("cache/full-prefix implementation comparison failed")
                        for step in range(1, stage["steps"] + 1):
                            budget.claim(f"trial_{row['trial_id']:02d}_{stage['name']}_{step:03d}")
                            started = time.monotonic()
                            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                                y, probabilities = loop.forward(x, cached=True)
                            archive.states.append(y.detach().numpy().copy())
                            archive.seconds.append(time.monotonic() - started)
                            archive.tokens.append(int(probabilities.argmax()) if bool(torch.isfinite(probabilities).all()) else -1)
                            stage["completed_steps"] = step
                            row["completed_steps"] += 1
                            if not np.isfinite(archive.states[-1]).all() or archive.tokens[-1] < 0:
                                raise ValueError("non-finite map output; attempted output preserved")
                            if reference.rss_bytes() >= budget.max_rss:
                                raise MemoryError("process peak RSS exceeds declared cap")
                            x = y
                            if step % 12 == 0:
                                archive.save()
                        stage["status"] = "completed"
                    except BaseException as error:
                        stage.update(status="incomplete" if isinstance(error, (TimeoutError, InterruptedError, MemoryError, KeyboardInterrupt))
                                     else "failed", error=f"{type(error).__name__}: {error}")
                        raise
                    finally:
                        archive.save(final=True)
                        recorder.event("stage_finished", trial_id=row["trial_id"], **stage)
                row.update(status="completed", outcome=classify_write(archive.states[1:], refs, row["desired_label"]))
                # Continue even after a wrong or unresolved classification, preserving
                # the actual last state. Only execution errors stop the attempt.
                carried[row["trial_id"]] = x.detach().numpy().copy()
            except BaseException as error:
                row.update(status="incomplete" if isinstance(error, (TimeoutError, InterruptedError, MemoryError, KeyboardInterrupt))
                           else "failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                raise
            finally:
                recorder.event("trial_finished", **row)
        summary["status"] = "passed" if all(row["outcome"]["classification"] == "correct" for row in rows) else "failed"
    except BaseException as error:
        summary.update(status="incomplete" if isinstance(error, (TimeoutError, InterruptedError, MemoryError, KeyboardInterrupt))
                       else "failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        recorder.event("attempt_stopped", status=summary["status"], error=summary["error"])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        summary.update(dtype_observations=observations, model_calls=budget.calls,
                       elapsed_seconds=time.monotonic() - recorder.started, peak_rss_bytes=reference.rss_bytes(),
                       outcome_counts={label: sum(row.get("outcome", {}).get("classification") == label for row in rows)
                                       for label in ("correct", "wrong", "unresolved")})
        recorder.json("results.json", summary)
        recorder.event("finished", status=summary["status"], model_calls=budget.calls)
        recorder.stream.close()
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
