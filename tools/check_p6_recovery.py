#!/usr/bin/env python3
"""Fixed finite-perturbation recovery pilot for a qualified period-six pair.

Twenty-six trials, 120 cached steps each, one CPU thread, native float64.
Run under an external 2160-second timeout; the internal cap is 2100 seconds.
This tests one saved direction and one finite amplitude, not basin geometry,
a basin radius, fractality, writable memory, or global numerical precision.
No model downloads, outcome-dependent stopping, or replacement trials.
"""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import check_p6_hold as hold_test
import validate_qwen_map as reference

HOLD = {"alpha": 0.07293697809562831, "log10_temperature": -0.15395151820007266}
LABELS = ("ad", "td")
STEPS = 120
TRIALS = 26
MATCH_TOLERANCE = 1e-6
DIRECTION_SEED = 20261025
MAX_SECONDS = 2100
MAX_RSS_GIB = 20


def classify_recovery(trajectory, references, source_index):
    """Return to an orbit, not a phase; never force an ambiguous nearest label."""
    import numpy as np

    if source_index not in (0, 1):
        raise ValueError("source_index must identify ad or td")
    refs, states = np.asarray(references), np.asarray(trajectory)
    result = {"classification": "unresolved", "reference_matches": [],
              "reference_distances": [], "match_tolerance": MATCH_TOLERANCE}
    if refs.shape != (2, 6, 896) or not np.isfinite(refs).all():
        return {**result, "reason": "invalid reference cycles", "cycle_gate": {"passed": False}}
    gate = hold_test.cycle_gate(states)
    result["cycle_gate"] = gate
    if states.ndim != 2 or states.shape[1:] != (896,) or len(states) < 6 or not np.isfinite(states).all():
        return {**result, "reason": "invalid trajectory"}
    for label, cycle in zip(LABELS, refs):
        distance = hold_test.separation_gate(states[-6:], cycle)
        value = distance["minimum_over_shifts_of_maximum_l2_distance"]
        result["reference_distances"].append({"label": label, "distance": value,
                                                "best_shift": distance["best_shift"]})
        if value <= MATCH_TOLERANCE:
            result["reference_matches"].append(label)
    if not gate["passed"]:
        return {**result, "reason": "primitive period-six closure gate failed"}
    matches = result["reference_matches"]
    if len(matches) != 1:
        return {**result, "reason": "ambiguous reference match" if len(matches) > 1 else "no reference match"}
    result["classification"] = "returned" if matches[0] == LABELS[source_index] else "switched"
    return result


def recovery_interpretation(rows):
    """Failed controls invalidate recovery interpretation without deleting rows."""
    controls = [row for row in rows if row.get("kind") == "control"]
    controls_pass = (len(controls) == 2 and {row.get("source_label") for row in controls} == set(LABELS)
                     and all(row.get("status") == "completed" and row.get("completed_steps") == STEPS
                             and row.get("outcome", {}).get("classification") == "returned"
                             for row in controls))
    complete = (len(rows) == TRIALS and len({row.get("trial_id") for row in rows}) == TRIALS
                and all(row.get("status") == "completed" and row.get("completed_steps") == STEPS for row in rows))
    counts = {label: sum(row.get("outcome", {}).get("classification") == label
                        for row in rows if row.get("kind") == "perturbed")
              for label in ("returned", "switched", "unresolved")}
    return {"controls_passed": controls_pass, "recovery_interpretation_valid": controls_pass and complete,
            "all_trials_completed": complete, "perturbed_trial_counts": counts,
            "all_tested_perturbations_returned": controls_pass and complete and counts["returned"] == 24}


def qualify_references(args):
    """Bind the extracted cycles to passed source results and raw trajectories.

    Manifest schema: hold; output.{sha256,key,shape,dtype,order};
    source_hold_results.sha256; source_hold_metadata.sha256. Source results,
    metadata.json, and the named final trajectories must be colocated.
    """
    import numpy as np

    provenance = {"references": reference.require_hash(args.references, args.references_sha256),
                  "reference_manifest": reference.require_hash(args.reference_manifest, args.reference_manifest_sha256)}
    manifest = json.loads(args.reference_manifest.read_text())
    contract = {"sha256": args.references_sha256, "key": "states", "shape": [2, 6, 896],
                "dtype": "float64", "order": list(LABELS)}
    if manifest["hold"] != HOLD or any(manifest["output"].get(key) != value for key, value in contract.items()):
        raise ValueError("reference manifest has incompatible controls or array contract")
    provenance["source_results"] = reference.require_hash(
        args.source_hold_results, manifest["source_hold_results"]["sha256"])
    source = json.loads(args.source_hold_results.read_text())
    metadata_path = args.source_hold_results.parent / "metadata.json"
    provenance["source_metadata"] = reference.require_hash(metadata_path, manifest["source_hold_metadata"]["sha256"])
    metadata = json.loads(metadata_path.read_text())
    if (metadata["hold"] != HOLD or metadata["model"] != reference.MODEL
            or metadata["revision"] != reference.REVISION or metadata["prefix_ids"] != reference.PREFIX_IDS
            or metadata["weights"]["sha256"] != reference.WEIGHTS_SHA256
            or metadata["vectors"]["sha256"] != reference.VECTOR_SHA256
            or metadata["config"]["sha256"] != hold_test.CONFIG_SHA256):
        raise ValueError("source hold metadata does not match this map")
    source_rows = source.get("rows", [])
    if (source.get("status") != "passed" or source.get("separation", {}).get("passed") is not True
            or len(source_rows) != 2 or [row.get("label") for row in source_rows] != list(LABELS)
            or any(row.get("status") != "passed" or row.get("completed_steps") != STEPS
                   or row.get("cycle_gate", {}).get("passed") is not True for row in source_rows)):
        raise ValueError("source hold results do not qualify both cycles")
    with np.load(args.references, allow_pickle=False) as archive:
        refs = archive["states"]
    if refs.shape != (2, 6, 896) or refs.dtype != np.float64 or not np.isfinite(refs).all():
        raise ValueError("references must be finite float64 with shape (2,6,896)")
    provenance["source_trajectories"] = []
    for index, row in enumerate(source_rows):
        record = row["trajectory"]
        relative = Path(record["path"])
        if relative.is_absolute() or relative.name != str(relative):
            raise ValueError("source trajectory must be a filename within the source results directory")
        path = args.source_hold_results.parent / relative
        provenance["source_trajectories"].append(reference.require_hash(path, record["sha256"]))
        with np.load(path, allow_pickle=False) as archive:
            trajectory = archive["states"]
            if float(archive["alpha"]) != HOLD["alpha"] or float(archive["log10_temperature"]) != HOLD["log10_temperature"]:
                raise ValueError("source trajectory used different hold controls")
        if (trajectory.shape != (STEPS + 1, 896) or trajectory.dtype != np.float64
                or not hold_test.cycle_gate(trajectory[1:])["passed"]
                or not np.array_equal(refs[index], trajectory[-6:])):
            raise ValueError("reference cycle does not equal its qualified source trajectory")
    if not hold_test.separation_gate(*refs)["passed"]:
        raise ValueError("recomputed source cycle separation failed")
    return refs, {"manifest": manifest, **provenance}


def trial_inputs(refs):
    import numpy as np

    direction = np.random.Generator(np.random.PCG64(DIRECTION_SEED)).normal(size=896)
    direction /= np.linalg.norm(direction)
    rms_state_norm = float(np.sqrt(np.mean(np.sum(refs * refs, axis=-1))))
    epsilon = 0.001 * rms_state_norm
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("perturbation amplitude is not finite and positive")
    rows, inputs = [], []

    def add(source, phase, sign, kind):
        rows.append({"trial_id": len(rows), "kind": kind, "source_index": source,
                     "source_label": LABELS[source], "phase": phase, "sign": sign,
                     "status": "not_started", "completed_steps": 0})
        inputs.append(refs[source, phase] + sign * epsilon * direction)

    for source in range(2):
        add(source, 0, 0, "control")
    for phase in range(6):
        for source in range(2):
            for sign in (-1, 1):
                add(source, phase, sign, "perturbed")
    return rows, np.stack(inputs), direction, epsilon, rms_state_norm


class TrialArchive:
    def __init__(self, recorder, row, initial):
        self.recorder, self.row = recorder, row
        self.states, self.tokens, self.seconds, self.first = [initial.copy()], [], [], 0

    def save(self, final=False):
        import numpy as np

        completed = len(self.states) - 1
        name = f"trial-{self.row['trial_id']:02d}-{'final' if final else 'block'}-{completed:03d}.npz"
        first = 0 if final else self.first
        path = self.recorder.directory / name
        with path.open("xb") as stream:
            np.savez_compressed(stream, states=np.stack(self.states[first:]), first_state_index=first,
                                token_ids=np.asarray(self.tokens[first:], dtype=np.int64),
                                step_seconds=np.asarray(self.seconds[first:]), **HOLD)
            stream.flush()
            os.fsync(stream.fileno())
        record = {"path": name, "bytes": path.stat().st_size, "sha256": reference.sha256(path)}
        self.row.setdefault("checkpoints", []).append(record)
        if final:
            self.row["trajectory"] = record
        self.first = completed
        self.recorder.event("checkpoint", trial_id=self.row["trial_id"], completed_steps=completed, final=final, **record)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache-dir", "vectors", "references", "reference-manifest", "source-hold-results", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("references-sha256", "reference-manifest-sha256"):
        p.add_argument("--" + name, required=True)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    for digest in (args.references_sha256, args.reference_manifest_sha256):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            p.error("reference hashes must be lowercase SHA-256 digests")
    recorder = reference.Recorder(args.output_dir)
    budget = reference.Budget(recorder, 1 + TRIALS * STEPS, MAX_SECONDS, MAX_RSS_GIB)
    summary = {"status": "failed", "evidence": "exploratory finite-precision observation",
               "scope": "one finite amplitude and one direction at all phases; no basin or write claim", "rows": []}
    observations = {}
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}

    def stop(signum, frame):
        if signum == signal.SIGALRM:
            raise TimeoutError("declared wall-time limit reached")
        raise InterruptedError(f"received signal {signum}")

    for sig in old_handlers:
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    try:
        execution = {"utc": reference.utc_now(), "arguments": {k: str(v) for k, v in vars(args).items()},
                     "source_commit": subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[1]),
                                                              "rev-parse", "HEAD"], text=True).strip(),
                     "source_hashes": {Path(path).name: reference.sha256(path)
                                       for path in (__file__, reference.__file__, hold_test.__file__)},
                     "max_model_calls": budget.max_calls, "max_seconds": MAX_SECONDS,
                     "max_rss_gib": MAX_RSS_GIB, "threads": 1, "checkpoint_every_steps": 12}
        recorder.json("attempt.json", execution)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "1"
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        refs, provenance = qualify_references(args)
        rows, inputs, direction, epsilon, rms_norm = trial_inputs(refs)
        summary["rows"] = rows
        import numpy as np

        input_path = recorder.directory / "fixed-trial-inputs.npz"
        with input_path.open("xb") as stream:
            np.savez_compressed(stream, references=refs, initial_states=inputs, direction=direction,
                                epsilon=epsilon, rms_state_norm=rms_norm, **HOLD)
            stream.flush()
            os.fsync(stream.fileno())
        recorder.json("trial-plan.json", {"trials": rows, "steps_per_trial": STEPS, "hold": HOLD,
                      "epsilon": epsilon, "epsilon_rule": "0.001 * RMS norm of twelve reference states",
                      "rms_state_norm": rms_norm, "generator": "numpy.PCG64", "generator_seed": DIRECTION_SEED,
                      "direction_norm": float(np.linalg.norm(direction)), "match_tolerance": MATCH_TOLERANCE,
                      "inputs": reference.require_hash(input_path, reference.sha256(input_path))})
        recorder.event("fixed_inputs_saved", trials=TRIALS, steps_per_trial=STEPS, epsilon=epsilon,
                       inputs_sha256=reference.sha256(input_path))
        snapshot = args.cache_dir / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / reference.REVISION
        recorder.json("metadata.json", {**execution,
                      "model": reference.MODEL, "revision": reference.REVISION, "hold": HOLD,
                      "prefix_ids": reference.PREFIX_IDS, "qualified_inputs": provenance,
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
        loop = reference.FeedbackMap(model, torch.tensor(vector, dtype=torch.float64), HOLD["alpha"],
                                     HOLD["log10_temperature"], budget, recorder)
        loop.mode = "native64"
        loop.prepare()
        for row, initial in zip(rows, inputs):
            archive = TrialArchive(recorder, row, initial)
            row["status"] = "running"
            try:
                x = torch.tensor(initial, dtype=torch.float64)
                recorder.event("trial_started", **row)
                for step in range(1, STEPS + 1):
                    budget.claim(f"trial_{row['trial_id']:02d}_step_{step:03d}")
                    started = time.monotonic()
                    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                        y, probabilities = loop.forward(x, cached=True)
                    archive.states.append(y.detach().cpu().numpy().copy())
                    archive.seconds.append(time.monotonic() - started)
                    archive.tokens.append(int(probabilities.argmax()) if bool(torch.isfinite(probabilities).all()) else -1)
                    row["completed_steps"] = step
                    if not np.isfinite(archive.states[-1]).all() or archive.tokens[-1] < 0:
                        raise ValueError("non-finite map output; attempted output preserved")
                    if reference.rss_bytes() >= budget.max_rss:
                        raise MemoryError("process peak RSS exceeds the declared cap")
                    x = y
                    if step % 12 == 0:
                        archive.save()
                        recorder.event("progress", trial_id=row["trial_id"], step=step,
                                       cycle_gate=hold_test.cycle_gate(archive.states[1:]))
                row.update(status="completed", outcome=classify_recovery(archive.states[1:], refs, row["source_index"]))
            except BaseException as error:
                row.update(status="incomplete" if isinstance(error, (TimeoutError, InterruptedError, MemoryError, KeyboardInterrupt))
                           else "failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                raise  # Resource or implementation failures stop; measured recovery failures do not.
            finally:
                archive.save(final=True)
                recorder.event("trial_finished", **row)
        interpretation = recovery_interpretation(rows)
        summary["status"] = "passed" if interpretation["all_tested_perturbations_returned"] else "failed"
    except BaseException as error:
        summary.update(status="incomplete" if isinstance(error, (TimeoutError, InterruptedError, MemoryError, KeyboardInterrupt))
                       else "failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        recorder.event("attempt_stopped", status=summary["status"], error=summary["error"])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        summary.update(interpretation=recovery_interpretation(summary["rows"]), dtype_observations=observations,
                       model_calls=budget.calls, elapsed_seconds=time.monotonic() - recorder.started,
                       peak_rss_bytes=reference.rss_bytes())
        recorder.json("results.json", summary)
        recorder.event("finished", status=summary["status"], model_calls=budget.calls)
        recorder.stream.close()
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
