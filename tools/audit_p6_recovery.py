#!/usr/bin/env python3
"""Independent NumPy audit of locally returned P6 recovery arrays; no model calls.

Partial mode audits the closed shard receipts present when the snapshot starts.
Missing trials and inputs remain explicit; a partial snapshot never qualifies
the complete run or extrapolates its outcomes. Audit success means consistency
of evidence, not that every perturbation returned to its source orbit.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
import traceback

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np

HOLD = {"alpha": 0.07293697809562831, "log10_temperature": -0.15395151820007266}
LABELS = ("ad", "td")
NTRIALS, STEPS, PERIOD, DIM = 26, 120, 6, 896
CLOSURE, DIVISOR_GAP, MATCH = 1e-8, 1e-4, 1e-6


def file_record(path, expected=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing, non-file or symbolic-link input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    record = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    if expected and (record["sha256"] != expected["sha256"]
                     or ("bytes" in expected and record["bytes"] != expected["bytes"])):
        raise ValueError(f"file hash/size mismatch: {path}; observed {record}")
    return record


def expected_trial(trial_id):
    if not 0 <= trial_id < NTRIALS:
        raise ValueError("trial ID outside the fixed schedule")
    if trial_id < 2:
        source, phase, sign, kind = trial_id, 0, 0, "control"
    else:
        offset = trial_id - 2
        source, phase = (offset % 4) // 2, offset // 4
        sign, kind = (-1 if offset % 2 == 0 else 1), "perturbed"
    return {"trial_id": trial_id, "source_index": source, "source_label": LABELS[source],
            "phase": phase, "sign": sign, "kind": kind}


def distances(a, b):
    difference = a - b
    return np.sqrt(np.einsum("ij,ij->i", difference, difference))


def recompute(trajectory, refs, source):
    """Independent lag-six, divisor and all-shift calculations, no producer helpers."""
    states, refs = np.asarray(trajectory), np.asarray(refs)
    result = {"classification": "unresolved", "cycle_gate_passed": False, "reference_matches": []}
    if (states.ndim != 2 or states.shape[1:] != (DIM,) or len(states) < 12
            or refs.shape != (2, 6, DIM) or source not in (0, 1)
            or not np.isfinite(states).all() or not np.isfinite(refs).all()):
        return {**result, "reason": "invalid or insufficient states/references"}
    cycle = states[-6:]
    lag = distances(cycle, states[-12:-6])
    divisor_rows = []
    for divisor in (1, 2, 3):
        gaps = distances(cycle, cycle[(np.arange(6) + divisor) % 6])
        divisor_rows.append({"divisor": divisor, "phase_l2_gaps": gaps.tolist(),
                             "maximum_l2_gap": float(gaps.max()), "excluded": bool(gaps.max() > DIVISOR_GAP)})
    comparisons = []
    for label, ref in zip(LABELS, refs):
        shifts = []
        for shift in range(6):
            gaps = distances(cycle, ref[(np.arange(6) + shift) % 6])
            shifts.append({"shift": shift, "phase_l2_distances": gaps.tolist(),
                           "maximum_l2_distance": float(gaps.max())})
        best = min(shifts, key=lambda row: row["maximum_l2_distance"])
        comparisons.append({"label": label, "distance": best["maximum_l2_distance"],
                            "best_shift": best["shift"], "all_shifts": shifts})
    if not all(math.isfinite(row["distance"]) for row in comparisons) or not np.isfinite(lag).all():
        return {**result, "reason": "non-finite computed distances"}
    matches = [row["label"] for row in comparisons if row["distance"] <= MATCH]
    closed = bool(lag.max() <= CLOSURE and all(row["excluded"] for row in divisor_rows))
    classification = "unresolved"
    if closed and len(matches) == 1:
        classification = "returned" if matches[0] == LABELS[source] else "switched"
    return {"classification": classification, "cycle_gate_passed": closed,
            "lag_six_phase_l2_residuals": lag.tolist(), "maximum_l2_residual": float(lag.max()),
            "proper_divisors": divisor_rows, "reference_matches": matches, "reference_distances": comparisons,
            "closure_tolerance": CLOSURE, "divisor_separation": DIVISOR_GAP, "match_tolerance": MATCH}


def load_trajectory(path, steps, first):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("states", "token_ids", "step_seconds", "first_state_index",
                                            "alpha", "log10_temperature")}
    if (data["states"].shape != (steps + 1, DIM) or data["states"].dtype != np.float64
            or not np.isfinite(data["states"]).all() or data["token_ids"].shape != (steps,)
            or data["token_ids"].dtype.kind not in "iu" or data["step_seconds"].shape != (steps,)
            or not np.isfinite(data["step_seconds"]).all() or np.any(data["step_seconds"] < 0)
            or data["first_state_index"].shape != () or data["first_state_index"].dtype.kind not in "iu"
            or int(data["first_state_index"]) != first
            or any(data[key].shape != () for key in HOLD)
            or any(float(data[key]) != value for key, value in HOLD.items())):
        raise ValueError(f"invalid trajectory schema, controls or non-finite values: {path}")
    return data


def check_context(base, refs, references_hash):
    bindings = []
    plan_path, inputs_path, metadata_path = (base / name for name in
                                           ("trial-plan.json", "fixed-trial-inputs.npz", "metadata.json"))
    bindings.append(file_record(plan_path))
    plan = json.loads(plan_path.read_text())
    bindings.append(file_record(inputs_path, plan["inputs"]))
    bindings.append(file_record(metadata_path))
    if (base / "attempt.json").exists():
        bindings.append(file_record(base / "attempt.json"))
    metadata = json.loads(metadata_path.read_text())
    if (plan["hold"] != HOLD or plan["steps_per_trial"] != STEPS or plan["match_tolerance"] != MATCH
            or plan["generator"] != "numpy.PCG64" or plan["generator_seed"] != 20261025
            or metadata["hold"] != HOLD
            or metadata["qualified_inputs"]["references"]["sha256"] != references_hash):
        raise ValueError("plan/metadata does not bind this recovery design and reference pair")
    planned = plan["trials"]
    if len(planned) != NTRIALS:
        raise ValueError("plan does not contain all 26 cases")
    for i, row in enumerate(planned):
        if any(row.get(key) != value for key, value in expected_trial(i).items()):
            raise ValueError(f"plan differs from fixed schedule at trial {i}")
    with np.load(inputs_path, allow_pickle=False) as archive:
        inputs, direction, input_refs = (archive[key] for key in ("initial_states", "direction", "references"))
        epsilon, rms = float(archive["epsilon"]), float(archive["rms_state_norm"])
        if any(float(archive[key]) != value for key, value in HOLD.items()):
            raise ValueError("fixed inputs carry different hold controls")
    if (inputs.shape != (NTRIALS, DIM) or inputs.dtype != np.float64 or not np.isfinite(inputs).all()
            or direction.shape != (DIM,) or direction.dtype != np.float64 or not np.isfinite(direction).all()
            or not np.array_equal(input_refs, refs) or not math.isfinite(epsilon) or epsilon <= 0
            or not math.isclose(float(direction @ direction), 1, rel_tol=0, abs_tol=1e-14)):
        raise ValueError("invalid fixed input arrays, reference identity or normalized direction")
    generated_direction = np.random.Generator(np.random.PCG64(20261025)).normal(size=DIM)
    generated_direction /= np.linalg.norm(generated_direction)
    # Allow only a few float64 rounding units in the normalization reduction.
    if not np.allclose(direction, generated_direction, rtol=2e-15, atol=0):
        raise ValueError("saved perturbation direction differs from PCG64(20261025)")
    expected_rms = float(np.sqrt(np.einsum("ijk,ijk->", refs, refs) / 12))
    if (not math.isclose(rms, expected_rms, rel_tol=1e-14, abs_tol=0)
            or not math.isclose(epsilon, 0.001 * expected_rms, rel_tol=1e-14, abs_tol=0)
            or plan["epsilon"] != epsilon or plan["rms_state_norm"] != rms):
        raise ValueError("perturbation amplitude disagrees with the declared RMS rule")
    for i, row in enumerate(planned):
        expected = refs[row["source_index"], row["phase"]] + row["sign"] * epsilon * direction
        if not np.array_equal(inputs[i], expected):
            raise ValueError(f"fixed input does not equal its declared perturbation: trial {i}")
    return inputs, {"status": "verified", "bindings": bindings, "epsilon": epsilon,
                    "direction_norm": float(np.linalg.norm(direction)),
                    "direction_regeneration": {"rtol": 2e-15, "atol": 0,
                                               "maximum_absolute_difference": float(np.max(np.abs(direction - generated_direction)))},
                    "source_commit": metadata.get("source_commit")}


def audit_shard(base, trial_id, refs, inputs, bindings=None):
    directory = base / "return-shards" / f"trial-{trial_id:02d}"
    if bindings is None:
        bindings = []
    bindings.append(file_record(directory / "receipt.json"))
    receipt = json.loads((directory / "receipt.json").read_text())
    if receipt.get("status") != "verified" or receipt.get("trial_id") != trial_id:
        raise ValueError("invalid collector receipt")
    bindings.append(file_record(directory / "manifest.json", receipt["manifest"]))
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["trial_id"] != trial_id or manifest["files"] != receipt["files"]
            or manifest["archive"] != receipt["archive"]):
        raise ValueError("shard manifest and receipt disagree")
    bindings.append(file_record(directory / "archive.tar", manifest["archive"]))
    prefix = f"trial-{trial_id:02d}"
    expected_names = {f"{prefix}-block-{end:03d}.npz" for end in range(12, 121, 12)} | {f"{prefix}-final-120.npz"}
    records = {record["path"]: record for record in manifest["files"]}
    if len(manifest["files"]) != 11 or set(records) != expected_names:
        raise ValueError("shard lacks exactly ten blocks and one final trajectory")
    with tarfile.open(directory / "archive.tar", "r:") as archive:
        members = archive.getmembers()
        if len(members) != 11 or {member.name for member in members} != expected_names:
            raise ValueError("archive inventory differs from the shard manifest")
        for member in members:
            record = records[member.name]
            if not member.isfile() or member.size != record["bytes"]:
                raise ValueError("archive member type/size mismatch")
            with archive.extractfile(member) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != record["sha256"]:
                raise ValueError("archive member hash mismatch")
    for record in manifest["files"]:
        bindings.append(file_record(base / record["path"], record))
    event = manifest["trial_finished"]
    if (any(event.get(key) != value for key, value in expected_trial(trial_id).items())
            or event.get("status") != "completed" or event.get("completed_steps") != STEPS
            or len(event.get("checkpoints", [])) != 11
            or {row["path"]: row for row in event["checkpoints"]} != records
            or event.get("trajectory") != records[f"{prefix}-final-120.npz"]):
        raise ValueError("closed-trial event is incomplete or disagrees with its inventory/schedule")
    final = load_trajectory(base / f"{prefix}-final-120.npz", STEPS, 0)
    for end in range(12, 121, 12):
        first = end - 12
        block = load_trajectory(base / f"{prefix}-block-{end:03d}.npz", 12, first)
        if (not np.array_equal(block["states"], final["states"][first:end + 1])
                or not np.array_equal(block["token_ids"], final["token_ids"][first:end])
                or not np.array_equal(block["step_seconds"], final["step_seconds"][first:end])):
            raise ValueError(f"block/final mismatch at trial {trial_id}, endpoint {end}")
    if inputs is not None and not np.array_equal(final["states"][0], inputs[trial_id]):
        raise ValueError("trajectory initial state differs from fixed trial inputs")
    outcome = recompute(final["states"][1:], refs, expected_trial(trial_id)["source_index"])
    reported = event.get("outcome", {})
    agreement = (outcome["classification"] == reported.get("classification")
                 and outcome["cycle_gate_passed"] == reported.get("cycle_gate", {}).get("passed")
                 and outcome["reference_matches"] == reported.get("reference_matches")
                 and math.isclose(outcome["maximum_l2_residual"],
                                  reported.get("cycle_gate", {}).get("maximum_l2_residual", float("nan")),
                                  rel_tol=1e-12, abs_tol=1e-18))
    for comparison in outcome["reference_distances"]:
        old = [row for row in reported.get("reference_distances", []) if row.get("label") == comparison["label"]]
        agreement &= (len(old) == 1 and math.isclose(comparison["distance"], old[0].get("distance", float("nan")),
                                                   rel_tol=1e-12, abs_tol=1e-18))
    return {**expected_trial(trial_id), "status": "verified" if agreement and inputs is not None else "unresolved",
            "integrity_passed": True, "blocks_equal_final": True, "initial_state_verified": inputs is not None,
            "producer_agreement": bool(agreement), "outcome": outcome, "bindings": bindings,
            "producer_trial": event}


def audit(run_dir, references, references_sha256, partial=False):
    base = Path(run_dir)
    report = {"mode": "partial" if partial else "complete", "status": "failed", "rows": [],
              "scope": "saved-array consistency only; no independent model evaluation or extrapolation",
              "reference_binding": file_record(references, {"sha256": references_sha256})}
    with np.load(references, allow_pickle=False) as archive:
        refs = archive["states"]
    if refs.shape != (2, 6, DIM) or refs.dtype != np.float64 or not np.isfinite(refs).all():
        raise ValueError("references must be finite float64 of shape (2,6,896)")
    closed = [i for i in range(NTRIALS) if (base / "return-shards" / f"trial-{i:02d}" / "receipt.json").exists()]
    report["closed_trials_at_snapshot_start"] = closed
    report["missing_trials"] = [i for i in range(NTRIALS) if i not in closed]
    inputs = None
    try:
        inputs, report["input_context"] = check_context(base, refs, references_sha256)
    except Exception as error:
        report["input_context"] = {"status": "unresolved", "error": f"{type(error).__name__}: {error}"}
    for trial_id in range(NTRIALS):
        if trial_id not in closed:
            report["rows"].append({**expected_trial(trial_id), "status": "missing", "outcome": {"classification": "not_checked"}})
            continue
        bindings = []
        try:
            row = audit_shard(base, trial_id, refs, inputs, bindings)
        except Exception as error:
            row = {**expected_trial(trial_id), "status": "failed", "integrity_passed": False,
                   "error": f"{type(error).__name__}: {error}", "bindings": bindings,
                   "outcome": {"classification": "not_checked"}}
        report["rows"].append(row)
    verified = all(report["rows"][i]["status"] == "verified" for i in closed)
    report["snapshot_audit_passed"] = bool(closed) and verified and report["input_context"]["status"] == "verified"
    report["final_results"] = {"status": "not_required_for_partial_snapshot" if partial else "unresolved"}
    if not partial:
        try:
            binding = file_record(base / "results.json")
            results = json.loads((base / "results.json").read_text())
            if (len(results.get("rows", [])) != NTRIALS
                    or [row.get("trial_id") for row in results["rows"]] != list(range(NTRIALS))):
                raise ValueError("final results do not contain the fixed 26 rows")
            for row in results["rows"]:
                saved = report["rows"][row["trial_id"]].get("producer_trial", {})
                if any(saved.get(key) != value for key, value in row.items()):
                    raise ValueError(f"final row and collected closed-trial event differ: {row['trial_id']}")
            report["final_results"] = {"status": "verified", "binding": binding, "reported_status": results.get("status")}
        except Exception as error:
            report["final_results"] = {"status": "unresolved", "error": f"{type(error).__name__}: {error}"}
    report["complete_audit_passed"] = (not partial and report["snapshot_audit_passed"] and not report["missing_trials"]
                                       and report["final_results"]["status"] == "verified")
    controls_returned = all(report["rows"][i]["status"] == "verified"
                            and report["rows"][i]["outcome"]["classification"] == "returned" for i in (0, 1))
    counts = {label: sum(row["kind"] == "perturbed" and row["status"] == "verified"
                         and row["outcome"]["classification"] == label for row in report["rows"])
              for label in ("returned", "switched", "unresolved")}
    report["measurements"] = {"controls_returned": controls_returned, "audited_perturbed_counts": counts,
                              "all_24_perturbations_returned": report["complete_audit_passed"] and controls_returned
                              and counts["returned"] == 24}
    if not partial and report["complete_audit_passed"]:
        expected_summary = {"controls_passed": controls_returned, "recovery_interpretation_valid": controls_returned,
                            "all_trials_completed": True, "perturbed_trial_counts": counts,
                            "all_tested_perturbations_returned": controls_returned and counts["returned"] == 24}
        expected_status = "passed" if expected_summary["all_tested_perturbations_returned"] else "failed"
        aggregate_agreement = results.get("interpretation") == expected_summary and results.get("status") == expected_status
        report["final_results"]["aggregate_agreement"] = aggregate_agreement
        report["complete_audit_passed"] &= aggregate_agreement
        report["measurements"]["all_24_perturbations_returned"] &= aggregate_agreement
    report["status"] = ("partial_audit_passed" if report["snapshot_audit_passed"] else "partial_audit_unresolved") if partial else (
        "complete_audit_passed" if report["complete_audit_passed"] else "complete_audit_failed")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--references-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--partial", action="store_true")
    args = p.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root):
        p.error("audit outputs must be outside the repository")
    if len(args.references_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.references_sha256):
        p.error("reference SHA-256 must be a lowercase digest")
    receipt = {"utc": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
               "source_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
               "script": file_record(__file__), "numpy_version": np.__version__,
               "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    # Reserve a fresh output before auditing; never overwrite an earlier snapshot.
    with args.output.open("x") as output:
        try:
            receipt["audit"] = audit(args.run_dir, args.references, args.references_sha256, args.partial)
        except Exception as error:
            receipt["audit"] = {"status": "failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        json.dump(receipt, output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps({"output": str(args.output), "status": receipt["audit"]["status"]}))
    return 0 if receipt["audit"]["status"] in ("partial_audit_passed", "complete_audit_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
