#!/usr/bin/env python3
"""Audit isolated P6 writes or six round trips with NumPy; no model calls.

An audit passes when all four complete trials and their reported outcomes agree
with the saved evidence. Wrong or unresolved writes remain valid measured
outcomes. Incomplete attempts cannot qualify the four-case result.
The roundtrip option separately checks all twelve pulses and exact inter-write
continuity, including after a wrong or unresolved first write.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import traceback

import audit_p6_recovery as independent

np = independent.np
HOLD = dict(independent.HOLD)
LABELS = ("ad", "td")
ENDPOINTS = {"ad": {"alpha": HOLD["alpha"], "log10_temperature": -0.15350},
             "td": {"alpha": 0.07299785589927693, "log10_temperature": HOLD["log10_temperature"]}}
STAGE_STEPS = {"endpoint": 120, "hold": 240}
CACHE_TOLERANCE, DIM, VOCAB = 1e-10, 896, 151936


def expected_case(trial_id, protocol="isolated"):
    if protocol == "sequence":
        if not 0 <= trial_id < 8:
            raise ValueError("trial ID outside eight-write sequence")
        desired = ("td", "ad", "td", "td", "ad", "ad", "ad", "td")[trial_id]
        return {"trial_id": trial_id, "source_label": "ad" if trial_id == 0 else "carried",
                "source_index": 0 if trial_id == 0 else None, "phase": 0,
                "desired_label": desired, "chain_id": 0, "pulse_index": trial_id,
                "predecessor_trial_id": None if trial_id == 0 else trial_id - 1}
    if protocol == "roundtrip":
        if not 0 <= trial_id < 12:
            raise ValueError("trial ID outside six-round-trip schedule")
        phase, pulse = divmod(trial_id, 2)
        return {"trial_id": trial_id, "source_label": "ad" if pulse == 0 else "carried",
                "source_index": 0 if pulse == 0 else None, "phase": phase,
                "desired_label": "td" if pulse == 0 else "ad", "chain_id": phase,
                "pulse_index": pulse, "predecessor_trial_id": None if pulse == 0 else trial_id - 1}
    if protocol != "isolated":
        raise ValueError("unknown protocol")
    if not 0 <= trial_id < 4:
        raise ValueError("trial ID outside fixed four-case schedule")
    return {"trial_id": trial_id, "source_label": LABELS[trial_id % 2], "source_index": trial_id % 2,
            "phase": 0, "desired_label": LABELS[trial_id // 2]}


def record_local(base, record, bindings):
    name = record["path"]
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("artifact must be a colocated filename")
    binding = independent.file_record(base / name, record)
    bindings.append(binding)
    return base / name


def check_controls(archive, controls):
    if any(archive[key].shape != () or archive[key].dtype != np.float64
           or float(archive[key]) != value for key, value in controls.items()):
        raise ValueError("saved controls differ from the prescribed stage")


def load_trajectory(path, steps, first, controls):
    with np.load(path, allow_pickle=False) as archive:
        check_controls(archive, controls)
        data = {key: archive[key] for key in ("states", "token_ids", "step_seconds", "first_state_index")}
    if (data["states"].shape != (steps + 1, DIM) or data["states"].dtype != np.float64
            or not np.isfinite(data["states"]).all() or data["token_ids"].shape != (steps,)
            or data["token_ids"].dtype.kind not in "iu" or np.any(data["token_ids"] < 0)
            or np.any(data["token_ids"] >= VOCAB) or data["step_seconds"].shape != (steps,)
            or not np.isfinite(data["step_seconds"]).all() or np.any(data["step_seconds"] < 0)
            or data["first_state_index"].shape != () or data["first_state_index"].dtype.kind not in "iu"
            or int(data["first_state_index"]) != first):
        raise ValueError(f"invalid trajectory schema or values: {path.name}")
    return data


def check_schedule(row, trial_id, protocol="isolated"):
    if any(row.get(key) != value for key, value in expected_case(trial_id, protocol).items()):
        raise ValueError("case differs from the fixed source/desired/phase schedule")
    stages = row.get("stages", [])
    if len(stages) != 2:
        raise ValueError("case must contain endpoint and hold stages")
    for stage, name in zip(stages, STAGE_STEPS):
        controls = HOLD if name == "hold" else ENDPOINTS[row["desired_label"]]
        if stage.get("name") != name or stage.get("steps") != STAGE_STEPS[name] or stage.get("controls") != controls:
            raise ValueError("stage schedule or controls differ from the prescribed pulse")


def check_context(base, refs, references_hash, bindings, protocol="isolated"):
    for name in ("metadata.json", "trial-plan.json", "fixed-trial-inputs.npz", "attempt.json"):
        bindings.append(independent.file_record(base / name))
    metadata = json.loads((base / "metadata.json").read_text())
    attempt = json.loads((base / "attempt.json").read_text())
    plan = json.loads((base / "trial-plan.json").read_text())
    if (metadata["hold"] != HOLD or metadata["candidate_endpoints"] != ENDPOINTS
            or metadata["qualified_inputs"]["references"]["sha256"] != references_hash
            or plan["hold"] != HOLD or plan["candidate_endpoints"] != ENDPOINTS
            or plan["cache_tolerance"] != CACHE_TOLERANCE or plan["match_tolerance"] != independent.MATCH
            or len(plan["trials"]) != ({"isolated": 4, "roundtrip": 12, "sequence": 8}[protocol])):
        raise ValueError("metadata/plan does not bind this design and reference pair")
    arguments = metadata["arguments"]
    if arguments.get("protocol", "isolated") != protocol:
        raise ValueError("selected protocol does not match producer metadata")
    if (metadata["source_commit"] != attempt["source_commit"] or metadata["source_hashes"] != attempt["source_hashes"]
            or arguments != attempt["arguments"] or arguments["references_sha256"] != references_hash
            or arguments["recovery_results_sha256"] != metadata["recovery_prerequisite"]["results"]["sha256"]):
        raise ValueError("attempt, metadata or recovery prerequisite hash binding disagrees")
    for i, row in enumerate(plan["trials"]):
        check_schedule(row, i, protocol)
    independent.file_record(base / "fixed-trial-inputs.npz", plan["inputs"])
    with np.load(base / "fixed-trial-inputs.npz", allow_pickle=False) as archive:
        saved_refs, inputs = archive["references"], archive["initial_states"]
    expected_inputs = (refs[0, :1] if protocol == "sequence" else
                       refs[0] if protocol == "roundtrip" else refs[[0, 1, 0, 1], 0])
    if (saved_refs.dtype != np.float64 or not np.array_equal(saved_refs, refs)
            or inputs.dtype != np.float64 or inputs.shape != expected_inputs.shape
            or not np.array_equal(inputs, expected_inputs)):
        raise ValueError("fixed inputs differ from the prescribed reference starts")
    return {"status": "verified", "producer_source_commit": metadata.get("source_commit"),
            "producer_source_hashes": metadata.get("source_hashes"),
            "declared_geometric_design": metadata.get("geometric_design"),
            "declared_recovery_prerequisite": metadata.get("recovery_prerequisite")}


def check_cache(base, prefix, stage, final, bindings):
    comparison = stage.get("cache_comparison", {})
    records = comparison.get("artifacts", [])
    names = [f"{prefix}-cached.npz", f"{prefix}-full.npz"]
    if len(records) != 2 or [record["path"] for record in records] != names:
        raise ValueError("missing or misordered cached/full comparison artifacts")
    outputs, probabilities = [], []
    for record in records:
        path = record_local(base, record, bindings)
        with np.load(path, allow_pickle=False) as archive:
            check_controls(archive, stage["controls"])
            x, y, p = (archive[key] for key in ("input", "output", "probabilities"))
        if (x.shape != (DIM,) or y.shape != (DIM,) or p.shape != (VOCAB,)
                or any(value.dtype != np.float64 or not np.isfinite(value).all() for value in (x, y, p))
                or not np.array_equal(x, final["states"][0]) or np.any(p < 0) or np.any(p > 1)
                or not math.isclose(float(p.sum()), 1.0, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("comparison schema, probabilities or stage-start input mismatch")
        outputs.append(y)
        probabilities.append(p)
    gap = float(np.max(np.abs(outputs[0] - outputs[1])))
    if (gap > CACHE_TOLERANCE or comparison.get("passed") is not True
            or comparison.get("input_unchanged") is not True or comparison.get("tolerance") != CACHE_TOLERANCE
            or not math.isclose(gap, comparison.get("maximum_coordinate_gap", float("nan")), rel_tol=1e-12, abs_tol=1e-18)):
        raise ValueError("cached/full comparison failed or disagrees with the saved outputs")
    first_step_gap = None
    if len(final["states"]) > 1:
        first_step_gap = float(np.max(np.abs(outputs[0] - final["states"][1])))
        if first_step_gap > CACHE_TOLERANCE:
            raise ValueError("cached comparison output disagrees with the first rollout update")
    return {"passed": True, "maximum_coordinate_gap": gap, "tolerance": CACHE_TOLERANCE,
            "both_inputs_equal_stage_start": True,
            "maximum_probability_gap": float(np.max(np.abs(probabilities[0] - probabilities[1]))),
            "cached_output_first_step_maximum_coordinate_gap": first_step_gap,
            "cached_output_first_step_gate_passed": first_step_gap is not None and first_step_gap <= CACHE_TOLERANCE,
            "cached_output_equals_first_saved_step": len(final["states"]) > 1
                and np.array_equal(outputs[0], final["states"][1])}


def audit_stage(base, trial_id, stage, bindings, expected_steps=None):
    name, completed = stage["name"], stage.get("completed_steps")
    limit = STAGE_STEPS[name] if expected_steps is None else expected_steps
    if not isinstance(completed, int) or isinstance(completed, bool) or not 0 <= completed <= limit:
        raise ValueError("invalid stage step count")
    prefix = f"trial-{trial_id:02d}-{name}"
    final_name = f"{prefix}-final-{completed:03d}.npz"
    expected = {f"{prefix}-block-{end:03d}.npz" for end in range(12, completed + 1, 12)} | {final_name}
    records = stage.get("checkpoints", [])
    inventory = {record["path"]: record for record in records}
    if len(records) != len(expected) or set(inventory) != expected or stage.get("trajectory") != inventory.get(final_name):
        raise ValueError("missing or inconsistent stage checkpoints/final trajectory")
    for record in records:
        record_local(base, record, bindings)
    final = load_trajectory(base / final_name, completed, 0, stage["controls"])
    for end in range(12, completed + 1, 12):
        first = end - 12
        block = load_trajectory(base / f"{prefix}-block-{end:03d}.npz", 12, first, stage["controls"])
        if (not np.array_equal(block["states"], final["states"][first:end + 1])
                or not np.array_equal(block["token_ids"], final["token_ids"][first:end])
                or not np.array_equal(block["step_seconds"], final["step_seconds"][first:end])):
            raise ValueError(f"block/final mismatch: {prefix}, block endpoint {end}")
    comparison = check_cache(base, prefix, stage, final, bindings)
    complete = stage.get("status") == "completed" and completed == limit
    return {"name": name, "status": "verified" if complete else "incomplete", "controls": stage["controls"],
            "completed_steps": completed, "blocks_equal_final": True, "cache_comparison": comparison}, final


def classify(trajectory, refs, desired):
    result = independent.recompute(trajectory, refs, LABELS.index(desired))
    result["classification"] = {"returned": "correct", "switched": "wrong", "unresolved": "unresolved"}[
        result["classification"]]
    return {**result, "desired_label": desired}


def outcome_agrees(actual, reported):
    gate = reported.get("cycle_gate", {})
    if (actual["classification"] != reported.get("classification") or actual["desired_label"] != reported.get("desired_label")
            or actual["cycle_gate_passed"] != gate.get("passed")
            or actual["reference_matches"] != reported.get("reference_matches")
            or reported.get("match_tolerance") != independent.MATCH
            or gate.get("closure_tolerance") != independent.CLOSURE
            or gate.get("divisor_separation") != independent.DIVISOR_GAP
            or not math.isclose(actual["maximum_l2_residual"], gate.get("maximum_l2_residual", float("nan")),
                                rel_tol=1e-12, abs_tol=1e-18)):
        return False
    for key, old in (("lag_six_phase_l2_residuals", gate.get("lag_six_phase_l2_residuals", [])),):
        if np.shape(old) != (6,) or not np.allclose(actual[key], old, rtol=1e-12, atol=1e-18):
            return False
    divisors = gate.get("proper_divisors", [])
    if len(divisors) != 3:
        return False
    for new, old in zip(actual["proper_divisors"], divisors):
        if (new["divisor"] != old.get("divisor") or new["excluded"] != old.get("excluded")
                or np.shape(old.get("phase_l2_gaps", [])) != (6,)
                or not np.allclose(new["phase_l2_gaps"], old["phase_l2_gaps"], rtol=1e-12, atol=1e-18)
                or not math.isclose(new["maximum_l2_gap"], old.get("maximum_l2_gap", float("nan")), rel_tol=1e-12, abs_tol=1e-18)):
            return False
    for comparison in actual["reference_distances"]:
        old = [row for row in reported.get("reference_distances", []) if row.get("label") == comparison["label"]]
        if len(old) != 1 or not math.isclose(comparison["distance"], old[0].get("distance", float("nan")), rel_tol=1e-12, abs_tol=1e-18):
            return False
    return True


def audit(run_dir, references, references_sha256, protocol="isolated"):
    count = {"isolated": 4, "roundtrip": 12, "sequence": 8}[protocol]
    base, bindings = Path(run_dir), []
    report = {"status": "audit_failed", "scope": "saved-array consistency only; no independent model evaluation",
              "bindings": bindings, "rows": [], "all_four_cases_verified": False, "aggregate_agreement": False,
              "all_four_writes_correct": False, "measured_outcome_counts": {key: 0 for key in ("correct", "wrong", "unresolved")},
              "reference_binding": independent.file_record(references, {"sha256": references_sha256})}
    with np.load(references, allow_pickle=False) as archive:
        refs = archive["states"]
    if refs.shape != (2, 6, DIM) or refs.dtype != np.float64 or not np.isfinite(refs).all():
        raise ValueError("references must be finite float64 of shape (2,6,896)")
    try:
        report["input_context"] = check_context(base, refs, references_sha256, bindings, protocol)
    except Exception as error:
        report["input_context"] = {"status": "unresolved", "error": f"{type(error).__name__}: {error}"}
    try:
        bindings.append(independent.file_record(base / "results.json"))
        results = json.loads((base / "results.json").read_text())
        if len(results.get("rows", [])) != count or [row.get("trial_id") for row in results["rows"]] != list(range(count)):
            raise ValueError("results must contain the complete ordered case inventory")
        if results.get("protocol", "isolated") != protocol:
            raise ValueError("results protocol mismatch")
    except Exception as error:
        report["results_error"] = f"{type(error).__name__}: {error}"
        report["rows"] = [{**expected_case(i, protocol), "status": "missing", "outcome": {"classification": "not_checked"}} for i in range(count)]
        return report
    carried = {}
    for trial_id, row in enumerate(results["rows"]):
        current = {**expected_case(trial_id, protocol), "status": "failed", "stages": [], "outcome": {"classification": "not_checked"}}
        try:
            check_schedule(row, trial_id, protocol)
            finals = []
            for stage in row["stages"]:
                try:
                    stage_report, final = audit_stage(base, trial_id, stage, bindings)
                except Exception as error:
                    stage_report, final = {"name": stage["name"], "status": "failed",
                                          "error": f"{type(error).__name__}: {error}"}, None
                current["stages"].append(stage_report)
                finals.append(final)
            if any(final is None for final in finals):
                raise ValueError("one or more stages lack verifiable raw evidence")
            predecessor = row.get("predecessor_trial_id")
            expected_initial = (carried[predecessor] if predecessor is not None
                                else refs[row["source_index"], row["phase"]])
            if not np.array_equal(finals[0]["states"][0], expected_initial):
                raise ValueError("endpoint initial state differs from reference or actual predecessor")
            if not np.array_equal(finals[0]["states"][-1], finals[1]["states"][0]):
                raise ValueError("endpoint-last and hold-first states are not identical")
            current.update(initial_reference_verified=predecessor is None,
                           predecessor_continuity_verified=predecessor is not None,
                           endpoint_hold_continuity_verified=True)
            complete = (row.get("status") == "completed" and row.get("completed_steps") == 360
                        and all(stage["status"] == "verified" for stage in current["stages"]))
            if complete:
                current["outcome"] = classify(finals[1]["states"][1:], refs, row["desired_label"])
                current["producer_agreement"] = outcome_agrees(current["outcome"], row.get("outcome", {}))
                current["status"] = "verified" if current["producer_agreement"] else "failed"
                if current["status"] == "verified":
                    carried[trial_id] = finals[1]["states"][-1].copy()
            else:
                current.update(status="incomplete", error="both fixed stage lengths and completed row are required")
        except Exception as error:
            current["error"] = f"{type(error).__name__}: {error}"
        report["rows"].append(current)
    recorded = {Path(item["path"]).name for item in bindings if item["path"].endswith(".npz")}
    report["unaccounted_raw_files"] = [independent.file_record(path) for path in sorted(base.glob("trial-*.npz")) if path.name not in recorded]
    verified = all(row["status"] == "verified" for row in report["rows"])
    counts = {label: sum(row["status"] == "verified" and row["outcome"]["classification"] == label for row in report["rows"])
              for label in ("correct", "wrong", "unresolved")}
    aggregate_agreement = verified and results.get("outcome_counts") == counts and results.get("status") == (
        "passed" if counts["correct"] == count else "failed")
    passed = (verified and report["input_context"]["status"] == "verified"
              and not report["unaccounted_raw_files"] and aggregate_agreement)
    report.update(status="audit_passed" if passed else "audit_failed", all_four_cases_verified=verified if count == 4 else None,
                  aggregate_agreement=aggregate_agreement, measured_outcome_counts=counts,
                  all_four_writes_correct=passed and counts["correct"] == 4 if count == 4 else None,
                  protocol=protocol, all_cases_verified=verified,
                  all_writes_correct=passed and counts["correct"] == count)
    if protocol == "roundtrip":
        report["round_trips"] = [{"initial_phase": phase,
            "both_writes_correct": all(report["rows"][i]["status"] == "verified" and
                report["rows"][i]["outcome"]["classification"] == "correct" for i in (2*phase, 2*phase+1))}
            for phase in range(6)]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--references-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=("isolated", "roundtrip", "sequence"), default="isolated")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root):
        parser.error("audit output must be outside the repository")
    if len(args.references_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.references_sha256):
        parser.error("reference SHA-256 must be a lowercase digest")
    receipt = {"utc": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
               "source_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
               "source_files": [independent.file_record(path) for path in (__file__, independent.__file__)],
               "numpy_version": np.__version__,
               "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    with args.output.open("x") as output:
        try:
            receipt["audit"] = audit(args.run_dir, args.references, args.references_sha256, args.protocol)
        except Exception as error:
            receipt["audit"] = {"status": "audit_failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        json.dump(receipt, output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        independent.os.fsync(output.fileno())
    print(json.dumps({"output": str(args.output), "status": receipt["audit"]["status"]}))
    return 0 if receipt["audit"]["status"] == "audit_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
