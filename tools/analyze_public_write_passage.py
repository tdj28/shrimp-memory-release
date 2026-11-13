#!/usr/bin/env python3
"""Write time against the distance of a write endpoint beyond a fold. Zero model calls.

Run in two steps, in this order (plan and predictions: docs/public-write-passage.md):

  predict   reads the companion study's saved branch, the audited fold receipt, the branch receipt and the control
            values in the design files of archived endpoint-perturbation runs. It never opens a saved state of those
            runs. It writes the predicted passage updates to a file that the plan then binds by hash.
  measure   reads the saved states of the same runs, measures each passage, and reports it beside the predictions.

Nothing is tuned after the states are opened: the passage definition, the thresholds and the fits are fixed here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_public_register_branch as arb  # noqa: E402  (strict stage loader, Codex Task 9 D2)

ROOT = Path(__file__).resolve().parents[1]
ALPHA_H = 0.07293697809562831
LOG10_T_H = -0.15395151820007266
PERIOD = 6
FIT_WINDOW = 0.25                # branch points with |multiplier - 1| at most this enter a fold fit
SENSITIVITY_WINDOWS = (0.15, 0.35)
THRESHOLDS = (0.25, 0.5, 0.75)   # fraction of the way from the starting cycle to the requested one
PRIMARY = "0.75"                 # the passage is read here; the other two show how much the choice matters
ENDPOINT_STEPS, HOLD_STEPS = 120, 240


def binding(path: Path) -> dict:
    raw = Path(path).read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_branch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points = json.loads(path.read_text())["points"]
    lam, mu = [], []
    for point in points:
        source = point.get("multipliers_refined") or point["multipliers"]
        if abs(source[0][1]) > 1e-12:
            raise SystemExit(f"leading multiplier of point {point['index']} is not real")
        lam.append(float(point["lambda"]))
        mu.append(float(source[0][0]))
    return np.array(lam), np.array(mu)


def fold_fit(lam: np.ndarray, mu: np.ndarray, crossing: int, window: float) -> dict:
    """alpha = alpha_fold + c2 m^2 + c3 m^3 with m = multiplier - 1, least squares over nearby branch points.

    At a fold the branch turns in alpha where the multiplier passes +1, so alpha has no linear term in m. For the
    one-dimensional normal form y -> y + a (alpha - alpha_fold) + b y^2 the multiplier of an orbit at distance D from
    the fold satisfies (1 - multiplier)^2 = 4 a b D, so c2 = -1 / (4 a b) up to the side on which the orbits exist.
    """
    idx = [j for j in range(max(0, crossing - 10), min(len(lam), crossing + 12)) if abs(mu[j] - 1.0) <= window]
    m = mu[idx] - 1.0
    design = np.stack([np.ones_like(m), m ** 2, m ** 3], axis=1)
    origin = float(lam[crossing])
    coef, *_ = np.linalg.lstsq(design, lam[idx] - origin, rcond=None)
    residual = lam[idx] - origin - design @ coef
    return {"window": window, "branch_points": [int(j) for j in idx], "alpha_fold": origin + float(coef[0]),
            "alpha_fold_minus_last_stable_neighbour": float(coef[0]), "c2": float(coef[1]), "c3": float(coef[2]),
            "ab_per_unit_alpha": 1.0 / (4.0 * abs(float(coef[1]))),
            "orbits_exist": "below the fold" if coef[1] < 0 else "above the fold",
            "max_abs_residual_alpha": float(np.abs(residual).max())}


def passage_updates(delta: float, hold_distance: float, ab: float) -> float:
    """Updates from the hold orbit, a distance hold_distance before the fold, to far beyond the bottleneck."""
    return PERIOD * (math.pi / 2 + math.atan(math.sqrt(hold_distance / delta))) / math.sqrt(ab * delta)


def reach_updates(delta: float, hold_distance: float, ab: float) -> float:
    """Updates from the hold orbit to the place of the hold's unstable orbit, its mirror image across the fold."""
    return PERIOD * 2.0 * math.atan(math.sqrt(hold_distance / delta)) / math.sqrt(ab * delta)


def regressor(delta: float, hold_distance: float) -> float:
    return (math.pi / 2 + math.atan(math.sqrt(hold_distance / delta))) / math.sqrt(delta)


def load_design(run: Path) -> dict:
    design = json.loads((run / "design.json").read_text())
    groups = []
    for group in design["groups"]:
        stages = {s["name"]: s for s in design["realized_stages"] if s["group_id"] == group["group_id"]}
        endpoint, hold = stages["endpoint"], stages["hold"]
        if endpoint["steps"] != ENDPOINT_STEPS or hold["steps"] != HOLD_STEPS:
            raise SystemExit(f"unexpected stage lengths in {run}")
        if hold["controls"]["alpha"] != ALPHA_H or hold["controls"]["log10_temperature"] != LOG10_T_H:
            raise SystemExit(f"hold controls of {run} are not the common hold")
        groups.append({"group_id": group["group_id"], "name": group["name"], "fraction": group.get("fraction"),
                       "endpoint_stage": endpoint["stage_id"], "hold_stage": hold["stage_id"],
                       "alpha": endpoint["controls"]["alpha"],
                       "log10_temperature": endpoint["controls"]["log10_temperature"]})
    return {"run": str(run), "requested_bit": design["requested_bit"],
            "profile": design.get("profile", "axis-stencil"), "groups": groups, "design": binding(run / "design.json")}


def predict(args) -> int:
    lam, mu = load_branch(args.branch_json)
    folds = [i for i in range(len(mu) - 1) if (mu[i] - 1.0) * (mu[i + 1] - 1.0) < 0]
    if len(folds) != 2:
        raise SystemExit(f"expected two fold crossings on the branch, found {folds}")
    first = fold_fit(lam, mu, folds[0], FIT_WINDOW)
    third = fold_fit(lam, mu, folds[1], FIT_WINDOW)
    sensitivity = {str(w): {"first_sheet_fold": fold_fit(lam, mu, folds[0], w),
                            "third_sheet_fold": fold_fit(lam, mu, folds[1], w)} for w in SENSITIVITY_WINDOWS}
    fold_receipt = json.loads(args.fold_receipt.read_text())["fold"]["refined"]
    if fold_receipt["alpha"] != ALPHA_H:
        raise SystemExit("fold receipt is not at the hold strength")
    t_fold = float(fold_receipt["log10_temperature"])
    branch_receipt = json.loads(args.branch_receipt.read_text())
    mu_cycle1 = float(branch_receipt["at_hold"]["cycle_1"]["leading_multiplier_cubic"])
    mu_cycle0 = float(branch_receipt["at_hold"]["cycle_0"]["leading_multiplier"])
    at_hold = branch_receipt["at_hold"]
    d0 = float(at_hold["unstable_orbit"]["distance_to_cycle_0"])
    d1 = float(at_hold["unstable_orbit"]["distance_to_cycle_1"])
    separation = float(at_hold["cycle_separation_at_this_phase"])
    unstable_coordinate = (d0 ** 2 + separation ** 2 - d1 ** 2) / (2.0 * separation ** 2)
    saved_phase = int(at_hold["phase_of_saved_branch_states"])

    d_alpha = first["alpha_fold"] - ALPHA_H          # hold to the fold of the cycle-0 sheet, in alpha
    d_log_t = t_fold - LOG10_T_H                      # hold to the fold of the cycle-1 sheet, in log10 T
    ab_alpha = first["ab_per_unit_alpha"]
    # Two rough coefficients for the temperature direction; no branch along the temperature was recorded.
    ab_t_from_hold = ((1.0 - mu_cycle1) / 2.0) ** 2 / d_log_t
    chord = d_log_t / (ALPHA_H - third["alpha_fold"])
    ab_t_from_chord = third["ab_per_unit_alpha"] / chord
    # The same constant-coefficient estimate on the alpha side, where the answer is known, as a gauge of its error.
    ab_alpha_from_hold = ((1.0 - mu_cycle0) / 2.0) ** 2 / d_alpha

    runs = [load_design(run) for run in args.run]
    rows = []
    for run in runs:
        bit = run["requested_bit"]
        for group in run["groups"]:
            row = {"run": run["run"], "profile": run["profile"], "requested_bit": bit, "group": group["name"],
                   "fraction": group["fraction"], "alpha": group["alpha"],
                   "log10_temperature": group["log10_temperature"]}
            if bit == 1:
                row["on_recorded_line"] = group["log10_temperature"] == LOG10_T_H
                if row["on_recorded_line"]:
                    delta = group["alpha"] - first["alpha_fold"]
                    row["alpha_beyond_fold"] = delta
                    if delta > 0:
                        row["predicted_passage_update"] = passage_updates(delta, d_alpha, ab_alpha)
                        row["predicted_update_reaching_hold_unstable_orbit"] = reach_updates(
                            delta, d_alpha, ab_alpha)
                        row["predicted_write_succeeds_with_120_updates"] = (
                            row["predicted_update_reaching_hold_unstable_orbit"] < ENDPOINT_STEPS)
                    else:
                        row["predicted_passage_update"] = None
                        row["predicted_update_reaching_hold_unstable_orbit"] = None
                        row["predicted_write_succeeds_with_120_updates"] = False
                        row["note"] = "the cycle-0 sheet is recorded at this strength; no passage expected"
            else:
                row["on_recorded_line"] = group["alpha"] == ALPHA_H
                if row["on_recorded_line"]:
                    delta = group["log10_temperature"] - t_fold
                    row["log10_temperature_beyond_fold"] = delta
                    if delta > 0:
                        row["rough_passage_update_coefficient_from_hold_multiplier"] = passage_updates(
                            delta, d_log_t, ab_t_from_hold)
                        row["rough_passage_update_coefficient_from_fold_chord"] = passage_updates(
                            delta, d_log_t, ab_t_from_chord)
            rows.append(row)

    out = {
        "schema": "public-write-passage-prediction-v1", "utc": now(), "model_calls": 0,
        "opened_saved_states_of_the_pilot_runs": False,
        "script": binding(Path(__file__).resolve()),
        "model": "one-dimensional fold normal form y -> y + a (control - fold) + b y^2 for the period-6 return map; "
                 "passage in periods from the hold orbit, a distance D before the fold, to far beyond the bottleneck "
                 "at a distance Delta beyond it: [pi/2 + arctan(sqrt(D/Delta))] / sqrt(a b Delta)",
        "fit_window_abs_multiplier_minus_one": FIT_WINDOW,
        "first_sheet_fold": first, "third_sheet_fold": third, "window_sensitivity": sensitivity,
        "hold_to_first_sheet_fold_alpha": d_alpha,
        "fold_on_hold_strength_line_log10_temperature": t_fold, "hold_to_that_fold_log10_temperature": d_log_t,
        "multiplier_cycle_0_at_hold": mu_cycle0, "multiplier_cycle_1_at_hold": mu_cycle1,
        "hold_unstable_orbit": {
            "coordinate_at_saved_phase": unstable_coordinate, "saved_phase": saved_phase,
            "note": "projection of the hold's unstable orbit on the segment between the two cycles, from the three "
                    "distances in the branch receipt; in the normal form it is the mirror image of the starting "
                    "orbit across the fold, and a write succeeds if the state is beyond it when the pulse ends"},
        "write1_coefficient_ab_per_unit_alpha": ab_alpha,
        "write1_constant_coefficient_estimate_from_hold_multiplier": ab_alpha_from_hold,
        "write0_rough_coefficients_per_unit_log10_temperature": {
            "from_hold_multiplier_constant_coefficient": ab_t_from_hold,
            "from_third_sheet_fold_along_a_straight_fold_curve": ab_t_from_chord,
            "fold_curve_chord_dlog10T_dalpha": chord},
        "rows": rows,
        "inputs": {"branch": binding(args.branch_json), "fold_receipt": binding(args.fold_receipt),
                   "branch_receipt": binding(args.branch_receipt), "designs": [r["design"] for r in runs]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(out, handle, indent=1)
        handle.write("\n")
    print(f"first-sheet fold: alpha_fold = last stable neighbour + {first['alpha_fold_minus_last_stable_neighbour']:.3e}, "
          f"ab = {ab_alpha:.1f}, points {first['branch_points']}, max residual {first['max_abs_residual_alpha']:.2e}")
    for w, s in sensitivity.items():
        f = s["first_sheet_fold"]
        print(f"  window {w}: fold + {f['alpha_fold_minus_last_stable_neighbour']:.3e}, ab = {f['ab_per_unit_alpha']:.1f}, "
              f"points {f['branch_points']}")
    print(f"third-sheet fold: alpha_fold = {third['alpha_fold']:.14f}, ab = {third['ab_per_unit_alpha']:.1f}, "
          f"points {third['branch_points']}, max residual {third['max_abs_residual_alpha']:.2e}")
    print(f"hold to first-sheet fold: {d_alpha:.4e} in alpha; hold to temperature fold: {d_log_t:.4e} in log10 T")
    print(f"constant-coefficient estimate on the alpha side: {ab_alpha_from_hold:.1f} against {ab_alpha:.1f} at the fold")
    print(f"write-0 rough coefficients: {ab_t_from_hold:.1f} (hold multiplier), {ab_t_from_chord:.1f} (chord {chord:.4f})")
    print(f"hold's unstable orbit: coordinate {unstable_coordinate:.4f} at phase {saved_phase}")
    for row in rows:
        if not row.get("on_recorded_line"):
            print(f"  bit {row['requested_bit']} {row['profile']:>24} {row['group']:>13}: off the recorded line, no prediction")
        elif row["requested_bit"] == 1:
            p, q = row["predicted_passage_update"], row["predicted_update_reaching_hold_unstable_orbit"]
            print(f"  bit 1 {row['profile']:>24} {row['group']:>13}: beyond fold {row['alpha_beyond_fold']:+.3e}  "
                  f"predicted passage {'none' if p is None else f'{p:.1f}'}  "
                  f"reaches the hold's unstable orbit {'never' if q is None else f'{q:.1f}'}")
        else:
            d = row["log10_temperature_beyond_fold"]
            a = row.get("rough_passage_update_coefficient_from_hold_multiplier")
            b = row.get("rough_passage_update_coefficient_from_fold_chord")
            print(f"  bit 0 {row['profile']:>24} {row['group']:>13}: beyond fold {d:+.3e}  rough "
                  f"{'none' if a is None else f'{a:.1f}'} / {'none' if b is None else f'{b:.1f}'}")
    return 0


# ----------------------------------------------------------------------------------------------------------------
# measure


def load_stage(run: Path, stage: int, alpha: float | None = None, log10_t: float | None = None) -> np.ndarray:
    """The shared strict loader (indices, names, dtype, finiteness, and the run's stage record), plus the design's
    controls when given."""
    states, a, t, _ = arb.load_stage(run, stage)
    if alpha is not None and (a, t) != (alpha, log10_t):
        raise SystemExit(f"controls of stage {stage} in {run} differ from the design")
    return states


def blocks_digest(run: Path) -> dict:
    digest, count, total = hashlib.sha256(), 0, 0
    for name in sorted(glob.glob(str(run / "stage-*"))):
        raw = Path(name).read_bytes()
        digest.update(Path(name).name.encode() + b"\0" + hashlib.sha256(raw).digest())
        count += 1
        total += len(raw)
    return {"files": count, "bytes": total, "sha256_of_names_and_file_digests": digest.hexdigest()}


def coordinate(traj: np.ndarray, refs: np.ndarray, start_phase: int) -> np.ndarray:
    """0 on reference cycle 0 and 1 on reference cycle 1, phase by phase; traj[k] is the state after k+1 updates."""
    phase = (start_phase + np.arange(1, len(traj) + 1)) % PERIOD
    span = refs[1, phase] - refs[0, phase]
    return np.einsum("ij,ij->i", traj - refs[0, phase], span) / np.einsum("ij,ij->i", span, span)


def passage_update(c: np.ndarray, bit: int, threshold: float) -> int | None:
    """First update from which the coordinate stays on the requested side of the threshold for one full period."""
    side = c >= threshold if bit == 1 else c <= 1.0 - threshold
    for k in range(len(c) - PERIOD + 1):
        if side[k:k + PERIOD].all():
            return k + 1
    return None


def level_crossing(c: np.ndarray, level: float, bit: int, phase: int) -> float | None:
    """Update at which the coordinate, sampled at one phase, first passes a level; linear between two samples."""
    updates = np.arange(1, len(c) + 1)
    pick = updates[updates % PERIOD == phase]
    values = c[pick - 1]
    beyond = values >= level if bit == 1 else values <= level
    if not beyond.any():
        return None
    k = int(np.argmax(beyond))
    if k == 0:
        return float(pick[0])
    return float(pick[k - 1] + PERIOD * (level - values[k - 1]) / (values[k] - values[k - 1]))


def displacement(start: np.ndarray, traj: np.ndarray) -> np.ndarray:
    """s[n] = |x(n+6) - x(n)| for n = 0 .. N-6, with x(0) the state the stage started from."""
    full = np.concatenate([start[None], traj], axis=0)
    return np.linalg.norm(full[PERIOD:] - full[:-PERIOD], axis=1)


def bottleneck(s: np.ndarray, limit: int) -> list[dict]:
    """Smallest per-period displacement at each phase among n with n + 6 <= limit, with a three-point parabola."""
    out = []
    for phase in range(PERIOD):
        n = np.arange(phase, limit - PERIOD + 1, PERIOD)
        if len(n) < 3:
            out.append({"phase": phase, "samples": int(len(n))})
            continue
        v = s[n]
        k = int(np.argmin(v))
        entry = {"phase": phase, "samples": int(len(n)), "update_at_minimum": int(n[k]), "minimum": float(v[k]),
                 "interior": bool(0 < k < len(v) - 1), "first": float(v[0]), "last": float(v[-1])}
        if entry["interior"]:
            curvature = float(v[k + 1] + v[k - 1] - 2.0 * v[k])
            if curvature > 0:
                refined = float(v[k] - (v[k + 1] - v[k - 1]) ** 2 / (8.0 * curvature))
                entry["minimum_interpolated"] = refined
                entry["relative_curvature_per_period_squared"] = curvature / (2.0 * refined)
        out.append(entry)
    return out


def tail(c: np.ndarray, s: np.ndarray) -> dict:
    """Is a lane that did not pass still drifting, or settling onto an orbit? Same-phase samples at the stage end."""
    last = len(s) - 1
    samples = [float(s[last - PERIOD * j]) for j in range(4)][::-1]
    return {"coordinate_at_end": float(c[-1]), "coordinate_one_period_earlier": float(c[-1 - PERIOD]),
            "coordinate_largest": float(c.max()), "coordinate_smallest": float(c.min()),
            "last_four_displacements_same_phase": samples,
            "ratios": [samples[j + 1] / samples[j] for j in range(3)]}


def nearest_label(state: np.ndarray, refs: np.ndarray, phase: int) -> tuple[int, float]:
    d = [float(np.linalg.norm(state - refs[label, phase])) for label in (0, 1)]
    return int(np.argmin(d)), min(d)


def line_fit(x: list[float], y: list[float]) -> dict | None:
    if len(x) < 3:
        return None
    design = np.stack([np.ones(len(x)), np.array(x)], axis=1)
    coef, *_ = np.linalg.lstsq(design, np.array(y), rcond=None)
    residual = np.array(y) - design @ coef
    return {"points": len(x), "intercept": float(coef[0]), "slope": float(coef[1]),
            "residuals": [float(r) for r in residual], "max_abs_residual": float(np.abs(residual).max())}


def measure(args) -> int:
    prediction = json.loads(args.prediction.read_text())
    predicted = {(r["run"], r["group"]): r for r in prediction["rows"]}
    with np.load(args.reference_cycles) as z:
        refs = z["states"]
    if refs.shape != (2, PERIOD, 896):
        raise SystemExit(f"unexpected reference shape {refs.shape}")
    alpha_fold = prediction["first_sheet_fold"]["alpha_fold"]
    t_fold = prediction["fold_on_hold_strength_line_log10_temperature"]
    d_alpha = prediction["hold_to_first_sheet_fold_alpha"]
    d_log_t = prediction["hold_to_that_fold_log10_temperature"]
    unstable = prediction["hold_unstable_orbit"]

    rows, run_bindings, endpoint_record = [], [], {}
    for run in args.run:
        design = load_design(run)
        bit = design["requested_bit"]
        run_bindings.append({"run": str(run), "design": design["design"], "result": binding(run / "result.json"),
                             "manifest": binding(run / "manifest.json"), "stage_files": blocks_digest(run)})
        for group in design["groups"]:
            start = np.load(run / f"stage-{group['endpoint_stage']:02d}-initial.npy")
            if not np.array_equal(start, refs[:, 0]):
                raise SystemExit(f"{run} group {group['name']} does not start on the phase-zero references")
            endpoint = load_stage(run, group["endpoint_stage"], group["alpha"], group["log10_temperature"])
            hold = load_stage(run, group["hold_stage"], ALPHA_H, LOG10_T_H)
            if endpoint.shape[0] != ENDPOINT_STEPS or hold.shape[0] != HOLD_STEPS:
                raise SystemExit(f"unexpected trajectory lengths in {run}")
            pred = predicted[(str(run), group["name"])]
            for lane in (0, 1):
                traj = np.concatenate([endpoint[:, lane], hold[:, lane]], axis=0)
                c = coordinate(traj, refs, 0)
                label, distance = nearest_label(traj[-1], refs, (ENDPOINT_STEPS + HOLD_STEPS) % PERIOD)
                row = {"run": str(run), "profile": design["profile"], "requested_bit": bit, "group": group["name"],
                       "fraction": group["fraction"], "alpha": group["alpha"],
                       "log10_temperature": group["log10_temperature"], "lane_start_cycle": lane,
                       "passage_lane": lane != bit, "final_nearest_cycle": label,
                       "final_distance_to_that_cycle": distance, "requested_bit_stored": label == bit,
                       "on_recorded_line": pred.get("on_recorded_line", False)}
                if row["passage_lane"]:
                    crossing = {str(t): passage_update(c, bit, t) for t in THRESHOLDS}
                    n_star = crossing[PRIMARY]
                    within = n_star is not None and n_star <= ENDPOINT_STEPS
                    s = displacement(start[lane], endpoint[:, lane])
                    limit = n_star if within else ENDPOINT_STEPS
                    row.update({"passage_update": crossing, "passage_within_endpoint_stage": within,
                                "coordinate_at_update_120": float(c[ENDPOINT_STEPS - 1]),
                                "update_reaching_hold_unstable_orbit": level_crossing(
                                    c[:ENDPOINT_STEPS], unstable["coordinate_at_saved_phase"], bit,
                                    unstable["saved_phase"]),
                                "bottleneck_by_phase": bottleneck(s, limit),
                                "predicted_passage_update": pred.get("predicted_passage_update"),
                                "predicted_update_reaching_hold_unstable_orbit": pred.get(
                                    "predicted_update_reaching_hold_unstable_orbit"),
                                "alpha_beyond_fold": pred.get("alpha_beyond_fold"),
                                "log10_temperature_beyond_fold": pred.get("log10_temperature_beyond_fold")})
                    if not within:
                        row["endpoint_stage_tail"] = tail(c[:ENDPOINT_STEPS], s)
                    endpoint_record[(bit, group["alpha"], group["log10_temperature"])] = (
                        start[lane].copy(), endpoint[:, lane].copy())
                rows.append(row)

    # ---- comparisons fixed in the plan
    passing = [r for r in rows if r["passage_lane"] and r["on_recorded_line"] and r["passage_within_endpoint_stage"]]
    comparisons = {}

    def unique(rows_in, key):
        seen, out = set(), []
        for r in sorted(rows_in, key=lambda r: r[key]):
            if r[key] not in seen:
                seen.add(r[key])
                out.append(r)
        return out

    w1 = unique([r for r in passing if r["requested_bit"] == 1], "alpha")
    comparisons["write1_prediction"] = [
        {"alpha_beyond_fold": r["alpha_beyond_fold"], "group": r["group"], "profile": r["profile"],
         "measured": r["passage_update"][PRIMARY], "predicted": r["predicted_passage_update"],
         "measured_over_predicted": r["passage_update"][PRIMARY] / r["predicted_passage_update"],
         "measured_at_thresholds": r["passage_update"]} for r in w1]
    comparisons["write1_line"] = line_fit([regressor(r["alpha_beyond_fold"], d_alpha) for r in w1],
                                          [float(r["passage_update"][PRIMARY]) for r in w1])
    if comparisons["write1_line"]:
        comparisons["write1_line"]["ab_from_slope"] = (PERIOD / comparisons["write1_line"]["slope"]) ** 2
        comparisons["write1_line"]["ab_from_branch"] = prediction["write1_coefficient_ab_per_unit_alpha"]

    w0 = unique([r for r in passing if r["requested_bit"] == 0], "log10_temperature")
    comparisons["write0_measured"] = [
        {"log10_temperature_beyond_fold": r["log10_temperature_beyond_fold"], "group": r["group"],
         "profile": r["profile"], "measured": r["passage_update"][PRIMARY],
         "measured_at_thresholds": r["passage_update"],
         "rough_from_hold_multiplier": predicted[(r["run"], r["group"])].get(
             "rough_passage_update_coefficient_from_hold_multiplier"),
         "rough_from_fold_chord": predicted[(r["run"], r["group"])].get(
             "rough_passage_update_coefficient_from_fold_chord")} for r in w0]
    fit0 = line_fit([regressor(r["log10_temperature_beyond_fold"], d_log_t) for r in w0],
                    [float(r["passage_update"][PRIMARY]) for r in w0])
    if fit0:
        fit0["ab_from_slope"] = (PERIOD / fit0["slope"]) ** 2
        fit0["rough_coefficients"] = prediction["write0_rough_coefficients_per_unit_log10_temperature"]
        failing = [r for r in rows if r["passage_lane"] and r["requested_bit"] == 0 and r["on_recorded_line"]
                   and not r["passage_within_endpoint_stage"] and (r["log10_temperature_beyond_fold"] or 0) > 0]
        fit0["extrapolated_to_failing_settings"] = [
            {"group": r["group"], "profile": r["profile"],
             "log10_temperature_beyond_fold": r["log10_temperature_beyond_fold"],
             "extrapolated_passage_update": fit0["intercept"] + fit0["slope"] * regressor(
                 r["log10_temperature_beyond_fold"], d_log_t)} for r in failing]
    comparisons["write0_line"] = fit0
    bare0 = line_fit([1.0 / math.sqrt(r["log10_temperature_beyond_fold"]) for r in w0],
                     [float(r["passage_update"][PRIMARY]) for r in w0])
    if bare0:
        bare0["ab_from_slope"] = (PERIOD * math.pi / bare0["slope"]) ** 2
    comparisons["write0_line_bare_inverse_square_root"] = bare0

    # slowest per-period displacement against the control: zero crossing = a fold estimate from the dynamics alone
    def slowest(rows_in, control, fold):
        out = []
        for phase in range(PERIOD):
            xs, ys = [], []
            for r in rows_in:
                e = r["bottleneck_by_phase"][phase]
                if e.get("interior") and "minimum_interpolated" in e:
                    xs.append(r[control])
                    ys.append(e["minimum_interpolated"])
            fit = line_fit(xs, ys)
            entry = {"phase": phase, "settings_with_an_interior_minimum": len(xs)}
            if fit and fit["slope"] != 0:
                zero = -fit["intercept"] / fit["slope"]
                entry.update({"zero_crossing": zero, "zero_crossing_minus_independent_fold": zero - fold,
                              "slope": fit["slope"], "max_abs_residual_over_largest": fit["max_abs_residual"] / max(ys)})
            out.append(entry)
        return out

    all_w1 = unique([r for r in rows if r["passage_lane"] and r["requested_bit"] == 1 and r["on_recorded_line"]
                     and (r["alpha_beyond_fold"] or 0) > 0], "alpha")
    all_w0 = unique([r for r in rows if r["passage_lane"] and r["requested_bit"] == 0 and r["on_recorded_line"]
                     and (r["log10_temperature_beyond_fold"] or 0) > 0], "log10_temperature")
    comparisons["write1_reaching_hold_unstable_orbit"] = [
        {"alpha_beyond_fold": r["alpha_beyond_fold"], "group": r["group"], "profile": r["profile"],
         "measured": r["update_reaching_hold_unstable_orbit"],
         "predicted": r["predicted_update_reaching_hold_unstable_orbit"],
         "requested_bit_stored": r["requested_bit_stored"]} for r in all_w1]
    comparisons["write0_reaching_hold_unstable_orbit"] = [
        {"log10_temperature_beyond_fold": r["log10_temperature_beyond_fold"], "group": r["group"],
         "profile": r["profile"], "measured": r["update_reaching_hold_unstable_orbit"],
         "requested_bit_stored": r["requested_bit_stored"]} for r in all_w0]
    comparisons["write1_slowest_displacement_against_alpha"] = slowest(all_w1, "alpha", alpha_fold)
    comparisons["write0_slowest_displacement_against_log10_temperature"] = slowest(all_w0, "log10_temperature", t_fold)

    def local_coefficient(rows_in, key):
        out = []
        for r in rows_in:
            values = [e["relative_curvature_per_period_squared"] / r[key] for e in r["bottleneck_by_phase"]
                      if "relative_curvature_per_period_squared" in e]
            if values:
                out.append({"group": r["group"], "profile": r["profile"], key: r[key],
                            "ab_by_phase": values, "ab_median": float(np.median(values))})
        return out

    comparisons["write1_coefficient_from_bottleneck_curvature"] = local_coefficient(all_w1, "alpha_beyond_fold")
    comparisons["write0_coefficient_from_bottleneck_curvature"] = local_coefficient(
        all_w0, "log10_temperature_beyond_fold")

    # ---- the continued-perturbation arms: another 120 updates at the full negative offset, then the hold
    continuations = []
    for run in args.continuation:
        design = json.loads((run / "design.json").read_text())
        bit, origin = design["requested_bit"], design["origin_reference_index"]
        phases = design["origin_reference_phases"]
        stages = {s["stage_id"]: s for s in design["realized_stages"]}
        entry = {"run": str(run), "requested_bit": bit, "origin_cycle": origin, "origin_phases": phases,
                 "design": binding(run / "design.json"), "result": binding(run / "result.json"),
                 "stage_files": blocks_digest(run), "arms": []}
        for group in design["groups"]:
            ep = stages[group["stage_ids"][0]]
            start = np.load(run / f"stage-{ep['stage_id']:02d}-initial.npy")
            traj = load_stage(run, ep["stage_id"], ep["controls"]["alpha"], ep["controls"]["log10_temperature"])
            hold = load_stage(run, group["stage_ids"][1], ALPHA_H, LOG10_T_H)
            arm = {"name": group["name"], "controls": ep["controls"], "lanes": []}
            for lane, phase in enumerate(phases):
                c = coordinate(traj[:, lane], refs, phase)
                s = displacement(start[lane], traj[:, lane])
                label, distance = nearest_label(hold[-1, lane], refs, (phase + len(traj) + len(hold)) % PERIOD)
                lane_entry = {"origin_phase": phase,
                              "coordinate_at_start": float(coordinate(start[lane][None], refs, phase - 1)[0]),
                              "passage_update_within_this_stage": passage_update(c, bit, float(PRIMARY)),
                              "final_nearest_cycle": label, "final_distance_to_that_cycle": distance,
                              "requested_bit_stored": label == bit, "tail": tail(c, s),
                              "bottleneck_by_phase": bottleneck(s, len(traj))}
                key = (bit, ep["controls"]["alpha"], ep["controls"]["log10_temperature"])
                if phase == 0 and group["name"] == "continue-perturbed-endpoint" and key in endpoint_record:
                    pilot_start, pilot_traj = endpoint_record[key]
                    gap = float(np.abs(start[lane] - pilot_traj[-1]).max())
                    lane_entry["start_minus_pilot_state_after_120_updates_max_abs"] = gap
                    if gap < 1e-9:
                        joined = np.concatenate([pilot_traj, traj[:, lane]], axis=0)
                        cj = coordinate(joined, refs, 0)
                        sj = displacement(pilot_start, joined)
                        lane_entry["joined_240_updates_at_this_endpoint"] = {
                            "passage_update": {str(t): passage_update(cj, bit, t) for t in THRESHOLDS},
                            "update_reaching_hold_unstable_orbit": level_crossing(
                                cj, unstable["coordinate_at_saved_phase"], bit, unstable["saved_phase"]),
                            "coordinate_at_updates_60_120_180_240": [float(cj[n - 1]) for n in (60, 120, 180, 240)],
                            "bottleneck_by_phase": bottleneck(sj, len(joined)), "tail": tail(cj, sj)}
                arm["lanes"].append(lane_entry)
            entry["arms"].append(arm)
        continuations.append(entry)

    receipt = {
        "schema": "public-write-passage-receipt-v1", "utc": now(), "model_calls": 0,
        "status": "zero_call_analysis_of_saved_records", "evidence_class": "adapted exploratory, post hoc",
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_note": "Plan, prediction file and this script were untracked working-tree files at that commit.",
        "script": binding(Path(__file__).resolve()), "plan": binding(ROOT / "docs" / "public-write-passage.md"),
        "prediction": binding(args.prediction), "reference_cycles": binding(args.reference_cycles),
        "definitions": {
            "coordinate": "projection of the state on the segment between the two reference cycles at the hold, "
                          "phase by phase: 0 on cycle 0, 1 on cycle 1",
            "passage_update": "first update from which the coordinate stays beyond the level three quarters of the "
                              "way to the requested cycle for six consecutive updates; also reported at one quarter "
                              "and one half",
            "update_reaching_hold_unstable_orbit": "update, sampled at the branch's saved phase and interpolated "
                                                   "linearly between two samples, at which the coordinate first "
                                                   "passes the projection of the hold's unstable orbit",
            "displacement": "norm of x(n+6) - x(n) during the endpoint stage",
            "bottleneck": "smallest displacement at each phase before the passage, refined by a three-point parabola "
                          "when the minimum is interior"},
        "rows": rows, "comparisons": comparisons, "continuations": continuations, "runs": run_bindings,
        "scope": "Saved trajectories of fixed pilot grids from one model at one hold. The write-1 predictions use a "
                 "coefficient taken from the companion study's multipliers near the fold; the nominal write-1 passage "
                 "was already known approximately when the method was fixed. The write-0 line is a two-parameter fit. "
                 "Not a located bifurcation beyond the receipts named in the plan, not a basin, not a reliability "
                 "estimate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=1)
        handle.write("\n")

    for r in rows:
        if not r["passage_lane"]:
            continue
        p = r["passage_update"]
        where = (f"beyond fold {r['alpha_beyond_fold']:+.3e}" if r.get("alpha_beyond_fold") is not None else
                 f"beyond fold {r['log10_temperature_beyond_fold']:+.3e}"
                 if r.get("log10_temperature_beyond_fold") is not None else "off line")
        pred = r.get("predicted_passage_update")
        slow = [e.get("minimum_interpolated", e.get("minimum")) for e in r["bottleneck_by_phase"]]
        inner = "".join("i" if e.get("interior") else "-" for e in r["bottleneck_by_phase"])
        at = [e.get("update_at_minimum") for e in r["bottleneck_by_phase"]]
        print(f"bit {r['requested_bit']} {r['profile'][:10]:>10} {r['group']:>13} {where:>24} stored {r['requested_bit_stored']!s:>5} "
              f"n* {p['0.25']}/{p['0.5']}/{p['0.75']} within {r['passage_within_endpoint_stage']!s:>5} "
              f"pred {'-' if pred is None else f'{pred:.1f}'} reach {r['update_reaching_hold_unstable_orbit']} "
              f"c120 {r['coordinate_at_update_120']:+.4f} "
              f"slow[{inner}] {min(v for v in slow if v is not None):.3e} at {at}")
        if "endpoint_stage_tail" in r:
            t = r["endpoint_stage_tail"]
            print(f"      tail: c {t['coordinate_one_period_earlier']:+.5f} -> {t['coordinate_at_end']:+.5f}; "
                  f"displacements {['%.3e' % v for v in t['last_four_displacements_same_phase']]} "
                  f"ratios {['%.4f' % v for v in t['ratios']]}")
    print(json.dumps({k: v for k, v in comparisons.items()}, indent=1)[:6000])
    for entry in continuations:
        for arm in entry["arms"]:
            for lane in arm["lanes"]:
                t = lane["tail"]
                print(f"continuation bit {entry['requested_bit']} {arm['name']:>28} phase {lane['origin_phase']} "
                      f"c start {lane['coordinate_at_start']:+.5f} end {t['coordinate_at_end']:+.5f} "
                      f"passage {lane['passage_update_within_this_stage']} stored {lane['requested_bit_stored']} "
                      f"disp {['%.3e' % v for v in t['last_four_displacements_same_phase']]} "
                      f"splice {lane.get('start_minus_pilot_state_after_120_updates_max_abs')}")
                j = lane.get("joined_240_updates_at_this_endpoint")
                if j:
                    print("      joined:", json.dumps({k: j[k] for k in (
                        "passage_update", "update_reaching_hold_unstable_orbit",
                        "coordinate_at_updates_60_120_180_240")}))
                    print("      joined slowest:", [(e.get("update_at_minimum"), e.get("interior"),
                                                    e.get("minimum_interpolated", e.get("minimum")),
                                                    e.get("relative_curvature_per_period_squared"))
                                                   for e in j["bottleneck_by_phase"]])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--branch-json", type=Path, required=True)
    p.add_argument("--fold-receipt", type=Path, required=True)
    p.add_argument("--branch-receipt", type=Path, required=True)
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=predict)
    m = sub.add_parser("measure")
    m.add_argument("--prediction", type=Path, required=True)
    m.add_argument("--reference-cycles", type=Path, required=True)
    m.add_argument("--run", type=Path, action="append", required=True)
    m.add_argument("--continuation", type=Path, action="append", default=[])
    m.add_argument("--output", type=Path, required=True)
    m.set_defaults(func=measure)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
