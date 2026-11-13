#!/usr/bin/env python3
"""Post hoc readings of the write-passage receipt. Zero model calls. No saved state is opened.

Written after `docs/receipts/public-write-passage.json` had been read. Nothing here was predicted or fixed before
the measurement, and nothing here changes a criterion of the plan (docs/public-write-passage.md). The script reads
the measurement receipt, the prediction file and the companion study's saved branch, and reports three things that
the fixed comparisons left open:

  1. the control at which the smallest per-period displacement extrapolates to zero when only the settings nearest
     the fold are used (the two nearest and the three nearest, every phase, nothing selected);
  2. the settling rate of the two write-1 lanes that stopped short of the fold, against the multiplier that the fold
     coefficient gives and against the multiplier the companion's branch records at that strength;
  3. the write-0 coefficient from the curvature of the bottleneck in the joined 240-update continuation at the
     setting nearest the fold.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_public_write_passage as W  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRANSIENT_UPDATES = 2 * W.PERIOD   # a minimum inside the first two periods is flagged, not removed


def unique_on_line(rows, key):
    out = {}
    for row in rows:
        if row["passage_lane"] and row["on_recorded_line"] and row.get(key) is not None:
            out.setdefault((row["alpha"], row["log10_temperature"]), row)
    return sorted(out.values(), key=lambda r: r[key])


def nearest_zero(points, fold, count):
    """Straight line through the `count` settings nearest the fold; where it reaches zero."""
    chosen = sorted(points, key=lambda p: abs(p["beyond"]))[:count]
    if len(chosen) < count:
        return None
    x = np.array([p["control"] for p in chosen])
    y = np.array([p["minimum"] for p in chosen])
    slope, intercept = np.polyfit(x - fold, y, 1)
    zero = fold - intercept / slope
    fitted = slope * (x - fold) + intercept
    return {"settings": [p["group"] for p in chosen], "beyond_fold": [p["beyond"] for p in chosen],
            "update_at_minimum": [p["update"] for p in chosen],
            "any_minimum_inside_first_two_periods": any(p["update"] < TRANSIENT_UPDATES for p in chosen),
            "slope": float(slope), "zero_crossing": float(zero), "zero_crossing_minus_independent_fold": float(zero - fold),
            "max_abs_residual_over_largest": float(np.max(np.abs(y - fitted)) / np.max(y)) if count > 2 else 0.0}


def fold_from_nearest(rows, key, control, fold, extra=None):
    out = []
    for phase in range(W.PERIOD):
        points = []
        for row in rows:
            entry = row["bottleneck_by_phase"][phase] if phase < len(row["bottleneck_by_phase"]) else None
            if entry and entry.get("interior"):
                points.append({"group": row["group"], "control": row[control], "beyond": row[key],
                               "minimum": entry["minimum_interpolated"], "update": entry["update_at_minimum"]})
        item = {"phase": phase, "pilot_stage_only": {str(k): nearest_zero(points, fold, k) for k in (2, 3)}}
        if extra is not None:
            entry = extra["bottleneck_by_phase"][phase]
            if entry.get("interior"):
                joined = points + [{"group": extra["group"], "control": extra["control"], "beyond": extra["beyond"],
                                    "minimum": entry["minimum_interpolated"], "update": entry["update_at_minimum"]}]
                item["with_joined_continuation_at_nearest_setting"] = {
                    str(k): nearest_zero(joined, fold, k) for k in (2, 3)}
        out.append(item)
    return out


def branch_multiplier(lam, mu, first_sheet_last, alpha):
    """Leading multiplier of the first sheet at one strength: linear between neighbours, cubic through four."""
    idx = np.arange(first_sheet_last + 1)
    order = np.argsort(lam[idx])
    x, y = lam[idx][order], mu[idx][order]
    if not x[0] <= alpha <= x[-1]:
        return None
    hi = int(np.searchsorted(x, alpha))
    lo = hi - 1
    linear = float(y[lo] + (y[hi] - y[lo]) * (alpha - x[lo]) / (x[hi] - x[lo]))
    start = min(max(lo - 1, 0), len(x) - 4)
    cubic = float(np.polyval(np.polyfit(x[start:start + 4] - alpha, y[start:start + 4], 3), 0.0))
    return {"neighbours_alpha": [float(x[lo]), float(x[hi])], "neighbours_multiplier": [float(y[lo]), float(y[hi])],
            "linear": linear, "cubic_through_four": cubic}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--branch-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    prediction = json.loads(args.prediction.read_text())
    lam, mu = W.load_branch(args.branch_json)
    ab = prediction["write1_coefficient_ab_per_unit_alpha"]
    alpha_fold = prediction["first_sheet_fold"]["alpha_fold"]
    t_fold = prediction["fold_on_hold_strength_line_log10_temperature"]

    w1 = unique_on_line([r for r in receipt["rows"] if r["requested_bit"] == 1], "alpha_beyond_fold")
    w0 = unique_on_line([r for r in receipt["rows"] if r["requested_bit"] == 0], "log10_temperature_beyond_fold")

    # joined 240-update continuation of the failed write of 0 (origin phase 0, same endpoint as the pilot)
    joined = None
    for cont in receipt["continuations"]:
        if cont["requested_bit"] != 0:
            continue
        for arm in cont["arms"]:
            if arm["name"] != "continue-perturbed-endpoint":
                continue
            for lane in arm["lanes"]:
                if "joined_240_updates_at_this_endpoint" in lane:
                    joined = {"group": "continued -0.1536 (240 updates)", "control": arm["controls"]["log10_temperature"],
                              "beyond": arm["controls"]["log10_temperature"] - t_fold,
                              "start_minus_pilot_state_after_120_updates_max_abs":
                                  lane["start_minus_pilot_state_after_120_updates_max_abs"],
                              "bottleneck_by_phase": lane["joined_240_updates_at_this_endpoint"]["bottleneck_by_phase"]}
    if joined is None:
        raise SystemExit("joined continuation not found in the receipt")

    settled = []
    for row in w1:
        if row["alpha_beyond_fold"] >= 0:
            continue
        short = -row["alpha_beyond_fold"]
        ratios = row["endpoint_stage_tail"]["ratios"]
        settled.append({"group": row["group"], "alpha": row["alpha"], "alpha_short_of_fold": short,
                        "measured_per_period_ratios_of_displacement": ratios,
                        "measured_last_ratio": ratios[-1],
                        "from_fold_coefficient_1_minus_2_sqrt_ab_D": 1.0 - 2.0 * math.sqrt(ab * short),
                        "from_branch_at_this_alpha": branch_multiplier(lam, mu, 16, row["alpha"])})

    curvature = []
    for entry in joined["bottleneck_by_phase"]:
        if entry.get("interior"):
            kappa = entry["relative_curvature_per_period_squared"]
            curvature.append({"phase": entry["phase"], "update_at_minimum": entry["update_at_minimum"],
                              "minimum_interpolated": entry["minimum_interpolated"],
                              "relative_curvature_per_period_squared": kappa, "ab": kappa / joined["beyond"],
                              "bottleneck_length_updates_pi_over_sqrt_curvature": W.PERIOD * math.pi / math.sqrt(kappa)})

    out = {
        "schema": "public-write-passage-addendum-v1",
        "utc": W.now(),
        "model_calls": 0,
        "status": "zero_call_arithmetic_on_receipts",
        "evidence_class": "post hoc; written after the measurement receipt was read; nothing here was predicted or "
                          "fixed beforehand, and no criterion of the plan is changed by it",
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "script": W.binding(Path(__file__).resolve()),
        "inputs": {"receipt": W.binding(args.receipt), "prediction": W.binding(args.prediction),
                   "branch": W.binding(args.branch_json)},
        "transient_flag_updates": TRANSIENT_UPDATES,
        "independent_fold_positions": {"alpha_on_hold_temperature_line": alpha_fold,
                                       "log10_temperature_on_hold_strength_line": t_fold},
        "write1_fold_from_nearest_settings": fold_from_nearest(w1, "alpha_beyond_fold", "alpha", alpha_fold),
        "write0_fold_from_nearest_settings": fold_from_nearest(w0, "log10_temperature_beyond_fold",
                                                               "log10_temperature", t_fold, joined),
        "write1_lanes_settled_short_of_fold": settled,
        "write0_coefficient_from_joined_continuation_curvature": {
            "log10_temperature_beyond_fold": joined["beyond"],
            "start_minus_pilot_state_after_120_updates_max_abs": joined["start_minus_pilot_state_after_120_updates_max_abs"],
            "by_phase": curvature, "ab_median": float(np.median([c["ab"] for c in curvature])) if curvature else None,
            "ab_from_passage_line_slope": receipt["comparisons"]["write0_line"]["ab_from_slope"]},
        "scope": "Arithmetic on two receipts and the companion's recorded multipliers. The nearest-setting lines are "
                 "two- and three-point extrapolations chosen after the fixed four- and five-point lines had missed; "
                 "they are reported for every phase and both counts, and are not a test.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(out, handle, indent=1)
        handle.write("\n")

    for name, spacing in (("write1_fold_from_nearest_settings", 2.5e-6), ("write0_fold_from_nearest_settings", 2.5e-5)):
        print(name, f"(setting spacing {spacing:g})")
        for item in out[name]:
            for variant in ("pilot_stage_only", "with_joined_continuation_at_nearest_setting"):
                if variant not in item:
                    continue
                for k, fit in item[variant].items():
                    if fit is None:
                        continue
                    print(f"  phase {item['phase']} {variant[:22]:22} k={k} zero-fold {fit['zero_crossing_minus_independent_fold']:+.3e} "
                          f"({fit['zero_crossing_minus_independent_fold'] / spacing:+.4f} spacings) "
                          f"transient {fit['any_minimum_inside_first_two_periods']} resid {fit['max_abs_residual_over_largest']:.4f} "
                          f"{fit['settings']}")
    print("settled lanes")
    for item in settled:
        b = item["from_branch_at_this_alpha"]
        print(f"  {item['group']:>13} short {item['alpha_short_of_fold']:.4e} measured {item['measured_last_ratio']:.4f} "
              f"fold coefficient {item['from_fold_coefficient_1_minus_2_sqrt_ab_D']:.4f} "
              f"branch linear {b['linear']:.4f} cubic {b['cubic_through_four']:.4f} neighbours {b['neighbours_multiplier']}")
    print("write-0 coefficient from joined continuation curvature")
    for item in curvature:
        print(f"  phase {item['phase']} n {item['update_at_minimum']} kappa {item['relative_curvature_per_period_squared']:.6f} "
              f"ab {item['ab']:.1f} length {item['bottleneck_length_updates_pi_over_sqrt_curvature']:.0f} updates")
    print("  median", out["write0_coefficient_from_joined_continuation_curvature"]["ab_median"],
          "line slope", out["write0_coefficient_from_joined_continuation_curvature"]["ab_from_passage_line_slope"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
