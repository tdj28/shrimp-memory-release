#!/usr/bin/env python3
"""Does the 30-degree pulse act on the slow coordinate as one fold close to its own cycle? Part A of
docs/public-toggle-fold.md, fixed before any quantity was computed.

Zero model calls. Reads the saved toggle check (batch 1: phases 0 to 5) and the sweep's 120-update pulse at the same
controls. C_p and v_p are defined exactly as in tools/analyze_public_toggle_mechanism.py; the receipt's phase-0 trace is
reproduced first.

  F1  one-update factors along the slow lines in the linear regime, by label; their product;
  F2  at the update with the smallest factor, the second-order part of the next deviation, divided by c^2;
  F3  pairs of label-0 visits six updates apart from all saved pulses at these controls: the one-hump ordering test;
  F5  each lane's first label-0 visit: fraction along v_p, image, and the on-curve image at the same coordinate.
Added after the first run, descriptive: F5b (each lane's off-line part at its first label-0 visit and its coefficient
three updates later) and F6 (where the check's eight neighbours end on the 30-degree slow line at label 0).
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_public_register_branch as arb  # noqa: E402  (binding)
import analyze_public_toggle_mechanism as mech  # noqa: E402  (group_file, load_group, WINDOW, PULSE)

PERIOD = 6
LINEAR = (1e-9, 1e-5)
ON_CURVE = 0.98
F2_RANGE = (1e-4, 1e-2)
FAR_SIDE = -0.001
ORDER_TOL = 0.05


def cycle_and_directions(P: np.ndarray):
    """As in analyze_public_toggle_mechanism.py: C_p is lane 0's last six states; v_p at label j is lane 0's latest
    deviation from C_p within mech.WINDOW, before the last period, normalised."""
    t_last = P.shape[0] - 1
    cp = {(t % PERIOD): P[t, 0] for t in range(t_last - PERIOD + 1, t_last + 1)}
    v = {}
    for j in range(PERIOD):
        ts = [t for t in range(t_last - PERIOD + 1) if t % PERIOD == j]
        inwin = [t for t in ts if mech.WINDOW[0] <= np.linalg.norm(P[t, 0] - cp[j]) <= mech.WINDOW[1]]
        e = P[inwin[-1], 0] - cp[j]
        v[j] = e / np.linalg.norm(e)
    return cp, v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-root", type=Path, required=True, help="toggle check batch-1 pod-returned directory")
    ap.add_argument("--sweep-root", type=Path, required=True, help="toggle sweep returned-pod directory")
    ap.add_argument("--mechanism-receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    bound: list[Path] = [a.mechanism_receipt]

    long = mech.load_group(mech.group_file(a.sweep_root, 26), bound)
    assert long["meta"]["pulse_steps"] == 120 and long["push_controls"] == mech.PULSE
    P = long["push"]
    cp, v = cycle_and_directions(P)
    runs = {0: P}                                      # start phase -> pulse states (updates, lane, 896)
    for k in range(1, PERIOD):
        g = mech.load_group(mech.group_file(a.check_root, k), bound)
        assert g["meta"]["start_phase"] == k and g["push_controls"] == mech.PULSE and g["meta"]["pulse_steps"] == 12
        runs[k] = g["push"]
    g0 = mech.load_group(mech.group_file(a.check_root, 0), bound)["push"]

    def label(k, t):
        return (k + t) % PERIOD

    def coef(k, t, lane):
        j = label(k, t)
        e = runs[k][t, lane] - cp[j]
        n = float(np.linalg.norm(e))
        c = float(v[j] @ e)
        return c, (abs(c) / n if n else None), n, e

    # Reproduce the mechanism receipt's phase-0 trace.
    old = json.loads(a.mechanism_receipt.read_text())["phase0_pulse_trace"]
    same_path = bool(np.array_equal(g0[:13], P[:13]))
    worst = max(abs(coef(0, row["update"], lane)[0] - row[f"lane{lane}_coefficient_along_v_p"])
                for row in old for lane in range(2))
    if not same_path or worst > 1e-15:
        raise SystemExit(f"cannot reproduce the mechanism trace (same path {same_path}, worst {worst})")

    # Per-update coefficients and fractions for every saved pulse at these controls (descriptive).
    traces = {}
    for k, R in runs.items():
        traces[str(k)] = [{"update": t, "label": label(k, t),
                           **{f"lane{l}": {"c": coef(k, t, l)[0], "fraction": coef(k, t, l)[1],
                                           "distance": coef(k, t, l)[2]} for l in range(2)}}
                          for t in range(min(R.shape[0], 25))]

    # F1: one-update factors in the linear regime (120-update pulse).
    factors = {j: [] for j in range(PERIOD)}
    for lane in range(2):
        for t in range(P.shape[0] - 1):
            c0, _, n0, _ = coef(0, t, lane)
            c1, _, n1, _ = coef(0, t + 1, lane)
            if LINEAR[0] <= n0 <= LINEAR[1] and LINEAR[0] <= n1 <= LINEAR[1]:
                factors[t % PERIOD].append(c1 / c0)
    f1 = {str(j): {"median": float(np.median(x)), "min": float(min(x)), "max": float(max(x)), "n": len(x)}
          for j, x in factors.items()}
    med = {j: float(np.median(x)) for j, x in factors.items()}
    k_f = min(med, key=lambda j: abs(med[j]))
    f1_summary = {"product_of_medians": float(np.prod(list(med.values()))), "fold_update_from_label": k_f,
                  "below_0.3": [j for j in med if abs(med[j]) < 0.3],
                  "holds": k_f in (0, 1) and sum(abs(m) < 0.3 for m in med.values()) == 1}

    # F2: second-order part at the fold update.
    visits = []
    for k, R in runs.items():
        for lane in range(2):
            for t in range(R.shape[0] - 1):
                if label(k, t) != k_f:
                    continue
                c, frac, n, _ = coef(k, t, lane)
                if not (F2_RANGE[0] <= abs(c) <= F2_RANGE[1] and frac >= ON_CURVE):
                    continue
                _, _, _, e1 = coef(k, t + 1, lane)
                q = (e1 - med[k_f] * c * v[(k_f + 1) % PERIOD]) / c ** 2
                visits.append({"start_phase": k, "lane": lane, "update": t, "c": c, "q": q})
    cos = [float(x["q"] @ y["q"] / (np.linalg.norm(x["q"]) * np.linalg.norm(y["q"])))
           for x, y in itertools.combinations(visits, 2)]
    norms = [float(np.linalg.norm(x["q"])) for x in visits]
    f2 = {"visits": [{k: x[k] for k in ("start_phase", "lane", "update", "c")} | {"q_norm": float(np.linalg.norm(x["q"]))}
                     for x in visits],
          "pairwise_cosine_min": min(cos) if cos else None, "pairwise_cosine_median": float(np.median(cos)) if cos else None,
          "norm_ratio_max": (max(norms) / min(norms)) if norms else None,
          "holds": bool(cos) and min(cos) >= 0.9 and max(norms) / min(norms) <= 2.0}

    # F3: pairs of label-0 visits six updates apart.
    pairs = []
    for k, R in runs.items():
        for lane in range(2):
            for t in range(R.shape[0] - PERIOD):
                if label(k, t) != 0:
                    continue
                s, fs, ns, _ = coef(k, t, lane)
                s2, fs2, _, _ = coef(k, t + PERIOD, lane)
                if ns < 1e-8:
                    continue
                pairs.append({"start_phase": k, "lane": lane, "update": t, "s": s, "image": s2,
                              "fraction": fs, "image_fraction": fs2,
                              "on_curve": fs >= ON_CURVE and fs2 >= ON_CURVE, "new": k != 0})
    on = [p for p in pairs if p["on_curve"]]
    near = sorted((p for p in on if p["s"] > 0), key=lambda p: p["s"])
    far = sorted((p for p in on if p["s"] < FAR_SIDE), key=lambda p: p["s"])
    viol = []
    for x, y in itertools.combinations(near, 2):           # x["s"] <= y["s"]
        if x["image"] > y["image"] + ORDER_TOL * max(abs(x["image"]), abs(y["image"])):
            viol.append(("near side", x["start_phase"], x["lane"], x["update"], y["start_phase"], y["lane"], y["update"]))
    for x, y in itertools.combinations(far, 2):
        if x["image"] < y["image"] - ORDER_TOL * max(abs(x["image"]), abs(y["image"])):
            viol.append(("far side", x["start_phase"], x["lane"], x["update"], y["start_phase"], y["lane"], y["update"]))
    wrong_sign = [(p["start_phase"], p["lane"], p["update"]) for p in near if not (0 < p["image"] < p["s"])]
    wrong_sign += [(p["start_phase"], p["lane"], p["update"]) for p in far if not p["image"] > 0]
    f3 = {"pairs": pairs, "on_curve": len(on), "near_side": len(near), "far_side": len(far),
          "middle": len([p for p in on if FAR_SIDE <= p["s"] <= 0]),
          "new_pairs_on_curve": len([p for p in on if p["new"]]),
          "order_violations": viol, "wrong_sign": wrong_sign, "holds": not viol and not wrong_sign}

    # F5: first label-0 visit of every lane; on-curve image at the same s by linear interpolation on the same side.
    def on_curve_image(s):
        side = [p for p in on if (p["s"] > 0) == (s > 0)]
        lo = [p for p in side if p["s"] <= s]
        hi = [p for p in side if p["s"] >= s]
        if not lo or not hi:
            return None
        p, q = max(lo, key=lambda p: p["s"]), min(hi, key=lambda p: p["s"])
        if q["s"] == p["s"]:
            return p["image"]
        return p["image"] + (q["image"] - p["image"]) * (s - p["s"]) / (q["s"] - p["s"])
    f5 = []
    for k in runs:
        for lane in range(2):
            first = [p for p in pairs if p["start_phase"] == k and p["lane"] == lane]
            if not first:
                continue
            p = min(first, key=lambda p: p["update"])
            ref = on_curve_image(p["s"]) if not p["on_curve"] else None
            f5.append({"start_phase": k, "lane": lane, "update": p["update"], "s": p["s"], "fraction": p["fraction"],
                       "image": p["image"], "on_curve": p["on_curve"], "on_curve_image_at_s": ref,
                       "image_minus_on_curve": (p["image"] - ref) if ref is not None else None})

    # Added after the first run (descriptive, post hoc). F5b: for each lane's first label-0 visit, the part of its
    # deviation off the slow line and its coefficient three updates later (label 3, after the fold update).
    f5b = []
    for k, R in runs.items():
        for lane in range(2):
            t = next(t for t in range(R.shape[0]) if label(k, t) == 0)
            c, frac, n, _ = coef(k, t, lane)
            c3 = coef(k, t + 3, lane)[0] if t + 3 < R.shape[0] else None
            off = n * float(np.sqrt(max(0.0, 1 - frac ** 2)))
            f5b.append({"start_phase": k, "lane": lane, "update": t, "s": c, "fraction": frac, "off_line_norm": off,
                        "label3_coefficient": c3, "label3_over_off_line": (c3 / off) if (c3 is not None and off) else None})
    # F6: where the eight angle/radius neighbours of the check end, on the 30-degree pulse's slow line at label 0.
    f6 = []
    for gid in range(6, 14):
        g = mech.load_group(mech.group_file(a.check_root, gid), bound)
        assert g["meta"]["pulse_steps"] == 12 and g["meta"]["start_phase"] == 0
        row = {"name": g["meta"]["name"], "mapping": g["meta"]["mapping"], "controls": list(g["push_controls"])}
        for lane in range(2):
            e = g["push"][12, lane] - cp[0]
            c = float(v[0] @ e)
            row[f"lane{lane}"] = {"s": c, "fraction": abs(c) / float(np.linalg.norm(e))}
        f6.append(row)

    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    receipt = {
        "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "kind": "public-toggle-fold",
        "plan": "docs/public-toggle-fold.md", "script": "tools/analyze_public_toggle_fold.py", "source_commit": commit,
        "working_tree_dirty": True, "model_calls": 0, "evidence_class": "exploratory, post hoc",
        "fixed_constants": {"linear": list(LINEAR), "on_curve": ON_CURVE, "f2_range": list(F2_RANGE),
                            "far_side": FAR_SIDE, "order_tolerance": ORDER_TOL},
        "reproduced_mechanism_trace": {"same_path": same_path, "worst_difference": worst},
        "F1": {"factors_by_label": f1, **f1_summary}, "F2": f2, "F3": f3, "F5": f5, "traces": traces,
        "F5b_added_after_first_run": f5b, "F6_added_after_first_run": f6,
        "inputs": [arb.binding(p) for p in sorted(set(bound))],
    }
    a.output.write_text(json.dumps(receipt, indent=1) + "\n")
    print("F1", json.dumps(f1), json.dumps(f1_summary))
    print("F2 visits", len(visits), "cos min %.3f median %.3f" % (f2["pairwise_cosine_min"] or 0, f2["pairwise_cosine_median"] or 0),
          "norm ratio %.2f" % (f2["norm_ratio_max"] or 0), "holds", f2["holds"])
    print("F3 on-curve", f3["on_curve"], "near", f3["near_side"], "far", f3["far_side"], "middle", f3["middle"],
          "new", f3["new_pairs_on_curve"], "violations", len(viol), "wrong sign", wrong_sign, "holds", f3["holds"])
    for p in sorted(on, key=lambda p: p["s"]):
        print("   s %+.6f -> %+.6f  phase %d lane %d update %3d  fractions %.3f %.3f" %
              (p["s"], p["image"], p["start_phase"], p["lane"], p["update"], p["fraction"], p["image_fraction"]))
    print("F5")
    for x in f5:
        print("  ", x)
    print("F5b (added after the first run)")
    for x in f5b:
        print("   phase %d lane %d u%d s %+.5f frac %.3f off %.5f label3 %s ratio %s" % (
            x["start_phase"], x["lane"], x["update"], x["s"], x["fraction"], x["off_line_norm"],
            "%+.5f" % x["label3_coefficient"] if x["label3_coefficient"] is not None else None,
            "%+.2f" % x["label3_over_off_line"] if x["label3_over_off_line"] is not None else None))
    print("F6 (added after the first run)")
    for x in f6:
        print("   %-32s %-8s lane0 %+.5f (%.3f)  lane1 %+.5f (%.3f)" % (x["name"], x["mapping"], x["lane0"]["s"],
              x["lane0"]["fraction"], x["lane1"]["s"], x["lane1"]["fraction"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
