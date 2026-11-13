#!/usr/bin/env python3
"""Score the prospective toggle timing test against the predictions in docs/public-toggle-timing.md. Zero model calls.

Reads the two groups of the timing run (18 and 24 updates at 30 degrees, radius 3.5, from phase 0), the pod's
120-update pulse at the same controls (whose first updates are the same paths), the reference cycles and the
companion continuation, and reports P0 (paths within 1e-12 of the pod's), P1 (24 updates: both lanes cycle 1), P2
(18 updates: the lane from cycle 0 reads cycle 1) and the measurement (18 updates: where the lane from cycle 1 goes).
The pulse cycle C_p and its slow direction at label 0 are recomputed exactly as in
tools/analyze_public_toggle_mechanism.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_public_register_branch as arb  # noqa: E402
import analyze_public_toggle_mechanism as mech  # noqa: E402

P0_TOLERANCE = 1e-12
PREDICTED = {18: (0.000209, 0.001040), 24: (0.000033, 0.000319)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch-dir", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--sweep-root", type=Path, required=True)
    ap.add_argument("--timing-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    bound: list[Path] = [args.branch_dir / "branch.json", args.branch_dir / "states.npy", args.reference]

    branch = json.loads((args.branch_dir / "branch.json").read_text())
    states = np.load(args.branch_dir / "states.npy")[:, 0, :]
    lam = np.array([p["lambda"] for p in branch["points"]])
    fold_1, fold_2 = (c["between_indices"] for c in branch["crossings"][1:3])
    i_mid = arb.bracket(lam, fold_1[1], fold_2[0], arb.ALPHA_H)
    _, saddle, _ = arb.interpolate(lam, states, i_mid, arb.ALPHA_H)
    with np.load(args.reference) as z:
        ref = z["states"]
    i_h = [i for i in range(len(lam)) if lam[i] == arb.ALPHA_H][0]
    saved_phase = int(np.argmin([np.linalg.norm(states[i_h] - ref[0][j]) for j in range(mech.PERIOD)]))

    # C_p and v_p at label 0, exactly as in the mechanism analysis (lane 0 of the pod's 120-update pulse).
    long = mech.load_group(mech.group_file(args.sweep_root, 26), bound)
    P = long["push"]
    t_last = P.shape[0] - 1
    cp0 = P[t_last - (t_last % mech.PERIOD), 0] if t_last % mech.PERIOD == 0 else None
    cp = {(t % mech.PERIOD): P[t, 0] for t in range(t_last - mech.PERIOD + 1, t_last + 1)}
    cp0 = cp[0]
    ts = [t for t in range(t_last - mech.PERIOD + 1) if t % mech.PERIOD == 0]
    inwin = [t for t in ts if mech.WINDOW[0] <= np.linalg.norm(P[t, 0] - cp0) <= mech.WINDOW[1]]
    e = P[inwin[-1], 0] - cp0
    v0 = e / np.linalg.norm(e)

    groups, verdict = {}, {}
    for gid in (0, 1):
        path = mech.group_file(args.timing_root, gid)
        g = mech.load_group(path, bound)
        meta, push = g["meta"], g["push"]
        D = int(meta["pulse_steps"])
        if g["push_controls"] != long["push_controls"] or meta["start_phase"] != 0:
            raise SystemExit(f"{path}: not the planned controls or phase")
        start_gap = float(np.abs(push[0] - P[0]).max())
        path_gap = [float(np.linalg.norm(push[t, lane] - P[t, lane])) for t in range(D + 1) for lane in (0, 1)]
        end_gap = [float(np.linalg.norm(push[D, lane] - P[D, lane])) for lane in (0, 1)]
        coef = [float(v0 @ (push[D, lane] - cp0)) for lane in (0, 1)]
        hold = mech.hold_dynamics(g["hold"], ref, saddle, saved_phase)
        finals = [h["final_cycle"] for h in hold]
        groups[meta["name"]] = {
            "pulse_updates": D, "runner_mapping": meta["mapping"], "runner_decoded_pair": meta["decoded_pair"],
            "diagnostics_passed": meta.get("diagnostics_passed"), "model_calls": meta.get("model_calls"),
            "start_equals_pod_start_max_abs": start_gap, "max_path_gap_to_pod": max(path_gap),
            "end_of_pulse_gap_to_pod": end_gap, "coefficients_along_v_p": coef,
            "predicted_coefficients": list(PREDICTED[D]), "recomputed_final_cycles": finals, "at_hold": hold,
        }
        verdict[D] = (max(end_gap) <= P0_TOLERANCE, finals, meta["mapping"])

    p0 = all(v[0] for v in verdict.values())
    f18, f24 = verdict[18][1], verdict[24][1]
    lane1_18 = f18[1]
    receipt = {
        "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": "public-toggle-timing-score", "plan": "docs/public-toggle-timing.md",
        "script": "tools/analyze_public_toggle_timing.py",
        "source_commit": subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                                        capture_output=True, text=True).stdout.strip(),
        "working_tree_dirty": True, "model_calls_in_this_script": 0,
        "verdicts": {
            "P0_paths_within_1e-12_of_pod": p0,
            "P1_24_updates_reset_to_1": f24 == [1, 1],
            "P2_18_updates_lane_from_cycle_0_reads_1": f18[0] == 1,
            "runner_and_recomputation_agree": all(
                (v[2] == {(1, 1): "reset1", (0, 0): "reset0", (1, 0): "toggle", (0, 1): "identity"}[tuple(v[1])])
                for v in verdict.values()),
            "measurement_18_updates_lane_from_cycle_1": lane1_18,
            "boundary_bracket_on_the_slow_line": ([0.000802, groups["t030-r3.5-d018-phase0"]["coefficients_along_v_p"][1]]
                                                  if lane1_18 == 0 else
                                                  [groups["t030-r3.5-d018-phase0"]["coefficients_along_v_p"][1], 0.002073]),
        },
        "groups": groups, "inputs": [arb.binding(p) for p in sorted(set(bound))],
    }
    args.output.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt["verdicts"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
