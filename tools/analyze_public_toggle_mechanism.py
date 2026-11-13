#!/usr/bin/env python3
"""Where the 30-degree toggle pulse leaves the two lanes, and why it exchanges them only from phase zero.

Zero model calls. Reads saved trajectories of the toggle check and the toggle sweep, the register's reference cycles
and the companion study's continuation (for the unstable orbit at H), and reports the quantities fixed in
docs/public-toggle-mechanism.md before they were computed:

  1. the pulse controls' own period-6 cycle C_p, from the 120-update pulse, with per-period ratios, the cosines between
     successive per-period deviations, and the slow direction v_p at each label;
  2. where every pulse at those controls ends: distance to C_p at label (k + D) mod 6 and at the nearest label, the
     lanes' separation, and each lane's coefficient along v_p;
  3. what happens back at H: the distance to the unstable orbit at each hold update at the branch's saved phase, its
     minimum, and the update after which the lane stays within 1e-6 of its final reference cycle.

Nothing is fitted and nothing is selected by outcome.
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
import analyze_public_register_branch as arb  # noqa: E402  (load_stage, interpolate, bracket, binding, leading)

PERIOD = 6
WINDOW = (1e-11, 1e-5)
SETTLE = 1e-6
PULSE = (0.07312150413133038, -0.1531613613499455)
HOLD = (arb.ALPHA_H, arb.LOG10_T_H)
CHECK_GROUPS = list(range(19))
SWEEP_GROUPS = [24, 25, 26, 14, 38]


def group_file(root: Path, gid: int) -> Path:
    hits = sorted(root.glob(f"shard-*/group-{gid:02d}.json"))
    if len(hits) != 1:
        raise SystemExit(f"expected one group-{gid:02d}.json under {root}, found {hits}")
    return hits[0]


def load_group(path: Path, bound: list[Path]) -> dict:
    g = json.loads(path.read_text())
    run = path.parent
    push_id, hold_id = g["stage_ids"]
    out = {"meta": g, "file": str(path)}
    for name, sid in (("push", push_id), ("hold", hold_id)):
        blocks, alpha, log10_t, files = arb.load_stage(run, sid)
        init_path = run / f"stage-{sid:02d}-initial.npy"
        init = np.load(init_path)
        out[name] = np.concatenate([init[None], blocks], 0)
        out[name + "_controls"] = (alpha, log10_t)
        bound.extend(files + [init_path])
    bound.append(path)
    out["carry_gap"] = float(np.abs(out["hold"][0] - out["push"][-1]).max())
    if out["hold_controls"] != HOLD:
        raise SystemExit(f"{path}: hold stage not at H")
    if tuple(g["controls"].values()) != out["push_controls"]:
        raise SystemExit(f"{path}: pulse controls differ from the group record")
    return out


def hold_dynamics(hold: np.ndarray, ref: np.ndarray, saddle: np.ndarray, saved_phase: int) -> list[dict]:
    lanes = []
    m_last = hold.shape[0] - 1
    for lane in range(hold.shape[1]):
        end = hold[m_last, lane]
        d = np.array([[np.linalg.norm(end - ref[c][j]) for j in range(PERIOD)] for c in range(2)])
        c, j = np.unravel_index(int(np.argmin(d)), d.shape)
        phase = [(int(j) + m - m_last) % PERIOD for m in range(m_last + 1)]
        to_final = np.array([np.linalg.norm(hold[m, lane] - ref[c][phase[m]]) for m in range(m_last + 1)])
        above = np.nonzero(to_final >= SETTLE)[0]
        settled = int(above[-1] + 1) if len(above) else 0
        samples = [(m, float(np.linalg.norm(hold[m, lane] - saddle))) for m in range(m_last + 1) if phase[m] == saved_phase]
        m_min, d_min = min(samples, key=lambda x: x[1])
        lanes.append({
            "final_cycle": int(c), "final_distance": float(d[c, j]),
            "first_six_saddle_distances": [{"hold_update": m, "distance": x} for m, x in samples[:6]],
            "closest_saddle_distance": d_min, "closest_at_hold_update": m_min,
            "settled_within_1e-6_from_hold_update": settled,
        })
    return lanes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch-dir", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True, help="reference-cycles.npz of the audited XOR chain")
    ap.add_argument("--check-root", type=Path, required=True, help="toggle check batch-1 pod-returned directory")
    ap.add_argument("--sweep-root", type=Path, required=True, help="toggle sweep returned-pod directory")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    bound: list[Path] = []

    # The unstable orbit at H, at the branch's saved phase, exactly as in the register-branch analysis.
    branch = json.loads((args.branch_dir / "branch.json").read_text())
    states = np.load(args.branch_dir / "states.npy")[:, 0, :]
    bound += [args.branch_dir / "branch.json", args.branch_dir / "states.npy", args.reference]
    lam = np.array([p["lambda"] for p in branch["points"]])
    fold_1, fold_2 = (c["between_indices"] for c in branch["crossings"][1:3])
    i_mid = arb.bracket(lam, fold_1[1], fold_2[0], arb.ALPHA_H)
    saddle_lin, saddle, _ = arb.interpolate(lam, states, i_mid, arb.ALPHA_H)
    with np.load(args.reference) as z:
        ref = z["states"]
    i_h = [i for i in range(len(lam)) if lam[i] == arb.ALPHA_H][0]
    saved_phase = int(np.argmin([np.linalg.norm(states[i_h] - ref[0][j]) for j in range(PERIOD)]))
    sep_saved = float(np.linalg.norm(ref[1][saved_phase] - ref[0][saved_phase]))
    saddle_info = {
        "saved_phase": saved_phase, "cycle_separation_at_saved_phase": sep_saved,
        "distance_to_cycle_0": float(np.linalg.norm(saddle - ref[0][saved_phase])),
        "distance_to_cycle_1": float(np.linalg.norm(saddle - ref[1][saved_phase])),
        "linear_minus_cubic": float(np.linalg.norm(saddle - saddle_lin)),
    }

    groups = {}
    for gid in CHECK_GROUPS:
        groups[f"check-{gid:02d}"] = load_group(group_file(args.check_root, gid), bound)
    for gid in SWEEP_GROUPS:
        groups[f"sweep-{gid:02d}"] = load_group(group_file(args.sweep_root, gid), bound)

    # 1. The pulse cycle C_p from the 120-update pulse at the pulse controls (started at phase 0).
    long = groups["sweep-26"]
    assert long["meta"]["pulse_steps"] == 120 and long["push_controls"] == PULSE
    P = long["push"]                                     # (121, 2, 896); t = 0..120
    t_last = P.shape[0] - 1
    cp = {(t % PERIOD): P[t, 0] for t in range(t_last - PERIOD + 1, t_last + 1)}
    lag6 = [float(np.linalg.norm(P[t, 0] - P[t - PERIOD, 0])) for t in range(t_last - PERIOD + 1, t_last + 1)]
    lanes_agree = float(np.abs(P[-PERIOD:, 0] - P[-PERIOD:, 1]).max())
    ratios, cosines, v_p = [], [], {}
    for lane in range(2):
        for j in range(PERIOD):
            ts = [t for t in range(t_last - PERIOD + 1) if t % PERIOD == j]
            dev = {t: P[t, lane] - cp[j] for t in ts}
            inwin = [t for t in ts if WINDOW[0] <= np.linalg.norm(dev[t]) <= WINDOW[1]]
            for t in inwin:
                if t + PERIOD in inwin:
                    a, b = dev[t], dev[t + PERIOD]
                    ratios.append(float(np.linalg.norm(b) / np.linalg.norm(a)))
                    cosines.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
            if lane == 0 and inwin:
                e = dev[inwin[-1]]
                v_p[j] = e / np.linalg.norm(e)
    cycle = {
        "lag_six_residuals": lag6, "lanes_agree_over_last_period": lanes_agree,
        "per_period_ratio_median": float(np.median(ratios)) if ratios else None,
        "per_period_ratio_range": [float(min(ratios)), float(max(ratios))] if ratios else None,
        "successive_deviation_cosine_median": float(np.median(cosines)) if cosines else None,
        "successive_deviation_cosine_range": [float(min(cosines)), float(max(cosines))] if cosines else None,
        "ratios_used": len(ratios), "labels_with_slow_direction": sorted(v_p),
        "distance_to_reference_cycles_by_label": {
            str(j): {"cycle_0": float(min(np.linalg.norm(cp[j] - ref[0][i]) for i in range(PERIOD))),
                     "cycle_1": float(min(np.linalg.norm(cp[j] - ref[1][i]) for i in range(PERIOD)))}
            for j in range(PERIOD)},
        "distance_label0_to_saddle_neighbourhood_note": "C_p is at the pulse controls; distances to H's cycles are context",
    }

    def landing(gname: str) -> dict:
        g = groups[gname]
        meta, push = g["meta"], g["push"]
        k = int(meta.get("start_phase") or 0)
        D = int(meta["pulse_steps"])
        label = (k + D) % PERIOD
        lanes = []
        for lane in range(2):
            x = push[D, lane]
            dists = [float(np.linalg.norm(x - cp[j])) for j in range(PERIOD)]
            e = x - cp[label]
            coef = float(v_p[label] @ e) if label in v_p else None
            lanes.append({"distance_to_cp_at_expected_label": dists[label], "nearest_label": int(np.argmin(dists)),
                          "distance_to_cp_at_nearest_label": min(dists), "coefficient_along_v_p": coef,
                          "fraction_of_deviation_along_v_p": (abs(coef) / float(np.linalg.norm(e))) if coef else None})
        return {"start_phase": k, "pulse_updates": D, "end_label": label,
                "start_separation": float(np.linalg.norm(push[0, 1] - push[0, 0])),
                "end_separation": float(np.linalg.norm(push[D, 1] - push[D, 0])), "lanes": lanes}

    same_controls = [n for n, g in groups.items() if g["push_controls"] == PULSE]
    report = {}
    for name, g in groups.items():
        meta = g["meta"]
        entry = {"name": meta["name"], "mapping": meta["mapping"], "decoded_pair": meta.get("decoded_pair"),
                 "pulse_controls": list(g["push_controls"]), "pulse_updates": meta["pulse_steps"],
                 "start_phase": meta.get("start_phase", 0), "carry_gap": g["carry_gap"],
                 "start_distance_to_reference": [
                     float(np.linalg.norm(g["push"][0, lane] - ref[lane][int(meta.get("start_phase") or 0)]))
                     for lane in range(2)],
                 "at_hold": hold_dynamics(g["hold"], ref, saddle, saved_phase)}
        if name in same_controls:
            entry["landing"] = landing(name)
        report[name] = entry

    # Every update of the phase-0 12-update pulse.
    g0 = groups["check-00"]["push"]
    trace = []
    for t in range(g0.shape[0]):
        j = t % PERIOD
        row = {"update": t, "label": j, "separation": float(np.linalg.norm(g0[t, 1] - g0[t, 0]))}
        for lane in range(2):
            e = g0[t, lane] - cp[j]
            row[f"lane{lane}_distance_to_cp"] = float(np.linalg.norm(e))
            if j in v_p:
                row[f"lane{lane}_coefficient_along_v_p"] = float(v_p[j] @ e)
        trace.append(row)

    # Added after the first run (see the plan's results): the same two paths continue in the 48- and 120-update pulses,
    # which share the 12-update pulse's first 12 updates bit for bit. Their coefficients along v_p at label 0, every
    # period, show where the lanes would be if the dials returned to H later.
    shared = {n: bool(np.array_equal(groups[n]["push"][:13], P[:13])) for n in ("check-00", "sweep-24", "sweep-25")}
    continuation = []
    for t in range(0, t_last - PERIOD + 1, PERIOD):
        row = {"update": t}
        for lane in range(2):
            e = P[t, lane] - cp[0]
            c = float(v_p[0] @ e)
            row[f"lane{lane}_coefficient_along_v_p"] = c
            row[f"lane{lane}_fraction_along_v_p"] = abs(c) / float(np.linalg.norm(e)) if np.linalg.norm(e) else None
        continuation.append(row)

    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    receipt = {
        "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": "public-toggle-mechanism", "plan": "docs/public-toggle-mechanism.md",
        "script": "tools/analyze_public_toggle_mechanism.py", "source_commit": commit, "working_tree_dirty": True,
        "model_calls": 0, "evidence_class": "adapted exploratory, post hoc",
        "fixed_constants": {"window": list(WINDOW), "settle": SETTLE, "period": PERIOD},
        "unstable_orbit_at_hold": saddle_info, "pulse_cycle": cycle, "groups": report, "phase0_pulse_trace": trace,
        "first_12_updates_shared_with_120_update_pulse": shared, "label0_continuation_every_period": continuation,
        "inputs": [arb.binding(p) for p in sorted(set(bound))],
    }
    args.output.write_text(json.dumps(receipt, indent=1) + "\n")
    print("receipt", args.output, "inputs", len(receipt["inputs"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
