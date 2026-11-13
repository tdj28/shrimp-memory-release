#!/usr/bin/env python3
"""Score Part C of docs/public-toggle-fold.md (zero-offset counterfactual to a readout). Zero model calls.

Z0: the first six pulse updates reproduce Part B's lambda = 0 path within 1e-12. Z1: no exchange; both lanes read
cycle 1. Also reports each lane's coefficient along C_p's slow line at updates 6 and 12, the closest approach to the
unstable orbit at H and the settle update, computed as in tools/analyze_public_toggle_mechanism.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_public_register_branch as arb  # noqa: E402
import analyze_public_toggle_fold as fold  # noqa: E402
import analyze_public_toggle_mechanism as mech  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="Part C run directory (shard-00)")
    ap.add_argument("--part-b-run", type=Path, required=True, help="Part B run directory (shard-00 is lambda = 0)")
    ap.add_argument("--sweep-root", type=Path, required=True)
    ap.add_argument("--branch-dir", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    bound: list[Path] = [a.reference, a.branch_dir / "branch.json", a.branch_dir / "states.npy"]
    problems = []

    d = a.run / "shard-00"
    manifest = json.loads((d / "manifest.json").read_text())
    for x in manifest["files"]:
        if sha(d / x["path"]) != x["sha256"]:
            problems.append(f"hash mismatch {x['path']}")
    r = json.loads((d / "result.json").read_text())
    g = r["groups"][0]
    if r["status"] != "completed" or g["status"] != "completed" or not g["diagnostics_passed"]:
        problems.append(f"status {r['status']} / {g['status']}")
    push_id, hold_id = g["stage_ids"]
    def stage(sid):
        blocks, alpha, logt, files = arb.load_stage(d, sid)
        init = np.load(d / f"stage-{sid:02d}-initial.npy")
        bound.extend(files + [d / f"stage-{sid:02d}-initial.npy"])
        return np.concatenate([init[None], blocks], 0), (alpha, logt)
    push, pc = stage(push_id)
    hold, hc = stage(hold_id)
    if pc != mech.PULSE or hc != mech.HOLD:
        problems.append(f"controls {pc} / {hc}")
    if hold[0].tobytes() != push[-1].tobytes():
        problems.append("pulse-to-hold carry differs")

    # Z0 against Part B's lambda = 0 path.
    bd = a.part_b_run / "shard-00"
    bblocks, _, _, bfiles = arb.load_stage(bd, 0, 6)
    binit = np.load(bd / "stage-00-initial.npy")
    bound.extend(bfiles + [bd / "stage-00-initial.npy"])
    bpath = np.concatenate([binit[None], bblocks], 0)
    z0_gap = float(np.abs(push[:7] - bpath).max())

    # Slow-line coefficients (C_p and v_p exactly as in the mechanism analysis).
    P = mech.load_group(mech.group_file(a.sweep_root, 26), bound)["push"]
    cp, v = fold.cycle_and_directions(P)
    coefs = {}
    for t in (0, 6, 12):
        coefs[str(t)] = []
        for lane in range(2):
            e = push[t, lane] - cp[t % 6]
            c = float(v[t % 6] @ e)
            coefs[str(t)].append({"s": c, "fraction": abs(c) / float(np.linalg.norm(e))})

    # Hold dynamics: unstable orbit at H at the branch's saved phase, as in the mechanism analysis.
    branch = json.loads((a.branch_dir / "branch.json").read_text())
    states = np.load(a.branch_dir / "states.npy")[:, 0, :]
    lam = np.array([p["lambda"] for p in branch["points"]])
    f1, f2 = (c["between_indices"] for c in branch["crossings"][1:3])
    _, saddle, _ = arb.interpolate(lam, states, arb.bracket(lam, f1[1], f2[0], arb.ALPHA_H), arb.ALPHA_H)
    with np.load(a.reference) as z:
        ref = z["states"]
    i_h = [i for i in range(len(lam)) if lam[i] == arb.ALPHA_H][0]
    saved_phase = int(np.argmin([np.linalg.norm(states[i_h] - ref[0][j]) for j in range(6)]))
    at_hold = mech.hold_dynamics(hold, ref, saddle, saved_phase)

    mapping = g.get("mapping")
    pair = g.get("decoded_pair")
    z1 = mapping == "reset1"
    receipt = {
        "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "kind": "public-toggle-offset-readout",
        "plan": "docs/public-toggle-fold.md (Part C)", "script": "tools/analyze_public_toggle_offset_readout.py",
        "source_commit": subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                                        capture_output=True, text=True).stdout.strip(),
        "working_tree_dirty": True, "model_calls_scored": r["model_calls"], "audit_problems": problems,
        "audit_passed": not problems, "runner_mapping": mapping, "runner_decoded_pair": pair,
        "recomputed_final_cycles": [x["final_cycle"] for x in at_hold],
        "Z0": {"max_gap_to_part_b_lambda0_path": z0_gap, "holds": z0_gap <= 1e-12},
        "Z1": {"no_exchange_both_cycle_1": z1, "holds": z1}, "slow_line_coefficients": coefs, "at_hold": at_hold,
        "inputs": [arb.binding(p) for p in sorted(set(bound))],
    }
    a.output.write_text(json.dumps(receipt, indent=1) + "\n")
    print("audit passed", not problems, problems[:4], "calls", r["model_calls"])
    print("runner mapping", mapping, pair, "| recomputed final cycles", receipt["recomputed_final_cycles"])
    print("Z0 gap %.2e holds %s | Z1 holds %s" % (z0_gap, z0_gap <= 1e-12, z1))
    for t, rows in coefs.items():
        print("  update", t, ["%+.6f (%.3f)" % (x["s"], x["fraction"]) for x in rows])
    for lane, x in enumerate(at_hold):
        print("  lane", lane, "final cycle", x["final_cycle"], "closest to unstable orbit %.2e" % x["closest_saddle_distance"],
              "settled from hold update", x["settled_within_1e-6_from_hold_update"])
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
