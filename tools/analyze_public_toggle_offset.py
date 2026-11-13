#!/usr/bin/env python3
"""Audit and score Part B of docs/public-toggle-fold.md (the offset test). Zero model calls.

Per shard: rehash the manifest, check the start equals the bound start file and the controls equal the 30-degree
pulse, rebuild the six-update path from the saved blocks. Then score B0 (lambda = 1 reproduces the pod's phase-0
path within 1e-12) and B1 (the landing of the lane from cycle 1 at update 6 decreases monotonically in lambda, and at
lambda = 0 lies within 0.0005 of C_p with s > 0), and report coefficients along the slow lines at every update.
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
import analyze_public_register_branch as arb  # noqa: E402  (binding, load_stage)
import analyze_public_toggle_fold as fold  # noqa: E402  (cycle_and_directions)
import analyze_public_toggle_mechanism as mech  # noqa: E402  (group_file, load_group, PULSE)

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
PERIOD = 6


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="Part B run directory (starts/, shard-00 .. shard-04)")
    ap.add_argument("--check-root", type=Path, required=True)
    ap.add_argument("--sweep-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    bound: list[Path] = []
    problems: list[str] = []

    starts_file = a.run / "starts" / "starts.npz"
    meta = json.loads((a.run / "starts" / "starts.json").read_text())
    if sha(starts_file) != meta["starts_sha256"]:
        problems.append("start file hash differs from its record")
    with np.load(starts_file) as z:
        starts = z["states"]
    bound += [starts_file, a.run / "starts" / "starts.json"]

    P = mech.load_group(mech.group_file(a.sweep_root, 26), bound)["push"]
    cp, v = fold.cycle_and_directions(P)
    pod = mech.load_group(mech.group_file(a.check_root, 0), bound)["push"]       # pod's phase-0 12-update pulse

    paths, calls = {}, 0
    for gid, lam in enumerate(LAMBDAS):
        d = a.run / f"shard-{gid:02d}"
        manifest = json.loads((d / "manifest.json").read_text())
        files = {x["path"]: x["sha256"] for x in manifest["files"]}
        if set(files) != {p.name for p in d.iterdir() if p.is_file()} - {"manifest.json"}:
            problems.append(f"{d.name}: manifest file set differs")
        for name, h in files.items():
            if sha(d / name) != h:
                problems.append(f"{d.name}: hash mismatch {name}")
        r = json.loads((d / "result.json").read_text())
        calls += r["model_calls"]
        g = r["groups"][0]
        if r["status"] != "completed" or g["status"] != "completed" or not g["diagnostics_passed"]:
            problems.append(f"{d.name}: status {r['status']} / {g['status']}, diagnostics {g.get('diagnostics_passed')}")
        if g["offset_lambda"] != lam or g["group_id"] != gid:
            problems.append(f"{d.name}: group is not lambda {lam}")
        start = np.load(d / f"group-{gid:03d}-start.npy")
        init = np.load(d / f"stage-{gid:02d}-initial.npy")
        blocks, alpha, log10_t, bfiles = arb.load_stage(d, gid, 6)
        bound += [d / "manifest.json", d / "result.json", d / f"group-{gid:03d}-start.npy",
                  d / f"stage-{gid:02d}-initial.npy", *bfiles]
        if start.tobytes() != starts[gid].tobytes() or init.tobytes() != start.tobytes():
            problems.append(f"{d.name}: start differs from the start file")
        if (alpha, log10_t) != mech.PULSE:
            problems.append(f"{d.name}: controls {(alpha, log10_t)} are not the pulse's")
        X = np.concatenate([init[None], blocks], 0)
        if X.shape != (7, 2, 896):
            problems.append(f"{d.name}: path shape {X.shape}")
        paths[lam] = X

    def coef(x, t):
        e = x - cp[t % PERIOD]
        c = float(v[t % PERIOD] @ e)
        n = float(np.linalg.norm(e))
        return c, abs(c) / n if n else None

    b0_gap = float(np.abs(paths[1.0] - pod[:7]).max())
    b0 = b0_gap <= 1e-12
    table = []
    for lam in LAMBDAS:
        X = paths[lam]
        table.append({"lambda": lam, **{f"lane{l}": [{"update": t, "s": coef(X[t, l], t)[0],
                                                      "fraction": coef(X[t, l], t)[1]} for t in range(7)]
                                       for l in range(2)}})
    s6 = {l: [row[f"lane{l}"][6]["s"] for row in table] for l in range(2)}
    s3 = {l: [row[f"lane{l}"][3]["s"] for row in table] for l in range(2)}
    mono = all(s6[1][i] > s6[1][i + 1] for i in range(len(LAMBDAS) - 1))
    near = 0 < s6[1][0] <= 5e-4
    linear = {l: [s6[l][i] - (s6[l][0] + LAMBDAS[i] * (s6[l][-1] - s6[l][0])) for i in range(len(LAMBDAS))]
              for l in range(2)}
    b1 = {"monotone_decreasing": mono, "lambda0_within_5e-4_positive": near, "holds": mono and near,
          "lambda0_landing": s6[1][0], "landings": s6[1]}
    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    receipt = {
        "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "kind": "public-toggle-offset",
        "plan": "docs/public-toggle-fold.md (Part B)", "script": "tools/analyze_public_toggle_offset.py",
        "source_commit": commit, "working_tree_dirty": True, "model_calls_scored": calls, "audit_problems": problems,
        "audit_passed": not problems, "B0": {"max_gap_to_pod_path": b0_gap, "holds": b0}, "B1": b1,
        "landing_update6": s6, "label3_update3": s3, "update6_minus_straight_line_in_lambda": linear,
        "start_off_line_norms": meta["starts"], "paths": table,
        "inputs": [arb.binding(p) for p in sorted(set(bound))],
    }
    a.output.write_text(json.dumps(receipt, indent=1) + "\n")
    print("audit passed", not problems, problems[:5], "calls", calls)
    print("B0 gap %.2e holds %s" % (b0_gap, b0))
    print("B1", json.dumps(b1))
    for lam, row in zip(LAMBDAS, table):
        print("lambda %.2f  lane1 s: %s" % (lam, " ".join("%+.5f(%.2f)" % (x["s"], x["fraction"]) for x in row["lane1"])))
    for lam, row in zip(LAMBDAS, table):
        print("lambda %.2f  lane0 s: %s" % (lam, " ".join("%+.5f(%.2f)" % (x["s"], x["fraction"]) for x in row["lane0"])))
    print("update-6 minus straight line in lambda:", {l: ["%+.1e" % x for x in linear[l]] for l in range(2)})
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
