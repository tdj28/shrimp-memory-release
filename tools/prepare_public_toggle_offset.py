#!/usr/bin/env python3
"""Start states for Part B of docs/public-toggle-fold.md. Zero model calls.

For each lane (the phase-0 points of reference cycles 0 and 1) and lambda in (0, 0.25, 0.5, 0.75, 1): the point
proj + lambda (start - proj), where proj is C_p's label-0 point plus the lane's coefficient along v_p (label 0) times
v_p. C_p and v_p come from the pod's 120-update pulse exactly as in tools/analyze_public_toggle_mechanism.py. The
lambda = 1 start is the reference state itself, and must equal the saved phase-0 start of the toggle check bit for bit.
Writes starts.npz (lambdas, states with shape (5, 2, 896)) and starts.json into --output-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_public_register_branch as arb  # noqa: E402  (binding)
import analyze_public_toggle_fold as fold  # noqa: E402  (cycle_and_directions)
import analyze_public_toggle_mechanism as mech  # noqa: E402  (group_file, load_group)

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-root", type=Path, required=True, help="toggle check batch-1 pod-returned directory")
    ap.add_argument("--sweep-root", type=Path, required=True, help="toggle sweep returned-pod directory")
    ap.add_argument("--references", type=Path, required=True, help="the instrument's qualified reference cycles")
    ap.add_argument("--xor-references", type=Path, required=True, help="reference cycles of the audited XOR chain")
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    bound: list[Path] = [a.references, a.xor_references]

    P = mech.load_group(mech.group_file(a.sweep_root, 26), bound)["push"]
    cp, v = fold.cycle_and_directions(P)
    g0 = mech.load_group(mech.group_file(a.check_root, 0), bound)
    assert g0["meta"]["start_phase"] == 0 and g0["push_controls"] == mech.PULSE
    with np.load(a.references) as z:
        refs = z["states"]
    with np.load(a.xor_references) as z:
        if not np.array_equal(z["states"], refs):
            raise SystemExit("instrument references differ from the XOR chain's")
    start = refs[:, 0].copy()                                         # (2, 896): lanes from cycle 0 and cycle 1
    if start.tobytes() != g0["push"][0].tobytes() or start.tobytes() != P[0].tobytes():
        raise SystemExit("reference phase-0 states differ from the saved phase-0 starts")

    states, lanes = [], []
    for lane in range(2):
        e = start[lane] - cp[0]
        c = float(v[0] @ e)
        proj = cp[0] + c * v[0]
        off = start[lane] - proj
        lanes.append({"lane": lane, "coefficient_along_v_p": c, "distance_to_cp": float(np.linalg.norm(e)),
                      "off_line_norm": float(np.linalg.norm(off)),
                      "fraction_along_v_p": abs(c) / float(np.linalg.norm(e))})
    for lam in LAMBDAS:
        x = np.empty_like(start)
        for lane in range(2):
            if lam == 1.0:
                x[lane] = start[lane]
            else:
                c = float(v[0] @ (start[lane] - cp[0]))
                proj = cp[0] + c * v[0]
                x[lane] = proj + lam * (start[lane] - proj)
        states.append(x)
    states = np.stack(states)
    check = []
    for i, lam in enumerate(LAMBDAS):
        row = {"lambda": lam}
        for lane in range(2):
            e = states[i, lane] - cp[0]
            c = float(v[0] @ e)
            row[f"lane{lane}"] = {"s": c, "off_line_norm": float(np.linalg.norm(e - c * v[0]))}
        check.append(row)

    a.output_dir.mkdir(parents=True, exist_ok=False)
    out = a.output_dir / "starts.npz"
    np.savez(out, lambdas=np.array(LAMBDAS), states=states)
    meta = {"plan": "docs/public-toggle-fold.md (Part B)", "script": "tools/prepare_public_toggle_offset.py",
            "model_calls": 0, "lambdas": list(LAMBDAS), "lanes": lanes, "starts": check,
            "cp_label0_sha256": hashlib.sha256(cp[0].tobytes()).hexdigest(),
            "v_p_label0_sha256": hashlib.sha256(v[0].tobytes()).hexdigest(),
            "starts_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "inputs": [arb.binding(p) for p in sorted(set(bound))]}
    (a.output_dir / "starts.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in ("lanes", "starts", "starts_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
