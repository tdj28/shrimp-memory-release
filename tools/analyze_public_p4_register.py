#!/usr/bin/env python3
"""Audit and score stage W of the second register (docs/public-p4-register.md). Zero model calls.

Per shard: rehash the manifest; check statuses, diagnostics, the start against the bound start file, each stage's
controls and length against the bound design, and the pulse-to-hold carry. Then read each lane's period-4 cycle from
its last four hold states (tools/p4_readout.py) and score W0 (hold only: identity), W1 (write 1: both lanes on U) and
W2 (write 0: both lanes on L). Also reports settle times, end-of-pulse distances to both cycles, the closest
approach to the earlier study's saved unstable-sheet state, and stage R's R1 report.
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
import p4_readout  # noqa: E402

UNSTABLE_INDEX = 48                      # earlier study's middle-sheet point nearest H4 (log10 T 0.065161)
CLASS = {(0, 1): "hold", (0, 0): "reset0", (1, 1): "reset1", (1, 0): "toggle"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run directory with inputs-W/ and W/shard-*")
    ap.add_argument("--branch-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    I = a.run / "inputs-W"
    rep = json.loads((I / "references-report.json").read_text())
    problems: list[str] = []
    for name, key in (("references-p4.npz", "references_sha256"), ("design-W.json", "design_sha256"),
                      ("starts-W.npz", "starts_sha256")):
        if sha(I / name) != rep[key]:
            problems.append(f"{name}: hash differs from the references report")
    with np.load(I / "references-p4.npz") as z:
        refs = z["states"]
    with np.load(I / "starts-W.npz") as z:
        starts = z["states"]
    design = json.loads((I / "design-W.json").read_text())
    M = np.load(a.branch_dir / "states.npy")[UNSTABLE_INDEX, 0, :]
    bound = [I / "references-p4.npz", I / "starts-W.npz", I / "design-W.json", I / "references-report.json",
             a.branch_dir / "states.npy"]

    groups, calls = [], 0
    for d in sorted((a.run / "W").glob("shard-*")):
        if not d.is_dir():
            continue
        manifest = json.loads((d / "manifest.json").read_text())
        files = {x["path"]: x["sha256"] for x in manifest["files"]}
        if set(files) != {p.name for p in d.iterdir() if p.is_file()} - {"manifest.json"}:
            problems.append(f"{d.name}: manifest file set differs")
        for name, h in files.items():
            if sha(d / name) != h:
                problems.append(f"{d.name}: hash mismatch {name}")
        r = json.loads((d / "result.json").read_text())
        calls += r["model_calls"]
        for g in r["groups"]:
            spec = design["groups"][g["group_id"]]
            if r["status"] != "completed" or g["status"] != "completed" or not g["diagnostics_passed"]:
                problems.append(f"{d.name} {g['name']}: status {r['status']} / {g['status']}")
                continue
            start = np.load(d / f"group-{g['group_id']:03d}-start.npy")
            if start.tobytes() != starts[spec["start_index"]].tobytes():
                problems.append(f"{d.name} {g['name']}: start differs from the start file")
            paths, prev = [], start
            for sid, st in zip(g["stage_ids"], spec["stages"]):
                blocks, alpha, log10_t, bfiles = arb.load_stage(d, sid, st["steps"])
                init = np.load(d / f"stage-{sid:02d}-initial.npy")
                bound += [d / "manifest.json", d / "result.json", d / f"stage-{sid:02d}-initial.npy", *bfiles]
                if (alpha, log10_t) != (st["controls"]["alpha"], st["controls"]["log10_temperature"]):
                    problems.append(f"{d.name} {g['name']}: stage {sid} controls differ from the design")
                if blocks.shape[0] != st["steps"]:
                    problems.append(f"{d.name} {g['name']}: stage {sid} has {blocks.shape[0]} updates")
                if init.tobytes() != prev.tobytes():
                    problems.append(f"{d.name} {g['name']}: stage {sid} does not start where the last ended")
                X = np.concatenate([init[None], blocks], 0)
                paths.append(X)
                prev = X[-1]
            hold = paths[-1]
            lanes = []
            for lane in range(2):
                out = p4_readout.read(hold[:, lane], refs)
                pulse_end = paths[0][-1, lane] if len(paths) == 2 else None
                lanes.append({**out,
                              "pulse_end_distance_to_cycles": None if pulse_end is None else
                              [float(min(np.linalg.norm(pulse_end - refs[c][j]) for j in range(4))) for c in range(2)],
                              "closest_to_saved_unstable_state": float(min(np.linalg.norm(hold[m, lane] - M)
                                                                           for m in range(hold.shape[0])))})
            bits = tuple(x["cycle"] for x in lanes)
            groups.append({"name": g["name"], "stages": [dict(s) for s in spec["stages"]], "lanes": lanes,
                           "outcome": CLASS.get(bits, "unresolved")})

    by = {g["name"]: g for g in groups}
    w0 = by.get("p4-hold-only", {}).get("outcome") == "hold"
    w1 = [g["outcome"] == "reset1" for n, g in by.items() if n.startswith("p4-write1")]
    w2 = [g["outcome"] == "reset0" for n, g in by.items() if n.startswith("p4-write0")]
    scores = {"W0_hold_only_identity": w0, "W1_write1_reset_to_U": all(w1) and len(w1) == 2,
              "W2_write0_reset_to_L": all(w2) and len(w2) == 2}
    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    receipt = {
        "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "kind": "public-p4-register",
        "plan": "docs/public-p4-register.md", "script": "tools/analyze_public_p4_register.py", "source_commit": commit,
        "working_tree_dirty": True, "model_calls_scored_stage_W": calls, "audit_problems": problems,
        "audit_passed": not problems, "R1": rep["R1"], "scores": scores, "groups": groups,
        "inputs": [arb.binding(p) for p in sorted(set(bound))],
    }
    a.output.write_text(json.dumps(receipt, indent=1) + "\n")
    print("audit passed", not problems, problems[:6], "stage W calls", calls)
    print("scores", scores)
    for g in groups:
        print(" ", g["name"], g["outcome"], [(x["cycle"], x["settled_from_update"], "%.1e" % x["distance"]) for x in g["lanes"]],
              "pulse-end distances", [x["pulse_end_distance_to_cycles"] for x in g["lanes"]])
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
