#!/usr/bin/env python3
"""Inputs for the second-register runs (docs/public-p4-register.md). Zero model calls.

  stage-r  From the earlier study's period-4 branch: check its settings equal the kite branch's, take the saved states
           nearest the hold H4 on each attracting sheet (L index 29, U index 61), write design-R.json and starts-R.npz.
  stage-w  From stage R's returned run: first establish that it is the planned Stage R (below), then test R1 (lag-4
           residual < 1e-12, lag-2 residual > 1e-3, cycles more than 5e-3 apart at every phase and at every cyclic
           shift), write references-p4.npz, design-W.json and starts-W.npz. If R1 fails, write the report and stop.

Stage R contract (Codex Task 9, D1), all checked before anything is computed or written: the prepared inputs
(inputs-R.json binds design-R.json and starts-R.npz by SHA-256; design-R.json is the literal one-group, one-stage
design); the run's own design record binds those same two files; its manifest covers exactly its files with matching
digests; the result is the completed single shard with the literal group (p4-relax-H4, stage 0, start 0, diagnostics
passed) and the literal stage (relax, 400 updates at alpha 0.06484 and log10 T 0.0651875, IEEE encodings, counts,
cache gates, no failed outputs); backend readiness is native64 with one thread; the call ledger is the literal
ordered 403 calls; the saved start equals starts-R.npz; and the path is read with the strict reader (initial and
every block checked against their bindings, finite float64, contiguous, named, at the planned controls). Any
violation writes preparation-failed.json, listing every problem, and nothing else.
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
import p4_readout  # noqa: E402
import task8_audit_common as C  # noqa: E402  (strict reader, manifest, ledger and record checks)

ALPHA = 0.06484
FOLD_LOW, FOLD_HIGH = 0.063981, 0.066394          # last attracting points of sheet U (low end) and sheet L (high end)
H4 = (FOLD_LOW + FOLD_HIGH) / 2                   # 0.0651875
DELTAS = (2.4e-4, 6.0e-4)
SHEET_L, SHEET_U = 29, 61
SAME = ("model", "layer", "dtype", "prompt", "opener", "window", "vectors", "concept", "alpha_scale")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def controls(log10_t):
    return {"alpha": ALPHA, "log10_temperature": float(round(log10_t, 7))}


def stage_r(a) -> int:
    b = a.branch_dir
    branch = json.loads((b / "branch.json").read_text())
    kite = json.loads((a.kite_branch_dir / "branch.json").read_text())
    if branch["period"] != 4 or branch["free"] != "temperature" or branch["args"]["alpha"] != ALPHA:
        raise SystemExit("not the period-4 temperature branch at alpha 0.06484")
    if any(branch["args"].get(k) != kite["args"].get(k) for k in SAME):
        raise SystemExit("branch settings differ from the kite branch")
    pts = branch["points"]
    states = np.load(b / "states.npy")[:, 0, :]
    for i in (SHEET_L, SHEET_U):
        if not (pts[i]["spectral_radius"] < 1 and pts[i]["residual"] < 1e-10):
            raise SystemExit(f"branch point {i} is not an attracting converged orbit")
    lam = [p["lambda"] for p in pts]
    near_l = min(range(0, 35), key=lambda i: abs(lam[i] - H4))
    near_u = min(range(53, 92), key=lambda i: abs(lam[i] - H4))
    if (near_l, near_u) != (SHEET_L, SHEET_U):
        raise SystemExit("plan's branch points are not the nearest to H4")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez(a.output_dir / "starts-R.npz", states=np.stack([states[SHEET_L], states[SHEET_U]])[None])
    design = {"profile": "public-p4-register", "stage_set": "R", "plan": "docs/public-p4-register.md",
              "groups": [{"name": "p4-relax-H4", "start_index": 0,
                          "stages": [{"name": "relax", "controls": controls(H4), "steps": 400}]}]}
    (a.output_dir / "design-R.json").write_text(json.dumps(design, indent=1) + "\n")
    info = {"H4": controls(H4), "branch_points": {"L": {"index": SHEET_L, "log10_temperature": lam[SHEET_L],
                                                       "spectral_radius": pts[SHEET_L]["spectral_radius"]},
                                                 "U": {"index": SHEET_U, "log10_temperature": lam[SHEET_U],
                                                       "spectral_radius": pts[SHEET_U]["spectral_radius"]}},
            "design_sha256": sha(a.output_dir / "design-R.json"), "starts_sha256": sha(a.output_dir / "starts-R.npz"),
            "inputs": [arb.binding(p) for p in (b / "branch.json", b / "states.npy", a.kite_branch_dir / "branch.json")]}
    (a.output_dir / "inputs-R.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: info[k] for k in ("H4", "branch_points", "design_sha256", "starts_sha256")}, indent=1))
    return 0


def stage_r_design():
    return {"profile": "public-p4-register", "stage_set": "R", "plan": "docs/public-p4-register.md",
            "groups": [{"name": "p4-relax-H4", "start_index": 0,
                        "stages": [{"name": "relax", "controls": controls(H4), "steps": 400}]}]}


def stage_r_evidence(d: Path, inputs: Path, problems: list):
    """Check the Stage R contract; returns (path (401, 2, 896), token ids, bound files) or None after recording why."""
    bound = []
    design_r, starts_r, info_r = inputs / "design-R.json", inputs / "starts-R.npz", inputs / "inputs-R.json"
    info = C.load_json(info_r, problems, "inputs-R")
    for f in (info_r, design_r, starts_r):
        if f.is_file():
            bound.append(arb.binding(f))
    if not isinstance(info, dict) or not design_r.is_file() or not starts_r.is_file() \
            or info.get("design_sha256") != sha(design_r) or info.get("starts_sha256") != sha(starts_r):
        problems.append("inputs-R: design-R.json or starts-R.npz is missing or not the prepared file")
        return None
    if not C.same(C.load_json(design_r, problems, "design-R"), stage_r_design()):
        problems.append("design-R.json is not the literal Stage R design")
    C.check_manifest(d, problems, "stage R run")
    for f in ("manifest.json", "result.json", "design.json", "backend-readiness.json", "group-000-start.npy"):
        if (d / f).is_file():
            bound.append(arb.binding(d / f))
    run_design = C.load_json(d / "design.json", problems, "stage R run")
    if not isinstance(run_design, dict) or run_design.get("stage_set") != "R" \
            or (run_design.get("design") or {}).get("sha256") != sha(design_r) \
            or (run_design.get("starts") or {}).get("sha256") != sha(starts_r):
        problems.append("stage R run: its design record does not bind the prepared design-R.json and starts-R.npz")
    r = C.load_json(d / "result.json", problems, "stage R run")
    if not isinstance(r, dict):
        return None
    groups, stages = r.get("groups"), r.get("stages")
    g = groups[0] if isinstance(groups, list) and len(groups) == 1 and isinstance(groups[0], dict) else {}
    if r.get("profile") != "public-p4-register" or r.get("status") != "completed" or not C.same(r.get("shard"), 0) \
            or not C.same(r.get("shards"), 1) or g.get("name") != "p4-relax-H4" or not C.same(g.get("group_id"), 0) \
            or not C.same(g.get("stage_ids"), [0]) or not C.same(g.get("start_index"), 0) \
            or g.get("status") != "completed" or g.get("diagnostics_passed") is not True:
        problems.append("stage R run: not the completed single shard with the literal group p4-relax-H4")
    if not isinstance(stages, list) or len(stages) != 1 or not isinstance(stages[0], dict):
        problems.append("stage R run: not exactly one stage record")
        return None
    st, want = stages[0], controls(H4)
    if not C.same(st.get("stage_id"), 0) or st.get("name") != "relax" or not C.same(st.get("group_id"), 0):
        problems.append("stage R run: the stage is not stage 0, relax, of group 0")
    C.check_stage_record(st, 400, problems, "stage R relax")
    C.check_controls(st, want, problems, "stage R relax")
    C.check_environment(d, None, None, problems, "stage R run")
    n_calls = C.check_call_ledger(r.get("calls"), [dict(stage_id=0, stage="relax", group_id=0, group="p4-relax-H4",
                                                        steps=400)], "public-p4-register", problems, "stage R run",
                                    layout="pair")
    if not C.same(r.get("model_calls"), n_calls):
        problems.append(f"stage R run: model_calls {r.get('model_calls')} differs from the ledger ({n_calls})")
    path = C.reconstruct(d, st, want, 400, problems, "stage R relax", bound)
    try:
        with np.load(starts_r, allow_pickle=False) as z:
            start = z["states"][0]
        if np.load(d / "group-000-start.npy", allow_pickle=False).tobytes() != start.tobytes() \
                or (path is not None and path[0].tobytes() != start.tobytes()):
            problems.append("stage R run: the saved start is not starts-R.npz")
    except Exception as exc:
        problems.append(f"stage R run: start files unreadable ({type(exc).__name__})")
    if problems or path is None:
        return None
    tokens = []
    for b in st["blocks"]:
        with np.load(d / b["path"], allow_pickle=False) as z:
            tokens.append(z["token_ids"])
    return path, np.concatenate(tokens, 0), bound


def stage_w(a) -> int:
    d = a.r_shard
    problems = []
    try:
        evidence = stage_r_evidence(d, a.inputs_r, problems)
    except Exception as exc:                            # malformed evidence is a recorded rejection, not a crash
        problems.append(f"stage R evidence could not be checked ({type(exc).__name__}: {exc})")
        evidence = None
    if evidence is None:
        a.output_dir.mkdir(parents=True, exist_ok=False)
        report = {"status": "rejected", "stage_r_run": str(d), "inputs_r": str(a.inputs_r), "problems": problems}
        (a.output_dir / "preparation-failed.json").write_text(json.dumps(report, indent=1) + "\n")
        for q in problems[:12]:
            print("  problem:", q)
        raise SystemExit("stage W is not prepared: the stage R evidence does not meet its contract")
    path, tokens, bound = evidence
    alpha, log10_t = ALPHA, controls(H4)["log10_temperature"]
    lag4 = [p4_readout.lag_residual(path[:, lane], 4) for lane in range(2)]
    lag2 = [p4_readout.lag_residual(path[:, lane], 2) for lane in range(2)]
    refs = p4_readout.references_from_relaxation(path)
    phase_matched = [float(np.linalg.norm(refs[0][j] - refs[1][j])) for j in range(4)]
    shifted = [p4_readout.align(refs[0], refs[1])[1]]
    r1 = {"lag4_residual": lag4, "lag2_residual": lag2, "phase_matched_distances": phase_matched,
          "best_cyclic_shift_distance": shifted[0],
          "holds": max(lag4) < 1e-12 and min(lag2) > 1e-3 and min(phase_matched) > 5e-3 and shifted[0] > 5e-3,
          "argmax_tokens_same_over_last_period": bool(np.array_equal(tokens[-4:, 0], tokens[-4:, 1]))
          if tokens.ndim == 2 else None,
          "controls": {"alpha": alpha, "log10_temperature": log10_t}}
    a.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"R1": r1, "stage_r_contract": "passed", "inputs": bound}
    if not r1["holds"]:
        (a.output_dir / "references-report.json").write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(r1, indent=1))
        raise SystemExit("R1 failed: stage W is not prepared")
    np.savez(a.output_dir / "references-p4.npz", states=refs)
    np.savez(a.output_dir / "starts-W.npz", states=np.stack([refs[0][0], refs[1][0]])[None])
    hold = {"name": "hold", "controls": controls(H4), "steps": 240}
    groups = [{"name": "p4-hold-only", "start_index": 0, "stages": [hold]}]
    for bit, edge, sign in ((1, FOLD_HIGH, +1), (0, FOLD_LOW, -1)):
        for delta in DELTAS:
            groups.append({"name": f"p4-write{bit}-delta{delta:.1e}", "start_index": 0,
                           "stages": [{"name": "pulse", "controls": controls(edge + sign * delta), "steps": 120},
                                      dict(hold)]})
    design = {"profile": "public-p4-register", "stage_set": "W", "plan": "docs/public-p4-register.md",
              "groups": groups}
    (a.output_dir / "design-W.json").write_text(json.dumps(design, indent=1) + "\n")
    report.update(references_sha256=sha(a.output_dir / "references-p4.npz"),
                  design_sha256=sha(a.output_dir / "design-W.json"), starts_sha256=sha(a.output_dir / "starts-W.npz"))
    (a.output_dir / "references-report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"R1": r1, **{k: report[k] for k in ("references_sha256", "design_sha256", "starts_sha256")}},
                     indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("stage-r")
    r.add_argument("--branch-dir", type=Path, required=True)
    r.add_argument("--kite-branch-dir", type=Path, required=True)
    r.add_argument("--output-dir", type=Path, required=True)
    w = sub.add_parser("stage-w")
    w.add_argument("--r-shard", type=Path, required=True)
    w.add_argument("--inputs-r", type=Path, required=True, help="directory with inputs-R.json, design-R.json, starts-R.npz")
    w.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    return stage_r(a) if a.cmd == "stage-r" else stage_w(a)


if __name__ == "__main__":
    sys.exit(main())
