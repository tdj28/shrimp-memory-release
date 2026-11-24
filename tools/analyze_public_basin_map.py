#!/usr/bin/env python3
"""Labels, maps and the fixed checks of the basin map at H (docs/public-basin-map.md). Zero model calls.

Reads every shard's result.json of the sets present under --run-root (M, Z, S on the pod; X on the second host via
--x-root), requires every shard completed with all groups' diagnostics passed, and writes labels-<set>.npz (one label
per start: 0 = A_0, 1 = A_1, -1 = unresolved) and a receipt with B1-B4, the measured fractions, label changes along rows
and columns, and basin stability with Wilson 95% intervals.
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
import analyze_public_toggle_mechanism as mech  # noqa: E402

N = 65
LABEL_NAMES = {0: "A0", 1: "A1", -1: "unresolved"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(set_root: Path, starts_meta: dict, bound: list) -> tuple[np.ndarray, list, dict]:
    n_starts = 2 * starts_meta["groups"]
    labels = np.full(n_starts, -2, dtype=int)
    terminal_equal, problems = {}, []
    shards = sorted(f for f in set_root.glob("shard-*/result.json") if "." not in f.parent.name)  # not .partial copies
    for f in shards:
        r = json.loads(f.read_text()); bound.append(f)
        design = json.loads((f.parent / "design.json").read_text())
        if design["starts_sha256"] != starts_meta["sha256"]:
            problems.append(f"{f.parent.name}: ran a different start file"); continue
        if r["status"] != "completed":
            problems.append(f"{f.parent.name}: {r['status']}"); continue
        for g in r["groups"]:
            if not g.get("diagnostics_passed"):
                problems.append(f"{f.parent.name} group {g['group_id']}: diagnostics")
            terminal_equal[g["group_id"]] = g.get("terminal_lanes_equal")
        for row in r["rows"]:
            labels[row["start_index"]] = -1 if row["label"] is None else row["label"]
    missing = int((labels == -2).sum())
    return labels, problems, dict(shards=len(shards), missing_starts=missing, terminal_equal=terminal_equal)


def changes(grid: np.ndarray) -> dict:
    rows = [int((grid[i, 1:] != grid[i, :-1]).sum()) for i in range(N)]
    cols = [int((grid[1:, j] != grid[:-1, j]).sum()) for j in range(N)]
    return dict(per_row=rows, per_column=cols, rows_with_more_than_one=sum(x > 1 for x in rows),
                columns_with_more_than_one=sum(x > 1 for x in cols))


def fractions(x: np.ndarray) -> dict:
    return {LABEL_NAMES[k]: float((x == k).mean()) for k in (0, 1, -1)}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--starts-dir", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True, help="directory holding M/, Z/, S/ shard folders")
    ap.add_argument("--x-root", type=Path, help="second host's directory holding X/")
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="receipt JSON; labels-<set>.npz are written beside it")
    ap.add_argument("--allow-incomplete", action="store_true", help="report partial sets (labels -2 = not yet run)")
    ap.add_argument("--sweep-root", type=Path, help="toggle sweep returned-pod directory: report the pixels nearest "
                    "the 12-, 18- and 24-update pulse ends")
    ap.add_argument("--b3", type=Path, help="receipt of tools/check_public_basin_b3.py")
    a = ap.parse_args()
    bound: list[Path] = [a.reference]
    with np.load(a.reference) as z:
        ref = z["states"]
    with np.load(a.starts_dir / "plane.npz") as z:
        c, e1, e2 = z["origin"], z["e1"], z["e2"]
    bound.append(a.starts_dir / "plane.npz")
    out: dict = {"utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "kind": "public-basin-map",
                 "plan": "docs/public-basin-map.md", "script": "tools/analyze_public_basin_map.py", "model_calls": 0,
                 "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))}
    labels = {}
    for name, root in (("M", a.run_root), ("Z", a.run_root), ("S", a.run_root), ("X", a.x_root)):
        if root is None or not (root / name).is_dir():
            continue
        meta = json.loads((a.starts_dir / f"starts-{name}.json").read_text())
        bound.append(a.starts_dir / f"starts-{name}.json")
        lab, problems, info = collect(root / name, meta, bound)
        if (problems or info["missing_starts"]) and not a.allow_incomplete:
            raise SystemExit(f"set {name}: {problems[:5]} missing {info['missing_starts']}")
        labels[name] = lab
        np.savez(a.output.parent / f"labels-{name}.npz", labels=lab)
        out[name] = dict(starts_sha256=meta["sha256"], problems=problems, shards=info["shards"],
                         missing_starts=info["missing_starts"], fractions=fractions(lab[lab != -2]))
        if name in ("M", "Z"):
            grid = lab[: N * N].reshape(N, N)
            out[name].update(label_changes=changes(grid), centre=meta.get("center", [0.0, 0.0]))
            if name == "M":
                last = meta["groups"] - 1
                out["B3_same_group_lanes"] = dict(
                    group=last, lanes_equal=info["terminal_equal"].get(last), labels=lab[2 * last: 2 * last + 2].tolist(),
                    note="design error: the lanes carry the corner and the centre start, not one start twice; see B3")
                if a.b3 is not None:
                    b3 = json.loads(a.b3.read_text()); bound.append(a.b3)
                    centre = (N * N // 2)
                    out["B3_determinism"] = dict({k: v for k, v in b3.items() if k != "inputs"}, receipt=str(a.b3),
                                                 labels=[int(lab[2 * last + 1]), int(lab[centre])])
    if "M" in labels:
        axis = np.linspace(-5e-3, 5e-3, N)
        grid = labels["M"][: N * N].reshape(N, N)
        row0 = grid[N // 2]
        a1_zone = [int(row0[j]) for j in range(N) if -1.0e-3 - 1e-12 <= axis[j] <= 1.0e-3 + 1e-12]
        a0_zone = [int(row0[j]) for j in range(N) if 2.1e-3 - 1e-12 <= axis[j] <= 3.6e-3 + 1e-12]
        out["B1_bracket"] = dict(a1_zone=a1_zone, a0_zone=a0_zone,
                                 holds=all(x == 1 for x in a1_zone) and all(x == 0 for x in a0_zone))
        b2 = {}
        for cyc in (0, 1):
            d = ref[cyc][0] - c; s, t = float(d @ e1), float(d @ e2)
            i, j = int(np.argmin(abs(axis - t))), int(np.argmin(abs(axis - s)))
            b2[f"A{cyc}"] = dict(s=s, t=t, pixel=[i, j], label=int(grid[i, j]), holds=int(grid[i, j]) == cyc)
        out["B2_cycles"] = b2
        if a.sweep_root is not None:
            P = mech.load_group(mech.group_file(a.sweep_root, 26), bound)["push"]
            near = []
            for u in (12, 18, 24):
                for lane in (0, 1):
                    d = P[u, lane] - c; s, t = float(d @ e1), float(d @ e2)
                    i, j = int(np.argmin(abs(axis - t))), int(np.argmin(abs(axis - s)))
                    near.append(dict(pulse_updates=u, from_cycle=lane, s=s, t=t, off_plane=float(
                        np.linalg.norm(d - s * e1 - t * e2)), pixel=[i, j], label=int(grid[i, j])))
            out["B1_landings_post_hoc"] = dict(
                note="added after M's partial labels were seen (amendment of 23 November, 19:10 UTC)", pixels=near)
        if "X" in labels:
            xm = json.loads((a.starts_dir / "starts-X.json").read_text())
            with np.load(a.starts_dir / "starts-X.npz") as z:
                picks = z["m_groups"]
            mine = np.concatenate([labels["M"][2 * g: 2 * g + 2] for g in picks])
            agree = int((mine == labels["X"]).sum())
            out["B4_hosts"] = dict(lanes=int(mine.size), agree=agree, holds=agree >= 124,
                                   disagreements=[dict(m_group=int(picks[k // 2]), lane=int(k % 2), pod=int(mine[k]),
                                                       local=int(labels["X"][k]))
                                                  for k in np.nonzero(mine != labels["X"])[0]])
    if "S" in labels:
        with np.load(a.starts_dir / "starts-S.npz") as z:
            eps = z["eps"]
        lab = labels["S"].reshape(-1, 2)
        bs = []
        for e in sorted(set(eps.tolist())):
            for cyc in (0, 1):
                x = lab[eps == e, cyc]; x = x[x != -2]; n = int(x.size); k = int((x == cyc).sum())
                if n:
                    bs.append(dict(eps=e, cycle=f"A{cyc}", n=n, returned=k, fraction=k / n, wilson95=wilson(k, n),
                                   went_to_other=int((x == 1 - cyc).sum()), unresolved=int((x == -1).sum())))
        out["basin_stability"] = bs
    out["inputs"] = [{"path": str(p), "sha256": sha256(p)} for p in dict.fromkeys(bound)]
    a.output.write_text(json.dumps(out, indent=1) + "\n")
    for k in ("B1_bracket", "B1_landings_post_hoc", "B2_cycles", "B3_determinism", "B4_hosts"):
        if k in out:
            print(k, json.dumps(out[k])[:300])
    for name in labels:
        print(name, out[name]["fractions"], out[name].get("label_changes", {}).get("rows_with_more_than_one"))
    for row in out.get("basin_stability", []):
        print("  BS eps %.3f %s %d/%d = %.3f [%.3f, %.3f]" % (row["eps"], row["cycle"], row["returned"], row["n"],
              row["fraction"], *row["wilson95"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
