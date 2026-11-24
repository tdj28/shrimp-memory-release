#!/usr/bin/env python3
"""Starts for the basin map at H (docs/public-basin-map.md), fixed before any model call. Zero model calls.

Builds the plane through C_p at label 0 spanned by v_p at label 0 and the part of A_0[0] - A_1[0] orthogonal to it,
then writes one start file per set: M (65 x 65 main slice), S (basin-stability kicks), X (second-host sample of M),
and, given M's labels, Z (65 x 65 zoom). Each file holds starts of shape (groups, 2, 896), float64, and the
coordinates of every start; a JSON sidecar binds the inputs and the file's hash.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_public_toggle_mechanism as mech  # noqa: E402
from analyze_public_toggle_fold import cycle_and_directions  # noqa: E402

N_GRID = 65
HALF_WIDTH = 5e-3
ZOOM_FACTOR = 16
KICKS = (0.005, 0.01, 0.02)
KICKS_PER_CYCLE = 500
SEED_S, SEED_X = 20261123, 20261124
X_GROUPS = 64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plane(sweep_root: Path, reference: Path, bound: list[Path]):
    P = mech.load_group(mech.group_file(sweep_root, 26), bound)["push"]
    cp, v = cycle_and_directions(P)
    with np.load(reference) as z:
        ref = z["states"]
    bound.append(reference)
    c, e1 = cp[0], v[0]
    w = ref[0][0] - ref[1][0]
    e2 = w - (w @ e1) * e1
    e2 /= np.linalg.norm(e2)
    return c, e1, e2, ref


def grid(center_s: float, center_t: float, half: float):
    axis = np.linspace(-half, half, N_GRID)
    s, t = np.meshgrid(center_s + axis, center_t + axis)      # row-major: t (rows) then s (columns)
    return s.ravel(), t.ravel()


def pair(points: np.ndarray, pad: np.ndarray) -> np.ndarray:
    if len(points) % 2:
        points = np.concatenate([points, pad[None]], 0)
    return points.reshape(-1, 2, points.shape[-1])


def write(out: Path, name: str, starts: np.ndarray, meta: dict, **coords) -> dict:
    path = out / f"starts-{name}.npz"
    np.savez(path, starts=starts, **coords)
    meta = dict(meta, set=name, groups=int(starts.shape[0]), file=path.name, sha256=sha256(path),
                calls=int(starts.shape[0]) * 243)
    (out / f"starts-{name}.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-root", type=Path, help="toggle sweep returned-pod directory (M, S and X)")
    ap.add_argument("--reference", type=Path, help="reference-cycles.npz (M, S and X)")
    ap.add_argument("--plane", type=Path, help="plane.npz written with M (Z only)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--zoom-from", type=Path, help="labels-M.npz (column 'labels', 4225 entries: 0, 1 or -1)")
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    bound: list[Path] = []
    if a.zoom_from is None:
        c, e1, e2, ref = plane(a.sweep_root, a.reference, bound)
    else:
        with np.load(a.plane) as z:
            c, e1, e2 = z["origin"], z["e1"], z["e2"]
        bound.append(a.plane)
    common = dict(utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), plan="docs/public-basin-map.md",
                  script="tools/prepare_public_basin_map.py", model_calls=0,
                  inputs=[{"path": str(p), "sha256": sha256(p)} for p in dict.fromkeys(bound)],
                  plane=dict(origin="C_p[0]", e1="v_p[0]", e2="(A0[0]-A1[0]) orthogonal to e1",
                             e1_dot_e2=float(e1 @ e2)))
    if a.zoom_from is None:
        np.savez(a.output_dir / "plane.npz", origin=c, e1=e1, e2=e2)

    def point(s, t):
        return c[None] + s[:, None] * e1[None] + t[:, None] * e2[None]

    if a.zoom_from is None:
        s, t = grid(0.0, 0.0, HALF_WIDTH)
        m = write(a.output_dir, "M", pair(point(s, t), c), common, s=s, t=t)
        rng = np.random.default_rng(SEED_S)
        kicks, kick_eps, kick_cycle = [], [], []
        for eps in KICKS:
            for _ in range(KICKS_PER_CYCLE):
                lanes = []
                for cyc in (0, 1):
                    r = rng.standard_normal(ref.shape[-1])
                    x = ref[cyc][0]
                    lanes.append(x + eps * np.linalg.norm(x) * r / np.linalg.norm(r))
                kicks.append(np.stack(lanes)); kick_eps.append(eps); kick_cycle.append((0, 1))
        write(a.output_dir, "S", np.stack(kicks), dict(common, seed=SEED_S, kicks=list(KICKS),
              per_cycle=KICKS_PER_CYCLE), eps=np.array(kick_eps), cycle=np.array(kick_cycle))
        pick = np.sort(np.random.default_rng(SEED_X).choice(m["groups"], X_GROUPS, replace=False))
        with np.load(a.output_dir / "starts-M.npz") as z:
            write(a.output_dir, "X", z["starts"][pick], dict(common, seed=SEED_X, from_set="M"), m_groups=pick)
        for name, x in (("A0[0]", ref[0][0]), ("A1[0]", ref[1][0])):
            d = x - c
            print(f"{name}: s={d @ e1:+.6f} t={d @ e2:+.6f}")
    else:
        with np.load(a.zoom_from) as z:
            labels = z["labels"][: N_GRID * N_GRID].reshape(N_GRID, N_GRID)
        axis = np.linspace(-HALF_WIDTH, HALF_WIDTH, N_GRID)
        row = labels[N_GRID // 2]
        hits = [j for j in range(N_GRID // 2, N_GRID - 1) if row[j] == 1 and row[j + 1] == 0]
        if hits:
            j = hits[0]; cs, ct, rule = 0.5 * (axis[j] + axis[j + 1]), 0.0, "t=0 row, first A1->A0 change at s>0"
        else:
            cand = []
            for i in range(N_GRID):
                for j in range(N_GRID - 1):
                    if labels[i, j] != labels[i, j + 1]:
                        ms = 0.5 * (axis[j] + axis[j + 1])
                        cand.append((np.hypot(ms - 1.5e-3, axis[i]), abs(axis[i]), ms, axis[i]))
            if not cand:
                print("M has no label change: Z is not run"); return 0
            _, _, cs, ct = min(cand); rule = "nearest label change to (+1.5e-3, 0)"
        s, t = grid(cs, ct, HALF_WIDTH / ZOOM_FACTOR)
        write(a.output_dir, "Z", pair(point(s, t), point(np.array([cs]), np.array([ct]))[0]),
              dict(common, center=[cs, ct], rule=rule, labels_from=str(a.zoom_from), labels_sha256=sha256(a.zoom_from)),
              s=s, t=t)
    for f in sorted(a.output_dir.glob("starts-*.json")):
        meta = json.loads(f.read_text()); print(f.name, meta["groups"], "groups", meta["calls"], "calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
