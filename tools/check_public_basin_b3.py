#!/usr/bin/env python3
"""B3 of the basin map at H (docs/public-basin-map.md, amendment of 23 November, 23:20 UTC). Zero model calls.

M's odd last start (the grid corner) is paired with a copy of the grid's centre start, so the two lanes of M's last
group carry different starts and their equality says nothing. The centre start itself is lane 0 of group
(N*N // 2) // 2. B3 compares that lane with lane 1 of the last group, update by update, bitwise, after checking every
stage file read against its shard's manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

N = 65


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(shard: Path, group: int, lane: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    manifest = {f["path"]: f["sha256"] for f in json.loads((shard / "manifest.json").read_text())["files"]}
    files = [shard / f"stage-{group}-initial.npy"] + [shard / f"stage-{group}-block-{b:03d}.npz" for b in range(12, 241, 12)]
    bound = []
    for f in files:
        h = sha256(f)
        if manifest.get(f.name) != h:
            raise SystemExit(f"{f}: not the file its manifest records")
        bound.append(dict(path=str(f), sha256=h))
    x0 = np.load(files[0])[lane]
    states = np.concatenate([np.load(f)["states"][:, lane] for f in files[1:]], 0)
    return x0, states, bound


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m-root", type=Path, required=True, help="directory holding M's shard folders")
    ap.add_argument("--shards", type=int, default=265)
    ap.add_argument("--groups", type=int, default=2113)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    last, centre = a.groups - 1, (N * N // 2) // 2
    sa, sb = a.m_root / f"shard-{last % a.shards:03d}", a.m_root / f"shard-{centre % a.shards:03d}"
    x0a, xa, ba = stage(sa, last, 1)
    x0b, xb, bb = stage(sb, centre, 0)
    diff = np.nonzero(np.any(xa != xb, axis=1))[0]
    out = dict(check="B3", plan="docs/public-basin-map.md", model_calls=0,
               compared=[dict(group=last, lane=1, shard=sa.name), dict(group=centre, lane=0, shard=sb.name)],
               same_start=bool(np.array_equal(x0a, x0b)), updates=int(xa.shape[0]),
               bitwise_equal=bool(diff.size == 0), first_differing_update=int(diff[0]) + 1 if diff.size else None,
               max_abs_difference=float(np.abs(xa - xb).max()),
               terminal_sha256=[hashlib.sha256(xa[-1].tobytes()).hexdigest(), hashlib.sha256(xb[-1].tobytes()).hexdigest()],
               holds=bool(np.array_equal(x0a, x0b) and diff.size == 0), inputs=ba + bb)
    a.output.write_text(json.dumps(out, indent=1) + "\n")
    print({k: v for k, v in out.items() if k != "inputs"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
