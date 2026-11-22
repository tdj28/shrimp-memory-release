"""Print the full-adder truth table from the last twelve measured states of each hold."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "inputs" / "adder-example"


def verify(path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Hash mismatch for {path.name}")


def decode(states, refs):
    if states.shape != (12, 896) or states.dtype != np.float64 or not np.isfinite(states).all():
        raise ValueError("Invalid trajectory tail")
    cycle = states[-6:]
    def max_distance(a, b):
        return np.linalg.norm(a - b, axis=1).max()
    if max_distance(cycle, states[:6]) > 1e-8:
        raise ValueError("Period-six closure failed")
    if any(max_distance(cycle, np.roll(cycle, i, axis=0)) <= 1e-4 for i in (1, 2, 3)):
        raise ValueError("Proper divisor not excluded")
    matches = [k for k, ref in enumerate(refs)
               if min(max_distance(cycle, np.roll(ref, i, axis=0)) for i in range(6)) <= 1e-6]
    if len(matches) != 1:
        raise ValueError("Ambiguous or missing reference match")
    return matches[0]


def main():
    ref = ROOT / "inputs" / "reference-cycles.npz"
    verify(ref, "d96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0")
    with np.load(ref, allow_pickle=False) as archive:
        refs = archive["states"]
    observed = {}
    for row in json.loads((DATA / "index.json").read_text()):
        x, y, stage = (row[k] for k in ("x", "y", "stage"))
        key = (x, y, stage)
        if key in observed:
            raise ValueError("Repeated stage")
        path = DATA / f"x{x}-y{y}-stage-{stage:02d}-block-240.npz"
        verify(path, row["sha256"])
        with np.load(path, allow_pickle=False) as block:
            if (int(block["first_state_index"]), int(block["last_state_index"])) != (229, 240):
                raise ValueError("Incorrect hold interval")
            states = block["states"]
            if states.shape != (12, 2, 896):
                raise ValueError("Incorrect lane layout")
            observed[key] = [decode(states[:, lane], refs) for lane in range(2)]
    if set(observed) != {(x,y,s) for x in (0,1) for y in (0,1) for s in (1,3,5)}:
        raise ValueError("Incomplete stage coverage")
    print(f"{'carry_in':>8} {'x':>2} {'y':>2} {'after_first':>11} {'sum':>4} {'carry_out':>9}")
    for c in (0,1):
        for x in (0,1):
            for y in (0,1):
                readings = [observed[x,y,s][c] for s in (1,3,5)]
                print(f"{c:>8} {x:>2} {y:>2} {readings[0]:>11} {readings[1]:>4} {readings[2]:>9}")
                if readings != [c ^ x, (c+x+y)%2, (c+x+y)//2]:
                    raise ValueError("Measured readouts differ from the truth table")
    print("All 24 readouts match the full-adder truth table (after_first = carry_in XOR x).")


if __name__ == "__main__":
    main()
