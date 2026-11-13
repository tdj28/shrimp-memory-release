#!/usr/bin/env python3
"""Are the nine toggles thin borders between a reset to 0 and a reset to 1? Tests fixed in
docs/public-toggle-border-rule.md before any adjacency was computed.

Zero model calls. Reads the audited sweep outcomes, the audited toggle checks and, for P3, the nine toggles' saved
pulse trajectories.

  P1  each toggle's eight check neighbours: B1 (a reset to 0 and a reset to 1 among them), B2 (no hold), B3 (no
      opposite pair with the same reset at both ends);
  P2  the coarse sweep grid: each toggle's sweep neighbours and the counts of outcome pairs between grid neighbours;
  P3  the lanes' separation at the end of each toggle pulse, relative to the start, and each lane's change over its last
      six pulse updates; ordinal claim: toggles failing P1 end with larger separations than every toggle that passes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_public_register_branch as arb  # noqa: E402  (binding)
import analyze_public_toggle_mechanism as mech  # noqa: E402  (group_file, load_group)

PERIOD = 6
DIRECTIONS = tuple(range(0, 360, 30))
RADII = (0.5, 1.0, 2.0, 3.5)
DWELLS = (12, 48, 120)
SHORT = {"identity": "hold", "reset0": "reset0", "reset1": "reset1", "toggle": "toggle", "unresolved": "unresolved"}
REFERENCE_TOGGLE = "t030-r3.5-d012"          # helped suggest the rule: reported, not counted
NB = re.compile(r"-nb-t(-?\d+)-r([0-9.]+)$")


def outcome(m: str) -> str:
    if m not in SHORT:
        raise SystemExit(f"unknown mapping {m!r}")
    return SHORT[m]


def p1(rows: list[dict]) -> dict:
    out = {}
    for cand in sorted({r["candidate"] for r in rows}):
        th, r, d = re.match(r"t(\d+)-r([0-9.]+)-d(\d+)$", cand).groups()
        th, r, d = int(th), float(r), int(d)
        nbs = {}
        for x in rows:
            if x["candidate"] == cand and x["role"] == "neighbour":
                t, rr = NB.search(x["name"]).groups()
                dth = int(t) - th
                f = round(float(rr) / r, 3)
                nbs[(dth, f)] = outcome(x["mapping"])
        expected = {(a, f) for a in (-10, 0, 10) for f in (0.9, 1.0, 1.1)} - {(0, 1.0)}
        if set(nbs) != expected:
            raise SystemExit(f"{cand}: neighbour grid {sorted(nbs)} differs from the check design")
        resolved = {k: v for k, v in nbs.items() if v != "unresolved"}
        vals = list(resolved.values())
        b1 = "reset0" in vals and "reset1" in vals
        b2 = "hold" not in vals
        pairs = [((-10, 0.9), (10, 1.1)), ((-10, 1.0), (10, 1.0)), ((-10, 1.1), (10, 0.9)), ((0, 0.9), (0, 1.1))]
        pair_rows, b3 = [], True
        for a, b in pairs:
            ea, eb = nbs[a], nbs[b]
            same_reset = ea == eb and ea in ("reset0", "reset1")
            b3 = b3 and not same_reset
            pair_rows.append({"ends": [f"{a[0]:+d} deg x{a[1]}", f"{b[0]:+d} deg x{b[1]}"], "outcomes": [ea, eb],
                              "same_reset": same_reset})
        other = {x["role"] + (f"-{x['phase']}" if x["role"] == "phase" else f"-{x['name'].rsplit('-', 1)[-1]}"):
                 outcome(x["mapping"]) for x in rows if x["candidate"] == cand and x["role"] != "neighbour"}
        out[cand] = {
            "theta": th, "radius": r, "dwell": d, "counted": cand != REFERENCE_TOGGLE,
            "grid": {f"{a:+d} deg, radius x{f}": nbs[(a, f)] for a, f in sorted(nbs)},
            "counts": dict(Counter(nbs.values())), "unresolved": sorted(f"{a:+d} x{f}" for (a, f), v in nbs.items()
                                                                        if v == "unresolved"),
            "B1": b1, "B2": b2, "B3": b3, "opposite_pairs": pair_rows, "passes": b1 and b2 and b3,
            "phases_dwells_rounding_recorded_only": other,
        }
    return out


def p2(groups: list[dict], toggles: list[str]) -> dict:
    grid = {}
    for g in groups:
        if g.get("theta") is None:
            continue
        grid[(int(g["theta"]), float(g["radius"]), int(g["dwell"]))] = outcome(g["mapping"])
    if len(grid) != len(DIRECTIONS) * len(RADII) * len(DWELLS):
        raise SystemExit(f"sweep grid has {len(grid)} pulses")
    per_toggle = {}
    for cand in toggles:
        th, r, d = re.match(r"t(\d+)-r([0-9.]+)-d(\d+)$", cand).groups()
        th, r, d = int(th), float(r), int(d)
        if grid[(th, r, d)] != "toggle":
            raise SystemExit(f"{cand} is not a toggle in the sweep")
        i = RADII.index(r)
        per_toggle[cand] = {
            "angle_minus_30": grid[((th - 30) % 360, r, d)], "angle_plus_30": grid[((th + 30) % 360, r, d)],
            "radius_in": grid[(th, RADII[i - 1], d)] if i > 0 else None,
            "radius_out": grid[(th, RADII[i + 1], d)] if i + 1 < len(RADII) else None,
            "other_lengths": {str(dd): grid[(th, r, dd)] for dd in DWELLS if dd != d},
        }
    def tally(pairs):
        c = Counter(tuple(sorted(p)) for p in pairs)
        unlike = sum(v for k, v in c.items() if k[0] != k[1])
        double = sum(v for k, v in c.items() if set(k) in ({"hold", "toggle"}, {"reset0", "reset1"}))
        return {"pairs": {" - ".join(k): v for k, v in sorted(c.items())}, "total": sum(c.values()),
                "unlike": unlike, "double_changes": double,
                "double_fraction_of_unlike": double / unlike if unlike else None}
    angle_pairs = [(grid[(t, r, d)], grid[((t + 30) % 360, r, d)]) for t in DIRECTIONS for r in RADII for d in DWELLS]
    radius_pairs = [(grid[(t, RADII[i], d)], grid[(t, RADII[i + 1], d)]) for t in DIRECTIONS
                    for i in range(len(RADII) - 1) for d in DWELLS]
    freq = Counter(grid.values())
    n = sum(freq.values())
    p = {k: v / n for k, v in freq.items()}
    same = sum(x * x for x in p.values())
    null = (2 * p.get("hold", 0) * p.get("toggle", 0) + 2 * p.get("reset0", 0) * p.get("reset1", 0)) / (1 - same)
    hidden = sorted(f"{a} {b}" for a, b in
                    [(f"t{t:03d}-r{r}-d{d:03d}", f"t{(t + 30) % 360:03d}-r{r}-d{d:03d}") for t in DIRECTIONS
                     for r in RADII for d in DWELLS
                     if {grid[(t, r, d)], grid[((t + 30) % 360, r, d)]} == {"reset0", "reset1"}])
    return {"class_counts": dict(freq), "per_toggle": per_toggle, "angle_neighbours": tally(angle_pairs),
            "radius_neighbours": tally(radius_pairs),
            "context_independent_double_fraction": null,
            "angle_reset0_reset1_adjacencies_predicted_to_hide_a_window": hidden}


def p3(sweep_root: Path, ref: np.ndarray, groups: list[dict], toggles: list[str], bound: list[Path]) -> dict:
    by_name = {g["name"]: g for g in groups}
    out = {}
    for cand in toggles:
        g = by_name[f"sweep-{cand}"]
        run = mech.load_group(mech.group_file(sweep_root, int(g["group_id"])), bound)
        push = run["push"]
        D = int(run["meta"]["pulse_steps"])
        label = D % PERIOD
        start = float(np.linalg.norm(push[0, 1] - push[0, 0]))
        end = float(np.linalg.norm(push[D, 1] - push[D, 0]))
        out[cand] = {
            "group_id": int(g["group_id"]), "pulse_updates": D, "end_label": label,
            "start_separation": start, "end_separation": end, "end_over_start": end / start,
            "reference_separation_at_end_label": float(np.linalg.norm(ref[1][label] - ref[0][label])),
            "last_six_change": [float(np.linalg.norm(push[D, lane] - push[D - PERIOD, lane])) for lane in range(2)],
        }
    return out


def p3b(sweep_root: Path, groups: list[dict], toggles: list[str], bound: list[Path]) -> dict:
    """Added after the first run (descriptive, post hoc): at each toggle's controls, does the 120-update pulse of the
    sweep settle both lanes onto one period-6 cycle? End separation and each lane's change over its last six updates."""
    by_name = {g["name"]: g for g in groups}
    out = {}
    for cand in toggles:
        th, r, _ = re.match(r"t(\d+)-r([0-9.]+)-d(\d+)$", cand).groups()
        g = by_name[f"sweep-t{int(th):03d}-r{float(r)}-d120"]
        run = mech.load_group(mech.group_file(sweep_root, int(g["group_id"])), bound)
        push = run["push"]
        D = int(run["meta"]["pulse_steps"])
        out[cand] = {"group_id": int(g["group_id"]), "outcome_120": outcome(g["mapping"]),
                     "end_separation_120": float(np.linalg.norm(push[D, 1] - push[D, 0])),
                     "last_six_change_120": [float(np.linalg.norm(push[D, k] - push[D - PERIOD, k])) for k in range(2)]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-audit", type=Path, required=True)
    ap.add_argument("--check-audit", type=Path, nargs="+", required=True)
    ap.add_argument("--sweep-root", type=Path, required=True, help="toggle sweep returned-pod directory")
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    bound = [a.sweep_audit, *a.check_audit, a.reference]

    sweep = json.loads(a.sweep_audit.read_text())
    if not sweep["passed"]:
        raise SystemExit("sweep audit did not pass")
    groups = sweep["hosts"]["pod"]["groups"]
    rows = []
    for f in a.check_audit:
        rep = json.loads(f.read_text())
        if not rep["passed"]:
            raise SystemExit(f"{f}: check audit did not pass")
        rows += rep["rows"]
    with np.load(a.reference) as z:
        ref = z["states"]

    r1 = p1(rows)
    toggles = sorted(r1)
    if len(toggles) != 9:
        raise SystemExit(f"expected nine checked toggles, found {len(toggles)}")
    r2 = p2(groups, toggles)
    r3 = p3(a.sweep_root, ref, groups, toggles, bound)
    r3b = p3b(a.sweep_root, groups, toggles, bound)

    counted = [c for c in toggles if r1[c]["counted"]]
    passing = [c for c in counted if r1[c]["passes"]]
    failing = [c for c in counted if not r1[c]["passes"]]
    if passing and failing:
        ordinal = {"tested": True,
                   "holds": min(r3[c]["end_separation"] for c in failing) > max(r3[c]["end_separation"] for c in passing)}
    else:
        ordinal = {"tested": False, "holds": None}
    summary = {
        "counted_toggles": len(counted), "passes": len(passing), "fails": failing,
        "passes_by_length": {str(d): f"{sum(r1[c]['passes'] for c in counted if r1[c]['dwell'] == d)}/"
                                     f"{sum(1 for c in counted if r1[c]['dwell'] == d)}" for d in DWELLS},
        "reference_toggle_passes": r1[REFERENCE_TOGGLE]["passes"], "p3_ordinal_claim": ordinal,
    }
    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    receipt = {
        "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "kind": "public-toggle-border-rule",
        "plan": "docs/public-toggle-border-rule.md", "script": "tools/analyze_public_toggle_border.py",
        "source_commit": commit, "working_tree_dirty": True, "model_calls": 0,
        "evidence_class": "exploratory, post hoc", "summary": summary, "P1": r1, "P2": r2, "P3": r3,
        "P3b_added_after_first_run": r3b,
        "inputs": [arb.binding(p) for p in sorted(set(map(Path, bound)))],
    }
    a.output.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    for c in toggles:
        x = r1[c]
        print(c, "counted" if x["counted"] else "reference", "B1", x["B1"], "B2", x["B2"], "B3", x["B3"],
              "PASS" if x["passes"] else "fail", x["counts"], "end sep %.2e (x%.3f)" % (r3[c]["end_separation"],
                                                                                    r3[c]["end_over_start"]))
        y = r3b[c]
        print("   120-update pulse at these controls:", y["outcome_120"], "end sep %.1e" % y["end_separation_120"],
              "last six", ["%.1e" % v for v in y["last_six_change_120"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
