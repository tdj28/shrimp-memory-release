#!/usr/bin/env python3
"""Place the public register on the companion study's recorded period-6 branch. Zero model calls.

The companion study (arXiv:2610.31692) continued a period-6 orbit in the steering strength along the register's hold
temperature and saved every converged point with its Floquet multipliers. This script reads that saved continuation
(read-only) and this study's saved trajectories and reports

  * which recorded branch points are the register's two cycles at the hold H, with their multipliers;
  * the unstable orbit between them at H (state and multiplier, interpolated between two recorded neighbours);
  * the folds and flips recorded on this temperature line, and where the E1 write endpoint lies relative to them;
  * whether saved trajectories settle onto the cycles, leave the unstable orbit, and approach the orbit at E1 at the
    rates those multipliers give.

Nothing is fitted and nothing is selected by outcome. Every number is read from a saved file, interpolated between
recorded neighbours, or is a ratio of distances measured on a saved trajectory. The linear-range windows below are
fixed constants. Plan and interpretation: docs/public-register-branch.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ALPHA_H = 0.07293697809562831
LOG10_T_H = -0.15395151820007266
ALPHA_E1 = 0.07299785589927693
# A deviation is treated as inside the linear range of an attracting cycle when its norm lies in this window: small
# against the 5.7e-3 distance from a cycle to the unstable orbit, large against the 1e-11 closure of the references.
LINEAR_WINDOW = (1e-9, 1e-5)
PERIOD = 6


def binding(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def leading(point: dict) -> complex:
    source = point.get("multipliers_refined") or point["multipliers"]
    return complex(source[0][0], source[0][1])


def second_modulus(point: dict) -> float:
    return float(abs(complex(*point["multipliers"][1])))


def bracket(lam: np.ndarray, lo: int, hi: int, target: float) -> int:
    hits = [i for i in range(lo, hi) if (lam[i] - target) * (lam[i + 1] - target) < 0]
    if len(hits) != 1:
        raise SystemExit(f"expected one bracket of {target} in points {lo}..{hi}, found {hits}")
    return hits[0]


def interpolate(lam: np.ndarray, values: np.ndarray, i: int, target: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Linear between i and i+1, and a cubic through i-1..i+2 as an interpolation-error check."""
    w = (target - lam[i]) / (lam[i + 1] - lam[i])
    linear = (1 - w) * values[i] + w * values[i + 1]
    idx = [i - 1, i, i + 1, i + 2]
    x = lam[idx] - target
    vander = np.vander(x, 4, increasing=True)
    coeffs = np.linalg.solve(vander, values[idx].reshape(4, -1))
    cubic = coeffs[0].reshape(values[i].shape)
    return linear, cubic, float(w)


def stage_records(run: Path) -> list[dict] | None:
    """The run's saved stage manifest (result.json "stages"), numbered 0..n-1 in order; None if the run saved none."""
    f = run / "result.json"
    if not f.is_file():
        return None
    stages = json.loads(f.read_text()).get("stages")
    if not isinstance(stages, list) or [s.get("stage_id") for s in stages] != list(range(len(stages))):
        raise SystemExit(f"{f}: stage records are not numbered 0..n-1 in order")
    return stages


def load_stage(run: Path, stage: int, expected_steps: int | None = None, *,
               partial: bool = False) -> tuple[np.ndarray, float, float, list[Path]]:
    """Post-update states of one saved stage, in chronological order. Strict (Codex Task 9, D2).

    Every block file must hold finite float64 states of one shape (n, lanes, 896); its stored indices must run
    contiguously from update 1 with no gap, duplicate or overlap; its name must state its last update; all blocks must
    share one set of controls. Where the run saved its stage manifest (result.json), the files must be exactly the
    blocks it lists, with the listed sizes and SHA-256 digests (each file is hashed from the bytes that are then
    parsed), the block controls must equal the record's, the record must be complete and the stage must have the
    record's length; expected_steps, when given, must agree too. Without a manifest the caller must give
    expected_steps, or ask for partial=True: an explicit partial analysis that accepts any contiguous history from
    update 1 (with a manifest, partial=True accepts an incomplete record whose completed updates are all present).
    Any violation raises SystemExit naming the stage.
    """
    where = f"stage {stage} in {run}"
    if expected_steps is not None and (partial or isinstance(expected_steps, bool)
                                       or not isinstance(expected_steps, (int, np.integer)) or expected_steps < 1):
        raise ValueError("expected_steps must be a positive integer, and is not combined with partial=True")
    files = sorted(Path(p) for p in glob.glob(str(run / f"stage-{stage:02d}-block-*.npz")))
    if not files:
        raise SystemExit(f"no blocks for {where}")
    blocks, controls, shape, digests = [], set(), None, {}
    for f in files:
        raw = f.read_bytes()
        digests[f.name] = (len(raw), hashlib.sha256(raw).hexdigest())
        try:
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                y = z["states"]
                first, last = int(z["first_state_index"]), int(z["last_state_index"])
                controls.add((float(z["alpha"]), float(z["log10_temperature"])))
        except (KeyError, ValueError, OSError) as exc:
            raise SystemExit(f"{where}: {f.name} is not a readable stage block ({type(exc).__name__})")
        name_end = f.stem.rsplit("-", 1)[-1]
        if not name_end.isdigit() or int(name_end) != last:
            raise SystemExit(f"{where}: {f.name} does not name its last update {last}")
        if y.dtype != np.float64 or y.ndim != 3 or y.shape[2] != 896 or len(y) != last - first + 1 \
                or not np.isfinite(y).all():
            raise SystemExit(f"{where}: {f.name} is not a finite float64 block of updates {first}..{last}")
        if shape is not None and y.shape[1:] != shape:
            raise SystemExit(f"{where}: lane shape changes at {f.name}")
        shape = y.shape[1:]
        blocks.append((first, last, y, f))
    blocks.sort(key=lambda b: b[0])
    cursor = 0
    for first, last, _, f in blocks:
        if first != cursor + 1:
            raise SystemExit(f"{where}: {f.name} starts at update {first}, expected {cursor + 1}")
        cursor = last
    if len(controls) != 1:
        raise SystemExit(f"{where} mixes controls {controls}")
    alpha, log10_t = next(iter(controls))
    records = stage_records(run)
    if records is not None:
        if stage >= len(records):
            raise SystemExit(f"{where}: no saved stage record")
        rec = records[stage]
        listed = {b["path"]: (b["bytes"], b["sha256"]) for b in rec.get("blocks", [])}
        if set(listed) != set(digests):
            raise SystemExit(f"{where}: block files {sorted(digests)} differ from the record {sorted(listed)}")
        bad = [name for name in listed if listed[name] != digests[name]]
        if bad:
            raise SystemExit(f"{where}: size or SHA-256 differs from the record for {bad}")
        c = rec.get("controls") or {}
        if (c.get("alpha"), c.get("log10_temperature")) != (alpha, log10_t):
            raise SystemExit(f"{where}: block controls differ from the record")
        if partial:
            if cursor != rec.get("completed_steps"):
                raise SystemExit(f"{where}: {cursor} updates on disk, record says {rec.get('completed_steps')} completed")
        else:
            if rec.get("status") != "completed" or not (cursor == rec.get("steps") == rec.get("completed_steps")):
                raise SystemExit(f"{where}: record {rec.get('status')}, {rec.get('completed_steps')} of "
                                 f"{rec.get('steps')} updates, {cursor} on disk")
            if expected_steps is not None and cursor != expected_steps:
                raise SystemExit(f"{where}: {cursor} updates, expected {expected_steps}")
    elif expected_steps is not None:
        if cursor != expected_steps:
            raise SystemExit(f"{where}: {cursor} updates, expected {expected_steps}")
    elif not partial:
        raise SystemExit(f"{where}: no saved stage manifest and no expected length "
                         "(give expected_steps, or partial=True for an explicit partial analysis)")
    return np.concatenate([b[2] for b in blocks], 0), alpha, log10_t, [b[3] for b in blocks]


def stage_count(run: Path) -> int:
    """Number of saved stages; the stages on disk must be exactly 0..n-1 and, where saved, match the manifest."""
    on_disk = {int(Path(p).name.split("-")[1]) for p in glob.glob(str(run / "stage-*-block-*.npz"))}
    records = stage_records(run)
    n = len(records) if records is not None else len(on_disk)
    if on_disk != set(range(n)):
        raise SystemExit(f"{run}: stages on disk {sorted(on_disk)} are not 0..{n - 1}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch-dir", type=Path, required=True, help="companion continuation (branch.json, states.npy)")
    ap.add_argument("--xor-run", type=Path, required=True, help="audited XOR-chain run directory")
    ap.add_argument("--write-run", type=Path, action="append", default=[], help="run directory with E1 write stages")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    branch_path, states_path = args.branch_dir / "branch.json", args.branch_dir / "states.npy"
    branch = json.loads(branch_path.read_text())
    points = branch["points"]
    states = np.load(states_path)[:, 0, :]
    lam = np.array([p["lambda"] for p in points])
    mult = np.array([leading(p) for p in points])
    if branch["free"] != "alpha" or branch["period"] != PERIOD:
        raise SystemExit("unexpected branch description")
    if branch["args"]["log_temperature"] != LOG10_T_H:
        raise SystemExit("the branch is not on the hold temperature")
    if not all(p["multipliers_converged"] for p in points):
        raise SystemExit("a branch point has unconverged multipliers")
    if np.abs(mult.imag).max() != 0.0:
        raise SystemExit("a leading multiplier is complex")
    mu = mult.real

    # Sheets, from the companion's own recorded crossings.
    crossings = [{"between_indices": c["between_indices"], "type": c["type"],
                  "alpha": [lam[c["between_indices"][0]], lam[c["between_indices"][1]]],
                  "leading_multiplier": [mu[c["between_indices"][0]], mu[c["between_indices"][1]]]}
                 for c in branch["crossings"]]
    kinds = [c["type"] for c in branch["crossings"]]
    if kinds != ["real -1 (flip)", "real +1 (fold)", "real +1 (fold)", "real -1 (flip)"]:
        raise SystemExit(f"unexpected crossings {kinds}")
    flip_a, fold_1, fold_2, flip_b = (c["between_indices"] for c in branch["crossings"])
    first = (0, fold_1[0])                 # attracting sheet that carries cycle 0
    middle = (fold_1[1], fold_2[0])        # one multiplier above +1
    third = (fold_2[1], flip_b[0])         # attracting sheet that carries cycle 1
    for name, (lo, hi), ok in (("first", first, lambda m: abs(m) <= 1.0 + 2e-9), ("middle", middle, lambda m: m > 1),
                               ("third", third, lambda m: abs(m) < 1)):
        bad = [i for i in range(lo, hi + 1) if not ok(mu[i])]
        if bad:
            raise SystemExit(f"{name} sheet has unexpected multipliers at {bad}")
    turn_1 = int(np.argmax(lam[: middle[1]]))            # largest alpha before the branch turns back
    turn_2 = middle[0] + int(np.argmin(lam[middle[0]: third[1]]))

    # The register's reference cycles, as used by the audited XOR chain.
    ref_path = args.xor_run / "reference-cycles.npz"
    with np.load(ref_path) as z:
        ref = z["states"]                                 # (2, 6, 896)

    exact_h = [i for i in range(len(points)) if lam[i] == ALPHA_H]
    exact_e1 = [i for i in range(len(points)) if lam[i] == ALPHA_E1]
    if len(exact_h) != 1 or not first[0] <= exact_h[0] <= first[1]:
        raise SystemExit("no unique first-sheet point at the hold")
    if len(exact_e1) != 1 or not third[0] <= exact_e1[0] <= third[1]:
        raise SystemExit("no unique third-sheet point at E1")
    i_h, i_e1 = exact_h[0], exact_e1[0]
    phase_distances = [float(np.linalg.norm(states[i_h] - ref[0][ph])) for ph in range(PERIOD)]
    phase = int(np.argmin(phase_distances))               # the phase at which the branch states were saved
    a0, a1 = ref[0][phase], ref[1][phase]
    separation = float(np.linalg.norm(a1 - a0))

    i_mid = bracket(lam, middle[0], middle[1], ALPHA_H)
    i_third = bracket(lam, third[0], third[1], ALPHA_H)
    saddle_lin, saddle, w_mid = interpolate(lam, states, i_mid, ALPHA_H)
    third_lin, third_state, w_third = interpolate(lam, states, i_third, ALPHA_H)
    mu_saddle_lin, mu_saddle, _ = interpolate(lam, mu, i_mid, ALPHA_H)
    mu_third_lin, mu_third, _ = interpolate(lam, mu, i_third, ALPHA_H)
    mu_saddle, mu_third = float(mu_saddle), float(mu_third)

    at_hold = {
        "phase_of_saved_branch_states": phase,
        "cycle_0": {
            "branch_index": i_h, "alpha_equals_hold_exactly": True,
            "distance_to_reference_cycle_0": phase_distances[phase],
            "newton_residual": points[i_h]["residual"],
            "leading_multiplier": float(mu[i_h]), "second_largest_modulus": second_modulus(points[i_h]),
        },
        "cycle_1": {
            "bracketing_indices": [i_third, i_third + 1],
            "bracketing_alpha_minus_hold": [float(lam[i_third] - ALPHA_H), float(lam[i_third + 1] - ALPHA_H)],
            "bracketing_multipliers": [float(mu[i_third]), float(mu[i_third + 1])],
            "interpolation_weight": w_third,
            "leading_multiplier_cubic": mu_third, "leading_multiplier_linear": float(mu_third_lin),
            "distance_to_reference_cycle_1_cubic": float(np.linalg.norm(third_state - a1)),
            "distance_to_reference_cycle_1_linear": float(np.linalg.norm(third_lin - a1)),
            "second_largest_modulus_at_neighbours": [second_modulus(points[i_third]),
                                                     second_modulus(points[i_third + 1])],
        },
        "unstable_orbit": {
            "bracketing_indices": [i_mid, i_mid + 1],
            "bracketing_alpha_minus_hold": [float(lam[i_mid] - ALPHA_H), float(lam[i_mid + 1] - ALPHA_H)],
            "bracketing_multipliers": [float(mu[i_mid]), float(mu[i_mid + 1])],
            "interpolation_weight": w_mid,
            "leading_multiplier_cubic": mu_saddle, "leading_multiplier_linear": float(mu_saddle_lin),
            "state_linear_minus_cubic": float(np.linalg.norm(saddle_lin - saddle)),
            "second_largest_modulus_at_neighbours": [second_modulus(points[i_mid]), second_modulus(points[i_mid + 1])],
            "distance_to_cycle_0": float(np.linalg.norm(saddle - a0)),
            "distance_to_cycle_1": float(np.linalg.norm(saddle - a1)),
        },
        "cycle_separation_at_this_phase": separation,
    }

    line = {
        "log10_temperature": LOG10_T_H,
        "points": len(points), "maximum_newton_residual": float(max(p["residual"] for p in points)),
        "alpha_range": [float(lam.min()), float(lam.max())],
        "recorded_crossings": crossings,
        "first_sheet": {"indices": list(first), "alpha": [float(lam[first[0]]), float(lam[first[1]])],
                        "leading_multiplier": [float(mu[first[0]]), float(mu[first[1]])]},
        "middle_sheet": {"indices": list(middle),
                         "leading_multiplier_range": [float(mu[middle[0]: middle[1] + 1].min()),
                                                      float(mu[middle[0]: middle[1] + 1].max())],
                         "largest_second_modulus": max(second_modulus(points[i])
                                                       for i in range(middle[0], middle[1] + 1))},
        "third_sheet": {"indices": list(third), "alpha": [float(lam[third[0]]), float(lam[third[1]])],
                        "leading_multiplier": [float(mu[third[0]]), float(mu[third[1]])]},
        "first_turn": {"index": turn_1, "alpha": float(lam[turn_1]), "leading_multiplier": float(mu[turn_1])},
        "second_turn": {"index": turn_2, "alpha": float(lam[turn_2]), "leading_multiplier": float(mu[turn_2])},
        # Both attracting sheets are recorded from the first sheet's flip to its fold.
        "both_sheets_attracting_alpha": [float(lam[first[0]]), float(lam[turn_1])],
        "hold_minus_lower_edge": float(ALPHA_H - lam[first[0]]),
        "upper_edge_minus_hold": float(lam[turn_1] - ALPHA_H),
    }
    e1 = {
        "branch_index": i_e1, "alpha_equals_e1_exactly": True, "newton_residual": points[i_e1]["residual"],
        "leading_multiplier": float(mu[i_e1]), "second_largest_modulus": second_modulus(points[i_e1]),
        "alpha_e1_minus_first_turn": float(ALPHA_E1 - lam[turn_1]),
        "alpha_e1_minus_fold_bracket": [float(ALPHA_E1 - lam[fold_1[0]]), float(ALPHA_E1 - lam[fold_1[1]])],
        "third_sheet_flip_bracket_minus_alpha_e1": [float(lam[flip_b[0]] - ALPHA_E1), float(lam[flip_b[1]] - ALPHA_E1)],
        "orbit_distance_to_cycle_1_at_hold": float(np.linalg.norm(states[i_e1] - a1)),
        "orbit_distance_to_cycle_0_at_hold": float(np.linalg.norm(states[i_e1] - a0)),
        "orbit_distance_to_unstable_orbit_at_hold": float(np.linalg.norm(states[i_e1] - saddle)),
    }

    # Saved dynamics at the hold: the XOR chain's holds after its toggles.
    bound = [binding(branch_path), binding(states_path), binding(ref_path)]
    settle = {0: [], 1: []}
    cosines = {0: [], 1: []}
    departures = []
    toggles = 0
    n_stages = stage_count(args.xor_run)
    for k in range(n_stages // 2):
        pulse, a_p, t_p, _ = load_stage(args.xor_run, 2 * k)
        hold, a_h, t_h, hold_blocks = load_stage(args.xor_run, 2 * k + 1)
        if (a_h, t_h) != (ALPHA_H, LOG10_T_H):
            raise SystemExit("an odd XOR stage is not a hold")
        if (a_p, t_p) == (ALPHA_H, LOG10_T_H):
            continue                                      # B = 0: no pulse was applied
        if pulse.shape[0] % PERIOD or hold.shape[0] % PERIOD:
            raise SystemExit("stage lengths are not whole periods")
        toggles += 1
        bound += [binding(b) for b in hold_blocks]
        # Hold update j (1-based) is at phase j mod 6, because every earlier stage is a whole number of periods.
        js = [j for j in range(1, hold.shape[0] + 1) if j % PERIOD == phase]
        row = {"toggle": toggles, "pulse_controls": {"alpha": a_p, "log10_temperature": t_p}, "lanes": []}
        for lane in (0, 1):
            x = hold[[j - 1 for j in js], lane]
            end = [float(np.linalg.norm(x[-1] - ref[c][phase])) for c in (0, 1)]
            c = int(np.argmin(end))
            dev = x - ref[c][phase]
            norm = np.linalg.norm(dev, axis=1)
            for i in range(len(js) - 1):
                if LINEAR_WINDOW[0] <= norm[i + 1] and norm[i] <= LINEAR_WINDOW[1]:
                    settle[c].append(float(norm[i + 1] / norm[i]))
                    cosines[c].append(float(dev[i + 1] @ dev[i] / (norm[i + 1] * norm[i])))
            from_saddle = np.linalg.norm(x[:6] - saddle, axis=1)
            row["lanes"].append({
                "lane": lane, "final_cycle": c, "final_distance": end[c],
                "updates_after_pulse": js[:6],
                "distance_from_unstable_orbit": [float(v) for v in from_saddle],
                "distance_ratio_per_period": [float(from_saddle[i + 1] / from_saddle[i]) for i in range(5)],
                "landing_distance_over_cycle_separation": float(from_saddle[0] / separation),
            })
        departures.append(row)

    def summary(values: list[float]) -> dict:
        v = np.array(values)
        return {"count": int(v.size), "median": float(np.median(v)), "minimum": float(v.min()), "maximum": float(v.max())}

    dynamics = {
        "linear_window": list(LINEAR_WINDOW),
        "toggles_examined": toggles,
        "settling_onto_cycle_0": {"ratio_per_period": summary(settle[0]), "cosine_of_successive_deviations":
                                  summary(cosines[0]), "branch_multiplier": float(mu[i_h])},
        "settling_onto_cycle_1": {"ratio_per_period": summary(settle[1]), "cosine_of_successive_deviations":
                                  summary(cosines[1]), "branch_multiplier": mu_third},
        "after_each_toggle": departures,
        "unstable_orbit_multiplier": mu_saddle,
    }

    # Saved dynamics at E1: write stages held at exactly the E1 controls.
    approach = []
    for run in args.write_run:
        for s in range(stage_count(run)):
            traj, a_s, t_s, blocks = load_stage(run, s)
            if (a_s, t_s) != (ALPHA_E1, LOG10_T_H):
                continue
            bound += [binding(b) for b in blocks]
            start = np.load(run / f"stage-{s:02d}-initial.npy")
            full = np.concatenate([start[None], traj], 0)   # index n = updates at E1
            lanes = []
            for lane in (0, 1):
                # the orbit's saved phase is found from the trajectory itself, at its last whole period
                tail = full[-PERIOD:, lane]
                offsets = np.linalg.norm(tail - states[i_e1], axis=1)
                last = full.shape[0] - PERIOD + int(np.argmin(offsets))
                ns = list(range(last % PERIOD, full.shape[0], PERIOD))
                dev = full[ns, lane] - states[i_e1]
                norm = np.linalg.norm(dev, axis=1)
                ratios = [float(norm[i + 1] / norm[i]) for i in range(len(ns) - 1)]
                cos = [float(dev[i + 1] @ dev[i] / (norm[i + 1] * norm[i])) for i in range(len(ns) - 1)]
                started_on = int(np.argmin([np.linalg.norm(full[0, lane] - ref[c][q])
                                            for c in (0, 1) for q in range(PERIOD)]) // PERIOD)
                # Near the fold the map is strongly curved, so with a negative multiplier the one-period ratio
                # alternates about its limit; the geometric mean over two periods cancels that to first order.
                two = [float(np.sqrt(norm[i + 2] / norm[i])) for i in range(len(ns) - 2)]
                half = next((ns[i] for i in range(len(ns)) if norm[i] < 0.5 * norm[0]), None)
                flips = next((ns[i + 1] for i in range(len(cos)) if cos[i] < 0), None)
                lanes.append({"lane": lane, "started_on_cycle": started_on, "updates": ns,
                              "distance_to_e1_orbit": [float(v) for v in norm],
                              "ratio_per_period": ratios, "cosine_of_successive_deviations": cos,
                              "two_period_geometric_mean_ratio": two,
                              "geometric_mean_ratio_last_four_periods": float((norm[-1] / norm[-5]) ** 0.25),
                              "first_sampled_update_below_half_start_distance": half,
                              "first_sampled_update_with_reversed_deviation": flips,
                              "lag6_residual_at_end": float(np.linalg.norm(full[-1, lane] - full[-1 - PERIOD, lane]))})
            approach.append({"run": str(run), "stage": s, "updates": int(traj.shape[0]), "lanes": lanes})

    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    receipt = {
        "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "zero_call_analysis_of_saved_records",
        "kind": "public_register_on_companion_branch",
        "source_commit": commit,
        "script": binding(Path(__file__).resolve()),
        "model_calls": 0,
        "evidence_class": "adapted exploratory",
        "scope": ("Reads the companion study's saved continuation and this study's saved trajectories. The multipliers "
                  "are the companion study's Arnoldi estimates at Newton-converged points; values at the hold on the "
                  "middle and third sheets are interpolated between two recorded neighbours. No basin, no certified "
                  "bifurcation and no reliability is measured."),
        "controls": {"alpha_hold": ALPHA_H, "log10_temperature_hold": LOG10_T_H, "alpha_e1": ALPHA_E1},
        "branch_arguments": {k: branch["args"][k] for k in ("model", "layer", "dtype", "concept", "log_temperature",
                                                             "tol", "k", "refine_k", "free", "ds", "ds_max")},
        "temperature_line": line,
        "at_hold": at_hold,
        "e1_endpoint": e1,
        "saved_dynamics_at_hold": dynamics,
        "saved_dynamics_at_e1": approach,
        "inputs": bound,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")

    def show(label: str, value) -> None:
        print(f"{label:<58} {value}")
    show("phase of saved branch states", phase)
    show("cycle 0: branch point, distance, multiplier", (i_h, f"{phase_distances[phase]:.2e}", f"{mu[i_h]:+.6f}"))
    show("cycle 1: distance (cubic), multiplier (cubic, linear)", (f"{at_hold['cycle_1']['distance_to_reference_cycle_1_cubic']:.2e}",
                                                                  f"{mu_third:+.6f}", f"{float(mu_third_lin):+.6f}"))
    show("unstable orbit: multiplier (cubic, linear)", (f"{mu_saddle:+.6f}", f"{float(mu_saddle_lin):+.6f}"))
    show("unstable orbit: distance to cycle 0, cycle 1; separation", (f"{at_hold['unstable_orbit']['distance_to_cycle_0']:.6f}",
                                                                      f"{at_hold['unstable_orbit']['distance_to_cycle_1']:.6f}", f"{separation:.6f}"))
    show("both sheets attracting for alpha in", line["both_sheets_attracting_alpha"])
    show("hold - lower edge, upper edge - hold", (f"{line['hold_minus_lower_edge']:.3e}", f"{line['upper_edge_minus_hold']:.3e}"))
    show("first turn (index, alpha, multiplier)", (turn_1, lam[turn_1], f"{mu[turn_1]:+.6f}"))
    show("E1: multiplier, alpha_E1 - first turn", (f"{mu[i_e1]:+.6f}", f"{e1['alpha_e1_minus_first_turn']:.3e}"))
    show("E1: third-sheet flip bracket - alpha_E1", [f"{v:.3e}" for v in e1["third_sheet_flip_bracket_minus_alpha_e1"]])
    show("settling onto cycle 0 (ratio per period)", dynamics["settling_onto_cycle_0"]["ratio_per_period"])
    show("settling onto cycle 1 (ratio per period)", dynamics["settling_onto_cycle_1"]["ratio_per_period"])
    show("cosines cycle 0 / cycle 1 (median)", (dynamics["settling_onto_cycle_0"]["cosine_of_successive_deviations"]["median"],
                                                dynamics["settling_onto_cycle_1"]["cosine_of_successive_deviations"]["median"]))
    for lane in departures[0]["lanes"]:
        show(f"toggle 1 lane {lane['lane']} -> cycle {lane['final_cycle']}: distance from unstable orbit",
             " ".join(f"{v:.6f}" for v in lane["distance_from_unstable_orbit"]))
        show("    ratio per period", " ".join(f"{v:.4f}" for v in lane["distance_ratio_per_period"]))
    for row in approach[:2]:
        for lane in row["lanes"]:
            show(f"E1 stage {row['stage']} lane {lane['lane']} (from cycle {lane['started_on_cycle']}): distance to E1 orbit",
                 " ".join(f"{v:.2e}" for v in lane["distance_to_e1_orbit"][::3]))
            show("    last-four-period mean ratio, half-distance update, lag-6 residual at end",
                 (f"{lane['geometric_mean_ratio_last_four_periods']:.4f}",
                  lane["first_sampled_update_below_half_start_distance"], f"{lane['lag6_residual_at_end']:.2e}"))
    print("receipt", args.output)


if __name__ == "__main__":
    sys.exit(main())
