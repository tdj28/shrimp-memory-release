#!/usr/bin/env python3
"""Figure for the write-passage analysis (docs/public-write-passage.md). Zero model calls.

Four panels, from the measurement receipt, the prediction file and the saved states the receipt binds:

  (a) the lane that starts on cycle 0, at the six write strengths on the hold-temperature line: its coordinate
      along the line between the two reference cycles, once per period, through the 120-update pulse and the
      240-update hold;
  (b) the per-period displacement of the same lanes during the pulse;
  (c) the update at which each of those writes passed, against the distance of its endpoint beyond the fold, with
      the prediction that has no fitted number;
  (d) the same for the write of 0 along the hold-strength line, with its two-parameter line.

Nothing is measured here that the receipt does not already hold. The trajectories of (a) and (b) are re-read from the
stage files of the pilot runs, and the digest of those files is checked against the receipt before anything is drawn.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_public_write_passage as W  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAMP = ("#9e9ac8", "#807dba", "#6a51a3", "#3f007d")   # beyond the fold, ordinal: darker = farther beyond
SHORT = ("#8c8c8c", "#b0b0b0")                        # short of the fold: nearer, farther
INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
UNIT_ALPHA, UNIT_T = 1e-6, 1e-5


def unique_on_line(rows, bit, key):
    out = {}
    for row in rows:
        if row["requested_bit"] == bit and row["passage_lane"] and row["on_recorded_line"]:
            out.setdefault((row["alpha"], row["log10_temperature"]), row)
    return sorted(out.values(), key=lambda r: r[key])


def lane_series(row, refs, digests, phase):
    """Coordinate and per-period displacement of one passage lane, sampled once per period at one phase."""
    run = Path(row["run"])
    if W.blocks_digest(run) != digests[row["run"]]:
        raise SystemExit(f"stage files of {run} differ from the receipt")
    group = next(g for g in W.load_design(run)["groups"] if g["name"] == row["group"])
    start = np.load(run / f"stage-{group['endpoint_stage']:02d}-initial.npy")
    endpoint = W.load_stage(run, group["endpoint_stage"], group["alpha"], group["log10_temperature"])
    hold = W.load_stage(run, group["hold_stage"], W.ALPHA_H, W.LOG10_T_H)
    lane = row["lane_start_cycle"]
    c = W.coordinate(np.concatenate([endpoint[:, lane], hold[:, lane]], axis=0), refs, 0)
    s = W.displacement(start[lane], endpoint[:, lane])
    updates = np.arange(1, len(c) + 1)
    pick = updates[updates % W.PERIOD == phase]
    n = np.arange(phase, len(s), W.PERIOD)
    return {"updates": pick.tolist(), "coordinate": c[pick - 1].tolist(),
            "displacement_from_update": n.tolist(), "displacement": s[n].tolist()}


def style_axis(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=SECONDARY, labelsize=7, length=2.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--reference-cycles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, help="optional PNG copy for looking at the layout")
    args = parser.parse_args()

    receipt = json.loads(args.receipt.read_text())
    prediction = json.loads(args.prediction.read_text())
    if W.binding(args.prediction)["sha256"] != receipt["prediction"]["sha256"]:
        raise SystemExit("prediction file is not the one the receipt binds")
    if W.binding(args.reference_cycles)["sha256"] != receipt["reference_cycles"]["sha256"]:
        raise SystemExit("reference cycles are not the ones the receipt binds")
    with np.load(args.reference_cycles) as z:
        refs = z["states"]
    digests = {r["run"]: r["stage_files"] for r in receipt["runs"]}
    phase = int(prediction["hold_unstable_orbit"]["saved_phase"])
    c_unstable = float(prediction["hold_unstable_orbit"]["coordinate_at_saved_phase"])
    ab = float(prediction["write1_coefficient_ab_per_unit_alpha"])
    d_alpha = float(prediction["hold_to_first_sheet_fold_alpha"])
    d_log_t = float(prediction["hold_to_that_fold_log10_temperature"])
    primary = W.PRIMARY

    w1 = unique_on_line(receipt["rows"], 1, "alpha_beyond_fold")
    w0 = unique_on_line(receipt["rows"], 0, "log10_temperature_beyond_fold")
    beyond = [r for r in w1 if r["alpha_beyond_fold"] > 0]
    short = [r for r in w1 if r["alpha_beyond_fold"] <= 0][::-1]          # nearest the fold first
    if len(beyond) != len(RAMP) or len(short) != len(SHORT):
        raise SystemExit("unexpected number of write-1 settings on the recorded line")
    colour = {id(r): RAMP[i] for i, r in enumerate(beyond)}
    colour.update({id(r): SHORT[i] for i, r in enumerate(short)})
    dash = {id(r): "-" for r in beyond}
    dash.update({id(short[0]): (0, (4, 1.5)), id(short[1]): (0, (1, 1.2))})
    series = {id(r): lane_series(r, refs, digests, phase) for r in w1}

    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5,
                         "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.7), constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes
    for ax in axes.ravel():
        style_axis(ax)

    def label(r):
        return f"{r['alpha_beyond_fold'] / UNIT_ALPHA:+.1f}".replace("-", "−")

    # (a) coordinate through pulse and hold
    ax_a.axvspan(0, W.ENDPOINT_STEPS, color="#f4f3ee", lw=0)
    ax_a.axhline(c_unstable, color=MUTED, lw=0.7, ls=(0, (1, 2)))
    for r in w1[::-1]:
        d = series[id(r)]
        ax_a.plot(d["updates"], d["coordinate"], color=colour[id(r)], ls=dash[id(r)], lw=1.5)
    ax_a.set_xlim(0, W.ENDPOINT_STEPS + W.HOLD_STEPS)
    ax_a.set_ylim(-0.08, 1.36)
    ax_a.set_xticks(range(0, 361, 60))
    ax_a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_a.set_yticklabels(["0\n(cycle 0)", "", "0.5", "", "1\n(cycle 1)"])
    ax_a.set_xlabel("updates since the write began")
    ax_a.set_ylabel("position between the two cycles")
    ax_a.set_title("(a) write of 1: the state through pulse and hold", loc="left", color=INK)
    ax_a.text(60, 1.29, "write pulse", ha="center", va="center", color=SECONDARY, fontsize=6.5)
    ax_a.text(240, 1.29, "hold at $H$", ha="center", va="center", color=SECONDARY, fontsize=6.5)
    ax_a.text(356, c_unstable - 0.025, "unstable orbit at $H$", ha="right", va="top", color=SECONDARY,
              fontsize=6.5)
    handles = [Line2D([], [], color=colour[id(r)], ls=dash[id(r)], lw=1.5, label=label(r)) for r in w1[::-1]]
    ax_a.legend(handles=handles, title="strength beyond the fold ($10^{-6}$)", title_fontsize=6.5,
                ncol=2, loc="center right", bbox_to_anchor=(1.0, 0.5), frameon=False, handlelength=2.2,
                labelspacing=0.25, columnspacing=1.2)

    # (b) per-period displacement during the pulse
    for r in w1[::-1]:
        d = series[id(r)]
        ax_b.semilogy(d["displacement_from_update"], d["displacement"], color=colour[id(r)], ls=dash[id(r)], lw=1.5)
    ax_b.set_xlim(0, W.ENDPOINT_STEPS)
    ax_b.set_xticks(range(0, 121, 30))
    ax_b.set_ylim(1e-9, 3e-2)
    ax_b.set_xlabel("updates since the write began")
    ax_b.set_ylabel("distance moved in one period")
    ax_b.set_title("(b) write of 1: a bottleneck, or settling", loc="left", color=INK)

    # (c) write of 1: passage against distance beyond the fold
    x = np.linspace(0.35, 18.0, 600)
    ax_c.axvspan(-5.5, 0, color="#f4f3ee", lw=0)
    ax_c.axhline(W.ENDPOINT_STEPS, color=MUTED, lw=0.7, ls=(0, (4, 2)))
    ax_c.plot(x, [W.passage_updates(v * UNIT_ALPHA, d_alpha, ab) for v in x], color=INK, lw=1.2)
    ax_c.plot(x, [W.reach_updates(v * UNIT_ALPHA, d_alpha, ab) for v in x], color=INK, lw=0.9, ls=(0, (3, 1.5)))
    points_c = []
    for r in w1:
        xv = r["alpha_beyond_fold"] / UNIT_ALPHA
        if r["passage_within_endpoint_stage"]:
            ax_c.plot(xv, r["passage_update"][primary], "o", ms=5.5, color=colour[id(r)], mec="white", mew=0.8,
                      zorder=4)
            ax_c.plot(xv, r["update_reaching_hold_unstable_orbit"], "o", ms=4.5, mfc="white", mec=colour[id(r)],
                      mew=1.2, zorder=4)
        elif r["alpha_beyond_fold"] > 0:      # beyond the fold: still creeping when the pulse ended
            ax_c.annotate("", xy=(xv, W.ENDPOINT_STEPS + 40), xytext=(xv, W.ENDPOINT_STEPS),
                          arrowprops={"arrowstyle": "-|>", "color": colour[id(r)], "lw": 1.2, "mutation_scale": 7})
            ax_c.plot(xv, W.ENDPOINT_STEPS, "s", ms=4.5, color=colour[id(r)], mec="white", mew=0.8, zorder=4)
        else:                                 # short of the fold: settled on the orbit that continues cycle 0
            ax_c.plot(xv, W.ENDPOINT_STEPS, "x", ms=5.0, color=SECONDARY, mew=1.3, zorder=4)
        points_c.append({"alpha_beyond_fold": r["alpha_beyond_fold"], "group": r["group"],
                         "passage_update": r["passage_update"][primary],
                         "update_reaching_hold_unstable_orbit": r["update_reaching_hold_unstable_orbit"],
                         "predicted_passage_update": r["predicted_passage_update"],
                         "predicted_update_reaching_hold_unstable_orbit":
                             r["predicted_update_reaching_hold_unstable_orbit"],
                         "requested_bit_stored": r["requested_bit_stored"]})
    ax_c.set_xlim(-5.5, 18.0)
    ax_c.set_ylim(0, 290)
    ax_c.set_xticks(range(-4, 17, 4))
    ax_c.set_xticklabels([str(v).replace("-", "−") for v in range(-4, 17, 4)])
    ax_c.set_xlabel("endpoint strength beyond the fold ($10^{-6}$)")
    ax_c.set_ylabel("update")
    ax_c.set_title("(c) write of 1: time to pass, no fitted number", loc="left", color=INK)
    ax_c.text(-2.75, 14, "short of\nthe fold", ha="center", va="bottom", color=SECONDARY, fontsize=6.5)
    ax_c.text(17.7, W.ENDPOINT_STEPS + 4, "pulse ends", ha="right", va="bottom", color=SECONDARY, fontsize=6.5)
    ax_c.legend(handles=[
        Line2D([], [], color=INK, lw=1.2, label="predicted: passed"),
        Line2D([], [], color=INK, lw=0.9, ls=(0, (3, 1.5)), label="predicted: at unstable orbit"),
        Line2D([], [], marker="o", ms=5.5, color=RAMP[2], mec="white", mew=0.8, ls="", label="measured: passed"),
        Line2D([], [], marker="o", ms=4.5, mfc="white", mec=RAMP[2], mew=1.2, ls="",
               label="measured: at unstable orbit"),
        Line2D([], [], marker="s", ms=4.5, color=RAMP[0], mec="white", mew=0.8, ls="",
               label="not passed within the pulse"),
        Line2D([], [], marker="x", ms=5.0, color=SECONDARY, mew=1.3, ls="",
               label="settled on cycle 0")],
        loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False, labelspacing=0.25, handlelength=2.0)

    # (d) write of 0: passage against distance beyond the fold candidate
    fit = receipt["comparisons"]["write0_line"]
    x0 = np.linspace(0.3, 22.0, 600)
    ax_d.axhline(W.ENDPOINT_STEPS, color=MUTED, lw=0.7, ls=(0, (4, 2)))
    ax_d.plot(x0, [fit["intercept"] + fit["slope"] * W.regressor(v * UNIT_T, d_log_t) for v in x0], color=INK, lw=1.2)
    points_d, longest = [], W.ENDPOINT_STEPS + W.ENDPOINT_STEPS
    for r in w0:
        xv = r["log10_temperature_beyond_fold"] / UNIT_T
        if r["passage_within_endpoint_stage"]:
            ax_d.plot(xv, r["passage_update"][primary], "o", ms=5.5, color=SECONDARY, mec="white", mew=0.8, zorder=4)
        else:
            ax_d.annotate("", xy=(xv, longest + 34), xytext=(xv, longest),
                          arrowprops={"arrowstyle": "-|>", "color": SECONDARY, "lw": 1.2, "mutation_scale": 7})
            ax_d.plot(xv, longest, "s", ms=4.5, color=SECONDARY, mec="white", mew=0.8, zorder=4)
        points_d.append({"log10_temperature_beyond_fold": r["log10_temperature_beyond_fold"], "group": r["group"],
                         "passage_update": r["passage_update"][primary],
                         "requested_bit_stored": r["requested_bit_stored"]})
    joined = None
    for cont in receipt["continuations"]:
        for arm in cont["arms"]:
            for lane in arm["lanes"]:
                if cont["requested_bit"] == 0 and "joined_240_updates_at_this_endpoint" in lane:
                    joined = lane["joined_240_updates_at_this_endpoint"]
    if joined is None or joined["passage_update"][primary] is not None:
        raise SystemExit("the joined 240-update continuation is missing or shows a passage")
    ax_d.set_xlim(0, 22.0)
    ax_d.set_ylim(0, 330)
    ax_d.set_xticks(range(0, 21, 5))
    ax_d.set_xlabel("endpoint $\\log_{10}T$ beyond the fold candidate ($10^{-5}$)")
    ax_d.set_ylabel("update")
    ax_d.set_title("(d) write of 0: the same form, two fitted numbers", loc="left", color=INK)
    ax_d.text(21.7, W.ENDPOINT_STEPS + 4, "pulse ends", ha="right", va="bottom", color=SECONDARY, fontsize=6.5)
    ax_d.legend(handles=[
        Line2D([], [], color=INK, lw=1.2, label="line fitted to the five passages"),
        Line2D([], [], marker="o", ms=5.5, color=SECONDARY, mec="white", mew=0.8, ls="", label="measured: passed"),
        Line2D([], [], marker="s", ms=4.5, color=SECONDARY, mec="white", mew=0.8, ls="",
               label="not passed after 240 updates")],
        loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False, labelspacing=0.25, handlelength=2.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    if args.preview:
        fig.savefig(args.preview, dpi=200)

    description = (
        "Four panels. (a) The lane that starts on cycle 0 during a write of 1 at six endpoint strengths on the "
        "hold-temperature line, from 3.8e-6 short of the fold of the cycle-0 sheet to 16.2e-6 beyond it: position "
        "between the two reference cycles, sampled once per period, over the 120-update pulse and the 240-update "
        "hold. The three endpoints farthest beyond the fold cross to cycle 1, later the nearer the fold; the "
        "endpoint 1.2e-6 beyond it and the two short of it stay near cycle 0 and return to it. (b) Distance moved "
        "in one period by the same lanes during the pulse, logarithmic axis: beyond the fold it falls to a minimum "
        "and rises again; short of the fold it shrinks by a constant factor per period. (c) Update at which the "
        "write passed three quarters of the way to cycle 1, and update at which it reached the hold's unstable "
        "orbit, against distance beyond the fold, with the two predictions of the fold normal form that use the "
        "coefficient from the companion study's multipliers and no fitted number. (d) The same passage update for "
        "the write of 0 against distance in log10 temperature beyond its fold candidate, with the two-parameter "
        "line fitted to the five passages; the endpoint nearest the fold had not passed after 240 updates.")
    out = {
        "schema_version": 1, "figure_id": "public-write-passage",
        "title": "Write time against the distance of the write endpoint beyond a fold",
        "description": description, "alt_text": description,
        "evidence": "post hoc plot of an adapted exploratory, zero-call analysis of saved pilot runs",
        "utc": W.now(), "model_calls": 0,
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "generator_sources": [W.binding(Path(__file__).resolve()), W.binding(Path(W.__file__).resolve())],
        "data_sources": [W.binding(args.receipt), W.binding(args.prediction), W.binding(args.reference_cycles)]
                        + [{"run": run, "stage_files": digests[run]} for run in sorted({r["run"] for r in w1})],
        "plotted_data": {
            "sampling_phase": phase, "hold_unstable_orbit_coordinate": c_unstable,
            "write1_lanes": [{"group": r["group"], "run": r["run"], "alpha": r["alpha"],
                              "alpha_beyond_fold": r["alpha_beyond_fold"], **series[id(r)]} for r in w1],
            "write1_points": points_c,
            "write1_prediction": {"ab_per_unit_alpha": ab, "hold_to_fold_alpha": d_alpha,
                                  "formula_passed": "6 [pi/2 + arctan(sqrt(D/Delta))] / sqrt(ab Delta)",
                                  "formula_at_unstable_orbit": "6 [2 arctan(sqrt(D/Delta))] / sqrt(ab Delta)"},
            "write0_points": points_d,
            "write0_line": {"intercept": fit["intercept"], "slope": fit["slope"], "hold_to_fold_log10_temperature": d_log_t,
                            "regressor": "[pi/2 + arctan(sqrt(D/Delta))] / sqrt(Delta)"},
            "write0_joined_continuation_coordinate_at_updates_60_120_180_240":
                joined["coordinate_at_updates_60_120_180_240"]},
        "matplotlib_version": matplotlib.__version__,
        "outputs": [W.binding(args.output)],
        "accessibility": "Strength is carried by an ordinal one-hue ramp and by a legend in the same order; the two "
                         "endpoints short of the fold are gray with dashed and dotted lines; measured and predicted "
                         "values differ by marker and line style, not by color.",
    }
    args.receipt_output.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", args.output, "and", args.receipt_output)
    for r in w1:
        d = series[id(r)]
        print(f"  {r['group']:>13} beyond {r['alpha_beyond_fold']:+.3e}: coordinate at pulse end {d['coordinate'][19]:.4f}, "
              f"largest {max(d['coordinate']):.4f}, final {d['coordinate'][-1]:.6f}, "
              f"smallest displacement {min(d['displacement']):.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
