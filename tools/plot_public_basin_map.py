#!/usr/bin/env python3
"""Figure for the basin map at H (docs/public-basin-map.md). Zero model calls.

Three panels, from the analysis receipt and the labels written beside it:

  (a) M, the 65 x 65 phase-zero slice through C_p spanned by v_p and the part of A_0[0] - A_1[0] orthogonal to it,
      with the projections of A_0[0] and A_1[0] and the end states of the 12-, 18- and 24-update pulses of the toggle
      timing test (the old bracket), and the outline of Z;
  (b) Z, the zoom of one sixteenth of M's width;
  (c) basin stability: the fraction of kicks of each size that return to the cycle they were kicked from, with
      Wilson 95% intervals.

The pulse end states are re-read from the toggle sweep's 120-update pulse (sweep group 26), whose first 12, 18 and
24 updates are those pulses' paths. Starts not yet run (labels -2, partial receipts only) are drawn pale.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_public_toggle_mechanism as mech  # noqa: E402

N, HALF_M, HALF_Z = 65, 5e-3, 3.125e-4
CYC = ("#2b8cbe", "#d6604d")          # cycle 0, cycle 1 (as in the XOR, adder and toggle figures)
UNRESOLVED, NOT_RUN = "#111111", "#ecebe6"
INK, MUTED = "#0b0b0b", "#898781"
CMAP = ListedColormap([NOT_RUN, UNRESOLVED, CYC[0], CYC[1]])   # labels -2, -1, 0, 1 shifted by +2
PULSES = (12, 18, 24)
SWEEP = Path("/archive/shrimp-memory-data/audits/public-toggle-sweep/20261109T230942Z/returned-pod")


def show(ax, labels, cs, ct, half, unit=1e-3):
    grid = labels[: N * N].reshape(N, N) + 2
    step = 2 * half / (N - 1)
    ext = [(cs - half - step / 2) / unit, (cs + half + step / 2) / unit,
           (ct - half - step / 2) / unit, (ct + half + step / 2) / unit]
    ax.imshow(grid, origin="lower", extent=ext, cmap=CMAP, vmin=0, vmax=3, interpolation="nearest", aspect="equal")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--receipt", type=Path, required=True, help="analysis receipt; labels-<set>.npz beside it")
    ap.add_argument("--starts-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    r = json.loads(a.receipt.read_text())
    lab = {k: np.load(a.receipt.parent / f"labels-{k}.npz")["labels"] for k in ("M", "Z")
           if (a.receipt.parent / f"labels-{k}.npz").exists()}
    with np.load(a.starts_dir / "plane.npz") as z:
        c, e1, e2 = z["origin"], z["e1"], z["e2"]
    P = mech.load_group(mech.group_file(SWEEP, 26), [])["push"]
    landings = {(u, lane): ((P[u, lane] - c) @ e1, (P[u, lane] - c) @ e2) for u in PULSES for lane in (0, 1)}

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5})
    fig, ax = plt.subplots(1, 3, figsize=(9.8, 3.3), constrained_layout=True,
                           gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    partial = any(r.get(k, {}).get("missing_starts", 0) for k in ("M", "Z", "S"))

    show(ax[0], lab["M"], 0.0, 0.0, HALF_M)
    for cyc, key in ((0, "A0"), (1, "A1")):
        b = r["B2_cycles"][key]
        ax[0].plot(b["s"] / 1e-3, b["t"] / 1e-3, marker="*", ms=11, mfc=CYC[cyc], mec="white", mew=0.9, ls="none")
        ax[0].annotate(f"$A_{cyc}$", (b["s"] / 1e-3, b["t"] / 1e-3), xytext=(6, 4) if cyc == 0 else (-17, -12),
                       textcoords="offset points", color="white", fontsize=8.5, weight="bold")
    ax[0].plot(0, 0, marker="+", ms=8, color="white", mew=1.4)
    ax[0].annotate("$C_p$", (0, 0), xytext=(-16, 5), textcoords="offset points", color="white", fontsize=8)
    for (u, lane), (s, t) in landings.items():
        ax[0].plot(s / 1e-3, t / 1e-3, marker="s" if lane else "o", ms=3.2, mfc="white", mec=INK, mew=0.6, ls="none")
    if "Z" in r:
        zs, zt = r["Z"]["centre"]
        ax[0].add_patch(Rectangle(((zs - HALF_Z) / 1e-3, (zt - HALF_Z) / 1e-3), 2 * HALF_Z / 1e-3, 2 * HALF_Z / 1e-3,
                                  fill=False, ec="white", lw=1.0))
    ax[0].set(xlabel=r"$s$ along $v_p$ ($10^{-3}$)", ylabel=r"$t$ toward $A_0-A_1$ ($10^{-3}$)",
              title="(a) phase-zero slice at $H$")

    if "Z" in lab:
        zs, zt = r["Z"]["centre"]
        show(ax[1], lab["Z"], zs, zt, HALF_Z)
        ax[1].set(xlabel=r"$s$ ($10^{-3}$)", ylabel=r"$t$ ($10^{-3}$)", title="(b) zoom, 1/16 of the width")
    else:
        ax[1].text(0.5, 0.5, "zoom not yet run", ha="center", va="center", transform=ax[1].transAxes, color=MUTED)
        ax[1].set(xticks=[], yticks=[], title="(b) zoom, 1/16 of the width")

    bs = r.get("basin_stability", [])
    for cyc, key in ((0, "A0"), (1, "A1")):
        rows = sorted((x for x in bs if x["cycle"] == key), key=lambda x: x["eps"])
        if rows:
            e = np.array([x["eps"] for x in rows]) * (1 + 0.04 * (2 * cyc - 1))
            f = np.array([x["fraction"] for x in rows])
            lo, hi = np.array([x["wilson95"] for x in rows]).T
            ax[2].errorbar(e, f, yerr=[f - lo, hi - f], color=CYC[cyc], marker="o", ms=4, capsize=2.5, lw=1.2,
                           label=f"kicked from $A_{cyc}$")
    ax[2].set(xscale="log", ylim=(-0.03, 1.03), xlabel=r"kick size $\varepsilon$ (fraction of $|A_c[0]|$)",
              ylabel="fraction returning", title="(c) basin stability at $H$")
    ax[2].set_xticks([0.005, 0.01, 0.02], ["0.005", "0.01", "0.02"])
    ax[2].minorticks_off()
    if bs:
        ax[2].legend(loc="lower left", frameon=False)
    else:
        ax[2].text(0.5, 0.5, "not yet run", ha="center", va="center", transform=ax[2].transAxes, color=MUTED)

    handles = [Patch(fc=CYC[0], label="$A_0$"), Patch(fc=CYC[1], label="$A_1$"),
               Patch(fc=UNRESOLVED, label="unresolved")]
    if partial:
        handles.append(Patch(fc=NOT_RUN, ec=MUTED, lw=0.5, label="not yet run"))
    handles += [plt.Line2D([], [], marker="o", ms=3.5, mfc="white", mec=INK, ls="none", label="pulse ends, from $A_0$"),
                plt.Line2D([], [], marker="s", ms=3.5, mfc="white", mec=INK, ls="none", label="pulse ends, from $A_1$")]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    fig.savefig(a.output, dpi=220)
    print(a.output, "partial" if partial else "complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
