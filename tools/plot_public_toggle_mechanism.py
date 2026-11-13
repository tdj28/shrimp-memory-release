#!/usr/bin/env python3
"""Figure: why the 30-degree toggle exchanges the cycles only from phase 0. Zero model calls.

(a) Each lane's distance to the pulse controls' own cycle C_p during the pod's 120-update pulse from phase 0, whose
first 12 updates are the toggle pulse; dotted lines mark when the dials returned to H in a measured run.
(b) The period map along C_p's slow line at label 0 (added 12 November 2026, docs/public-toggle-fold.md): every saved
pair of label-0 visits six updates apart lying 98 % or more on the line, and the two phase-0 lanes' first two periods.
(c) Each lane's coefficient along C_p's slow line at label 0, every period, filled with the cycle it reached when the
dials returned to H at that time (open: never returned there); the band is the bracket of the basin boundary.
(d) For the 12-update pulse from each start phase, each lane's closest approach to the unstable orbit at H, filled
with the cycle it reached.
Reads docs/receipts/public-toggle-mechanism.json, docs/receipts/public-toggle-timing.json,
docs/receipts/public-toggle-fold.json and the saved pulse.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_public_toggle_mechanism as mech  # noqa: E402

CYC = ('#2b8cbe', '#d6604d')          # cycle 0, cycle 1 (as in the XOR and adder figures)
GOLD = '#ffd23f'
MARK = ('o', 's')                     # lane from cycle 0, lane from cycle 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mechanism', type=Path, required=True)
    ap.add_argument('--timing', type=Path, help='timing-test score; without it, 18 and 24 updates are open')
    ap.add_argument('--fold', type=Path, required=True, help='docs/receipts/public-toggle-fold.json')
    ap.add_argument('--sweep-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    M = json.loads(a.mechanism.read_text())
    T = json.loads(a.timing.read_text()) if a.timing else {'groups': {}, 'verdicts': {
        'boundary_bracket_on_the_slow_line': [0.000802, 0.002073]}}
    plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5})

    P = mech.load_group(mech.group_file(a.sweep_root, 26), [])['push']        # (121, 2, 896), from phase 0
    cp = {(t % 6): P[t, 0] for t in range(115, 121)}
    returned = {12: M['groups']['check-00']['at_hold'], 48: M['groups']['sweep-25']['at_hold'],
                120: M['groups']['sweep-26']['at_hold']}
    for g in T['groups'].values():
        returned[g['pulse_updates']] = g['at_hold']
    fates = {t: [h['final_cycle'] for h in v] for t, v in returned.items()}

    F = json.loads(a.fold.read_text())
    fig, ax = plt.subplots(1, 4, figsize=(9.8, 2.6), constrained_layout=True,
                           gridspec_kw=dict(width_ratios=[1.0, 1.05, 1.0, 0.72]))
    ax = [ax[0], ax[2], ax[3], ax[1]]          # keep the old panel code; the new map is drawn in the second slot
    # (a) distance to C_p
    tt = np.arange(0, 49)
    for lane in (0, 1):
        d = [np.linalg.norm(P[t, lane] - cp[t % 6]) for t in tt]
        ax[0].semilogy(tt, d, color=CYC[lane], lw=1.2, label=f'lane started on cycle {lane}')
        ax[0].semilogy(tt[::6], d[::6], MARK[lane], ms=3.5, color=CYC[lane])
    for t in sorted(fates):
        if t <= 48:
            f = fates[t]
            word = {(1, 0): 'exchange', (1, 1): 'reset to 1', (0, 0): 'reset to 0', (0, 1): 'hold'}[tuple(f)]
            ax[0].axvline(t, color='0.45', lw=0.7, ls=':')
            ax[0].text(t - 0.5 if t == 48 else t + 0.5, 2.4e-2, f'{t}: {word}', rotation=90, va='top',
                       ha='right' if t == 48 else 'left', fontsize=6.5, color='0.2',
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.6))
    ax[0].set_xlabel('update since the pulse began'); ax[0].set_ylabel(r'distance to $C_p$ (same label)')
    ax[0].set_ylim(1e-9, 3e-2); ax[0].set_xlim(0, 49.5)
    ax[0].legend(loc='lower left', frameon=False)
    ax[0].set_title('(a) distance to $C_p$', loc='left')

    # (b) the slow line at label 0
    lo, hi = T['verdicts']['boundary_bracket_on_the_slow_line']
    ax[1].axhspan(lo, hi, color='0.88', lw=0)
    ax[1].text(25, np.sqrt(lo * hi) * 1.9, 'outcome bracket at $H$', ha='left', va='bottom', fontsize=6.3, color='0.3')
    ax[1].axhline(0, color='0.3', lw=0.7)
    for row in M['label0_continuation_every_period']:
        t = row['update']
        if t > 42:
            continue
        for lane in (0, 1):
            c = row[f'lane{lane}_coefficient_along_v_p']
            f = fates.get(t, [None, None])[lane]
            ax[1].plot(t + (lane - 0.5) * 0.9, c, MARK[lane], ms=5, mec=CYC[lane], mew=1.1,
                       mfc=CYC[f] if f is not None else 'white', zorder=3)
    ax[1].set_yscale('symlog', linthresh=1e-5, linscale=0.6)
    ax[1].set_xticks(range(0, 43, 6)); ax[1].set_xlim(-2, 44)
    ax[1].set_xlabel('update (label 0)'); ax[1].set_ylabel(r'position $s$ on the slow line')
    ax[1].set_title('(c) the slow line at each return', loc='left')
    ax[1].text(43.5, -3e-6, r'$C_p$’s point: to cycle 1', ha='right', va='top', fontsize=6.3,
               color=CYC[1])

    # (c) landing by start phase
    ax[2].axvspan(-0.5, 0.5, color=GOLD, alpha=0.5, lw=0)
    for k in range(6):
        hold = M['groups'][f'check-{k:02d}']['at_hold']
        for lane in (0, 1):
            h = hold[lane]
            ax[2].semilogy(k + (lane - 0.5) * 0.25, h['closest_saddle_distance'], MARK[lane], ms=5, mec=CYC[lane],
                           mew=1.1, mfc=CYC[h['final_cycle']])
    ax[2].set_xticks(range(6)); ax[2].set_xlim(-0.6, 5.6)
    ax[2].set_xlabel('start phase'); ax[2].set_ylabel('closest approach to\nunstable orbit at $H$')
    ax[2].set_title('(d) by start phase', loc='left')

    # (b) the period map along the slow line at label 0
    m = ax[3]
    u = 1e3                                     # plot in units of 1e-3
    on = [q for q in F['F3']['pairs'] if q['on_curve']]
    lo, hi = T['verdicts']['boundary_bracket_on_the_slow_line']
    m.axhspan(lo * u, hi * u, color='0.88', lw=0)
    m.text(-2.9, (lo + hi) / 2 * u, 'outcome bracket at $H$', ha='left', va='center', fontsize=6.3, color='0.3')
    m.plot([-8, 4], [-8, 4], color='0.6', lw=0.6, ls='--')
    m.axhline(0, color='0.3', lw=0.6); m.axvline(0, color='0.3', lw=0.6)
    old = [q for q in on if q['start_phase'] == 0]
    new = [q for q in on if q['start_phase'] != 0]
    m.plot([q['s'] * u for q in old], [q['image'] * u for q in old], '.', color='0.25', ms=3.5, zorder=2,
           label='saved states on the line')
    m.plot([q['s'] * u for q in new], [q['image'] * u for q in new], 'D', mfc='white', mec='0.25', ms=3.5, mew=0.9,
           zorder=2, label='same, from phases 1 to 3')
    first = {x['lane']: x for x in F['F5'] if x['start_phase'] == 0}
    for lane in (0, 1):
        x0, y0 = first[lane]['s'] * u, first[lane]['image'] * u
        q = next(q for q in old if q['lane'] == lane and q['update'] == 6)
        x1, y1 = q['s'] * u, q['image'] * u
        m.plot(x0, y0, MARK[lane], mfc='white', mec=CYC[lane], mew=1.1, ms=5, zorder=4)
        m.plot(x1, y1, MARK[lane], mfc=CYC[lane], mec=CYC[lane], ms=5, zorder=4)
        m.annotate('', xy=(x1, x1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-', color=CYC[lane], lw=0.7, ls=':'))
        m.annotate('', xy=(x1, y1), xytext=(x1, x1), arrowprops=dict(arrowstyle='->', color=CYC[lane], lw=0.9))
    m.text(-6.9, -2.6, 'far side lands on\nthe near side:\norder reversed', fontsize=6.3, color='0.2', va='top')
    m.text(-6.9, -5.3, 'open: start, off the line', fontsize=6.3, color='0.2', va='top')
    m.set_xlim(-8, 4); m.set_ylim(-8, 3)
    m.set_xlabel(r'$s$ at label 0 ($10^{-3}$)'); m.set_ylabel(r'$s$ one period later ($10^{-3}$)', labelpad=1)
    m.set_title('(b) period map on the slow line', loc='left')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output)
    print('figure', a.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
