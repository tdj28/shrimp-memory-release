#!/usr/bin/env python3
"""Figure: single-state paths against two-lane paths, per update (docs/public-lane-layout-check.md). Zero model calls.

For each of the four moves (12 idle updates, the writes of 0 and 1, the toggle, each followed by the 240-update hold)
and each phase-0 start, the largest coordinate gap at every update between the single-state run and the same lane of
the two-lane run in the usual arrangement (cycle 0 in lane 0, cycle 1 in lane 1). Paths are read with the strict
reader of tools/task8_audit_common.py from the frozen runs. The script also confirms that every other two-lane
arrangement of the same start and move is bit-for-bit equal to the usual one (so its gap to the single state is the
same curve), and writes a receipt binding every file it read.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task8_audit_common as C  # noqa: E402

OPS = ('W', 'E0', 'E1', 'T')
NAMES = {'W': 'idle', 'E0': 'write 0', 'E1': 'write 1', 'T': 'toggle'}
COLORS = {'W': '#2a78d6', 'E0': '#eb6834', 'E1': '#1baf7a', 'T': '#eda100'}   # categorical slots 1-4, fixed order
LAYOUTS = (('P', (0, 1)), ('R', (1, 0)), ('D0', (0, 0)), ('D1', (1, 1)))


def read(d, sids, op, lanes, bound):
    r = json.loads((d/'result.json').read_text())
    probs = []
    ctrl, steps = C.MOVES[op]
    a = C.reconstruct(d, r['stages'][sids[0]], ctrl, steps, probs, f'{d.name} move', bound, lanes=lanes)
    b = C.reconstruct(d, r['stages'][sids[1]], C.H, C.HOLD_STEPS, probs, f'{d.name} hold', bound, lanes=lanes)
    if probs:
        raise SystemExit(f'{d.name}: {probs[:3]}')
    bound += [C.binding(d/'result.json'), C.binding(d/'manifest.json')]
    return np.concatenate([a, b[1:]])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--single-run', type=Path, required=True)
    p.add_argument('--lane-run', type=Path, required=True)
    p.add_argument('--output', type=Path, nargs='+', required=True, help='figure files (pdf, png)')
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args(argv)
    bound, curves, layout_checks = [], {}, []
    for i, op in enumerate(OPS):
        pair = {}
        for j, (lay, starts) in enumerate(LAYOUTS):
            k = 4*i + j
            pair[lay] = (starts, read(a.lane_run/f'lanes-{k:02d}', [2*k, 2*k + 1], op, 2, bound))
        for c in (0, 1):
            k = 2*i + c
            single = read(a.single_run/f'single-{k:02d}', [2*k, 2*k + 1], op, 1, bound)[:, 0]
            usual = pair['P'][1][:, c]
            for lay, (starts, path) in pair.items():
                for lane, s in enumerate(starts):
                    if s == c:
                        layout_checks.append(dict(operation=op, start=c, layout=lay, lane=lane,
                                                  bitwise_equal_to_usual=path[:, lane].tobytes() == usual.tobytes()))
            curves[(op, c)] = np.abs(single - usual).max(axis=1)
    if not all(x['bitwise_equal_to_usual'] for x in layout_checks):
        raise SystemExit('a two-lane arrangement differs from the usual one')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.85), sharex=True, sharey=True, constrained_layout=True)
    for c, ax in enumerate(axes):
        for op in OPS:
            g = curves[(op, c)]
            per = g[1:].reshape(-1, C.PERIOD).max(axis=1)           # largest gap within each period of six updates
            ax.semilogy(C.PERIOD*np.arange(1, len(per) + 1), per, color=COLORS[op], lw=1.2, label=NAMES[op])
        ax.set_xlim(0, 360)
        ax.set_ylim(5e-16, 1e-11)
        ax.grid(True, which='major', axis='y', color='0.9', lw=0.5)
        ax.text(0.015, 0.93, f'from $A_{c}$', transform=ax.transAxes, ha='left', va='top', fontsize=7)
    g = curves[('E1', 0)]
    m = int(g.argmax())
    axes[0].annotate(r'$1.45\times10^{-12}$ at update %d' % m, (m, g[m]), xytext=(m + 40, g[m] * 1.3),
                     fontsize=6.3, color='0.25', va='center', arrowprops=dict(arrowstyle='-', lw=0.5, color='0.45'))
    for ax in axes:
        ax.set_xlabel('update since the start')
    axes[0].set_ylabel('largest gap per period,\none state vs. two lanes', fontsize=6.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=4, frameon=False, fontsize=6.5, handlelength=1.8,
               columnspacing=1.2)
    for f in a.output:
        fig.savefig(f, dpi=300)
    rec = dict(kind='lane-gap-figure', model_calls=0, script=C.binding(Path(__file__).resolve()),
               utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
               outputs=[C.binding(f) for f in a.output],
               cases=[dict(operation=op, start=c, updates=len(curves[(op, c)]) - 1,
                           max_gap=float(curves[(op, c)].max()), at_update=int(curves[(op, c)].argmax()),
                           at_move_end=float(curves[(op, c)][C.MOVES[op][1]]), at_hold_end=float(curves[(op, c)][-1]))
                      for op in OPS for c in (0, 1)],
               two_lane_arrangements_bitwise_equal_to_usual=len(layout_checks), bound_inputs=bound)
    a.receipt.write_text(json.dumps(rec, indent=1) + '\n')
    print('largest gap per case:', {f'{op}{c}': f'{curves[(op, c)].max():.2e}' for op in OPS for c in (0, 1)})
    print('two-lane occurrences checked bitwise equal to the usual arrangement:', len(layout_checks))
    return 0


if __name__ == '__main__':
    sys.exit(main())
