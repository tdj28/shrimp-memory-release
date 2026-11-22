#!/usr/bin/env python3
"""Opening figure of the short paper: adding 11 + 6 in one loop, both lanes, from a saved public adder run.

The trajectories and the twelve readouts are reconstructed by the independent audit (tools/audit_public_adder.py,
check(), which also checks the run against the fixed design; a run it does not pass is refused). For each lane the
figure shows the distance of the state to the nearest of the six points of each reference cycle (A_0 blue, A_1 red)
at every update, a band for each move, the digit pair each group of three moves encodes (read from the run's own move
list and checked against its x and y), the readout after every hold (sum digits bold, final carry boxed) and the
result decoded from those readouts. A JSON receipt is written next to the figure. Zero model calls.

Usage (the paper's figure, ARCHIVE being this study's data archive; the receipt goes to the same path with .json):
  python tools/plot_public_adder.py \
    --run ARCHIVE/audits/public-adder/20261110T040000Z-pod-returned/run --output public-adder.pdf
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_public_adder as AD  # noqa: E402  (independent reconstruction and readout)

CYCLE = ('#2b8cbe', '#d6604d')                      # A_0 blue, A_1 red (as in public-toggle-mechanism)
BAND = {'T': '#ffd23f', 'W': '#dddddd', 'E1': '#fcbba1', 'E0': '#9ecae1'}
BAND_NAME = {'T': 'toggle', 'W': 'idle', 'E1': 'write of 1', 'E0': 'write of 0'}
ROLE_TAG = {'carry xor x': 'c⊕x', 'sum bit': 'sum', 'next carry': 'carry'}
FLOOR = 1e-16                                       # display floor: the start lies exactly on its cycle


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digit_pairs(r):
    """(x_i, y_i) per digit from the run's move list (toggle = 1, idle = 0), checked against the run's x and y."""
    out = []
    for i in range(len(r['moves'])//3):
        step = {m['step']: m['move'] for m in r['moves'] if m['bit'] == i}
        if step.get('x') not in ('T', 'W') or step.get('y') not in ('T', 'W'):
            raise SystemExit(f'digit {i}: moves {step} do not encode a digit pair')
        out.append((int(step['x'] == 'T'), int(step['y'] == 'T')))
    if [((r['x'] >> i) & 1, (r['y'] >> i) & 1) for i in range(len(out))] != out:
        raise SystemExit(f'moves encode {out}, the run states x={r["x"]}, y={r["y"]}')
    return out


def decoded(readouts):
    """Per lane: the final carry then the sum digits, most significant first, and its value."""
    sums = [o['bits'] for o in readouts if o['role'] == 'sum bit']
    if readouts[-1]['role'] != 'next carry':
        raise SystemExit('the last readout is not a carry')
    text = [str(readouts[-1]['bits'][lane]) + ''.join(str(s[lane]) for s in reversed(sums)) for lane in (0, 1)]
    return text, [int(t, 2) for t in text]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, nargs='+', required=True, help='figure files (pdf, png)')
    p.add_argument('--receipt', type=Path, help='default: the first output with .json')
    a = p.parse_args(argv)
    run = a.run.resolve()
    rep, traj, spans, refs = AD.check(run)
    if not rep['passed'] or traj is None:
        raise SystemExit(f'the audit does not pass {run}: {rep["problems"][:5]}')
    r = json.loads((run/'result.json').read_text())
    digits = digit_pairs(r)
    reads = rep['readouts']
    text, numbers = decoded(reads)
    if numbers != rep['numbers'] or numbers != AD.decode([o['bits'] for o in reads]):
        raise SystemExit(f'decoded {numbers}, audit {rep["numbers"]}')
    dist = np.stack([[np.min(np.linalg.norm(refs[c][None] - traj[:, lane, None], axis=2), axis=1) for c in (0, 1)]
                     for lane in (0, 1)])                       # lane, cycle, update
    moves = [n.split('-')[1] for _, _, n in spans if not n.endswith('-hold')]
    holds = [(s, e) for s, e, n in spans if n.endswith('-hold')]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory
    plt.rcParams.update({'font.size': 7, 'legend.fontsize': 6.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                         'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                         'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'pdf.fonttype': 42})
    W, Hgt = 7.0, 3.4                                    # inches
    left, right, top, digit_row, gap, bottom = 0.64, 0.90, 0.19, 0.24, 0.09, 0.37
    h = (Hgt - top - digit_row - gap - bottom)/2
    fig = plt.figure(figsize=(W, Hgt))
    ax_a = fig.add_axes([left/W, (bottom + h + gap)/Hgt, (W - left - right)/W, h/Hgt])
    ax_b = fig.add_axes([left/W, bottom/Hgt, (W - left - right)/W, h/Hgt], sharex=ax_a)
    t = np.arange(dist.shape[2])
    for lane, ax in enumerate((ax_a, ax_b)):
        for (s, e, name) in spans:
            if not name.endswith('-hold'):
                ax.axvspan(s, e, color=BAND[name.split('-')[1]], alpha=0.9, lw=0, zorder=0.6)
        for c in (0, 1):
            ax.semilogy(t, np.maximum(dist[lane, c], FLOOR), color=CYCLE[c], lw=0.5, zorder=2)
        for o, (s, e) in zip(reads, holds):
            last = o['move_index'] == len(reads) - 1
            ax.text((s + e)/2, 3e-18, f'{ROLE_TAG[o["role"]]} {o["bits"][lane]}', ha='center', va='baseline',
                    fontsize=6.5, zorder=3, color='k' if o['match'] else 'red',
                    fontweight='bold' if o['role'] == 'sum bit' or last else 'normal',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='k', lw=0.5) if last else None)
        ax.text(1.0 + 0.08/(W - left - right), 0.5, f'{text[lane]} = {numbers[lane]}', transform=ax.transAxes,
                ha='left', va='center', fontsize=8, fontweight='bold')
        ax.set_ylim(1e-18, 1.0)
        ax.set_yticks([1e-15, 1e-10, 1e-5, 1.0])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(True, which='major', axis='y', color='0.9', lw=0.5)
        ax.set_axisbelow(True)
        ax.set_ylabel(f'({"ab"[lane]}) lane started\nwith carry {lane}', labelpad=3)
    ax_a.set_xlim(0, spans[-1][1])
    ax_a.tick_params(labelbottom=False)
    ax_b.set_xlabel('loop update', labelpad=2)
    # one label per digit, centered over its three moves
    tr = blended_transform_factory(ax_a.transData, ax_a.transAxes)
    for i, (x, y) in enumerate(digits):
        s, e = spans[6*i][0], spans[6*i + 5][1]
        inset = 0.012*(spans[-1][1])
        ax_a.plot([s + inset, s + inset, e - inset, e - inset], [1.02, 1.05, 1.05, 1.02], color='0.35', lw=0.6,
                  transform=tr, clip_on=False)
        ax_a.text((s + e)/2, 1.075, f'digit {i}: $x={x}$, $y={y}$', transform=tr, ha='center', va='bottom',
                  fontsize=7)
    handles = [Line2D([], [], color=CYCLE[0], lw=1.2, label='distance to $A_0$'),
               Line2D([], [], color=CYCLE[1], lw=1.2, label='distance to $A_1$')]
    handles += [Patch(color=BAND[m], lw=0, label=BAND_NAME[m]) for m in ('T', 'W', 'E1', 'E0') if m in moves]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=((left + (W - left - right)/2)/W, 1.0),
               ncol=len(handles), frameon=False, handlelength=1.8, columnspacing=1.8, borderaxespad=0.25)
    for f in a.output:
        fig.savefig(f, dpi=600, metadata={'CreationDate': None} if f.suffix == '.pdf' else None)
    script = Path(__file__).resolve()
    receipt = a.receipt or a.output[0].with_suffix('.json')
    rec = dict(kind='public-adder-figure', model_calls=0, run=str(run),
               result_json_sha256=sha256(run/'result.json'), manifest_json_sha256=sha256(run/'manifest.json'),
               script=str(script), script_sha256=sha256(script),
               audit_script=str(Path(AD.__file__).resolve()), audit_script_sha256=sha256(AD.__file__),
               audit_passed=rep['passed'], x=r['x'], y=r['y'], digits_least_significant_first=digits,
               moves=moves, readouts=[dict(move_index=o['move_index'], role=o['role'], bits=o['bits'])
                                      for o in reads],
               decoded={f'carry_in_{lane}': dict(bits_most_significant_first=text[lane], number=numbers[lane])
                        for lane in (0, 1)},
               outputs=[dict(path=str(f.resolve()), sha256=sha256(f)) for f in a.output],
               utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip())
    receipt.write_text(json.dumps(rec, indent=1) + '\n')
    print('decoded', text, numbers, '| smallest distance drawn', float(dist[dist > 0].min()),
          '| zero distances', int((dist == 0).sum()), '| wrote', [str(f) for f in a.output], str(receipt))
    return 0


if __name__ == '__main__':
    sys.exit(main())
