#!/usr/bin/env python3
"""Figure for the public XOR chain: both lanes' distance to each reference cycle over all 16 carried stages.

Distance at each update = smallest L2 distance from the state to any of the six states of reference cycle 0 or 1.
Bands mark the 12-update stages (toggle pulse when B = 1, idle at H when B = 0); each 240-update hold follows. Labels
give the incoming bit and the audited readout after every hold. Data: a returned run of tools/run_public_xor_chain.py.
Drawn at 7.0 in wide for the paper; no model call.

Usage (the paper's figure; ARCHIVE is this study's data archive):
  python tools/plot_public_xor_chain.py \
    --run ARCHIVE/audits/public-xor-chain/20261110T013000Z-pod-returned/run --output public-xor-chain.pdf
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_public_xor_chain as X

CYCLE = ('#2b8cbe', '#d6604d')                      # A_0 blue, A_1 red (as in public-toggle-mechanism)
TOGGLE, IDLE = '#ffd23f', '#dddddd'
FLOOR = 1e-16                                       # display floor: a state exactly on a cycle has distance 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-output', type=Path)
    a = p.parse_args(argv)
    r = json.loads((a.run/'result.json').read_text())
    refs = np.load(a.run/'reference-cycles.npz')['states']
    xs = []
    spans = []
    t = 0
    for st in r['stages']:
        s, _, _ = X.prior.reconstruct(a.run, st)
        xs.append(s if not xs else s[1:])
        spans.append((t, t + st['steps'], st['name']))
        t += st['steps']
    traj = np.concatenate(xs)                                    # (updates+1, 2, 896)
    dist = np.stack([[np.min(np.linalg.norm(refs[c][None] - traj[:, lane, None], axis=2), axis=1) for c in (0, 1)]
                     for lane in (0, 1)])                       # lane, cycle, time
    if a.data_output:
        np.savez_compressed(a.data_output, distance=dist, spans=np.array([(s, e) for s, e, _ in spans]),
                            names=np.array([n for _, _, n in spans]))

    plt.rcParams.update({'font.size': 7, 'legend.fontsize': 6.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                         'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                         'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.0), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.915, bottom=0.115, hspace=0.12)
    for lane, ax in enumerate(axes):
        for s, e, name in spans:
            if not name.endswith('-hold'):
                ax.axvspan(s, e, color=TOGGLE if name.endswith('toggle') else IDLE, alpha=0.9, lw=0, zorder=0.6)
        for c in (0, 1):
            ax.semilogy(np.arange(dist.shape[2]), np.maximum(dist[lane, c], FLOOR), color=CYCLE[c], lw=0.5,
                        zorder=2)
        for item in r['readouts']:
            k = item['chain_index']; end = spans[2*k + 1][1]
            ax.text(end - 126, 3e-18, rf'$B={item["incoming_bit"]}\rightarrow{item["bits"][lane]}$', ha='center',
                    va='baseline', fontsize=6.5, color='k' if item['match'] else 'red', zorder=3)
        ax.set_ylim(1e-18, 1.0)
        ax.set_yticks([1e-15, 1e-10, 1e-5, 1.0])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(True, which='major', axis='y', color='0.9', lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(spans[0][0] - 20, spans[-1][1] + 20)
        ax.set_ylabel(f'({"ab"[lane]}) lane started\nat $A={lane}$', labelpad=3)
    axes[-1].set_xlabel('loop update', labelpad=2)
    handles = [Line2D([], [], color=CYCLE[0], lw=1.2, label='distance to $A_0$'),
               Line2D([], [], color=CYCLE[1], lw=1.2, label='distance to $A_1$'),
               Patch(color=TOGGLE, lw=0, label='toggle pulse ($B=1$)'),
               Patch(color=IDLE, lw=0, label='12 idle updates at $H$ ($B=0$)')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.55, 1.0), ncol=4, frameon=False,
               handlelength=1.8, columnspacing=1.6, borderaxespad=0.3)
    fig.savefig(a.output, dpi=600, metadata={'CreationDate': None} if a.output.suffix == '.pdf' else None)
    print('wrote', a.output)


if __name__ == '__main__':
    main()
