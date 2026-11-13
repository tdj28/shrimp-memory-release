#!/usr/bin/env python3
"""Figure for the public XOR chain: both lanes' distance to each reference cycle over all 16 carried stages.

Distance at each update = smallest L2 distance from the state to any of the six states of reference cycle 0 or 1.
Bands mark the 12-update stages (toggle pulse when B = 1, idle at H when B = 0); each 240-update hold follows. Labels
give the incoming bit and the audited readout after every hold. Data: a returned run of tools/run_public_xor_chain.py.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_public_xor_chain as X


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
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
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True, constrained_layout=True)
    colors = ('#2b8cbe', '#d6604d')
    for lane, ax in enumerate(axes):
        for s, e, name in spans:
            if not name.endswith('-hold'):
                ax.axvspan(s, e, color='#ffd23f' if name.endswith('toggle') else '#dddddd', alpha=0.9, lw=0)
        for c in (0, 1):
            ax.semilogy(np.arange(dist.shape[2]), np.maximum(dist[lane, c], 1e-16), color=colors[c], lw=1.0,
                        label=f'distance to cycle {c}')
        for item in r['readouts']:
            k = item['chain_index']; end = spans[2*k + 1][1]
            ax.annotate(f'B={item["incoming_bit"]} → {item["bits"][lane]}', (end - 126, 2.5e-17), ha='center',
                        fontsize=9, color='k' if item['match'] else 'red', fontweight='bold')
        ax.set_ylim(1e-17, 0.5)
        ax.set_ylabel(f'lane started at A={lane}')
        ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('loop update')
    toggle = r['toggle']
    fig.suptitle(f'XOR in one loop: incoming bits {"".join(map(str, r["b_sequence"]))}; yellow = toggle pulse '
                 f'(theta {toggle["theta"]} deg, radius {toggle["radius"]}, {toggle["dwell"]} updates), grey = idle at H; '
                 f'{r["matched"]}/8 stages matched A XOR B', fontsize=11)
    fig.savefig(a.output, dpi=180)
    print('wrote', a.output)


if __name__ == '__main__':
    main()
