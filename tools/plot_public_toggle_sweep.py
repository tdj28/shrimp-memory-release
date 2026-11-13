#!/usr/bin/env python3
"""Figure for the public toggle sweep: every pulse target on the real Qwen kite map, coloured by its mapping.

Background: the earlier study's fixed-start scans (hr10 under hr11) with the computed fold curve, the two flip curves
and the located cusp, read from /data2/neural-shrimp-data/explore/llm-soft-loop/pod64 (read only). Points: the audited
sweep report (tools/audit_public_toggle_sweep.py). One panel per dwell. No model call.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

POD64 = Path('/data2/neural-shrimp-data/explore/llm-soft-loop/pod64')
COLORS = dict(identity='#9aa5b1', reset0='#2b8cbe', reset1='#d6604d', toggle='#ffd23f', unresolved='#111111')
MARKERS = dict(identity='o', reset0='v', reset1='^', toggle='*', unresolved='x')


def background(scan):
    d = np.load(POD64/scan/'scan.npz', allow_pickle=True)
    per, ly = d['period'], d['lyapunov']
    img = np.ones(per.shape + (3,))*0.93
    img[(per >= 1) & (ly < 0)] = (0.80, 0.84, 0.88)
    img[(per == 6) & (ly < 0)] = (0.62, 0.86, 0.92)
    img[(per <= 0) & (ly > 1e-3)] = (0.93, 0.78, 0.90)
    return img, d['alpha'], d['log10_temperature']


def curve(path):
    pts = [q for q in json.loads(path.read_text())['points'] if not q.get('rejected')]
    return np.array([[q['alpha'], q['log10_temperature']] for q in pts])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--host', required=True, help='host whose complete sweep is drawn')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    rep = json.loads(a.report.read_text())
    groups = rep['hosts'][a.host]['groups']
    folds = np.vstack([curve(POD64/'kite/kite-la-curve-plus/curve.json')[::-1],
                       curve(POD64/'kite/kite-la-curve-minus/curve.json')])
    flips = [np.vstack([curve(POD64/f'p6s/p6s-{k}-curve-plus/curve.json')[::-1],
                        curve(POD64/f'p6s/p6s-{k}-curve-minus/curve.json')]) for k in ('ad', 'td')]
    cusp = [json.loads(f.read_text())['final'] for f in sorted((POD64/'kite').glob('locate-*/locate.json'))
            if json.loads(f.read_text()).get('located')]
    base = [background(s) for s in ('hr10', 'hr11')]
    hold = groups[0]['controls']; e0 = groups[1]['controls']; e1 = groups[2]['controls']
    probes = [g for g in groups if g['group_id'] >= 3]
    dwells = sorted({g['dwell'] for g in probes})
    al = [g['controls']['alpha'] for g in probes]; lt = [g['controls']['log10_temperature'] for g in probes]
    pad_a, pad_t = 0.08*(max(al) - min(al)), 0.08*(max(lt) - min(lt))
    fig, axes = plt.subplots(1, len(dwells), figsize=(6.2*len(dwells), 5.6), sharey=True, constrained_layout=True)
    for ax, dwell in zip(np.atleast_1d(axes), dwells):
        for img, x, y in base:
            ax.imshow(img, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], aspect='auto', interpolation='nearest')
        ax.plot(folds[:, 0], folds[:, 1], color='k', lw=1.4, label='fold')
        for k, f in enumerate(flips):
            ax.plot(f[:, 0], f[:, 1], color='#b2182b', lw=1.1, ls='--', label='flip' if k == 0 else None)
        for c in cusp[:1]:
            ax.plot(c['alpha'], c['log10_temperature'], 'P', color='k', ms=9, label='cusp')
        for name, pt, mk in (('H', hold, 's'), ('E0', e0, 'D'), ('E1', e1, 'D')):
            ax.plot(pt['alpha'], pt['log10_temperature'], mk, color='#1a9850' if name == 'H' else 'k', ms=8,
                    mfc='none' if name != 'H' else '#1a9850')
            ax.annotate(name, (pt['alpha'], pt['log10_temperature']), xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        for m in COLORS:
            sel = [g for g in probes if g['dwell'] == dwell and g['mapping'] == m]
            if sel:
                ax.scatter([g['controls']['alpha'] for g in sel], [g['controls']['log10_temperature'] for g in sel],
                           s=170 if m == 'toggle' else 60, c=COLORS[m], marker=MARKERS[m], edgecolors='k',
                           linewidths=0.6, zorder=6, label=f'{m} ({len(sel)})')
        ax.set_xlim(min(al) - pad_a, max(al) + pad_a); ax.set_ylim(min(lt) - pad_t, max(lt) + pad_t)
        ax.set_title(f'pulse {dwell} updates, then 240 at H')
        ax.set_xlabel('steering strength alpha'); ax.ticklabel_format(useOffset=False)
        ax.legend(fontsize=8, loc='lower right', framealpha=0.85)
    np.atleast_1d(axes)[0].set_ylabel('log10 temperature')
    fig.suptitle('Public toggle sweep: where one pulse holds, resets or exchanges the two period-6 cycles at H '
                 f'({a.host}; phase-0 starts)', fontsize=12)
    fig.savefig(a.output, dpi=180)
    print('wrote', a.output)


if __name__ == '__main__':
    main()
