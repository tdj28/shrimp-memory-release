#!/usr/bin/env python3
"""Figure for the public toggle sweep: every pulse target on the real Qwen kite map, colored by its mapping.

Background: the earlier study's fixed-start scans (hr10 under hr11) with the computed fold curve, the two flip curves
and the located cusp, read from the earlier study's pod64 data folder (--map-root, read only). Points: the audited
sweep report (tools/audit_public_toggle_sweep.py). One panel per dwell, drawn at 7.0 in wide for the paper. No model
call.

Usage (the paper's figure; ARCHIVE is this study's data archive, PRIOR the earlier study's data folder):
  python tools/plot_public_toggle_sweep.py --host pod --map-root PRIOR/explore/llm-soft-loop/pod64 \
    --report ARCHIVE/audits/public-toggle-sweep/20261109T230942Z/audit-pod-and-local.json \
    --output public-toggle-sweep.pdf
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = dict(identity='#9aa5b1', reset0='#2b8cbe', reset1='#d6604d', toggle='#ffd23f', unresolved='#111111')
MARKERS = dict(identity='o', reset0='v', reset1='^', toggle='*', unresolved='x')
NAMES = dict(identity='hold', reset0='reset to 0', reset1='reset to 1', toggle='exchange', unresolved='unresolved')
SIZES = dict(identity=12, reset0=15, reset1=15, toggle=40, unresolved=15)        # marker areas (points^2)
HOLD_GREEN = '#1a9850'
FLIP_RED = '#b2182b'


def background(root, scan):
    d = np.load(root/scan/'scan.npz', allow_pickle=True)
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--host', required=True, help='host whose complete sweep is drawn')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--map-root', type=Path, required=True,
                   help="the earlier study's pod64 folder: scans hr10 and hr11, kite/ and p6s/ curves")
    a = p.parse_args(argv)
    root = a.map_root
    rep = json.loads(a.report.read_text())
    groups = rep['hosts'][a.host]['groups']
    folds = np.vstack([curve(root/'kite/kite-la-curve-plus/curve.json')[::-1],
                       curve(root/'kite/kite-la-curve-minus/curve.json')])
    flips = [np.vstack([curve(root/f'p6s/p6s-{k}-curve-plus/curve.json')[::-1],
                        curve(root/f'p6s/p6s-{k}-curve-minus/curve.json')]) for k in ('ad', 'td')]
    located = [json.loads(f.read_text())['final'] for f in sorted((root/'kite').glob('locate-*/locate.json'))
               if json.loads(f.read_text()).get('located')]
    base = [background(root, s) for s in ('hr10', 'hr11')]
    hold = groups[0]['controls']; e0 = groups[1]['controls']; e1 = groups[2]['controls']
    probes = [g for g in groups if g['group_id'] >= 3]
    dwells = sorted({g['dwell'] for g in probes})
    al = [g['controls']['alpha'] for g in probes]; lt = [g['controls']['log10_temperature'] for g in probes]
    pad_a, pad_t = 0.08*(max(al) - min(al)), 0.08*(max(lt) - min(lt))
    xlim, ylim = (min(al) - pad_a, max(al) + pad_a), (min(lt) - pad_t, max(lt) + pad_t)
    # the located cusp of this window (the other located points lie far outside it; sorted order alone put one of
    # those first, so the earlier version of this figure listed a cusp it did not show)
    cusp = [c for c in located if xlim[0] <= c['alpha'] <= xlim[1] and ylim[0] <= c['log10_temperature'] <= ylim[1]]

    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'legend.fontsize': 6.5, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
                         'ytick.major.size': 2.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'pdf.fonttype': 42})
    W, Hgt = 7.0, 2.6                                   # inches
    left, right, gap, top, bottom = 0.70, 0.06, 0.13, 0.40, 0.66
    n = len(dwells)
    w, h = (W - left - right - gap*(n - 1))/n, Hgt - top - bottom
    fig = plt.figure(figsize=(W, Hgt))
    axes = []
    for k in range(n):
        axes.append(fig.add_axes([(left + k*(w + gap))/W, bottom/Hgt, w/W, h/Hgt],
                                 sharey=axes[0] if axes else None))
    halo = [pe.withStroke(linewidth=1.6, foreground='white')]
    for k, (ax, dwell) in enumerate(zip(axes, dwells)):
        for img, x, y in base:
            ax.imshow(img, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], aspect='auto', interpolation='nearest')
        ax.plot(folds[:, 0], folds[:, 1], color='k', lw=0.9, label='fold')
        for j, f in enumerate(flips):
            ax.plot(f[:, 0], f[:, 1], color=FLIP_RED, lw=0.8, ls=(0, (4, 2)), label='flip' if j == 0 else None)
        for c in cusp:
            ax.plot(c['alpha'], c['log10_temperature'], 'P', color='k', mec='white', mew=0.4, ms=5.5, zorder=7,
                    label='cusp')
        for name, pt, mk in (('H', hold, 's'), ('E0', e0, 'D'), ('E1', e1, 'D')):
            ax.plot(pt['alpha'], pt['log10_temperature'], mk, color=HOLD_GREEN if name == 'H' else 'k',
                    ms=3.6 if name == 'H' else 5.6, mew=0.8, mfc='none' if name != 'H' else HOLD_GREEN)
        for m in COLORS:
            sel = [g for g in probes if g['dwell'] == dwell and g['mapping'] == m]
            if sel:
                ax.scatter([g['controls']['alpha'] for g in sel], [g['controls']['log10_temperature'] for g in sel],
                           s=SIZES[m], c=COLORS[m], marker=MARKERS[m], edgecolors='k', linewidths=0.35, zorder=6,
                           label=f'{NAMES[m]} ({len(sel)})')
        # H itself is named in the legend: the twelve pulses 0.5 units from it leave no room for a label at this size
        for name, pt, off in (('$E_0$', e0, (3.0, 2.0)), ('$E_1$', e1, (3.0, 2.0))):
            ax.annotate(name, (pt['alpha'], pt['log10_temperature']), xytext=off, textcoords='offset points',
                        fontsize=7, ha='left', va='bottom', zorder=9, path_effects=halo)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_title(f'({"abc"[k]}) {dwell}-update pulse', loc='left', pad=3)
        ax.set_xlabel(r'steering strength $\alpha$', labelpad=2); ax.ticklabel_format(useOffset=False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1e-4))
        if k:
            ax.tick_params(labelleft=False)
        hs, ls = ax.get_legend_handles_labels()
        keep = [i for i, s in enumerate(ls) if s.split(' (')[0] in NAMES.values()]
        ax.legend([hs[i] for i in keep], [ls[i] for i in keep], ncol=2, loc='upper center', frameon=False,
                  bbox_to_anchor=((left + k*(w + gap) + w/2), 0.30), bbox_transform=fig.dpi_scale_trans,
                  handletextpad=0.3, columnspacing=1.0, borderaxespad=0, handlelength=1.2, labelspacing=0.35)
    axes[0].set_ylabel(r'$\log_{10}T$', labelpad=2)
    shared = [Line2D([], [], color='k', lw=0.9, label='fold'),
              Line2D([], [], color=FLIP_RED, lw=0.8, ls=(0, (4, 2)), label='flip')]
    if cusp:
        shared.append(Line2D([], [], ls='none', marker='P', color='k', mec='white', mew=0.4, ms=5.5, label='cusp'))
    shared += [Line2D([], [], ls='none', marker='s', color=HOLD_GREEN, ms=3.6, label='hold setting $H$'),
               Line2D([], [], ls='none', marker='D', mfc='none', mec='k', mew=0.8, ms=5.6,
                      label='write endpoints $E_0$, $E_1$')]
    fig.legend(handles=shared, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(shared), frameon=False,
               handlelength=2.0, columnspacing=1.6, handletextpad=0.5, borderaxespad=0.2)
    fig.savefig(a.output, dpi=600, metadata={'CreationDate': None} if a.output.suffix == '.pdf' else None)
    print('wrote', a.output, '| counts per dwell:',
          {d: {NAMES[m]: sum(g['dwell'] == d and g['mapping'] == m for g in probes) for m in COLORS} for d in dwells},
          '| cusp drawn at', [(c['alpha'], c['log10_temperature']) for c in cusp])


if __name__ == '__main__':
    main()
