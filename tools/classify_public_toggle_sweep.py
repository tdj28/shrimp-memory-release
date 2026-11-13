#!/usr/bin/env python3
"""Which region of the earlier study's fixed-start map each public toggle-sweep pulse target lies in.

Reads the audited sweep report (tools/audit_public_toggle_sweep.py) and the fixed-start scans drawn behind the sweep
figure (hr10 under hr11, /data2/neural-shrimp-data/explore/llm-soft-loop/pod64, read only), with the figure's classes:
period p >= 1 with Lyapunov < 0 is periodic (period 6 = the shrimp), period <= 0 with Lyapunov > 1e-3 is chaotic, and
anything else is undetermined. Each target takes the class of the nearest pixel of the finest scan that covers it; the
3 x 3 neighbourhood is reported so a target on a boundary is visible. No model call. The map describes where a loop
held at that setting settles from a fixed start; a pulse lasts only 12-120 updates, so the class is context, not cause.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

POD64 = Path('/data2/neural-shrimp-data/explore/llm-soft-loop/pod64')
SCANS = ('hr10', 'hr11')                      # later entries lie on top, as in plot_public_toggle_sweep.py


def label(period, lyapunov):
    if period >= 1 and lyapunov < 0:
        return 'period6' if period == 6 else 'periodic_other'
    if period <= 0 and lyapunov > 1e-3:
        return 'chaotic'
    return 'undetermined'


def load(root=POD64, scans=SCANS):
    out = []
    for s in scans:
        d = np.load(root/s/'scan.npz', allow_pickle=True)
        assert d['period'].shape == (len(d['log10_temperature']), len(d['alpha']))
        out.append((s, d['alpha'], d['log10_temperature'], d['period'], d['lyapunov']))
    return out


def classify(scans, alpha, logt):
    found = None
    for name, a, t, per, ly in scans:
        if a.min() <= alpha <= a.max() and t.min() <= logt <= t.max():
            i, j = int(np.abs(t - logt).argmin()), int(np.abs(a - alpha).argmin())
            near = [label(per[y, x], ly[y, x]) for y in range(max(i - 1, 0), min(i + 2, len(t)))
                    for x in range(max(j - 1, 0), min(j + 2, len(a)))]
            found = dict(scan=name, row=i, col=j, period=int(per[i, j]), lyapunov=float(ly[i, j]),
                         region=label(per[i, j], ly[i, j]), neighbourhood=dict(Counter(near)))
    return found or dict(scan=None, region='outside_scans')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    rep = json.loads(a.report.read_text())
    scans = load()
    rows = []
    for g in rep['hosts'][a.host]['groups']:
        if g['group_id'] < 3:
            continue
        c = g['controls']
        rows.append(dict(group_id=g['group_id'], name=g['name'], theta=g['theta'], radius=g['radius'], dwell=g['dwell'],
                         mapping=g['mapping'], alpha=c['alpha'], log10_temperature=c['log10_temperature'],
                         **classify(scans, c['alpha'], c['log10_temperature'])))
    table = Counter((r['mapping'], r['region']) for r in rows)
    out = dict(schema='public-toggle-sweep-map-classes-v1', report=str(a.report), host=a.host, scans=list(SCANS),
               probes=len(rows), by_mapping_and_region={f'{m}|{g}': n for (m, g), n in sorted(table.items())},
               toggles=[r for r in rows if r['mapping'] == 'toggle'], rows=rows)
    a.output.write_text(json.dumps(out, indent=1))
    for r in out['toggles']:
        print(f"toggle theta {r['theta']:3d} radius {r['radius']} dwell {r['dwell']:3d}: {r['region']} "
              f"(period {r['period']}, Lyapunov {r['lyapunov']:+.3f}; 3x3 {r['neighbourhood']})")
    for k, n in out['by_mapping_and_region'].items():
        print(k, n)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
