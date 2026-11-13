#!/usr/bin/env python3
"""Independent check of returned Stage 1b runs (tools/run_public_toggle_check.py).

Per shard: rehash the manifest; rebuild the fixed check list from the bound candidates file; per group, confirm the
controls, that the saved start equals the rule's start (reference phase k, optionally times 1 + 1e-15 u for seed s),
the pulse->hold carry, and recompute both readouts with audit_public_toggle_sweep.readout. Then summarise every
candidate against the pre-registered Stage 1b rule. No model call.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_public_toggle_sweep as A
import run_public_toggle_check as C


def check_shard(d, problems):
    d = Path(d)
    manifest = json.loads((d/'manifest.json').read_text())
    if {x['path'] for x in manifest['files']} != {p.name for p in d.iterdir() if p.is_file()} - {'manifest.json'}:
        problems.append(f'{d}: manifest file set differs')
    for item in manifest['files']:
        if A.sha(d/item['path']) != item['sha256']:
            problems.append(f'{d}: hash mismatch {item["path"]}')
    r = json.loads((d/'result.json').read_text()); design = json.loads((d/'design.json').read_text())
    groups, stages, _ = C.plans(design['candidates'])
    if r['status'] != 'completed':
        problems.append(f'{d}: status {r["status"]}')
    refs = np.load(d/'reference-cycles.npz')['states']
    out = []
    for g in r['groups']:
        plan = groups[g['group_id']]
        if g['control_bits'] != plan['control_bits'] or g['start_phase'] != plan['start_phase']:
            problems.append(f'{d} g{g["group_id"]}: controls or phase differ from the rule')
        start = np.load(d/f'group-{g["group_id"]:03d}-start.npy')
        if start.tobytes() != C.start_state(refs, plan['start_phase'], plan['perturbation_seed']).tobytes():
            problems.append(f'{d} g{g["group_id"]}: start state differs from the rule')
        pulse, hold = (r['stages'][s] for s in plan['stage_ids'])
        p, _, _ = C.prior.reconstruct(d, pulse); h, _, _ = C.prior.reconstruct(d, hold)
        if p[0].tobytes() != start.tobytes() or h[0].tobytes() != p[-1].tobytes():
            problems.append(f'{d} g{g["group_id"]}: start or carry differs')
        bits = [A.readout(h[1:, lane], refs)['bit'] for lane in (0, 1)]
        mapping = 'unresolved' if None in bits else A.MAPPING[tuple(bits)]
        if mapping != g['mapping']:
            problems.append(f'{d} g{g["group_id"]}: recomputed {mapping}, saved {g["mapping"]}')
        out.append(dict(group_id=g['group_id'], name=g['name'], candidate=g['candidate'], role=g['role'],
                        phase=g['start_phase'], seed=g['perturbation_seed'], mapping=mapping))
    return out, r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard', type=Path, nargs='+', required=True)
    p.add_argument('--rerun', type=Path, nargs='*', default=[], help='second-host runs of the same check list')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    problems, rows, cost = [], [], dict(model_calls=0, elapsed_seconds=0.0, peak_rss_bytes=0)
    for d in a.shard:
        out, r = check_shard(d, problems); rows += out
        cost['model_calls'] += r['model_calls']; cost['elapsed_seconds'] += r['elapsed_seconds']
        cost['peak_rss_bytes'] = max(cost['peak_rss_bytes'], r['peak_rss_bytes'])
    reruns = []
    for d in a.rerun:
        out, _ = check_shard(d, problems); reruns += out
    verdicts = {}
    for cand in sorted({x['candidate'] for x in rows}):
        mine = [x for x in rows if x['candidate'] == cand]
        phases = {x['phase']: x['mapping'] for x in mine if x['role'] == 'phase'}
        rounding = [x['mapping'] for x in mine if x['role'] == 'rounding']
        neighbours = [x['mapping'] for x in mine if x['role'] == 'neighbour']
        dwells = [x['mapping'] for x in mine if x['role'] == 'dwell']
        rerun = [x['mapping'] for x in reruns if x['candidate'] == cand and x['role'] == 'phase' and x['phase'] == 0]
        robust = (len(phases) == 6 and all(m == 'toggle' for m in phases.values()) and len(rounding) == 3
                  and all(m == 'toggle' for m in rounding) and rerun == ['toggle']
                  and sum(m == 'toggle' for m in neighbours) >= 4)
        verdicts[cand] = dict(phases=phases, rounding=rounding, neighbours_toggling=sum(m == 'toggle' for m in neighbours),
                              neighbours=len(neighbours), dwells=dwells, second_host_phase0=rerun,
                              robust_by_preregistered_rule=robust)
    report = dict(schema='public-toggle-check-audit-v1', passed=not problems, problems=problems, cost=cost,
                  verdicts=verdicts, rows=rows, rerun_rows=reruns)
    a.output.write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(passed=report['passed'], problems=problems[:8], cost=cost), indent=1))
    for k, v in verdicts.items():
        print(k, 'phases', ''.join({'toggle': 'T', 'identity': 'I', 'reset0': '0', 'reset1': '1'}.get(v['phases'].get(i), '?')
                                     for i in range(6)), 'rounding', v['rounding'], f"neighbours {v['neighbours_toggling']}/{v['neighbours']}",
              'dwells', v['dwells'], 'second host', v['second_host_phase0'], 'ROBUST' if v['robust_by_preregistered_rule'] else 'not robust')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
