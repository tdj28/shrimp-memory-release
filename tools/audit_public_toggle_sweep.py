#!/usr/bin/env python3
"""Independent check of returned public-toggle-sweep shards (docs/public-toggle-sweep.md).

For every shard directory: rehash the manifest, confirm the design and call ledger, then for every group rebuild both
lanes from the saved blocks, check the literal reference start and the pulse->hold carry, and recompute the neutral
readout with this file's own implementation of the published rules (closure 1e-8 over states 229-240, proper divisors
1-3 excluded at 1e-4, a unique best-shift reference match within 1e-6). Saved readouts must agree. Across shards:
every one of the 147 groups exactly once per host, baseline terminal states equal to the 29 October pilot, and any
group run on two hosts compared bit for bit. Writes one JSON report. No model call.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_public_toggle_sweep as sweep

CLOSURE, DIVISOR_GAP, MATCH = 1e-8, 1e-4, 1e-6
MAPPING = {(0, 1): 'identity', (1, 0): 'toggle', (0, 0): 'reset0', (1, 1): 'reset1'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readout(hold, refs):
    """hold: (240, 896) states after the 240 hold updates; refs: (2, 6, 896)."""
    last, prev = hold[-6:], hold[-12:-6]
    closure = float(np.max(np.linalg.norm(last - prev, axis=1)))
    gaps = {d: float(np.max(np.linalg.norm(last - np.roll(last, -d, axis=0), axis=1))) for d in (1, 2, 3)}
    period_ok = closure <= CLOSURE and all(g > DIVISOR_GAP for g in gaps.values())
    dist = []
    for cycle in refs:
        dist.append(min(float(np.max(np.linalg.norm(last - np.roll(cycle, -s, axis=0), axis=1))) for s in range(6)))
    matches = [k for k, d in enumerate(dist) if d <= MATCH]
    bit = matches[0] if period_ok and len(matches) == 1 else None
    return dict(closure=closure, divisor_gaps=gaps, period6=period_ok, reference_distances=dist, bit=bit)


def check_shard(directory, problems):
    d = Path(directory)
    manifest = json.loads((d/'manifest.json').read_text())
    listed = {x['path'] for x in manifest['files']}
    present = {p.name for p in d.iterdir() if p.is_file()} - {'manifest.json'}
    if listed != present:
        problems.append(f'{d.name}: manifest lists {len(listed)} files, directory has {len(present)}')
    for item in manifest['files']:
        if sha(d/item['path']) != item['sha256']:
            problems.append(f'{d.name}: hash mismatch {item["path"]}')
    result = json.loads((d/'result.json').read_text())
    design = json.loads((d/'design.json').read_text())
    groups, stages, _ = sweep.plans()
    ids = sweep.shard_group_ids(len(groups), design['shard'], design['shards'])
    if design['shard_group_ids'] != ids or [g['group_id'] for g in result['groups']] != ids:
        problems.append(f'{d.name}: shard partition differs from the fixed plan')
    if result['status'] != 'completed' or result['model_calls'] != len(sweep.shard_calls(groups, stages, ids)):
        problems.append(f'{d.name}: status {result["status"]}, calls {result["model_calls"]}')
    refs = np.load(d/'reference-cycles.npz')['states']
    out = []
    for g in result['groups']:
        gid = g['group_id']; plan = groups[gid]
        if g['control_bits'] != plan['control_bits']:
            problems.append(f'{d.name} g{gid}: controls differ from the fixed plan')
        pulse, hold = (result['stages'][s] for s in plan['stage_ids'])
        p_states, _, _ = sweep.prior.reconstruct(d, pulse)
        h_states, _, _ = sweep.prior.reconstruct(d, hold)
        if p_states[0].tobytes() != refs[:, 0].tobytes():
            problems.append(f'{d.name} g{gid}: pulse does not start at the literal references')
        if h_states[0].tobytes() != p_states[-1].tobytes():
            problems.append(f'{d.name} g{gid}: hold does not start from the pulse terminal state')
        if len(p_states) != plan['pulse_steps'] + 1 or len(h_states) != 241:
            problems.append(f'{d.name} g{gid}: wrong trajectory lengths')
        lanes = [readout(h_states[1:, lane], refs) for lane in (0, 1)]
        bits = [x['bit'] for x in lanes]
        mapping = 'unresolved' if None in bits else MAPPING[tuple(bits)]
        if bits != g['decoded_pair'] or mapping != g['mapping']:
            problems.append(f'{d.name} g{gid}: recomputed {bits}/{mapping}, saved {g["decoded_pair"]}/{g["mapping"]}')
        out.append(dict(group_id=gid, name=g['name'], theta=plan['theta_degrees'], radius=plan['radius'],
                        dwell=plan['pulse_steps'], controls=plan['controls'], bits=bits, mapping=mapping,
                        terminal_sha256=sweep.prior.array_hash(h_states[-1]),
                        lanes=[{k: v for k, v in x.items() if k != 'bit'} for x in lanes]))
    return dict(directory=str(d), shard=design['shard'], model_calls=result['model_calls'],
                elapsed_seconds=result['elapsed_seconds'], peak_rss_bytes=result['peak_rss_bytes'],
                source_commit=result['source']['source_commit'], groups=out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', action='append', nargs='+', metavar=('NAME', 'SHARD_DIR'), required=True,
                   help='host name followed by its shard directories')
    p.add_argument('--pilot-result', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    problems, hosts = [], {}
    for name, *dirs in a.host:
        shards = [check_shard(x, problems) for x in dirs]
        hosts[name] = dict(shards=shards, groups={g['group_id']: g for s in shards for g in s['groups']})
    pilot = json.loads(a.pilot_result.read_text())
    pilot_terminal = {g['name']: g['terminal_state_sha256'] for g in pilot['groups'][:3]}
    parity = {}
    for name, h in hosts.items():
        for gid in (0, 1, 2):
            if gid in h['groups']:
                g = h['groups'][gid]
                parity[f'{name}:{g["name"]}'] = dict(mapping=g['mapping'],
                                                    terminal_equals_pilot=g['terminal_sha256'] == pilot_terminal[g['name']])
    names = list(hosts)
    cross = []
    for i, x in enumerate(names):
        for y in names[i+1:]:
            for gid in sorted(set(hosts[x]['groups']) & set(hosts[y]['groups'])):
                gx, gy = hosts[x]['groups'][gid], hosts[y]['groups'][gid]
                cross.append(dict(group_id=gid, hosts=[x, y], same_mapping=gx['mapping'] == gy['mapping'],
                                  same_terminal_bits=gx['terminal_sha256'] == gy['terminal_sha256']))
    full = [n for n, h in hosts.items() if sorted(h['groups']) == list(range(147))]
    summary = {}
    for n in full:
        counts = {}
        for g in hosts[n]['groups'].values():
            if g['group_id'] >= 3:
                counts[g['mapping']] = counts.get(g['mapping'], 0) + 1
        summary[n] = counts
    report = dict(schema='public-toggle-sweep-audit-v1', problems=problems, passed=not problems,
                  baseline_parity=parity, cross_host=cross, complete_hosts=full, probe_mappings=summary,
                  toggles={n: [g['group_id'] for g in hosts[n]['groups'].values() if g['mapping'] == 'toggle']
                           for n in hosts},
                  hosts={n: dict(shards=[{k: v for k, v in s.items() if k != 'groups'} for s in h['shards']],
                                 groups=[h['groups'][k] for k in sorted(h['groups'])]) for n, h in hosts.items()})
    a.output.write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(passed=report['passed'], problems=problems[:10], baseline_parity=parity,
                          probe_mappings=summary, toggles=report['toggles'],
                          cross_host_agree=all(c['same_mapping'] for c in cross), cross_host_groups=len(cross)),
                     indent=1))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
