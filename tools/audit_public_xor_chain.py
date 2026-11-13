#!/usr/bin/env python3
"""Independent check of a returned public XOR chain (tools/run_public_xor_chain.py). Zero model calls.

Revised (schema v2) to carry the Task 8 validator repair here, as Codex asked in Task 9
(docs/audits/2026-11-13-repository-scan/review.md). Everything scored against is literal, nothing is taken from the
producer: the B sequence 1, 0, 1, 1, 0, 0, 1, 0; the toggle (30 degrees, radius 3.5, 12 updates) and its controls;
the 16 stages in order with their ids, names, chain indices, incoming bits, controls, IEEE encodings and lengths;
the expected register after each step (lanes start on cycles 0 and 1, and a 1 flips both); the canonical reference
cycles (by SHA-256); the ordered call ledger. Per run: manifest; completed status; native64/one-thread readiness;
every stage record complete with its cache gates; every saved array checked on its own against its binding (initial
state and every block); contiguous carried states from the literal phase-0 start; the eight readouts recomputed with
audit_public_toggle_sweep.readout (its own implementation of the published rules) and compared with the saved ones
and the expectation. A malformed run is reported as failed with its problems; the receipt is always written.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_public_toggle_sweep as A  # noqa: E402  (independent readout)
import task8_audit_common as C  # noqa: E402  (strict reader, records, ledger)

PROFILE = 'public-xor-chain'
B_SEQUENCE = (1, 0, 1, 1, 0, 0, 1, 0)
TOGGLE = {'theta': 30, 'radius': 3.5, 'dwell': 12}
REFERENCES_SHA256 = 'd96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0'


def expected():
    bits, out = [0, 1], []
    for b in B_SEQUENCE:
        bits = [x ^ b for x in bits]
        out.append(list(bits))
    return out


def literal_stages():
    out = []
    for k, b in enumerate(B_SEQUENCE):
        out.append(dict(stage_id=2*k, name=f'b{k}-{"toggle" if b else "idle"}', controls=C.T if b else C.H,
                        steps=TOGGLE['dwell'], chain_index=k, incoming_bit=b))
        out.append(dict(stage_id=2*k + 1, name=f'b{k}-hold', controls=C.H, steps=C.HOLD_STEPS, chain_index=k,
                        incoming_bit=b))
    return out


def check(directory):
    d, problems, out = Path(directory), [], []
    C.check_manifest(d, problems, d.name)
    r = C.load_json(d/'result.json', problems, d.name)
    if not isinstance(r, dict):
        return dict(directory=str(d), passed=False, problems=problems + [f'{d.name}: no result record'], readouts=[])
    spec = r.get('toggle') if isinstance(r.get('toggle'), dict) else {}
    if r.get('profile') != PROFILE or r.get('status') != 'completed' or not C.same(r.get('b_sequence'), list(B_SEQUENCE)) \
            or any(not C.same(spec.get(k), v) for k, v in TOGGLE.items()) or not C.same(r.get('toggle_controls'), C.T):
        problems.append('profile, status, B sequence or toggle differs from the fixed design')
    C.check_environment(d, None, None, problems, d.name)
    try:
        if C.sha(d/'reference-cycles.npz') != REFERENCES_SHA256:
            problems.append('reference cycles are not the canonical file')
        refs = np.load(d/'reference-cycles.npz')['states']
    except Exception as exc:
        return dict(directory=str(d), passed=False, readouts=[],
                    problems=problems + [f'reference cycles unreadable ({type(exc).__name__})'])
    plan, want = literal_stages(), expected()
    stages = r.get('stages') if isinstance(r.get('stages'), list) else []
    if len(stages) != len(plan):
        problems.append(f'{len(stages)} stage records, expected the {len(plan)} literal stages')
    saved = r.get('readouts') if isinstance(r.get('readouts'), list) else []
    if [q.get('chain_index') if isinstance(q, dict) else None for q in saved] != list(range(len(B_SEQUENCE))):
        problems.append('saved readouts are not chain steps 0..7 in order')
    prev = refs[:, 0]
    for i, lit in enumerate(plan[:len(stages)]):
        st, w = stages[i], f'stage {i}'
        ident = dict(stage_id=i, group_id=0, group='xor-chain', local_stage_id=i, name=lit['name'],
                     chain_index=lit['chain_index'], incoming_bit=lit['incoming_bit'], profile=PROFILE)
        if not isinstance(st, dict) or any(not C.same(st.get(k), v) for k, v in ident.items()):
            problems.append(f'{w}: identity differs from the literal stage {lit["name"]}')
            prev = None
            continue
        C.check_stage_record(st, lit['steps'], problems, w)
        C.check_controls(st, lit['controls'], problems, w)
        x_ = C.reconstruct(d, st, lit['controls'], lit['steps'], problems, w)
        if x_ is None:
            prev = None
            continue
        if prev is None or x_[0].tobytes() != prev.tobytes():
            problems.append(f'{w}: does not start from the previous terminal state')
        prev = x_[-1]
        if lit['name'].endswith('-hold'):
            k = lit['chain_index']
            bits = [A.readout(x_[1:, lane], refs)['bit'] for lane in (0, 1)]
            sb = saved[k].get('bits') if k < len(saved) and isinstance(saved[k], dict) else None
            if not C.same(sb, bits):
                problems.append(f'chain {k}: recomputed {bits}, saved {sb}')
            out.append(dict(chain_index=k, incoming_bit=lit['incoming_bit'], bits=bits, expected=want[k],
                            match=bits == want[k]))
    ledger = [dict(stage_id=s['stage_id'], stage=s['name'], group_id=0, group='xor-chain', steps=s['steps'])
              for s in plan]
    n = C.check_call_ledger(r.get('calls'), ledger, PROFILE, problems, 'calls', layout='pair')
    if not C.same(r.get('model_calls'), n):
        problems.append(f'model_calls {r.get("model_calls")} differs from the ledger ({n})')
    return dict(directory=str(d), passed=not problems and len(out) == 8 and all(o['match'] for o in out),
                problems=problems, readouts=out, model_calls=r.get('model_calls'),
                elapsed_seconds=r.get('elapsed_seconds'), peak_rss_bytes=r.get('peak_rss_bytes'),
                source_commit=(r.get('source') or {}).get('source_commit'), toggle=spec,
                toggle_controls=r.get('toggle_controls'))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    reports = []
    for x in a.run:
        try:
            reports.append(check(x))
        except Exception as exc:                          # never abort the receipt
            reports.append(dict(directory=str(x), passed=False, readouts=[],
                                problems=[f'audit exception {type(exc).__name__}: {exc}']))
    a.output.write_text(json.dumps(dict(schema='public-xor-chain-audit-v2', model_calls=0, runs=reports,
                                        passed=all(x['passed'] for x in reports)), indent=1))
    for x in reports:
        print(x['directory'], 'passed' if x['passed'] else 'FAILED', x['problems'][:5],
              ''.join(str(o['bits']) for o in x['readouts']))
    return 0 if all(x['passed'] for x in reports) else 1


if __name__ == '__main__':
    raise SystemExit(main())
