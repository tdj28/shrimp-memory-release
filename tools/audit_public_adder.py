#!/usr/bin/env python3
"""Independent check of a returned public adder run (tools/run_public_adder.py), and its figure. Zero model calls.

Revised (schema v2) to carry the Task 8 validator repair here, as Codex asked in Task 9
(docs/audits/2026-11-13-repository-scan/review.md). Everything scored against is literal, nothing is taken from the
producer: 11 + 6 in four bits, least significant first; the twelve moves (T, W, T / T, T, E1 / W, T, T / T, W, T),
their bits and roles; the 24 stages in order with their ids, names, move indices, controls, IEEE encodings and
lengths; the expected register after every move for carry in 0 and 1 (a full adder evaluated here); the results 17
and 18; the canonical reference cycles (by SHA-256); the ordered call ledger. Per run: manifest; completed status;
native64/one-thread readiness; every stage record complete with its cache gates; every saved array checked on its own
against its binding; contiguous carried states from the literal phase-0 start; the twelve readouts recomputed with
audit_public_toggle_sweep.readout and compared with the saved ones and the expectation. A malformed run is reported
as failed with its problems; the receipt is always written. Optionally draws both lanes' distances to the two
reference cycles over the whole run (--figure) and saves them (--data-output).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_public_toggle_sweep as A  # noqa: E402  (independent readout)
import task8_audit_common as C  # noqa: E402  (strict reader, records, ledger)

PROFILE = 'public-adder'
X, Y, BITS = 11, 6, 4
MOVES = ('T', 'W', 'T', 'T', 'T', 'E1', 'W', 'T', 'T', 'T', 'W', 'T')
ROLES = ('carry xor x', 'sum bit', 'next carry')
NUMBERS = [17, 18]
REFERENCES_SHA256 = 'd96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0'


def expected():
    """Register after every move, lanes with carry in 0 and 1: c xor x_i, then the sum bit, then the next carry."""
    out, carry = [], [0, 1]
    for i in range(BITS):
        xi, yi = (X >> i) & 1, (Y >> i) & 1
        out += [[c ^ xi for c in carry], [c ^ xi ^ yi for c in carry]]
        carry = [int(c + xi + yi >= 2) for c in carry]
        out.append(list(carry))
    return out


def decode(per_move):
    """Sum bits least significant first, then the final carry, for each lane."""
    return [sum(per_move[3*i + 1][lane] << i for i in range(BITS)) + (per_move[-1][lane] << BITS) for lane in (0, 1)]


def literal_stages():
    out = []
    for k, m in enumerate(MOVES):
        ctrl, steps = C.MOVES[m]
        for name, c, n in ((f'm{k:02d}-{m}', ctrl, steps), (f'm{k:02d}-hold', C.H, C.HOLD_STEPS)):
            out.append(dict(stage_id=len(out), name=name, controls=c, steps=n, move_index=k, move=m, bit=k // 3,
                            role=ROLES[k % 3]))
    return out


def check(directory):
    d, problems = Path(directory), []
    C.check_manifest(d, problems, d.name)
    r = C.load_json(d/'result.json', problems, d.name)
    if not isinstance(r, dict):
        return dict(directory=str(d), passed=False, problems=problems + [f'{d.name}: no result record'], readouts=[],
                    numbers=None), None, [], None
    saved_moves = r.get('moves') if isinstance(r.get('moves'), list) else []
    if r.get('profile') != PROFILE or r.get('status') != 'completed' or not C.same(r.get('x'), X) \
            or not C.same(r.get('y'), Y) or [(m.get('move'), m.get('bit'), m.get('role')) if isinstance(m, dict)
                                              else None for m in saved_moves] != \
            [(m, k // 3, ROLES[k % 3]) for k, m in enumerate(MOVES)]:
        problems.append('profile, status, digits or moves differ from the fixed design')
    C.check_environment(d, None, None, problems, d.name)
    try:
        if C.sha(d/'reference-cycles.npz') != REFERENCES_SHA256:
            problems.append('reference cycles are not the canonical file')
        refs = np.load(d/'reference-cycles.npz')['states']
    except Exception as exc:
        return dict(directory=str(d), passed=False, readouts=[], numbers=None,
                    problems=problems + [f'reference cycles unreadable ({type(exc).__name__})']), None, [], None
    plan, want = literal_stages(), expected()
    stages = r.get('stages') if isinstance(r.get('stages'), list) else []
    if len(stages) != len(plan):
        problems.append(f'{len(stages)} stage records, expected the {len(plan)} literal stages')
    saved = r.get('readouts') if isinstance(r.get('readouts'), list) else []
    if [q.get('move_index') if isinstance(q, dict) else None for q in saved] != list(range(len(MOVES))):
        problems.append('saved readouts are not moves 0..11 in order')
    prev, traj, out, spans, t = refs[:, 0], [refs[:, 0][None]], [], [], 0
    for i, lit in enumerate(plan[:len(stages)]):
        st, w = stages[i], f'stage {i}'
        ident = dict(stage_id=i, group_id=0, group='adder', local_stage_id=i, name=lit['name'],
                     move_index=lit['move_index'], move=lit['move'], bit=lit['bit'], role=lit['role'], profile=PROFILE)
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
        traj.append(x_[1:])
        spans.append((t, t + lit['steps'], lit['name']))
        t += lit['steps']
        if lit['name'].endswith('-hold'):
            k = lit['move_index']
            bits = [A.readout(x_[1:, lane], refs)['bit'] for lane in (0, 1)]
            sb = saved[k].get('bits') if k < len(saved) and isinstance(saved[k], dict) else None
            if not C.same(sb, bits):
                problems.append(f'move {k}: recomputed {bits}, saved {sb}')
            out.append(dict(move_index=k, move=lit['move'], bit=lit['bit'], role=lit['role'], bits=bits,
                            expected=want[k], match=bits == want[k]))
    ledger = [dict(stage_id=s['stage_id'], stage=s['name'], group_id=0, group='adder', steps=s['steps']) for s in plan]
    n = C.check_call_ledger(r.get('calls'), ledger, PROFILE, problems, 'calls', layout='pair')
    if not C.same(r.get('model_calls'), n):
        problems.append(f'model_calls {r.get("model_calls")} differs from the ledger ({n})')
    complete = len(out) == len(MOVES) and all(o['bits'] is not None and None not in o['bits'] for o in out)
    numbers = decode([o['bits'] for o in out]) if complete else None
    rep = dict(directory=str(d), passed=not problems and complete and all(o['match'] for o in out)
               and numbers == NUMBERS, problems=problems, readouts=out, numbers=numbers,
               model_calls=r.get('model_calls'), elapsed_seconds=r.get('elapsed_seconds'),
               peak_rss_bytes=r.get('peak_rss_bytes'), source_commit=(r.get('source') or {}).get('source_commit'))
    whole = np.concatenate(traj) if len(traj) == len(plan) + 1 else None
    return rep, whole, spans, refs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figure', type=Path, action='append', default=[], help='figure(s) of the first run (png/pdf)')
    p.add_argument('--data-output', type=Path)
    a = p.parse_args(argv)
    reports, first = [], None
    for k, run in enumerate(a.run):
        try:
            rep, traj, spans, refs = check(run)
        except Exception as exc:                          # never abort the receipt
            rep, traj, spans, refs = dict(directory=str(run), passed=False, readouts=[], numbers=None,
                                          problems=[f'audit exception {type(exc).__name__}: {exc}']), None, [], None
        reports.append(rep)
        if k == 0:
            first = (rep, traj, spans, refs)
    a.output.write_text(json.dumps(dict(schema='public-adder-audit-v2', model_calls=0, runs=reports,
                                        passed=all(x['passed'] for x in reports)), indent=1))
    for x in reports:
        print(x['directory'], 'passed' if x['passed'] else 'FAILED', x['problems'][:5], 'numbers', x['numbers'],
              sum(o['match'] for o in x['readouts']), '/ 12 moves matched')
    if (a.figure or a.data_output) and first[1] is not None:
        rep, traj, spans, refs = first
        dist = np.stack([[np.min(np.linalg.norm(refs[c][None] - traj[:, lane, None], axis=2), axis=1) for c in (0, 1)]
                         for lane in (0, 1)])
        if a.data_output:
            np.savez_compressed(a.data_output, distance=dist, spans=np.array([(s, e) for s, e, _ in spans]),
                                names=np.array([n for _, _, n in spans]))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(14, 6.6), sharex=True, constrained_layout=True)
        colors = {'T': '#ffd23f', 'W': '#dddddd', 'E0': '#9ecae1', 'E1': '#fcbba1'}
        for lane, ax in enumerate(axes):
            for s, e, name in spans:
                if not name.endswith('-hold'):
                    ax.axvspan(s, e, color=colors[name.split('-')[1]], alpha=0.9, lw=0)
            for c, col in ((0, '#2b8cbe'), (1, '#d6604d')):
                ax.semilogy(np.arange(dist.shape[2]), np.maximum(dist[lane, c], 1e-16), color=col, lw=1.0,
                            label=f'distance to cycle {c}')
            for o in rep['readouts']:
                end = spans[2*o['move_index'] + 1][1]
                tag = {'carry xor x': 'c⊕x', 'sum bit': 'sum', 'next carry': 'carry'}[o['role']]
                ax.annotate(f'{tag} {o["bits"][lane]}', (end - 120, 2.5e-17), ha='center', fontsize=8,
                            color='k' if o['match'] else 'red', fontweight='bold' if o['role'] == 'sum bit' else 'normal')
            ax.set_ylim(1e-17, 0.5); ax.set_ylabel(f'carry in {lane}'); ax.legend(loc='upper right', fontsize=8)
        axes[-1].set_xlabel('loop update')
        fig.suptitle(f'Adding {X} + {Y} in one loop (bits least significant first). Yellow = toggle push, grey = wait, '
                     f'blue/red = write 0/1. Result: {rep["numbers"][0]} (carry in 0), {rep["numbers"][1]} (carry in 1)',
                     fontsize=11)
        for f in a.figure:
            fig.savefig(f, dpi=180)
    return 0 if all(x['passed'] for x in reports) else 1


if __name__ == '__main__':
    raise SystemExit(main())
