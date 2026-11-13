#!/usr/bin/env python3
"""Audit of the full-adder truth-table runs (docs/public-full-adder-truth-table.md). Zero model calls.

Revised after Codex's Task 8 review (docs/audits/2026-11-13-task8/review.md) and recheck
(docs/audits/2026-11-13-task8-recheck/review.md). The scoring oracle is literal: the job list, each job's moves,
every stage's identity, controls and length, the ordered call ledger and the Boolean truth table are written here,
not taken from the producer. Each scheduled job is checked independently; a missing or malformed job is recorded as
an operational failure and the receipt is still written.

Per job: manifest; identity (profile, digits, moves); status; native64/one-thread readiness, model-input hashes
against the canonical audited run, source commit equal to the frozen commit and a clean checkout; reference cycles
equal to the canonical file; exactly six stage records in order, each with its literal id, name, group, move index
and move, literal controls and bit encodings, exact step counts (declared, attempted, returned, completed and
reconstructed), cache and final-cache gates and no failed outputs; each saved array checked on its own against its
binding (initial state and every block); contiguous carried states from the literal phase-0 start; the literal
ordered call ledger; the three readouts, recomputed with audit_public_toggle_sweep.readout and compared with the
saved bits and the truth table.

Reported: settle update per move and lane (first hold update after which the lane stays within 1e-6 of its final
cycle, phase-aligned; censored if never), the measured cycle and phase at each move end and hold end, and, for group
(1, 0), the largest gap to the first three moves of the saved 11 + 6 run. That comparison is secondary and qualified
on its own: its manifest, records and every file it reads are checked and bound, a value is given only when all of
its own checks pass, and a failed or malformed historical run is reported as unavailable without touching the
primary results.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_public_toggle_sweep as A  # noqa: E402  (independent readout)
import task8_audit_common as C  # noqa: E402

JOBS = ((0, 0, ('W', 'W', 'E0')), (0, 1, ('W', 'T', 'T')), (1, 0, ('T', 'W', 'T')), (1, 1, ('T', 'T', 'E1')))


def truth(x, y):
    """Register after steps 1, 2 and 3 for lanes with carry in 0 and 1."""
    return [[c ^ x for c in (0, 1)], [c ^ x ^ y for c in (0, 1)], [int(c + x + y >= 2) for c in (0, 1)]]


PROFILE = 'public-full-adder'


def literal_stages(moves):
    out = []
    for k, m in enumerate(moves):
        ctrl, steps = C.MOVES[m]
        out.append((f'm{k:02d}-{m}', ctrl, steps, k, False))
        out.append((f'm{k:02d}-hold', C.H, C.HOLD_STEPS, k, True))
    return out


def check_identity(st, i, name, k, move, problems, what, profile=PROFILE):
    want = dict(stage_id=i, group_id=0, group='adder', local_stage_id=i, name=name, move_index=k, move=move,
                profile=profile)
    if not isinstance(st, dict) or any(not C.same(st.get(key), v) for key, v in want.items()):
        problems.append(f'{what}: identity differs from the literal stage {i} ({name})')


def check_job(d, x, y, moves, refs, canon_inputs, frozen):
    what = d.name
    problems, rep = [], dict(job=what, x=x, y=y, moves=list(moves))
    if not d.is_dir():
        return dict(rep, status='missing', problems=[f'{what}: missing'])
    C.check_manifest(d, problems, what)
    r = C.load_json(d/'result.json', problems, what)
    if not isinstance(r, dict):
        return dict(rep, status='operational_failure', problems=problems + [f'{what}: no result record'])
    saved_moves = r.get('moves')
    if r.get('profile') != PROFILE or not C.same(r.get('x'), x) or not C.same(r.get('y'), y) \
            or not isinstance(saved_moves, list) \
            or [m.get('move') if isinstance(m, dict) else None for m in saved_moves] != list(moves):
        problems.append(f'{what}: identity (profile, digits or moves) differs from the schedule')
    if r.get('status') != 'completed':
        problems.append(f'{what}: run status {r.get("status")}')
    rep['environment'] = C.check_environment(d, canon_inputs, frozen, problems, what)
    try:
        run_refs = np.load(d/'reference-cycles.npz')['states']
        if run_refs.tobytes() != refs.tobytes():
            problems.append(f'{what}: reference cycles differ from the canonical file')
    except Exception as exc:
        problems.append(f'{what}: reference cycles unreadable ({type(exc).__name__})')
    plan = literal_stages(moves)
    stages = r.get('stages')
    if not isinstance(stages, list) or len(stages) != len(plan):
        problems.append(f'{what}: stage list is not exactly the {len(plan)} literal stages in order')
        stages = stages if isinstance(stages, list) else []
    prev, holds, ends, path = refs[:, 0], {}, [], [refs[:, 0][None]]
    for i, (name, ctrl, steps, k, is_hold) in enumerate(plan):
        if i >= len(stages):
            break
        st, w = stages[i], f'{what} stage {i}'
        check_identity(st, i, name, k, moves[k], problems, w)
        if not isinstance(st, dict):
            prev = None
            continue
        C.check_stage_record(st, steps, problems, w)
        C.check_controls(st, ctrl, problems, w)
        x_ = C.reconstruct(d, st, ctrl, steps, problems, w)
        if x_ is None:
            prev = None
            continue
        if prev is not None and x_[0].tobytes() != prev.tobytes():
            problems.append(f'{w}: does not start from the previous terminal state')
        prev = x_[-1]
        path.append(x_[1:])
        ends.append(dict(stage=name, lanes=[dict(zip(('cycle', 'phase', 'distance'), C.nearest(x_[-1, lane], refs)))
                                            for lane in (0, 1)]))
        if is_hold:
            holds[k] = x_
    ledger = [dict(stage_id=i, stage=name, group_id=0, group='adder', steps=steps)
              for i, (name, _, steps, _, _) in enumerate(plan)]
    n_calls = C.check_call_ledger(r.get('calls'), ledger, PROFILE, problems, what, layout='pair')
    if not C.same(r.get('model_calls'), n_calls):
        problems.append(f'{what}: model_calls {r.get("model_calls")} differs from the ledger ({n_calls})')
    want = truth(x, y)
    saved = r.get('readouts') if isinstance(r.get('readouts'), list) else []
    indices = [q.get('move_index') if isinstance(q, dict) else None for q in saved]
    if indices != [0, 1, 2]:
        problems.append(f'{what}: readout set {indices}, expected [0, 1, 2]')
    readouts, settles = [], []
    for k in range(3):
        if k not in holds:
            readouts.append(dict(move_index=k, bits=None, expected=want[k], match=False))
            continue
        bits = [A.readout(holds[k][1:, lane], refs)['bit'] for lane in (0, 1)]
        saved_bits = next((q.get('bits') for q in saved if isinstance(q, dict) and q.get('move_index') == k), None)
        if not C.same(saved_bits, bits):
            problems.append(f'{what}: move {k} recomputed {bits}, saved {saved_bits}')
        readouts.append(dict(move_index=k, move=moves[k], bits=bits, expected=want[k], match=bits == want[k]))
        settles.append(dict(move_index=k, lanes=[C.settle(holds[k][:, lane], refs) for lane in (0, 1)]))
    rep.update(status='checked', problems=problems, readouts=readouts, settle=settles, stage_ends=ends,
               model_calls=r.get('model_calls'), elapsed_seconds=r.get('elapsed_seconds'),
               path=np.concatenate(path) if len(path) == len(plan) + 1 else None)
    return rep


def historical_11_6(d, new_path, bound):
    """Secondary comparison with the first three moves (T, W, T) of the saved 11 + 6 run, qualified on its own."""
    out = dict(run=str(d), status='unavailable', problems=[])
    probs = out['problems']
    try:
        if new_path is None:
            out['reason'] = 'group (1, 0) has no qualified path'
            return out
        C.check_manifest(d, probs, '11+6')
        for f in ('manifest.json', 'result.json'):
            if (d/f).is_file():
                bound.append(C.binding(d/f))
        old = C.load_json(d/'result.json', probs, '11+6')
        if not isinstance(old, dict) or old.get('profile') != 'public-adder' or not C.same(old.get('x'), 11) \
                or not C.same(old.get('y'), 6):
            probs.append('11+6: not the saved public-adder run of 11 + 6')
            return out
        stages = old.get('stages')
        plan = literal_stages(('T', 'W', 'T'))
        if not isinstance(stages, list) or len(stages) < len(plan):
            probs.append('11+6: fewer than six stage records')
            return out
        pieces = []
        for i, (name, ctrl, steps, k, is_hold) in enumerate(plan):
            w = f'11+6 stage {i}'
            check_identity(stages[i], i, name, k, ('T', 'W', 'T')[k], probs, w, profile='public-adder')
            C.check_stage_record(stages[i], steps, probs, w)
            C.check_controls(stages[i], ctrl, probs, w)
            x_ = C.reconstruct(d, stages[i], ctrl, steps, probs, w, bound)
            if x_ is None:
                return out
            if pieces and x_[0].tobytes() != pieces[-1][-1].tobytes():
                probs.append(f'{w}: does not start from the previous terminal state')
            pieces.append(x_ if not pieces else x_[1:])
        if probs:
            return out
        old_path = np.concatenate(pieces)
        out.update(C.compare_paths(new_path, old_path, old_path.shape[0], '11 + 6, first three moves (T, W, T)'))
        return out
    except Exception as exc:
        probs.append(f'11+6: comparison could not be evaluated ({type(exc).__name__}: {exc})')
        return dict(out, status='invalid')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True, help='execution root holding adder-x?-y? jobs')
    p.add_argument('--canonical-references', type=Path, required=True)
    p.add_argument('--canonical-model-inputs', type=Path, required=True)
    p.add_argument('--frozen-commit', required=True)
    p.add_argument('--adder-11-6', type=Path, help='saved 11 + 6 run directory on this machine')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    refs = np.load(a.canonical_references)['states']
    canon_inputs = json.loads(a.canonical_model_inputs.read_text())
    bound = [C.binding(a.canonical_references), C.binding(a.canonical_model_inputs)]
    jobs = []
    for x, y, moves in JOBS:
        d = a.run/f'adder-x{x}-y{y}'
        try:
            rep = check_job(d, x, y, moves, refs, canon_inputs, a.frozen_commit)
        except Exception as exc:                      # never abort the aggregate audit
            rep = dict(job=d.name, x=x, y=y, status='operational_failure',
                       problems=[f'{d.name}: audit exception {type(exc).__name__}: {exc}'])
        if (d/'manifest.json').is_file():
            bound.append(C.binding(d/'manifest.json'))
        jobs.append(rep)
    instrument_ok = all(j['status'] == 'checked' and not j['problems'] for j in jobs)
    table = []
    for j in jobs:
        for lane, c in ((0, 0), (1, 1)):
            rd = j.get('readouts') or []
            table.append(dict(c=c, x=j['x'], y=j['y'],
                              sum=rd[1]['bits'][lane] if len(rd) > 1 and rd[1].get('bits') else None,
                              carry=rd[2]['bits'][lane] if len(rd) > 2 and rd[2].get('bits') else None,
                              expected_sum=c ^ j['x'] ^ j['y'], expected_carry=int(c + j['x'] + j['y'] >= 2)))
    matches = sum(1 for j in jobs for q in (j.get('readouts') or []) for lane in (0, 1)
                  if q.get('bits') is not None and q['bits'][lane] == q['expected'][lane])
    a1 = instrument_ok and matches == 24
    j10 = next(j for j in jobs if (j['x'], j['y']) == (1, 0))
    historical = (historical_11_6(a.adder_11_6, j10.get('path'), bound) if a.adder_11_6
                  else dict(status='not_requested'))
    gap = historical.get('max_gap') if historical.get('status') == 'compared' else None
    for j in jobs:
        j.pop('path', None)
    receipt = {'kind': 'public-full-adder-audit', 'plan': 'docs/public-full-adder-truth-table.md', 'model_calls': 0,
               'frozen_commit': a.frozen_commit, 'instrument_checks_passed': instrument_ok,
               'readouts_matching_truth_table': matches, 'A1_truth_table_passed': a1,
               'operational_or_instrument_problems': [q for j in jobs for q in j['problems']],
               'truth_table': table, 'jobs': jobs, 'group_1_0_largest_gap_to_11_plus_6': gap,
               'group_1_0_against_11_plus_6': historical, 'bound_inputs': bound}
    a.output.write_text(json.dumps(receipt, indent=1) + '\n')
    print('instrument checks:', instrument_ok, '| readouts matching:', matches, 'of 24 | A1:', a1)
    for q in receipt['operational_or_instrument_problems'][:12]:
        print('  problem:', q)
    for row in table:
        print('  c=%d x=%d y=%d  sum %s (want %d)  carry %s (want %d)' % (row['c'], row['x'], row['y'], row['sum'],
              row['expected_sum'], row['carry'], row['expected_carry']))
    print('group (1, 0) against the first three moves of 11 + 6:', historical.get('status'), gap,
          *historical.get('problems', [])[:3])
    return 0 if a1 else 1


if __name__ == '__main__':
    sys.exit(main())
