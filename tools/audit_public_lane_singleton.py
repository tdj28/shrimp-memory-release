#!/usr/bin/env python3
"""Audit of the single-lane arm (docs/public-lane-layout-check.md, "Single-lane arm"). Zero model calls.

The schedule is literal: job k (single-00 .. single-07) runs operation OPS[k // 2] from reference cycle k % 2, one
state at a time, with literal controls, steps and expected bit. Each job's identity is checked against its directory.
A missing or malformed job is recorded and the receipt is still written.

Per job: manifest; identity (profile, shard of 8, group, operation, start cycle, expected bit, move controls and
encodings, stage and row ids); completed status with diagnostics; native64/one-thread readiness, model inputs against
the canonical audited run, the frozen commit and a clean tree; canonical reference cycles; the runner's whole plan in
order (16 stages and 8 rows; its own complete, the others unstarted with no evidence); the literal start; both stages
read with the strict reader at one lane (every array checked on its own against its binding); the literal ordered
call ledger, with every comparison and update showing the scalar path (no batch expansion); the readout recomputed
with audit_public_toggle_sweep.readout against the saved bit and the literal expectation.

S1: the 8 recomputed labels equal the literal expectations and every instrument check passes.
S2 (with --lane-run, the two-lane run of the lane-layout check): each single-lane path against every occurrence of
the same operation and start cycle there (P, R and the duplicate layout: 4 each, 32 in all, every one qualified by
the lane audit's own checks), within 1e-10 at every update. Missing coverage makes S2 incomplete, not passed.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_public_lane_layout as LA  # noqa: E402  (the two-lane job checks, for S2)
import audit_public_toggle_sweep as A  # noqa: E402  (independent readout)
import task8_audit_common as C  # noqa: E402

PROFILE = 'public-lane-singleton'
OPS = ('W', 'E0', 'E1', 'T')
EXPECT = {'W': lambda c: c, 'E0': lambda c: 0, 'E1': lambda c: 1, 'T': lambda c: 1 - c}
TOL = 1e-10


def schedule():
    return [dict(k=2*i + c, op=op, start=c, expected=EXPECT[op](c), name=f'{op}-S{c}',
                 stage_ids=[2*(2*i + c), 2*(2*i + c) + 1], row_id=2*i + c) for i, op in enumerate(OPS) for c in (0, 1)]


def scaffold():
    stages, rows = [], []
    for job in schedule():
        ctrl, steps = C.MOVES[job['op']]
        for local, (name, c, n) in enumerate((('move', ctrl, steps), ('hold', C.H, C.HOLD_STEPS))):
            stages.append(dict(stage_id=2*job['k'] + local, group_id=job['k'], group=job['name'], local_stage_id=local,
                               name=name, controls=c, steps=n, profile=PROFILE))
        rows.append(dict(row_id=job['k'], group_id=job['k'], group=job['name'], lane=0,
                         source_reference_index=job['start'], source_reference_phase=0, profile=PROFILE))
    return stages, rows


def check_job(d, job, refs, canon_inputs, frozen):
    what = d.name
    problems, rep = [], dict(job=what, operation=job['op'], start_cycle=job['start'], expected=job['expected'])
    if not d.is_dir():
        return dict(rep, status='missing', problems=[f'{what}: missing']), None
    C.check_manifest(d, problems, what)
    r = C.load_json(d/'result.json', problems, what)
    if not isinstance(r, dict):
        return dict(rep, status='operational_failure', problems=problems + [f'{what}: no result record']), None
    move_ctrl, move_steps = C.MOVES[job['op']]
    groups = r.get('groups')
    g = groups[0] if isinstance(groups, list) and len(groups) == 1 and isinstance(groups[0], dict) else {}
    want = dict(group_id=job['k'], name=job['name'], operation=job['op'], start_cycle=job['start'],
                expected_bit=job['expected'], stage_ids=job['stage_ids'], row_ids=[job['row_id']],
                pulse_steps=move_steps, controls=move_ctrl, profile=PROFILE)
    if r.get('profile') != PROFILE or not C.same(r.get('shard'), job['k']) or not C.same(r.get('shards'), 8) \
            or not g or any(not C.same(g.get(key), v) for key, v in want.items()) \
            or not C.same(g.get('control_bits'), C.literal_bits(move_ctrl)):
        problems.append(f'{what}: identity differs from the schedule')
    if r.get('status') != 'completed' or g.get('status') != 'completed' or g.get('diagnostics_passed') is not True:
        problems.append(f'{what}: status {r.get("status")}/{g.get("status")}, diagnostics {g.get("diagnostics_passed")}')
    rep['environment'] = C.check_environment(d, canon_inputs, frozen, problems, what)
    try:
        if np.load(d/'reference-cycles.npz')['states'].tobytes() != refs.tobytes():
            problems.append(f'{what}: reference cycles differ from the canonical file')
    except Exception as exc:
        problems.append(f'{what}: reference cycles unreadable ({type(exc).__name__})')
    start = refs[job['start'], 0][None]
    try:
        if np.load(d/f'group-{job["k"]:03d}-start.npy').tobytes() != start.tobytes():
            problems.append(f'{what}: start differs from reference cycle {job["start"]} at phase 0')
    except Exception as exc:
        problems.append(f'{what}: start file unreadable ({type(exc).__name__})')
    lit_stages, lit_rows = scaffold()
    stages, rows = r.get('stages'), r.get('rows')
    if not isinstance(stages, list) or len(stages) != len(lit_stages):
        problems.append(f'{what}: stage list is not the 16-stage scaffold')
        stages = stages if isinstance(stages, list) else []
    own = {}
    for i, w_ in enumerate(lit_stages[:len(stages)]):
        st = stages[i]
        if not isinstance(st, dict) or any(not C.same(st.get(key), v) for key, v in w_.items() if key != 'controls'):
            problems.append(f'{what} stage {i}: identity differs from the literal scaffold')
            continue
        C.check_controls(st, w_['controls'], problems, f'{what} stage {i}')
        if i in job['stage_ids']:
            own[i] = st
        else:
            C.check_unstarted_stage(st, problems, f'{what} stage {i}')
    if not isinstance(rows, list) or len(rows) != len(lit_rows):
        problems.append(f'{what}: row list is not the 8-row scaffold')
        rows = rows if isinstance(rows, list) else []
    for i, w_ in enumerate(lit_rows[:len(rows)]):
        row = rows[i]
        if not isinstance(row, dict) or any(not C.same(row.get(key), v) for key, v in w_.items()):
            problems.append(f'{what} row {i}: identity differs from the literal scaffold')
        elif i != job['row_id'] and (row.get('status') != 'not_started' or row.get('decoded_bit') is not None):
            problems.append(f'{what} row {i}: a row of another shard was started')
    plan = [(job['stage_ids'][0], move_ctrl, move_steps), (job['stage_ids'][1], C.H, C.HOLD_STEPS)]
    prev, parts = start, []
    for sid, ctrl, steps in plan:
        st = own.get(sid)
        if st is None:
            prev = None
            continue
        w = f'{what} stage {sid}'
        C.check_stage_record(st, steps, problems, w)
        x_ = C.reconstruct(d, st, ctrl, steps, problems, w, lanes=1)
        if x_ is None:
            prev = None
            continue
        if prev is None or x_[0].tobytes() != prev.tobytes():
            problems.append(f'{w}: does not start from the previous state')
        prev = x_[-1]
        parts.append(x_ if not parts else x_[1:])
    ledger = [dict(stage_id=sid, stage=name, group_id=job['k'], group=job['name'], steps=steps)
              for (sid, _, steps), name in zip(plan, ('move', 'hold'))]
    n_calls = C.check_call_ledger(r.get('calls'), ledger, PROFILE, problems, what, layout='single')
    calls = r.get('calls') if isinstance(r.get('calls'), list) else []
    if any(isinstance(c, dict) and c.get('category') != 'prefix' and c.get('scalar_path') is not True for c in calls):
        problems.append(f'{what}: a comparison or update does not show the scalar path')
    if not C.same(r.get('model_calls'), n_calls):
        problems.append(f'{what}: model_calls {r.get("model_calls")} differs from the ledger ({n_calls})')
    path = np.concatenate(parts) if len(parts) == 2 else None
    bit = None
    if path is not None:
        bit = A.readout(path[1 + move_steps:, 0], refs)['bit']
        row = rows[job['row_id']] if job['row_id'] < len(rows) and isinstance(rows[job['row_id']], dict) else {}
        if row.get('status') != 'completed' or not C.same(row.get('decoded_bit'), bit) \
                or not C.same(g.get('decoded_bit'), bit):
            problems.append(f'{what}: recomputed bit {bit}, saved row {row.get("decoded_bit")}, group {g.get("decoded_bit")}')
    rep.update(status='checked', problems=problems, bit=bit, match=bit == job['expected'],
               model_calls=r.get('model_calls'))
    return rep, (path[:, 0] if path is not None else None)


def against_lanes(lane_run, single, refs, canon_inputs, frozen):
    """S2: every qualified two-lane occurrence of each operation and start cycle against the single-lane path."""
    occurrences, lane_problems = {}, []
    for job in LA.schedule():
        d = Path(lane_run)/f'lanes-{job["k"]:02d}'
        try:
            rep, path = LA.check_job(d, job, refs, canon_inputs, frozen)
        except Exception as exc:
            rep, path = dict(status='operational_failure', problems=[f'{d.name}: {type(exc).__name__}: {exc}']), None
        lane_problems += rep.get('problems', [])
        if path is None or rep.get('status') != 'checked' or rep.get('problems'):
            continue
        for lane, c in enumerate(job['starts']):
            occurrences.setdefault((job['op'], c), []).append((job['layout'], lane, path[:, lane]))
    gaps, notes = [], []
    for job in schedule():
        key, mine = (job['op'], job['start']), single.get(job['k'])
        occ = occurrences.get(key, [])
        if mine is None or len(occ) != 4:
            notes.append(f'{job["name"]}: single-lane path {"present" if mine is not None else "missing"}, '
                         f'{len(occ)} of 4 two-lane occurrences qualified')
            continue
        for layout, lane, pth in occ:
            ok = pth.shape == mine.shape
            gaps.append(dict(operation=job['op'], start_cycle=job['start'], layout=layout, lane=lane,
                             max_gap=float(np.abs(pth - mine).max()) if ok else None,
                             bitwise_identical=ok and pth.tobytes() == mine.tobytes()))
    complete = not notes and len(gaps) == 32 and all(x['max_gap'] is not None for x in gaps)
    return dict(status='compared' if complete else 'incomplete', comparisons=len(gaps), tolerance=TOL,
                largest_gap=max((x['max_gap'] for x in gaps if x['max_gap'] is not None), default=None),
                bitwise_identical=sum(x['bitwise_identical'] for x in gaps), coverage_notes=notes, gaps=gaps,
                lane_run_problems=lane_problems[:40])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True, help='execution root holding single-00 .. single-07')
    p.add_argument('--canonical-references', type=Path, required=True)
    p.add_argument('--canonical-model-inputs', type=Path, required=True)
    p.add_argument('--frozen-commit', required=True)
    p.add_argument('--lane-run', type=Path, help='execution root of the two-lane lane-layout check, for S2')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    refs = np.load(a.canonical_references)['states']
    canon_inputs = json.loads(a.canonical_model_inputs.read_text())
    bound = [C.binding(a.canonical_references), C.binding(a.canonical_model_inputs)]
    jobs, single = [], {}
    for job in schedule():
        d = a.run/f'single-{job["k"]:02d}'
        try:
            rep, path = check_job(d, job, refs, canon_inputs, a.frozen_commit)
        except Exception as exc:
            rep, path = dict(job=d.name, operation=job['op'], start_cycle=job['start'], status='operational_failure',
                             problems=[f'{d.name}: audit exception {type(exc).__name__}: {exc}']), None
        if (d/'manifest.json').is_file():
            bound.append(C.binding(d/'manifest.json'))
        jobs.append(rep)
        if path is not None and rep['status'] == 'checked' and not rep['problems']:
            single[job['k']] = path
    instrument_ok = all(j['status'] == 'checked' and not j['problems'] for j in jobs)
    s1 = instrument_ok and all(j.get('match') for j in jobs)
    if a.lane_run:
        s2 = against_lanes(a.lane_run, single, refs, canon_inputs, a.frozen_commit)
        for k in range(16):
            m = a.lane_run/f'lanes-{k:02d}'/'manifest.json'
            if m.is_file():
                bound.append(C.binding(m))
    else:
        s2 = dict(status='not_requested')
    s2_ok = s2['status'] == 'compared' and all(x['max_gap'] <= TOL for x in s2['gaps'])
    receipt = {'kind': 'public-lane-singleton-audit', 'plan': 'docs/public-lane-layout-check.md', 'model_calls': 0,
               'frozen_commit': a.frozen_commit, 'instrument_checks_passed': instrument_ok,
               'S1_all_8_lanes_as_expected': s1, 'S2_all_within_tolerance': s2_ok if a.lane_run else None,
               'operational_or_instrument_problems': [q for j in jobs for q in j['problems']],
               'jobs': jobs, 'against_two_lane_runs': s2, 'bound_inputs': bound}
    a.output.write_text(json.dumps(receipt, indent=1) + '\n')
    print('instrument checks:', instrument_ok, '| S1:', s1, '| S2:', s2['status'],
          s2.get('comparisons'), 'comparisons, largest gap', s2.get('largest_gap'))
    for q in receipt['operational_or_instrument_problems'][:12]:
        print('  problem:', q)
    return 0 if s1 and (s2_ok or not a.lane_run) else 1


if __name__ == '__main__':
    sys.exit(main())
