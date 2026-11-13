#!/usr/bin/env python3
"""Audit of the lane-layout runs (docs/public-lane-layout-check.md). Zero model calls.

Revised after Codex's Task 8 review (docs/audits/2026-11-13-task8/review.md) and recheck
(docs/audits/2026-11-13-task8-recheck/review.md). The schedule is literal here: job k is operation OPS[k // 4] under
layout LAYOUTS[k % 4], with literal starts, controls, steps and expected labels. Each job's identity is checked
against its directory, never derived from the result's own records. A missing or malformed job is recorded and the
receipt is still written.

Each job's result keeps the runner's whole plan: 16 groups' 32 stages and 32 rows, of which only its own two are
executed. All 32 stage records are checked in order against the literal scaffold (id, group, name, controls,
steps); the job's own two must be complete and the other 30 must be unstarted and carry no evidence. Rows likewise.
The call ledger must be the literal ordered sequence for the job's two stages. Every saved array is checked on its
own against its binding before anything is joined.

Per job, the same instrument checks as the full-adder audit (manifest, readiness, model inputs, source commit,
references, literal controls and encodings for both stages including H, block controls, exact step counts, cache
gates, call ledger, carried states). L1: the 32 recomputed labels equal the literal expectations. L2: for each of the
8 operation/start combinations there must be exactly 4 occurrences (exactly one in P) and exactly 24 comparisons in
all, each path within 1e-10 of the P path at every update; missing coverage makes L2 fail. Reported: measured phase at
the end of each stage, bitwise identity, the largest gaps, and the P paths against earlier runs. Those comparisons
are secondary and each is qualified on its own (manifest, identities, records and every file read, all bound): the
pod's phase-0 toggle check (12 + 240), Part B's full-offset run on this machine (the first 6 pulse updates), the
sweep's pod write baselines (120 + 240), and the sweep's hold baseline over the first 252 updates at H (its stage
boundary is at update 120, the W run's at 12). A comparison that cannot be qualified, including one whose P baseline
is missing, is reported unavailable with its reasons and never stops the receipt.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_public_toggle_sweep as A  # noqa: E402  (independent readout)
import task8_audit_common as C  # noqa: E402

PROFILE = 'public-lane-layout'
OPS = ('W', 'E0', 'E1', 'T')
LAYOUTS = (('P', (0, 1)), ('R', (1, 0)), ('D0', (0, 0)), ('D1', (1, 1)))
EXPECT = {'W': lambda a, b: [a, b], 'E0': lambda a, b: [0, 0], 'E1': lambda a, b: [1, 1],
          'T': lambda a, b: [1 - a, 1 - b]}
TOL = 1e-10


def schedule():
    out = []
    for i, op in enumerate(OPS):
        for j, (lay, starts) in enumerate(LAYOUTS):
            k = 4*i + j
            out.append(dict(k=k, op=op, layout=lay, starts=starts, expected=EXPECT[op](*starts),
                            name=f'{op}-{lay}', stage_ids=[2*k, 2*k + 1], row_ids=[2*k, 2*k + 1]))
    return out


def scaffold():
    """The 32 literal stage records and 32 literal rows every job's result must hold, in order."""
    stages, rows = [], []
    for job in schedule():
        ctrl, steps = C.MOVES[job['op']]
        for local, (name, c, n) in enumerate((('move', ctrl, steps), ('hold', C.H, C.HOLD_STEPS))):
            stages.append(dict(stage_id=2*job['k'] + local, group_id=job['k'], group=job['name'], local_stage_id=local,
                               name=name, controls=c, steps=n, profile=PROFILE))
        for lane in (0, 1):
            rows.append(dict(row_id=2*job['k'] + lane, group_id=job['k'], group=job['name'], lane=lane,
                             source_reference_index=job['starts'][lane], source_reference_phase=0, profile=PROFILE))
    return stages, rows


def check_job(d, job, refs, canon_inputs, frozen):
    what = d.name
    problems, rep = [], dict(job=what, operation=job['op'], layout=job['layout'], expected=job['expected'])
    if not d.is_dir():
        return dict(rep, status='missing', problems=[f'{what}: missing']), None
    C.check_manifest(d, problems, what)
    r = C.load_json(d/'result.json', problems, what)
    if not isinstance(r, dict):
        return dict(rep, status='operational_failure', problems=problems + [f'{what}: no result record']), None
    move_ctrl, move_steps = C.MOVES[job['op']]
    groups = r.get('groups')
    g = groups[0] if isinstance(groups, list) and len(groups) == 1 and isinstance(groups[0], dict) else {}
    want_group = dict(group_id=job['k'], name=job['name'], operation=job['op'], layout=job['layout'],
                      stage_ids=job['stage_ids'], row_ids=job['row_ids'], start_cycles=list(job['starts']),
                      expected_pair=job['expected'], pulse_steps=move_steps, controls=move_ctrl, profile=PROFILE)
    if not C.same(r.get('profile'), PROFILE) or not C.same(r.get('shard'), job['k']) or not C.same(r.get('shards'), 16) \
            or not g or any(not C.same(g.get(key), v) for key, v in want_group.items()) \
            or not C.same(g.get('control_bits'), C.literal_bits(move_ctrl)):
        problems.append(f'{what}: identity differs from the schedule (profile, shard, group, stages, rows, starts, '
                        'expected pair or move controls)')
    if r.get('status') != 'completed' or g.get('status') != 'completed' or g.get('diagnostics_passed') is not True:
        problems.append(f'{what}: status {r.get("status")}/{g.get("status")}, diagnostics {g.get("diagnostics_passed")}')
    rep['environment'] = C.check_environment(d, canon_inputs, frozen, problems, what)
    try:
        run_refs = np.load(d/'reference-cycles.npz')['states']
        if run_refs.tobytes() != refs.tobytes():
            problems.append(f'{what}: reference cycles differ from the canonical file')
    except Exception as exc:
        problems.append(f'{what}: reference cycles unreadable ({type(exc).__name__})')
    a, b = job['starts']
    start = np.stack([refs[a, 0], refs[b, 0]])
    try:
        if np.load(d/f'group-{job["k"]:03d}-start.npy').tobytes() != start.tobytes():
            problems.append(f'{what}: start differs from the layout')
    except Exception as exc:
        problems.append(f'{what}: start file unreadable ({type(exc).__name__})')

    # The whole planned scaffold, in order: own stages complete, the other 30 unstarted.
    lit_stages, lit_rows = scaffold()
    stages, rows = r.get('stages'), r.get('rows')
    if not isinstance(stages, list) or len(stages) != len(lit_stages):
        problems.append(f'{what}: stage list is not the 32-stage scaffold')
        stages = stages if isinstance(stages, list) else []
    own = {}
    for i, want in enumerate(lit_stages[:len(stages)]):
        st, w = stages[i], f'{what} stage {i}'
        if not isinstance(st, dict) or any(not C.same(st.get(key), v) for key, v in want.items() if key != 'controls'):
            problems.append(f'{w}: identity differs from the literal scaffold')
            continue
        C.check_controls(st, want['controls'], problems, w)
        if i in job['stage_ids']:
            own[i] = st
        else:
            C.check_unstarted_stage(st, problems, w)
    if not isinstance(rows, list) or len(rows) != len(lit_rows):
        problems.append(f'{what}: row list is not the 32-row scaffold')
        rows = rows if isinstance(rows, list) else []
    for i, want in enumerate(lit_rows[:len(rows)]):
        row = rows[i]
        if not isinstance(row, dict) or any(not C.same(row.get(key), v) for key, v in want.items()):
            problems.append(f'{what} row {i}: identity differs from the literal scaffold')
        elif i not in job['row_ids'] and (row.get('status') != 'not_started' or row.get('decoded_bit') is not None
                                          or row.get('readout') is not None):
            problems.append(f'{what} row {i}: a row of another shard was started')

    plan = [(job['stage_ids'][0], move_ctrl, move_steps), (job['stage_ids'][1], C.H, C.HOLD_STEPS)]
    prev, paths, ends = start, [], []
    for sid, ctrl, steps in plan:
        st = own.get(sid)
        if st is None:
            prev = None
            continue
        w = f'{what} stage {sid}'
        C.check_stage_record(st, steps, problems, w)
        x_ = C.reconstruct(d, st, ctrl, steps, problems, w)
        if x_ is None:
            prev = None
            continue
        if prev is not None and x_[0].tobytes() != prev.tobytes():
            problems.append(f'{w}: does not start from the previous state')
        prev = x_[-1]
        paths.append(x_ if not paths else x_[1:])
        ends.append(dict(stage=sid, lanes=[dict(zip(('cycle', 'phase', 'distance'), C.nearest(x_[-1, lane], refs)))
                                           for lane in (0, 1)]))
    ledger = [dict(stage_id=sid, stage=name, group_id=job['k'], group=job['name'], steps=steps)
              for (sid, _, steps), name in zip(plan, ('move', 'hold'))]
    n_calls = C.check_call_ledger(r.get('calls'), ledger, PROFILE, problems, what, layout='pair')
    if not C.same(r.get('model_calls'), n_calls):
        problems.append(f'{what}: model_calls {r.get("model_calls")} differs from the ledger ({n_calls})')
    path = np.concatenate(paths) if len(paths) == 2 else None
    bits = None
    if path is not None:
        hold = path[1 + move_steps:]
        bits = [A.readout(hold[:, lane], refs)['bit'] for lane in (0, 1)]
        if not C.same(g.get('decoded_pair'), bits):
            problems.append(f'{what}: recomputed {bits}, saved pair {g.get("decoded_pair")}')
        for lane, rid in enumerate(job['row_ids']):
            row = rows[rid] if rid < len(rows) and isinstance(rows[rid], dict) else {}
            if row.get('status') != 'completed' or not C.same(row.get('decoded_bit'), bits[lane]):
                problems.append(f'{what} row {rid}: saved {row.get("status")}/{row.get("decoded_bit")}, '
                                f'recomputed {bits[lane]}')
    rep.update(status='checked', problems=problems, bits=bits, match=bits == job['expected'], stage_ends=ends,
               model_calls=r.get('model_calls'))
    return rep, path


def load_old(root, gid, name, push_ctrl, push_steps, bound, label):
    """An earlier pulse-then-hold group (toggle check or sweep), qualified on its own and read with the strict
    reader: one group file, manifest, an ordered stage list, the group's literal name, stage names, controls, steps
    and completion, every file read bound. Returns (path or None, problems)."""
    probs = []
    try:
        hits = sorted(Path(root).glob(f'shard-*/group-{gid:02d}.json'))
        if len(hits) != 1:
            return None, [f'{label}: expected one group-{gid:02d}.json, found {len(hits)}']
        d = hits[0].parent
        C.check_manifest(d, probs, label)
        for f in (hits[0], d/'result.json', d/'manifest.json'):
            if f.is_file():
                bound.append(C.binding(f))
        g, r = C.load_json(hits[0], probs, label), C.load_json(d/'result.json', probs, label)
        if not isinstance(g, dict) or not isinstance(r, dict):
            return None, probs + [f'{label}: group or result record unreadable']
        stages = r.get('stages')
        sids = g.get('stage_ids')
        if g.get('name') != name or not C.same(g.get('group_id'), gid) or not isinstance(sids, list) or len(sids) != 2 \
                or not isinstance(stages, list) or [s.get('stage_id') for s in stages] != list(range(len(stages))) \
                or any(not isinstance(i, int) or not 0 <= i < len(stages) for i in sids):
            return None, probs + [f'{label}: group {gid} is not "{name}" with two indexed stages']
        parts = []
        for sid, stage_name, ctrl, steps in ((sids[0], 'pulse', push_ctrl, push_steps),
                                             (sids[1], 'hold', C.H, C.HOLD_STEPS)):
            st, w = stages[sid], f'{label} {stage_name}'
            if st.get('name') != stage_name or not C.same(st.get('group_id'), gid):
                probs.append(f'{w}: stage {sid} is not this group\'s {stage_name}')
            C.check_stage_record(st, steps, probs, w)
            C.check_controls(st, ctrl, probs, w)
            parts.append(C.reconstruct(d, st, ctrl, steps, probs, w, bound))
        if probs or any(x is None for x in parts):
            return None, probs
        if parts[1][0].tobytes() != parts[0][-1].tobytes():
            return None, [f'{label}: hold does not start where the pulse ended']
        return np.concatenate([parts[0], parts[1][1:]]), probs
    except Exception as exc:
        return None, probs + [f'{label}: could not be evaluated ({type(exc).__name__}: {exc})']


def load_part_b(run, bound, label):
    """Part B's full-offset pulse: shard-04, stage 4, group lambda 1.00, six updates at the toggle controls."""
    probs = []
    try:
        d = Path(run)/'shard-04'
        C.check_manifest(d, probs, label)
        for f in (d/'result.json', d/'manifest.json'):
            if f.is_file():
                bound.append(C.binding(f))
        r = C.load_json(d/'result.json', probs, label)
        stages = r.get('stages') if isinstance(r, dict) else None
        if not isinstance(stages, list) or [s.get('stage_id') for s in stages] != list(range(len(stages))) \
                or len(stages) != 5:
            return None, probs + [f'{label}: stage list is not the five offset stages']
        st = stages[4]
        if st.get('name') != 'pulse' or not C.same(st.get('group_id'), 4) \
                or st.get('group') != 't030-r3.5-d006-offset-lambda1.00':
            probs.append(f'{label}: stage 4 is not the lambda = 1.00 pulse')
        C.check_stage_record(st, 6, probs, label)
        C.check_controls(st, C.T, probs, label)
        x_ = C.reconstruct(d, st, C.T, 6, probs, label, bound)
        return (x_ if not probs else None), probs
    except Exception as exc:
        return None, probs + [f'{label}: could not be evaluated ({type(exc).__name__}: {exc})']


def comparison(p_path, op, loader, n, label, note=None):
    """One secondary comparison: unavailable, with reasons, unless both the P baseline and the earlier run qualify."""
    if op not in p_path:
        return dict(earlier=label, status='unavailable', reason=f'P baseline for {op} missing or not qualified')
    old, probs = loader()
    if old is None or probs:
        return dict(earlier=label, status='unavailable', reason='earlier run did not qualify', problems=probs)
    return C.compare_paths(p_path[op], old, n, label, note)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True, help='execution root holding lanes-00 .. lanes-15')
    p.add_argument('--canonical-references', type=Path, required=True)
    p.add_argument('--canonical-model-inputs', type=Path, required=True)
    p.add_argument('--frozen-commit', required=True)
    p.add_argument('--check-root', type=Path, help='toggle check batch-1 pod-returned directory')
    p.add_argument('--sweep-root', type=Path, help='toggle sweep returned-pod directory')
    p.add_argument('--part-b-run', type=Path, help='Part B offset run directory (shard-04 is the full offset)')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    refs = np.load(a.canonical_references)['states']
    canon_inputs = json.loads(a.canonical_model_inputs.read_text())
    bound = [C.binding(a.canonical_references), C.binding(a.canonical_model_inputs)]
    jobs, paths = [], {}
    for job in schedule():
        d = a.run/f'lanes-{job["k"]:02d}'
        try:
            rep, path = check_job(d, job, refs, canon_inputs, a.frozen_commit)
        except Exception as exc:
            rep, path = dict(job=d.name, operation=job['op'], layout=job['layout'], status='operational_failure',
                             problems=[f'{d.name}: audit exception {type(exc).__name__}: {exc}']), None
        if (d/'manifest.json').is_file():
            bound.append(C.binding(d/'manifest.json'))
        jobs.append(rep)
        if path is not None and rep['status'] == 'checked' and not rep['problems']:
            for lane, c in enumerate(job['starts']):
                paths.setdefault((job['op'], c), []).append((job['layout'], lane, path[:, lane]))
    instrument_ok = all(j['status'] == 'checked' and not j['problems'] for j in jobs)
    l1 = instrument_ok and all(j.get('match') for j in jobs)
    gaps, coverage_ok, coverage_notes = [], True, []
    for op in OPS:
        for c in (0, 1):
            occ = paths.get((op, c), [])
            ps = [pth for lay, _, pth in occ if lay == 'P']
            if len(occ) != 4 or len(ps) != 1:
                coverage_ok = False
                coverage_notes.append(f'{op} from cycle {c}: {len(occ)} qualified occurrences, {len(ps)} in P')
                continue
            for lay, lane, pth in occ:
                if lay != 'P':
                    gaps.append(dict(operation=op, start_cycle=c, layout=lay, lane=lane,
                                     max_gap=float(np.abs(pth - ps[0]).max()), bitwise_identical=pth.tobytes() == ps[0].tobytes()))
    coverage_ok = coverage_ok and len(gaps) == 24
    l2 = instrument_ok and coverage_ok and all(x['max_gap'] <= TOL for x in gaps)

    # Secondary comparisons with earlier runs: each needs both P paths of its operation, and qualifies on its own.
    p_path = {}
    for op in OPS:
        p0 = [pth for lay, _, pth in paths.get((op, 0), []) if lay == 'P']
        p1 = [pth for lay, _, pth in paths.get((op, 1), []) if lay == 'P']
        if len(p0) == 1 and len(p1) == 1:
            p_path[op] = np.stack([p0[0], p1[0]], axis=1)
    cross = []
    if a.check_root:
        cross.append(comparison(p_path, 'T', lambda: load_old(a.check_root, 0, 't030-r3.5-d012-phase0', C.T, 12, bound,
                                                              'toggle check phase 0 (pod)'),
                                253, 'toggle check phase 0 (pod), 12 + 240 updates'))
    if a.part_b_run:
        cross.append(comparison(p_path, 'T', lambda: load_part_b(a.part_b_run, bound, 'Part B lambda = 1'),
                                7, 'Part B full-offset run (this machine), first 6 pulse updates'))
    if a.sweep_root:
        for op, gid, name, ctrl, steps, n, label, note in (
                ('E0', 1, 'baseline-write0', C.E0, 120, 361, 'sweep baseline-write0 (pod), 120 + 240 updates', None),
                ('E1', 2, 'baseline-write1', C.E1, 120, 361, 'sweep baseline-write1 (pod), 120 + 240 updates', None),
                ('W', 0, 'baseline-hold', C.H, 120, 253, 'sweep baseline-hold (pod), first 252 updates at H',
                 'both are at H throughout; the baseline changes stage at update 120 and the W run at update 12')):
            cross.append(comparison(p_path, op, lambda gid=gid, name=name, ctrl=ctrl, steps=steps, label=label:
                                    load_old(a.sweep_root, gid, name, ctrl, steps, bound, label), n, label, note))
    receipt = {'kind': 'public-lane-layout-audit', 'plan': 'docs/public-lane-layout-check.md', 'model_calls': 0,
               'frozen_commit': a.frozen_commit, 'instrument_checks_passed': instrument_ok,
               'L1_all_32_lanes_as_expected': l1, 'L2_coverage_complete': coverage_ok,
               'L2_all_paths_within_tolerance': l2, 'tolerance': TOL, 'comparisons': len(gaps),
               'coverage_notes': coverage_notes, 'largest_gap': max((x['max_gap'] for x in gaps), default=None),
               'operational_or_instrument_problems': [q for j in jobs for q in j['problems']],
               'jobs': jobs, 'gaps': gaps, 'against_earlier_runs': cross, 'bound_inputs': bound}
    a.output.write_text(json.dumps(receipt, indent=1) + '\n')
    print('instrument checks:', instrument_ok, '| L1:', l1, '| coverage:', coverage_ok, f'({len(gaps)} comparisons)',
          '| L2:', l2, '| largest gap:', receipt['largest_gap'])
    for q in receipt['operational_or_instrument_problems'][:12]:
        print('  problem:', q)
    for x in cross:
        print('  earlier run:', {k: v for k, v in x.items() if k != 'problems'})
    return 0 if (l1 and l2) else 1


if __name__ == '__main__':
    sys.exit(main())
