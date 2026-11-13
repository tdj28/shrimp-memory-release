"""Regression tests for the revised Task 8 validators and launcher (Codex review docs/audits/2026-11-13-task8).

All fixtures are synthetic files in the exact archive formats; no model is loaded. Positive fixtures must pass;
every fault Codex reproduced against the first versions must now be rejected, and missing or broken jobs must still
produce a receipt.
"""
import ast
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import task8_audit_common as C  # noqa: E402
import audit_public_full_adder as AA  # noqa: E402
import audit_public_lane_layout as LA  # noqa: E402
import run_public_full_adder as FA  # noqa: E402  (fixtures use the runners' own planned scaffolds)
import run_public_lane_layout as LR  # noqa: E402

FROZEN = 'f' * 40


def refs():
    r = np.zeros((2, 6, 896))
    for c in range(2):
        for ph in range(6):
            r[c, ph, ph] = .2
            r[c, ph, 10] = c * .1
    return r


def save(p, v):
    p.write_text(json.dumps(v, indent=1))


def binding(p):
    return {'path': p.name, 'bytes': p.stat().st_size, 'sha256': C.sha(p)}


def manifest(d):
    save(d/'manifest.json', {'files': [binding(p) for p in sorted(d.iterdir())
                                       if p.is_file() and p.name != 'manifest.json']})


def env_files(d, R, canon):
    np.savez(d/'reference-cycles.npz', states=R)
    save(d/'backend-readiness.json', {'mode': 'native64', 'torch_threads': 1, 'torch_interop_threads': 1,
                                      'thread_environment': {k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                                                               'OPENBLAS_NUM_THREADS',
                                                                               'NUMEXPR_NUM_THREADS')}})
    save(d/'model-inputs.json', canon)
    save(d/'source.json', {'source_commit': FROZEN, 'working_tree_status': '', 'files': [], 'arguments': {}})


def fill_stage(d, st, initial, end_bits, R):
    """Complete one planned stage record the way the instrument does: initial-state binding with its array digest,
    shape and dtype, 12-update blocks named for their last update, counts, cache gates."""
    sid, steps, ctrl = st['stage_id'], st['steps'], st['controls']
    f = d/f'stage-{sid:02d}-initial.npy'
    np.save(f, initial)
    st.update(status='completed', attempted_steps=steps, returned_steps=steps, completed_steps=steps,
              failed_outputs=[], cache_comparison={'passed': True}, final_scalar_cache_unchanged=True,
              initial=dict(binding(f), array_sha256=hashlib.sha256(np.ascontiguousarray(initial).tobytes()).hexdigest(),
                           shape=[2, 896], dtype='float64'), blocks=[])
    y = np.stack([R[list(end_bits), (t + 1) % 6] for t in range(steps)])
    for s0 in range(0, steps, 12):
        z = y[s0:s0 + 12]
        f = d/f'stage-{sid:02d}-block-{s0 + len(z):03d}.npz'
        np.savez_compressed(f, states=z, token_ids=np.zeros((len(z), 2), dtype=np.int64),
                            step_seconds=np.ones(len(z)), first_state_index=np.int64(s0 + 1),
                            last_state_index=np.int64(s0 + len(z)), **ctrl)
        st['blocks'].append(binding(f))
    return y[-1]


def returned(calls):
    """Mark planned call slots as returned with the evidence the instrument records for each kind."""
    for c in calls:
        c['status'] = 'returned'
        if c['category'] == 'prefix':
            c['dtype_observations'] = {'norm': {'input': 'torch.float64'}}
        else:
            c.update(precision_passed=True, outputs_valid=True, input_unchanged=True, cache_unchanged=True,
                     expanded_cache_valid=True)
            if c['category'] == 'update':
                c['state_carried'] = True
    return calls


def adder_plan(x, y):
    old = FA.X, FA.Y
    FA.X, FA.Y = x, y
    try:
        stages, calls = FA.plans()
        return stages, calls, FA.moves()
    finally:
        FA.X, FA.Y = old


@pytest.fixture
def canon(tmp_path):
    R = refs()
    np.savez(tmp_path/'canonical-refs.npz', states=R)
    inputs = {k: {'sha256': k * 8} for k in ('config', 'vectors', 'weights')}
    save(tmp_path/'canonical-inputs.json', inputs)
    return R, inputs, tmp_path/'canonical-refs.npz', tmp_path/'canonical-inputs.json'


def build_adder(root, canon):
    R, inputs, _, _ = canon
    for x, y, moves in AA.JOBS:
        d = root/f'adder-x{x}-y{y}'
        d.mkdir(parents=True)
        env_files(d, R, inputs)
        stages, calls, planned = adder_plan(x, y)
        assert [m['move'] for m in planned] == list(moves)
        want = AA.truth(x, y)
        prev = R[:, 0]
        for st in stages:
            prev = fill_stage(d, st, prev, want[st['move_index']], R)
        save(d/'result.json', dict(profile=FA.PROFILE, x=x, y=y, moves=planned, status='completed', stages=stages,
                                   calls=returned(calls), model_calls=len(calls),
                                   readouts=[{'move_index': k, 'bits': want[k]} for k in range(3)],
                                   elapsed_seconds=1.0))
        manifest(d)


def build_lanes(root, canon):
    """Each job keeps the runner's whole plan (16 groups, 32 stages, 32 rows); only its own group is executed."""
    R, inputs, _, _ = canon
    for job in LA.schedule():
        k = job['k']
        d = root/f'lanes-{k:02d}'
        d.mkdir(parents=True)
        env_files(d, R, inputs)
        groups, stages, rows = LR.plans()
        calls = LR.sweep.shard_calls(groups, stages, [k])
        for c in calls:
            c['profile'] = LR.PROFILE
        a, b = job['starts']
        start = np.stack([R[a, 0], R[b, 0]])
        np.save(d/f'group-{k:03d}-start.npy', start)
        prev = start
        for sid in job['stage_ids']:
            prev = fill_stage(d, stages[sid], prev, job['expected'], R)
        for lane, rid in enumerate(job['row_ids']):
            rows[rid].update(status='completed', readout={'qualified_label': ('ad', 'td')[job['expected'][lane]]},
                             decoded_bit=job['expected'][lane])
        g = groups[k]
        g.update(status='completed', diagnostics_passed=True, decoded_pair=list(job['expected']), matches_expected=True)
        save(d/'result.json', dict(profile=LR.PROFILE, shard=k, shards=16, status='completed', outcome='checked',
                                   groups=[g], stages=stages, rows=rows, calls=returned(calls),
                                   model_calls=len(calls)))
        manifest(d)


def fill_stage_path(d, st, segment):
    """Like fill_stage, from a given path segment (steps + 1 states) instead of the synthetic cycle walk."""
    sid, steps, ctrl = st['stage_id'], st['steps'], st['controls']
    assert segment.shape[0] == steps + 1
    f = d/f'stage-{sid:02d}-initial.npy'
    np.save(f, segment[0])
    st.update(status='completed', attempted_steps=steps, returned_steps=steps, completed_steps=steps,
              failed_outputs=[], cache_comparison={'passed': True}, final_scalar_cache_unchanged=True,
              initial=dict(binding(f), array_sha256=hashlib.sha256(np.ascontiguousarray(segment[0]).tobytes()).hexdigest(),
                           shape=list(segment.shape[1:]), dtype='float64'), blocks=[])
    for s0 in range(0, steps, 12):
        z = segment[1 + s0:1 + s0 + 12]
        f = d/f'stage-{sid:02d}-block-{s0 + len(z):03d}.npz'
        np.savez_compressed(f, states=z, token_ids=np.zeros(z.shape[:2], dtype=np.int64),
                            step_seconds=np.ones(len(z)), first_state_index=np.int64(s0 + 1),
                            last_state_index=np.int64(s0 + len(z)), **ctrl)
        st['blocks'].append(binding(f))


def build_lanes_with(root, canon, path_of):
    """build_lanes with prescribed lane paths: path_of(operation, start cycle, move steps) -> (steps + 1, 896)."""
    R, inputs, _, _ = canon
    for job in LA.schedule():
        k = job['k']
        d = root/f'lanes-{k:02d}'
        d.mkdir(parents=True)
        env_files(d, R, inputs)
        groups, stages, rows = LR.plans()
        calls = LR.sweep.shard_calls(groups, stages, [k])
        for c in calls:
            c['profile'] = LR.PROFILE
        a, b = job['starts']
        np.save(d/f'group-{k:03d}-start.npy', np.stack([R[a, 0], R[b, 0]]))
        steps = C.MOVES[job['op']][1]
        path = np.stack([path_of(job['op'], a, steps), path_of(job['op'], b, steps)], axis=1)
        fill_stage_path(d, stages[2*k], path[:steps + 1])
        fill_stage_path(d, stages[2*k + 1], path[steps:])
        for lane, rid in enumerate(job['row_ids']):
            rows[rid].update(status='completed', readout={'qualified_label': ('ad', 'td')[job['expected'][lane]]},
                             decoded_bit=job['expected'][lane])
        g = groups[k]
        g.update(status='completed', diagnostics_passed=True, decoded_pair=list(job['expected']), matches_expected=True)
        save(d/'result.json', dict(profile=LR.PROFILE, shard=k, shards=16, status='completed', outcome='checked',
                                   groups=[g], stages=stages, rows=rows, calls=returned(calls),
                                   model_calls=len(calls)))
        manifest(d)


def own_stage(r, local):
    return r['stages'][r['groups'][0]['stage_ids'][local]]


def audit(module, root, canon, tmp_path, name, extra=()):
    _, _, refs_path, inputs_path = canon
    out = tmp_path/f'{name}.json'
    rc = module.main(['--run', str(root), '--canonical-references', str(refs_path), '--canonical-model-inputs',
                      str(inputs_path), '--frozen-commit', FROZEN, '--output', str(out), *extra])
    return rc, json.loads(out.read_text())


def edit_result(d, change):
    r = json.loads((d/'result.json').read_text())
    change(d, r)
    save(d/'result.json', r)
    manifest(d)


# ----------------------------------------------------------------- full adder
def test_adder_positive(tmp_path, canon):
    build_adder(tmp_path/'run', canon)
    rc, rep = audit(AA, tmp_path/'run', canon, tmp_path, 'a')
    assert rc == 0 and rep['A1_truth_table_passed'] and rep['readouts_matching_truth_table'] == 24
    assert rep['instrument_checks_passed'] and not rep['operational_or_instrument_problems']


@pytest.mark.parametrize('fault', ['extra_stage', 'truncated_hold', 'failed_cache', 'wrong_recorded_bit',
                                   'wrong_source_commit', 'not_native64', 'wrong_moves', 'missing_prefix_call'])
def test_adder_faults_rejected(tmp_path, canon, fault):
    build_adder(tmp_path/'run', canon)
    d = tmp_path/'run'/'adder-x0-y0'
    if fault == 'extra_stage':
        edit_result(d, lambda d, r: r['stages'].append(copy.deepcopy(r['stages'][-1])))
    elif fault == 'truncated_hold':
        def f(d, r):
            st = r['stages'][1]
            st['blocks'] = st['blocks'][:1]
            st['completed_steps'] = 12
        edit_result(d, f)
    elif fault == 'failed_cache':
        edit_result(d, lambda d, r: r['stages'][0].update(cache_comparison={'passed': False}))
    elif fault == 'wrong_recorded_bit':
        edit_result(d, lambda d, r: r['readouts'][0]['bits'].__setitem__(0, 1))
    elif fault == 'wrong_source_commit':
        save(d/'source.json', {'source_commit': 'e' * 40, 'working_tree_status': ''}); manifest(d)
    elif fault == 'not_native64':
        save(d/'backend-readiness.json', {'mode': 'float32', 'torch_threads': 1, 'torch_interop_threads': 1,
                                          'thread_environment': {'OMP_NUM_THREADS': '1'}}); manifest(d)
    elif fault == 'wrong_moves':
        edit_result(d, lambda d, r: r.update(moves=[{'move': 'T'}, {'move': 'W'}, {'move': 'E0'}]))
    elif fault == 'missing_prefix_call':
        edit_result(d, lambda d, r: r.update(calls=r['calls'][1:], model_calls=len(r['calls']) - 1))
    rc, rep = audit(AA, tmp_path/'run', canon, tmp_path, fault)
    assert rc == 1 and not rep['A1_truth_table_passed'] and not rep['instrument_checks_passed']


@pytest.mark.parametrize('how', ['missing', 'malformed'])
def test_adder_missing_or_broken_job_still_writes_receipt(tmp_path, canon, how):
    build_adder(tmp_path/'run', canon)
    d = tmp_path/'run'/'adder-x1-y1'
    if how == 'missing':
        shutil.rmtree(d)
    else:
        (d/'result.json').write_text('{not json')
    rc, rep = audit(AA, tmp_path/'run', canon, tmp_path, how)
    assert rc == 1 and not rep['A1_truth_table_passed']
    assert len(rep['jobs']) == 4 and any(j['status'] in ('missing', 'operational_failure') for j in rep['jobs'])


def test_adder_truth_table_is_literal():
    assert AA.truth(0, 0) == [[0, 1], [0, 1], [0, 0]] and AA.truth(1, 1) == [[1, 0], [0, 1], [1, 1]]
    assert AA.truth(0, 1) == [[0, 1], [1, 0], [0, 1]] and AA.truth(1, 0) == [[1, 0], [1, 0], [0, 1]]


# ----------------------------------------------------------------- lanes
def test_lanes_positive(tmp_path, canon):
    build_lanes(tmp_path/'run', canon)
    rc, rep = audit(LA, tmp_path/'run', canon, tmp_path, 'l')
    assert rc == 0 and rep['L1_all_32_lanes_as_expected'] and rep['L2_all_paths_within_tolerance']
    assert rep['comparisons'] == 24 and rep['L2_coverage_complete'] and rep['instrument_checks_passed']


def test_lanes_sixteen_copies_rejected(tmp_path, canon):
    build_lanes(tmp_path/'good', canon)
    dup = tmp_path/'dup'
    dup.mkdir()
    for k in range(16):
        shutil.copytree(tmp_path/'good'/'lanes-00', dup/f'lanes-{k:02d}')
    rc, rep = audit(LA, dup, canon, tmp_path, 'dup')
    assert rc == 1 and not rep['L1_all_32_lanes_as_expected'] and not rep['L2_all_paths_within_tolerance']
    assert not rep['instrument_checks_passed'] and rep['comparisons'] == 0


def test_lanes_wrong_hold_controls_rejected(tmp_path, canon):
    build_lanes(tmp_path/'run', canon)
    for d in sorted((tmp_path/'run').iterdir()):
        def f(d, r):
            st = own_stage(r, 1)
            st['controls']['alpha'] += .01
            st['control_bits'] = C.literal_bits(st['controls'])
            for b in st['blocks']:
                p = d/b['path']
                with np.load(p) as z:
                    arrays = {k: z[k] for k in z.files}
                arrays['alpha'] = np.float64(st['controls']['alpha'])
                np.savez_compressed(p, **arrays)
                b.update(binding(p))
        edit_result(d, f)
    rc, rep = audit(LA, tmp_path/'run', canon, tmp_path, 'hold')
    assert rc == 1 and not rep['instrument_checks_passed']


def test_lanes_trajectory_gap_fails_l2_only(tmp_path, canon):
    build_lanes(tmp_path/'run', canon)
    d = tmp_path/'run'/'lanes-01'
    def f(d, r):
        b = own_stage(r, 0)['blocks'][0]
        p = d/b['path']
        with np.load(p) as z:
            arrays = {k: z[k] for k in z.files}
        arrays['states'][0, 0, 20] += 1e-5
        np.savez_compressed(p, **arrays)
        b.update(binding(p))
    edit_result(d, f)
    rc, rep = audit(LA, tmp_path/'run', canon, tmp_path, 'gap')
    assert rc == 1 and rep['L1_all_32_lanes_as_expected'] and not rep['L2_all_paths_within_tolerance']


def test_lanes_missing_job_fails_coverage_and_writes_receipt(tmp_path, canon):
    build_lanes(tmp_path/'run', canon)
    shutil.rmtree(tmp_path/'run'/'lanes-07')
    rc, rep = audit(LA, tmp_path/'run', canon, tmp_path, 'missing')
    assert rc == 1 and not rep['L2_coverage_complete'] and not rep['L2_all_paths_within_tolerance']
    assert len(rep['jobs']) == 16 and rep['jobs'][7]['status'] == 'missing'


# ----------------------------------------------------------------- launcher
LAUNCHER = ROOT/'tools'/'launch_full_adder_and_lane_layout.sh'


def fake_source(tmp_path):
    src = tmp_path/'src'
    src.mkdir()
    subprocess.run(['git', 'init', '-q', str(src)], check=True)
    (src/'README').write_text('fixture\n')
    subprocess.run(['git', '-C', str(src), 'add', 'README'], check=True)
    subprocess.run(['git', '-C', str(src), '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'x'],
                   check=True)
    head = subprocess.run(['git', '-C', str(src), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    return src, head


def stub(tmp_path, code):
    s = tmp_path/'stub.sh'
    s.write_text(f'#!/bin/sh\necho "$@" >> "{tmp_path}/stub-args"\nexit {code}\n')
    s.chmod(0o755)
    return s


def meminfo(tmp_path, gib):
    """A controlled memory sensor in /proc/meminfo format (the controller reads MEMINFO)."""
    f = tmp_path/f'meminfo-{gib}'
    f.write_text(f'MemTotal:       {gib * 2 * 1048576} kB\nMemAvailable:   {gib * 1048576} kB\n')
    return f


def launch(tmp_path, mode, src, head, run, py, **overrides):
    env = dict(os.environ, PY=str(py), JOB_GIB='1', RESERVE_GIB='0', MAX_JOBS='4', DEADLINE='60s', GRACE='5',
               MEMINFO=str(meminfo(tmp_path, 1024)), LAUNCH_LOCK=str(tmp_path/'launch.lock'))
    env.update(overrides)
    return subprocess.run(['bash', str(LAUNCHER), mode, str(src), head, str(run)], env=env, capture_output=True,
                          text=True, timeout=300, start_new_session=True)


def test_launcher_propagates_failures_and_lists_every_job(tmp_path):
    src, head = fake_source(tmp_path)
    cp = launch(tmp_path, 'execute', src, head, tmp_path/'run', stub(tmp_path, 7))
    assert cp.returncode == 1
    summary = json.loads((tmp_path/'run'/'launch-summary.json').read_text())
    assert len(summary['jobs']) == 20 and all(j['exit'] == 7 for j in summary['jobs'])
    assert '--preflight-only' not in (tmp_path/'stub-args').read_text()


def test_launcher_preflight_mode_and_success(tmp_path):
    src, head = fake_source(tmp_path)
    cp = launch(tmp_path, 'preflight', src, head, tmp_path/'pre', stub(tmp_path, 0))
    assert cp.returncode == 0
    args = [a for a in (tmp_path/'stub-args').read_text().splitlines() if a.startswith('-u ')]   # not the -V probe
    assert len(args) == 20 and all('--preflight-only' in a for a in args)


@pytest.mark.parametrize('case', ['existing_root', 'wrong_commit', 'dirty_source'])
def test_launcher_refusals(tmp_path, case):
    src, head = fake_source(tmp_path)
    run = tmp_path/'run'
    if case == 'existing_root':
        run.mkdir()
    if case == 'wrong_commit':
        head = 'a' * 40
    if case == 'dirty_source':
        (src/'README').write_text('changed\n')
    cp = launch(tmp_path, 'execute', src, head, run, stub(tmp_path, 0))
    assert cp.returncode == 2 and not (tmp_path/'stub-args').exists()


# ----------------------------------------------------------------- copied runner functions
def test_full_adder_runner_copies_audited_functions_unchanged():
    def funcs(path):
        tree = ast.parse(Path(path).read_text())
        return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    old, new = funcs(ROOT/'tools'/'run_public_adder.py'), funcs(ROOT/'tools'/'run_public_full_adder.py')
    for name in ('moves', 'expected', 'decode', 'controls', 'plans'):
        assert old[name] == new[name], name


# ----------------------------------------------------------------- Codex Task 8 recheck (R3-R7)
# docs/audits/2026-11-13-task8-recheck/review.md: each probe there is a rejection or preservation test here.
def planned(sid, name, gid, group, ctrl, steps, profile):
    return dict(stage_id=sid, name=name, group_id=gid, group=group, local_stage_id=sid, controls=dict(ctrl),
                control_bits=C.literal_bits(ctrl), steps=steps, profile=profile, status='not_started',
                attempted_steps=0, returned_steps=0, completed_steps=0, blocks=[], initial=None, failed_outputs=[],
                cache_comparison=None)


def old_root(root, R, check_ok=True):
    """Synthetic earlier runs in their archive layout: the toggle check's phase-0 group, the sweep's three baselines
    and Part B's shard-04. Built from the phase-0 starts so that, like the real runs, they share the P paths."""
    start = R[:, 0]
    d = root/'check'/'shard-00'
    d.mkdir(parents=True)
    st = [planned(0, 'pulse', 0, 't030-r3.5-d012-phase0', C.T, 12, 'public-toggle-check'),
          planned(1, 'hold', 0, 't030-r3.5-d012-phase0', C.H, 240, 'public-toggle-check')]
    prev = fill_stage(d, st[0], start, [1, 0], R)
    fill_stage(d, st[1], prev, [1, 0], R)
    save(d/'group-00.json', dict(group_id=0, name='t030-r3.5-d012-phase0', stage_ids=[0, 1]))
    save(d/'result.json', dict(profile='public-toggle-check', stages=st))
    manifest(d)
    d = root/'sweep'/'shard-00'
    d.mkdir(parents=True)
    st = []
    for gid, name, ctrl, bits in ((0, 'baseline-hold', C.H, [0, 1]), (1, 'baseline-write0', C.E0, [0, 0]),
                                  (2, 'baseline-write1', C.E1, [1, 1])):
        a = planned(2*gid, 'pulse', gid, name, ctrl, 120, 'public-toggle-sweep')
        b = planned(2*gid + 1, 'hold', gid, name, C.H, 240, 'public-toggle-sweep')
        prev = fill_stage(d, a, start, bits, R)
        fill_stage(d, b, prev, bits, R)
        st += [a, b]
        save(d/f'group-{gid:02d}.json', dict(group_id=gid, name=name, stage_ids=[2*gid, 2*gid + 1]))
    save(d/'result.json', dict(profile='public-toggle-sweep', stages=st))
    manifest(d)
    d = root/'partb'/'shard-04'
    d.mkdir(parents=True)
    st = [planned(i, 'pulse', i, f't030-r3.5-d006-offset-lambda{lam:.2f}', C.T, 6, 'public-toggle-offset')
          for i, lam in enumerate((0, .25, .5, .75, 1))]
    fill_stage(d, st[4], start, [1, 0], R)
    save(d/'result.json', dict(profile='public-toggle-offset', stages=st))
    manifest(d)
    return ['--check-root', str(root/'check'), '--sweep-root', str(root/'sweep'), '--part-b-run', str(root/'partb')]


def test_lanes_earlier_run_comparisons_qualify(tmp_path, canon):
    build_lanes(tmp_path/'run', canon)
    extra = old_root(tmp_path/'old', canon[0])
    rc, rep = audit(LA, tmp_path/'run', canon, tmp_path, 'cross', extra)
    assert rc == 0 and rep['L1_all_32_lanes_as_expected'] and rep['L2_all_paths_within_tolerance']
    assert [x['status'] for x in rep['against_earlier_runs']] == ['compared'] * 5
    assert all(x['max_gap'] == 0.0 for x in rep['against_earlier_runs'])
    assert [x['compared_updates'] for x in rep['against_earlier_runs']] == [252, 6, 360, 360, 252]


@pytest.mark.parametrize('k', [0, 4, 8, 12])
def test_lanes_missing_p_baseline_still_writes_receipt(tmp_path, canon, k):
    """R3: removing any P job (lanes-00 was Codex's probe) must not abort the audit."""
    build_lanes(tmp_path/'run', canon)
    extra = old_root(tmp_path/'old', canon[0])
    shutil.rmtree(tmp_path/'run'/f'lanes-{k:02d}')
    rc, rep = audit(LA, tmp_path/'run', canon, tmp_path, f'no-p{k}', extra)
    assert rc == 1 and not rep['L2_coverage_complete'] and not rep['L2_all_paths_within_tolerance']
    assert len(rep['jobs']) == 16 and rep['jobs'][k]['status'] == 'missing'
    assert sum(j['status'] == 'checked' and not j['problems'] for j in rep['jobs']) == 15
    op = LA.OPS[k // 4]
    for x in rep['against_earlier_runs']:
        uses = {'toggle': 'T', 'Part B': 'T', 'baseline-write0': 'E0', 'baseline-write1': 'E1', 'baseline-hold': 'W'}
        mine = next(v for key, v in uses.items() if key in x['earlier'])
        assert x['status'] == ('unavailable' if mine == op else 'compared')


def test_lanes_failed_p_job_is_not_a_baseline(tmp_path, canon):
    build_lanes(tmp_path/'run', canon)
    extra = old_root(tmp_path/'old', canon[0])
    edit_result(tmp_path/'run'/'lanes-12', lambda d, r: own_stage(r, 0).update(cache_comparison={'passed': False}))
    rc, rep = audit(LA, tmp_path/'run', canon, tmp_path, 'failed-p', extra)
    assert rc == 1 and not rep['instrument_checks_passed'] and len(rep['jobs']) == 16
    t = [x for x in rep['against_earlier_runs'] if 'toggle' in x['earlier'] or 'Part B' in x['earlier']]
    assert len(t) == 2 and all(x['status'] == 'unavailable' for x in t)


@pytest.mark.parametrize('fault', ['hash_mismatch', 'malformed_result', 'wrong_group', 'float32_block', 'missing_root'])
def test_lanes_bad_earlier_run_is_unavailable_and_primary_kept(tmp_path, canon, fault):
    """R6: a secondary comparison that cannot be qualified is reported unavailable; L1/L2 are untouched."""
    build_lanes(tmp_path/'run', canon)
    extra = old_root(tmp_path/'old', canon[0])
    d = tmp_path/'old'/'check'/'shard-00'
    if fault == 'hash_mismatch':
        p = d/'stage-01-block-012.npz'
        with np.load(p) as z:
            arrays = {k: z[k] for k in z.files}
        arrays['states'][0, 0, 20] += .125
        np.savez_compressed(p, **arrays)                      # stage record and manifest left unchanged
    elif fault == 'malformed_result':
        save(d/'result.json', {'stages': []})
        manifest(d)
    elif fault == 'wrong_group':
        save(d/'group-00.json', dict(group_id=0, name='t060-r3.5-d012-phase0', stage_ids=[0, 1]))
        manifest(d)
    elif fault == 'float32_block':
        r = json.loads((d/'result.json').read_text())
        b = r['stages'][1]['blocks'][3]
        p = d/b['path']
        with np.load(p) as z:
            arrays = {k: z[k] for k in z.files}
        arrays['states'] = arrays['states'].astype(np.float32)
        np.savez_compressed(p, **arrays)
        b.update(binding(p))
        save(d/'result.json', r)
        manifest(d)
    elif fault == 'missing_root':
        shutil.rmtree(tmp_path/'old'/'check')
    rc, rep = audit(LA, tmp_path/'run', canon, tmp_path, fault, extra)
    assert rc == 0 and rep['L1_all_32_lanes_as_expected'] and rep['L2_all_paths_within_tolerance']
    first = rep['against_earlier_runs'][0]
    assert first['earlier'].startswith('toggle check') and first['status'] == 'unavailable' and 'max_gap' not in first
    assert [x['status'] for x in rep['against_earlier_runs'][1:]] == ['compared'] * 4


@pytest.mark.parametrize('fault', ['duplicate_stage', 'foreign_group_stage', 'wrong_stage_name', 'missing_scaffold_stage',
                                   'other_shard_stage_started', 'other_shard_row_started', 'wrong_row_lane',
                                   'calls_reordered', 'update_steps_all_one', 'wrong_call_group'])
def test_lanes_identity_and_ledger_faults_rejected(tmp_path, canon, fault):
    """R4: the 32-stage scaffold, the rows and the call ledger are checked in order, not filtered or counted."""
    build_lanes(tmp_path/'run', canon)
    d = tmp_path/'run'/'lanes-05'

    def change(d, r):
        if fault == 'duplicate_stage':
            r['stages'].append(copy.deepcopy(own_stage(r, 1)))
        elif fault == 'foreign_group_stage':
            st = copy.deepcopy(r['stages'][0]); st.update(stage_id=99, group_id=99); r['stages'].append(st)
        elif fault == 'wrong_stage_name':
            own_stage(r, 0)['name'] = 'wrong-stage'
        elif fault == 'missing_scaffold_stage':
            del r['stages'][20]
        elif fault == 'other_shard_stage_started':
            r['stages'][20].update(status='completed', completed_steps=r['stages'][20]['steps'])
        elif fault == 'other_shard_row_started':
            r['rows'][3].update(status='completed', decoded_bit=1)
        elif fault == 'wrong_row_lane':
            r['rows'][10]['lane'] = 1
        elif fault == 'calls_reordered':
            r['calls'][4], r['calls'][5] = r['calls'][5], r['calls'][4]
        elif fault == 'update_steps_all_one':
            for c in r['calls']:
                if c['category'] == 'update':
                    c['step'] = 1
        elif fault == 'wrong_call_group':
            r['calls'][7]['group_id'] = 6
    edit_result(d, change)
    rc, rep = audit(LA, tmp_path/'run', canon, tmp_path, fault)
    assert rc == 1 and not rep['instrument_checks_passed'] and not rep['L1_all_32_lanes_as_expected']
    assert any(q.startswith('lanes-05') for q in rep['operational_or_instrument_problems'])


@pytest.mark.parametrize('fault', ['update_steps_all_one', 'calls_reordered', 'cached_swapped', 'update_not_carried',
                                   'duplicate_call_id', 'prefix_float32', 'comparison_precision_failed',
                                   'duplicate_stage_id', 'wrong_move_index'])
def test_adder_identity_and_ledger_faults_rejected(tmp_path, canon, fault):
    build_adder(tmp_path/'run', canon)
    d = tmp_path/'run'/'adder-x0-y0'

    def change(d, r):
        calls = r['calls']
        if fault == 'update_steps_all_one':
            for c in calls:
                if c['category'] == 'update':
                    c['step'] = 1
        elif fault == 'calls_reordered':
            calls[10], calls[11] = calls[11], calls[10]
        elif fault == 'cached_swapped':
            calls[1]['cached'], calls[2]['cached'] = False, True
        elif fault == 'update_not_carried':
            calls[40]['state_carried'] = False
        elif fault == 'duplicate_call_id':
            calls[8]['call'] = 8
        elif fault == 'prefix_float32':
            calls[0]['dtype_observations'] = {'norm': {'input': 'torch.float32'}}
        elif fault == 'comparison_precision_failed':
            calls[2]['precision_passed'] = False
        elif fault == 'duplicate_stage_id':
            r['stages'][3]['stage_id'] = 2
        elif fault == 'wrong_move_index':
            r['stages'][3]['move_index'] = 2
    edit_result(d, change)
    rc, rep = audit(AA, tmp_path/'run', canon, tmp_path, fault)
    assert rc == 1 and not rep['instrument_checks_passed'] and not rep['A1_truth_table_passed']


@pytest.mark.parametrize('stage_index,block_index', [(0, 0), (1, 0), (1, 7), (5, 19)])
def test_adder_reduced_precision_block_rejected(tmp_path, canon, stage_index, block_index):
    """R5: a float32 block anywhere (Codex's probe was the first interior hold block) is rejected on its own,
    with its bindings refreshed so only the precision is wrong."""
    build_adder(tmp_path/'run', canon)

    def change(d, r):
        b = r['stages'][stage_index]['blocks'][block_index]
        p = d/b['path']
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays['states'] = arrays['states'].astype(np.float32)
        np.savez_compressed(p, **arrays)
        b.update(binding(p))
    edit_result(tmp_path/'run'/'adder-x0-y0', change)
    rc, rep = audit(AA, tmp_path/'run', canon, tmp_path, 'f32')
    assert rc == 1 and not rep['instrument_checks_passed'] and not rep['A1_truth_table_passed']
    assert any('not a finite float64' in q for q in rep['operational_or_instrument_problems'])


@pytest.mark.parametrize('field,value', [('sha256', '0' * 64), ('array_sha256', '0' * 64), ('bytes', 1),
                                         ('shape', [896, 2]), ('dtype', 'float32'), ('path', 'stage-01-initial.npy')])
def test_adder_initial_binding_faults_rejected(tmp_path, canon, field, value):
    """R7: the initial state's stage-level binding is checked like the blocks' (Codex's probe: 64 zeroes)."""
    build_adder(tmp_path/'run', canon)
    edit_result(tmp_path/'run'/'adder-x0-y0', lambda d, r: r['stages'][0]['initial'].update({field: value}))
    rc, rep = audit(AA, tmp_path/'run', canon, tmp_path, 'initial')
    assert rc == 1 and not rep['instrument_checks_passed'] and not rep['A1_truth_table_passed']


def historical(tmp_path, canon):
    """A synthetic saved 11 + 6 run: the (1, 0) job's first three moves recorded the way run_public_adder.py did."""
    build_adder(tmp_path/'run', canon)
    old = tmp_path/'historical'
    shutil.copytree(tmp_path/'run'/'adder-x1-y0', old)

    def as_11_6(d, r):
        r.update(profile='public-adder', x=11, y=6)
        for st in r['stages']:
            st['profile'] = 'public-adder'
    edit_result(old, as_11_6)
    return old


def test_historical_comparison_valid(tmp_path, canon):
    old = historical(tmp_path, canon)
    rc, rep = audit(AA, tmp_path/'run', canon, tmp_path, 'hist', ['--adder-11-6', str(old)])
    h = rep['group_1_0_against_11_plus_6']
    assert rc == 0 and h['status'] == 'compared' and h['max_gap'] == 0.0 and h['compared_updates'] == 756
    assert rep['group_1_0_largest_gap_to_11_plus_6'] == 0.0
    assert sum(1 for b in rep['bound_inputs'] if 'historical' in b['path']) >= 2 + 6 + 63   # manifest, result, files


@pytest.mark.parametrize('fault', ['hash_mismatch', 'empty_stages', 'wrong_run', 'unreadable', 'float32_block'])
def test_historical_failure_is_unavailable_and_primary_kept(tmp_path, canon, fault):
    """R6: Codex's hash-mismatched block (formerly a silent gap of 0.125) and empty stage list (formerly IndexError)."""
    old = historical(tmp_path, canon)
    r = json.loads((old/'result.json').read_text())
    if fault == 'hash_mismatch':
        p = old/r['stages'][0]['blocks'][0]['path']
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays['states'][0, 0, 20] += .125
        np.savez_compressed(p, **arrays)                      # recorded digest deliberately left unchanged
    elif fault == 'empty_stages':
        save(old/'result.json', {'stages': []})
    elif fault == 'wrong_run':
        edit_result(old, lambda d, r: r.update(x=1, y=0))
    elif fault == 'unreadable':
        (old/'result.json').write_text('{not json')
    elif fault == 'float32_block':
        def f(d, r):
            b = r['stages'][1]['blocks'][2]
            p = d/b['path']
            with np.load(p, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            arrays['states'] = arrays['states'].astype(np.float32)
            np.savez_compressed(p, **arrays)
            b.update(binding(p))
        edit_result(old, f)
    rc, rep = audit(AA, tmp_path/'run', canon, tmp_path, fault, ['--adder-11-6', str(old)])
    h = rep['group_1_0_against_11_plus_6']
    assert rc == 0 and rep['A1_truth_table_passed'] and rep['instrument_checks_passed']
    assert h['status'] in ('unavailable', 'invalid') and h['problems'] and 'max_gap' not in h
    assert rep['group_1_0_largest_gap_to_11_plus_6'] is None


# ----------------------------------------------------------------- launcher: Codex Task 8 recheck R1 and R2
def counting_stub(tmp_path, seconds=.3):
    """A job that records how many jobs run at once (Codex's probe design), then sleeps briefly."""
    state = tmp_path/'active.json'
    state.write_text(json.dumps(dict(active=0, maximum=0, started=0)))
    s = tmp_path/'count-stub'
    s.write_text('#!' + sys.executable + '\n' + f'''import fcntl, json, sys, time
from pathlib import Path
if '-V' in sys.argv:
    print('Python stub'); raise SystemExit(0)
p = Path({str(state)!r})
def update(delta):
    with p.open('r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        r = json.load(f); r['active'] += delta; r['maximum'] = max(r['maximum'], r['active'])
        r['started'] += max(delta, 0); f.seek(0); json.dump(r, f); f.truncate()
update(1); time.sleep({seconds}); update(-1)
''')
    s.chmod(0o755)
    return s, state


@pytest.mark.parametrize('gib,most', [(4, 1), (6, 2)])
def test_launcher_admits_memory_centrally(tmp_path, gib, most):
    """R1: Codex's probe (4 GiB available, 2 GiB jobs, 1 GiB reserve, cap 4) admitted all four at once. Budgets not
    yet resident are now reserved, so at most (available - reserve) // job run together."""
    src, head = fake_source(tmp_path)
    py, state = counting_stub(tmp_path)
    cp = launch(tmp_path, 'execute', src, head, tmp_path/'run', py, JOB_GIB='2', RESERVE_GIB='1', MAX_JOBS='4',
                MEMINFO=str(meminfo(tmp_path, gib)))
    seen = json.loads(state.read_text())
    record = json.loads((tmp_path/'run'/'launch-record.json').read_text())
    assert cp.returncode == 0 and seen['started'] == 20 and 1 <= seen['maximum'] <= most
    assert record['initial_capacity_jobs'] == most


def test_launcher_not_admitted_when_memory_never_suffices(tmp_path):
    src, head = fake_source(tmp_path)
    py, state = counting_stub(tmp_path)
    cp = launch(tmp_path, 'execute', src, head, tmp_path/'run', py, JOB_GIB='2', RESERVE_GIB='1', ADMIT_WAIT='1',
                MEMINFO=str(meminfo(tmp_path, 2)))
    summary = json.loads((tmp_path/'run'/'launch-summary.json').read_text())
    assert cp.returncode == 1 and json.loads(state.read_text())['started'] == 0
    assert [j['state'] for j in summary['jobs']] == ['not_admitted'] * 20 and all(j['exit'] == 4 for j in summary['jobs'])


@pytest.mark.parametrize('name,value', [('MAX_JOBS', '0'), ('MAX_JOBS', '-1'), ('MAX_JOBS', 'x'), ('JOB_GIB', '0'),
                                        ('RESERVE_GIB', '-1'), ('GRACE', '0'), ('ADMIT_WAIT', '0'), ('DEADLINE', '0'),
                                        ('DEADLINE', '4x'), ('MEMINFO', '/nonexistent/meminfo')])
def test_launcher_rejects_invalid_limits(tmp_path, name, value):
    src, head = fake_source(tmp_path)
    cp = launch(tmp_path, 'execute', src, head, tmp_path/'run', stub(tmp_path, 0), **{name: value})
    assert cp.returncode == 2 and not (tmp_path/'stub-args').exists() and not (tmp_path/'run').exists()


def slow_stub(tmp_path, body='sleep 30'):
    s = tmp_path/'slow-stub'
    s.write_text(f'#!/bin/sh\nif [ "$1" = "-V" ]; then echo "Python stub"; exit 0; fi\n{body}\n')
    s.chmod(0o755)
    return s


def stop_scenario(tmp_path, py, **overrides):
    """Codex's probe: a wrapper in a private session starts a harmless sibling and the launcher; TERM goes only to
    the launcher. Returns (launcher exit status, sibling saw TERM, sibling still alive, run root)."""
    import signal
    import time
    src, head = fake_source(tmp_path)
    run, mark = tmp_path/'run', tmp_path/'sibling-terminated'
    sentinel = tmp_path/'sentinel.py'
    sentinel.write_text('import signal, time\nfrom pathlib import Path\n'
                        f'signal.signal(signal.SIGTERM, lambda *a: (Path({str(mark)!r}).write_text("TERM"), exit(0)))\n'
                        'time.sleep(60)\n')
    ids = tmp_path/'ids.json'
    wrapper = tmp_path/'wrapper.py'
    wrapper.write_text('import json, os, subprocess, sys, time\nfrom pathlib import Path\n'
                       f's = subprocess.Popen([{sys.executable!r}, {str(sentinel)!r}])\n'
                       f'l = subprocess.Popen({["bash", str(LAUNCHER), "execute", str(src), head, str(run)]!r})\n'
                       f'Path({str(ids)!r}).write_text(json.dumps(dict(group=os.getpgrp(), launcher=l.pid, sibling=s.pid)))\n'
                       f'code = l.wait()\nPath({str(tmp_path/"launcher-exit")!r}).write_text(str(code))\n'
                       'time.sleep(30)\n')
    env = dict(os.environ, PY=str(py), JOB_GIB='1', RESERVE_GIB='0', MAX_JOBS='2', DEADLINE='60s', GRACE='1',
               MEMINFO=str(meminfo(tmp_path, 1024)), LAUNCH_LOCK=str(tmp_path/'launch.lock'), **overrides)
    proc = subprocess.Popen([sys.executable, str(wrapper)], env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        deadline = time.monotonic() + 20
        while not ((run/'adder-x0-y0.started').exists() and (run/'adder-x0-y1.started').exists()) \
                and time.monotonic() < deadline:
            time.sleep(.05)
        who = json.loads(ids.read_text())
        assert who['group'] == proc.pid and who['group'] != os.getpgrp()
        os.kill(who['launcher'], signal.SIGTERM)
        deadline = time.monotonic() + 20
        while not (tmp_path/'launcher-exit').exists() and time.monotonic() < deadline:
            time.sleep(.05)
        time.sleep(.3)
        try:
            os.kill(who['sibling'], 0)
            alive = True
        except ProcessLookupError:
            alive = False
        return int((tmp_path/'launcher-exit').read_text()), mark.exists(), alive, run
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)         # this session exists only for the fixture
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


@pytest.mark.parametrize('body,stopped_exit', [('sleep 30', 143), ("trap '' TERM; sleep 30", 137)])
def test_launcher_stop_reaches_only_its_jobs_and_records_every_job(tmp_path, body, stopped_exit):
    """R2: TERM to the launcher leaves an unrelated sibling untouched, stops its own running jobs (KILL after GRACE
    for a job that ignores TERM), and writes a record for every scheduled job and the summary."""
    status, sibling_saw_term, sibling_alive, run = stop_scenario(tmp_path, slow_stub(tmp_path, body))
    summary = json.loads((run/'launch-summary.json').read_text())
    states = [j['state'] for j in summary['jobs']]
    assert status == 143 and not sibling_saw_term and sibling_alive and summary['stopped_by'] == 'SIGTERM'
    assert states[:2] == ['stopped', 'stopped'] and states[2:] == ['not_started'] * 18
    assert [j['exit'] for j in summary['jobs'][:2]] == [stopped_exit] * 2
    for name in ('adder-x0-y0', 'adder-x0-y1'):
        rec = json.loads((run/f'{name}.json').read_text())
        with pytest.raises(ProcessLookupError):
            os.killpg(rec['pgid'], 0)                   # the job's whole group is gone
    assert all((run/f'{n}.json').exists() for n in ('adder-x1-y0', 'lanes-15'))


def test_launcher_deadline_terminates_jobs(tmp_path):
    src, head = fake_source(tmp_path)
    cp = launch(tmp_path, 'execute', src, head, tmp_path/'run', slow_stub(tmp_path), DEADLINE='1s', GRACE='1',
                MAX_JOBS='20')
    summary = json.loads((tmp_path/'run'/'launch-summary.json').read_text())
    assert cp.returncode == 1 and all(j['state'] == 'timed_out' and j['exit'] == 143 for j in summary['jobs'])


def test_launcher_ends_descendants_left_by_a_job(tmp_path):
    src, head = fake_source(tmp_path)
    marker = tmp_path/'orphans'
    body = f'sleep 300 &\necho $! >> {marker}\nexit 0'
    cp = launch(tmp_path, 'execute', src, head, tmp_path/'run', slow_stub(tmp_path, body), MAX_JOBS='20')
    summary = json.loads((tmp_path/'run'/'launch-summary.json').read_text())
    assert cp.returncode == 0 and all(j['state'] == 'finished' and j['exit'] == 0 for j in summary['jobs'])
    for pid in map(int, marker.read_text().split()):
        try:
            os.kill(pid, 0)
            alive = Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
        except ProcessLookupError:
            alive = False
        assert not alive


# ----------------------------------------------------------------- launcher: the single-lane job set
def test_launcher_single_lane_job_set(tmp_path):
    src, head = fake_source(tmp_path)
    env = dict(os.environ, PY=str(stub(tmp_path, 0)), JOB_GIB='1', RESERVE_GIB='0', MAX_JOBS='4', DEADLINE='60s',
               GRACE='5', MEMINFO=str(meminfo(tmp_path, 1024)), LAUNCH_LOCK=str(tmp_path/'launch.lock'))
    cp = subprocess.run(['bash', str(LAUNCHER), 'preflight', str(src), head, str(tmp_path/'pre'), 'single-lane'],
                        env=env, capture_output=True, text=True, timeout=300, start_new_session=True)
    summary = json.loads((tmp_path/'pre'/'launch-summary.json').read_text())
    args = [a for a in (tmp_path/'stub-args').read_text().splitlines() if a.startswith('-u ')]
    assert cp.returncode == 0 and summary['job_set'] == 'single-lane'
    assert [j['job'] for j in summary['jobs']] == [f'single-{k:02d}' for k in range(8)]
    assert len(args) == 8 and all('tools/run_public_lane_singleton.py' in a and '--shards 8' in a
                                  and '--preflight-only' in a for a in args)


def test_launcher_refuses_an_unknown_job_set(tmp_path):
    src, head = fake_source(tmp_path)
    env = dict(os.environ, PY=str(stub(tmp_path, 0)), MEMINFO=str(meminfo(tmp_path, 1024)),
               LAUNCH_LOCK=str(tmp_path/'launch.lock'))
    cp = subprocess.run(['bash', str(LAUNCHER), 'execute', str(src), head, str(tmp_path/'run'), 'everything'],
                        env=env, capture_output=True, text=True, timeout=60, start_new_session=True)
    assert cp.returncode == 2 and not (tmp_path/'run').exists() and not (tmp_path/'stub-args').exists()


def test_launcher_refuses_a_second_controller(tmp_path):
    """Admission is per controller, so a second launch while one holds the lock is refused before it starts."""
    import fcntl
    src, head = fake_source(tmp_path)
    lock = open(tmp_path/'launch.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)        # stands in for a running controller
    try:
        cp = launch(tmp_path, 'execute', src, head, tmp_path/'run', stub(tmp_path, 0))
        assert cp.returncode == 2 and 'another launch controller' in cp.stderr
        assert not (tmp_path/'run').exists() and not (tmp_path/'stub-args').exists()
    finally:
        lock.close()
    cp = launch(tmp_path, 'preflight', src, head, tmp_path/'run', stub(tmp_path, 0))
    assert cp.returncode == 0


# ----------------------------------------------------------------- Codex Task 8/9 recheck, N1: saved per-call gates
GATES = ('input_unchanged', 'cache_unchanged', 'expanded_cache_valid', 'precision_passed', 'outputs_valid')


def set_gate(calls, category, field, value, stage_id=None):
    """Change one call's saved gate: a value, or 'missing' to drop it."""
    c = next(c for c in calls if c['category'] == category and (stage_id is None or c['stage_id'] == stage_id))
    if value == 'missing':
        c.pop(field, None)
    else:
        c[field] = value


@pytest.mark.parametrize('category', ['comparison', 'update'])
@pytest.mark.parametrize('value', [False, 'missing'])
@pytest.mark.parametrize('field', GATES)
def test_adder_failed_or_missing_call_gate_rejected(tmp_path, canon, field, value, category):
    build_adder(tmp_path/'run', canon)
    edit_result(tmp_path/'run'/'adder-x0-y0', lambda d, r: set_gate(r['calls'], category, field, value, stage_id=3))
    rc, rep = audit(AA, tmp_path/'run', canon, tmp_path, 'gate')
    assert rc == 1 and not rep['instrument_checks_passed'] and not rep['A1_truth_table_passed']
    assert any('call ledger' in q for q in rep['operational_or_instrument_problems'])


@pytest.mark.parametrize('category', ['comparison', 'update'])
@pytest.mark.parametrize('value', [False, 'missing'])
@pytest.mark.parametrize('field', GATES)
def test_lanes_failed_or_missing_call_gate_rejected(tmp_path, canon, field, value, category):
    build_lanes(tmp_path/'run', canon)
    edit_result(tmp_path/'run'/'lanes-00', lambda d, r: set_gate(r['calls'], category, field, value))
    rc, rep = audit(LA, tmp_path/'run', canon, tmp_path, 'gate')
    assert rc == 1 and not rep['instrument_checks_passed'] and not rep['L1_all_32_lanes_as_expected']


def test_pair_call_claiming_the_scalar_path_rejected(tmp_path, canon):
    build_adder(tmp_path/'run', canon)
    edit_result(tmp_path/'run'/'adder-x0-y0', lambda d, r: r['calls'][40].update(scalar_path=True))
    rc, rep = audit(AA, tmp_path/'run', canon, tmp_path, 'scalar')
    assert rc == 1 and not rep['instrument_checks_passed']


def test_ledger_layout_is_explicit():
    plan = [dict(stage_id=0, stage='s', group_id=0, group='g', steps=2)]
    calls = returned(C.expected_calls(plan, 'p'))
    problems = []
    assert C.check_call_ledger(calls, plan, 'p', problems, 'x', layout='pair') == 5 and not problems
    C.check_call_ledger(calls, plan, 'p', problems, 'x', layout='single')         # pair evidence, single layout
    assert problems
    with pytest.raises(TypeError):
        C.check_call_ledger(calls, plan, 'p', [], 'x')                             # no default layout
    with pytest.raises(ValueError):
        C.check_call_ledger(calls, plan, 'p', [], 'x', layout='batch')
