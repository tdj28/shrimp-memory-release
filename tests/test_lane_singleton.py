"""Single-lane arm: the runner end to end with a fake one-state backend, and its audit. No model is loaded.

The fake moves a state along synthetic period-6 cycles (coordinate 0 is the phase, coordinate 1 the cycle: 0 or .1),
like the fakes of the pair runner tests: H keeps the cycle, E0 and E1 set it, the toggle flips it once per stage.
It accepts only one state at a time and, like the real scalar path, returns no expanded-cache evidence.
"""
import copy
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import run_public_lane_singleton as S  # noqa: E402
import task8_audit_common as C  # noqa: E402
import audit_public_lane_singleton as SA  # noqa: E402
sys.path.insert(0, str(ROOT/'tests'))
from test_public_batch_writes import OBS, caches, fake_valid  # noqa: E402
from test_public_command_memory import refs  # noqa: E402
import test_task8_revised as F  # noqa: E402

CALLS = {'W': 258, 'E0': 366, 'E1': 366, 'T': 258}


class FakeScalar:
    shapes_seen = set()

    def __init__(self, args, recorder, budget):
        self.recorder, self.budget = recorder, budget
        self.observations = copy.deepcopy(OBS)
        self.cache, self.controls, self.flipped = caches(), None, False

    def prepare(self, controls):
        self.controls = dict(controls)
        self.budget.claim('native64_prefix')
        self.cache, self.observations, self.flipped = caches(controls['alpha']), copy.deepcopy(OBS), False

    def cache_arrays(self):
        return {k: v.copy() for k, v in self.cache.items()}

    def forward(self, state, cached):
        FakeScalar.shapes_seen.add(state.shape)
        if state.shape != (1, 896):
            raise AssertionError('the single-lane arm must send one state')
        slot = self.budget.slots[self.budget.calls - 1]
        y = state.copy()
        y[0, 0] = (y[0, 0] + 1) % 6
        if self.controls == C.E0:
            y[0, 1] = 0.
        elif self.controls == C.E1:
            y[0, 1] = .1
        elif self.controls == C.T and slot['category'] == 'update' and not self.flipped:
            y[0, 1] = .1 - y[0, 1]
            self.flipped = True
        elif self.controls not in (C.H, C.T):
            raise AssertionError(f'unexpected controls {self.controls}')
        p = np.zeros((1, 4), dtype=np.float64)
        p[0, 1] = 1.
        self.observations = copy.deepcopy(OBS)
        return y, p, 1e-3, {'input_after': state.copy(), 'observed_output_dtype': 'torch.float64',
                            'observed_probability_dtype': 'torch.float64', 'input_exact': True}


@pytest.fixture(autouse=True)
def cheap(monkeypatch):
    monkeypatch.setattr(S.prior.memory.os, 'fsync', lambda fd: None)
    monkeypatch.setattr(S.batch, 'VOCAB', 4)
    monkeypatch.setattr(S.batch, 'valid_cache', fake_valid)


def run_shard(root, k, backend=FakeScalar, preflight=False):
    args = SimpleNamespace(output_dir=root/f'single-{k:02d}', shard=k, shards=8, preflight_only=preflight)
    return S.run(args, backend_factory=backend, qualifier=lambda a: (refs(), {}), rss=lambda: 0, require_runtime=False)


def test_plan_is_the_lane_check_operations_one_state_at_a_time():
    groups, stages, rows = S.plans()
    assert [g['name'] for g in groups] == [f'{op}-S{c}' for op in ('W', 'E0', 'E1', 'T') for c in (0, 1)]
    assert [g['expected_bit'] for g in groups] == [0, 1, 0, 0, 1, 1, 1, 0]
    assert len(stages) == 16 and len(rows) == 8 and all(r['lane'] == 0 for r in rows)
    calls = sum(len(S.sweep.shard_calls(groups, stages, [k])) for k in range(8))
    assert calls == 2496


@pytest.mark.parametrize('k', range(8))
def test_runner_end_to_end(tmp_path, k):
    result = run_shard(tmp_path, k)
    g = result['groups'][0]
    assert result['status'] == 'completed' and result['model_calls'] == CALLS[g['operation']]
    assert g['decoded_bit'] == g['expected_bit'] and g['matches_expected'] and g['diagnostics_passed']
    assert FakeScalar.shapes_seen == {(1, 896)}
    own = [c for c in result['calls']]
    assert all(c['status'] == 'returned' for c in own)
    assert all(c['scalar_path'] is True for c in own if c['category'] != 'prefix')
    for sid in g['stage_ids']:
        x, ids, _ = S.reconstruct(tmp_path/f'single-{k:02d}', result['stages'][sid])
        assert x.shape == (result['stages'][sid]['steps'] + 1, 1, 896) and ids.shape[1] == 1


def test_preflight_makes_no_call(tmp_path):
    result = run_shard(tmp_path, 3, preflight=True)
    assert result['status'] == 'preflight_passed' and result['model_calls'] == 0


def test_a_batch_expansion_is_refused(tmp_path):
    class Expanded(FakeScalar):
        def forward(self, state, cached):
            y, p, s, extra = super().forward(state, cached)
            if cached:
                extra['used_cache_before'] = {'key': np.repeat(self.cache['key'], 2, axis=0)}
            return y, p, s, extra
    result = run_shard(tmp_path, 0, backend=Expanded)
    assert result['status'] == 'instrument_failed'


def test_a_two_lane_return_is_refused(tmp_path):
    class Pair(FakeScalar):
        def forward(self, state, cached):
            y, p, s, extra = super().forward(state, cached)
            return np.repeat(y, 2, axis=0), np.repeat(p, 2, axis=0), s, extra
    result = run_shard(tmp_path, 1, backend=Pair)
    assert result['status'] == 'instrument_failed'


def test_wrong_shard_count_is_refused(tmp_path):
    with pytest.raises(ValueError):
        S.run(SimpleNamespace(output_dir=tmp_path/'x', shard=0, shards=16, preflight_only=True),
              backend_factory=FakeScalar, qualifier=lambda a: (refs(), {}), rss=lambda: 0, require_runtime=False)


# ----------------------------------------------------------------------------------------------- the audit
def finished_runs(root, canon_inputs):
    """Run all eight shards with the fake, then add the environment records a real run writes."""
    for k in range(8):
        run_shard(root, k)
        d = root/f'single-{k:02d}'
        F.save(d/'backend-readiness.json', {'mode': 'native64', 'torch_threads': 1, 'torch_interop_threads': 1,
                                            'thread_environment': {v: '1' for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                                                                    'OPENBLAS_NUM_THREADS',
                                                                                    'NUMEXPR_NUM_THREADS')}})
        F.save(d/'model-inputs.json', canon_inputs)
        src = json.loads((d/'source.json').read_text())
        src.update(source_commit=F.FROZEN, working_tree_status='')
        F.save(d/'source.json', src)
        r = json.loads((d/'result.json').read_text())
        r['source'] = src
        F.save(d/'result.json', r)
        F.manifest(d)
    return root


@pytest.fixture(scope='module')
def base(tmp_path_factory):
    """One set of eight finished fake runs and a matching two-lane run, copied by each audit test."""
    root = tmp_path_factory.mktemp('singleton-base')
    R = refs()
    np.savez(root/'canonical-refs.npz', states=R)
    inputs = {k: {'sha256': k * 8} for k in ('config', 'vectors', 'weights')}
    F.save(root/'canonical-inputs.json', inputs)
    canon = (R, inputs, root/'canonical-refs.npz', root/'canonical-inputs.json')
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S.prior.memory.os, 'fsync', lambda fd: None)
        mp.setattr(S.batch, 'VOCAB', 4)
        mp.setattr(S.batch, 'valid_cache', fake_valid)
        finished_runs(root/'run', inputs)
    build_matching_lanes(root/'lanes', canon)
    return root, canon


def copy_runs(base, tmp_path, lanes=False):
    root, canon = base
    shutil.copytree(root/'run', tmp_path/'run')
    if lanes:
        shutil.copytree(root/'lanes', tmp_path/'lanes')
    return tmp_path/'run', canon


def audit(root, canon, tmp_path, extra=()):
    out = tmp_path/'single-audit.json'
    rc = SA.main(['--run', str(root), '--canonical-references', str(canon[2]), '--canonical-model-inputs',
                  str(canon[3]), '--frozen-commit', F.FROZEN, '--output', str(out), *extra])
    return rc, json.loads(out.read_text())


def test_audit_positive(base, tmp_path):
    root, canon = copy_runs(base, tmp_path)
    rc, rep = audit(root, canon, tmp_path)
    assert rc == 0 and rep['instrument_checks_passed'] and rep['S1_all_8_lanes_as_expected']
    assert rep['against_two_lane_runs']['status'] == 'not_requested'


@pytest.mark.parametrize('fault', ['missing_job', 'float32_block', 'two_lane_block', 'wrong_start', 'steps_all_one',
                                   'other_shard_started', 'not_scalar_path', 'wrong_initial_digest', 'wrong_row_bit'])
def test_audit_faults_rejected_with_receipt(base, tmp_path, fault):
    root, canon = copy_runs(base, tmp_path)
    d = root/'single-05'                                 # E1 from cycle 1

    def change(d, r):
        own = [r['stages'][i] for i in r['groups'][0]['stage_ids']]
        if fault in ('float32_block', 'two_lane_block'):
            b = own[1]['blocks'][3]
            p = d/b['path']
            with np.load(p, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            arrays['states'] = (arrays['states'].astype(np.float32) if fault == 'float32_block'
                                else np.repeat(arrays['states'], 2, axis=1))
            np.savez(p, **arrays)
            b.update(F.binding(p))
        elif fault == 'steps_all_one':
            for c in r['calls']:
                if c['category'] == 'update':
                    c['step'] = 1
        elif fault == 'other_shard_started':
            r['stages'][0].update(status='completed', completed_steps=12)
        elif fault == 'not_scalar_path':
            r['calls'][5]['scalar_path'] = False
        elif fault == 'wrong_initial_digest':
            own[0]['initial']['array_sha256'] = '0' * 64
        elif fault == 'wrong_row_bit':
            r['rows'][5]['decoded_bit'] = 1 - r['rows'][5]['decoded_bit']
    if fault == 'missing_job':
        shutil.rmtree(d)
    elif fault == 'wrong_start':
        np.save(d/'group-005-start.npy', refs()[0, 0][None])   # cycle 0, not the planned cycle 1
        F.manifest(d)
    else:
        F.edit_result(d, change)
    rc, rep = audit(root, canon, tmp_path)
    assert rc == 1 and not rep['instrument_checks_passed'] and not rep['S1_all_8_lanes_as_expected']
    assert len(rep['jobs']) == 8


def test_audit_against_two_lane_runs(base, tmp_path):
    """S2: each single-lane path against all four two-lane occurrences (same synthetic dynamics: gap zero)."""
    root, canon = copy_runs(base, tmp_path, lanes=True)
    lanes = tmp_path/'lanes'
    rc, rep = audit(root, canon, tmp_path, ['--lane-run', str(lanes)])
    s2 = rep['against_two_lane_runs']
    assert rc == 0 and s2['status'] == 'compared' and s2['comparisons'] == 32 and s2['largest_gap'] == 0.0
    assert rep['S2_all_within_tolerance'] and s2['bitwise_identical'] == 32
    shutil.rmtree(lanes/'lanes-12')                     # a missing P job: S2 incomplete, S1 and the receipt kept
    rc, rep = audit(root, canon, tmp_path, ['--lane-run', str(lanes)])
    assert rc == 1 and rep['against_two_lane_runs']['status'] == 'incomplete' and rep['S1_all_8_lanes_as_expected']
    assert not rep['S2_all_within_tolerance']


def test_audit_detects_a_single_lane_divergence(base, tmp_path):
    root, canon = copy_runs(base, tmp_path, lanes=True)
    d = tmp_path/'lanes'/'lanes-01'                     # W from (1, 0): lane 1 starts on cycle 0

    def nudge(d, r):
        b = LR_own(r, 1)['blocks'][10]
        p = d/b['path']
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays['states'][0, 1, 300] += 1e-9
        np.savez_compressed(p, **arrays)
        b.update(F.binding(p))
    F.edit_result(d, nudge)
    rc, rep = audit(root, canon, tmp_path, ['--lane-run', str(tmp_path/'lanes')])
    s2 = rep['against_two_lane_runs']
    assert rc == 1 and s2['status'] == 'compared' and not rep['S2_all_within_tolerance']
    assert s2['largest_gap'] == pytest.approx(1e-9)


def LR_own(r, local):
    return r['stages'][r['groups'][0]['stage_ids'][local]]


def build_matching_lanes(root, canon):
    """Two-lane runs with the same synthetic dynamics as FakeScalar, built by the pair fixture."""
    R, inputs, _, _ = canon
    F.build_lanes_with(root, canon, path_of=lambda op, c, steps_move: synthetic_path(op, c, steps_move))


def synthetic_path(op, c, steps_move):
    """FakeScalar's path from cycle c: phase advances by one; the cycle coordinate follows the operation."""
    x = refs()[c, 0].copy()
    out = [x.copy()]
    for t in range(steps_move + 240):
        x = x.copy()
        x[0] = (x[0] + 1) % 6
        if t < steps_move:
            if op == 'E0':
                x[1] = 0.
            elif op == 'E1':
                x[1] = .1
            elif op == 'T' and t == 0:
                x[1] = .1 - x[1]
        out.append(x)
    return np.stack(out)


# ----------------------------------------------------------------- Codex Task 8/9 recheck, N1: saved per-call gates
@pytest.mark.parametrize('category', ['comparison', 'update'])
@pytest.mark.parametrize('value', [False, 'missing'])
@pytest.mark.parametrize('field', ['input_unchanged', 'cache_unchanged', 'scalar_path', 'precision_passed',
                                   'outputs_valid'])
def test_audit_failed_or_missing_call_gate_rejected(base, tmp_path, field, value, category):
    root, canon = copy_runs(base, tmp_path)

    def change(d, r):
        c = next(c for c in r['calls'] if c['category'] == category)
        if value == 'missing':
            c.pop(field, None)
        else:
            c[field] = value
    F.edit_result(root/'single-02', change)
    rc, rep = audit(root, canon, tmp_path)
    assert rc == 1 and not rep['instrument_checks_passed'] and not rep['S1_all_8_lanes_as_expected']


@pytest.mark.parametrize('value', [True, False, 'missing'])
def test_audit_requires_an_explicit_absence_of_batch_expansion(base, tmp_path, value):
    """The single-lane evaluator records expanded_cache_valid = None; any other value, or none, is rejected."""
    root, canon = copy_runs(base, tmp_path)

    def change(d, r):
        c = next(c for c in r['calls'] if c['category'] == 'update')
        if value == 'missing':
            c.pop('expanded_cache_valid')
        else:
            c['expanded_cache_valid'] = value
    F.edit_result(root/'single-06', change)
    rc, rep = audit(root, canon, tmp_path)
    assert rc == 1 and not rep['instrument_checks_passed']


def test_real_runner_records_every_gate(tmp_path):
    """The fake-backend run's own ledger carries the complete single-lane gate set on every call."""
    result = run_shard(tmp_path, 6)
    calls = [c for c in result['calls'] if c['category'] != 'prefix']
    assert calls and all(c['input_unchanged'] is True and c['cache_unchanged'] is True and c['scalar_path'] is True
                         and 'expanded_cache_valid' in c and c['expanded_cache_valid'] is None for c in calls)
