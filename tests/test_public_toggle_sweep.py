"""Model-free contracts for the fixed public toggle sweep (docs/public-toggle-sweep.md)."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import run_public_toggle_sweep as s
import test_public_toggle_pilot as tp
from test_public_command_memory import refs


class FakeBackend(tp.FakeBackend):
    """Scripted file-contract control (pulse stage sets each lane's cycle), not a numerical prediction."""
    pairs = {0: None, 1: (0, 0), 2: (1, 1), 50: (1, 0), 53: (0, 1), 100: (1, 1)}

    def forward(self, state, cached):
        slot = self.budget.slots[self.budget.calls-1]
        y, probs, t, extra = super().forward(state, cached)
        if slot['stage_id'] % 2 == 0:
            y[:, 1] = state[:, 1]
            pair = self.pairs.get(slot['group_id'])
            if pair is not None:
                y[:, 1] = np.asarray(pair)*.1
        return y, probs, t, extra


@pytest.fixture(autouse=True)
def compact(monkeypatch):
    monkeypatch.setattr(s.memory.os, 'fsync', lambda _fd: None)
    monkeypatch.setattr(s.batch, 'VOCAB', 4)
    monkeypatch.setattr(s.batch, 'valid_cache', tp.fake_valid)


def execute(path, shard, shards, backend=FakeBackend):
    return s.run(SimpleNamespace(output_dir=path, preflight_only=False, shard=shard, shards=shards),
                 backend_factory=backend, qualifier=lambda _: (refs(), {'synthetic': True}), rss=lambda: 0,
                 install_signals=False, require_runtime=False)


def test_fixed_candidate_list_and_ledger():
    groups, stages, rows = s.plans()
    assert len(groups) == 3 + 12*4*3 == 147 and len(stages) == len(rows) == 294
    total = sum(len(s.shard_calls(groups, stages, [g])) for g in range(len(groups)))
    assert total == 45162
    h, e0, e1 = s.HOLD, s.ENDPOINTS[0], s.ENDPOINTS[1]
    assert groups[0]['controls'] == h and groups[1]['controls'] == e0 and groups[2]['controls'] == e1
    # radius 1 at 0 and 90 degrees reproduces the audited write endpoints
    r0 = next(g for g in groups if g['name'] == 'sweep-t000-r1.0-d120')['controls']
    r90 = next(g for g in groups if g['name'] == 'sweep-t090-r1.0-d120')['controls']
    assert r0['alpha'] == pytest.approx(e1['alpha'], abs=1e-15) and r0['log10_temperature'] == h['log10_temperature']
    assert r90['log10_temperature'] == pytest.approx(e0['log10_temperature'], abs=1e-15)
    assert all(stages[2*g['group_id']+1]['controls'] == h and stages[2*g['group_id']+1]['steps'] == 240 for g in groups)
    assert [r['source_reference_index'] for r in rows] == [0, 1]*147
    assert len({g['name'] for g in groups}) == 147


@pytest.mark.parametrize('shards', [1, 7, 16])
def test_shards_partition_groups_exactly(shards):
    ids = [g for k in range(shards) for g in s.shard_group_ids(147, k, shards)]
    assert sorted(ids) == list(range(147))
    with pytest.raises(ValueError):
        s.shard_group_ids(147, shards, shards)


def test_shard_run_classifies_and_closes(tmp_path):
    result = execute(tmp_path/'shard0', 0, 50)
    assert result['status'] == 'completed', result.get('traceback')
    assert [g['group_id'] for g in result['groups']] == [0, 50, 100]
    assert [g['mapping'] for g in result['groups']] == ['identity', 'toggle', 'reset1']
    assert result['baseline_qualified'] is True and result['toggle_candidate_group_ids'] == [50]
    assert result['outcome'] == 'toggle_observed'
    assert result['model_calls'] == len(result['calls']) == sum(g['model_calls'] for g in result['groups'])
    tp.verify_manifest(tmp_path/'shard0')
    design = json.loads((tmp_path/'shard0'/'design.json').read_text())
    assert design['shard_group_ids'] == [0, 50, 100] and design['all_calls'] == 45162


def test_probe_only_shard_has_no_baseline_verdict(tmp_path):
    result = execute(tmp_path/'shard3', 3, 50)
    assert result['status'] == 'completed', result.get('traceback')
    assert [g['group_id'] for g in result['groups']] == [3, 53, 103]
    assert result['baseline_qualified'] is None
    assert [g['mapping'] for g in result['groups']] == ['identity', 'identity', 'identity']
    assert result['outcome'] == 'no_toggle_observed'


def test_preflight_loads_no_model(tmp_path):
    result = s.run(SimpleNamespace(output_dir=tmp_path/'pre', preflight_only=True, shard=0, shards=16),
                   backend_factory=FakeBackend, qualifier=lambda _: (refs(), {'synthetic': True}), rss=lambda: 0,
                   install_signals=False, require_runtime=False)
    assert result['status'] == 'preflight_passed' and result['model_calls'] == 0 and not result['model_loaded']
    assert all(g['status'] == 'not_run' and g['closed'] for g in result['groups'])


def test_check_specs_follow_stage_1b_rules(tmp_path):
    import run_public_toggle_check as c
    specs = c.check_specs([dict(theta=30, radius=3.5, dwell=12)])
    assert len(specs) == 19
    roles = [x['role'] for x in specs]
    assert roles.count('phase') == 6 and roles.count('neighbour') == 8 and roles.count('dwell') == 2
    assert roles.count('rounding') == 3
    assert sorted(x['pulse'] for x in specs if x['role'] == 'dwell') == [8, 16]
    r = refs()
    a = c.start_state(r, 2, None); b = c.start_state(r, 0, 1)
    assert a.tobytes() == r[:, 2].tobytes()
    assert 0 < np.max(np.abs(b - r[:, 0])) <= 2e-15*np.max(np.abs(r[:, 0]))  # 1+1e-15 rounds to 1+1.11e-15


def test_check_shard_runs(tmp_path):
    import run_public_toggle_check as c
    cand = tmp_path/'cand.json'; cand.write_text(json.dumps(dict(candidates=[dict(theta=30, radius=3.5, dwell=12)])))
    result = c.run(SimpleNamespace(output_dir=tmp_path/'chk', preflight_only=False, shard=0, shards=10,
                                   candidates=cand),
                   backend_factory=FakeBackend, qualifier=lambda _: (refs(), {'synthetic': True}), rss=lambda: 0,
                   install_signals=False, require_runtime=False)
    assert result['status'] == 'completed', result.get('traceback')
    assert [g['group_id'] for g in result['groups']] == [0, 10]
    tp.verify_manifest(tmp_path/'chk')


def test_xor_chain_expected_and_fake_run(tmp_path):
    import run_public_xor_chain as x
    assert x.expected() == [[1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1]]

    class Toggle(FakeBackend):
        """Scripted: a toggle stage swaps the two lanes' cycle marker; idle and hold stages keep it."""
        def forward(self, state, cached):
            slot = self.budget.slots[self.budget.calls-1]
            y, probs, t, extra = tp.FakeBackend.forward(self, state, cached)
            y[:, 1] = state[:, 1]
            st = self.budget.stages[slot['stage_id']]
            if st['name'].endswith('-toggle') and slot['category'] == 'update' and slot['step'] == 1:
                y[:, 1] = .1 - state[:, 1]
            return y, probs, t, extra
    tog = tmp_path/'t.json'; tog.write_text(json.dumps(dict(theta=30, radius=3.5, dwell=12)))
    r = x.run(SimpleNamespace(output_dir=tmp_path/'xor', toggle=tog), backend_factory=Toggle,
              qualifier=lambda _: (refs(), {'synthetic': True}), rss=lambda: 0, install_signals=False,
              require_runtime=False)
    assert r['status'] == 'completed', r.get('traceback')
    assert r['outcome'] == 'xor_chain_matched' and r['matched'] == 8
    tp.verify_manifest(tmp_path/'xor')


def test_adder_schedule_and_fake_run(tmp_path):
    import run_public_adder as ad
    assert [m['move'] for m in ad.moves()] == ['T', 'W', 'T', 'T', 'T', 'E1', 'W', 'T', 'T', 'T', 'W', 'T']
    want = ad.expected()
    assert [w[0] for w in want] == [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
    assert [w[1] for w in want] == [0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1]
    assert ad.decode(want) == [17, 18]

    class Ideal(FakeBackend):
        """Scripted ideal register: T swaps the lane marker, E0/E1 set it, W and holds keep it."""
        def forward(self, state, cached):
            slot = self.budget.slots[self.budget.calls-1]
            y, probs, t, extra = tp.FakeBackend.forward(self, state, cached)
            y[:, 1] = state[:, 1]
            st = self.budget.stages[slot['stage_id']]
            if slot['category'] == 'update' and slot['step'] == 1 and not st['name'].endswith('-hold'):
                if st['move'] == 'T':
                    y[:, 1] = .1 - state[:, 1]
                elif st['move'] in ('E0', 'E1'):
                    y[:, 1] = .1 * int(st['move'][1])
            return y, probs, t, extra
    r = ad.run(SimpleNamespace(output_dir=tmp_path/'add'), backend_factory=Ideal,
               qualifier=lambda _: (refs(), {'synthetic': True}), rss=lambda: 0, install_signals=False, require_runtime=False)
    assert r['status'] == 'completed', r.get('traceback')
    assert r['outcome'] == 'adder_matched' and r['numbers'] == [17, 18]
    tp.verify_manifest(tmp_path/'add')
