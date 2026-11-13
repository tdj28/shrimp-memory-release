from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import run_public_path_comparison as p


def test_fixed_controls_counts_and_equal_exposures():
    groups, stages, calls = p.plans()
    assert [len(g['stage_ids']) for g in groups] == [2, 2, 5]
    assert len(calls) == 2907
    assert sum(s['steps'] for s in stages) == 2880
    assert [s['steps'] for s in stages] == [240, 240, 960, 240, 240, 240, 240, 240, 240]
    assert stages[0]['controls'] == stages[2]['controls'] == stages[7]['controls'] == p.controls(.78125)
    assert all(stages[g['stage_ids'][-1]]['controls'] == p.controls(.75) for g in groups)
    assert sum(stages[i]['steps'] for i in groups[1]['stage_ids']) == sum(stages[i]['steps'] for i in groups[2]['stage_ids'])
    assert [s['path_fraction'] for s in stages[4:]] == [.7578125, .765625, .7734375, .78125, .75]
    assert all(s['controls']['alpha'] == p.H['alpha'] for s in stages)


def test_no_configurable_search():
    for args in [('other', 240), (p.PROFILE, 480)]:
        with pytest.raises(ValueError):
            p.plans(*args)


def test_actual_pinned_p_source_when_available():
    base = p.ROOT.parent/'shrimp-memory-data/audits/public-root-continuation/20261029T110830Z-public-root-continuation'
    if not base.is_dir():
        pytest.skip('external pinned input bundle is not installed')
    reference, provenance = p.qualify_inputs(SimpleNamespace(
        cycle_packet=base/'cycle-packet.json', cycle_states=base/'cycle-states.npz',
        source_audit=base/'source-audit.json'))
    with np.load(base/'cycle-states.npz', allow_pickle=False) as source:
        assert reference[:, 0].tobytes() == source['source_terminal_states'].tobytes()
    assert provenance['original_history_order'] == [0, 1]


@pytest.mark.parametrize('gap,status', [(1e-8, 'same_within_tolerance'),
                                      (5e-5, 'intermediate'), (.1, 'distinct')])
def test_observer_preserves_intermediate_band(gap, status):
    from test_public_command_memory import refs
    reference = refs()
    states = np.tile(reference.transpose(1, 0, 2), (3, 1, 1))
    states[:, 1, 1] = gap
    observed = p.observe_stage(states, reference, p.plans()[1][0])
    assert observed['both_primitive_period_six']
    assert observed['pair_distance'] == pytest.approx(gap)
    assert observed['pair_status'] == status


def test_actual_serializer_and_carries(tmp_path, monkeypatch):
    from test_public_mechanism import Fake, fake_valid
    from test_public_command_memory import refs
    m = p.mechanism
    monkeypatch.setattr(m.memory.os, 'fsync', lambda _: None)
    monkeypatch.setattr(m.batch, 'VOCAB', 4)
    monkeypatch.setattr(m.batch, 'valid_cache', fake_valid)
    args = SimpleNamespace(mode=p.PROFILE, dwell=240, output_dir=tmp_path/'run')
    result = p.toggle.run_fixed(args, profile=p.PROFILE, backend_factory=Fake,
                   qualifier=lambda _: (refs(), {}), rss=lambda: 0, install_signals=False,
                   plan_factory=p.plans, design_factory=p.design, observe_stage=p.observe_stage,
                   require_runtime=False)
    assert result['status'] == 'completed', result.get('traceback')
    assert result['model_calls'] == 2907 and result['initializations'] == 3
    assert len(result['rows']) == 6
    assert [row['original_history_index'] for row in result['rows']] == [0, 1]*3
    assert all(len(row['readout']['all_reference_alignments']) == 2 for row in result['rows'])
    assert all((args.output_dir/f'row-{i:02d}-completed.json').is_file() for i in range(6))
    for group in result['groups']:
        previous = refs()[:, 0].copy()
        for sid in group['stage_ids']:
            states, _, _ = m.prior.reconstruct(args.output_dir, result['stages'][sid])
            assert np.array_equal(states[0], previous)
            previous = states[-1]
    assert (args.output_dir/'manifest.json').is_file()
