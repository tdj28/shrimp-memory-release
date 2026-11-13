"""Control provenance, immutable return contracts and failure retention without models."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import run_public_control_tolerance as pilot
sys.path.pop(0)
from test_public_batch_writes import FakeBackend as OldFake, fake_valid, historical, HOLD, ENDPOINT, ZERO_ENDPOINT
from test_public_command_memory import refs


class FakeBackend(OldFake):
    """Synthetic six-cycle with a bit controlled only during the endpoint stage."""
    group_fault = None

    def __init__(self, args, recorder, budget):
        for name in ('planned-groups.json', 'planned-stages.json', 'planned-rows.json',
                     'planned-calls.json', 'frozen-rows.json'):
            assert (recorder.directory/name).is_file()
        super().__init__(args, recorder, budget)
        self.requested_bit = args.requested_bit

    def forward(self, state, cached):
        slot = self.budget.slots[self.budget.calls-1]
        actual = self.controls
        self.controls = (ZERO_ENDPOINT if self.requested_bit == 0 else ENDPOINT) if slot['stage_id']%2 == 0 else HOLD
        previous_fault = self.fault
        if self.group_fault is not None and slot['group_id'] == self.group_fault[0]: self.fault = self.group_fault[1]
        try: return super().forward(state, cached)
        finally: self.controls = actual; self.fault = previous_fault


@pytest.fixture(autouse=True)
def cheap_files(monkeypatch):
    monkeypatch.setattr(pilot.memory.os, 'fsync', lambda fd:None)
    now = pilot.reference.utc_now(); monkeypatch.setattr(pilot.reference, 'utc_now', lambda:now)
    monkeypatch.setattr(pilot.batch, 'VOCAB', 4)
    monkeypatch.setattr(pilot.batch, 'valid_cache', fake_valid)


def run(tmp_path, bit=1, backend=FakeBackend, profile=None, **kwargs):
    args = SimpleNamespace(output_dir=tmp_path/'run', requested_bit=bit)
    if profile is not None: args.profile = profile
    return pilot.run(args, backend_factory=backend,
        qualifier=lambda args:(refs(), {}), historical_qualifier=historical,
        rss=lambda:0, install_signals=False, **kwargs)


def validate_files(path, result):
    assert json.loads((path/'result.json').read_text()) == result
    assert len(result['rows']) == 10 and len(result['groups']) == 5 and len(result['stages']) == 10
    assert len(result['calls']) == 1830
    manifest = json.loads((path/'manifest.json').read_text())
    assert {b['path'] for b in manifest['files']} == {p.name for p in path.iterdir()}-{'manifest.json'}
    for item in manifest['files']:
        data = (path/item['path']).read_bytes()
        assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
    for stage in result['stages']:
        if stage['trajectory']:
            states, tokens, times = pilot.prior.reconstruct(path, stage)
            assert len(states) == stage['completed_steps']+1
            assert len(tokens) == len(times) == stage['completed_steps']
    for row in result['rows']:
        assert json.loads((path/f'row-{row["row_id"]:02d}.json').read_text()) == row
    for group in result['groups']:
        marker = path/f'group-{group["group_id"]:02d}.json'
        if group['status'] != 'completed':
            assert not marker.exists(); continue
        assert json.loads(marker.read_text()) == group
        names = [item['path'] for item in group['artifacts']]
        assert names == sorted(set(names))
        assert not ({'events.jsonl', 'result.json', 'manifest.json', marker.name} & set(names))
        for item in group['artifacts']:
            assert item in manifest['files']
        stage_ids = group['stage_ids']
        expected = {p.name for p in path.iterdir() if any(p.name.startswith(f'stage-{i:02d}-') for i in stage_ids)}
        expected |= {slot['artifact']['path'] for slot in result['calls']
            if slot['group_id'] == group['group_id'] and 'artifact' in slot}
        expected |= {f'row-{i:02d}-completed.json' for i in group['row_ids']}
        assert set(names) == expected


@pytest.mark.parametrize('bit', [0,1])
@pytest.mark.parametrize('profile', pilot.PROFILES)
def test_all_groups_reset_carry_and_preserve_nominal_vs_realized_controls(tmp_path, bit, profile):
    original_globals = (copy.deepcopy(pilot.batch.CONTROLS), pilot.prior.MAX_CALLS,
        copy.deepcopy(pilot.memory.writes.ENDPOINTS))
    result = run(tmp_path, bit, profile=profile); directory = tmp_path/'run'
    assert result['status'] == 'completed' and result['nominal_equivalence_passed']
    assert result['call_counts'] == dict(prefix=10, comparison=20, update=1800)
    assert result['model_calls'] == 1830 and result['completed_row_updates'] == 3600
    assert result['reference_initializations'] == 5 and result['between_group_resets'] == 4
    assert result['outcome_counts'] == dict(passed=10, failed=0, unresolved=0, not_evaluated=0)
    assert all(slot['status'] == 'returned' for slot in result['calls'])
    assert result['profile'] == result['source']['arguments']['profile'] == profile
    design = json.loads((directory/'design.json').read_text())
    assert design['profile'] == profile
    for collection in ('groups', 'stages', 'rows', 'calls'):
        assert all(item['profile'] == profile for item in result[collection])
        assert all(item['profile'] == profile for item in json.loads((directory/f'planned-{collection}.json').read_text()))
    seen_initials = []
    for group in result['groups']:
        endpoint, hold = [result['stages'][i] for i in group['stage_ids']]
        left, _, _ = pilot.prior.reconstruct(directory, endpoint)
        right, _, _ = pilot.prior.reconstruct(directory, hold)
        assert left[0].tobytes() == refs()[:,0].tobytes()
        assert left[-1].tobytes() == right[0].tobytes()
        seen_initials.append(endpoint['initial']['path'])
        assert hold['controls'] == HOLD and hold['nominal_controls'] == HOLD
        assert endpoint['nominal_controls'] == (ZERO_ENDPOINT if bit == 0 else ENDPOINT)
        for key, delta in group['endpoint_delta'].items():
            assert endpoint['controls'][key] == endpoint['nominal_controls'][key]+delta
        for i in group['row_ids']:
            row = result['rows'][i]
            assert row['public_action']['stages'][0]['alpha'] == endpoint['nominal_controls']['alpha']
            assert row['realized_stages'][0]['controls'] == endpoint['controls']
            assert row['dispatch_count'] == 1 and row['parser_outcome'] == 'accepted'
            assert row['numerical_diagnostics_passed']
            if group['group_id']:
                assert row['historical_comparison'] is None
                assert endpoint['historical_comparison'] is None and hold['historical_comparison'] is None
            else: assert row['historical_comparison']['passed']
        if profile == 'negative-axis-refinement':
            fraction = (0., .25, .5, .75, 1.)[group['group_id']]
            assert group['fraction'] == fraction
            assert group['endpoint_delta'] == dict(alpha=-fraction*1e-5 if bit else 0.,
                log10_temperature=0. if bit else -fraction*1e-4)
            assert group['name'] == ('fraction-0','fraction-025','fraction-050','fraction-075','fraction-100')[group['group_id']]
    assert len(set(seen_initials)) == 5
    assert original_globals == (pilot.batch.CONTROLS, pilot.prior.MAX_CALLS, pilot.memory.writes.ENDPOINTS)
    validate_files(directory, result)


@pytest.mark.parametrize('bit,old_group', [(0,3), (1,1)])
def test_refinement_controls_match_prior_nominal_and_failed_offset_without_requiring_failure(bit, old_group):
    original = pilot.plans(bit)
    refined = pilot.plans(bit, 'negative-axis-refinement')
    assert original == pilot.plans(bit, 'axis-stencil')
    assert [stage['controls'] for stage in refined[2][:2]] == [stage['controls'] for stage in original[2][:2]]
    assert [stage['controls'] for stage in refined[2][-2:]] == [stage['controls'] for stage in original[2][2*old_group:2*old_group+2]]
    assert all(row['outcome'] == 'not_evaluated' for row in refined[3])
    assert refined[1][-1]['status'] == 'not_started'


@pytest.mark.parametrize('fault,expected', [('wrong','failed'), ('unresolved','unresolved'), ('finite-diagnostic','passed')])
def test_bad_scientific_group_retained_without_selecting_later_groups(tmp_path, fault, expected):
    result = run(tmp_path, backend=type('Fault', (FakeBackend,), {'group_fault':(1,fault)}))
    assert result['status'] == 'completed' and result['model_calls'] == 1830
    assert [r['outcome'] for r in result['rows'][2:4]] == [expected,expected]
    assert all(r['outcome'] == 'passed' for r in result['rows'][4:])
    if fault == 'finite-diagnostic': assert not result['groups'][1]['diagnostics_passed']
    validate_files(tmp_path/'run', result)


def test_wrong_nominal_control_does_not_skip_perturbed_groups(tmp_path):
    result = run(tmp_path, backend=type('NominalWrong', (FakeBackend,), {'group_fault':(0,'wrong')}))
    assert result['status'] == 'completed' and result['model_calls'] == 1830
    assert result['nominal_equivalence_passed'] is False
    assert [row['outcome'] for row in result['rows'][:2]] == ['failed','failed']
    assert all(row['outcome'] == 'passed' for row in result['rows'][2:])
    validate_files(tmp_path/'run', result)


@pytest.mark.parametrize('call', [1,2,8,123,124,366,367,490,732,733,1465,1830])
def test_failed_execution_retains_all_slots_and_closed_return_markers(tmp_path, call):
    result = run(tmp_path, backend=type('Failure', (FakeBackend,), {'fail_at':call}))
    assert result['status'] == 'failed' and result['model_calls'] == call
    assert result['calls'][call-1]['status'] == 'failed'
    assert all(s['status'] == 'not_started' for s in result['calls'][call:])
    assert sum(s['completed_steps'] for s in result['stages']) == sum(s.get('state_carried',False) for s in result['calls'])
    completed_groups = (call-1)//366
    assert sum(g['status'] == 'completed' for g in result['groups']) == completed_groups
    assert all(r['outcome'] == 'not_evaluated' for r in result['rows'][2*completed_groups:])
    validate_files(tmp_path/'run', result)


def test_refinement_late_failure_keeps_complete_control_and_unstarted_fraction_slots(tmp_path):
    result = run(tmp_path, profile='negative-axis-refinement',
        backend=type('FailureAfterNominal', (FakeBackend,), {'fail_at':368}))
    assert result['status'] == 'failed' and result['model_calls'] == 368
    assert result['groups'][0]['status'] == 'completed'
    assert all(row['outcome'] == 'passed' for row in result['rows'][:2])
    assert all(row['outcome'] == 'not_evaluated' for row in result['rows'][2:])
    assert all(slot['profile'] == 'negative-axis-refinement' for slot in result['calls'])
    assert all(slot['status'] == 'not_started' for slot in result['calls'][368:])
    validate_files(tmp_path/'run', result)


@pytest.mark.parametrize('fault', ['nonfinite','dtype','shape','precision','input','cache','expanded','length','missing-extra'])
def test_structural_return_not_admitted_and_failed_arrays_preserved(tmp_path, fault):
    result = run(tmp_path, backend=type('Structural', (FakeBackend,), {'fault':fault}))
    assert result['status'] == 'failed' and result['model_calls'] == 8
    assert result['stages'][0]['completed_steps'] == 4
    assert result['stages'][0]['failed_outputs']
    assert all(row['outcome'] == 'not_evaluated' for row in result['rows'])
    validate_files(tmp_path/'run', result)


def test_preforward_cache_inspection_consumes_slot_but_retains_previous_state(tmp_path):
    result = run(tmp_path, backend=type('Inspect', (FakeBackend,), {'fault':'cache-inspect'}))
    assert result['calls'][7]['forward_attempted'] is False
    assert result['stages'][0]['completed_steps'] == 4
    validate_files(tmp_path/'run', result)


@pytest.mark.parametrize('profile', pilot.PROFILES)
def test_zero_forward_preflight_uses_actual_saved_contract_and_rejects_corruption(tmp_path, profile):
    saved = tmp_path/'references.npy'; np.save(saved, refs())
    expected = hashlib.sha256(saved.read_bytes()).hexdigest()
    args = SimpleNamespace(requested_bit=0, profile=profile)
    def qualifier(args):
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == expected
        return np.load(saved, allow_pickle=False), {'binding':pilot.memory.binding(saved)}
    result = pilot.preflight(args, qualifier=qualifier, historical_qualifier=historical)
    assert result['status'] == 'passed' and result['model_calls'] == 0 and not result['model_loaded']
    assert result['profile'] == result['design']['profile'] == profile
    assert len(result['design']['realized_stages']) == 10
    saved.write_bytes(b'changed')
    with pytest.raises(AssertionError): pilot.preflight(args, qualifier=qualifier, historical_qualifier=historical)


def test_reconstruct_rejects_changed_realized_control_even_if_file_hash_rebound(tmp_path):
    result = run(tmp_path)
    stage = copy.deepcopy(result['stages'][2]); record = stage['blocks'][0]
    path = tmp_path/'run'/record['path']
    with np.load(path, allow_pickle=False) as source: arrays = {k:source[k].copy() for k in source.files}
    arrays['alpha'] += 1e-6
    with path.open('wb') as stream: np.savez_compressed(stream, **arrays)
    stage['blocks'][0] = pilot.memory.binding(path)
    with pytest.raises(ValueError, match='invalid/gapped block'): pilot.prior.reconstruct(tmp_path/'run', stage)


def test_existing_output_directory_refused_before_overwrite(tmp_path):
    (tmp_path/'run').mkdir(); sentinel = tmp_path/'run'/'keep'; sentinel.write_text('immutable')
    with pytest.raises(FileExistsError): run(tmp_path)
    assert sentinel.read_text() == 'immutable' and list((tmp_path/'run').iterdir()) == [sentinel]


def test_marker_write_failure_preserves_completed_rows_without_false_completed_group(tmp_path, monkeypatch):
    original = pilot.Recorder.json
    def fail_marker(self, name, value, final=False):
        if name == 'group-00.json': raise OSError('injected marker write failure')
        return original(self, name, value, final)
    monkeypatch.setattr(pilot.Recorder, 'json', fail_marker)
    result = run(tmp_path)
    assert result['status'] == 'failed' and result['model_calls'] == 366
    assert result['groups'][0]['status'] == 'failed'
    assert all(row['status'] == 'completed' and row['outcome'] == 'passed' for row in result['rows'][:2])
    assert all(row['status'] == 'not_started' for row in result['rows'][2:])
    validate_files(tmp_path/'run', result)


def test_budget_rejects_hold_change_and_checks_real_cap(tmp_path):
    plan, groups, stages, rows, slots = pilot.plans(0)
    recorder = pilot.Recorder(tmp_path/'budget', clock=lambda:0., rss=lambda:0)
    budget = pilot.Budget(recorder, slots, 0)
    corrupted = copy.deepcopy(stages[0]); corrupted['controls']['alpha'] += 1e-9
    with pytest.raises(ValueError, match='realized-control'): budget.begin(corrupted)
    budget.begin(stages[0]); budget.claim('prefix')
    with pytest.raises(RuntimeError, match='fixed schedule'): budget.claim('prefix')
    recorder.stream.close()
    design = pilot.derive_design(0)
    assert design['limits']['internal_seconds'] == 1800 and design['limits']['total_raw_bytes'] == 160*1024**2
    assert design['artifact_envelope']['estimated_uncompressed_envelope_bytes']+pilot.prior.FAILED_RETURN_BYTES+16384 < pilot.WORKING_OUTPUT


def test_nominal_controls_cannot_hide_cross_profile_stage_or_slot_injection(tmp_path):
    original, refinement = pilot.plans(0), pilot.plans(0, 'negative-axis-refinement')
    recorder = pilot.Recorder(tmp_path/'budget', clock=lambda:0., rss=lambda:0)
    try:
        with pytest.raises(ValueError, match='profile schedule'):
            pilot.Budget(recorder, original[4], 0, 'negative-axis-refinement')
        slots = refinement[4]
        budget = pilot.Budget(recorder, slots, 0, 'negative-axis-refinement')
        assert original[2][0]['controls'] == refinement[2][0]['controls']
        with pytest.raises(ValueError, match='realized-control'):
            budget.begin(original[2][0])
        budget.begin(refinement[2][0])
        for key, corrupt in [('profile','axis-stencil'), ('group','nominal'), ('step',1), ('cached',True)]:
            old = slots[0][key]; slots[0][key] = corrupt
            with pytest.raises(RuntimeError, match='fixed schedule'): budget.claim('prefix')
            assert budget.calls == 0
            slots[0][key] = old
        budget.claim('prefix')
        assert budget.calls == 1
    finally: recorder.stream.close()


@pytest.mark.parametrize('profile', ['', 'refinement', None, True, ['axis-stencil']])
def test_invalid_profile_rejected_before_output_or_qualification(tmp_path, profile):
    args = SimpleNamespace(output_dir=tmp_path/'run', requested_bit=0, profile=profile)
    def forbidden(args): raise AssertionError('input loading must not occur')
    with pytest.raises(ValueError, match='profile'): pilot.run(args, qualifier=forbidden, install_signals=False)
    with pytest.raises(ValueError, match='profile'): pilot.preflight(args, qualifier=forbidden)
    assert not args.output_dir.exists()


@pytest.mark.parametrize('bit', [-1,2,True,1.,None])
def test_invalid_bit_rejected_without_any_output(tmp_path, bit):
    with pytest.raises(ValueError): run(tmp_path, bit=bit)
    assert not (tmp_path/'run').exists()
