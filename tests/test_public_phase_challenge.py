"""Literal two-phase starts, independent history scope and bounded raw contracts."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import run_public_phase_challenge as pilot
sys.path.pop(0)
from test_public_control_tolerance import FakeBackend
from test_public_batch_writes import fake_valid, historical as old_history, HOLD, ENDPOINT, ZERO_ENDPOINT
from test_public_command_memory import refs


def history(args, cycles, plan=None):
    all_values, metadata = old_history(args, cycles, plan)
    index = 1-plan.requested_bit
    trial = (1,2)[plan.requested_bit]
    for source, control in enumerate(metadata['controls']):
        control.update(source_index=source, trial_id=plan.historical_trials[source])
    return all_values[index], dict(source_commit=metadata['source_commit'], selected_trial_id=trial,
        selected_source_index=index, selected_source_phase=0,
        selected_control=metadata['controls'][index], original_qualification=metadata)


@pytest.fixture(autouse=True)
def cheap_files(monkeypatch):
    monkeypatch.setattr(pilot.memory.os, 'fsync', lambda fd:None)
    now = pilot.reference.utc_now(); monkeypatch.setattr(pilot.reference, 'utc_now', lambda:now)
    monkeypatch.setattr(pilot.batch, 'VOCAB', 4)
    monkeypatch.setattr(pilot.batch, 'valid_cache', fake_valid)


def run(tmp_path, bit=1, backend=FakeBackend, **kwargs):
    args = SimpleNamespace(output_dir=tmp_path/'run', requested_bit=bit)
    return pilot.run(args, backend_factory=backend, qualifier=lambda args:(refs(), {}),
        historical_qualifier=history, rss=lambda:0, install_signals=False, **kwargs)


def validate_files(path, result):
    assert json.loads((path/'result.json').read_text()) == result
    assert len(result['rows']) == 4 and len(result['groups']) == 2 and len(result['stages']) == 4
    assert len(result['calls']) == 732
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
        assert not ({'events.jsonl','result.json','manifest.json',marker.name} & set(names))
        assert all(item in manifest['files'] for item in group['artifacts'])
        expected = {p.name for p in path.iterdir() if any(p.name.startswith(f'stage-{i:02d}-')
            for i in group['stage_ids'])}
        expected |= {slot['artifact']['path'] for slot in result['calls']
            if slot['group_id'] == group['group_id'] and 'artifact' in slot}
        expected |= {f'row-{i:02d}-completed.json' for i in group['row_ids']}
        assert set(names) == expected


@pytest.mark.parametrize('bit', [0,1])
def test_complete_phase_pair_literal_resets_carry_and_row0_history(tmp_path, bit):
    originals = copy.deepcopy((pilot.batch.CONTROLS, pilot.memory.writes.ENDPOINTS,
        pilot.prior.MAX_CALLS, pilot.shared.MAX_CALLS, pilot.shared.GROUPS))
    result = run(tmp_path, bit); path = tmp_path/'run'
    assert result['status'] == 'completed' and result['nominal_equivalence_passed']
    assert result['model_calls'] == 732 and result['call_counts'] == dict(prefix=4,comparison=8,update=720)
    assert result['completed_batch_updates'] == 720 and result['completed_row_updates'] == 1440
    assert result['reference_initializations'] == 2 and result['between_group_resets'] == 1
    assert result['outcome_counts'] == dict(passed=4,failed=0,unresolved=0,not_evaluated=0)
    assert all(call['status'] == 'returned' for call in result['calls'])
    expected_initial = refs()[1-bit,:2]
    phase = json.loads((path/'phase-inputs.json').read_text())
    assert phase['batch_initial_sha256'] == pilot.prior.array_hash(expected_initial)
    for i, slot in enumerate(phase['initial_rows']):
        assert slot['source_reference_index'] == 1-bit and slot['source_phase'] == i
        assert slot['array_sha256'] == pilot.prior.array_hash(refs()[1-bit,i])
    assert phase['initial_rows'][0]['array_sha256'] != phase['initial_rows'][1]['array_sha256']
    for group in result['groups']:
        endpoint, hold = [result['stages'][i] for i in group['stage_ids']]
        left, _, _ = pilot.prior.reconstruct(path, endpoint)
        right, _, _ = pilot.prior.reconstruct(path, hold)
        assert left[0].tobytes() == expected_initial.tobytes()
        assert left[-1].tobytes() == right[0].tobytes()
        assert hold['controls'] == hold['nominal_controls'] == HOLD
        assert endpoint['nominal_controls'] == (ZERO_ENDPOINT if bit == 0 else ENDPOINT)
        for key, delta in group['endpoint_delta'].items():
            assert endpoint['controls'][key] == endpoint['nominal_controls'][key]+delta
        for i in group['row_ids']:
            row = result['rows'][i]
            assert row['source_reference_index'] == 1-bit and row['source_phase'] == row['batch_index']
            assert row['public_action']['stages'][0]['alpha'] == endpoint['nominal_controls']['alpha']
            assert row['realized_stages'][0]['controls'] == endpoint['controls']
            assert row['dispatch_count'] == 1 and row['numerical_diagnostics_passed']
            if i == 0:
                assert row['historical_comparison']['passed']
                assert row['historical_comparison']['historical_trial_id'] == (1,2)[bit]
            else: assert row['historical_comparison'] is None
        if group['group_id']:
            assert endpoint['historical_comparison'] is None and hold['historical_comparison'] is None
        else:
            assert len(endpoint['historical_comparison']['rows']) == 1
            assert endpoint['historical_comparison']['scope'] == 'nominal_phase0_only'
    assert originals == (pilot.batch.CONTROLS,pilot.memory.writes.ENDPOINTS,
        pilot.prior.MAX_CALLS,pilot.shared.MAX_CALLS,pilot.shared.GROUPS)
    for collection in ('rows','groups','stages','calls'):
        frozen = json.loads((path/f'planned-{collection}.json').read_text())
        assert all(item['profile'] == pilot.PROFILE for item in frozen)
    validate_files(path, result)


@pytest.mark.parametrize('bit,old_group', [(0,3),(1,1)])
def test_full_offset_fixed_from_original_stencil_not_refinement(bit, old_group):
    original = pilot.shared.plans(bit, 'axis-stencil')
    current = pilot.plans(bit)
    assert [s['controls'] for s in current[2][:2]] == [s['controls'] for s in original[2][:2]]
    assert [s['controls'] for s in current[2][2:]] == [s['controls'] for s in original[2][2*old_group:2*old_group+2]]
    assert [g['name'] for g in current[1]] == ['nominal','full-negative-offset']
    assert all(row['outcome'] == 'not_evaluated' for row in current[3])


@pytest.mark.parametrize('fault,expected', [('wrong','failed'),('unresolved','unresolved'),('finite-diagnostic','passed')])
def test_finite_nominal_failures_continue_fixed_offset_group(tmp_path, fault, expected):
    result = run(tmp_path, backend=type('Fault',(FakeBackend,),{'group_fault':(0,fault)}))
    assert result['status'] == 'completed' and result['model_calls'] == 732
    assert [r['outcome'] for r in result['rows'][:2]] == [expected,expected]
    assert [r['outcome'] for r in result['rows'][2:]] == ['passed','passed']
    assert not result['nominal_equivalence_passed']
    if fault == 'finite-diagnostic':
        assert not result['groups'][0]['diagnostics_passed']
        assert all(not r['numerical_diagnostics_passed'] for r in result['rows'][:2])
    validate_files(tmp_path/'run', result)


@pytest.mark.parametrize('call',[1,8,124,366,367,368,732])
def test_failure_preserves_all_phase_slots_and_closed_groups(tmp_path, call):
    result = run(tmp_path, backend=type('Failure',(FakeBackend,),{'fail_at':call}))
    assert result['status'] == 'failed' and result['model_calls'] == call
    assert result['calls'][call-1]['status'] == 'failed'
    assert all(c['status'] == 'not_started' for c in result['calls'][call:])
    assert sum(s['completed_steps'] for s in result['stages']) == sum(c.get('state_carried',False) for c in result['calls'])
    completed = (call-1)//366
    assert sum(g['status'] == 'completed' for g in result['groups']) == completed
    assert all(r['outcome'] == 'not_evaluated' for r in result['rows'][2*completed:])
    validate_files(tmp_path/'run', result)


@pytest.mark.parametrize('fault',['nonfinite','precision','cache','expanded','cache-inspect'])
def test_structural_failure_retains_partial_blocks(tmp_path, fault):
    result = run(tmp_path, backend=type('Structural',(FakeBackend,),{'fault':fault}))
    assert result['status'] == 'failed' and result['model_calls'] == 8
    assert result['stages'][0]['completed_steps'] == 4
    if fault == 'cache-inspect': assert result['calls'][7]['forward_attempted'] is False
    else: assert result['stages'][0]['failed_outputs']
    validate_files(tmp_path/'run', result)


def test_phase1_is_never_compared_with_phase0_history():
    plan = pilot.prior.WritePlan(1); old, _ = history(None, refs(), plan)
    saved = np.stack([old[0]['states'], old[0]['states']+123.], axis=1)
    tokens = np.stack([old[0]['token_ids'], old[0]['token_ids']+10], axis=1)
    compared = pilot.compare_phase0(saved, tokens, old, 0, 1)
    assert compared['passed'] and len(compared['rows']) == 1
    saved[4,0,3] += 1e-5
    assert not pilot.compare_phase0(saved, tokens, old, 0, 1)['passed']


@pytest.mark.parametrize('bit',[0,1])
def test_history_wrapper_selects_literal_trial_and_rejects_mislabelling(monkeypatch, bit):
    plan = pilot.prior.WritePlan(bit)
    values, metadata = old_history(None, refs(), plan)
    for i, row in enumerate(metadata['controls']): row.update(source_index=i,trial_id=plan.historical_trials[i])
    monkeypatch.setattr(pilot.prior, 'qualify_historical', lambda *a,**kw:(values,metadata))
    selected, proof = pilot.qualify_phase_history(None, refs(), plan)
    assert selected is values[1-bit] and proof['selected_trial_id'] == (1,2)[bit]
    assert proof['original_qualification'] is metadata
    metadata['controls'][1-bit]['source_index'] = bit
    with pytest.raises(ValueError,match='phase-zero'): pilot.qualify_phase_history(None,refs(),plan)


@pytest.mark.parametrize('bit',[0,1])
def test_zero_forward_preflight_binds_saved_phase_bytes(tmp_path, bit):
    saved = tmp_path/'references.npy'; np.save(saved, refs())
    digest = hashlib.sha256(saved.read_bytes()).hexdigest()
    args = SimpleNamespace(requested_bit=bit)
    def qualify(args):
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == digest
        return np.load(saved,allow_pickle=False),{'binding':pilot.memory.binding(saved)}
    result = pilot.preflight(args, qualifier=qualify, historical_qualifier=history)
    assert result['status'] == 'passed' and result['model_calls'] == 0 and not result['model_loaded']
    assert result['reference_initial_state_sha256'] == pilot.prior.array_hash(refs()[1-bit,:2])
    assert [r['source_phase'] for r in result['initial_rows']] == [0,1]
    saved.write_bytes(b'changed')
    with pytest.raises(AssertionError): pilot.preflight(args,qualifier=qualify,historical_qualifier=history)


def test_initial_file_failure_does_not_claim_runtime_initialization(tmp_path, monkeypatch):
    original = pilot.Recorder.state
    def fail(self,name,value,**kwargs):
        if name == 'stage-00-initial.npy': raise OSError('initial save failed')
        return original(self,name,value,**kwargs)
    monkeypatch.setattr(pilot.Recorder,'state',fail)
    result = run(tmp_path)
    assert result['model_calls'] == result['reference_initializations'] == result['between_group_resets'] == 0
    assert result['terminal_state_sha256'] == pilot.prior.array_hash(refs()[0,:2])
    assert all(row['status'] == 'not_started' for row in result['rows'])
    validate_files(tmp_path/'run',result)


def test_marker_failure_keeps_completed_rows_but_not_false_closed_group(tmp_path, monkeypatch):
    original = pilot.Recorder.json
    def fail(self,name,value,final=False):
        if name == 'group-00.json': raise OSError('marker save failed')
        return original(self,name,value,final)
    monkeypatch.setattr(pilot.Recorder,'json',fail)
    result = run(tmp_path)
    assert result['status'] == result['groups'][0]['status'] == 'failed'
    assert result['model_calls'] == 366
    assert all(r['status'] == 'completed' for r in result['rows'][:2])
    assert all(r['status'] == 'not_started' for r in result['rows'][2:])
    validate_files(tmp_path/'run',result)


def test_timeout_after_return_preserves_actual_state(tmp_path):
    clock = [0.]
    class Timeout(FakeBackend):
        def forward(self,state,cached):
            output = super().forward(state,cached)
            if self.budget.calls == 8: clock[0] = 900.
            return output
    result = run(tmp_path,backend=Timeout,clock=lambda:clock[0])
    assert result['status'] == 'incomplete' and result['model_calls'] == 8
    assert result['error'].startswith('TimeoutError:')
    assert result['stages'][0]['completed_steps'] == 5
    assert result['calls'][7]['state_carried']
    assert not result['resource_limits_passed']
    validate_files(tmp_path/'run',result)


def test_budget_schedule_deadline_and_artifact_reserve(tmp_path):
    _,_,stages,_,slots = pilot.plans(0)
    clock=[0.]; recorder=pilot.Recorder(tmp_path/'budget',clock=lambda:clock[0],rss=lambda:0)
    try:
        budget=pilot.Budget(recorder,slots,0)
        changed=copy.deepcopy(stages[0]); changed['controls']['alpha'] += 1e-9
        with pytest.raises(ValueError,match='realized-control'): budget.begin(changed)
        budget.begin(stages[0]); slots[0]['profile']='axis-stencil'
        with pytest.raises(RuntimeError,match='fixed schedule'): budget.claim('prefix')
        slots[0]['profile']=pilot.PROFILE;budget.claim('prefix')
        clock[0]=900.
        with pytest.raises(TimeoutError,match='900-second'):budget.check()
        design=pilot.derive_design(0)
        assert design['schedule']['total_calls'] == 732
        assert design['limits']['internal_seconds'] == 900 and design['limits']['external_seconds'] == 1080
        assert design['limits']['total_raw_bytes'] == 160*1024**2
        assert design['artifact_envelope']['estimated_uncompressed_envelope_bytes'] == 57254784
        assert 57254784+2473984+16384 < pilot.WORKING_OUTPUT
        recorder.size=lambda:pilot.WORKING_OUTPUT
        with pytest.raises(RuntimeError,match='raw artifact'):recorder.admit(1)
        recorder.admit(1,final=True)
    finally: recorder.stream.close()


def test_existing_output_directory_is_immutable(tmp_path):
    (tmp_path/'run').mkdir(); sentinel=tmp_path/'run'/'keep';sentinel.write_text('immutable')
    with pytest.raises(FileExistsError):run(tmp_path)
    assert sentinel.read_text() == 'immutable' and list((tmp_path/'run').iterdir()) == [sentinel]


@pytest.mark.parametrize('bit',[-1,2,True,1.,None])
def test_invalid_bit_refused_before_any_output(tmp_path, bit):
    with pytest.raises(ValueError):run(tmp_path,bit=bit)
    assert not (tmp_path/'run').exists()
