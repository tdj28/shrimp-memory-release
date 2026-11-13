"""Model-free contracts for a small fixed toggle panel and shared failure guard."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import run_public_toggle_pilot as p
from test_public_batch_writes import OBS, caches, fake_valid
from test_public_command_memory import refs
NATIVE_VALID_CACHE = p.batch.valid_cache


class FakeBackend:
    """Scripted file-contract control, not a numerical prediction of a toggle."""
    fail_at = None
    error = OSError
    fault = None
    native = False

    def __init__(self, args, recorder, budget):
        self.recorder, self.budget = recorder, budget
        self.observations = copy.deepcopy(OBS)
        self.controls = None
        self.cache = self.make_cache()

    def make_cache(self):
        if not self.native:
            return caches()
        values = {f'layer_{i:02d}_{kind}': np.full((1,2,37,64), i/100., dtype=np.float64)
                  for i in range(24) for kind in ('key','value')}
        values['length'] = np.asarray(37, dtype=np.int64)
        return values

    def prepare(self, controls):
        self.controls = controls.copy(); self.budget.claim('native64_prefix')
        if self.budget.calls == self.fail_at:
            raise self.error('injected prefix failure')
        self.observations = copy.deepcopy(OBS); self.cache = self.make_cache()

    def cache_arrays(self):
        return {k: v.copy() for k,v in self.cache.items()}

    def forward(self, state, cached):
        if self.budget.calls == self.fail_at:
            raise self.error('injected forward failure')
        slot = self.budget.slots[self.budget.calls-1]
        group = slot['group_id']; stage = slot['stage_id']
        y = state.copy(); y[:,0] = (y[:,0]+1)%6
        if stage % 2 == 0:
            pairs = {1: (0,0), 2: (1,1), 3: (1,0), 4: (0,1), 5: (0,0),
                     6: (1,1), 7: (0,1), 8: (0,1), 9: (0,0), 10: (1,1)}
            if group in pairs:
                y[:,1] = np.asarray(pairs[group])*.1
            if group == 7:
                y[:,2] = .02
        probs = np.zeros((2,p.batch.VOCAB), dtype=np.float64); probs[:,1] = 1.
        extra = dict(input_after=state.copy(), observed_output_dtype='torch.float64',
                     observed_probability_dtype='torch.float64', input_exact=True)
        if cached:
            extra.update(used_cache_before={k: np.repeat(v,2,axis=0) for k,v in self.cache.items() if k != 'length'},
                         used_cache_length_after=38)
        self.observations = copy.deepcopy(OBS)
        if self.fault == 'comparison' and not cached:
            y[:,4] += .01
        if self.budget.calls == 8:
            if self.fault == 'nonfinite': y[0,3] = np.nan
            if self.fault == 'precision': self.observations['norm']['reduction'] = 'torch.float32'
            if self.fault == 'cache': self.cache[next(k for k in self.cache if k != 'length')].flat[0] = -0.
        return y, probs, .01, extra


def execute(path, backend=FakeBackend, **kwargs):
    return p.run(SimpleNamespace(output_dir=path, preflight_only=False), backend_factory=backend,
                 qualifier=lambda _: (refs(), {'synthetic': True}), rss=lambda: 0,
                 install_signals=False, require_runtime=False, **kwargs)


@pytest.fixture(autouse=True)
def compact(monkeypatch):
    monkeypatch.setattr(p.memory.os, 'fsync', lambda _fd: None)
    monkeypatch.setattr(p.batch, 'VOCAB', 4)
    monkeypatch.setattr(p.batch, 'valid_cache', fake_valid)


def verify_manifest(base):
    manifest = json.loads((base/'manifest.json').read_text())
    assert {x['path'] for x in manifest['files']} == {x.name for x in base.iterdir()}-{'manifest.json'}
    for item in manifest['files']:
        raw = (base/item['path']).read_bytes()
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']


def test_fixed_coordinates_schedule_and_no_control_selection():
    groups, stages, rows, calls = p.plans()
    assert len(groups) == 11 and len(stages) == len(rows) == 22 and len(calls) == 3258
    assert sum(s['steps'] for s in stages) == 3192
    assert [sum(c['group_id'] == g for c in calls) for g in range(11)] == [366]*3+[258,282]*4
    h = {'alpha': .07293697809562831, 'log10_temperature': -.15395151820007266}
    da, dt = .07299785589927693-h['alpha'], -.1535-h['log10_temperature']
    for offset, (sa,st) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
        expected = {'alpha': h['alpha']+sa*da, 'log10_temperature': h['log10_temperature']+st*dt}
        assert groups[3+2*offset]['controls'] == groups[4+2*offset]['controls'] == expected
        assert stages[2*(3+2*offset)+1]['controls'] == h
        for key, value in expected.items():
            assert groups[3+2*offset]['control_bits'][key]['hex'] == value.hex()
    assert [r['source_reference_index'] for r in rows] == [0,1]*11
    assert all(r['source_reference_phase'] == 0 for r in rows)
    with pytest.raises(TypeError): p.plans([1,0])


@pytest.mark.parametrize('pair,mapping', [([1,0],'toggle'),([0,1],'identity'),([0,0],'reset0'),
    ([1,1],'reset1'),([None,0],'unresolved'),([0,None],'unresolved'),([None,None],'unresolved')])
def test_neutral_pair_classes(pair,mapping):
    assert p.classify_pair(pair) == mapping


@pytest.mark.parametrize('pair', [[True,0],[2,0],[1],[0,1,0]])
def test_invalid_pair_rejected(pair):
    with pytest.raises(ValueError): p.classify_pair(pair)


def test_complete_compact_contract_and_all_mappings(tmp_path):
    base = tmp_path/'complete'; result = execute(base)
    assert result['status'] == 'completed', result.get('traceback')
    assert result['model_calls'] == 3258 and result['call_counts'] == dict(prefix=22,comparison=44,update=3192)
    assert result['initializations'] == 11 and result['baseline_qualified']
    assert result['outcome'] == 'toggle_observed' and result['toggle_candidate_group_ids'] == [3]
    assert [g['mapping'] for g in result['groups']] == ['identity','reset0','reset1','toggle','identity','reset0','reset1','unresolved','identity','reset0','reset1']
    initial = refs()[:,0]
    for group in result['groups']:
        pulse, hold = [result['stages'][i] for i in group['stage_ids']]
        left, _, _ = p.prior.reconstruct(base,pulse); right, _, _ = p.prior.reconstruct(base,hold)
        assert left[0].tobytes() == initial.tobytes()
        assert left[-1].tobytes() == right[0].tobytes()
        assert group['terminal_state_sha256'] == p.prior.array_hash(right[-1])
        marker = json.loads((base/f'group-{group["group_id"]:02d}.json').read_text())
        assert marker == group and marker['closed']
        for binding in marker['artifacts']:
            assert p.memory.binding(base/binding['path']) == binding
    assert len(list(base.glob('call-*-diagnostic.npz'))) == 44
    assert not list(base.glob('*.pending'))
    assert all(len(row['readout']['all_reference_alignments']) == 2 for row in result['rows'])
    verify_manifest(base)


@pytest.mark.parametrize('error,status', [(OSError,'execution_failed'),(ValueError,'instrument_failed'),
    (TimeoutError,'incomplete'),(MemoryError,'incomplete'),(p.ResourceLimit,'incomplete')])
def test_failure_category_and_unrun_markers(tmp_path,error,status):
    backend = type('Fail',(FakeBackend,),{'fail_at':8,'error':error})
    base = tmp_path/'failed'; result = execute(base,backend)
    assert result['status'] == status and result['model_calls'] == 8
    assert result['stages'][0]['completed_steps'] == 4
    assert result['groups'][0]['status'] == status and result['groups'][0]['mapping'] == 'unavailable'
    assert all(g['status'] == 'not_run' and g['closed'] and g['model_calls'] == 0 for g in result['groups'][1:])
    with np.load(base/'call-008-failed.npz') as z:
        assert z['input'].shape == (2,896)
    verify_manifest(base)


@pytest.mark.parametrize('fault',['nonfinite','precision','cache'])
def test_returned_invalid_output_is_retained(tmp_path,fault):
    result = execute(tmp_path/'failed',type('Bad',(FakeBackend,),{'fault':fault}))
    assert result['status'] == 'instrument_failed' and result['model_calls'] == 8
    with np.load(tmp_path/'failed/call-008-failed.npz') as z:
        assert 'output' in z and 'probabilities' in z
    assert result['stages'][0]['completed_steps'] == 4


def test_comparison_disagreement_stops_before_update(tmp_path):
    result = execute(tmp_path/'mismatch',type('Mismatch',(FakeBackend,),{'fault':'comparison'}))
    assert result['status'] == 'instrument_failed' and result['model_calls'] == 3
    assert result['stages'][0]['completed_steps'] == 0
    assert not result['stages'][0]['cache_comparison']['passed']
    assert len(list((tmp_path/'mismatch').glob('call-*-diagnostic.npz'))) == 2
    assert result['calls'][3]['status'] == 'not_started'


def test_legacy_mechanism_uses_guard_without_global_changes(tmp_path):
    before = (p.mechanism.Budget,p.mechanism.Recorder,p.engine.execute_stage,p.prior.evaluate)
    args = SimpleNamespace(mode='hysteresis',dwell=240,output_dir=tmp_path/'legacy')
    result = p.mechanism.run(args,
        backend_factory=p.guarded_backend_factory(type('Mismatch',(FakeBackend,),{'fault':'comparison'}),require_runtime=False),
        qualifier=lambda _: (refs(),{}), rss=lambda:0, install_signals=False)
    assert result['status'] == 'failed' and result['model_calls'] == 3
    assert before == (p.mechanism.Budget,p.mechanism.Recorder,p.engine.execute_stage,p.prior.evaluate)


def test_raw_limit_is_resource_and_failed_call_uses_reserved_space(tmp_path,monkeypatch):
    original = p.Recorder.admit
    def limited(self,size,final=False):
        if not (final or self.finalizing) and self.size() > 200000:
            raise p.ResourceLimit('test working raw limit')
        return original(self,size,final)
    monkeypatch.setattr(p.Recorder,'admit',limited)
    result = execute(tmp_path/'raw')
    assert result['status'] == 'incomplete' and (tmp_path/'raw/result.json').is_file()
    assert all(g['closed'] for g in result['groups'])


def test_preflight_closes_every_slot_without_backend(tmp_path):
    def forbidden(*_): raise AssertionError('no model allowed')
    args = SimpleNamespace(output_dir=tmp_path/'preflight',preflight_only=True)
    result = p.run(args,backend_factory=forbidden,qualifier=lambda _:(refs(),{}),rss=lambda:0,install_signals=False)
    assert result['status'] == 'preflight_passed' and result['model_calls'] == 0 and not result['model_loaded']
    assert all(g['closed'] and g['status'] == 'not_run' for g in result['groups'])
    assert all(c['status'] == 'not_started' for c in result['calls'])
    verify_manifest(args.output_dir)


def test_preflight_rejects_nonoriginal_pins_before_io():
    args = SimpleNamespace(**{**p.PINS,'references_sha256':'0'*64})
    with pytest.raises(ValueError,match='original'): p.qualify_inputs(args)


def test_exact_native_envelope_and_restricted_cli():
    envelope = p.artifact_envelope()
    assert envelope['total_bytes'] == 336247616 < p.WORKING_OUTPUT
    assert not envelope['compression_assumed']
    options = p.parser()._option_string_actions
    assert '--mode' not in options and '--steps' not in options and '--alpha' not in options
    assert {'--references-sha256','--recovery-results-sha256','--preflight-only'} <= options.keys()


def test_other_profile_rejected_before_files(tmp_path):
    with pytest.raises(ValueError,match='two fixed'):
        p.run_fixed(SimpleNamespace(output_dir=tmp_path/'forbidden'),profile='arbitrary',
                    plan_factory=p.plans,design_factory=p.design,qualifier=lambda _:(refs(),{}))
    assert not (tmp_path/'forbidden').exists()


def test_thread_startup_not_silently_repaired(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'STARTUP_THREADS',{key:'2' for key in p.THREAD_KEYS})
    result = execute(tmp_path/'threads')
    assert result['status'] == 'instrument_failed' and result['model_calls'] == 0


def test_native_runtime_guard_records_failure(tmp_path):
    backend = type('RuntimeBad',(FakeBackend,),{'patch':{'norms':['x']*49,'rotaries':['model.rotary_emb']},
        'torch':SimpleNamespace(get_num_threads=lambda:1,get_num_interop_threads=lambda:1),
        'loop':SimpleNamespace(mode='native64')})
    result = p.run(SimpleNamespace(output_dir=tmp_path/'runtime',preflight_only=False),backend_factory=backend,
        qualifier=lambda _:(refs(),{}),rss=lambda:0,install_signals=False)
    assert result['status'] == 'instrument_failed' and result['model_calls'] == 0
    assert (tmp_path/'runtime/backend-readiness.json').is_file()


def test_path_preflight_has_six_explicit_history_rows(tmp_path):
    import run_public_path_comparison as path
    args = SimpleNamespace(output_dir=tmp_path/'path-preflight',preflight_only=True)
    result = p.run_fixed(args,profile='public-path-comparison',plan_factory=path.plans,
        design_factory=path.design,qualifier=lambda _:(refs(),{}),rss=lambda:0,install_signals=False)
    assert result['status'] == 'preflight_passed' and result['model_calls'] == 0
    assert len(result['rows']) == 6
    assert [r['original_history_index'] for r in result['rows']] == [0,1]*3
    assert all(r['status'] == 'not_run' and 'not H memory bits' in r['reference_label_scope'] for r in result['rows'])
    assert [g['row_ids'] for g in result['groups']] == [[0,1],[2,3],[4,5]]
    verify_manifest(args.output_dir)


@pytest.mark.parametrize('error,status',[(MemoryError,'incomplete'),(OSError,'execution_failed')])
def test_final_cache_inspection_keeps_original_failure_category(tmp_path,error,status):
    class LastCache(FakeBackend):
        inspections = 0
        def cache_arrays(self):
            if self.budget.calls == 123:
                self.inspections += 1
                if self.inspections >= 3:
                    raise error('injected final cache inspection')
            return super().cache_arrays()
    result = execute(tmp_path/'final-cache',LastCache)
    assert result['status'] == status and result['model_calls'] == 123
    assert result['stages'][0]['completed_steps'] == 120
    assert result['stages'][0]['final_cache_error'].startswith(error.__name__+':')


def test_failed_return_can_use_final_reserve(tmp_path,monkeypatch):
    rec = p.Recorder(tmp_path/'reserve',rss=lambda:0)
    monkeypatch.setattr(p,'WORKING_OUTPUT',1)
    with pytest.raises(p.ResourceLimit): rec.npz('ordinary.npz',input=refs()[:,0])
    item = rec.npz('call-004-failed.npz',input=refs()[:,0])
    assert (rec.directory/item['path']).is_file()
    rec.stream.close()


def test_native_comparison_payload_and_cache_contract(tmp_path,monkeypatch):
    monkeypatch.setattr(p.batch,'VOCAB',151936)
    monkeypatch.setattr(p.batch,'valid_cache',NATIVE_VALID_CACHE)
    backend = type('NativeUnit',(FakeBackend,),{'native':True,'fail_at':4})
    result = execute(tmp_path/'native-unit',backend)
    assert result['status'] == 'execution_failed' and result['model_calls'] == 4
    for call in (2,3):
        with np.load(tmp_path/f'native-unit/call-{call:03d}-diagnostic.npz') as z:
            assert z['probabilities'].shape == (2,151936) and z['probabilities'].dtype == np.float64
    with np.load(tmp_path/'native-unit/stage-00-scalar-prefix.npz') as z:
        assert NATIVE_VALID_CACHE(dict(z))
    with np.load(tmp_path/'native-unit/stage-00-expanded-prefix.npz') as z:
        assert NATIVE_VALID_CACHE(dict(z),2)
    verify_manifest(tmp_path/'native-unit')


def test_actual_reference_preflight_has_no_model(tmp_path):
    base = Path('/data2/shrimp-memory-data/audits')
    recovery = base/'p6-recovery/20261025T221011Z-recovery'
    args = SimpleNamespace(output_dir=tmp_path/'actual-preflight',preflight_only=True,
        cache_dir=Path('/not-read-in-preflight-cache'),vectors=Path('/not-read-in-preflight.npz'),
        references=recovery/'reference-cycles.npz',reference_manifest=recovery/'reference-manifest.json',
        source_hold_results=base/'p6-followup/20261025T220342Z-followup/continued-hold/results.json',
        recovery_results=recovery/'run/results.json',**p.PINS)
    if not args.references.is_file(): pytest.skip('local historical evidence is not available')
    def forbidden(*_): raise AssertionError('preflight must not load a model')
    result = p.run(args,backend_factory=forbidden,rss=lambda:0,install_signals=False)
    assert result['status'] == 'preflight_passed', result.get('traceback')
    assert result['model_calls'] == 0 and not result['model_loaded']
    with np.load(args.references) as z:
        assert np.load(args.output_dir/'reference-starts.npy').tobytes() == z['states'][:,0].tobytes()
    verify_manifest(args.output_dir)
