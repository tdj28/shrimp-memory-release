"""Offline orchestration/control fixtures; these never load or evaluate a model."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import run_public_nominal_handoffs as runner


def actual_args(output):
    base = Path('/data2/shrimp-memory-data/audits/p6-writes/20261025T222347Z-writes')
    return SimpleNamespace(trajectory=base/'run/trial-01-endpoint-final-120.npz',
        fixed_inputs=base/'run/fixed-trial-inputs.npz', historical_metadata=base/'run/metadata.json',
        historical_plan=base/'run/trial-plan.json', historical_audit=base/'complete-audit.json',
        historical_manifest=base/'final-manifest.json', cache_dir=Path('/tmp/unused-model'),
        vectors=Path('/tmp/unused-vectors'), output_dir=Path(output), preflight_only=False)


def synthetic_payload():
    references = np.zeros((2, 6, runner.DIM), dtype=np.float64)
    references[1, :, 0] = 10.
    for i in range(6): references[:, i, i+1] = .1
    states = np.repeat(references[1, 0][None], 121, axis=0)
    states[12, 20] = .012; states[36, 20] = .036; states[60, 20] = .05
    return runner.derive_handoffs(references, states)


def synthetic_source(args):
    return dict(source_commit='synthetic', files=[], working_tree_status=[], synthetic=True)


def synthetic_model(args):
    return dict(synthetic=True, no_model_loaded=True)


class FakeBackend:
    """Scripted phase cycles and real-sized arrays for serialization, not a model."""
    def __init__(self, args, budget, recorder, *, payload=None, bits=None, failure=None, defect=None):
        self.budget, self.recorder = budget, recorder
        self.payload = synthetic_payload() if payload is None else payload
        self.bits = [1, 0, 0, 0] if bits is None else bits
        self.failure, self.defect = failure, defect
        self.observations = self.good_observations()
        self.cache = None; self.expanded = None; self.reads = 0

    @staticmethod
    def good_observations():
        return {group: {k: 'torch.float64' for k in keys} for group, keys in runner.PRECISION_KEYS.items()}

    def runtime_info(self):
        return dict(patch=dict(norms=[f'model.layers.{i}.{k}' for i in range(24)
            for k in ('input_layernorm', 'post_attention_layernorm')]+['model.norm'], rotaries=['model.rotary_emb']),
            torch_threads=1, interop_threads=1, thread_env={k:'1' for k in runner.THREAD_KEYS},
            startup_thread_env={k:'1' for k in runner.THREAD_KEYS}, mode='native64')

    def maybe_fail(self):
        if self.failure and self.budget.calls == self.failure[0]: raise self.failure[1]('injected offline failure')

    def probability(self, width):
        p = np.zeros((width, runner.VOCAB), dtype=np.float64)
        p[:, 0] = .75; p[:, 1] = .25
        return p

    def prepare(self):
        self.budget.claim('native64_prefix'); self.maybe_fail()
        self.observations = self.good_observations()
        self.cache = {'length':np.asarray(37, dtype=np.int64)}
        for i in range(24):
            for kind in ('key', 'value'):
                self.cache[f'layer_{i:02d}_{kind}'] = np.full(runner.batch.CACHE_SHAPE, (i+1)/100,
                    dtype=np.float64)
        self.expanded = {k:np.repeat(v, 2, axis=0) for k,v in self.cache.items() if k != 'length'}
        ids = np.asarray([runner.instrument.PREFIX_IDS], dtype=np.int64)
        return np.zeros((1, runner.DIM), dtype=np.float64), self.probability(1), .01, dict(
            input_after=ids.copy(), input_exact=True, observed_output_dtype='torch.float64',
            observed_probability_dtype='torch.float64')

    def cache_arrays(self):
        self.reads += 1
        if self.defect == ('precall_cache', self.budget.calls) and self.reads == 2:
            self.cache['layer_00_key'].flat[0] += 1
        # Real backend returns copies. Copying ensures tests cannot hide alias mutation.
        return {k:v.copy() for k,v in self.cache.items()}

    def forward(self, x, cached):
        self.maybe_fail(); self.observations = self.good_observations()
        row = self.budget.slots[self.budget.calls-1]; group = row['group_id']
        lanes = list(range(2)) if len(x) == 2 else [int(row['label'][-1])]
        phase = (row['step'] if row['category'] == 'update' else 1) % 6
        y = np.stack([self.payload['references'][self.bits[2*group+lane] or 0, phase] for lane in lanes])
        for j,lane in enumerate(lanes):
            if self.bits[2*group+lane] is None: y[j, 25] += .01 * (row['step'] + 1)
        extra = dict(input_after=x.copy(), input_exact=True, observed_output_dtype='torch.float64',
                     observed_probability_dtype='torch.float64')
        if cached:
            extra.update(used_cache_before={k:v.copy() for k,v in self.expanded.items()},
                         used_cache_length_after=np.asarray(38, dtype=np.int64))
        p = self.probability(len(x))
        if self.defect and self.budget.calls == self.defect[1]:
            mode = self.defect[0]
            if mode == 'precision': self.observations['norm']['reduction'] = 'torch.float32'
            if mode == 'nonfinite': y[0, 0] = np.nan
            if mode == 'input': extra['input_after'][0, 0] += 1
            if mode == 'cache': self.cache['layer_00_key'].flat[0] += 1
            if mode == 'expanded': extra['used_cache_before']['layer_00_key'].flat[0] += 1
            if mode == 'comparison': y[0, 0] += 1e-6
            if mode == 'probabilities': p[0, 0] += 1e-4
        return y.copy(), p, .01, extra


def run_fixture(path, *, payload=None, bits=None, failure=None, defect=None, preflight=False):
    payload = synthetic_payload() if payload is None else payload
    args = actual_args(path); args.preflight_only = preflight
    def factory(a,b,r): return FakeBackend(a,b,r,payload=payload,bits=bits,failure=failure,defect=defect)
    return runner.run(args, backend_factory=factory, input_qualifier=lambda args:(payload,dict(synthetic=True,bindings={})),
        model_qualifier=synthetic_model, source_factory=synthetic_source)


@pytest.fixture(autouse=True)
def small_arrays(request, monkeypatch):
    if request.node.name != 'test_maximum_actual_serialization':
        monkeypatch.setattr(runner, 'VOCAB', 7)
        monkeypatch.setattr(runner.batch, 'CACHE_SHAPE', (1,2,37,1))


def test_fixed_schedule_and_readout_plan():
    slots = runner.call_plan()
    assert len(slots) == 490
    assert [sum(s['group_id']==g for s in slots) for g in range(2)] == [245]*2
    assert [s['category'] for s in slots[:6]] == ['prefix']+['comparison']*4+['update']
    assert slots[-1]['step'] == 240 and all(s['controls']==runner.HOLD for s in slots)
    assert runner.derive_design()['readout']['closure_pairs'] == [[i,i-6] for i in range(235,241)]
    assert runner.derive_design()['source_indices'] == [[0,60],[12,36]]
    assert runner.derive_design()['literal_phase_carry'] is True


def test_source_inventory_binds_new_profile_and_reused_dependencies(tmp_path):
    source=runner.source_metadata(actual_args(tmp_path/'unused'))
    records={row['path']:row for row in source['files']}
    required={'tools/run_public_nominal_handoffs.py','tests/test_run_public_nominal_handoffs.py',
        'tools/audit_public_nominal_handoffs.py','tests/test_audit_public_nominal_handoffs.py',
        'docs/public-nominal-handoffs.md','tools/run_public_state_intervention.py',
        'tools/benchmark_qwen_batch.py','tools/validate_qwen_map.py','tools/check_p6_writes.py',
        'tools/check_p6_hold.py','tools/check_p6_recovery.py','pyproject.toml','uv.lock'}
    assert set(records)==required
    assert len(source['source_commit'])==40 and isinstance(source['working_tree_status'],list)
    for name in ('tools/run_public_nominal_handoffs.py','tests/test_run_public_nominal_handoffs.py',
                 'tools/run_public_state_intervention.py'):
        assert records[name]['sha256']==runner.instrument.sha256(runner.ROOT/name)


def test_literal_handoff_indices_preserve_all_bytes_and_phases():
    payload = synthetic_payload(); refs = payload['references']
    states = np.repeat(refs[1, 0][None], 121, axis=0)
    states[12] = refs[0, 3] + .001
    states[36] = refs[1, 5] + .002
    states[60] = refs[0, 2] + .003
    before = states.tobytes(), refs.tobytes()
    output = runner.derive_handoffs(refs, states)
    assert (states.tobytes(), refs.tobytes()) == before
    assert np.array_equal(output['historical_state_indices'], [[0,60],[12,36]])
    assert runner.same_bytes(output['initial_states'], states[np.asarray([[0,60],[12,36]])])
    for index in (12,36,60): assert runner.same_bytes(output[f'x{index}'], states[index])
    assert not any(k in output for k in ('d','r','draw','u','projected_once','projected_twice'))


@pytest.mark.parametrize('bad', ['nan','wrong_phase','wrong_dtype','wrong_shape'])
def test_literal_invalid_inputs_fail(bad):
    payload=synthetic_payload(); refs=payload['references']; states=np.repeat(refs[1,0][None],121,axis=0)
    if bad=='nan': states[12,20]=np.nan
    elif bad=='wrong_phase': states[0]=refs[1,1]
    elif bad=='wrong_dtype': refs=refs.astype(np.float32)
    else: states=states[:-1]
    with pytest.raises(runner.InstrumentFailure): runner.derive_handoffs(refs,states)


def test_identical_handoff_states_are_not_silently_reselected():
    refs=synthetic_payload()['references']; states=np.repeat(refs[1,0][None],121,axis=0)
    output=runner.derive_handoffs(refs,states)
    assert np.array_equal(output['initial_states'],np.repeat(refs[1,0][None],4,axis=0).reshape(2,2,896))


def test_mature_numerical_and_evidence_primitives_are_unchanged():
    def nodes(path):
        tree=ast.parse(path.read_text())
        return {node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    tools=Path(__file__).resolve().parents[1]/'tools'
    old,new=nodes(tools/'run_public_state_intervention.py'),nodes(tools/'run_public_nominal_handoffs.py')
    for name in ('InstrumentFailure','ResourceLimit','require','finite','same_bytes','array_hash','cache_hash',
        'binding','checked_norm','binding_index','require_historical_binding','qualify_model','readout',
        'failure_status','Budget','QwenBackend','precision_ok','validate_readiness','raw_controls','evaluate',
        'compare','execute_group','parser','main'):
        assert old[name]==new[name],name


def test_tail_extra_state_is_archived_but_not_seventh_gate():
    refs=synthetic_payload()['references']; tail=np.concatenate((refs[1,-1:],refs[1],refs[1]))
    tail[0,50]=1000.
    row=runner.readout(tail,refs)
    assert row['decoded_bit']==1 and row['maximum_l2_residual']==0
    assert len(row['lag_six_phase_l2_residuals'])==6
    assert len(row['reference_distances'])==2 and all(len(x['by_shift'])==6 for x in row['reference_distances'])


def test_neutral_ambiguous_and_nonprimitive_readout():
    refs=synthetic_payload()['references']; tail=np.concatenate((refs[1,-1:],refs[1],refs[1]))
    ambiguous=np.stack((refs[1],refs[1]))
    assert runner.readout(tail,ambiguous)['decoded_bit'] is None
    fixed=np.repeat(refs[1,0][None],13,axis=0)
    assert runner.readout(fixed,refs)['cycle_gate_passed'] is False


@pytest.mark.parametrize('bits,outcome', [([1,0,0,0],'both_early_handoffs_h0'),
    ([1,0,1,1],'both_early_handoffs_h1'),([1,0,0,1],'mixed_early_handoffs'),
    ([1,0,1,0],'mixed_early_handoffs'),([1,0,0,None],'early_handoff_unresolved'),
    ([1,0,None,1],'early_handoff_unresolved'),([None,0,0,0],'control_unqualified'),
    ([1,1,None,None],'control_unqualified')])
def test_panel_classification(bits,outcome):
    assert runner.panel_outcome(bits)==outcome


@pytest.mark.parametrize('bits,outcome', [([1,0,0,0],'both_early_handoffs_h0'),
    ([1,0,1,1],'both_early_handoffs_h1'),([1,0,0,1],'mixed_early_handoffs'),
    ([1,0,0,None],'early_handoff_unresolved'),([None,0,0,0],'control_unqualified')])

def test_complete_file_contract_and_finite_nulls_do_not_skip(tmp_path,bits,outcome):
    path=tmp_path/'run'; result=run_fixture(path,bits=bits)
    assert result['status']=='completed' and result['outcome']==outcome and result['model_calls']==490
    assert result['resource_limits_passed']
    assert result['terminal_resources']['final_raw_bytes']==sum(p.stat().st_size for p in path.iterdir() if p.is_file())
    assert [g['status'] for g in result['groups']]==['completed']*2
    assert [g['model_calls'] for g in result['groups']]==[245]*2
    owned=[]
    for group in result['groups']:
        marker=json.loads((path/f'group-{group["group_id"]:02d}.json').read_text())
        assert marker==group and group['closed']
        for entry in marker['artifacts']:
            assert runner.binding(path/entry['path'],flat=True)==entry; owned.append(entry['path'])
    assert len(owned)==len(set(owned))
    for group in range(2):
        with np.load(path/f'group-{group:02d}-states.npz') as z:
            assert z['states'].shape==(241,2,896)
            assert np.array_equal(z['states'][0],synthetic_payload()['initial_states'][group])
        with np.load(path/f'group-{group:02d}-tail.npz') as z: assert z['state_indices'].tolist()==list(range(228,241))
    events=[json.loads(s) for s in (path/'events.jsonl').read_text().splitlines()]
    assert sum(e['event']=='model_call_started' for e in events)==490
    assert sum(e['event']=='model_call_completed' for e in events)==490
    assert sum(e['event']=='group_comparison' for e in events)==2
    with np.load(path/'call-001-g00-prefix.npz') as z:
        assert z['input'].shape==(1,37) and z['probabilities'].shape==(1,7) and z['output'].shape==(1,896)
    with np.load(path/'call-006-g00-update-001.npz') as z:
        assert np.array_equal(z['input'],synthetic_payload()['initial_states'][0])
    manifest=json.loads((path/'manifest.json').read_text())
    assert {r['path'] for r in manifest['files']}=={p.name for p in path.iterdir()}-{'manifest.json'}


@pytest.mark.parametrize('exception,status', [(RuntimeError,'execution_failed'),(OSError,'execution_failed'),
    (runner.ResourceLimit,'incomplete'),(TimeoutError,'incomplete'),(MemoryError,'incomplete'),(ValueError,'instrument_failed')])
def test_failed_attempt_preserved_and_unrun_closed(tmp_path,exception,status):
    path=tmp_path/'run'; result=run_fixture(path,failure=(6,exception))
    assert result['status']==status and result['model_calls']==6
    assert [g['model_calls'] for g in result['groups']]==[6,0]
    assert [g['status'] for g in result['groups']]==[status,'not_run']
    row=json.loads((path/'call-006-g00-update-001.json').read_text())
    assert row['status']=='failed' and row['exception_category']==status
    with np.load(path/'call-006-g00-update-001.npz') as z: assert 'input' in z and 'output' not in z
    assert all(g['closed'] for g in result['groups'])


@pytest.mark.parametrize('mode', ['precision','nonfinite','input','cache','expanded','probabilities'])
def test_returned_defect_keeps_raw_and_partial_state(tmp_path,mode):
    path=tmp_path/'run'; result=run_fixture(path,defect=(mode,6))
    assert result['status']=='instrument_failed' and result['model_calls']==6
    with np.load(path/'call-006-g00-update-001.npz') as z: assert 'probabilities' in z and 'output' in z
    with np.load(path/'group-00-states.npz') as z: assert z['states'].shape==(2,2,896)
    summary=json.loads((path/'group-00-summary.json').read_text())
    assert summary['returned_updates']==1 and summary['completed_updates']==0


def test_comparison_disagreement_stops_before_trajectory(tmp_path):
    path=tmp_path/'run'; result=run_fixture(path,defect=('comparison',3))
    assert result['status']=='instrument_failed' and result['model_calls']==5
    summary=json.loads((path/'group-00-summary.json').read_text())
    assert not summary['comparison']['cached_vs_full']['passed'] and summary['returned_updates']==0


def test_preinvocation_cache_failure_is_not_erased_or_counted(tmp_path):
    path=tmp_path/'run'; result=run_fixture(path,defect=('precall_cache',1))
    assert result['status']=='instrument_failed' and result['model_calls']==1
    row=json.loads((path/'blocked-call-002-g00-compare-cached.json').read_text())
    assert row['status']=='blocked_before_invocation' and row['model_calls']==0
    with np.load(path/'blocked-call-002-g00-compare-cached.npz') as z: assert 'cache_before_layer_00_key' in z
    assert result['calls'][1]['blocked_evidence']=='blocked-call-002-g00-compare-cached.json'


def test_prefix_exception_retains_input_and_zero_return(tmp_path):
    path=tmp_path/'run'; result=run_fixture(path,failure=(1,OSError))
    assert result['model_calls']==1 and result['status']=='execution_failed'
    with np.load(path/'call-001-g00-prefix.npz') as z: assert z['input'].shape==(1,37) and 'probabilities' not in z


def test_preflight_never_constructs_model(tmp_path):
    args=actual_args(tmp_path/'run'); args.preflight_only=True
    def forbidden(*a): raise AssertionError('model loading forbidden')
    result=runner.run(args,backend_factory=forbidden,input_qualifier=lambda a:(synthetic_payload(),dict(bindings={})),
        source_factory=synthetic_source,model_qualifier=synthetic_model)
    assert result['status']=='preflight_passed' and result['outcome']=='not_run_preflight'
    assert not result['model_loaded'] and result['model_calls']==0
    assert all(g['closed'] and g['status']=='not_run' for g in result['groups'])


def test_input_pin_failure_precedes_model_and_closes_groups(tmp_path):
    args=actual_args(tmp_path/'run'); args.trajectory=tmp_path/'wrong.npz'; args.trajectory.write_bytes(b'wrong')
    result=runner.run(args,backend_factory=lambda *a:pytest.fail('model'),source_factory=synthetic_source,
        model_qualifier=synthetic_model)
    assert result['status']=='instrument_failed' and result['model_calls']==0 and not result['model_loaded']
    assert all(g['closed'] for g in result['groups'])


def test_repeated_provenance_deduplicates_only_identical_bindings():
    row=dict(path='/a/run/../run/x',bytes=3,sha256='abc')
    assert runner.binding_index([row,row])=={'/a/run/x':(3,'abc')}
    with pytest.raises(runner.InstrumentFailure): runner.binding_index([row,{**row,'bytes':4}])


def test_completed_group_artifacts_never_change_on_late_failure(tmp_path):
    path=tmp_path/'run'; result=run_fixture(path,failure=(489,OSError))
    assert result['status']=='execution_failed' and result['model_calls']==489
    assert [g['model_calls'] for g in result['groups']]==[245,244]
    for group in result['groups'][:1]:
        assert group['status']=='completed'
        for row in group['artifacts']: assert row==runner.binding(path/row['path'],flat=True)
    with np.load(path/'group-01-states.npz') as z: assert z['states'].shape[0]==239


def test_raw_working_cap_preserves_terminal_files(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'RAW_CAP',runner.FINAL_RESERVE+1024*1024)
    result=run_fixture(tmp_path/'run')
    assert result['status']=='incomplete' and result['model_calls']==0
    assert (tmp_path/'run/result.json').is_file() and (tmp_path/'run/manifest.json').is_file()


def test_readiness_rejects_wrong_threads_and_patches(tmp_path):
    args=actual_args(tmp_path/'run')
    class Bad(FakeBackend):
        def runtime_info(self):
            row=super().runtime_info(); row['interop_threads']=2; return row
    result=runner.run(args,backend_factory=Bad,input_qualifier=lambda a:(synthetic_payload(),dict(bindings={})),
        source_factory=synthetic_source,model_qualifier=synthetic_model)
    assert result['status']=='instrument_failed' and result['model_calls']==0
    assert not json.loads((args.output_dir/'backend-readiness.json').read_text())['passed']


def test_cli_rejects_arbitrary_controls_and_limits():
    args=actual_args('/tmp/unused'); cli=[]
    for k,v in vars(args).items():
        if k!='preflight_only': cli.extend(['--'+k.replace('_','-'),str(v)])
    assert runner.parser().parse_args(cli).trajectory==args.trajectory
    for pair in (['--alpha','.1'],['--steps','1'],['--seed','3']):
        with pytest.raises(SystemExit): runner.parser().parse_args(cli+pair)


def test_maximum_actual_serialization(tmp_path):
    """Stream the fixed maximum with production dimensions; never retain all probabilities in RAM."""
    requested=os.environ.get('SHRIMP_NOMINAL_FULL_FIXTURE')
    if not requested: pytest.skip('Full-size retained fixture is run once in the preparation harness')
    path=Path(requested)
    payload,_=runner.qualify_inputs(actual_args(path))
    result=run_fixture(path,payload=payload)
    assert result['status']=='completed' and result['model_calls']==490
    assert result['resource_limits_passed']
    assert len(list(path.glob('call-*.npz')))==490
    probabilities=0
    for file in sorted(path.glob('call-*.npz')):
        with np.load(file,allow_pickle=False) as z:
            assert z['probabilities'].dtype==np.float64 and z['probabilities'].shape[1]==151936
            probabilities+=z['probabilities'].nbytes
    # Two prefix + four scalar +484 batch returns; no compression assumption.
    assert probabilities==(6+484*2)*151936*8
    actual=sum(p.stat().st_size for p in path.iterdir() if p.is_file())
    assert result['terminal_resources']['final_raw_bytes']==actual
    assert actual+runner.FINAL_RESERVE < runner.RAW_CAP
    receipt=dict(synthetic=True,model_calls=490,files=len(list(path.iterdir())),probability_bytes=probabilities,
        actual_raw_bytes=actual,raw_plus_final_reserve_bytes=actual+runner.FINAL_RESERVE,
        raw_cap_bytes=runner.RAW_CAP,threads={k:os.environ[k] for k in runner.THREAD_KEYS},
        measured_scope='Scripted full-vocabulary serialization, not numerical model evidence')
    (path.parent/(path.name+'-serialization.json')).write_text(json.dumps(receipt,indent=2)+'\n')
