#!/usr/bin/env python3
"""Two isolated public writes with a declared half-duration endpoint pulse."""
import argparse
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')

import numpy as np
import run_public_command_memory as memory

contract, reference = memory.contract, memory.reference
MESSAGES = ('Write bit 1 to register R0.', 'Write bit 0 to register R0.')
STARTS, OLD_TRIALS = (0, 1), (2, 1)
PROFILE = 'public-isolated-half-pulse60-hold240-v1'
MAX_CALLS, MAX_UPDATES = 612, 600
MAX_SECONDS, EXTERNAL_SECONDS, GRACE = 600, 780, 30
MAX_RSS, MAX_VM = 8 * 1024**3, 16 * 1024**3
MAX_OUTPUT, FINAL_RESERVE = 32 * 1024**2, 2 * 1024**2
WORKING_OUTPUT = MAX_OUTPUT-FINAL_RESERVE
ARCHIVE_CAP, TRANSFER_RESERVATION = 48 * 1024**2, 64 * 1024**2
HISTORICAL_SOURCE = 'f0c757ed2c7142bd2dd5cb78a796f69f7f49f4c4'
PRECISION_SCOPE = ('Existing native float64 Qwen map, per-call normalization/rotary instrumentation '
    'and four cache/full-prefix comparisons. No new derivative test or global precision proof.')


@dataclass(frozen=True)
class ExperimentalAction:
    profile_id: str
    trial_id: int
    canonical_action: contract.WriteAction
    stages: tuple[contract.Stage, contract.Stage]


def compile_profile(text, canonical_action, trial_id):
    contract.verify_action(text, canonical_action)
    if type(trial_id) is not int or trial_id not in (0, 1) or text != MESSAGES[trial_id]:
        raise ValueError('message differs from the fixed isolated trial')
    return ExperimentalAction(PROFILE, trial_id, canonical_action,
        (replace(canonical_action.stages[0], steps=60), canonical_action.stages[1]))


def verify_profile(text, canonical_action, action, trial_id):
    expected = compile_profile(text, canonical_action, trial_id)
    if (type(action) is not ExperimentalAction or type(action.profile_id) is not str
            or type(action.trial_id) is not int or type(action.stages) is not tuple
            or len(action.stages) != 2):
        raise ValueError('invalid typed experimental action')
    contract.verify_action(text, action.canonical_action)
    for stage in action.stages:
        if (type(stage) is not contract.Stage or type(stage.name) is not str
                or type(stage.steps) is not int or type(stage.alpha) is not float
                or type(stage.log10_temperature) is not float):
            raise ValueError('invalid typed experimental stage')
    if action != expected: raise ValueError('experimental action differs from public profile')


def json_action(action):
    result = asdict(action)
    result['stages'] = [asdict(stage) for stage in action.stages]
    if isinstance(action, ExperimentalAction): result['canonical_action'] = json_action(action.canonical_action)
    return result


def message_plan():
    return [dict(message_id=i, trial_id=i, message=text, profile_id=PROFILE,
        reference_index=STARTS[i], reference_label=('ad','td')[STARTS[i]], reference_phase=0,
        historical_trial_id=OLD_TRIALS[i], status='not_started', parser_outcome='not_started',
        action=None, experimental_action=None, dispatch_count=0, call_start=None, call_end=None,
        model_calls=0, initial_state=None, state_before=None, state_after=None, completed_steps=0,
        stages=[], readout=None, prefix_comparison=None, frontend_seconds=0.0, seconds=0.0)
        for i, text in enumerate(MESSAGES)]


def artifact_envelope():
    # Uncompressed payloads plus ZIP/header and generous metadata allowances.
    # Blocks duplicate every update and include 50 shared boundary states.
    parts = dict(trajectory_state_bytes=(600+4+600+50)*896*8,
        trajectory_token_time_bytes=2*600*16,
        comparison_array_bytes=8*(2*896+151936)*8,
        historical_prefix_bytes=2*61*896*8+2*60*8,
        fixed_reference_bytes=2*6*896*8,
        state_artifact_allowance_bytes=16*(896*8+256),
        archive_container_allowance_bytes=80*4096,
        metadata_events_allowance_bytes=4*1024**2)
    total = sum(parts.values())
    if total >= WORKING_OUTPUT: raise ValueError('declared artifacts cannot fit working raw limit')
    # At most one partial/final hold (241 states) must fit after an admission failure.
    final_need = 241*896*8+240*16+8192+256*1024
    if final_need >= FINAL_RESERVE: raise ValueError('finalization reserve cannot hold one late stage')
    return dict(parts=parts, estimated_uncompressed_envelope_bytes=total,
        finalization_required_allowance_bytes=final_need,
        scope='Conservative planned payload estimate; actual raw admission remains mandatory.')


def derive_design():
    updates = 0
    for i, text in enumerate(MESSAGES):
        action = compile_profile(text, contract.parse_command(text), i)
        verify_profile(text, action.canonical_action, action, i)
        updates += sum(stage.steps for stage in action.stages)
    if updates != MAX_UPDATES or updates+4+8 != MAX_CALLS: raise ValueError('declared call envelope differs')
    return dict(schema_version=1, evidence='adapted exploratory isolated public half-pulse pilot',
        messages=list(MESSAGES), profile_id=PROFILE, source_reference_indices=list(STARTS),
        source_phase=0, historical_trial_ids=list(OLD_TRIALS),
        adaptation='One half-dose challenge after full-pulse successes and the separate shorter-hold test; '
                   '60 is not an optimized duration or a safe general bound.',
        map=dict(model=reference.MODEL, revision=reference.REVISION, prefix_ids=reference.PREFIX_IDS,
                 hold=memory.writes.HOLD, endpoints=memory.writes.ENDPOINTS),
        schedule=dict(trials=2, endpoint_steps=60, hold_steps=240, numerical_updates=600,
                      prefix_calls=4, comparison_calls=8, total_calls=612, calls_per_trial=306),
        limits=dict(model_calls=MAX_CALLS, internal_seconds=MAX_SECONDS, external_seconds=EXTERNAL_SECONDS,
            termination_grace_seconds=GRACE, peak_rss_bytes=MAX_RSS, external_vm_bytes=MAX_VM,
            working_raw_bytes=WORKING_OUTPUT, finalization_reserve_bytes=FINAL_RESERVE,
            total_raw_bytes=MAX_OUTPUT, archive_bytes=ARCHIVE_CAP,
            transfer_reservation_bytes=TRANSFER_RESERVATION, threads=1),
        artifact_envelope=artifact_envelope(), precision_scope=PRECISION_SCOPE,
        cache_tolerance=memory.writes.CACHE_TOLERANCE, match_tolerance=memory.recovery.MATCH_TOLERANCE,
        checkpoint_every_steps=12,
        evaluation='Neutral terminal orbit qualification only; literal request correctness is independently evaluated.',
        stopping='Two isolated reference starts, one fixed reset; no extension, retry, pulse scan or outcome-based reset. '
                 'Wrong/unresolved readout continues; execution, resource, precision or historical-prefix mismatch aborts.')


class Recorder(memory.Recorder):
    def admit(self, size, final=False):
        if self.size()+size > (MAX_OUTPUT if final or self.finalizing else WORKING_OUTPUT):
            raise RuntimeError('raw artifact limit reached')

    def npz(self, name, final=False, **arrays):
        # Bad model output must use working allowance, leaving the reserve for the
        # validated partial trajectory. Stage finals may use finalization reserve.
        if name.startswith('invalid-message-'):
            final = False
        return super().npz(name, final=final, **arrays)


class Budget:
    def __init__(self, recorder):
        self.recorder, self.calls = recorder, 0
        self.counts = dict(prefix=0, comparison=0, update=0)
        self.scope = self.stage_counts = None
        self.next_stage = 0

    def check(self):
        if self.recorder.clock()-self.recorder.started >= MAX_SECONDS:
            raise TimeoutError('600-second internal deadline reached')
        if self.recorder.rss() >= MAX_RSS: raise MemoryError('8 GiB RSS cap reached')
        if self.recorder.size() > WORKING_OUTPUT: raise RuntimeError('working raw artifact limit reached')

    def begin(self, message_id, stage):
        if self.stage_counts is not None and self.stage_counts != dict(prefix=1, comparison=2, update=self.scope['steps']):
            raise RuntimeError('preceding stage did not complete')
        if type(message_id) is not int or self.next_stage >= 4 or message_id != self.next_stage//2:
            raise RuntimeError('undeclared message/stage order')
        action = compile_profile(MESSAGES[message_id], contract.parse_command(MESSAGES[message_id]), message_id)
        declared = action.stages[self.next_stage % 2]
        if (stage['name'] != declared.name or type(stage['steps']) is not int or stage['steps'] != declared.steps
                or stage['controls'] != dict(alpha=declared.alpha, log10_temperature=declared.log10_temperature)):
            raise RuntimeError('stage differs from declared half-pulse profile')
        self.next_stage += 1
        self.scope = dict(message_id=message_id, trial_id=message_id, profile_id=PROFILE,
                          stage=stage['name'], steps=stage['steps'])
        self.stage_counts = dict(prefix=0, comparison=0, update=0)

    def claim(self, label, category='prefix'):
        self.check()
        if self.scope is None or category not in self.counts: raise RuntimeError('call outside declared stage')
        if category in ('update', 'comparison'):
            # Reserve working space for a returned invalid native-shape array,
            # as well as its event, before asking the model to produce it.
            self.recorder.admit((2*896+151936)*8+16384)
        per_stage = dict(prefix=1, comparison=2, update=self.scope['steps'])
        total = dict(prefix=4, comparison=8, update=600)
        if (self.calls >= MAX_CALLS or self.stage_counts[category] >= per_stage[category]
                or self.counts[category] >= total[category]): raise RuntimeError('declared model-call limit reached')
        self.calls += 1; self.counts[category] += 1; self.stage_counts[category] += 1
        self.recorder.event('model_call_started', call=self.calls, category=category, label=label, **self.scope)


def require_bound(directory, item):
    name = item['path']
    if Path(name).name != name: raise ValueError('historical raw path must be a basename')
    path = Path(directory)/name
    if path.is_symlink() or not path.is_file(): raise ValueError('missing or linked historical file')
    if path.stat().st_size != item['bytes']: raise ValueError('historical file size differs')
    reference.require_hash(path, item['sha256'])
    return path


def qualify_historical(args, refs, receipt_path=None):
    """Bind saved controls without feeding old desired labels to dispatch."""
    root = Path(__file__).resolve().parents[1]
    receipt_path = Path(receipt_path) if receipt_path is not None else root/'docs/receipts/p6-writes.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['independent_audit_status'] != 'audit_passed' or receipt['experiment_source_commit'] != HISTORICAL_SOURCE:
        raise ValueError('historical receipt is not the declared audited run')
    bindings = []
    loaded = {}
    for name in ('results.json', 'metadata.json'):
        candidates = [b for b in receipt['bindings'] if b['path'].endswith('/run/'+name)]
        if len(candidates) != 1: raise ValueError('historical receipt binding missing or ambiguous')
        item = {**candidates[0], 'path': name}
        path = require_bound(args.historical_run, item)
        loaded[name] = json.loads(path.read_text()); bindings.append(memory.binding(path))
    result, meta = loaded['results.json'], loaded['metadata.json']
    if result['status'] != 'passed' or meta['source_commit'] != HISTORICAL_SOURCE:
        raise ValueError('historical attempt is not the completed fixed source')
    if (meta['model'] != reference.MODEL or meta['revision'] != reference.REVISION
            or meta['prefix_ids'] != reference.PREFIX_IDS or meta['hold'] != memory.writes.HOLD
            or meta['candidate_endpoints'] != memory.writes.ENDPOINTS or meta['threads'] != 1):
        raise ValueError('historical map inputs differ')
    for name, digest in (('weights', reference.WEIGHTS_SHA256), ('config', memory.hold_test.CONFIG_SHA256),
                         ('vectors', reference.VECTOR_SHA256)):
        if meta[name]['sha256'] != digest: raise ValueError('historical model input differs: '+name)
    for key in ('references_sha256','reference_manifest_sha256','recovery_results_sha256'):
        if meta['arguments'][key] != getattr(args, key): raise ValueError('historical prerequisite differs: '+key)
    for module in (reference, memory.hold_test, memory.recovery):
        path = Path(module.__file__)
        if meta['source_hashes'][path.name] != reference.sha256(path):
            raise ValueError('historical mathematical source differs: '+path.name)
    prefixes, tokens, controls = [], [], []
    for i, old_id in enumerate(OLD_TRIALS):
        row = result['rows'][old_id]
        if (row['trial_id'] != old_id or row['source_index'] != STARTS[i] or row['phase'] != 0
                or row['status'] != 'completed' or row['completed_steps'] != 360):
            raise ValueError('historical row identity or completion differs')
        canonical = contract.parse_command(MESSAGES[i])
        previous = refs[STARTS[i],0]
        row_bindings = []
        hold_states = None
        for stage, public_stage in zip(row['stages'], canonical.stages, strict=True):
            expected = dict(alpha=public_stage.alpha, log10_temperature=public_stage.log10_temperature)
            if (stage['name'] != public_stage.name or stage['controls'] != expected
                    or stage['steps'] != public_stage.steps or stage['completed_steps'] != public_stage.steps
                    or stage['status'] != 'completed' or not stage['cache_comparison']['passed']):
                raise ValueError('historical stage controls/completion differs')
            path = require_bound(args.historical_run, stage['trajectory'])
            bindings.append(memory.binding(path)); row_bindings.append(memory.binding(path))
            with np.load(path, allow_pickle=False) as data:
                states, ids = data['states'], data['token_ids']
                if (states.dtype != np.float64 or states.shape != (public_stage.steps+1,896)
                        or not np.isfinite(states).all() or ids.dtype != np.int64
                        or ids.shape != (public_stage.steps,) or not np.isfinite(data['step_seconds']).all()
                        or data['step_seconds'].shape != ids.shape
                        or float(data['alpha']) != public_stage.alpha
                        or float(data['log10_temperature']) != public_stage.log10_temperature
                        or int(data['first_state_index']) != 0
                        or states[0].tobytes() != previous.tobytes()):
                    raise ValueError('historical array shape, values or continuity differs')
                previous = states[-1].copy()
                if stage['name'] == 'endpoint':
                    prefixes.append(states[:61].copy()); tokens.append(ids[:60].copy())
                else: hold_states = states[1:].copy()
        neutral = memory.neutral_readout(hold_states, refs)
        if neutral['qualified_label'] is None: raise ValueError('historical control fails current neutral gate')
        controls.append(dict(trial_id=old_id, source_index=STARTS[i], trajectories=row_bindings,
                             neutral_readout=neutral, model_calls=366))
    return np.stack(prefixes), np.stack(tokens), dict(receipt=memory.binding(receipt_path),
        source_commit=HISTORICAL_SOURCE, bindings=bindings, controls=controls,
        timing_scope='Saved historical controls, not contemporaneous timing or independent new trials.')


def compare_prefix(row, recorder, prefix, tokens):
    path = recorder.directory/row['stages'][0]['trajectory']['path']
    with np.load(path, allow_pickle=False) as arrays:
        actual, actual_tokens = arrays['states'], arrays['token_ids']
        if (actual.shape != (61,896) or actual.dtype != np.float64 or not np.isfinite(actual).all()
                or actual_tokens.shape != (60,) or actual_tokens.dtype != np.int64
                or prefix.shape != (61,896) or prefix.dtype != np.float64 or not np.isfinite(prefix).all()
                or tokens.shape != (60,) or tokens.dtype != np.int64):
            raise ValueError('new half-pulse trajectory has invalid shape/precision')
        states_equal = actual.tobytes() == prefix.tobytes()
        tokens_equal = actual_tokens.tobytes() == tokens.tobytes()
        return dict(historical_trial_id=row['historical_trial_id'], compared_states=61, compared_updates=60,
            state_bytes_equal=states_equal, token_bytes_equal=tokens_equal,
            maximum_coordinate_gap=float(np.max(np.abs(actual-prefix))),
            actual_array_sha256=memory.array_hash(actual), historical_array_sha256=memory.array_hash(prefix),
            passed=bool(states_equal and tokens_equal))


def execute_action(action, row, state, backend, refs, recorder, budget, prefix, tokens):
    """Existing stage machinery with an endpoint gate before hold admission."""
    row['dispatch_count'] += 1
    if row['dispatch_count'] != 1: raise RuntimeError('duplicate dispatcher invocation')
    row['stages'] = [dict(name=s.name, controls=dict(alpha=s.alpha, log10_temperature=s.log10_temperature),
        steps=s.steps, status='not_started', attempted_steps=0, completed_steps=0) for s in action.stages]
    for stage in row['stages']:
        archive = memory.StageArchive(recorder, row['message_id'], stage, state)
        stage['status'] = 'running'
        try:
            budget.begin(row['message_id'], stage)
            backend.prepare(stage['controls'])
            stage['cache_comparison'] = memory.cache_check(backend, state, row, stage, recorder, budget)
            if not stage['cache_comparison']['passed']: raise ValueError('cache/full-prefix comparison failed')
            for step in range(1, stage['steps']+1):
                budget.claim(f'message_{row["message_id"]:02d}_{stage["name"]}_{step:03d}', 'update')
                stage['attempted_steps'] += 1; started = recorder.clock()
                y, probabilities = backend.forward(state.copy(), True)
                elapsed = recorder.clock()-started
                try: memory.check_output(y, probabilities)
                except BaseException:
                    failed = recorder.npz(f'invalid-message-{row["message_id"]:02d}-{stage["name"]}-{step:03d}.npz',
                                          output=y, probabilities=probabilities)
                    stage.setdefault('failed_outputs', []).append(failed)
                    raise
                archive.states.append(y.copy()); archive.seconds.append(elapsed)
                archive.tokens.append(int(np.argmax(probabilities)))
                state[:] = y
                stage['completed_steps'] = step; row['completed_steps'] += 1
                budget.check()
                if step % memory.CHECKPOINT_EVERY == 0: archive.save()
            stage['status'] = 'completed'
        except BaseException as exc:
            stage.update(status=memory.error_status(exc), error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            archive.save(final=True)
            recorder.event('stage_finished', message_id=row['message_id'], _final=True, **stage)
        if stage['name'] == 'endpoint':
            row['prefix_comparison'] = compare_prefix(row, recorder, prefix, tokens)
            recorder.event('historical_prefix_comparison', message_id=row['message_id'], **row['prefix_comparison'])
            if not row['prefix_comparison']['passed']: raise ValueError('historical endpoint prefix mismatch')
    row['readout'] = memory.neutral_readout(np.stack(archive.states[1:]), refs)


def execute_message(row, state, backend, refs, recorder, budget, prefix, tokens):
    row['status'] = 'running'; row['call_start'] = budget.calls
    started = recorder.clock()
    try:
        planned = message_plan()[row['message_id']]
        for key in ('trial_id','message','profile_id','reference_index','reference_label','reference_phase','historical_trial_id'):
            if type(row[key]) is not type(planned[key]) or row[key] != planned[key]:
                raise ValueError('message metadata differs from isolated plan')
        row['state_before'] = recorder.state(f'message-{row["message_id"]:02d}-before.npy', state)
        if memory.array_hash(state) != row['initial_state']['array_sha256']:
            raise ValueError('trial start differs from frozen reference bytes')
        frontend = recorder.clock()
        canonical = contract.parse_command(row['message'])
        experimental = compile_profile(row['message'], canonical, row['trial_id'])
        verify_profile(row['message'], canonical, experimental, row['trial_id'])
        row.update(parser_outcome='accepted', action=json_action(canonical), experimental_action=json_action(experimental))
        def dispatch(parsed):
            verify_profile(row['message'], parsed, experimental, row['trial_id'])
            row['frontend_seconds'] = recorder.clock()-frontend
            execute_action(experimental, row, state, backend, refs, recorder, budget, prefix, tokens)
        contract.dispatch_command(row['message'], dispatch, prepared_action=canonical)
        budget.check(); row['status'] = 'completed'
    except BaseException as exc:
        row.update(status=memory.error_status(exc), error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        row.update(call_end=budget.calls, model_calls=budget.calls-row['call_start'], seconds=recorder.clock()-started)
        row['state_after'] = recorder.state(f'message-{row["message_id"]:02d}-after.npy', state, final=True)
        recorder.event('message_finished', _final=True, **row)


def digital_control():
    rows = []
    for i, text in enumerate(MESSAGES):
        latch = contract.DigitalLatch(STARTS[i]); before = latch.state_bytes.hex()
        start = time.monotonic(); latch.execute(text)
        rows.append(dict(message_id=i, message=text, state_before=before, state_after=latch.state_bytes.hex(),
                         dispatch_calls=latch.dispatch_count, seconds=time.monotonic()-start))
    return dict(scope='Separate isolated digital-latch control; never feeds the numerical receiver.', rows=rows)


def run(args, backend_factory=memory.Backend, qualifier=memory.qualify_inputs,
        historical_qualifier=qualify_historical, clock=time.monotonic, rss=reference.rss_bytes, install_signals=True):
    recorder = Recorder(args.output_dir, clock, rss); budget = Budget(recorder)
    rows = message_plan(); backend = state = None
    result = dict(schema_version=1, status='failed', evidence='adapted exploratory isolated public half-pulse pilot',
        rows=rows, reset_count=0, reset=None, precision_scope=PRECISION_SCOPE,
        evaluation='Independent literal request evaluation required; producer neutral labels are not requested-bit scores.')
    old_handlers = {}
    if install_signals:
        def stop(signum, frame):
            if signum == signal.SIGALRM: raise TimeoutError('declared internal deadline reached')
            raise InterruptedError(f'received signal {signum}')
        old_handlers = {sig:signal.getsignal(sig) for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
        for sig in old_handlers: signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    try:
        recorder.json('design.json', derive_design()); recorder.json('messages.json', list(MESSAGES))
        recorder.json('planned-messages.json', rows)
        root = Path(__file__).resolve().parents[1]
        paths = [Path(__file__), Path(memory.__file__), Path(contract.__file__), Path(memory.writes.__file__),
            Path(memory.recovery.__file__), Path(memory.hold_test.__file__), Path(reference.__file__),
            root/'docs/public-half-pulse.md', root/'docs/public-command-memory.md',
            root/'docs/transparent-command-protocol.md', root/'docs/receipts/p6-writes.json',
            root/'docs/receipts/p6-write-path-design.json', root/'pyproject.toml', root/'uv.lock']
        result['source'] = dict(utc=reference.utc_now(), source_commit=subprocess.check_output(
            ['git','-C',str(root),'rev-parse','HEAD'], text=True).strip(),
            arguments={k:str(v) for k,v in vars(args).items()}, files=[memory.binding(p) for p in paths])
        recorder.json('source.json', result['source']); recorder.json('digital-control.json', digital_control())
        refs, provenance = qualifier(args)
        if refs.shape != (2,6,896) or refs.dtype != np.float64 or not np.isfinite(refs).all():
            raise ValueError('qualified references must be finite float64 (2,6,896)')
        recorder.json('qualified-inputs.json', provenance)
        prefixes, tokens, historical = historical_qualifier(args, refs)
        if (prefixes.shape != (2,61,896) or prefixes.dtype != np.float64 or not np.isfinite(prefixes).all()
                or tokens.shape != (2,60) or tokens.dtype != np.int64
                or any(prefixes[i,0].tobytes() != refs[STARTS[i],0].tobytes() for i in (0,1))):
            raise ValueError('historical prefix inventory differs from declared inputs')
        recorder.json('historical-inputs.json', historical)
        recorder.npz('historical-prefixes.npz', states=prefixes, token_ids=tokens)
        recorder.npz('fixed-inputs.npz', references=refs, initial_states=refs[list(STARTS),0])
        for row in rows:
            row['initial_state'] = recorder.state(f'trial-{row["trial_id"]:02d}-initial.npy', refs[row['reference_index'],0])
        recorder.json('trial-initials.json', [r['initial_state'] for r in rows])
        recorder.event('inputs_frozen', initials=[r['initial_state'] for r in rows], historical_source=historical.get('source_commit'))
        state = refs[STARTS[0],0].copy()
        budget.check(); started = clock(); backend = backend_factory(args, recorder, budget)
        result['cold_initialization_seconds'] = clock()-started
        for row in rows:
            budget.check()
            if row['trial_id'] == 1:
                before = recorder.state('between-trials-before-reset.npy', state)
                result['reset'] = dict(from_trial=0, to_trial=1, call=budget.calls, state_before=before,
                    state_after=None, source=rows[1]['initial_state'], mutation_applied=False)
                state[:] = refs[STARTS[1],0]
                result['reset']['mutation_applied'] = True; result['reset_count'] += 1
                result['reset']['state_after'] = recorder.state('between-trials-after-reset.npy', state)
                recorder.event('between_trial_reset', **result['reset'])
            execute_message(row, state, backend, refs, recorder, budget, prefixes[row['trial_id']], tokens[row['trial_id']])
        if (budget.calls != MAX_CALLS or budget.counts != dict(prefix=4,comparison=8,update=600)
                or [r['model_calls'] for r in rows] != [306,306] or result['reset_count'] != 1):
            raise ValueError('completed trials differ from declared call/reset counts')
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status=memory.error_status(exc), error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        if install_signals:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in old_handlers.items(): signal.signal(sig, handler)
        recorder.finalizing = True
        result.update(model_calls=budget.calls, call_counts=budget.counts, elapsed_seconds=clock()-recorder.started,
            peak_rss_bytes=rss(), precision_observations=getattr(backend,'observations',{}),
            accepted_messages=sum(r['parser_outcome']=='accepted' for r in rows),
            completed_messages=sum(r['status']=='completed' for r in rows))
        for row in rows: recorder.json(f'message-{row["message_id"]:02d}.json', row, final=True)
        if state is not None: result['terminal_state'] = recorder.state('terminal-state.npy', state, final=True)
        result['resource_limits_passed'] = (budget.calls <= MAX_CALLS and result['elapsed_seconds'] < MAX_SECONDS
            and result['peak_rss_bytes'] < MAX_RSS and recorder.size() <= MAX_OUTPUT)
        recorder.json('result.json', result, final=True)
        recorder.event('finished', status=result['status'], model_calls=budget.calls)
        recorder.stream.close()
        recorder.json('manifest.json', dict(files=[memory.binding(p) for p in sorted(recorder.directory.iterdir())
            if p.is_file() and p.name != 'manifest.json']), final=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache-dir','vectors','references','reference-manifest','source-hold-results',
                 'recovery-results','historical-run','output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('references-sha256','reference-manifest-sha256','recovery-results-sha256'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args(argv)
    for digest in (args.references_sha256,args.reference_manifest_sha256,args.recovery_results_sha256):
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            p.error('input hashes must be lowercase SHA-256 digests')
    return 0 if run(args)['status']=='completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
