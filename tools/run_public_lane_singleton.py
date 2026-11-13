#!/usr/bin/env python3
"""Single-lane arm of the lane-layout check (docs/public-lane-layout-check.md, "Single-lane arm", fixed 13 November
2026 before any call).

Eight fixed groups: the four operations of the lane-layout check (W = 12 idle updates at H, E0 and E1 = the audited
120-update writes, T = the 30-degree, radius-3.5, 12-update toggle; move controls from tools/run_public_adder.py),
each followed by 240 updates at H and the neutral readout, from each phase-0 reference start, run one state at a time.

Everything else is the pair instrument's own, unchanged: the backend (run_public_batch_writes.Backend, whose forward
sends a one-state input through FeedbackMap.forward, the canonical scalar map), its control setting and prefix call,
the guarded factory (native64 one-thread readiness; an update is refused unless the stage's cached and full
comparisons agree), the budget, recorder, block format, call ledger, group closing and neutral readout. Three
things cannot be reused because the pair code fixes the batch at two, so they are written out here instead of
changing it:
  * the return check requires (1, 896) and (1, vocabulary) shapes;
  * a cached call uses the scalar map's own shallow copy of the stored prefix cache, so there is no expanded batch
    cache to check; instead each call must show that no batch expansion happened, and the stored cache must be
    unchanged before and after, as in the pair evaluator;
  * the archive's metadata and its reader hold (n, 1, 896) blocks and (n, 1) token ids.
One group per shard, one single-thread process per shard.
"""
import copy
import math
from pathlib import Path
import signal
import time
import traceback

import numpy as np

import run_public_adder as adder
import run_public_toggle_check as chk

sweep, pilot, prior, memory, engine, reference = chk.sweep, chk.pilot, chk.prior, chk.memory, chk.engine, chk.reference
batch = prior.batch
PROFILE = 'public-lane-singleton'
PLAN = 'docs/public-lane-layout-check.md'
OPERATIONS = ('W', 'E0', 'E1', 'T')
EXPECTED = {'W': lambda c: c, 'E0': lambda c: 0, 'E1': lambda c: 1, 'T': lambda c: 1 - c}
LANE_SHAPE = (1, 896)
SHARDS = 8


def specs():
    return [dict(name=f'{op}-S{c}', operation=op, start_cycle=c, expected=EXPECTED[op](c))
            for op in OPERATIONS for c in (0, 1)]


def plans():
    groups, stages, rows = [], [], []
    for gid, s in enumerate(specs()):
        ct, steps = adder.controls(s['operation'])
        groups.append(dict(profile=PROFILE, group_id=gid, name=s['name'], kind='probe', role='lane-singleton',
                           operation=s['operation'], start_cycle=s['start_cycle'], expected_bit=s['expected'],
                           controls=ct, control_bits=pilot.control_bits(ct), pulse_steps=steps, start_phase=0,
                           perturbation_seed=None, stage_ids=[2*gid, 2*gid + 1], row_ids=[gid], status='not_started',
                           mapping='unavailable', decoded_bit=None, closed=False, observations=[]))
        for local, (c, n) in enumerate(((ct, steps), (sweep.HOLD, sweep.HOLD_STEPS))):
            stage = copy.deepcopy(prior.stage_plan(prior.WritePlan(0))[0])
            stage.update(profile=PROFILE, stage_id=len(stages), group_id=gid, group=s['name'], local_stage_id=local,
                         name='move' if local == 0 else 'hold', controls=dict(c), control_bits=pilot.control_bits(c),
                         nominal_controls=dict(c), steps=n,
                         historical_comparison_scope='not_applicable_fixed_toggle_exploration')
            stages.append(stage)
        rows.append(dict(profile=PROFILE, row_id=gid, group_id=gid, group=s['name'], lane=0,
                         source_reference_index=s['start_cycle'], source_reference_phase=0, status='not_started',
                         readout=None, decoded_bit=None))
    return groups, stages, rows


def check_return(y, p, seconds, extra):
    return (y.shape == LANE_SHAPE and p.shape == (1, batch.VOCAB) and y.dtype == p.dtype == np.float64
            and np.isfinite(y).all() and np.isfinite(p).all() and np.all(p >= 0) and np.all(p <= 1)
            and np.allclose(p.sum(-1), 1., rtol=0, atol=prior.DIAGNOSTIC_TOLERANCE)
            and math.isfinite(seconds) and seconds > 0 and extra['observed_output_dtype'] == 'torch.float64'
            and extra['observed_probability_dtype'] == 'torch.float64')


class SingletonArchive(prior.BlockArchive):
    """The pair archive's block format; only the recorded state shape differs."""
    def finalize(self):
        self.flush(final=True)
        report = dict(stage_id=self.stage['stage_id'], initial=self.stage['initial'], blocks=self.stage['blocks'],
                      completed_steps=self.stage['completed_steps'],
                      state_shape=[self.stage['completed_steps'] + 1, *LANE_SHAPE], status=self.stage['status'],
                      controls=self.stage['controls'], format='initial plus nonoverlapping post-update blocks')
        self.stage['trajectory'] = self.recorder.json(f'stage-{self.stage["stage_id"]:02d}-trajectory.json', report,
                                                      final=True)


def reconstruct(directory, stage):
    """The pair reader's contract for (n, 1, 896) blocks and (n, 1) token ids."""
    directory = Path(directory)
    initial = np.load(prior.historical_source.require_bound(directory, stage['initial']), allow_pickle=False)
    if initial.shape != LANE_SHAPE or initial.dtype != np.float64 or not np.isfinite(initial).all():
        raise ValueError('invalid stage initial state')
    states, tokens, seconds, cursor = [initial[None]], [], [], 0
    for record in stage['blocks']:
        with np.load(prior.historical_source.require_bound(directory, record), allow_pickle=False) as data:
            y, ids, t = data['states'], data['token_ids'], data['step_seconds']
            n = len(y)
            if (not 1 <= n <= prior.BLOCK or y.shape != (n, *LANE_SHAPE) or y.dtype != np.float64
                    or not np.isfinite(y).all() or ids.shape != (n, 1) or ids.dtype != np.int64 or np.any(ids < 0)
                    or np.any(ids >= batch.VOCAB) or t.shape != (n,) or t.dtype != np.float64
                    or not np.isfinite(t).all() or np.any(t <= 0)
                    or data['first_state_index'].shape != () or data['first_state_index'].dtype != np.int64
                    or data['last_state_index'].shape != () or data['last_state_index'].dtype != np.int64
                    or int(data['first_state_index']) != cursor + 1 or int(data['last_state_index']) != cursor + n
                    or any(float(data[k]) != v for k, v in stage['controls'].items())):
                raise ValueError('invalid/gapped block contract')
            states.append(y.copy()); tokens.append(ids.copy()); seconds.append(t.copy()); cursor += n
    if cursor != stage['completed_steps']:
        raise ValueError('block count differs from completed steps')
    return (np.concatenate(states), np.concatenate(tokens) if tokens else np.empty((0, 1), dtype=np.int64),
            np.concatenate(seconds) if seconds else np.empty(0, dtype=np.float64))


def evaluate(backend, state, stage, recorder, budget, prefix, archive):
    """The pair evaluator (run_public_batch_writes.evaluate) for one state: same slots, gates, artifacts and carry."""
    if state.shape != LANE_SHAPE:
        raise ValueError('a single-lane state is required')
    planned = budget.slots[budget.calls]
    category, cached = planned['category'], planned['cached']
    slot = budget.claim(f"{stage['name']}-{category}-{planned['step']}-{cached}", category)
    slot['forward_attempted'] = False
    original = state.copy()
    arrays = dict(input=original)
    if category == 'update':
        stage['attempted_steps'] += 1
    returned = None
    try:
        before = backend.cache_arrays()
        slot['forward_attempted'] = True
        y, p, seconds, extra = backend.forward(original.copy(), cached)
        returned = (y, p)
        arrays.update(output=y, probabilities=p)
        if category == 'update':
            stage['returned_steps'] += 1
        arrays['input_after'] = extra['input_after']
        slot.update(seconds=seconds if math.isfinite(seconds) else None, input_sha256=prior.array_hash(original),
                    output_sha256=prior.array_hash(y), probabilities_sha256=prior.array_hash(p),
                    dtype_observations=copy.deepcopy(backend.observations),
                    output_dtype=extra['observed_output_dtype'], probability_dtype=extra['observed_probability_dtype'])
        after = backend.cache_arrays()
        slot.update(cache_before_sha256=prior.cache_hash(before), cache_after_sha256=prior.cache_hash(after),
                    cache_length_before=int(before['length']), cache_length_after=int(after['length']),
                    cache_unchanged=bool(batch.valid_cache(before) and batch.valid_cache(after)
                                         and prior.same_bytes(before, prefix) and prior.same_bytes(after, prefix)),
                    input_unchanged=bool(extra['input_after'].shape == original.shape
                                         and extra['input_after'].dtype == original.dtype
                                         and extra['input_after'].tobytes() == original.tobytes()),
                    precision_passed=bool(prior.precision(backend.observations)),
                    outputs_valid=bool(check_return(y, p, seconds, extra)),
                    scalar_path=bool('used_cache_before' not in extra and 'used_cache_length_after' not in extra),
                    expanded_cache_valid=None)
        if not all(slot[k] for k in ('outputs_valid', 'input_unchanged', 'cache_unchanged', 'precision_passed',
                                     'scalar_path')):
            raise ValueError('finite/native64/input/cache/scalar-path structural gate failed')
        slot['top_tokens'] = p.argmax(-1).tolist()
        if category == 'update':
            archive.append(y, np.asarray(slot['top_tokens'], dtype=np.int64), seconds)
            state[:] = y
            slot['state_carried'] = True
        budget.check()
        slot['status'] = 'returned'
    except BaseException as exc:
        slot.update(status=prior.memory.error_status(exc), error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        with batch.preserve_call_files():
            if category == 'comparison' or slot['status'] != 'returned':
                name = f'call-{slot["call"]:03d}-' + ('diagnostic' if category == 'comparison' else 'failed') + '.npz'
                slot['artifact'] = recorder.npz(name, **arrays)
                if category == 'update':
                    stage['failed_outputs'].append(slot['artifact'])
            recorder.event('model_call_completed', _final=slot['status'] != 'returned', **slot)
            if category == 'update':
                archive.flush()
    return returned


def execute_stage(backend, state, stage, recorder, budget):
    """The pair stage executor (run_public_control_continuation.execute_stage) with the single-lane evaluator and
    archive: prefix call under the stage's controls, cached and full comparisons, then the updates."""
    archive = prefix = None
    stage['status'] = 'running'
    try:
        budget.begin(stage)
        archive = SingletonArchive(recorder, stage, state)
        try:
            backend.prepare(stage['controls'])
            if not prior.precision(backend.observations):
                raise ValueError('prefix native64 observations missing')
            slot = budget.slots[budget.calls - 1]
            slot['dtype_observations'] = copy.deepcopy(backend.observations)
            prefix = backend.cache_arrays()
            if not batch.valid_cache(prefix):
                raise ValueError('invalid scalar prefix cache')
            stage['scalar_prefix_cache'] = recorder.npz(f'stage-{stage["stage_id"]:02d}-scalar-prefix.npz', **prefix)
            slot['status'] = 'returned'
            budget.check()
        except BaseException as exc:
            if budget.calls and budget.slots[budget.calls - 1]['category'] == 'prefix':
                budget.slots[budget.calls - 1].update(status=prior.memory.error_status(exc),
                                                      error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if budget.calls and budget.slots[budget.calls - 1]['category'] == 'prefix':
                recorder.event('model_call_completed', _final=True, **budget.slots[budget.calls - 1])
        diagnostics = [evaluate(backend, state, stage, recorder, budget, prefix, archive) for _ in range(2)]
        stage['cache_comparison'] = batch.compare(*diagnostics)
        for _ in range(stage['steps']):
            evaluate(backend, state, stage, recorder, budget, prefix, archive)
        stage['status'] = 'completed'
    except BaseException as exc:
        stage.update(status=prior.memory.error_status(exc), error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        with batch.preserve_call_files():
            if prefix is not None:
                try:
                    after = backend.cache_arrays()
                    stage['final_scalar_cache_sha256'] = prior.cache_hash(after)
                    stage['final_scalar_cache_unchanged'] = prior.same_bytes(prefix, after)
                    if not stage['final_scalar_cache_unchanged']:
                        stage.update(status='failed', final_cache_error='final scalar cache changed')
                except BaseException as exc:
                    stage.update(status='failed', final_scalar_cache_unchanged=False,
                                 final_cache_error=f'{type(exc).__name__}: {exc}')
            if archive is not None:
                archive.finalize()
            recorder.event('stage_finished', _final=True, stage_id=stage['stage_id'], group_id=stage['group_id'],
                           status=stage['status'], completed_steps=stage['completed_steps'])
    if stage['status'] != 'completed':
        raise ValueError('stage failed final cache gate')


def run(args, backend_factory=prior.Backend, qualifier=pilot.qualify_inputs, clock=time.monotonic,
        rss=reference.rss_bytes, require_runtime=True):
    if args.shards != SHARDS or not 0 <= args.shard < SHARDS:
        raise ValueError(f'the single-lane arm runs as {SHARDS} shards of one group each')
    groups, stages, rows = plans()
    ids = sweep.shard_group_ids(len(groups), args.shard, args.shards)
    slots = sweep.shard_calls(groups, stages, ids)
    for c in slots:
        c['profile'] = PROFILE
    mine = [groups[g] for g in ids]
    recorder = sweep.Recorder(args.output_dir, clock, rss)
    budget = sweep.Budget(recorder, stages, slots)
    result = dict(profile=PROFILE, shard=args.shard, shards=args.shards, status='execution_failed',
                  outcome='not_evaluated', model_loaded=False, groups=mine, stages=stages, rows=rows, calls=slots,
                  initializations=0)
    handlers, active = {}, None

    def stop(signum, _frame):
        if signum == signal.SIGALRM:
            raise TimeoutError('internal wall cap')
        raise InterruptedError(f'signal {signum}')
    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        handlers[sig] = signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, sweep.MAX_SECONDS)
    try:
        recorder.json('design.json', dict(profile=PROFILE, rules=PLAN, specs=specs(), shard=args.shard,
                      shards=args.shards, shard_group_ids=ids, all_groups=len(groups), shard_calls=len(slots),
                      lane_shape=list(LANE_SHAPE), artifact_envelope=sweep.artifact_envelope(stages, ids, groups)))
        recorder.json('planned-groups.json', mine)
        recorder.json('planned-calls.json', slots)
        here = Path(__file__).resolve()
        result['source'] = pilot.source_metadata(args, extra_sources=(
            here, Path(chk.__file__).resolve(), Path(sweep.__file__).resolve(), Path(adder.__file__).resolve(),
            pilot.ROOT/PLAN))
        recorder.json('source.json', result['source'])
        if any(v != '1' for v in pilot.STARTUP_THREADS.values()):
            raise pilot.InstrumentFailure('four thread limits must equal one before process startup')
        refs, provenance = qualifier(args)
        engine.shared.check_references(refs)
        recorder.json('qualified-inputs.json', provenance)
        recorder.npz('reference-cycles.npz', states=refs)
        if args.preflight_only:
            result.update(status='preflight_passed', outcome='not_run_preflight')
        else:
            backend = pilot.guarded_backend_factory(backend_factory, require_runtime=require_runtime)(
                args, recorder, budget)
            result['model_loaded'] = True
            if hasattr(backend, 'bindings'):
                recorder.json('model-inputs.json', backend.bindings)
            for group in mine:
                active = group
                state = refs[group['start_cycle'], 0][None].copy()
                recorder.state(f'group-{group["group_id"]:03d}-start.npy', state)
                result['initializations'] += 1
                group.update(status='running', initial_state_sha256=prior.array_hash(state))
                for sid in group['stage_ids']:
                    execute_stage(backend, state, stages[sid], recorder, budget)
                saved, _, _ = reconstruct(recorder.directory, stages[group['stage_ids'][-1]])
                row = rows[group['row_ids'][0]]
                readout = pilot.neutral_readout(saved[1:, 0], refs)
                label = readout['qualified_label']
                row.update(status='completed', readout=readout,
                           decoded_bit=None if label is None else ('ad', 'td').index(label))
                bit = row['decoded_bit']
                group.update(status='completed', decoded_bit=bit, matches_expected=bit == group['expected_bit'],
                             mapping='unresolved' if bit is None else f'cycle{bit}',
                             terminal_state_sha256=prior.array_hash(state),
                             diagnostics_passed=all(stages[i]['cache_comparison']['passed']
                                                    for i in group['stage_ids']))
                pilot.close_group(group, result, recorder)
                active = None
            result.update(status='completed', outcome='checked')
    except BaseException as exc:
        status = pilot.failure_status(exc)
        result.update(status=status, error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        if active is not None:
            active.update(status=status, error=result['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        recorder.finalizing = True
        for group in mine:
            if group['status'] == 'not_started':
                group.update(status='not_run', not_run_reason=result['status'])
            pilot.close_group(group, result, recorder)
        result.update(model_calls=budget.calls, call_counts=budget.counts,
                      elapsed_seconds=clock() - recorder.started, peak_rss_bytes=rss())
        recorder.stream.close()
        recorder.json('result.json', result, final=True)
        recorder.json('manifest.json', dict(files=[memory.binding(p) for p in sorted(recorder.directory.iterdir())
                                                   if p.is_file() and p.name != 'manifest.json']), final=True)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return result


def main(argv=None):
    p = sweep.parser()
    p.description = __doc__
    result = run(p.parse_args(argv))
    return 0 if result['status'] in ('completed', 'preflight_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
