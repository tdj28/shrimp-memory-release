#!/usr/bin/env python3
"""Five fixed endpoint settings of one explicit public write, two starts each."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import signal
import subprocess
import time
import traceback

import run_public_batch_writes as prior
import numpy as np

memory, reference, contract, batch = prior.memory, prior.reference, prior.contract, prior.batch
ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('nominal', 'alpha-minus', 'alpha-plus', 'log10T-minus', 'log10T-plus')
DELTAS = ((0., 0.), (-1e-5, 0.), (1e-5, 0.), (0., -1e-4), (0., 1e-4))
DEFAULT_PROFILE = 'axis-stencil'
PROFILES = (DEFAULT_PROFILE, 'negative-axis-refinement')
REFINEMENT_GROUPS = ('fraction-0', 'fraction-025', 'fraction-050', 'fraction-075', 'fraction-100')
REFINEMENT_FRACTIONS = (0., .25, .5, .75, 1.)
MAX_CALLS, MAX_UPDATES = 1830, 1800
MAX_SECONDS, EXTERNAL_SECONDS, GRACE = 1800, 2100, 30
MAX_RSS, MAX_VM = 8*1024**3, 16*1024**3
MAX_OUTPUT, FINAL_RESERVE = 160*1024**2, 8*1024**2
WORKING_OUTPUT = MAX_OUTPUT-FINAL_RESERVE
ARCHIVE_CAP, TRANSFER_RESERVATION, SOURCE_UPLOAD_CAP = 224*1024**2, 320*1024**2, 8*1024**2
PRECISION_SCOPE = ('Existing native64 normalization/rotary patches with fresh per-call observations, '
    'CPU math SDPA and one thread; ten stage-specific cached/full batch checks. '
    'No new derivative-convergence test or global precision proof.')


def validate_profile(profile):
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError('unknown fixed control-tolerance profile')
    return profile


def plans(requested_bit, profile=DEFAULT_PROFILE):
    plan = prior.validate_plan(prior.WritePlan(requested_bit))
    validate_profile(profile)
    action = prior.actions(plan)[0]
    groups, stages, rows, calls = [], [], [], []
    names, deltas = GROUPS, DELTAS
    if profile == 'negative-axis-refinement':
        names = REFINEMENT_GROUPS
        deltas = tuple((0., -fraction*1e-4) if requested_bit == 0 else (-fraction*1e-5, 0.)
            for fraction in REFINEMENT_FRACTIONS)
    for group_id, (name, delta) in enumerate(zip(names, deltas, strict=True)):
        group = dict(profile=profile, group_id=group_id, name=name, endpoint_delta=dict(alpha=delta[0], log10_temperature=delta[1]),
            status='not_started', row_ids=[2*group_id, 2*group_id+1],
            stage_ids=[2*group_id, 2*group_id+1], initial_state_sha256=None, terminal_state_sha256=None)
        if profile == 'negative-axis-refinement': group['fraction'] = REFINEMENT_FRACTIONS[group_id]
        groups.append(group)
        for local_id, canonical in enumerate(prior.stage_plan(plan)):
            stage = copy.deepcopy(canonical)
            stage.update(profile=profile, stage_id=len(stages), group_id=group_id, group=name, local_stage_id=local_id,
                nominal_controls=canonical['controls'].copy(), historical_comparison=None,
                historical_comparison_scope='nominal_only' if group_id == 0 else 'not_applicable_perturbed_control')
            if local_id == 0:
                for key, value in group['endpoint_delta'].items(): stage['controls'][key] += value
            stages.append(stage)
            for category, step, cached in [('prefix', 0, None), ('comparison', 0, True), ('comparison', 0, False)]+[
                    ('update', step, True) for step in range(1, stage['steps']+1)]:
                calls.append(dict(profile=profile, call=len(calls)+1, group_id=group_id, group=name, stage_id=stage['stage_id'],
                    stage=stage['name'], category=category, step=step, cached=cached, status='not_started'))
        for start in range(2):
            rows.append(dict(profile=profile, row_id=len(rows), group_id=group_id, group=name, batch_index=start,
                message=plan.messages[start], requested_bit=requested_bit, source_reference_index=start,
                source_reference_label=('ad', 'td')[start], source_phase=0, status='not_started',
                parser_outcome='not_started', public_action=None, realized_stages=[dict(profile=profile, stage_id=s['stage_id'],
                    name=s['name'], steps=s['steps'], controls=s['controls'].copy()) for s in stages[-2:]],
                dispatch_count=0, completed_updates=0, readout=None, outcome='not_evaluated',
                historical_comparison=None))
    if len(calls) != MAX_CALLS or sum(s['steps'] for s in stages) != MAX_UPDATES:
        raise ValueError('fixed control-tolerance schedule changed')
    return plan, groups, stages, rows, calls


def derive_design(requested_bit, profile=DEFAULT_PROFILE):
    plan, groups, stages, rows, calls = plans(requested_bit, profile)
    envelope = 5*prior.artifact_envelope()['estimated_uncompressed_envelope_bytes']
    if envelope+prior.FAILED_RETURN_BYTES+16384 >= WORKING_OUTPUT:
        raise ValueError('uncompressed evidence does not fit working allowance')
    perturbation_scope = ('Engineering amplitudes only: endpoint alpha plus/minus 1e-5 or log10 temperature '
        'plus/minus 1e-4. Hold and durations unchanged. These are not validated text tolerances.')
    if profile == 'negative-axis-refinement':
        perturbation_scope = ('Adapted after the axis-stencil pilot: fixed fractions 0, 1/4, 1/2, 3/4, 1 '
            'of endpoint log10 temperature -1e-4 for WRITE0 or endpoint alpha -1e-5 for WRITE1. '
            'Zero and full fractions are matched nominal and reproducibility controls; all outcomes are retained. '
            'Hold and durations unchanged. No monotonicity, intervening safe region or validated text tolerance is assumed.')
    return dict(schema_version=1, profile=profile, evidence='adapted exploratory public-controller endpoint perturbation pilot',
        evidence_rung=0, requested_bit=requested_bit, public_messages=list(plan.messages),
        groups=groups, realized_stages=[{k:s[k] for k in ('profile', 'stage_id', 'group_id', 'name', 'steps',
            'controls', 'nominal_controls')} for s in stages],
        schedule=dict(groups=5, batch_size=2, rows=10, stages=10, endpoint_steps=120, hold_steps=240,
            batch_updates=1800, row_updates=3600, prefix_calls=10, comparison_calls=20, total_calls=1830,
            reference_initializations=5, between_group_resets=4),
        limits=dict(model_calls=MAX_CALLS, internal_seconds=MAX_SECONDS, external_seconds=EXTERNAL_SECONDS,
            termination_grace_seconds=GRACE, threads=1, peak_rss_bytes=MAX_RSS, external_vm_bytes=MAX_VM,
            working_raw_bytes=WORKING_OUTPUT, finalization_reserve_bytes=FINAL_RESERVE, total_raw_bytes=MAX_OUTPUT,
            archive_bytes=ARCHIVE_CAP, transfer_reservation_bytes=TRANSFER_RESERVATION, source_upload_bytes=SOURCE_UPLOAD_CAP),
        artifact_envelope=dict(estimated_uncompressed_envelope_bytes=envelope,
            failed_return_reserved_bytes=prior.FAILED_RETURN_BYTES,
            scope='Five complete old-format groups plus one failed native-shape return; no compression assumption.'),
        map=dict(model=reference.MODEL, revision=reference.REVISION, prefix_ids=reference.PREFIX_IDS,
            hold=memory.writes.HOLD, endpoints=memory.writes.ENDPOINTS),
        perturbation_scope=perturbation_scope,
        precision_scope=PRECISION_SCOPE, diagnostic_tolerance=prior.DIAGNOSTIC_TOLERANCE,
        nominal_trajectory_tolerance=prior.TRAJECTORY_TOLERANCE,
        outcome_definition='Readout uses only the hold trajectory and reference cycles. Qualified requested label '
            'is passed, qualified opposite label failed, no qualified label unresolved. Execution and diagnostic '
            'validity remain separate gates; unexecuted rows are not_evaluated.',
        stopping='Fixed order and no retries. Finite diagnostic or nominal trajectory disagreement is retained '
            'while the schedule continues. Structural, precision, execution or resource errors abort all remaining slots.',
        scope='A visible public controller sensitivity test at two reference phases and five discrete control points; '
            'not a continuous tolerance bound, language success, hidden carrier, general reliability or dynamics-specific advantage.',
        timing_scope='Report measured absolute times and calls. Prior 366-call groups took about 225 seconds; '
            'this is a planning observation, not a timing guarantee or matched speed comparison.')


class Recorder(memory.Recorder):
    def admit(self, size, final=False):
        if self.size()+size > (MAX_OUTPUT if final or self.finalizing else WORKING_OUTPUT):
            raise RuntimeError('raw artifact limit reached')


class Budget:
    def __init__(self, recorder, slots, requested_bit, profile=DEFAULT_PROFILE):
        self.recorder, self.slots, self.requested_bit = recorder, slots, requested_bit
        self.profile = validate_profile(profile)
        _, _, self.expected_stages, _, self.expected_slots = plans(requested_bit, profile)
        if slots != self.expected_slots: raise ValueError('call slots differ from fixed profile schedule')
        self.calls = 0; self.counts = dict(prefix=0, comparison=0, update=0)
        self.stage_id = None; self.max_rss = MAX_RSS

    def check(self):
        if self.recorder.clock()-self.recorder.started >= MAX_SECONDS: raise TimeoutError('1800-second internal cap')
        if self.recorder.rss() >= MAX_RSS: raise MemoryError('8 GiB RSS cap')
        if self.recorder.size() > WORKING_OUTPUT: raise RuntimeError('working raw artifact limit reached')

    def begin(self, stage):
        expected = self.expected_stages[stage['stage_id']]
        if any(stage[k] != expected[k] for k in ('profile', 'group_id', 'group', 'local_stage_id',
                'name', 'controls', 'nominal_controls', 'steps')):
            raise ValueError('stage differs from fixed realized-control schedule')
        if self.calls >= MAX_CALLS or self.slots[self.calls]['stage_id'] != stage['stage_id']:
            raise ValueError('stage order differs from fixed call schedule')
        self.stage_id = stage['stage_id']

    def claim(self, label, category='prefix'):
        self.check(); self.recorder.admit(prior.FAILED_RETURN_BYTES+16384)
        if self.calls >= MAX_CALLS: raise RuntimeError('model-call cap reached')
        slot = self.slots[self.calls]
        expected = self.expected_slots[self.calls]
        if (slot['stage_id'] != self.stage_id or slot['category'] != category or any(
                slot.get(key) != expected[key] for key in ('profile', 'call', 'group_id', 'group',
                    'stage_id', 'stage', 'category', 'step', 'cached'))):
            raise RuntimeError('model call differs from fixed schedule')
        self.calls += 1; self.counts[category] += 1; slot.update(status='started', label=label)
        self.recorder.event('model_call_started', **slot)
        return slot


def check_references(refs):
    if refs.shape != (2,6,896) or refs.dtype != np.float64 or not np.isfinite(refs).all():
        raise ValueError('invalid qualified reference cycles')


def preflight(args, qualifier=memory.qualify_inputs, historical_qualifier=prior.qualify_historical):
    """Load actual immutable evidence; never load a model or execute a forward."""
    profile = validate_profile(getattr(args, 'profile', DEFAULT_PROFILE))
    plan = plans(args.requested_bit, profile)[0]
    refs, provenance = qualifier(args); check_references(refs)
    _, historical = historical_qualifier(args, refs, plan=plan)
    return dict(status='passed', profile=profile, model_calls=0, model_loaded=False, design=derive_design(args.requested_bit, profile),
        qualified_inputs=provenance, historical_inputs=historical,
        reference_initial_state_sha256=prior.array_hash(refs[:,0]),
        initial_rows=[dict(batch_index=i, array_sha256=prior.array_hash(refs[i,0])) for i in range(2)])


def classify_readout(readout, requested_bit):
    label = readout['qualified_label']
    if label is None: return 'unresolved'
    if label not in ('ad', 'td'): raise ValueError('unknown neutral readout label')
    return 'passed' if label == ('ad', 'td')[requested_bit] else 'failed'


def immutable_group_artifacts(directory, group, stages, slots):
    """Closed group evidence only; its last-written marker is the return contract."""
    bindings = []
    for index in group['stage_ids']:
        stage = stages[index]
        for name in ('initial', 'trajectory', 'scalar_prefix_cache', 'expanded_prefix_cache'):
            if stage.get(name) is not None: bindings.append(stage[name])
        bindings.extend(stage['blocks']); bindings.extend(stage['failed_outputs'])
    bindings.extend(slot['artifact'] for slot in slots if slot['group_id'] == group['group_id'] and 'artifact' in slot)
    bindings.extend(memory.binding(Path(directory)/f'row-{index:02d}-completed.json') for index in group['row_ids'])
    unique = {}
    for item in bindings:
        path = prior.historical_source.require_bound(directory, item)
        actual = memory.binding(path)
        if actual['path'] in unique and actual != unique[actual['path']]:
            raise ValueError('conflicting group artifact binding')
        unique[actual['path']] = actual
    return [unique[name] for name in sorted(unique)]


def execute_stage(backend, state, stage, recorder, budget, histories, plan):
    archive = prefix = None; stage['status'] = 'running'
    try:
        budget.begin(stage); archive = prior.BlockArchive(recorder, stage, state)
        try:
            backend.prepare(stage['controls'])
            if not prior.precision(backend.observations): raise ValueError('prefix native64 observations missing')
            slot = budget.slots[budget.calls-1]; slot['dtype_observations'] = copy.deepcopy(backend.observations)
            prefix = backend.cache_arrays()
            if not batch.valid_cache(prefix): raise ValueError('invalid scalar prefix cache')
            stage['scalar_prefix_cache'] = recorder.npz(f'stage-{stage["stage_id"]:02d}-scalar-prefix.npz', **prefix)
            slot['status'] = 'returned'; budget.check()
        except BaseException as exc:
            if budget.calls and budget.slots[budget.calls-1]['category'] == 'prefix':
                budget.slots[budget.calls-1].update(status=memory.error_status(exc), error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if budget.calls and budget.slots[budget.calls-1]['category'] == 'prefix':
                recorder.event('model_call_completed', _final=True, **budget.slots[budget.calls-1])
        diagnostics = [prior.evaluate(backend, state, stage, recorder, budget, prefix, archive) for _ in range(2)]
        stage['cache_comparison'] = batch.compare(*diagnostics)
        for _ in range(stage['steps']): prior.evaluate(backend, state, stage, recorder, budget, prefix, archive)
        stage['status'] = 'completed'
    except BaseException as exc:
        stage.update(status=memory.error_status(exc), error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        with batch.preserve_call_files():
            if prefix is not None:
                try:
                    after = backend.cache_arrays(); stage['final_scalar_cache_sha256'] = prior.cache_hash(after)
                    stage['final_scalar_cache_unchanged'] = prior.same_bytes(prefix, after)
                    if not stage['final_scalar_cache_unchanged']:
                        stage.update(status='failed', final_cache_error='final scalar cache changed')
                except BaseException as exc:
                    stage.update(status='failed', final_scalar_cache_unchanged=False,
                        final_cache_error=f'{type(exc).__name__}: {exc}')
            if archive is not None:
                archive.finalize()
                saved, ids, _ = prior.reconstruct(recorder.directory, stage)
                if stage['group_id'] == 0:
                    stage['historical_comparison'] = prior.compare_history(saved, ids, histories, stage['local_stage_id'], plan)
            recorder.event('stage_finished', _final=True, stage_id=stage['stage_id'], group_id=stage['group_id'],
                status=stage['status'], completed_steps=stage['completed_steps'])
    if stage['status'] != 'completed': raise ValueError('stage failed final cache gate')


def run(args, backend_factory=prior.Backend, qualifier=memory.qualify_inputs,
        historical_qualifier=prior.qualify_historical, clock=time.monotonic,
        rss=reference.rss_bytes, install_signals=True):
    profile = validate_profile(getattr(args, 'profile', DEFAULT_PROFILE))
    plan, groups, stages, rows, slots = plans(args.requested_bit, profile)
    recorder = Recorder(args.output_dir, clock, rss); budget = Budget(recorder, slots, args.requested_bit, profile)
    result = dict(schema_version=1, profile=profile, status='failed', requested_bit=plan.requested_bit, groups=groups,
        rows=rows, stages=stages, calls=slots, reference_initializations=0, between_group_resets=0,
        evidence=derive_design(plan.requested_bit, profile)['evidence'], evidence_rung=0,
        precision_scope=PRECISION_SCOPE, nominal_equivalence_passed=None)
    state = backend = None; old_handlers = {}
    if install_signals:
        def stop(signum, frame):
            if signum == signal.SIGALRM: raise TimeoutError('declared internal deadline reached')
            raise InterruptedError(f'received signal {signum}')
        old_handlers = {s:signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
        for s in old_handlers: signal.signal(s, stop)
        signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    try:
        recorder.json('design.json', derive_design(plan.requested_bit, profile)); recorder.json('planned-rows.json', rows)
        recorder.json('planned-groups.json', groups); recorder.json('planned-stages.json', stages)
        recorder.json('planned-calls.json', slots)
        modules = (prior, memory, contract, memory.writes, memory.recovery, memory.hold_test,
            reference, batch, prior.historical_source, prior.trajectory_source)
        paths = [Path(m.__file__) for m in modules]+[Path(__file__), ROOT/'docs/public-control-tolerance.md',
            ROOT/'docs/receipts/p6-writes.json', ROOT/'docs/receipts/p6-write-path-design.json',
            ROOT/'pyproject.toml', ROOT/'uv.lock']
        result['source'] = dict(utc=reference.utc_now(), source_commit=subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
            files=[memory.binding(p) for p in paths], arguments={**{k:str(v) for k,v in vars(args).items()}, 'profile':profile})
        recorder.json('source.json', result['source'])
        refs, provenance = qualifier(args); check_references(refs)
        recorder.json('qualified-inputs.json', provenance)
        histories, historical = historical_qualifier(args, refs, plan=plan)
        recorder.json('historical-inputs.json', historical)
        # Freeze every public request before any numerical call. Realized controls
        # are a separately logged engineering perturbation, never a changed parse.
        for row in rows:
            action = prior.actions(plan)[row['batch_index']]
            row['public_action'] = {**asdict(action), 'stages':[asdict(s) for s in action.stages]}
            row['parser_outcome'] = 'accepted'
            def dispatch(parsed):
                contract.verify_action(row['message'], parsed); row['dispatch_count'] += 1
            contract.dispatch_command(row['message'], dispatch, prepared_action=action)
        recorder.json('frozen-rows.json', rows)
        state = refs[:,0].copy(); stages[0]['initial'] = recorder.state('stage-00-initial.npy', state)
        recorder.event('inputs_frozen', initial_sha256=prior.array_hash(state), historical_source=historical.get('source_commit'))
        budget.check(); started = clock(); backend = backend_factory(args, recorder, budget)
        result['cold_initialization_seconds'] = clock()-started
        if hasattr(backend, 'bindings'): recorder.json('model-inputs.json', backend.bindings)
        for group in groups:
            state = refs[:,0].copy(); result['reference_initializations'] += 1
            if group['group_id']: result['between_group_resets'] += 1
            group.update(status='running', initial_state_sha256=prior.array_hash(state))
            recorder.event('group_started', group_id=group['group_id'], initial_sha256=group['initial_state_sha256'])
            for i in group['row_ids']: rows[i]['status'] = 'running'
            for i in group['stage_ids']: execute_stage(backend, state, stages[i], recorder, budget, histories, plan)
            hold = stages[group['stage_ids'][1]]
            saved, _, _ = prior.reconstruct(recorder.directory, hold)
            for i in group['row_ids']:
                row = rows[i]; index = row['batch_index']
                row['readout'] = memory.neutral_readout(saved[1:,index], refs)
                row['outcome'] = classify_readout(row['readout'], plan.requested_bit)
                row['numerical_diagnostics_passed'] = all(stages[k]['cache_comparison']['passed'] for k in group['stage_ids'])
                if group['group_id'] == 0:
                    comparisons = [stages[k]['historical_comparison']['rows'][index] for k in group['stage_ids']]
                    old_label = historical['controls'][index]['neutral_readout']['qualified_label']
                    row['historical_comparison'] = dict(compared_states=361, compared_updates=360,
                        maximum_coordinate_gap=max(c['maximum_coordinate_gap'] for c in comparisons),
                        token_mismatch_count=sum(c['token_mismatch_count'] for c in comparisons),
                        state_bytes_equal=all(c['state_bytes_equal'] for c in comparisons),
                        token_bytes_equal=all(c['token_bytes_equal'] for c in comparisons),
                        final_labels_equal=row['readout']['qualified_label'] == old_label,
                        historical_qualified_label=old_label,
                        passed=all(c['passed'] for c in comparisons) and row['readout']['qualified_label'] == old_label)
                row.update(status='completed', completed_updates=360)
                recorder.json(f'row-{i:02d}-completed.json', row)
            group.update(terminal_state_sha256=prior.array_hash(state),
                diagnostics_passed=all(stages[k]['cache_comparison']['passed'] for k in group['stage_ids']))
            if group['group_id'] == 0:
                result['nominal_equivalence_passed'] = group['diagnostics_passed'] and all(
                    rows[i]['historical_comparison']['passed'] for i in group['row_ids'])
            group['artifacts'] = immutable_group_artifacts(recorder.directory, group, stages, slots)
            group['status'] = 'completed'
            try:
                recorder.json(f'group-{group["group_id"]:02d}.json', group)
            except BaseException:
                group['status'] = 'failed'
                raise
            recorder.event('group_finished', group_id=group['group_id'], status=group['status'])
        if budget.calls != MAX_CALLS or budget.counts != dict(prefix=10, comparison=20, update=1800):
            raise ValueError('completed schedule count differs')
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status=memory.error_status(exc), error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        for item in groups+rows:
            if item['status'] == 'running': item.update(status=memory.error_status(exc), error=result['error'])
    finally:
        if install_signals:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s in old_handlers: signal.signal(s, signal.SIG_IGN)
        recorder.finalizing = True
        try:
            for row in rows:
                group = groups[row['group_id']]
                row['completed_updates'] = sum(stages[i]['completed_steps'] for i in group['stage_ids'])
                recorder.json(f'row-{row["row_id"]:02d}.json', row, final=True)
            result.update(model_calls=budget.calls, call_counts=budget.counts,
                completed_batch_updates=sum(s['completed_steps'] for s in stages),
                completed_row_updates=2*sum(s['completed_steps'] for s in stages),
                elapsed_seconds=clock()-recorder.started, peak_rss_bytes=rss(),
                outcome_counts={k:sum(r['outcome'] == k for r in rows) for k in ('passed','failed','unresolved','not_evaluated')})
            result['terminal_state_sha256'] = prior.array_hash(state) if state is not None else None
            result['resource_limits_passed'] = (budget.calls <= MAX_CALLS and result['elapsed_seconds'] < MAX_SECONDS
                and result['peak_rss_bytes'] < MAX_RSS and recorder.size() <= MAX_OUTPUT)
            recorder.json('result.json', result, final=True)
            recorder.event('finished', status=result['status'], model_calls=budget.calls)
            recorder.stream.close()
            recorder.json('manifest.json', dict(files=[memory.binding(p) for p in sorted(recorder.directory.iterdir())
                if p.is_file() and p.name != 'manifest.json']), final=True)
        finally:
            for s, handler in old_handlers.items(): signal.signal(s, handler)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requested-bit', type=int, choices=(0,1), required=True)
    parser.add_argument('--profile', choices=PROFILES, default=DEFAULT_PROFILE)
    parser.add_argument('--preflight-only', action='store_true')
    for name in ('cache-dir','vectors','references','reference-manifest','source-hold-results',
                 'recovery-results','historical-run','output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('references-sha256','reference-manifest-sha256','recovery-results-sha256'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    for digest in (args.references_sha256, args.reference_manifest_sha256, args.recovery_results_sha256):
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            parser.error('input hashes must be lowercase SHA-256 digests')
    if args.preflight_only:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        try:
            result = preflight(args)
        except BaseException as exc:
            result = dict(status='failed', model_calls=0, model_loaded=False,
                error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        with (args.output_dir/'input-preflight.json').open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
        return 0 if result['status'] == 'passed' else 1
    return 0 if run(args)['status'] == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
