#!/usr/bin/env python3
"""Two public control settings from literal phases zero and one of one source cycle."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import signal
import subprocess
import time
import traceback

import run_public_control_tolerance as shared
prior = shared.prior
import numpy as np

memory, reference, contract, batch = prior.memory, prior.reference, prior.contract, prior.batch
ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'public-two-phase-challenge'
GROUPS = ('nominal', 'full-negative-offset')
MAX_CALLS, MAX_UPDATES = 732, 720
MAX_SECONDS, EXTERNAL_SECONDS, GRACE = 900, 1080, 30
MAX_RSS, MAX_VM = 8*1024**3, 16*1024**3
MAX_OUTPUT, FINAL_RESERVE = 160*1024**2, 8*1024**2
WORKING_OUTPUT = MAX_OUTPUT-FINAL_RESERVE
ARCHIVE_CAP, TRANSFER_RESERVATION, SOURCE_UPLOAD_CAP = 224*1024**2, 320*1024**2, 8*1024**2
PRECISION_SCOPE = ('Existing native64 normalization/rotary patches, per-call observations, '
    'CPU math SDPA and one thread; four stage-specific cached/full batch comparisons. '
    'No new derivative-convergence test or global precision proof.')


def source_index(bit):
    prior.validate_plan(prior.WritePlan(bit))
    return 1-bit


def selected_initials(refs, bit):
    shared.check_references(refs)
    return np.stack([refs[source_index(bit), phase].copy() for phase in (0,1)])


def initial_slots(refs, bit):
    values = selected_initials(refs, bit)
    return [dict(batch_index=i, source_reference_index=source_index(bit),
        source_reference_label=('ad','td')[source_index(bit)], source_phase=i,
        shape=[896], dtype='float64', array_sha256=prior.array_hash(values[i])) for i in (0,1)]


def qualify_phase_history(args, refs, plan=None):
    plan = prior.validate_plan(plan)
    histories, original = prior.qualify_historical(args, refs, plan=plan)
    index = source_index(plan.requested_bit)
    selected = original['controls'][index]
    trial = (1,2)[plan.requested_bit]
    if selected['trial_id'] != trial or selected['source_index'] != index:
        raise ValueError('selected phase-zero historical control differs')
    return histories[index], dict(source_commit=original['source_commit'],
        selected_trial_id=trial, selected_source_index=index, selected_source_phase=0,
        selected_control=selected, original_qualification=original,
        scope='The original scalar pair is qualified as prerequisite evidence; only its matching '
            'phase-zero trial is compared with current nominal row0. No phase-one historical comparison.')


def plans(requested_bit):
    profile = PROFILE
    plan = prior.validate_plan(prior.WritePlan(requested_bit))
    groups, stages, rows, calls = [], [], [], []
    names = GROUPS
    deltas = ((0.,0.), (0.,-1e-4) if requested_bit == 0 else (-1e-5,0.))
    for group_id, (name, delta) in enumerate(zip(names, deltas, strict=True)):
        group = dict(profile=profile, group_id=group_id, name=name, endpoint_delta=dict(alpha=delta[0], log10_temperature=delta[1]),
            status='not_started', row_ids=[2*group_id, 2*group_id+1],
            stage_ids=[2*group_id, 2*group_id+1], initial_state_sha256=None, terminal_state_sha256=None)
        groups.append(group)
        for local_id, canonical in enumerate(prior.stage_plan(plan)):
            stage = copy.deepcopy(canonical)
            stage.update(profile=profile, stage_id=len(stages), group_id=group_id, group=name, local_stage_id=local_id,
                nominal_controls=canonical['controls'].copy(), historical_comparison=None,
                historical_comparison_scope='nominal_phase0_only' if group_id == 0 else 'not_applicable_perturbed_control')
            if local_id == 0:
                for key, value in group['endpoint_delta'].items(): stage['controls'][key] += value
            stages.append(stage)
            for category, step, cached in [('prefix', 0, None), ('comparison', 0, True), ('comparison', 0, False)]+[
                    ('update', step, True) for step in range(1, stage['steps']+1)]:
                calls.append(dict(profile=profile, call=len(calls)+1, group_id=group_id, group=name, stage_id=stage['stage_id'],
                    stage=stage['name'], category=category, step=step, cached=cached, status='not_started'))
        for start in range(2):
            rows.append(dict(profile=profile, row_id=len(rows), group_id=group_id, group=name, batch_index=start,
                message=plan.messages[start], requested_bit=requested_bit, source_reference_index=source_index(requested_bit),
                source_reference_label=('ad', 'td')[source_index(requested_bit)], source_phase=start, status='not_started',
                parser_outcome='not_started', public_action=None, realized_stages=[dict(profile=profile, stage_id=s['stage_id'],
                    name=s['name'], steps=s['steps'], controls=s['controls'].copy()) for s in stages[-2:]],
                dispatch_count=0, completed_updates=0, readout=None, outcome='not_evaluated',
                historical_comparison=None, historical_comparison_scope=(
                    'nominal_phase0_only' if group_id == 0 and start == 0 else
                    'not_applicable_phase1' if start == 1 else 'not_applicable_perturbed_control')))
    if len(calls) != MAX_CALLS or sum(s['steps'] for s in stages) != MAX_UPDATES:
        raise ValueError('fixed two-phase schedule changed')
    return plan, groups, stages, rows, calls


def derive_design(requested_bit):
    plan, groups, stages, rows, calls = plans(requested_bit)
    envelope = 2*prior.artifact_envelope()['estimated_uncompressed_envelope_bytes']
    if envelope+prior.FAILED_RETURN_BYTES+16384 >= WORKING_OUTPUT:
        raise ValueError('uncompressed evidence does not fit working allowance')
    return dict(schema_version=1, profile=PROFILE,
        evidence='adapted exploratory public-controller two-phase challenge', evidence_rung=0,
        requested_bit=requested_bit, public_messages=list(plan.messages),
        source_reference_index=source_index(requested_bit), source_reference_label=('ad','td')[source_index(requested_bit)],
        source_phases=[0,1], nominal_historical_trial_id=(1,2)[requested_bit],
        groups=groups, realized_stages=[{k:s[k] for k in ('profile','stage_id','group_id','name','steps',
            'controls','nominal_controls')} for s in stages],
        schedule=dict(groups=2,batch_size=2,rows=4,stages=4,endpoint_steps=120,hold_steps=240,
            batch_updates=720,row_updates=1440,prefix_calls=4,comparison_calls=8,total_calls=732,
            reference_initializations=2,between_group_resets=1),
        limits=dict(model_calls=MAX_CALLS,internal_seconds=MAX_SECONDS,external_seconds=EXTERNAL_SECONDS,
            termination_grace_seconds=GRACE,threads=1,peak_rss_bytes=MAX_RSS,external_vm_bytes=MAX_VM,
            working_raw_bytes=WORKING_OUTPUT,finalization_reserve_bytes=FINAL_RESERVE,total_raw_bytes=MAX_OUTPUT,
            archive_bytes=ARCHIVE_CAP,transfer_reservation_bytes=TRANSFER_RESERVATION,source_upload_bytes=SOURCE_UPLOAD_CAP),
        artifact_envelope=dict(estimated_uncompressed_envelope_bytes=envelope,
            failed_return_reserved_bytes=prior.FAILED_RETURN_BYTES,
            scope='Two original-format complete groups plus one failed native-shape return; no compression assumption.'),
        map=dict(model=reference.MODEL,revision=reference.REVISION,prefix_ids=reference.PREFIX_IDS,
            hold=memory.writes.HOLD,endpoints=memory.writes.ENDPOINTS),
        precision_scope=PRECISION_SCOPE,diagnostic_tolerance=prior.DIAGNOSTIC_TOLERANCE,
        nominal_trajectory_tolerance=prior.TRAJECTORY_TOLERANCE,
        initial_state_scope='Literal saved reference phases0 and1 of the opposite-bit cycle; no advancement or settling.',
        historical_scope='Only nominal row0 compares against matching scalar phase0 trial; phase1 has no historical scalar equivalent.',
        stopping='Fixed two groups, no required offset failure, retry, extension or outcome-selected control. Finite wrong, '
            'unresolved, diagnostic and nominal-equivalence failures remain outcomes; structural/precision/cache/resource errors abort.',
        scope='Two phases of one source cycle at two discrete public control settings. No all-phase reliability, '
            'continuous tolerance, language-interface recovery, hidden channel or speedup claim.',
        timing_scope='Record actual elapsed time and calls. The protocol describes a historical proportional '
            'planning estimate only; no matched timing, deadline prediction or achieved speedup claim.')


class Recorder(memory.Recorder):
    def admit(self, size, final=False):
        if self.size()+size > (MAX_OUTPUT if final or self.finalizing else WORKING_OUTPUT):
            raise RuntimeError('raw artifact limit reached')


class Budget:
    def __init__(self, recorder, slots, requested_bit):
        self.recorder, self.slots, self.requested_bit = recorder, slots, requested_bit
        self.profile = PROFILE
        _, _, self.expected_stages, _, self.expected_slots = plans(requested_bit)
        if slots != self.expected_slots: raise ValueError('call slots differ from fixed profile schedule')
        self.calls = 0; self.counts = dict(prefix=0, comparison=0, update=0)
        self.stage_id = None; self.max_rss = MAX_RSS

    def check(self):
        if self.recorder.clock()-self.recorder.started >= MAX_SECONDS: raise TimeoutError('900-second internal cap')
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


def preflight(args, qualifier=memory.qualify_inputs, historical_qualifier=qualify_phase_history):
    """Qualify immutable inputs; no weights, model load or numerical forward."""
    plan = plans(args.requested_bit)[0]
    refs, provenance = qualifier(args)
    initial = selected_initials(refs, plan.requested_bit)
    _, historical = historical_qualifier(args, refs, plan=plan)
    return dict(status='passed',profile=PROFILE,model_calls=0,model_loaded=False,
        design=derive_design(plan.requested_bit),qualified_inputs=provenance,historical_inputs=historical,
        reference_initial_state_sha256=prior.array_hash(initial),initial_rows=initial_slots(refs,plan.requested_bit))


classify_readout = shared.classify_readout
immutable_group_artifacts = shared.immutable_group_artifacts


def compare_phase0(states, tokens, history, local_stage, bit):
    """Compare only batch row0; phase1 is deliberately outside this comparator."""
    old = history[local_stage]; count = len(states)
    if states.shape != (count,2,896) or tokens.shape != (count-1,2) or len(old['states']) < count:
        raise ValueError('historical comparison shape/horizon mismatch')
    actual, previous = states[:,0], old['states'][:count]
    mismatch = np.flatnonzero(tokens[:,0] != old['token_ids'][:count-1])
    gap = float(np.max(np.abs(actual-previous)))
    row = dict(row_id=0,batch_index=0,source_phase=0,historical_trial_id=(1,2)[bit],
        compared_states=count,compared_updates=count-1,maximum_coordinate_gap=gap,
        coordinate_tolerance=prior.TRAJECTORY_TOLERANCE,
        token_mismatch_update_indices=(mismatch+1).tolist(),token_mismatch_count=len(mismatch),
        state_bytes_equal=actual.tobytes()==previous.tobytes(),
        token_bytes_equal=tokens[:,0].tobytes()==old['token_ids'][:count-1].tobytes(),
        passed=gap<=prior.TRAJECTORY_TOLERANCE and not len(mismatch))
    return dict(rows=[row],passed=row['passed'],scope='nominal_phase0_only')


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
                    stage['historical_comparison'] = compare_phase0(saved, ids, histories, stage['local_stage_id'], plan.requested_bit)
            recorder.event('stage_finished', _final=True, stage_id=stage['stage_id'], group_id=stage['group_id'],
                status=stage['status'], completed_steps=stage['completed_steps'])
    if stage['status'] != 'completed': raise ValueError('stage failed final cache gate')


def run(args, backend_factory=prior.Backend, qualifier=memory.qualify_inputs,
        historical_qualifier=qualify_phase_history, clock=time.monotonic,
        rss=reference.rss_bytes, install_signals=True):
    profile = PROFILE
    plan, groups, stages, rows, slots = plans(args.requested_bit)
    recorder = Recorder(args.output_dir, clock, rss); budget = Budget(recorder, slots, args.requested_bit)
    result = dict(schema_version=1, profile=profile, status='failed', requested_bit=plan.requested_bit, groups=groups,
        rows=rows, stages=stages, calls=slots, reference_initializations=0, between_group_resets=0,
        evidence=derive_design(plan.requested_bit)['evidence'], evidence_rung=0,
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
        recorder.json('design.json', derive_design(plan.requested_bit)); recorder.json('planned-rows.json', rows)
        recorder.json('planned-groups.json', groups); recorder.json('planned-stages.json', stages)
        recorder.json('planned-calls.json', slots)
        modules = (shared, prior, memory, contract, memory.writes, memory.recovery, memory.hold_test,
            reference, batch, prior.historical_source, prior.trajectory_source)
        paths = [Path(m.__file__) for m in modules]+[Path(__file__), ROOT/'docs/public-phase-challenge.md', ROOT/'tests/test_public_phase_challenge.py',
            ROOT/'docs/receipts/p6-writes.json', ROOT/'docs/receipts/p6-write-path-design.json',
            ROOT/'docs/receipts/public-control-tolerance.json',
            ROOT/'pyproject.toml', ROOT/'uv.lock']
        result['source'] = dict(utc=reference.utc_now(), source_commit=subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
            files=[memory.binding(p) for p in paths], arguments={**{k:str(v) for k,v in vars(args).items()}, 'profile':profile})
        recorder.json('source.json', result['source'])
        refs, provenance = qualifier(args); initial = selected_initials(refs, plan.requested_bit)
        recorder.json('qualified-inputs.json', provenance)
        histories, historical = historical_qualifier(args, refs, plan=plan)
        recorder.json('historical-inputs.json', historical)
        recorder.json('phase-inputs.json', dict(initial_rows=initial_slots(refs, plan.requested_bit),
            batch_initial_sha256=prior.array_hash(initial), reference_source=provenance))
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
        state = initial.copy(); stages[0]['initial'] = recorder.state('stage-00-initial.npy', state)
        recorder.event('inputs_frozen', initial_sha256=prior.array_hash(state), historical_source=historical.get('source_commit'))
        budget.check(); started = clock(); backend = backend_factory(args, recorder, budget)
        result['cold_initialization_seconds'] = clock()-started
        if hasattr(backend, 'bindings'): recorder.json('model-inputs.json', backend.bindings)
        for group in groups:
            state = initial.copy(); result['reference_initializations'] += 1
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
                if group['group_id'] == 0 and index == 0:
                    comparisons = [stages[k]['historical_comparison']['rows'][index] for k in group['stage_ids']]
                    old_label = historical['selected_control']['neutral_readout']['qualified_label']
                    row['historical_comparison'] = dict(historical_trial_id=(1,2)[plan.requested_bit],compared_states=361, compared_updates=360,
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
                result['nominal_equivalence_passed'] = group['diagnostics_passed'] and rows[0]['historical_comparison']['passed']
            group['artifacts'] = immutable_group_artifacts(recorder.directory, group, stages, slots)
            group['status'] = 'completed'
            try:
                recorder.json(f'group-{group["group_id"]:02d}.json', group)
            except BaseException:
                group['status'] = 'failed'
                raise
            recorder.event('group_finished', group_id=group['group_id'], status=group['status'])
        if budget.calls != MAX_CALLS or budget.counts != dict(prefix=4, comparison=8, update=720):
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
