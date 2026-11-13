#!/usr/bin/env python3
"""Audit fresh fixed-horizon state readout; negative outcomes remain outcomes."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np

import audit_p6_readback as context
from audit_adaptive_readback import check_calls
from audit_readout_horizons import curve
from audit_shadow_readout import check_shadow
from evaluate_readout_policy import proposal
from validate_qwen_map import PREFIX_IDS

common = context.common
record = common.independent.file_record
REFERENCE_HASH = 'd96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0'
SEED = 2026102602
LIMITS = {'rows': 8, 'updates_per_row': 600, 'model_calls': 4817,
          'internal_seconds': 2700, 'external_seconds': 2760,
          'termination_grace_seconds': 30, 'threads': 1,
          'peak_rss_bytes': 8 * 1024**3, 'virtual_memory_bytes': 16 * 1024**3,
          'transfer_reservation_bytes': 512 * 1024**2}
POLICY_NUMBERS = {'checkpoint_interval': 12, 'consecutive_qualified': 3,
                  'maximum_steps_per_packet': 600, 'initial_state_observed': False,
                  'period': 6, 'closure_l2_tolerance': 1e-8,
                  'proper_divisors': [1, 2, 3],
                  'proper_divisor_maximum_phase_l2_gap_must_exceed': 1e-4,
                  'reference_match_maximum_phase_l2_tolerance': 1e-6,
                  'unique_reference_match_required': True}


def read_json(path):
    def reject(value):
        raise ValueError('non-finite JSON value: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def equal(actual, expected, label):
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError(label + ' differs')


def finite(value, label, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
        raise ValueError('invalid ' + label)
    return value


def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def check_design(packet, design_path, references, bindings):
    design = read_json(design_path)
    bindings += [record(design_path), record(packet, design['packet']),
                 record(references, {'sha256': REFERENCE_HASH}), record(references, design['references'])]
    for key, expected in {'schema_version': 1, 'design_name': 'fresh-uniform-readout-v1',
                          'generator': 'numpy.random.Generator(numpy.random.PCG64)', 'seed': SEED,
                          'endpoints': [['ad', 0], ['td', 3]], 'controls': common.HOLD,
                          'distribution': {'name': 'uniform', 'low': 0., 'high': 1., 'size': 8,
                                           'finite_generator_interval': '[0,1)', 'dtype': 'float64'},
                          'limits': LIMITS}.items():
        equal(design[key], expected, 'fresh design ' + key)
    equal({key: design['policy'][key] for key in POLICY_NUMBERS}, POLICY_NUMBERS, 'readout policy')
    weights = np.random.Generator(np.random.PCG64(SEED)).uniform(0.0, 1.0, 8)
    equal(design['weights'], weights.tolist(), 'unfiltered draw-order weights')
    equal(design['weight_hex'], [float(value).hex() for value in weights], 'hexadecimal weights')
    with np.load(references, allow_pickle=False) as archive:
        refs = archive['states']
    if refs.shape != (2, 6, 896) or refs.dtype != np.float64 or not np.isfinite(refs).all():
        raise ValueError('invalid reference-cycle schema')
    with np.load(packet, allow_pickle=False) as archive:
        if set(archive.files) != {'states'}:
            raise ValueError('packet must contain states only, without labels/design')
        states = archive['states']
    expected = np.stack([(1.0 - weight) * refs[0, 0] + weight * refs[1, 3] for weight in weights])
    if states.shape != (8, 896) or states.dtype != np.float64 or not np.isfinite(states).all() or not np.array_equal(states, expected):
        raise ValueError('fresh packet differs from exact prescribed interpolation')
    source = design['source']
    if re.fullmatch('[0-9a-f]{40}', source['source_commit']) is None:
        raise ValueError('invalid preparation source commit')
    sources = {}
    for item in source['source_files']:
        name = Path(item['path']).name
        if name in sources or not digest(item['sha256']) or type(item['bytes']) is not int or item['bytes'] < 0:
            raise ValueError('invalid preparation source bindings')
        sources[name] = item['sha256']
    if not {'run_p6_readback.py', 'streaming_readout.py', 'audit_p6_recovery.py'} <= sources.keys():
        raise ValueError('missing runner/observer source bindings')
    preparation = Path(design_path).with_name('preparation.json')
    if preparation.exists():
        bindings.append(record(preparation))
        receipt = read_json(preparation)
        if receipt['status'] != 'prepared' or type(receipt['model_forward_calls']) is not int or receipt['model_forward_calls'] != 0 or receipt['reference_sha256'] != REFERENCE_HASH or receipt['source_commit'] != source['source_commit']:
            raise ValueError('preparation receipt differs')
        bindings += [record(packet, receipt['packet']), record(design_path, receipt['design'])]
    return design, weights, states, refs, sources


def check_events(events, raw_rows, elapsed_seconds):
    count = check_calls(events, [600] * 8)
    if count != 4817:
        raise ValueError('fixed-horizon model call count differs')
    previous = -1.
    for event in events:
        elapsed = finite(event['elapsed_seconds'], 'event time', LIMITS['external_seconds'])
        if elapsed < previous:
            raise ValueError('event time moved backwards')
        previous = elapsed
        rss = finite(event['peak_rss_bytes'], 'event RSS')
        if rss >= LIMITS['peak_rss_bytes']:
            raise ValueError('event RSS cap exceeded')
        if event['event'] == 'model_call_started' and elapsed >= 2700:
            raise ValueError('model call started outside the internal budget')
        if event['event'] != 'finished' and elapsed > elapsed_seconds:
            raise ValueError('nonfinal event exceeds recorded execution time')
    finished = [event for event in events if event['event'] == 'finished']
    if len(finished) != 1 or finished[0]['status'] != 'completed' or events[-1] is not finished[0]:
        raise ValueError('missing or nonfinal completion event')
    prefixes = [event for event in events if event['event'] == 'prefix_ready']
    if len(prefixes) != 1 or prefixes[0]['mode'] != 'native64' or prefixes[0]['call'] != 1 or prefixes[0]['length'] != len(PREFIX_IDS):
        raise ValueError('native prefix preparation differs')
    finite(prefixes[0]['seconds'], 'prefix duration', elapsed_seconds)
    ended = [event for event in events if event['event'] == 'packet_finished']
    if len(ended) != 8:
        raise ValueError('missing/repeated packet completion event')
    for i, event in enumerate(ended):
        equal({key: event[key] for key in raw_rows[i]}, raw_rows[i], f'packet {i} completion event')
    return {'status': 'verified', 'model_calls': count, 'executed_updates': 4800,
            'prefix_calls': 1, 'cache_check_calls': 16, 'packet_completion_events': len(ended)}


def audit(base, packet, design, references, *, design_checker=None):
    base, packet, design, references = map(Path, (base, packet, design, references))
    bindings = []
    checker = check_design if design_checker is None else design_checker
    specification, weights, starts, refs, source_hashes = checker(packet, design, references, bindings)
    for name in ('metadata.json', 'results.json', 'events.jsonl'):
        bindings.append(record(base / name))
    meta, result = read_json(base / 'metadata.json'), read_json(base / 'results.json')
    if any(meta.get(key, {}).get('sha256') != expected for key, expected in context.INPUT_HASHES.items()):
        raise ValueError('model/config/vector hashes differ from the fixed map')
    bindings.append(record(packet, meta['packet']))
    equal(meta['controls'], common.HOLD, 'common-hold controls')
    equal([meta['max_calls'], meta['max_seconds'], meta['max_rss_gib']], [4817, 2700, 8], 'runner limits')
    arguments = meta['arguments']
    equal(arguments['steps'], 600, 'declared fixed horizon')
    if arguments['packet_sha256'] != specification['packet']['sha256'] or not arguments['shadow_references']:
        raise ValueError('runner packet/reference arguments differ')
    if any(key in arguments for key in ('design', 'expected_labels', 'expected_bits', 'stopping_policy')):
        raise ValueError('runner received evaluator-only labels/design or stopping configuration')
    if not re.fullmatch('[0-9a-f]{40}', meta['source_commit']):
        raise ValueError('invalid execution source commit')
    if set(meta['source_hashes']) != {'run_p6_readback.py', 'validate_qwen_map.py', 'check_p6_writes.py'} or any(not digest(value) for value in meta['source_hashes'].values()):
        raise ValueError('invalid runner source hashes')
    if meta['source_hashes']['run_p6_readback.py'] != source_hashes['run_p6_readback.py']:
        raise ValueError('runner source differs from preparation')
    definition = meta['shadow_readout']
    bindings.append(record(references, definition['references']))
    if set(definition['source_hashes']) != {'streaming_readout.py', 'audit_p6_recovery.py'} or any(definition['source_hashes'][key] != source_hashes[key] for key in definition['source_hashes']):
        raise ValueError('observer source differs from preparation')
    if definition['scope'] != 'Shadow observation only; full fixed horizon remains unchanged.':
        raise ValueError('not the fixed-horizon shadow runner')
    for group, keys in {'rotary': ('input', 'inverse_frequency', 'angles', 'cos', 'sin'),
                        'norm': ('input', 'reduction', 'weight', 'output')}.items():
        if any(result.get('dtype_observations', {}).get(group, {}).get(key) != 'torch.float64' for key in keys):
            raise ValueError('native float64 operation observations missing')
    # A finite overrun is a resource failure, not a reason to discard otherwise
    # inspectable completed rows from an interrupted attempt.
    elapsed = finite(result['elapsed_seconds'], 'whole-run elapsed seconds')
    peak = finite(result['peak_rss_bytes'], 'whole-run peak RSS')
    resource_passed = elapsed <= LIMITS['internal_seconds'] and peak < LIMITS['peak_rss_bytes']
    rows = []
    for i in range(8):
        row = {'packet_index': i, 'weight': float(weights[i])}
        rows.append(row)
        try:
            raw = result['rows'][i]
            stage = raw['stage']
            equal(raw['packet_index'], i, 'packet index')
            if stage['name'] != 'readback' or type(stage['steps']) is not int or stage['steps'] != 600 or stage['controls'] != common.HOLD:
                raise ValueError('row schedule differs from full 600 updates')
            checked, trajectory = common.audit_stage(base, i, stage, bindings, expected_steps=600)
            if checked['status'] != 'verified' or not np.array_equal(trajectory['states'][0], starts[i]):
                raise ValueError('incomplete trajectory or changed initial state')
            # audit_stage has already verified this artifact's binding and its
            # unchanged stage-start input. Its cached forward is repeated as
            # the first rollout update, including the retained top-token ID.
            with np.load(base / f'trial-{i:02d}-readback-cached.npz', allow_pickle=False) as archive:
                cached_first_token = int(np.argmax(archive['probabilities']))
            if cached_first_token != int(trajectory['token_ids'][0]):
                raise ValueError('cached probability argmax differs from first saved token')
            shadow = check_shadow(base, i, raw['shadow_readout'], trajectory['states'], refs, definition, bindings)
            grid = curve(trajectory['states'], trajectory['step_seconds'], refs)
            policy = proposal(grid['grid'])
            outcome = common.independent.recompute(trajectory['states'][1:], refs, 0)
            outcome.pop('classification')  # No desired label was assigned to these fresh states.
            label = outcome['reference_matches'][0] if outcome['cycle_gate_passed'] and len(outcome['reference_matches']) == 1 else None
            if label != policy['final_label'] or shadow['decision'] != policy['decision']:
                raise ValueError('full final gate/proposal disagrees with shadow audit')
            static = context.static_decode(starts[i], refs)
            decision = policy['decision']
            later = [] if decision is None else [point for point in grid['grid'] if point['steps'] > decision['steps']]
            row.update(status='verified', stage=checked, outcome=outcome, qualified_state=label,
                       first_token_matches_cached_argmax=True, first_token=cached_first_token,
                       static_baseline=static, proposed_policy=policy, shadow=shadow,
                       decision_matches_qualified_final=None if decision is None or label is None else decision['label'] == label,
                       static_nearest_matches_qualified_final=None if static['nearest_label'] is None or label is None else static['nearest_label'] == label,
                       later_unresolved_steps=[point['steps'] for point in later if point['qualified_label'] is None],
                       later_different_label_steps=[point['steps'] for point in later if point['qualified_label'] not in (None, decision['label'])],
                       horizon_diagnostics=grid, saved_forward_seconds=float(trajectory['step_seconds'].sum()),
                       hypothetical_unused_updates=policy['remaining_updates_after_decision'])
            if row['saved_forward_seconds'] > elapsed or shadow['observer_seconds'] > elapsed:
                raise ValueError('row durations exceed whole-run duration')
        except Exception as error:
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
    call_audit = {'status': 'failed'}
    try:
        raw_events = (base / 'events.jsonl').read_text()
        if not raw_events.endswith('\n'):
            raise ValueError('incomplete last event line')
        events = [json.loads(line) for line in raw_events.splitlines()]
        call_audit = check_events(events, result['rows'], elapsed)
    except Exception as error:
        call_audit['error'] = f'{type(error).__name__}: {error}'
    inventory = {'status': 'failed'}
    try:
        bound = {Path(item['path']).resolve() for item in bindings}
        files = {path.resolve() for path in base.glob('trial-*')}
        if files != {path for path in bound if path.parent == base.resolve() and path.name.startswith('trial-')}:
            raise ValueError('unbound or absent trial artifact')
        for item in bindings:
            record(item['path'], item)
        inventory = {'status': 'verified', 'trial_artifacts': len(files)}
    except Exception as error:
        inventory['error'] = f'{type(error).__name__}: {error}'
    verified = [row for row in rows if row['status'] == 'verified']
    measured_seconds = sum(row['saved_forward_seconds'] + row['shadow']['observer_seconds'] for row in verified)
    resource_passed = resource_passed and measured_seconds <= elapsed
    complete = (result['status'] == 'completed' and len(result['rows']) == 8 and len(verified) == 8
                and type(result['model_calls']) is int and result['model_calls'] == 4817
                and call_audit['status'] == 'verified' and inventory['status'] == 'verified' and resource_passed)
    comparable = [row for row in verified if row['decision_matches_qualified_final'] is not None]
    static_comparable = [row for row in verified if row['static_nearest_matches_qualified_final'] is not None]
    return {'status': 'audit_passed' if complete else 'audit_failed', 'rows': rows, 'bindings': bindings,
            'planned_rows': 8, 'verified_rows': len(verified), 'readback_steps': 600,
            'qualified_state_counts': {label: sum(row['qualified_state'] == label for row in verified) for label in ('ad', 'td', None)},
            'proposal_comparison': {'qualified_final_and_decision_denominator': len(comparable),
                'matching': sum(row['decision_matches_qualified_final'] for row in comparable),
                'wrong_against_qualified_final': sum(not row['decision_matches_qualified_final'] for row in comparable),
                'no_decision': sum(row['proposed_policy']['decision'] is None for row in verified),
                'final_unresolved_after_decision': sum(row['proposed_policy']['final_unresolved_after_decision'] for row in verified),
                'later_loss': sum(row['proposed_policy']['later_loss'] for row in verified),
                'later_label_change': sum(row['proposed_policy']['later_label_change'] for row in verified)},
            'static_comparison': {'qualified_final_and_nearest_denominator': len(static_comparable),
                'matching': sum(row['static_nearest_matches_qualified_final'] for row in static_comparable),
                'wrong_against_qualified_final': sum(not row['static_nearest_matches_qualified_final'] for row in static_comparable),
                'nearest_ties': sum(row['static_baseline']['nearest_label'] is None for row in verified)},
            'raw_call_audit': call_audit, 'artifact_inventory': inventory,
            'model_calls': result['model_calls'], 'executed_updates': sum(row['stage']['completed_steps'] for row in verified),
            'actual_avoided_updates': 0 if complete else None,
            'hypothetical_unused_updates_after_decisions': sum(row['hypothetical_unused_updates'] or 0 for row in verified),
            'observer_seconds': sum(row['shadow']['observer_seconds'] for row in verified),
            'saved_forward_seconds': sum(row['saved_forward_seconds'] for row in verified),
            'elapsed_seconds': elapsed, 'peak_rss_bytes': peak, 'resource_limits_passed': resource_passed,
            'producer_source_commit': meta['source_commit'], 'preparation_source_commit': specification['source']['source_commit'],
            'scope': 'Eight fresh draw-order states on one previously selected line. Full 600-update trajectories and all 4817 model calls were required for a complete audit; no actual early stopping or measured speedup here. Wrong, absent and unresolved proposed readouts are retained outcomes. Recorded precision/source/cache evidence is checked without model replay. No guarantee between twelve-update checkpoints, beyond 600 updates, of population reliability, fractality, secrecy or language-mediated communication.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'packet', 'design', 'references', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    with args.output.open('x', encoding='utf-8') as stream:
        try:
            report = audit(args.base, args.packet, args.design, args.references)
        except Exception as error:
            report = {'status': 'audit_failed', 'error': f'{type(error).__name__}: {error}'}
        receipt = {'utc': subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
                   'audit': report}
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status']}))
    return int(report['status'] != 'audit_passed')


if __name__ == '__main__':
    raise SystemExit(main())
