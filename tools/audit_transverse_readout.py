#!/usr/bin/env python3
"""Independently audit transverse preparation and fixed-horizon saved evidence."""
import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np

import audit_fresh_readout as fixed

common = fixed.common
record, read_json, equal = fixed.record, fixed.read_json, fixed.equal
REFERENCE_HASH = fixed.REFERENCE_HASH
SEED = 2026102603
DIRECTION_SEEDS = (2026102603, 2026102604)
LABELS = ('ad', 'td')
PRIOR_HASHES = {
    'coarse': 'b18a7b2f2583b1939d0b6f1bd0e2d0eb4fcddac96357761dcf8e93cb829a3a1e',
    'thirds': 'fd03d9831b6094a1be16aa6903dfd57b84dea9161e5df0c1c524e336cbaa4176',
    'nested': '07eb07e9c9a447f40063b979ec88dafd93486e37d6322ff526b112ada395a28e',
    'fresh': 'f48fc32007233cf4ec2b6b0d093e2fbf634ad3776d7570b6c43d885aa171f335',
}
RECEIPT_NAMES = dict(zip(PRIOR_HASHES, ('p6-phase-mixed-long.json', 'p6-transition-thirds.json',
                                     'p6-nested-thirds.json', 'fresh-readout.json')))
WEIGHTS = (7 / 27, 8 / 27, 0., 1.)
INTENDED = ('ad', 'td', 'ad', 'td')
GEOMETRY_GATES = {'unit_norm_absolute_tolerance': 1e-12,
                  'unit_orthogonality_absolute_tolerance': 1e-12,
                  'displacement_norm_relative_tolerance': 1e-10,
                  'displacement_projection_over_epsilon_tolerance': 1e-10}


def array(value, shape, label, dtype=np.float64):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
        raise ValueError('invalid ' + label)
    return value


def qualified(states, refs):
    outcome = common.independent.recompute(states, refs, 0)
    return (outcome['reference_matches'][0]
            if outcome['cycle_gate_passed'] and len(outcome['reference_matches']) == 1 else None)


def geometry(refs, relative_epsilon=0.001, *, direction_seed=SEED):
    """Reconstruct the stated first draw and scalar interpolation arithmetic."""
    if relative_epsilon not in (0.001, 0.003, 0.01):
        raise ValueError("unsupported transverse amplitude")
    if type(direction_seed) is not int or direction_seed not in DIRECTION_SEEDS:
        raise ValueError('unsupported transverse direction seed')
    if direction_seed != SEED and relative_epsilon != 0.01:
        raise ValueError('the additional direction requires relative epsilon 0.01')
    refs = array(refs, (2, 6, 896), 'reference cycles')
    a, b = refs[0, 0], refs[1, 3]
    delta = b - a
    norm = float(np.linalg.norm(delta))
    if norm == 0. or not np.isfinite(norm):
        raise ValueError('degenerate line direction')
    direction = delta / norm
    draw = np.random.Generator(np.random.PCG64(direction_seed)).normal(size=896)
    projected = draw - np.dot(draw, direction) * direction
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm == 0. or not np.isfinite(projected_norm):
        raise ValueError('degenerate transverse draw; no redraw permitted')
    q = projected / projected_norm
    rms = float(np.sqrt(np.mean(np.sum(refs * refs, axis=-1))))
    epsilon = relative_epsilon * rms
    if epsilon <= 0. or not np.isfinite(epsilon):
        raise ValueError('invalid transverse displacement scale')
    anchors = np.stack([(1.0 - w) * a + w * b for w in WEIGHTS])
    states = np.stack([anchor + sign * epsilon * q for anchor in anchors for sign in (-1, 1)])
    if abs(float(np.dot(q, direction))) > 1e-12 or abs(float(np.linalg.norm(q)) - 1.) > 1e-12:
        raise ValueError('transverse unit-vector gate failed')
    checks = []
    for i, state in enumerate(states):
        displacement = state - anchors[i // 2]
        length = float(np.linalg.norm(displacement))
        relative = abs(length / epsilon - 1.)
        projection = abs(float(np.dot(displacement, direction))) / epsilon
        if relative > 1e-10 or projection > 1e-10:
            raise ValueError('actual transverse displacement gate failed')
        checks.append({'displacement_norm': length, 'relative_norm_error': relative,
                       'normalized_line_projection': projection})
    if len({state.tobytes() for state in states}) != 8:
        raise ValueError('duplicate transverse packet state')
    displacements = states - np.repeat(anchors, 2, axis=0)
    saved_checks = {'direction_unit_norm_error': abs(float(np.linalg.norm(q)) - 1.),
                    'direction_line_absolute_dot': abs(float(np.dot(q, direction))),
                    'displacement_norm_relative_errors': (np.abs(np.linalg.norm(displacements, axis=1) - epsilon) / epsilon).tolist(),
                    'displacement_projection_over_epsilon': (np.abs(displacements @ direction) / epsilon).tolist(),
                    'line_length': norm, 'rms_reference_norm': rms, 'epsilon': epsilon, 'passed': True}
    if (saved_checks['direction_unit_norm_error'] > 1e-12
            or saved_checks['direction_line_absolute_dot'] > 1e-12
            or max(saved_checks['displacement_norm_relative_errors']) > 1e-10
            or max(saved_checks['displacement_projection_over_epsilon']) > 1e-10):
        raise ValueError('recorded-convention transverse geometry gate failed')
    return {'direction': direction, 'draw': draw, 'projected_draw': projected,
            'transverse_direction': q, 'reference_rms_norm': rms, 'epsilon': epsilon,
            'anchors': anchors, 'states': states, 'checks': checks, 'saved_checks': saved_checks}


def nearest_neighbor(state, states, labels):
    distances = np.linalg.norm(states - state, axis=1)
    if not len(distances) or not np.isfinite(distances).all():
        raise ValueError('invalid nearest-neighbor library/distances')
    minimum = float(distances.min())
    nearest = np.flatnonzero(distances == minimum).tolist()
    label = LABELS[int(labels[nearest[0]])] if len(nearest) == 1 else None
    return {'label': label, 'distances': distances.tolist(), 'nearest_indices': nearest,
            'minimum_distance': minimum, 'exact_tie': len(nearest) != 1}


def nearest_phase(state, refs):
    distances = np.linalg.norm(refs - state, axis=2)
    minimums = distances.min(axis=1)
    minimum = float(minimums.min())
    nearest = np.flatnonzero(minimums == minimum).tolist()
    return {'label': LABELS[nearest[0]] if len(nearest) == 1 else None,
            'phase_distances': distances.tolist(), 'label_distances': minimums.tolist(),
            'nearest_label_indices': nearest, 'minimum_distance': minimum,
            'exact_tie': len(nearest) != 1}


def deduplicate(candidates):
    """Exact coordinates, never a preferred outcome when a duplicate conflicts."""
    if not 1 <= len(candidates) <= 44:
        raise ValueError('library requires 1 through 44 raw qualified candidates')
    states, labels, entries = [], [], []
    for candidate in candidates:
        state = array(candidate['state'], (896,), 'library candidate state')
        label = candidate['label']
        if label not in LABELS:
            raise ValueError('unqualified library label')
        matches = [i for i, prior in enumerate(states) if np.array_equal(prior, state)]
        origin = candidate['origin']
        if matches:
            i = matches[0]
            if labels[i] != LABELS.index(label):
                raise ValueError('conflicting labels on an identical library state')
            entries[i]['origins'].append(origin)
        else:
            states.append(state.copy())
            labels.append(LABELS.index(label))
            entries.append({'library_index': len(states) - 1, 'label': label, 'origins': [origin]})
    return np.stack(states), np.asarray(labels, dtype=np.int8), entries


def prior_library(histories, refs, bindings):
    """Recheck every prior hash and recalculate labels from its final raw states."""
    if not isinstance(histories, list) or [history['name'] for history in histories] != list(PRIOR_HASHES):
        raise ValueError('historical cohorts missing, repeated or out of order')
    candidates, starts, summaries, excluded = [], [], [], []
    for history in histories:
        name = history['name']
        item = history['audit']
        path = Path(item['path'])
        bindings += [record(path, item), record(path, {'sha256': PRIOR_HASHES[name]})]
        receipt = history['receipt']
        if Path(receipt['path']).name != RECEIPT_NAMES[name]:
            raise ValueError('wrong historical receipt filename')
        bindings.append(record(receipt['path'], receipt))
        if not any(Path(entry['path']).name == 'complete-audit.json' and entry['sha256'] == PRIOR_HASHES[name]
                   for entry in read_json(receipt['path'])['bindings']):
            raise ValueError('historical receipt does not bind the prescribed audit')
        audit = read_json(path)['audit']
        if audit['status'] != 'audit_passed' or len(audit['rows']) != 8:
            raise ValueError('prior ' + name + ' audit is not complete')
        found = {}
        inventory = {}
        for binding in audit['bindings']:
            actual = record(binding['path'], binding)
            bindings.append(actual)
            found.setdefault(Path(actual['path']).name, {})[actual['path']] = actual
            inventory[actual['path']] = actual
        equal(history['verified_bindings'], list(inventory.values()), 'historical verified inventory')
        if not any(entry['sha256'] == REFERENCE_HASH for entry in inventory.values()):
            raise ValueError('prior audit does not bind these reference cycles')
        def unique(basename):
            values = found.get(basename, {})
            if len(values) != 1:
                raise ValueError('missing or ambiguous bound prior ' + basename)
            return Path(next(iter(values)))
        packet = unique('packet.npz')
        metadata = read_json(unique('metadata.json'))
        raw = read_json(unique('results.json'))
        for key, filename in (('packet', 'packet.npz'), ('metadata', 'metadata.json'), ('results', 'results.json')):
            equal(history[key], record(unique(filename)), 'historical ' + key + ' binding')
        if raw['status'] != 'completed' or raw['model_calls'] != 4817 or len(raw['rows']) != 8:
            raise ValueError('prior raw result is not the completed fixed horizon')
        equal(metadata['controls'], common.HOLD, 'prior common hold')
        equal(metadata['arguments']['steps'], 600, 'prior fixed horizon')
        if any(metadata.get(key, {}).get('sha256') != expected for key, expected in fixed.context.INPUT_HASHES.items()):
            raise ValueError('prior model/config/vector identity differs')
        record(packet, metadata['packet'])
        with np.load(packet, allow_pickle=False) as archive:
            if set(archive.files) != {'states'}:
                raise ValueError('prior packet must contain states only')
            packet_states = array(archive['states'], (8, 896), 'prior packet')
        labels, expected_rows = [], []
        for index, row in enumerate(audit['rows']):
            if row['status'] != 'verified' or type(row['packet_index']) is not int or row['packet_index'] != index:
                raise ValueError('prior audited row missing, reordered or unverified')
            stage = row['stage']
            if stage['status'] != 'verified' or stage['controls'] != common.HOLD or stage['completed_steps'] != 600:
                raise ValueError('prior audited stage is incomplete')
            final_path = unique(f'trial-{index:02d}-readback-final-600.npz')
            trajectory = common.load_trajectory(final_path, 600, 0, common.HOLD)
            if not np.array_equal(trajectory['states'][0], packet_states[index]):
                raise ValueError('prior raw initial state differs from packet')
            label = qualified(trajectory['states'][1:], refs)
            if row['qualified_state'] != label:
                raise ValueError('prior reported label differs from independently recomputed raw label')
            reported = row['outcome']
            reported_label = (reported['reference_matches'][0]
                              if reported['cycle_gate_passed'] is True and len(reported['reference_matches']) == 1 else None)
            if reported_label != label:
                raise ValueError('prior reported outcome disagrees with independently recomputed label')
            labels.append(label)
            starts.append(packet_states[index])
            expected_rows.append({'history': name, 'packet_index': index, 'qualified_label': label,
                                  'included_as_labelled_exemplar': label is not None,
                                  'trajectory': record(final_path)})
            origin = {'kind': 'historical_packet', 'history': name, 'packet_index': index}
            if label is not None:
                candidates.append({'state': packet_states[index], 'label': label,
                                   'origin': origin})
            else:
                excluded.append(origin)
        equal(history['rows'], expected_rows, 'historical raw-derived rows')
        summaries.append({'cohort': name, 'rows': 8, 'qualified': sum(x is not None for x in labels),
                          'unresolved': labels.count(None), 'qualified_labels': labels})
    for index, label in enumerate(LABELS):
        if qualified(np.concatenate([refs[index], refs[index]]), refs) != label:
            raise ValueError('reference cycle fails its own primitive-period qualification')
        for phase in range(6):
            candidates.append({'state': refs[index, phase], 'label': label,
                               'origin': {'kind': 'reference_phase', 'label': label, 'phase': phase}})
    library_states, library_labels, entries = deduplicate(candidates)
    return {'states': library_states, 'labels': library_labels, 'entries': entries,
            'all_prior_starts': np.stack(starts), 'prior_audits': summaries,
            'excluded_unresolved_origins': excluded,
            'qualified_candidates_before_deduplication': len(candidates)}


def inspect_preparation(packet, design_path, references, bindings):
    packet, design_path, references = map(Path, (packet, design_path, references))
    design = read_json(design_path)
    relative_epsilon = design.get('epsilon_relative_rms')
    if relative_epsilon not in (0.001, 0.003, 0.01):
        raise ValueError('unsupported transverse amplitude')
    direction_seed = design.get('seed')
    if type(direction_seed) is not int or direction_seed not in DIRECTION_SEEDS:
        raise ValueError('unsupported transverse direction seed')
    if direction_seed != SEED and relative_epsilon != 0.01:
        raise ValueError('the additional direction requires relative epsilon 0.01')
    bindings += [record(design_path), record(packet, design['packet']),
                 record(references, {'sha256': REFERENCE_HASH}), record(references, design['references'])]
    limits = dict(fixed.LIMITS)
    if direction_seed != SEED:
        limits['transfer_reservation_bytes'] = 384*1024**2
    for key, expected in {'schema_version': 1, 'design_name': 'transverse-readout-v1',
                          'generator': 'numpy.random.Generator(numpy.random.PCG64)', 'seed': direction_seed,
                          'endpoints': [['ad', 0], ['td', 3]], 'anchor_weights': list(WEIGHTS),
                          'anchor_weight_hex': [float(w).hex() for w in WEIGHTS],
                          'anchor_labels': list(INTENDED), 'signs': [-1, 1],
                          'epsilon_relative_rms': relative_epsilon, 'geometry_gates': GEOMETRY_GATES,
                          'controls': common.HOLD, 'limits': limits,
                          'policy': fixed.POLICY_NUMBERS, 'historical_packet_rows': 32}.items():
        equal(design[key], expected, 'transverse design ' + key)
    with np.load(references, allow_pickle=False) as archive:
        refs = array(archive['states'], (2, 6, 896), 'references')
    calculated = geometry(refs, relative_epsilon, direction_seed=direction_seed)
    library = prior_library(design['histories'], refs, bindings)

    def local(key, name):
        binding = design[key]
        if Path(binding['path']).name != name:
            raise ValueError('wrong preparation artifact filename: ' + key)
        path = design_path.parent / name
        bindings.append(record(path, binding))
        return path

    with np.load(packet, allow_pickle=False) as archive:
        if set(archive.files) != {'states'}:
            raise ValueError('receiver packet must contain states only')
        starts = array(archive['states'], (8, 896), 'transverse packet')
    if not np.array_equal(starts, calculated['states']):
        raise ValueError('transverse packet differs from prescribed independent construction')
    with np.load(local('geometry', 'geometry.npz'), allow_pickle=False) as archive:
        expected = {'gaussian': calculated['draw'], 'line_unit': calculated['direction'],
                    'direction': calculated['transverse_direction'], 'anchors': calculated['anchors'],
                    'epsilon': np.asarray(calculated['epsilon'], dtype=np.float64)}
        if set(archive.files) != set(expected):
            raise ValueError('geometry archive fields differ')
        for key, value in expected.items():
            if not np.array_equal(array(archive[key], value.shape, 'saved geometry ' + key), value):
                raise ValueError('saved geometry differs: ' + key)
    with np.load(local('library', 'library.npz'), allow_pickle=False) as archive:
        if set(archive.files) != {'states', 'labels'}:
            raise ValueError('library archive fields differ')
        for key in ('states', 'labels'):
            expected = library[key]
            value = array(archive[key], expected.shape, 'saved library ' + key, expected.dtype)
            if not np.array_equal(value, expected):
                raise ValueError('library differs from independently qualified prior states/labels')
    origins = [entry['origins'] for entry in library['entries']]
    equal(design['library_unique_count'], len(library['states']), 'unique library count')
    equal(design['library_origins'], origins, 'library origins')
    equal(design['excluded_unresolved_origins'], library['excluded_unresolved_origins'], 'excluded unresolved origins')
    equal(read_json(local('library_origin_artifact', 'library-origins.json')),
          {'origins': origins, 'excluded_unresolved_origins': library['excluded_unresolved_origins']}, 'library origin artifact')
    equal(design['geometry_checks'], calculated['saved_checks'], 'recorded geometry metrics')
    equal(read_json(local('geometry_check_artifact', 'geometry-checks.json')), calculated['saved_checks'], 'geometry metric artifact')
    prior_states = np.concatenate([library['all_prior_starts'], refs.reshape(12, 896)])
    if any(np.array_equal(state, old) for state in starts for old in prior_states):
        raise ValueError('new transverse state duplicates an earlier input; no replacement allowed')
    equal(design['novelty'], {'prior_rows_checked': 44, 'new_rows': 8,
                             'equal_to_prior': [], 'new_duplicate_pairs': []}, 'novelty record')
    for anchor_index, anchor in enumerate(calculated['anchors']):
        matches = [i for i, state in enumerate(library['states']) if np.array_equal(state, anchor)]
        if len(matches) != 1 or LABELS[int(library['labels'][matches[0]])] != INTENDED[anchor_index]:
            raise ValueError('anchor lacks the prescribed independently qualified historical label')
    expected_rows = []
    for index, state in enumerate(starts):
        nn = nearest_neighbor(state, library['states'], library['labels'])
        phase = nearest_phase(state, refs)
        expected_rows.append({'packet_index': index, 'anchor_index': index // 2,
            'sign': (-1, 1)[index % 2], 'stratum': 'interior' if index < 4 else 'near_orbit',
            'intended_label': INTENDED[index // 2],
            'one_nn': {'distances': nn['distances'], 'nearest_indices': nn['nearest_indices'], 'prediction': nn['label']},
            'nearest_phase': {'phase_distances': phase['phase_distances'], 'label_distances': phase['label_distances'],
                              'nearest_label_indices': phase['nearest_label_indices'], 'prediction': phase['label']}})
    equal(design['rows'], expected_rows, 'frozen rows and geometric predictions')
    source = design['source']
    if not re.fullmatch('[0-9a-f]{40}', source['source_commit']):
        raise ValueError('invalid preparation source commit')
    sources = {}
    for item in source['source_files']:
        name = Path(item['path']).name
        if name in sources or not fixed.digest(item['sha256']) or type(item['bytes']) is not int or item['bytes'] < 0:
            raise ValueError('invalid preparation source bindings')
        sources[name] = item['sha256']
    if not {'prepare_transverse_readout.py', 'run_p6_readback.py', 'streaming_readout.py', 'audit_p6_recovery.py'} <= sources.keys():
        raise ValueError('missing preparation/runner/observer source bindings')
    receipt_path = design_path.with_name('preparation.json')
    bindings.append(record(receipt_path))
    receipt = read_json(receipt_path)
    if (receipt['status'] != 'prepared' or type(receipt['model_forward_calls']) is not int
            or receipt['model_forward_calls'] != 0 or receipt['reference_sha256'] != REFERENCE_HASH
            or receipt['source_commit'] != source['source_commit'] or receipt['library_unique_count'] != len(library['states'])):
        raise ValueError('preparation receipt does not qualify completed zero-call preparation')
    for key, path in (('packet', packet), ('design', design_path),
                      ('geometry', design_path.with_name('geometry.npz')), ('library', design_path.with_name('library.npz'))):
        bindings.append(record(path, receipt[key]))
    # A replacement during a long prior-inventory pass must not be accepted.
    for binding in bindings:
        record(binding['path'], binding)
    summary = {'status': 'verified', 'new_rows': 8, 'historical_rows_checked': 32,
        'direction_seed': direction_seed, 'epsilon_relative_rms': relative_epsilon,
        'reference_phases': 12, 'library_unique_count': len(library['states']),
        'qualified_candidates_before_deduplication': library['qualified_candidates_before_deduplication'],
        'excluded_unresolved_origins': library['excluded_unresolved_origins'],
        'histories': library['prior_audits'], 'geometry': calculated['saved_checks'], 'rows': expected_rows,
        'model_forward_calls': 0, 'scope': 'Independent saved-array, geometry, historical-label and frozen-predictor audit; no model replay or new dynamical outcome.'}
    callback_values = (design, np.repeat(np.asarray(WEIGHTS), 2), starts, refs, sources)
    return callback_values, summary


def preparation_audit(packet, design, references):
    bindings = []
    _, summary = inspect_preparation(packet, design, references, bindings)
    return {'status': 'preparation_audit_passed', 'preparation': summary, 'bindings': bindings,
            'model_forward_calls': 0}


def outcome_summary(rows):
    """Preserve every attempt; wrong labels and erasures cannot fail integrity."""
    verified = [row for row in rows if row['status'] == 'verified']
    result = {'planned_rows': len(rows), 'verified_rows': len(verified), 'unverified_rows': len(rows) - len(verified),
              'intended_label_retained': sum(row['qualified_state'] == row['intended_label'] for row in verified),
              'wrong_qualified_label': sum(row['qualified_state'] is not None and row['qualified_state'] != row['intended_label'] for row in verified),
              'final_unresolved': sum(row['qualified_state'] is None for row in verified)}
    result['geometric_decoders'] = {}
    for name in ('one_nn', 'nearest_phase'):
        comparable = [row for row in verified if row[name]['prediction'] is not None and row['qualified_state'] is not None]
        result['geometric_decoders'][name] = {
            'qualified_final_and_prediction_denominator': len(comparable),
            'matching': sum(row[name]['prediction'] == row['qualified_state'] for row in comparable),
            'wrong_against_qualified_final': sum(row[name]['prediction'] != row['qualified_state'] for row in comparable),
            'prediction_erasure': sum(row[name]['prediction'] is None for row in verified),
            'final_unresolved': sum(row['qualified_state'] is None for row in verified)}
    decisions = [row for row in verified if row['proposed_policy']['decision'] is not None]
    result['first_decision'] = {'present': len(decisions), 'absent': len(verified) - len(decisions),
        'intended_label': sum(row['proposed_policy']['decision']['label'] == row['intended_label'] for row in decisions),
        'other_label': sum(row['proposed_policy']['decision']['label'] != row['intended_label'] for row in decisions),
        'steps': [row['proposed_policy']['decision']['steps'] for row in decisions],
        'later_loss': sum(row['proposed_policy']['later_loss'] for row in verified),
        'later_label_change': sum(row['proposed_policy']['later_label_change'] for row in verified)}
    return result


def audit(base, packet, design, references):
    preparation = {}
    def checker(packet, design_path, references, bindings):
        values, summary = inspect_preparation(packet, design_path, references, bindings)
        preparation.update(summary)
        return values
    report = fixed.audit(base, packet, design, references, design_checker=checker)
    for row, planned in zip(report['rows'], preparation['rows']):
        row['anchor_weight'] = row.pop('weight')
        row.update(planned)
        if row['status'] == 'verified':
            row['intended_label_retained'] = row['qualified_state'] == row['intended_label'] if row['qualified_state'] is not None else None
            decision = row['proposed_policy']['decision']
            point = next((point for point in row['horizon_diagnostics']['grid'] if decision is not None and point['steps'] == decision['steps']), None)
            row['first_decision_saved_forward_seconds'] = None if point is None else point['cumulative_saved_forward_seconds']
    report['preparation'] = preparation
    report['transverse_comparison'] = outcome_summary(report['rows'])
    report['strata'] = {name: outcome_summary([row for row in report['rows'] if row['stratum'] == name])
                       for name in ('interior', 'near_orbit')}
    report['scope'] = ('Adapted exploratory eight-state transverse test: four interior probes and four near-orbit controls, '
        'one fixed direction/amplitude and two known reference cycles. Full 600 updates and 4817 model calls; '
        'shadow decisions provide hypothetical latency only, no actual stopping savings. Intended-label retention '
        'and static-decoder disagreement are outcomes, not integrity gates. Saved evidence is rechecked without model replay. '
        'No basin-volume, fractality, shrimp-specificity, secrecy, population-reliability or language-channel claim.')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--preparation-only', action='store_true')
    for name in ('packet', 'design', 'references', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.preparation_only and args.base is None:
        parser.error('--base is required unless --preparation-only is selected')
    with args.output.open('x', encoding='utf-8') as stream:
        try:
            report = (preparation_audit(args.packet, args.design, args.references) if args.preparation_only
                      else audit(args.base, args.packet, args.design, args.references))
        except Exception as error:
            report = {'status': 'audit_failed', 'error': f'{type(error).__name__}: {error}'}
        receipt = {'utc': subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(), 'audit': report}
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status']}))
    return int(report['status'] not in ('audit_passed', 'preparation_audit_passed'))


if __name__ == '__main__':
    raise SystemExit(main())
