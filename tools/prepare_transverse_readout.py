#!/usr/bin/env python3
"""Freeze eight transverse probes and static predictions; never load a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np

import audit_p6_recovery as gates

ROOT = Path(__file__).resolve().parents[1]
RECEIPT_DIR = ROOT/'docs/receipts'
REFERENCE_SHA256 = 'd96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0'
HISTORY_PINS = {
    'coarse': {'receipt': 'p6-phase-mixed-long.json', 'sha256': 'b18a7b2f2583b1939d0b6f1bd0e2d0eb4fcddac96357761dcf8e93cb829a3a1e'},
    'thirds': {'receipt': 'p6-transition-thirds.json', 'sha256': 'fd03d9831b6094a1be16aa6903dfd57b84dea9161e5df0c1c524e336cbaa4176'},
    'nested': {'receipt': 'p6-nested-thirds.json', 'sha256': '07eb07e9c9a447f40063b979ec88dafd93486e37d6322ff526b112ada395a28e'},
    'fresh': {'receipt': 'fresh-readout.json', 'sha256': 'f48fc32007233cf4ec2b6b0d093e2fbf634ad3776d7570b6c43d885aa171f335'},
}
SEED, DIM = 2026102603, 896
DIRECTION_SEEDS = (SEED, 2026102604)
LABELS = ('ad', 'td')
HOLD = {'alpha': 0.07293697809562831, 'log10_temperature': -0.15395151820007266}
GEOMETRY_GATES = {'unit_norm_absolute_tolerance': 1e-12,
                  'unit_orthogonality_absolute_tolerance': 1e-12,
                  'displacement_norm_relative_tolerance': 1e-10,
                  'displacement_projection_over_epsilon_tolerance': 1e-10}
LIMITS = {'rows': 8, 'updates_per_row': 600, 'model_calls': 4817,
          'internal_seconds': 2700, 'external_seconds': 2760,
          'termination_grace_seconds': 30, 'threads': 1,
          'peak_rss_bytes': 8*1024**3, 'virtual_memory_bytes': 16*1024**3,
          'transfer_reservation_bytes': 512*1024**2}
POLICY = {'checkpoint_interval': 12, 'consecutive_qualified': 3,
          'maximum_steps_per_packet': 600, 'initial_state_observed': False,
          'period': 6, 'closure_l2_tolerance': 1e-8, 'proper_divisors': [1, 2, 3],
          'proper_divisor_maximum_phase_l2_gap_must_exceed': 1e-4,
          'reference_match_maximum_phase_l2_tolerance': 1e-6,
          'unique_reference_match_required': True}


def read_json(path):
    def reject(value):
        raise ValueError('nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def binding(path, expected=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('require a regular nonsymlink input: '+str(path))
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    item = {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest}
    if expected is not None:
        expected = {'sha256': expected} if isinstance(expected, str) else expected
        if digest != expected['sha256'] or ('bytes' in expected and item['bytes'] != expected['bytes']):
            raise ValueError('input hash/size mismatch: '+str(path))
    return item


def finite_array(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
        raise ValueError(name+' must be finite float64 with shape '+str(shape))
    return value


def qualified_label(outcome):
    matches = outcome['reference_matches']
    if any(label not in LABELS for label in matches):
        raise ValueError('unknown reference label')
    return matches[0] if outcome['cycle_gate_passed'] is True and len(matches) == 1 else None


def load_history(name, audit_path, refs):
    """Rehash the pinned audit's complete inventory and recompute final labels."""
    pin = HISTORY_PINS[name]
    receipt_path = RECEIPT_DIR/pin['receipt']
    receipt_binding = binding(receipt_path)
    receipt = read_json(receipt_path)
    if not any(Path(item['path']).name == 'complete-audit.json' and item['sha256'] == pin['sha256']
               for item in receipt['bindings']):
        raise ValueError('committed receipt does not bind the pinned audit')
    audit_binding = binding(audit_path, pin['sha256'])
    audit = read_json(audit_path)['audit']
    if audit['status'] != 'audit_passed' or len(audit['rows']) != 8:
        raise ValueError('require a passed eight-row historical audit')
    bound = {}
    for item in audit['bindings']:
        checked = binding(item['path'], item)
        previous = bound.get(checked['path'])
        if previous is not None and previous != checked:
            raise ValueError('conflicting historical binding')
        bound[checked['path']] = checked

    def one(basename):
        matches = [item for item in bound.values() if Path(item['path']).name == basename]
        if len(matches) != 1:
            raise ValueError('ambiguous/missing historical '+basename)
        return matches[0]

    packet, metadata, result_binding = (one(filename) for filename in ('packet.npz', 'metadata.json', 'results.json'))
    meta, result = read_json(metadata['path']), read_json(result_binding['path'])
    if meta['controls'] != HOLD or meta['arguments']['steps'] != 600 or meta['packet']['sha256'] != packet['sha256']:
        raise ValueError('historical packet/hold/horizon mismatch')
    if result['status'] != 'completed' or result['model_calls'] != 4817 or len(result['rows']) != 8:
        raise ValueError('historical run is not complete fixed-horizon evidence')
    if not any(item['sha256'] == REFERENCE_SHA256 for item in bound.values()):
        raise ValueError('historical audit does not bind the reference archive')
    with np.load(packet['path'], allow_pickle=False) as archive:
        if set(archive.files) != {'states'}:
            raise ValueError('historical packet is not state-only')
        starts = finite_array(archive['states'], (8, DIM), 'historical states').copy()
    rows = []
    for index, row in enumerate(audit['rows']):
        stage = row['stage']
        if (row['packet_index'] != index or row['status'] != 'verified' or stage['status'] != 'verified'
                or stage['controls'] != HOLD or stage['completed_steps'] != 600):
            raise ValueError('historical row failed its audited schedule')
        final = one(f'trial-{index:02d}-readback-final-600.npz')
        with np.load(final['path'], allow_pickle=False) as archive:
            states = finite_array(archive['states'], (601, DIM), 'historical trajectory')
            if not np.array_equal(states[0], starts[index]):
                raise ValueError('historical packet differs from saved initial state')
            if any(float(archive[key]) != value for key, value in HOLD.items()):
                raise ValueError('historical trajectory controls differ')
            outcome = gates.recompute(states[1:], refs, 0)
        label = qualified_label(outcome)
        if label != row['qualified_state'] or label != qualified_label(row['outcome']):
            raise ValueError('historical label disagrees with qualified raw trajectory')
        rows.append({'history': name, 'packet_index': index, 'qualified_label': label,
                     'included_as_labelled_exemplar': label is not None,
                     'trajectory': final})
    return starts, rows, {'name': name, 'receipt': receipt_binding, 'audit': audit_binding,
        'packet': packet, 'metadata': metadata, 'results': result_binding,
        'verified_bindings': list(bound.values()), 'rows': rows}


def build_library(histories, refs):
    states, labels, origins, excluded = [], [], [], []

    def add(state, label, origin):
        if label is None:
            excluded.append(origin)
            return
        for index, previous in enumerate(states):
            if np.array_equal(previous, state):
                if labels[index] != LABELS.index(label):
                    raise ValueError('identical library vector has conflicting labels')
                origins[index].append(origin)
                return
        states.append(state.copy()); labels.append(LABELS.index(label)); origins.append([origin])

    for starts, rows, _ in histories:
        for state, row in zip(starts, rows):
            add(state, row['qualified_label'], {'kind': 'historical_packet',
                'history': row['history'], 'packet_index': row['packet_index']})
    for label_index, label in enumerate(LABELS):
        for phase in range(6):
            add(refs[label_index, phase], label, {'kind': 'reference_phase', 'label': label, 'phase': phase})
    return np.stack(states), np.asarray(labels, dtype=np.int8), origins, excluded


def validate_direction(direction_seed, relative_epsilon):
    if type(direction_seed) is not int or direction_seed not in DIRECTION_SEEDS:
        raise ValueError('direction seed must be one of the two declared seeds')
    if direction_seed != SEED and relative_epsilon != 0.01:
        raise ValueError('the additional direction requires relative epsilon 0.01')


def _construct(refs, raw, relative_epsilon=0.001, direction_seed=SEED):
    validate_direction(direction_seed, relative_epsilon)
    if relative_epsilon not in (0.001, 0.003, 0.01):
        raise ValueError("relative epsilon must be one of the declared amplitudes")
    refs = finite_array(refs, (2, 6, DIM), 'references')
    a, b = refs[0, 0], refs[1, 3]
    delta = b-a
    length = float(np.linalg.norm(delta))
    if not np.isfinite(length) or length <= 0:
        raise ValueError('zero/nonfinite line length')
    line_unit = delta/length
    raw['line_unit'] = line_unit
    gaussian = np.random.Generator(np.random.PCG64(direction_seed)).normal(size=DIM)
    raw['gaussian'] = gaussian
    perpendicular = gaussian-np.dot(gaussian, line_unit)*line_unit
    norm = float(np.linalg.norm(perpendicular))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('zero/nonfinite transverse draw')
    direction = perpendicular/norm
    raw['direction'] = direction
    rms = float(np.sqrt(np.mean(np.sum(refs*refs, axis=-1))))
    epsilon = relative_epsilon*rms
    raw['epsilon'] = np.asarray(epsilon, dtype=np.float64)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('nonpositive/nonfinite amplitude')
    anchors = np.stack([(1.0-7/27)*a+(7/27)*b, (1.0-8/27)*a+(8/27)*b, a, b])
    raw['anchors'] = anchors
    states = np.stack([anchor+sign*epsilon*direction for anchor in anchors for sign in (-1, 1)])
    finite_array(states, (8, DIM), 'transverse states')
    displacements = states-np.repeat(anchors, 2, axis=0)
    norms = np.linalg.norm(displacements, axis=1)
    projections = np.abs(displacements@line_unit)/epsilon
    check = {'direction_unit_norm_error': abs(float(np.linalg.norm(direction))-1.0),
             'direction_line_absolute_dot': abs(float(np.dot(direction, line_unit))),
             'displacement_norm_relative_errors': (np.abs(norms-epsilon)/epsilon).tolist(),
             'displacement_projection_over_epsilon': projections.tolist(),
             'line_length': length, 'rms_reference_norm': rms, 'epsilon': epsilon}
    check['passed'] = not (check['direction_unit_norm_error'] > GEOMETRY_GATES['unit_norm_absolute_tolerance']
            or check['direction_line_absolute_dot'] > GEOMETRY_GATES['unit_orthogonality_absolute_tolerance']
            or max(check['displacement_norm_relative_errors']) > GEOMETRY_GATES['displacement_norm_relative_tolerance']
            or max(check['displacement_projection_over_epsilon']) > GEOMETRY_GATES['displacement_projection_over_epsilon_tolerance'])
    return states, raw, check


def construct(refs, preserve=None, *, relative_epsilon=0.001, direction_seed=SEED):
    raw = {}
    try:
        return _construct(refs, raw, relative_epsilon, direction_seed)
    finally:
        if preserve is not None and raw:
            preserve(raw)


def predictions(states, library, labels, refs):
    rows = []
    for index, state in enumerate(states):
        distances = np.linalg.norm(library-state, axis=1)
        phase_distances = np.linalg.norm(refs-state, axis=2)
        if not np.isfinite(distances).all() or not np.isfinite(phase_distances).all():
            raise ValueError('nonfinite static distance')
        nearest = np.flatnonzero(distances == distances.min()).tolist()
        per_label = phase_distances.min(axis=1)
        tied_labels = np.flatnonzero(per_label == per_label.min()).tolist()
        rows.append({'packet_index': index,
            'one_nn': {'distances': distances.tolist(), 'nearest_indices': nearest,
                       'prediction': LABELS[int(labels[nearest[0]])] if len(nearest) == 1 else None},
            'nearest_phase': {'phase_distances': phase_distances.tolist(), 'label_distances': per_label.tolist(),
                              'nearest_label_indices': tied_labels,
                              'prediction': LABELS[tied_labels[0]] if len(tied_labels) == 1 else None}})
    return rows


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def save_arrays(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    return binding(path)


def provenance():
    names = ['tools/prepare_transverse_readout.py', 'tools/run_p6_readback.py',
             'tools/validate_qwen_map.py', 'tools/check_p6_writes.py',
             'tools/streaming_readout.py', 'tools/audit_p6_recovery.py',
             'docs/transverse-readout-protocol.md', 'pyproject.toml', 'uv.lock']
    return {'utc': subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
            'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
            'numpy_version': np.__version__, 'source_files': [binding(ROOT/name) for name in names]}


def prepare_contents(references, audit_paths, output, progress, *, relative_epsilon=0.001, direction_seed=SEED):
    validate_direction(direction_seed, relative_epsilon)
    if set(audit_paths) != set(HISTORY_PINS):
        raise ValueError('all four prescribed prior audits are required')
    progress['stage'] = 'references'
    reference = binding(references, REFERENCE_SHA256)
    progress['reference'] = reference
    with np.load(references, allow_pickle=False) as archive:
        refs = finite_array(archive['states'], (2, 6, DIM), 'references').copy()
    histories = []
    for name in HISTORY_PINS:
        progress['stage'] = 'history_'+name
        loaded = load_history(name, audit_paths[name], refs)
        histories.append(loaded)
        progress.setdefault('completed_histories', []).append(loaded[2])
    progress['stage'] = 'library'
    library, labels, origins, excluded = build_library(histories, refs)
    lib = save_arrays(output/'library.npz', states=library, labels=labels)
    write_json(output/'library-origins.json', {'origins': origins, 'excluded_unresolved_origins': excluded})
    progress['stage'] = 'geometry'
    geometry_binding = {}
    def preserve_geometry(raw):
        geometry_binding.update(save_arrays(output/'geometry.npz', **raw))
    states, geometry, checks = construct(refs, preserve_geometry, relative_epsilon=relative_epsilon,
                                        direction_seed=direction_seed)
    packet = save_arrays(output/'packet.npz', states=states)
    geom = dict(geometry_binding)
    write_json(output/'geometry-checks.json', checks)
    if not checks['passed']:
        raise ValueError('fixed transverse geometry gate failed')
    progress['stage'] = 'novelty_and_anchor_labels'
    prior_states = np.concatenate([item[0] for item in histories]+[refs.reshape(12, DIM)])
    for index, state in enumerate(states):
        if any(np.array_equal(state, old) for old in prior_states) or any(np.array_equal(state, old) for old in states[:index]):
            raise ValueError('new input duplicates an earlier or newly planned input; no redraw permitted')
    intended = ['ad', 'td', 'ad', 'td']
    for index, anchor in enumerate(geometry['anchors']):
        matches = [i for i, vector in enumerate(library) if np.array_equal(anchor, vector)]
        if len(matches) != 1 or LABELS[int(labels[matches[0]])] != intended[index]:
            raise ValueError('anchor lacks its prescribed qualified historical label')
    progress['stage'] = 'static_predictions'
    static = predictions(states, library, labels, refs)
    source = provenance()
    progress['stage'] = 'final_input_recheck'
    # Preserve generated arrays even when this final source-stability check fails.
    for _, _, history in histories:
        for item in [history['receipt'], history['audit']]+history['verified_bindings']:
            binding(item['path'], item)
    binding(references, reference)
    with np.load(output/'packet.npz', allow_pickle=False) as archive:
        if set(archive.files) != {'states'} or not np.array_equal(archive['states'], states):
            raise ValueError('written receiver packet differs')
    rows = [{'packet_index': index, 'anchor_index': index//2, 'sign': (-1, 1)[index%2],
             'stratum': 'interior' if index < 4 else 'near_orbit',
             'intended_label': intended[index//2], **{key: row[key] for key in ('one_nn', 'nearest_phase')}}
            for index, row in enumerate(static)]
    design = {'schema_version': 1, 'design_name': 'transverse-readout-v1',
        'evidence_status': ('adapted exploratory; prior line outcomes informed anchors, new outcomes not observed'
                            if direction_seed == SEED else
                            'adapted exploratory; all three prior amplitude cohorts informed this single-direction follow-up, new direction outcomes not observed'),
        'references': reference, 'packet': packet, 'geometry': geom, 'library': lib,
        'library_origin_artifact': binding(output/'library-origins.json'),
        'geometry_check_artifact': binding(output/'geometry-checks.json'),
        'source': source, 'histories': [item[2] for item in histories],
        'generator': 'numpy.random.Generator(numpy.random.PCG64)', 'seed': direction_seed,
        'endpoints': [['ad', 0], ['td', 3]], 'anchor_weights': [7/27, 8/27, 0., 1.],
        'anchor_weight_hex': [float(w).hex() for w in (7/27, 8/27, 0., 1.)],
        'anchor_labels': intended, 'signs': [-1, 1], 'epsilon_relative_rms': relative_epsilon,
        'geometry_gates': GEOMETRY_GATES, 'geometry_checks': checks,
        'library_unique_count': len(library), 'library_origins': origins,
        'excluded_unresolved_origins': excluded,
        'historical_packet_rows': sum(len(item[0]) for item in histories),
        'novelty': {'prior_rows_checked': len(prior_states), 'new_rows': 8,
                    'equal_to_prior': [], 'new_duplicate_pairs': []},
        'rows': rows, 'controls': HOLD,
        'limits': {**LIMITS, **({'transfer_reservation_bytes': 384*1024**2} if direction_seed != SEED else {})},
        'policy': POLICY,
        'one_nn_rule': 'Float64 Euclidean distance over unique labelled states; any exact nearest-exemplar tie erases, including same-label ties.',
        'nearest_phase_rule': 'Minimum float64 Euclidean distance over six phases per label; exact label-distance ties erase; no rejection radius.',
        'execution': 'Full600 scalar updates per row with shadow observer; no actual early stopping.',
        'receiver_access': 'Only state packet and separately supplied reference cycles; local design, intended labels, geometry and library withheld.',
        'scope': 'Four interior probes and four near-orbit controls, one direction/amplitude. No basin-volume, fractality, shrimp-specificity, secrecy, population-reliability or language-channel claim.'}
    write_json(output/'design.json', design)
    receipt = {'status': 'prepared', 'model_forward_calls': 0, 'packet': packet, 'geometry': geom,
        'library': lib, 'design': binding(output/'design.json'), 'reference_sha256': REFERENCE_SHA256,
        'library_unique_count': len(library), 'source_commit': source['source_commit'],
        'utc': subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()}
    return receipt


def prepare(references, audit_paths, output, *, relative_epsilon=0.001, direction_seed=SEED):
    output = Path(output)
    if output.resolve().is_relative_to(ROOT):
        raise ValueError('output belongs outside the source repository')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    progress = {'stage': 'input_arguments'}
    try:
        receipt = prepare_contents(references, audit_paths, output, progress, relative_epsilon=relative_epsilon,
                                   direction_seed=direction_seed)
    except BaseException as error:
        receipt = {'status': 'failed', 'model_forward_calls': 0,
            'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc(),
            'progress': progress, 'retained_artifacts': [binding(path) for path in sorted(output.iterdir()) if path.is_file()],
            'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
            'utc': subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()}
        write_json(output/'preparation.json', receipt)
        raise
    write_json(output/'preparation.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, required=True)
    for name in HISTORY_PINS:
        parser.add_argument('--'+name+'-audit', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--relative-epsilon', type=float, choices=(0.001, 0.003, 0.01), default=0.001)
    parser.add_argument('--direction-seed', type=int, choices=DIRECTION_SEEDS, default=SEED)
    args = parser.parse_args(argv)
    result = prepare(args.references, {name: getattr(args, name+'_audit') for name in HISTORY_PINS}, args.output_dir,
                     relative_epsilon=args.relative_epsilon, direction_seed=args.direction_seed)
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
