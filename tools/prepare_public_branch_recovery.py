#!/usr/bin/env python3
"""Freeze one public, fixed small-perturbation recovery experiment.

This prepares literal states from an independently audited numerical trajectory;
it performs no model inference and supplies no communication interface.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = '3a0c04011bb4230bdab196f43ec8147682906e71'
AUDIT_SHA = 'ede01c781a2109c99d4813596750db6c78bd89bfc2fc7a8c51f06aacdf998309'
TAIL_SHA = '116372a9143979210f64e6b63636b1b73ea28e06b31b0c657e754df3e6164019'
CONTROLS = dict(alpha=0.07293697809562831, log10_temperature=-0.15361287955001818)
SEED, FACTOR = 20261029, 1e-5
GROUPS = ('unperturbed', 'positive-kick', 'negative-kick')
SIGNS = (0, 1, -1)
SCALE = 'sqrt(mean(sum(source_cycle_states**2, axis=-1)))'


def require(ok, label):
    if not bool(ok):
        raise ValueError(label)


def binding(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'missing or symlink input')
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def array_hash(array):
    header = str(array.dtype).encode() + b'\0' + json.dumps(list(array.shape), separators=(',', ':')).encode() + b'\0'
    return hashlib.sha256(header + array.tobytes(order='C')).hexdigest()


def same_bytes(left, right):
    return left.shape == right.shape and left.dtype == right.dtype and left.tobytes(order='C') == right.tobytes(order='C')


def derived_arrays(tail):
    require(tail.shape == (12, 2, 896) and tail.dtype == np.float64 and np.isfinite(tail).all(), 'finite historical tail')
    cycles = tail[-6:].copy()
    residuals = np.linalg.norm(tail[-6:] - tail[-12:-6], axis=-1).max(axis=0)
    require(np.all(residuals <= 1e-8), 'historical period-six recurrence')
    for divisor in (1, 2, 3):
        gaps = np.linalg.norm(cycles - np.roll(cycles, divisor, axis=0), axis=-1).max(axis=0)
        require(np.all(gaps > 1e-4), 'historical primitive period six')
    separation = min(float(np.linalg.norm(cycles[:, 0] - np.roll(cycles[:, 1], shift, axis=0), axis=1).max()) for shift in range(6))
    require(separation > 1e-4, 'two distinct historical cycles')
    direction = np.random.Generator(np.random.PCG64(SEED)).normal(size=896)
    direction /= np.linalg.norm(direction)
    scale = float(np.sqrt(np.mean(np.sum(cycles ** 2, axis=-1))))
    epsilon = FACTOR * scale
    terminal = tail[-1].copy()
    initial = np.stack([terminal.copy(), terminal + epsilon * direction, terminal - epsilon * direction])
    return dict(initial_states=initial, source_cycle_states=cycles, source_terminal_states=terminal,
                source_tail=tail.copy(), direction=direction), scale, epsilon


def validate_packet(packet_path, states_path, packet_sha256):
    """Remote-safe validation: original source files are checked at preparation/audit."""
    require(binding(packet_path)['sha256'] == packet_sha256, 'frozen packet hash')
    packet = json.loads(Path(packet_path).read_text())
    require(packet['schema_version'] == 1 and packet['profile'] == 'public-branch-recovery', 'packet profile')
    require(packet['source_commit'] == SOURCE_COMMIT and packet['source_stage_id'] == 0, 'historical source identity')
    require(packet['controls'] == CONTROLS and packet['steps_per_group'] == 240, 'fixed recovery controls/dwell')
    require(packet['seed'] == SEED and packet['epsilon_factor'] == FACTOR and packet['scale_definition'] == SCALE, 'fixed perturbation design')
    require(set(packet['source_bindings']) == {'audit', 'result', 'manifest', 'trajectory', 'tail'}, 'historical source inventory')
    require(packet['source_bindings']['audit']['sha256'] == AUDIT_SHA and packet['source_bindings']['tail']['sha256'] == TAIL_SHA, 'pinned source evidence')
    state_binding = binding(states_path)
    require(Path(packet['state_file']['path']).name == packet['state_file']['path'], 'packet state filename')
    require(all(state_binding[k] == packet['state_file'][k] for k in ('bytes', 'sha256')), 'state packet binding')
    with np.load(states_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    require(set(arrays) == {'initial_states', 'source_cycle_states', 'source_terminal_states', 'source_tail', 'direction'}, 'state packet arrays')
    expected, scale, epsilon = derived_arrays(arrays['source_tail'])
    require(all(same_bytes(arrays[k], expected[k]) for k in arrays), 'literal selection or prescribed perturbation differs')
    require(packet['scale'] == scale and packet['epsilon'] == epsilon, 'perturbation scale')
    require(packet['direction_array_sha256'] == array_hash(arrays['direction']), 'direction fingerprint')
    groups = [dict(group_id=i, name=name, sign=sign, initial_array_sha256=array_hash(arrays['initial_states'][i]))
              for i, (name, sign) in enumerate(zip(GROUPS, SIGNS, strict=True))]
    require(packet['groups'] == groups, 'group input fingerprints')
    return arrays, packet


def prepare(source_run, source_audit, output_dir):
    source_run, source_audit, output_dir = map(Path, (source_run, source_audit, output_dir))
    audit_binding = binding(source_audit)
    require(audit_binding['sha256'] == AUDIT_SHA, 'independent historical audit hash')
    audit = json.loads(source_audit.read_text())
    require(audit['status'] == 'passed' and audit['audited_source_commit'] == SOURCE_COMMIT, 'historical audit status/source')
    result = json.loads((source_run / 'result.json').read_text())
    require(result['status'] == 'completed' and result['source']['source_commit'] == SOURCE_COMMIT, 'historical completion/source')
    stage = result['stages'][0]
    require(stage['status'] == 'completed' and stage['controls'] == CONTROLS and stage['completed_steps'] == 480, 'historical stage')
    paths = dict(audit=source_audit, result=source_run / 'result.json', manifest=source_run / 'manifest.json',
                 trajectory=source_run / stage['trajectory']['path'], tail=source_run / stage['blocks'][-1]['path'])
    bindings = {k: binding(v) for k, v in paths.items()}
    require(bindings['tail']['sha256'] == TAIL_SHA, 'fixed tail hash')
    audit_records = {(x['path'], x['bytes'], x['sha256']) for x in audit['bindings']}
    for key, rec in bindings.items():
        if key != 'audit':
            require((rec['path'], rec['bytes'], rec['sha256']) in audit_records, 'historical audit lacks ' + key)
    manifest = json.loads(paths['manifest'].read_text())
    for key in ('result', 'trajectory', 'tail'):
        rec = bindings[key]
        require(dict(path=paths[key].name, bytes=rec['bytes'], sha256=rec['sha256']) in manifest['files'], 'historical manifest lacks ' + key)
    trajectory = json.loads(paths['trajectory'].read_text())
    require(trajectory['blocks'] == stage['blocks'] and trajectory['completed_steps'] == 480, 'historical trajectory contract')
    with np.load(paths['tail'], allow_pickle=False) as z:
        require(int(z['first_state_index']) == 469 and int(z['last_state_index']) == 480, 'historical tail indices')
        require({k: float(z[k]) for k in CONTROLS} == CONTROLS, 'historical tail controls')
        arrays, scale, epsilon = derived_arrays(z['states'])
    output_dir.mkdir(parents=True, exist_ok=False)
    states_path = output_dir / 'cycle-states.npz'
    with states_path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    packet = dict(schema_version=1, profile='public-branch-recovery', source_commit=SOURCE_COMMIT,
                  source_stage_id=0, controls=CONTROLS, steps_per_group=240, seed=SEED, epsilon_factor=FACTOR,
                  scale_definition=SCALE, scale=scale, epsilon=epsilon,
                  direction_array_sha256=array_hash(arrays['direction']), source_bindings=bindings,
                  state_file={**binding(states_path), 'path': states_path.name},
                  groups=[dict(group_id=i, name=name, sign=sign, initial_array_sha256=array_hash(arrays['initial_states'][i]))
                          for i, (name, sign) in enumerate(zip(GROUPS, SIGNS, strict=True))],
                  preparation_source=dict(utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
                                          commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                                          working_tree_dirty=bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()),
                                          files=[binding(__file__)]),
                  scope='Fixed finite-horizon perturbation recovery, not a stability, basin-volume, bifurcation or causal-hysteresis proof.')
    packet_path = output_dir / 'cycle-packet.json'
    with packet_path.open('x') as stream:
        json.dump(packet, stream, indent=2, allow_nan=False)
        stream.write('\n')
    validate_packet(packet_path, states_path, binding(packet_path)['sha256'])
    return dict(packet=binding(packet_path), states=binding(states_path), epsilon=epsilon, model_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--source-audit', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_run, args.source_audit, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
