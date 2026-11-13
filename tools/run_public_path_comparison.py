#!/usr/bin/env python3
"""Fixed direct-versus-staircase approach to N from two qualified P histories."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import run_public_mechanism as mechanism
import run_public_toggle_pilot as toggle

PROFILE = 'public-path-comparison'
ROOT = Path(__file__).resolve().parents[1]
PACKET_SHA = '86da50426fbef108d7f779e1bdb0573aa0926f35c765f835548b4b59df924332'
STATES_SHA = '57c5ba33135673f7935483fe10f2a7bf6c3a28a9696c4a937c1f1fb875569584'
AUDIT_SHA = 'ede01c781a2109c99d4813596750db6c78bd89bfc2fc7a8c51f06aacdf998309'
H = mechanism.HOLD
E0 = mechanism.ENDPOINTS[0]


def controls(fraction):
    return {k: H[k] + fraction * (E0[k]-H[k]) for k in H}


def plans(mode=PROFILE, dwell=240):
    if mode != PROFILE or dwell != 240:
        raise ValueError('fixed path comparison only')
    cases = [('direct-short', [(.78125, 240), (.75, 240)]),
             ('direct-long', [(.78125, 960), (.75, 240)]),
             ('staircase', [(f, 240) for f in (.7578125, .765625, .7734375, .78125, .75)])]
    groups, stages, calls = [], [], []
    for gid, (name, legs) in enumerate(cases):
        group = dict(group_id=gid, name=name, case=dict(arm=name), stage_ids=[],
                     status='not_started', initial_state_sha256=None, observations=[])
        for lid, (fraction, steps) in enumerate(legs):
            stage = copy.deepcopy(mechanism.prior.stage_plan(mechanism.prior.WritePlan(0))[0])
            sid = len(stages); ct = controls(fraction)
            stage.update(stage_id=sid, group_id=gid, group=name, local_stage_id=lid,
                         name=f'leg-{lid}', controls=ct, nominal_controls=ct.copy(),
                         steps=steps, path_fraction=fraction,
                         historical_comparison_scope='new_matched_path_comparison', historical_comparison=None)
            stages.append(stage); group['stage_ids'].append(sid)
            slots = [('prefix', 0, None), ('comparison', 0, True), ('comparison', 0, False)]
            slots += [('update', i, True) for i in range(1, steps+1)]
            for category, step, cached in slots:
                calls.append(dict(call=len(calls)+1, group_id=gid, group=name, stage_id=sid,
                                  stage=stage['name'], category=category, step=step,
                                  cached=cached, status='not_started'))
        groups.append(group)
    return groups, stages, calls


def design(mode=PROFILE, dwell=240):
    groups, stages, calls = plans(mode, dwell)
    return dict(profile=PROFILE, evidence_class='adapted exploratory', evidence_rung=0,
                groups=groups, stages=stages, calls=len(calls), updates=sum(s['steps'] for s in stages),
                initial_histories='literal source terminal states at P, original history order',
                initializations=3, limits=dict(seconds=3600, rss=8*2**30, raw=512*2**20),
                controls=dict(P=controls(.75), N=controls(.78125)),
                primary='Qualified cyclic separation at N; original-history recovery after return to P.',
                scope='Finite route/dwell comparison. Separation is not branch identity, stability, a fold or causal hysteresis.')


def qualify_inputs(args):
    bindings = {key: mechanism.reference.require_hash(getattr(args, key), sha)
                for key, sha in [('cycle_packet', PACKET_SHA), ('cycle_states', STATES_SHA), ('source_audit', AUDIT_SHA)]}
    packet = json.loads(args.cycle_packet.read_text())
    audit = json.loads(args.source_audit.read_text())
    if packet['controls'] != controls(.75) or audit.get('status') != 'passed':
        raise ValueError('qualified P source required')
    with np.load(args.cycle_states, allow_pickle=False) as z:
        cycles = z['source_cycle_states'].copy()
        terminal = z['source_terminal_states'].copy()
    if cycles.dtype != np.float64 or cycles.shape != (6, 2, 896) or not np.isfinite(cycles).all():
        raise ValueError('P cycles shape/precision')
    if terminal.tobytes() != cycles[-1].tobytes():
        raise ValueError('terminal phase does not match cycle')
    refs = np.roll(cycles.transpose(1, 0, 2), 1, axis=1).copy()
    mechanism.engine.shared.check_references(refs)
    return refs, dict(bindings=bindings, original_history_order=[0, 1],
                      phase='terminal first, followed by chronological cycle', source_controls=controls(.75))


def observe_stage(states, refs, stage):
    reads = [mechanism.memory.neutral_readout(states[1:, lane], refs) for lane in (0, 1)]
    cycles = states[-6:]
    gaps = [float(np.linalg.norm(cycles[:, 0]-np.roll(cycles[:, 1], -shift, axis=0), axis=1).max())
            for shift in range(6)]
    both = all(row['cycle_gate']['passed'] for row in reads)
    separation = min(gaps)
    return dict(stage_id=stage['stage_id'], controls=stage['controls'], path_fraction=stage['path_fraction'],
                readouts=reads, off_P=stage['controls'] != controls(.75),
                pair_distance=separation, pair_distances_by_shift=gaps,
                both_primitive_period_six=both,
                pair_status=('distinct' if separation > 1e-4 else
                             'same_within_tolerance' if separation <= 1e-6 else
                             'intermediate') if both else 'unresolved',
                reference_label_scope='Original P history identity only at P; descriptive similarity elsewhere.')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache-dir', 'vectors', 'cycle-packet', 'cycle-states', 'source-audit', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--preflight-only', action='store_true')
    return p


def main():
    args = parser().parse_args(); args.mode = PROFILE; args.dwell = 240
    if args.preflight_only:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        refs, provenance = qualify_inputs(args)
        (args.output_dir/'input-preflight.json').write_text(json.dumps(dict(
            status='passed', design=design(), provenance=provenance, model_calls=0), indent=2)+'\n')
        return 0
    result = toggle.run_fixed(args, profile=PROFILE, qualifier=qualify_inputs,
                              plan_factory=plans, design_factory=design, observe_stage=observe_stage,
                              extra_sources=(Path(__file__), ROOT/'docs/public-path-comparison.md',
                                             ROOT/'tests/test_public_path_comparison.py',
                                             ROOT/'tools/audit_public_toggle_pilot.py',
                                             ROOT/'tests/test_public_toggle_pilot_audit.py',
                                             ROOT/'tools/audit_public_repair_half_pulse.py'))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
