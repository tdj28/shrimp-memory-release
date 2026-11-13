#!/usr/bin/env python3
"""A bounded, public search for a second period-six solution at fixed M.

The search does not test attraction or certify a branch. Its inputs are literal,
hash-bound numerical evidence; it contains no communication interface.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np
import validate_qwen_map as instrument
import prepare_public_branch_recovery as packet_tool
import bounded_periodic_corrector as solver

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'public-periodic-corrector'
CONTROLS = dict(alpha=0.07293697809562831, log10_temperature=-0.1535846596625136)
DIM, PERIOD = 896, 6
MAX_CALLS, MAX_SECONDS, RAW_CAP = 403, 1800, 768 * 2**20
FINAL_RESERVE, NEXT_CALL_RESERVE = 16 * 2**20, 2 * 2**20
PACKET_SHA = '86da50426fbef108d7f779e1bdb0573aa0926f35c765f835548b4b59df924332'
STATES_SHA = '57c5ba33135673f7935483fe10f2a7bf6c3a28a9696c4a937c1f1fb875569584'
SOURCE_AUDIT_SHA = 'ede01c781a2109c99d4813596750db6c78bd89bfc2fc7a8c51f06aacdf998309'
DERIVATIVE_AUDIT_SHA = '6a1b1042c1e79a464d6e2befdc42e68fddfabdc2cd1d0f17fb7f356cd9c63495'
MIDPOINT_AUDIT_SHA = 'daca5e3136ecd6391d06cdf6e1a3926b27d897771183ec82013a06962b859e59'
MIDPOINT_RESULT_SHA = '84cbc18a89277302c1ae6571b513907f319de5ed84d128d85c4f9d3e3cb263ab'
MIDPOINT_TAIL_SHA = '1c2883281384efafda795df397abf6876b282b9be8a1f55dfc91fe447f80c954'
MIDPOINT_COMMIT = '23a34b09a3ab097bc3c0791ad21076f1ab7619be'
GROUP_NAMES = ('instrument', 'newton-0', 'newton-1', 'newton-2', 'newton-3', 'fresh-root-check')
GATES = dict(root_l2=1e-10, phase_closure_l2=1e-10, primitive_gap=1e-4,
    distinct_cycle_gap=1e-4, cache_linf=1e-10, primal_linf=1e-10,
    fd_relative=1e-6, linearity_relative=1e-10, fresh_linear_relative=0.1,
    action_agreement_relative=1e-10, arnoldi_breakdown_absolute=1e-13,
    direction_norm_floor=1e-14)
FD_STEPS = (1e-5, 5e-6, 2.5e-6)


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


class InstrumentFailure(RuntimeError):
    pass


def instrument_gate(ok, message):
    if not bool(ok):
        raise InstrumentFailure(message)


def binding(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'missing or symlink input: ' + str(path))
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=instrument.sha256(path))


def checked(path, digest):
    row = binding(path)
    require(row['sha256'] == digest, 'input hash differs: ' + str(path))
    return row


def finite(array, shape, name):
    require(isinstance(array, np.ndarray) and array.shape == shape and array.dtype == np.float64
            and np.isfinite(array).all(), 'finite float64 ' + name + ' required')


def array_hash(array):
    return packet_tool.array_hash(np.asarray(array))


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def terminal_report(value):
    bad = []
    def clean(item, path):
        if isinstance(item, dict):
            return {k: clean(v, path + '.' + k) for k, v in item.items()}
        if isinstance(item, list):
            return [clean(v, f'{path}[{i}]') for i, v in enumerate(item)]
        if isinstance(item, float) and not math.isfinite(item):
            bad.append(dict(path=path, representation=repr(item)))
            return None
        return item
    result = clean(plain(value), 'result')
    if bad:
        result.update(status='instrument_failed', nonfinite_report_fields=bad)
    return result


def design():
    return dict(profile=PROFILE, evidence_class='adapted exploratory', evidence_rung=0,
        controls=CONTROLS, period=PERIOD, seed='literal original P history-1 terminal',
        comparator='saved merged M history-0 terminal and cycle',
        counts=dict(prefix=1, instrument_ordinary=72, instrument_jvp=18,
            correction_ordinary_max=72, correction_jvp_max=216, final_ordinary_max=24),
        maximum_model_calls=MAX_CALLS, internal_seconds=MAX_SECONDS, external_seconds=1860,
        termination_grace_seconds=30, outer_seconds=2040, return_worker_seconds=3600,
        max_rss_bytes=8 * 2**30, max_virtual_bytes=16 * 2**30, threads=1,
        raw_cap_bytes=RAW_CAP, finalization_reserve_bytes=FINAL_RESERVE,
        package_cap_bytes=896 * 2**20, archive_cap_bytes=1024 * 2**20,
        transfer_reservation_bytes=4 * 2**30, upload_cap_bytes=8 * 2**20,
        newton_iterations=4, gmres_steps=8, damping=[1.0, 0.5, 0.25], armijo=1e-4,
        trust_ball='fixed center xP1; radius 2*L2(xP1-xP0); no clipping or recentering',
        direction_seeds=[0, 1], direction_normalization='PCG64 standard normal, NumPy L2 normalization',
        finite_difference_steps=list(FD_STEPS), gates=GATES, groups=list(GROUP_NAMES),
        scope='A distinct qualified numerical period-six solution would establish a finite-precision '
              'periodic candidate at M. No attraction, branch identity, branch disappearance, '
              'causal hysteresis or hidden-channel claim. An unresolved search is not nonexistence.')


def qualify_inputs(args):
    bindings = dict(packet=checked(args.cycle_packet, PACKET_SHA),
        states=checked(args.cycle_states, STATES_SHA),
        source_audit=checked(args.source_audit, SOURCE_AUDIT_SHA),
        derivative_audit=checked(args.derivative_audit, DERIVATIVE_AUDIT_SHA),
        midpoint_audit=checked(args.midpoint_audit, MIDPOINT_AUDIT_SHA),
        midpoint_result=checked(args.midpoint_result, MIDPOINT_RESULT_SHA),
        midpoint_tail=checked(args.midpoint_tail, MIDPOINT_TAIL_SHA))
    arrays, packet = packet_tool.validate_packet(args.cycle_packet, args.cycle_states, PACKET_SHA)
    source_audit = json.loads(args.source_audit.read_text())
    require(source_audit['status'] == 'passed'
            and source_audit['audited_source_commit'] == packet_tool.SOURCE_COMMIT,
            'original P source audit differs')
    prior_index = {}
    for row in source_audit['bindings']:
        key = os.path.normpath(row['path']); pair = (row['bytes'],row['sha256'])
        require(key not in prior_index or prior_index[key] == pair, 'conflicting original source binding')
        prior_index[key] = pair
    for name,row in packet['source_bindings'].items():
        if name != 'audit':
            require(prior_index.get(os.path.normpath(row['path'])) == (row['bytes'],row['sha256']),
                    'packet record absent from original source audit')
    derivative = json.loads(args.derivative_audit.read_text())
    require(derivative['status'] == 'audit_passed' and derivative['all_numerical_rows_passed']
            and derivative['source_commit'] == '0800126ac8af336f7da5b17fbdc3da9a7b443b29'
            and [r['history'] for r in derivative['rows']] == [0,1]
            and all(r['derivative_passed'] and r['implementation_passed']
                    and r['composition_steps'] == 6 for r in derivative['rows']),
            'original two-history derivative qualification differs')
    audit = json.loads(args.midpoint_audit.read_text())
    result = json.loads(args.midpoint_result.read_text())
    require(audit['status'] == 'audit_passed' and audit['integrity_status'] == 'passed'
            and audit['audited_source_commit'] == MIDPOINT_COMMIT
            and audit['return_status'] == 'verified' and audit['transfer_reservation_closed']
            and audit['all_planned_rows_retained'] and audit['all_closed_stage_members_immutable']
            and audit['outcome'] == 'm_unresolved', 'accepted midpoint audit required')
    index = {}
    for row in audit['bindings']:
        key = os.path.normpath(row['path']); pair = (row['bytes'], row['sha256'])
        require(key not in index or index[key] == pair, 'conflicting audit binding')
        index[key] = pair
    for name, row in [('result.json', bindings['midpoint_result']),
                      ('stage-00-block-480.npz', bindings['midpoint_tail'])]:
        require(index.get(name) == (row['bytes'], row['sha256']), 'midpoint file absent from audit')
    require(result['profile'] == 'public-endpoint-midpoint' and result['status'] == 'completed'
            and result['source']['source_commit'] == MIDPOINT_COMMIT
            and result['outcome'] == 'm_unresolved' and result['model_calls'] == 483
            and result['state_resets'] == 0 and result['stages'][0]['controls'] == CONTROLS
            and result['stages'][1]['status'] == 'not_run_m_unresolved', 'midpoint result identity differs')
    require(result['literal_inputs']['sha256'] == STATES_SHA
            and result['groups'][0]['initial_state_sha256'] == array_hash(arrays['source_terminal_states']),
            'midpoint and corrector must use identical P states')
    with np.load(args.midpoint_tail, allow_pickle=False) as z:
        tail = z['states'].copy()
        require(int(z['first_state_index']) == 469 and int(z['last_state_index']) == 480,
                'midpoint terminal indices differ')
        require(all(float(z[k]) == v for k, v in CONTROLS.items()), 'midpoint controls differ')
    finite(tail, (12, 2, DIM), 'midpoint tail')
    require(array_hash(tail[-1]) == result['terminal_state_sha256'], 'midpoint terminal differs')
    x0 = arrays['source_terminal_states'][1].copy()
    radius = 2.0 * float(np.linalg.norm(x0 - arrays['source_terminal_states'][0]))
    require(math.isfinite(radius) and radius > 0, 'positive fixed trust radius required')
    payload = dict(seed=x0, p_cycles=arrays['source_cycle_states'].copy(),
        p_terminal=arrays['source_terminal_states'].copy(), m_cycles=tail[-6:].copy(),
        comparator=tail[-1, 0].copy(), trust_radius=radius)
    return payload, dict(bindings=bindings, packet_controls=packet['controls'],
        seed_array_sha256=array_hash(x0), comparator_array_sha256=array_hash(payload['comparator']),
        fixed_trust_radius=radius, midpoint_source_commit=MIDPOINT_COMMIT,
        provenance_scope='Pinned midpoint audit binds the returned P inputs and literal M terminal. '
                         'Original trajectories are not rerun for preparation.')


def qualify_model(args):
    snapshot = args.cache_dir/'models--Qwen--Qwen2.5-0.5B-Instruct'/'snapshots'/instrument.REVISION
    row = dict(weights=instrument.require_hash(snapshot/'model.safetensors', instrument.WEIGHTS_SHA256),
        config=instrument.require_hash(snapshot/'config.json',
            '18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45'),
        tokenizer=instrument.require_hash(snapshot/'tokenizer.json',
            'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'),
        tokenizer_config=instrument.require_hash(snapshot/'tokenizer_config.json',
            '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583'),
        vectors=instrument.require_hash(args.vectors, instrument.VECTOR_SHA256),
        model=instrument.MODEL, revision=instrument.REVISION, prefix_ids=instrument.PREFIX_IDS)
    return snapshot, row


def source_metadata(args):
    files = [Path(__file__), Path(solver.__file__), Path(instrument.__file__), Path(packet_tool.__file__),
        ROOT/'tools/composed_feedback_action.py', ROOT/'pyproject.toml', ROOT/'uv.lock',
        ROOT/'tests/test_bounded_periodic_corrector.py', ROOT/'tests/test_public_periodic_corrector.py',
        ROOT/'docs/hysteresis-and-logic-protocol.md']
    rows = [{**binding(p), 'path': str(p.relative_to(ROOT))} for p in files]
    return dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        files=rows, working_tree_status=subprocess.check_output(['git','status','--porcelain','--',
            *[row['path'] for row in rows]],cwd=ROOT,text=True).splitlines(),
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})


class Recorder(instrument.Recorder):
    def __init__(self, path):
        super().__init__(path)
        self.assigned = set()
        self.groups = [dict(profile=PROFILE, group_id=i, name=name, status='not_started',
            closed=False, model_calls=0, artifacts=[]) for i, name in enumerate(GROUP_NAMES)]
        self.active_group = 0
        self.event_index = 0

    def size(self):
        return sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file())

    def save(self, name, **arrays):
        require(Path(name).name == name, 'flat raw filename required')
        arrays = {k:np.asarray(v) for k,v in arrays.items()}
        require(all(a.dtype.kind in 'biufcUS' for a in arrays.values()), 'nonobject evidence arrays required')
        if self.size() + sum(a.nbytes for a in arrays.values()) + 65536 > RAW_CAP - FINAL_RESERVE:
            raise RuntimeError('working raw byte limit')
        with (self.directory/name).open('xb') as out:
            np.savez(out, **arrays)

    def close_group(self, gid, status, summary):
        group = self.groups[gid]
        if group['closed']:
            return
        self.json(f'group-{gid:02d}-summary.json', terminal_report(summary))
        artifacts = []
        for p in sorted(self.directory.iterdir()):
            if not p.is_file() or p.name in self.assigned or p.name in ('events.jsonl','result.json','manifest.json'):
                continue
            if p.name.startswith('group-') and p.name.endswith('.json') and '-summary' not in p.name:
                continue
            require(not p.name.startswith('.partial'), 'unfinished marker at group close')
            row = binding(p); row['path'] = p.name; artifacts.append(row)
        group.update(status=status, closed=True, artifacts=artifacts)
        if status.startswith('not_run_'):
            group['not_run_reason'] = status.removeprefix('not_run_')
        self.assigned.update(row['path'] for row in artifacts)
        name = f'group-{gid:02d}.json'
        temporary = self.directory/('.partial-'+name)
        with temporary.open('x') as out:
            json.dump(terminal_report(group), out, indent=2, allow_nan=False); out.write('\n')
            out.flush(); os.fsync(out.fileno())
        os.link(temporary, self.directory/name); temporary.unlink()


class Budget(instrument.Budget):
    def check_returned(self):
        if time.monotonic()-self.recorder.started >= self.max_seconds:
            raise TimeoutError('wall-time limit after returned call')
        if instrument.rss_bytes() >= self.max_rss:
            raise MemoryError('peak RSS limit after returned call')
        if self.calls > self.max_calls or self.recorder.size() > RAW_CAP-FINAL_RESERVE:
            raise RuntimeError('call or raw limit after returned call')

    def claim(self, label):
        if self.recorder.size() + NEXT_CALL_RESERVE > RAW_CAP - FINAL_RESERVE:
            raise RuntimeError('map raw byte limit')
        super().claim(label)
        self.recorder.groups[self.recorder.active_group]['model_calls'] += 1


def apply_resource_check(rec,budget,summary):
    limits=dict(elapsed_seconds=time.monotonic()-rec.started,
        peak_rss_bytes=instrument.rss_bytes(),model_calls=budget.calls,
        current_raw_bytes=rec.size(),terminal_rewrite_reserve_bytes=65536)
    passed=(limits['elapsed_seconds'] < MAX_SECONDS and limits['peak_rss_bytes'] < 8*2**30
        and limits['model_calls'] <= MAX_CALLS and limits['current_raw_bytes']+65536 <= RAW_CAP)
    summary.update(resource_limits_passed=bool(passed),terminal_resources=limits)
    if not passed:
        summary.setdefault('status_before_resource_gate',summary['status'])
        summary.update(status='incomplete',outcome='not_qualified',resource_gate_failure=True)


def cycle_distance(left, right):
    values = [float(np.linalg.norm(left - np.roll(right, shift, axis=0), axis=1).max())
              for shift in range(PERIOD)]
    require(all(math.isfinite(v) for v in values), 'nonfinite cycle distance')
    return dict(distance=min(values), best_shift=int(np.argmin(values)), by_shift=values)


def cycle_checks(trace):
    finite(trace, (13, DIM), 'two-cycle trace')
    closure = float(np.linalg.norm(trace[6:]-trace[:-6],axis=1).max())
    cycle = trace[1:7]
    gaps = {str(k):float(np.linalg.norm(cycle-np.roll(cycle,k,axis=0),axis=1).max()) for k in (1,2,3)}
    require(all(math.isfinite(v) for v in [closure,*gaps.values()]), 'nonfinite cycle metrics')
    return dict(maximum_phase_closure=closure, proper_divisor_gaps=gaps,
        passed=closure<=GATES['phase_closure_l2'] and all(v>GATES['primitive_gap'] for v in gaps.values()))


def run_protocol(backend, payload, rec, summary):
    seed, comparator = payload['seed'].copy(), payload['comparator'].copy()
    def chain(label, state, tangent=None, steps=6):
        state = state.copy(); tangent = None if tangent is None else tangent.copy()
        differentiated = tangent is not None
        trace = [state.copy()]
        for i in range(steps):
            state, tangent = backend.evaluate(f'{label}-{i}',state,tangent)
            finite(state, (DIM,), 'map state')
            if differentiated:
                finite(tangent, (DIM,), 'map tangent')
            trace.append(state.copy())
        return np.stack(trace), tangent

    control, _ = chain('control', comparator, steps=12)
    control_check = cycle_checks(control)
    # Saved M cycles are chronological t475..480; comparator is t480, so the
    # next six states correspond to t475..480 without a shifted phase choice.
    expected = payload['m_cycles'][:,0]
    control_check['saved_cycle_linf'] = float(np.max(np.abs(control[1:7]-expected)))
    control_cache = []
    for i in range(6):
        y, _ = backend.evaluate(f'control-full-{i}',control[i].copy(),cached=False)
        control_cache.append(float(np.max(np.abs(y-control[i+1]))))
    rec.save('control-trace.npz', states=control)
    summary['control'] = dict(**control_check, cache_gaps=control_cache)
    instrument_gate(control_check['passed'] and control_check['saved_cycle_linf']<=1e-10
                    and max(control_cache)<=1e-10, 'merged M control or its cache comparison failed')

    baseline, _ = chain('seed-baseline',seed)
    cache_gaps = []
    for i in range(6):
        y, _ = backend.evaluate(f'seed-full-{i}',baseline[i].copy(),cached=False)
        cache_gaps.append(float(np.max(np.abs(y-baseline[i+1]))))
    rec.save('seed-baseline.npz', states=baseline)
    summary['seed_cache_gaps'] = cache_gaps
    instrument_gate(max(cache_gaps)<=1e-10, 'seed cached/full comparison failed')
    directions = []
    for s in (0,1):
        v = np.random.Generator(np.random.PCG64(s)).normal(size=DIM)
        directions.append(v/np.linalg.norm(v))
    rec.save('directions.npz', seed0=directions[0], seed1=directions[1])
    jvps = []
    for i,v in enumerate(directions):
        trace, action = chain(f'direction-{i}',seed,v)
        instrument_gate(float(np.max(np.abs(trace-baseline)))<=1e-10
                        and np.linalg.norm(action)>1e-14, 'seed JVP primal or norm gate failed')
        jvps.append(action)
    rows = []
    for i,h in enumerate(FD_STEPS):
        plus,_ = chain(f'fd-{i}-plus',backend.offset(seed,directions[0],h))
        minus,_ = chain(f'fd-{i}-minus',backend.offset(seed,directions[0],-h))
        delta = (plus[-1]-minus[-1])/(2*h)
        rows.append(dict(h=h, relative_error=float(np.linalg.norm(delta-jvps[0])/np.linalg.norm(jvps[0]))))
    fd = instrument.convergence_report(rows,1e-6)
    combined = 0.6*directions[0]-0.8*directions[1]
    trace, action = chain('linearity',seed,combined)
    predicted = 0.6*jvps[0]-0.8*jvps[1]
    linearity = float(np.linalg.norm(action-predicted)/max(np.linalg.norm(action),np.linalg.norm(predicted),1e-14))
    summary['derivative'] = dict(fd_rows=rows,fd=fd,linearity_relative_error=linearity,
        combined_primal_linf=float(np.max(np.abs(trace-baseline))))
    instrument_gate(fd['passed'] and linearity<=1e-10 and summary['derivative']['combined_primal_linf']<=1e-10,
                    'fresh M seed derivative qualification failed')

    residual_cache = {}
    first_residual = True
    counters = dict(residual=0,jvp=0)
    def residual(x):
        nonlocal first_residual
        counters['residual'] += 1
        if first_residual:
            require(x.tobytes()==seed.tobytes(), 'solver initial state differs')
            trace,_ = chain('seed-repeat',x)
            gap = float(np.max(np.abs(trace-baseline)))
            summary['seed_repeat_linf'] = gap
            instrument_gate(gap==0.0, 'seed repeat is not exact')
            first_residual = False
            rec.close_group(0,'completed',dict(status='completed',control=summary['control'],
                derivative=summary['derivative'],seed_cache_gaps=cache_gaps,seed_repeat_linf=gap))
        else:
            trace,_ = chain(f'root-r-{counters["residual"]}',x)
        residual_cache[array_hash(x)] = trace.copy()
        return trace[-1]-x

    def jacobian_action(x,v):
        counters['jvp'] += 1
        require(array_hash(x) in residual_cache, 'missing same-iterate nonlinear baseline')
        trace,j = chain(f'root-j-{counters["jvp"]}',x,v)
        instrument_gate(float(np.max(np.abs(trace-residual_cache[array_hash(x)])))<=1e-10,
                        'corrector JVP primal differs at current iterate')
        return j-v

    def event(row):
        event_type = row.get('type')
        if event_type == 'iteration_begin':
            rec.active_group = 1+row['iteration']
        rec.event_index += 1
        rec.json(f'solver-event-{rec.event_index:04d}.json',terminal_report(row))
        if event_type == 'iteration_end':
            rec.close_group(1+row['iteration'],row.get('status','completed'),row)

    config = solver.CorrectorConfig(root_tolerance=1e-10, max_newton_iterations=4, max_gmres_actions=8,
        max_residual_calls=13, max_jacobian_calls=36,
        linear_relative_tolerance=0.1, action_consistency_tolerance=1e-10,
        damping=(1.0,0.5,0.25), sufficient_decrease=1e-4, trust_radius=payload['trust_radius'])
    corrected = solver.correct(residual,jacobian_action,seed,config=config,event=event)
    rec.json('corrector-evidence.json',solver.serializable(corrected))
    summary['corrector'] = corrected
    summary['callback_counts'] = counters
    if corrected['status'] == 'failed':
        raise InstrumentFailure('corrector callback or instrument failure: '+corrected['reason'])
    if corrected['status'] != 'converged':
        summary.update(status='completed',outcome='root_search_unresolved')
        return summary
    candidate = np.asarray(corrected['x'],dtype=np.float64)
    rec.active_group = 5
    final,_ = chain('final-cycles',candidate,steps=12)
    cache = []
    for i in range(6):
        y,_ = backend.evaluate(f'final-full-{i}',final[i].copy(),cached=False)
        cache.append(float(np.max(np.abs(y-final[i+1]))))
    repeat,_ = chain('final-repeat',candidate)
    repeat_gap = float(np.max(np.abs(repeat-final[:7])))
    instrument_gate(max(cache)<=1e-10 and repeat_gap==0.0, 'final implementation checks failed')
    checks = cycle_checks(final)
    fresh_root = float(np.linalg.norm(final[6]-candidate))
    cycle = final[1:7]
    comparison = cycle_distance(cycle,payload['m_cycles'][:,0])
    p_distances = [cycle_distance(cycle,payload['p_cycles'][:,h]) for h in (0,1)]
    rec.save('final-root-check.npz',candidate=candidate,states=final,repeat=repeat)
    summary['final'] = dict(**checks,root_residual_l2=fresh_root,cache_gaps=cache,repeat_linf=repeat_gap,
        merged_cycle_distance=comparison,original_P_distances=p_distances)
    if not checks['passed'] or fresh_root>1e-10:
        outcome = 'root_qualification_unresolved'
    elif comparison['distance']>1e-4:
        outcome = 'distinct_periodic_candidate'
    else:
        outcome = 'merged_cycle_candidate'
    summary.update(status='completed',outcome=outcome)
    rec.close_group(5,'completed',summary['final'])
    return summary


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cycle-packet','cycle-states','source-audit','derivative-audit',
                 'midpoint-audit','midpoint-result','midpoint-tail',
                 'cache-dir','vectors','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    return p


def fresh_precision(observations):
    return all(observations.get(group,{}).get(key)=='torch.float64' for group,keys in {
        'norm':('input','reduction','weight','output'),
        'rotary':('input','inverse_frequency','angles','cos','sin')}.items() for key in keys)


def make_backend(args,snapshot,rec,budget):
    """Load exactly one native64 scalar map; only the production CLI uses this."""
    for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key]='1'
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    ids=tokenizer.apply_chat_template(instrument.MESSAGES,add_generation_prompt=True)
    ids=list(ids['input_ids'] if hasattr(ids,'keys') else ids)
    require(ids==instrument.PREFIX_IDS,'fixed prefix differs')
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
        dtype=torch.float64,attn_implementation='sdpa').cpu().eval().requires_grad_(False)
    rec.model_loaded=True
    cfg=model.config
    require((cfg.model_type,cfg.hidden_size,cfg.num_hidden_layers,cfg.vocab_size)==('qwen2',896,24,151936),
            'unexpected model architecture')
    require(all(t.device.type=='cpu' and t.dtype==torch.float64 for t in model.parameters()),
            'model parameter dtype/device differs')
    with np.load(args.vectors,allow_pickle=False) as z:vector=z['philosophy'].copy()
    finite(vector,(DIM,),'steering vector')
    observations={};patches=instrument.patch_native_float64(model,observations)
    loop=instrument.FeedbackMap(model,torch.tensor(vector,dtype=torch.float64),
        budget=budget,recorder=rec,**CONTROLS)
    loop.mode='native64';observations.clear();loop.prepare()
    rec.json('prefix-precision.json',dict(call=budget.calls,observations=copy.deepcopy(observations),
        passed=fresh_precision(observations)))
    instrument_gate(fresh_precision(observations),'prefix lacks fresh native64 observations')
    rec.json('backend-readiness.json',dict(status='ready',mode=loop.mode,patches=patches,
        observations=copy.deepcopy(observations),threads=dict(intraop=torch.get_num_threads(),interop=torch.get_num_interop_threads()),
        thread_environment={k:os.environ[k] for k in
            ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        versions={k:importlib.metadata.version(k) for k in ('numpy','torch','transformers')},
        scope='Fresh finite-difference qualification is still required at the literal M seed.'))
    class Backend:
        @staticmethod
        def evaluate(label,state,tangent=None,cached=True):
            observations.clear()
            y,j=loop.evaluate(label,torch.tensor(state),
                tangent=None if tangent is None else torch.tensor(tangent),cached=cached)
            rec.json(f'precision-{budget.calls:04d}.json',dict(call=budget.calls,label=label,
                passed=fresh_precision(observations),observations=copy.deepcopy(observations)))
            instrument_gate(fresh_precision(observations),'map lacks fresh native64 observations')
            budget.check_returned()
            return y.detach().numpy().copy(),None if j is None else j.detach().numpy().copy()
        @staticmethod
        def offset(x,v,h):return (torch.tensor(x)+h*torch.tensor(v)).numpy()
    return Backend(), observations


def run(args, *, backend_factory=make_backend, input_qualifier=qualify_inputs,
        model_qualifier=qualify_model, source_factory=source_metadata, protocol=run_protocol):
    """Dependency injection is for model-free contract tests, never a CLI option."""
    rec=Recorder(args.output_dir);budget=Budget(rec,MAX_CALLS,MAX_SECONDS,8)
    summary=dict(profile=PROFILE,status='execution_failed',outcome='not_evaluated',model_calls=0,
        design=design(),groups=rec.groups,model_loaded=False)
    handlers={}
    try:
        summary['source']=source_factory(args)
        rec.json('source.json',summary['source']);rec.json('design.json',design())
        rec.json('planned-groups.json',rec.groups)
        payload,inputs=input_qualifier(args)
        snapshot,model_inputs=model_qualifier(args)
        summary['inputs']=inputs;summary['model_inputs']=model_inputs
        rec.json('qualified-inputs.json',inputs);rec.json('model-inputs.json',model_inputs)
        rec.save('literal-inputs.npz',**payload)
        if args.preflight_only:
            summary.update(status='preflight_passed',outcome='not_run_preflight')
        else:
            def alarm(signum,frame):
                raise TimeoutError('declared numerical deadline or termination')
            for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):
                handlers[sig]=signal.getsignal(sig);signal.signal(sig,alarm)
            signal.setitimer(signal.ITIMER_REAL,MAX_SECONDS)
            backend,observations=backend_factory(args,snapshot,rec,budget)
            summary['model_loaded']=True
            protocol(backend,payload,rec,summary)
            summary['dtype_observations']=observations
    except BaseException as error:
        status='instrument_failed' if isinstance(error,InstrumentFailure) else (
            'incomplete' if isinstance(error,(TimeoutError,MemoryError,KeyboardInterrupt)) else 'execution_failed')
        summary.update(status=status,outcome='not_qualified',error=f'{type(error).__name__}: {error}',
                       traceback=traceback.format_exc())
        rec.event('attempt_stopped',status=status,error=summary['error'])
    finally:
        if handlers:
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,handler in handlers.items():signal.signal(sig,handler)
        summary['model_loaded']=summary['model_loaded'] or getattr(rec,'model_loaded',False)
        apply_resource_check(rec,budget,summary)
        for gid,group in enumerate(rec.groups):
            if not group['closed']:
                status = summary['status'] if group['model_calls'] else 'not_run_'+summary['outcome']
                row=dict(status=status,outcome=summary['outcome'],error=summary.get('error'))
                if status.startswith('not_run_'):
                    row['not_run_reason']=summary['outcome']
                rec.close_group(gid,status,row)
        summary.update(model_calls=budget.calls,unused_reserved_calls=MAX_CALLS-budget.calls,
            elapsed_seconds=time.monotonic()-rec.started,peak_rss_bytes=instrument.rss_bytes(),utc=instrument.utc_now())
        summary=terminal_report(summary)
        rec.json('result.json',summary)
        rec.event('attempt_finished',status=summary['status'],outcome=summary['outcome'],model_calls=budget.calls)
        rec.stream.close()
        files=[{**binding(p),'path':p.name} for p in sorted(rec.directory.iterdir()) if p.is_file()]
        rec.json('manifest.json',dict(files=files))
        # Include the terminal serialization and manifest itself. A late breach
        # is retained as incomplete, even if the numerical protocol had passed.
        apply_resource_check(rec,budget,summary)
        temporary=rec.directory/'.partial-result.json'
        temporary.write_text(json.dumps(terminal_report(summary),indent=2,allow_nan=False)+'\n')
        os.replace(temporary,rec.directory/'result.json')
        files=[{**binding(p),'path':p.name} for p in sorted(rec.directory.iterdir())
               if p.is_file() and p.name!='manifest.json']
        temporary=rec.directory/'.partial-manifest.json'
        temporary.write_text(json.dumps(dict(files=files),indent=2,allow_nan=False)+'\n')
        os.replace(temporary,rec.directory/'manifest.json')
    return 0 if summary['status'] in ('completed','preflight_passed') else 1


def main(argv=None):
    return run(parser().parse_args(argv))


if __name__=='__main__':
    raise SystemExit(main())
