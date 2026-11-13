#!/usr/bin/env python3
"""Audit actual stopping against raw prefixes of the prior 600-step replay."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import audit_p6_readback as context
from audit_shadow_readout import load_audit
from audit_readout_horizons import curve
from evaluate_readout_policy import proposal

common = context.common
record = common.independent.file_record
PACKET_HASH = '8c8ef80e02fbfca92b317a15580a5048d64270f7c6d922f974c7ed6bae7e7232'


def check_calls(events, lengths):
    expected = ['native64_prefix']
    for i, n in enumerate(lengths):
        expected += [f'trial-{i:02d}-readback-{kind}.npz' for kind in ('cached', 'full')]
        expected += [f'packet_{i:02d}_readback_{step:03d}' for step in range(1, n + 1)]
    calls = [event for event in events if event['event'] == 'model_call_started']
    if ([event['call'] for event in calls] != list(range(1, len(expected) + 1))
            or [event['label'] for event in calls] != expected):
        raise ValueError('raw model-call schedule differs from consumed updates')
    return len(calls)


def check_stopping(base, index, row, states, refs, definition, bindings):
    """Recompute each checkpoint and require the first decision to end execution."""
    n = len(states) - 1
    if n not in range(12, 601, 12):
        raise ValueError('invalid completed horizon')
    if not np.isfinite(row['elapsed_seconds']) or row['elapsed_seconds'] < 0:
        raise ValueError('invalid row elapsed time')
    shadow = row['shadow_readout']
    if (shadow['status'] != 'completed' or shadow['completed_steps'] != n
            or shadow['definition'] != definition):
        raise ValueError('incomplete observer or changed definition')
    timings = shadow['observer_seconds_per_update']
    if (np.shape(timings) != (n,) or not np.isfinite(timings).all()
            or np.any(np.asarray(timings) < 0)
            or not np.isfinite(shadow['total_observer_seconds'])
            or shadow['total_observer_seconds'] != sum(timings)):
        raise ValueError('invalid observer timings')
    grid = curve(states, np.zeros(n), refs)['grid']
    expected = [{k: r[k] for k in ('steps', 'qualified_label',
                'maximum_closure_residual', 'reference_matches')} for r in grid]
    decision = proposal(grid)['decision']
    if shadow['checkpoints'] != expected or shadow['decision'] != decision:
        raise ValueError('observer differs from independent gates')
    if decision is None:
        if n != 600 or row['stop_reason'] != 'horizon_without_decision' or row['readout_status'] != 'unresolved':
            raise ValueError('unresolved readout did not exhaust its horizon')
    elif (decision['steps'] != n or row['stop_reason'] != 'decision'
          or row['readout_status'] != 'qualified'):
        raise ValueError('execution did not stop on its first decision')
    names = [f'trial-{index:02d}-shadow-block-{end:03d}.json' for end in range(12, n + 1, 12)]
    names.append(f'trial-{index:02d}-shadow-final-{n:03d}.json')
    artifacts = shadow['artifacts']
    if [r['path'] for r in artifacts] != names:
        raise ValueError('observer artifact contract')
    for i, artifact in enumerate(artifacts):
        path = base / artifact['path']
        bindings.append(record(path, artifact))
        saved = json.loads(path.read_text())
        end = min((i + 1) * 12, n)
        if (saved['completed_steps'] != end or saved['artifacts'] != artifacts[:i]
                or saved['definition'] != definition
                or saved['checkpoints'] != expected[:end // 12]
                or saved['observer_seconds_per_update'] != timings[:end]
                or saved['total_observer_seconds'] != sum(timings[:end])
                or saved['decision'] != proposal(grid[:end // 12])['decision']
                or saved['status'] != ('completed' if i == len(artifacts) - 1 else 'running')):
            raise ValueError('observer sidecar progression differs')
    return {'decision': decision, 'checkpoint_count': len(grid),
            'observer_seconds': shadow['total_observer_seconds']}


def audit(base, packet, baseline, references):
    old, prior, bindings = load_audit(baseline)
    bindings += [record(packet, {'sha256': PACKET_HASH}),
                 record(references, {'sha256': context.REF_HASH}),
                 record(base / 'metadata.json'), record(base / 'results.json'),
                 record(base / 'events.jsonl')]
    with np.load(packet, allow_pickle=False) as data:
        if set(data.files) != {'states'}:
            raise ValueError('packet has fields other than states')
        starts = data['states']
    if starts.shape != (8, 896) or starts.dtype != np.float64 or not np.isfinite(starts).all():
        raise ValueError('invalid packet')
    with np.load(references, allow_pickle=False) as data:
        refs = data['states']
    meta = json.loads((base / 'metadata.json').read_text())
    result = json.loads((base / 'results.json').read_text())
    if any(meta.get(k, {}).get('sha256') != v for k, v in context.INPUT_HASHES.items()):
        raise ValueError('model inputs differ')
    if (meta['packet']['sha256'] != PACKET_HASH or meta['controls'] != common.HOLD
            or (meta['max_calls'], meta['max_seconds'], meta['max_rss_gib'],
                meta['maximum_steps'], meta['threads']) != (4817, 2700, 8, 600, 1)):
        raise ValueError('context or limits differ')
    policy = meta['stopping_policy']
    for k, value in {'checkpoint_interval': 12, 'consecutive_qualified': 3,
                     'maximum_steps_per_packet': 600, 'initial_state_observed': False}.items():
        if policy[k] != value:
            raise ValueError('stopping policy differs')
    definition = meta['shadow_readout']
    if (definition['references']['sha256'] != context.REF_HASH
            or definition['stopping_policy'] != policy or result['stopping_policy'] != policy):
        raise ValueError('observer declaration differs')
    for group, keys in {'rotary': ('input', 'inverse_frequency', 'angles', 'cos', 'sin'),
                        'norm': ('input', 'reduction', 'weight', 'output')}.items():
        if any(result['dtype_observations'].get(group, {}).get(k) != 'torch.float64' for k in keys):
            raise ValueError('missing native float64 observations')
    rows = []
    for i in range(8):
        row = {'packet_index': i}
        rows.append(row)
        try:
            raw = result['rows'][i]
            stage = raw['stage']
            n = stage['completed_steps']
            if (raw['packet_index'] != i or stage['name'] != 'readback'
                    or stage['steps'] != 600 or stage['controls'] != common.HOLD
                    or not isinstance(n, int) or isinstance(n, bool) or n not in range(12, 601, 12)):
                raise ValueError('row schedule differs')
            if raw['elapsed_seconds'] > result['elapsed_seconds']:
                raise ValueError('row duration exceeds whole-run duration')
            check, trajectory = common.audit_stage(base, i, stage, bindings, expected_steps=n)
            if check['status'] != 'verified' or not np.array_equal(trajectory['states'][0], starts[i]):
                raise ValueError('incomplete trajectory or changed initial state')
            row.update(check_stopping(base, i, raw, trajectory['states'], refs, definition, bindings))
            oldpath = prior / f'trial-{i:02d}-readback-final-600.npz'
            if not any(Path(b['path']) == oldpath for b in old['bindings']):
                raise ValueError('unbound baseline trajectory')
            full = common.load_trajectory(oldpath, 600, 0, common.HOLD)
            if not np.array_equal(full['states'][0], starts[i]):
                raise ValueError('baseline initial state differs')
            decision = row['decision']
            final_label = old['rows'][i]['qualified_state']
            full_policy = proposal(curve(full['states'], full['step_seconds'], refs)['grid'])
            if full_policy['final_label'] != final_label:
                raise ValueError('baseline final label differs from raw gates')
            row.update(status='verified', stage=check, completed_steps=n,
                       states_exactly_equal=bool(np.array_equal(trajectory['states'], full['states'][:n + 1])),
                       tokens_exactly_equal=bool(np.array_equal(trajectory['token_ids'], full['token_ids'][:n])),
                       maximum_coordinate_gap=float(np.max(np.abs(trajectory['states'] - full['states'][:n + 1]))),
                       baseline_final_label=final_label,
                       decision_matches_final=decision is not None and decision['label'] == final_label,
                       baseline_policy=full_policy,
                       saved_forward_seconds=float(trajectory['step_seconds'].sum()),
                       row_elapsed_seconds=raw['elapsed_seconds'])
        except Exception as error:
            row.update(status='failed', error=str(error))
    updates = sum(r.get('completed_steps', 0) for r in rows)
    expected_calls = 17 + updates
    call_audit = {'status': 'failed'}
    try:
        if not all(r['status'] == 'verified' for r in rows):
            raise ValueError('cannot reconcile complete calls with failed rows')
        events = [json.loads(line) for line in (base / 'events.jsonl').read_text().splitlines()]
        call_audit = {'status': 'verified', 'count': check_calls(events, [r['completed_steps'] for r in rows])}
    except Exception as error:
        call_audit['error'] = str(error)
    complete = (len(result['rows']) == 8 and result['status'] == 'completed'
                and all(r['status'] == 'verified' for r in rows)
                and call_audit['status'] == 'verified'
                and result['model_calls'] == result['expected_actual_calls'] == expected_calls
                and result['resource_limits_passed'] is True
                and 0 <= result['elapsed_seconds'] <= 2700
                and 0 <= result['peak_rss_bytes'] < 8 * 1024**3
                and result['qualified_rows'] == sum(r.get('decision') is not None for r in rows)
                and result['unresolved_rows'] == sum(r.get('decision') is None for r in rows))
    return {'status': 'audit_passed' if complete else 'audit_failed', 'rows': rows, 'bindings': bindings,
            'raw_call_audit': call_audit,
            'producer_source_commit': meta['source_commit'], 'model_calls': result['model_calls'],
            'executed_updates': updates, 'avoided_updates_against_600': 4800 - updates if complete else None,
            'elapsed_seconds': result['elapsed_seconds'], 'peak_rss_bytes': result['peak_rss_bytes'],
            'all_prefixes_exact': all(r.get('states_exactly_equal', False) and r.get('tokens_exactly_equal', False) for r in rows),
            'all_decisions_match_baseline_final': all(r.get('decision_matches_final', False) for r in rows),
            'scope': 'Actual stopping on reused exploratory inputs. Saved calls are measured; separate-run wall times are descriptive. No new-state reliability or text channel.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'packet', 'baseline', 'references', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.base, args.packet, args.baseline, args.references)
    with args.output.open('x') as stream:
        json.dump({'utc': subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
                   'audit': result}, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'model_calls', 'all_prefixes_exact', 'all_decisions_match_baseline_final')}))
    raise SystemExit(result['status'] != 'audit_passed')
