"""Shared checks for the full-adder and lane-layout audits (Task 8; docs/public-full-adder-truth-table.md and
docs/public-lane-layout-check.md). Zero model calls.

Everything the audits score against is literal here (controls, step counts, call order), not taken from the
producer's plan functions. Every check records problems instead of raising, so an audit can enumerate all scheduled
jobs and still write a receipt when some are missing or malformed.

Revised after Codex's Task 8 recheck (docs/audits/2026-11-13-task8-recheck/review.md): each saved array is checked on
its own before anything is joined (R5); the initial state's binding is checked like the blocks' (R7); the call
ledger is compared with the literal ordered sequence, not counted (R4); reconstruction returns nothing when any of
its own checks fails, so no comparison is built from evidence that failed (R6); every file read is bound.
"""
from __future__ import annotations

import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np

# Literal controls (values recorded in docs/receipts and the instrument constants; see the plans).
H = {'alpha': 0.07293697809562831, 'log10_temperature': -0.15395151820007266}
E0 = {'alpha': 0.07293697809562831, 'log10_temperature': -0.1535}
E1 = {'alpha': 0.07299785589927693, 'log10_temperature': -0.15395151820007266}
T = {'alpha': 0.07312150413133038, 'log10_temperature': -0.1531613613499455}
MOVES = {'W': (H, 12), 'T': (T, 12), 'E0': (E0, 120), 'E1': (E1, 120)}
HOLD_STEPS = 240
PERIOD = 6
SETTLE = 1e-6
STATE_SHAPE = (2, 896)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def binding(p: Path) -> dict:
    p = Path(p)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def load_json(p: Path, problems: list, what: str):
    try:
        return json.loads(Path(p).read_text())
    except Exception as exc:                      # missing or malformed: an operational failure, recorded
        problems.append(f'{what}: cannot read {Path(p).name} ({type(exc).__name__})')
        return None


def literal_bits(controls: dict) -> dict:
    return {k: {'value': v, 'hex': float(v).hex(), 'bits': struct.unpack('<Q', struct.pack('<d', v))[0]}
            for k, v in controls.items()}


def same(a, b) -> bool:
    """Equality that does not let True stand for 1 or 1.0 for 1 (JSON round trips keep these types)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def check_manifest(d: Path, problems: list, what: str) -> None:
    m = load_json(d/'manifest.json', problems, what)
    if m is None:
        return
    try:
        files = {x['path']: x['sha256'] for x in m.get('files', [])}
    except Exception as exc:
        problems.append(f'{what}: manifest malformed ({type(exc).__name__})')
        return
    present = {p.name for p in d.iterdir() if p.is_file()} - {'manifest.json'}
    if set(files) != present:
        problems.append(f'{what}: manifest file set differs')
    for name, h in files.items():
        if (d/name).is_file() and sha(d/name) != h:
            problems.append(f'{what}: hash mismatch {name}')


def check_controls(stage: dict, want: dict, problems: list, what: str) -> None:
    if not same(stage.get('controls'), want):
        problems.append(f'{what}: controls {stage.get("controls")} differ from the plan')
    if not same(stage.get('control_bits'), literal_bits(want)):
        problems.append(f'{what}: control bit encodings differ from the plan')


def _read_bound(d: Path, rec, problems: list, what: str, kind: str, bound: list | None):
    """Bytes of the file a stage record names, checked against the record's size and SHA-256 (the bytes that are
    then parsed, so the check and the use cannot diverge). The file actually read is bound whatever the outcome."""
    if not isinstance(rec, dict) or not isinstance(rec.get('path'), str) or Path(rec['path']).name != rec['path']:
        problems.append(f'{what}: {kind} binding missing or not a plain file name')
        return None
    f = d/rec['path']
    try:
        raw = f.read_bytes()
    except OSError as exc:
        problems.append(f'{what}: {kind} {f.name} unreadable ({type(exc).__name__})')
        return None
    if bound is not None:
        bound.append({'path': str(f), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    if not same(rec.get('bytes'), len(raw)) or rec.get('sha256') != hashlib.sha256(raw).hexdigest():
        problems.append(f'{what}: {kind} {f.name} size or SHA-256 differs from the stage record')
        return None
    return raw


def reconstruct(d: Path, stage: dict, want: dict, steps: int, problems: list, what: str, bound: list | None = None,
                lanes: int = 2):
    """A stage's saved path (steps + 1, lanes, 896) read against its record. The initial state must match its binding
    (size, SHA-256, array digest, shape and dtype) and be a finite float64 (lanes, 896) array; each block must match
    its binding, be a finite float64 (n, lanes, 896) array on its own (before any join can promote a dtype), carry
    the planned controls, run contiguously from update 1 and be named for its last update. Returns None if any of
    these fails, after recording why. lanes is 2 for the pair instrument and 1 for the single-lane arm."""
    n0 = len(problems)
    shape = (lanes, 896)
    sid = stage.get('stage_id') if isinstance(stage, dict) else None
    if not isinstance(sid, int) or isinstance(sid, bool):
        problems.append(f'{what}: stage record has no integer stage_id')
        return None
    try:
        init_rec = stage.get('initial')
        raw = _read_bound(d, init_rec, problems, what, 'initial state', bound)
        if raw is None:
            return None
        init = np.load(io.BytesIO(raw), allow_pickle=False)
        if init_rec['path'] != f'stage-{sid:02d}-initial.npy' or init.dtype != np.float64 \
                or init.shape != shape or not np.isfinite(init).all() or init_rec.get('dtype') != 'float64' \
                or not same(init_rec.get('shape'), list(shape)) \
                or init_rec.get('array_sha256') != hashlib.sha256(np.ascontiguousarray(init).tobytes()).hexdigest():
            problems.append(f'{what}: initial state is not the recorded finite float64 {shape} array')
        blocks = stage.get('blocks')
        if not isinstance(blocks, list) or not blocks:
            problems.append(f'{what}: no blocks recorded')
            return None
        states, cursor = [init[None]], 0
        for b in blocks:
            raw = _read_bound(d, b, problems, what, 'block', bound)
            if raw is None:
                continue
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                y = z['states']
                first, last = int(z['first_state_index']), int(z['last_state_index'])
                ctrl = (float(z['alpha']), float(z['log10_temperature']))
            if y.dtype != np.float64 or y.ndim != 3 or y.shape[1:] != shape or not np.isfinite(y).all():
                problems.append(f'{what}: block {b["path"]} is not a finite float64 (n, {lanes}, 896) array')
                continue
            if first != cursor + 1 or last != cursor + len(y):
                problems.append(f'{what}: block {b["path"]} holds updates {first}..{last}, expected from {cursor + 1}')
            if b['path'] != f'stage-{sid:02d}-block-{last:03d}.npz':
                problems.append(f'{what}: block {b["path"]} is not named for stage {sid} update {last}')
            if ctrl != (want['alpha'], want['log10_temperature']):
                problems.append(f'{what}: block {b["path"]} controls differ from the plan')
            states.append(y)
            cursor += len(y)
        if len(problems) > n0:
            return None
        x = np.concatenate(states)
    except Exception as exc:
        problems.append(f'{what}: stage files unreadable ({type(exc).__name__})')
        return None
    if x.shape != (steps + 1,) + shape:
        problems.append(f'{what}: reconstructed shape {x.shape}, expected ({steps + 1}, {lanes}, 896)')
        return None
    return x


def check_stage_record(stage: dict, steps: int, problems: list, what: str) -> None:
    for key in ('steps', 'completed_steps', 'attempted_steps', 'returned_steps'):
        if not same(stage.get(key), steps):
            problems.append(f'{what}: {key} = {stage.get(key)}, expected {steps}')
    if stage.get('status') != 'completed':
        problems.append(f'{what}: status {stage.get("status")}')
    if stage.get('failed_outputs') != []:
        problems.append(f'{what}: failed outputs recorded')
    if (stage.get('cache_comparison') or {}).get('passed') is not True:
        problems.append(f'{what}: cache comparison did not pass')
    if stage.get('final_scalar_cache_unchanged') is not True:
        problems.append(f'{what}: final scalar cache not confirmed unchanged')


def check_unstarted_stage(stage: dict, problems: list, what: str) -> None:
    """A planned stage of another shard: present in the scaffold, never started, with no evidence attached."""
    if stage.get('status') != 'not_started' or any(not same(stage.get(k), 0) for k in
                                                   ('attempted_steps', 'returned_steps', 'completed_steps')) \
            or stage.get('blocks') != [] or stage.get('initial') is not None or stage.get('failed_outputs') != [] \
            or stage.get('cache_comparison') is not None:
        problems.append(f'{what}: a stage of another shard was started or carries evidence')


def expected_calls(plan: list, profile: str) -> list:
    """The literal ordered ledger: per stage, a prefix, a cached and a full comparison, then updates 1..N, numbered
    consecutively from 1 across the stages in execution order."""
    out = []
    for p in plan:
        for category, step, cached in [('prefix', 0, None), ('comparison', 0, True), ('comparison', 0, False)] + [
                ('update', i, True) for i in range(1, p['steps'] + 1)]:
            out.append(dict(profile=profile, call=len(out) + 1, group_id=p['group_id'], group=p['group'],
                            stage_id=p['stage_id'], stage=p['stage'], category=category, step=step, cached=cached))
    return out


LAYOUTS = ('pair', 'single')


def check_call_ledger(calls, plan: list, profile: str, problems: list, what: str, *, layout: str) -> int:
    """Every call must be the literal one at its position, have returned, and carry its own runtime gates as saved by
    the evaluator: float64-only precision observations for the prefix; for every comparison and update, passed
    precision, valid outputs, an unchanged input and an unchanged stored prefix cache; a carried state for each
    update. layout says which evaluator wrote the ledger and is never inferred: 'pair' calls must record a passed
    batch-expansion check and no scalar-path flag; 'single' calls must record the scalar path and, explicitly, no
    batch expansion (expanded_cache_valid present and None). A missing gate fails like a false one (Codex's Task
    8/9 recheck, N1)."""
    if layout not in LAYOUTS:
        raise ValueError(f'layout must be one of {LAYOUTS}')
    if not isinstance(calls, list):
        problems.append(f'{what}: no call ledger')
        return 0
    want = expected_calls(plan, profile)
    if len(calls) != len(want):
        problems.append(f'{what}: {len(calls)} calls, expected {len(want)}')
    bad = []
    for i, (c, w) in enumerate(zip(calls, want)):
        ok = isinstance(c, dict) and all(same(c.get(k), v) for k, v in w.items()) and c.get('status') == 'returned'
        if ok and w['category'] == 'prefix':
            obs = json.dumps(c.get('dtype_observations', {}))
            ok = 'float64' in obs and not any(t in obs for t in ('float32', 'float16', 'bfloat16'))
        elif ok:
            ok = all(c.get(g) is True for g in ('precision_passed', 'outputs_valid', 'input_unchanged',
                                                 'cache_unchanged'))
            if layout == 'pair':
                ok = ok and c.get('expanded_cache_valid') is True and 'scalar_path' not in c
            else:
                ok = ok and c.get('scalar_path') is True and 'expanded_cache_valid' in c \
                    and c['expanded_cache_valid'] is None
            if w['category'] == 'update':
                ok = ok and c.get('state_carried') is True
        if not ok:
            bad.append(i + 1)
    if bad:
        problems.append(f'{what}: call ledger differs from the literal sequence at call(s) {bad[:6]}'
                        + (f' and {len(bad) - 6} more' if len(bad) > 6 else ''))
    return len(calls)


def check_environment(d: Path, canonical_inputs: dict | None, frozen_commit: str | None, problems: list,
                      what: str) -> dict:
    out = {}
    br = load_json(d/'backend-readiness.json', problems, what)
    if br is not None:
        if br.get('mode') != 'native64' or br.get('torch_threads') != 1 or br.get('torch_interop_threads') != 1 \
                or any(v != '1' for v in (br.get('thread_environment') or {}).values()) or not br.get('thread_environment'):
            problems.append(f'{what}: backend readiness is not native64 with one thread')
        out['backend_readiness'] = binding(d/'backend-readiness.json')
    mi = load_json(d/'model-inputs.json', problems, what)
    if mi is not None:
        out['model_inputs'] = binding(d/'model-inputs.json')
        if canonical_inputs is not None:
            for k in ('config', 'vectors', 'weights'):
                if (mi.get(k) or {}).get('sha256') != (canonical_inputs.get(k) or {}).get('sha256'):
                    problems.append(f'{what}: model input {k} differs from the canonical audited run')
    src = load_json(d/'source.json', problems, what)
    if src is not None:
        out['source'] = binding(d/'source.json')
        out['source_commit'] = src.get('source_commit')
        out['working_tree_clean'] = src.get('working_tree_status', None) == ''
        if frozen_commit is not None:
            if src.get('source_commit') != frozen_commit:
                problems.append(f'{what}: source commit {src.get("source_commit")} is not the frozen commit')
            if src.get('working_tree_status') != '':
                problems.append(f'{what}: source checkout was not clean')
    return out


def compare_paths(new: np.ndarray, old: np.ndarray, n: int, label: str, note: str | None = None) -> dict:
    """Largest coordinate gap over exactly the first n states of both paths; unavailable if either is shorter."""
    out = dict(earlier=label, planned_states=n)
    if note:
        out['note'] = note
    if new.shape[0] < n or old.shape[0] < n or new.shape[1:] != old.shape[1:]:
        return dict(out, status='unavailable', reason=f'paths of {new.shape} and {old.shape} cannot give {n} states')
    return dict(out, status='compared', compared_updates=n - 1, max_gap=float(np.abs(new[:n] - old[:n]).max()))


def nearest(state: np.ndarray, refs: np.ndarray) -> tuple[int, int, float]:
    d = [[float(np.linalg.norm(state - refs[c, j])) for j in range(refs.shape[1])] for c in range(refs.shape[0])]
    c, j = np.unravel_index(int(np.argmin(d)), (refs.shape[0], refs.shape[1]))
    return int(c), int(j), d[c][j]


def settle(trajectory: np.ndarray, refs: np.ndarray) -> dict:
    """First update after which the lane stays within SETTLE of its final cycle, phase-aligned by counting back from
    the measured end phase; censored if it is not within SETTLE at the end."""
    c, j, dist = nearest(trajectory[-1], refs)
    n = trajectory.shape[0] - 1
    d = [float(np.linalg.norm(trajectory[m] - refs[c, (j - (n - m)) % refs.shape[1]])) for m in range(n + 1)]
    if d[-1] >= SETTLE:
        return {'final_cycle': c, 'censored': True, 'settled_from_update': None}
    above = [m for m, x in enumerate(d) if x >= SETTLE]
    return {'final_cycle': c, 'censored': False, 'settled_from_update': (above[-1] + 1) if above else 0}
