"""Bounded first-order continuation in caller-supplied, fixed scaled coordinates.

For orbit continuation the caller supplies y=(x/r,(logT-logTH)/Delta),
G(y)=R/r and DG(y)v. No scale, control, phase or normalization is changed here.
Every accepted corrector satisfies G, the arclength constraint AND their full
stacked residual. SciPy GMRES success alone is insufficient: a separate fresh
action must satisfy the full linear equation. A parameter turn is only a
numerical bracket; it is not a certified fold, simple null mode or stability
result. ``solve_system`` also supports a caller-defined extended fold system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Callable, Mapping

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


@dataclass(frozen=True)
class NewtonConfig:
    max_newton_iterations: int = 6
    gmres_restart: int = 10
    max_linear_actions: int = 12  # Includes the separate fresh validation.
    max_residual_calls: int = 128
    max_action_calls: int = 1024
    residual_tolerance: float = 1e-10
    linear_relative_tolerance: float = 1e-3
    linear_absolute_tolerance: float = 1e-14
    damping: tuple[float, ...] = (1.0, 0.5, 0.25)
    sufficient_decrease: float = 1e-4

    def __post_init__(self):
        for name in ('max_newton_iterations', 'max_residual_calls', 'max_action_calls'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        for name in ('gmres_restart', 'max_linear_actions'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('residual_tolerance', 'linear_relative_tolerance', 'linear_absolute_tolerance'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.linear_relative_tolerance >= 1:
            raise ValueError('linear_relative_tolerance must be below one')
        if not math.isfinite(self.sufficient_decrease) or not 0 <= self.sufficient_decrease < 1:
            raise ValueError('sufficient_decrease must lie in [0,1)')
        damping = tuple(self.damping)
        if (not damping or any(not math.isfinite(x) or not 0 < x <= 1 for x in damping)
                or any(a <= b for a, b in zip(damping, damping[1:]))):
            raise ValueError('damping must be positive, strictly decreasing and at most one')
        object.__setattr__(self, 'damping', damping)


@dataclass(frozen=True)
class PACConfig(NewtonConfig):
    equation_tolerance: float = 1e-10
    arc_tolerance: float = 1e-10
    tangent_tolerance: float = 1e-8
    tangent_norm_tolerance: float = 1e-12
    max_correction_to_step_ratio: float = 1.0
    minimum_tangent_dot: float = 0.5
    step_sizes: tuple[float, ...] = (0.02, 0.01, 0.005)
    max_accepted_steps: int = 16
    max_attempted_steps: int = 24
    max_rejected_steps: int = 8

    def __post_init__(self):
        super().__post_init__()
        for name in ('equation_tolerance', 'arc_tolerance', 'tangent_tolerance', 'tangent_norm_tolerance', 'max_correction_to_step_ratio'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.minimum_tangent_dot) or not 0 < self.minimum_tangent_dot <= 1:
            raise ValueError('minimum_tangent_dot must lie in (0,1]')
        for name in ('max_accepted_steps', 'max_attempted_steps', 'max_rejected_steps'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        steps = tuple(self.step_sizes)
        if (not steps or any(not math.isfinite(x) or x <= 0 for x in steps)
                or any(a <= b for a, b in zip(steps, steps[1:]))):
            raise ValueError('step_sizes must be positive and strictly decreasing')
        object.__setattr__(self, 'step_sizes', steps)


def serializable(value):
    """Detached JSON-safe trace; arrays retain dtype, shape, bytes hash and values."""
    if isinstance(value, np.ndarray):
        return dict(dtype=str(value.dtype), shape=list(value.shape),
                    sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                    values=serializable(value.tolist()))
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return dict(python_type=type(value).__name__, repr=repr(value))


class SolverFailure(RuntimeError):
    """Fatal callback/arithmetic/persistence failure; partial ``result`` is retained.

    The original exception is ``__cause__``. Timeout, resource and I/O causes
    must not be reclassified as a scientific nonconvergence by the caller.
    """

    def __init__(self, result):
        self.result = result
        super().__init__(result['reason'])


class _Stop(Exception):
    pass


def _vector(value, size, name):
    a = np.asarray(value)
    if a.dtype != np.dtype('float64') or a.shape != (size,) or not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must be a finite real float64 vector of shape {(size,)}')
    return a.copy()


def _norm(a):
    scale = float(np.max(np.abs(a), initial=0.0))
    out = 0.0 if scale == 0 else float(scale * np.sqrt(np.sum((a / scale) ** 2)))
    if not math.isfinite(out):
        raise FloatingPointError('nonfinite Euclidean norm')
    return out


def _input(value, name):
    a = np.asarray(value)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f'{name} must be a nonempty vector')
    return _vector(a, a.size, name)


class _Ledger:
    def __init__(self, config, event, result):
        self.config, self.event, self.result = config, event, result
        result.update(counts=dict(residual_calls=0, action_calls=0), calls=[])

    def emit(self, kind, **fields):
        if self.event is not None:
            self.event(serializable(dict(event=kind, **fields)))

    def invoke(self, kind, function, y, v, size, purpose):
        key = kind + '_calls'
        record = dict(id=len(self.result['calls']), kind=kind, purpose=purpose,
                      y=y.copy(), called=False, status='pending')
        if v is not None:
            record['direction'] = v.copy()
        self.result['calls'].append(record)
        if self.result['counts'][key] >= getattr(self.config, 'max_' + key):
            record['status'] = 'not_called_budget'
            self.emit('call_end', call=record)
            raise _Stop(key + '_budget')
        try:
            self.emit('call_begin', call=record)
        except Exception as exc:
            record.update(status='not_called_event_failure', error_type=type(exc).__name__, error=str(exc))
            raise
        self.result['counts'][key] += 1
        record['called'] = True
        try:
            raw = function(y.copy()) if v is None else function(y.copy(), v.copy())
            record['output'] = raw.copy() if isinstance(raw, np.ndarray) else raw
            value = _vector(raw, size, kind + ' output')
            _norm(value)
        except Exception as exc:
            record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            try:
                self.emit('call_end', call=record)
            except Exception as observer_error:
                record['event_error'] = dict(type=type(observer_error).__name__, message=str(observer_error))
            raise
        record.update(status='completed', output=value.copy())
        self.emit('call_end', call=record)
        return value, record['id']


class _Problem:
    def __init__(self, residual, action, size, ledger, predictor=None, tangent=None):
        self.residual, self.derivative, self.size, self.ledger = residual, action, size, ledger
        self.predictor, self.tangent = predictor, tangent
        self.output_size = size if tangent is None else size - 1

    def evaluate(self, y, purpose):
        value, call_id = self.ledger.invoke('residual', self.residual, y, None, self.output_size, purpose)
        if self.tangent is not None:
            value = np.append(value, np.dot(self.tangent, y - self.predictor))
        return _vector(value, self.size, 'full residual'), call_id

    def action(self, y, v, purpose):
        value, call_id = self.ledger.invoke('action', self.derivative, y, v, self.output_size, purpose)
        if self.tangent is not None:
            value = np.append(value, np.dot(self.tangent, v))
        return _vector(value, self.size, 'full action'), call_id


def _blocks(blocks, size, tolerance):
    if blocks is None:
        return {'equations': (0, size, tolerance)}
    result, covered = {}, []
    for name, item in blocks.items():
        if not isinstance(name, str) or not name or len(item) != 3:
            raise ValueError('residual_blocks must map names to (start, stop, tolerance)')
        start, stop, tol = item
        if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= size
                or not math.isfinite(tol) or tol <= 0):
            raise ValueError('invalid residual block')
        covered.extend(range(start, stop))
        result[name] = (start, stop, float(tol))
    if sorted(covered) != list(range(size)):
        raise ValueError('residual blocks must cover every equation exactly once')
    return result


def _gates(residual, blocks, config):
    norm = _norm(residual)
    checks = {k: dict(norm=_norm(residual[a:b]), tolerance=tol)
              for k, (a, b, tol) in blocks.items()}
    for check in checks.values():
        check['passed'] = check['norm'] <= check['tolerance']
    return dict(full_norm=norm, full_tolerance=config.residual_tolerance,
                full_passed=norm <= config.residual_tolerance, blocks=checks,
                passed=norm <= config.residual_tolerance and all(c['passed'] for c in checks.values()))


def _linear(problem, y, rhs, config, record, purpose):
    """One GMRES restart cycle, explicit action ceiling including fresh check."""
    record.update(rhs=rhs.copy(), rhs_norm=_norm(rhs), actions=[], krylov_residuals=[], gmres_info=None)
    action_limit = config.max_linear_actions

    def apply(v, fresh=False):
        # Reserve one action for the independently evaluated full-space check.
        if len(record['actions']) >= action_limit - (0 if fresh else 1):
            raise _Stop('linear_action_budget')
        v = _vector(v, y.size, 'linear direction')
        value, call_id = problem.action(y, v, purpose + ('_fresh' if fresh else '_gmres'))
        record['actions'].append(dict(call_id=call_id, direction=v.copy(), output=value.copy(), fresh=fresh))
        return value

    def progress(value):
        value = float(value)
        if not math.isfinite(value):
            raise FloatingPointError('nonfinite GMRES callback residual')
        record['krylov_residuals'].append(value)
        if len(record['krylov_residuals']) > config.gmres_restart:
            raise _Stop('gmres_iteration_budget')

    operator = LinearOperator((y.size, y.size), matvec=apply, dtype=np.float64)
    delta, info = gmres(operator, rhs, x0=np.zeros_like(y),
                        rtol=config.linear_relative_tolerance, atol=config.linear_absolute_tolerance,
                        restart=config.gmres_restart, maxiter=1, callback=progress, callback_type='pr_norm')
    record['gmres_info'] = int(info)
    record['solution'] = np.asarray(delta).copy()
    delta = _vector(delta, y.size, 'linear solution')
    actual = apply(delta, fresh=True)
    error = actual - rhs
    limit = max(config.linear_absolute_tolerance, config.linear_relative_tolerance * record['rhs_norm'])
    record.update(fresh_action=actual.copy(), residual=error.copy(), residual_norm=_norm(error),
                  residual_limit=limit, relative_residual=_norm(error) / max(record['rhs_norm'], 1e-300))
    record['passed'] = info == 0 and record['residual_norm'] <= limit
    record['reason'] = ('gmres_not_converged' if info != 0 else
                        'fresh_linear_residual_failed' if not record['passed'] else 'qualified')
    problem.ledger.emit('linear_end', linear=record)
    return delta if record['passed'] else None


def _newton(problem, initial, blocks, config, record):
    record.update(status='running', reason=None, initial_y=initial.copy(), iterations=[])
    y = initial.copy()
    residual, call_id = problem.evaluate(y, 'initial_residual')
    record.update(y=y.copy(), residual=residual.copy(), residual_call_id=call_id)
    for index in range(config.max_newton_iterations + 1):
        gates = _gates(residual, blocks, config)
        record['gates'] = gates
        if gates['passed']:
            record.update(status='converged', reason='full_residual_qualified')
            return record
        if index == config.max_newton_iterations:
            record.update(status='unresolved', reason='newton_iteration_budget')
            return record
        iteration = dict(index=index, y=y.copy(), residual=residual.copy(), gates=gates,
                         linear={}, trials=[], status='running')
        record['iterations'].append(iteration)
        problem.ledger.emit('newton_begin', iteration=iteration)
        delta = _linear(problem, y, -residual, config, iteration['linear'], 'newton')
        if delta is None:
            iteration.update(status='unresolved', reason=iteration['linear']['reason'])
            record.update(status='unresolved', reason=iteration['reason'])
            problem.ledger.emit('newton_end', iteration=iteration)
            return record
        for damping in config.damping:
            trial_y = _vector(y + damping * delta, y.size, 'trial state')
            trial = dict(damping=damping, y=trial_y.copy(), status='running')
            iteration['trials'].append(trial)
            trial_r, trial_id = problem.evaluate(trial_y, 'line_search')
            trial_gates = _gates(trial_r, blocks, config)
            trial.update(residual=trial_r.copy(), residual_call_id=trial_id, gates=trial_gates)
            trial['accepted'] = (trial_gates['passed'] or
                                 trial_gates['full_norm'] <= (1 - config.sufficient_decrease * damping) * gates['full_norm'])
            trial['status'] = 'accepted' if trial['accepted'] else 'rejected'
            problem.ledger.emit('trial_end', trial=trial)
            if trial['accepted']:
                y, residual = trial_y, trial_r
                record.update(y=y.copy(), residual=residual.copy(), residual_call_id=trial_id)
                break
        else:
            iteration.update(status='unresolved', reason='line_search_exhausted')
            record.update(status='unresolved', reason='line_search_exhausted')
            problem.ledger.emit('newton_end', iteration=iteration)
            return record
        iteration['status'] = 'accepted'
        problem.ledger.emit('newton_end', iteration=iteration)
    raise AssertionError('unreachable')


def _execute(config, event, result, body):
    ledger = _Ledger(config, event, result)
    result.update(status='running', reason=None, config=asdict(config))
    try:
        ledger.emit('run_begin', result=result)
        body(ledger)
    except _Stop as exc:
        result.update(status='unresolved', reason=str(exc))
        _close_open(result, 'unresolved', str(exc))
    except Exception as exc:
        result.update(status='failed', reason='callback_or_arithmetic_failure',
                      error_type=type(exc).__name__, error=str(exc))
        _close_open(result, 'failed', str(exc))
        try:
            ledger.emit('run_end', result=result)
        except Exception as observer_error:
            result['terminal_event_error'] = dict(type=type(observer_error).__name__, message=str(observer_error))
        raise SolverFailure(result) from exc
    try:
        ledger.emit('run_end', result=result)
    except Exception as exc:
        result.update(status='failed', reason='event_callback_failure', error_type=type(exc).__name__, error=str(exc))
        raise SolverFailure(result) from exc
    return result


def _close_open(value, status, reason):
    """Close in-progress trial/iteration/attempt records on abnormal termination."""
    if isinstance(value, dict):
        if value.get('status') == 'running':
            value.update(status=status, reason=reason)
        for child in value.values():
            _close_open(child, status, reason)
    elif isinstance(value, list):
        for child in value:
            _close_open(child, status, reason)


def solve_system(residual: Callable, jacobian_action: Callable, z0: np.ndarray, *,
                 config: NewtonConfig, residual_blocks: Mapping | None = None,
                 event: Callable | None = None) -> dict:
    """Bounded square Newton solve; no model or finite differences are supplied.

    Residual/action return vectors of z0's length. Optional residual_blocks maps
    names to (start, exclusive_stop, absolute_L2_tolerance), covering every
    equation once. ALL blocks and the full residual must pass. The full linear
    gate uses max(atol, rtol*||rhs||), AND requires SciPy info==0. Nonconvergence
    returns ``unresolved``; a callback failure raises SolverFailure from the
    original exception, with complete partial result. Event snapshots are strict
    JSON-safe copies. Attempted callbacks count even when they fail.
    """
    if not isinstance(config, NewtonConfig):
        raise TypeError('config must be NewtonConfig')
    z0 = _input(z0, 'z0')
    blocks = _blocks(residual_blocks, z0.size, config.residual_tolerance)
    result = dict(kind='square_system', residual_blocks=blocks)
    def run(ledger):
        _newton(_Problem(residual, jacobian_action, z0.size, ledger), z0, blocks, config, result)
    return _execute(config, event, result, run)


def correct(residual: Callable, jacobian_action: Callable, predictor: np.ndarray,
            tangent: np.ndarray, *, config: PACConfig, initial_y: np.ndarray | None = None,
            event: Callable | None = None) -> dict:
    """Solve [G(y), tangent·(y−predictor)]=0 without dropping the arc equation.

    G and DGv have one fewer coordinate than y. Tangent must already be unit;
    its orientation and the predictor remain fixed throughout this correction.
    initial_y defaults to predictor; an explicit starting guess is allowed but
    receives the identical full/arc gates, even when G is already zero.
    """
    if not isinstance(config, PACConfig):
        raise TypeError('config must be PACConfig')
    predictor = _input(predictor, 'predictor')
    if predictor.size < 2:
        raise ValueError('continuation needs at least one equation and one parameter')
    tangent = _vector(tangent, predictor.size, 'tangent')
    if abs(_norm(tangent) - 1) > config.tangent_norm_tolerance:
        raise ValueError('tangent must be unit without implicit renormalization')
    initial = predictor.copy() if initial_y is None else _vector(initial_y, predictor.size, 'initial_y')
    blocks = {'equation': (0, predictor.size - 1, config.equation_tolerance),
              'arc': (predictor.size - 1, predictor.size, config.arc_tolerance)}
    result = dict(kind='pseudoarclength_corrector', predictor=predictor.copy(), tangent=tangent.copy(), residual_blocks=blocks)
    def run(ledger):
        _newton(_Problem(residual, jacobian_action, predictor.size, ledger, predictor, tangent), initial, blocks, config, result)
    return _execute(config, event, result, run)


def _tangent(action, y, reference, config, ledger, record):
    record.update(reference=reference.copy(), linear={}, status='running')
    p = _Problem(None, action, y.size, ledger, y, reference)
    rhs = np.zeros_like(y)
    rhs[-1] = 1.0
    direction = _linear(p, y, rhs, config, record['linear'], 'tangent')
    if direction is None:
        record.update(status='unresolved', reason=record['linear']['reason'])
        return None
    norm = _norm(direction)
    if norm == 0:
        record.update(status='unresolved', reason='zero_tangent')
        return None
    unit = direction / norm
    if np.dot(unit, reference) < 0:
        unit = -unit
    # Normalization changes the vector; check the actual normalized action too.
    fresh, call_id = ledger.invoke('action', action, y, unit, y.size - 1, 'normalized_tangent_fresh')
    record.update(unnormalized=direction.copy(), tangent=unit.copy(), norm_error=abs(_norm(unit) - 1),
                  normalized_action=fresh.copy(), normalized_action_call_id=call_id,
                  null_residual_norm=_norm(fresh), orientation=float(np.dot(unit, reference)))
    passed = (record['norm_error'] <= config.tangent_norm_tolerance
              and record['null_residual_norm'] <= config.tangent_tolerance and record['orientation'] > 0)
    record.update(status='qualified' if passed else 'unresolved',
                  reason='tangent_qualified' if passed else 'normalized_tangent_failed')
    ledger.emit('tangent_end', tangent=record)
    return unit if passed else None


def continue_branch(residual: Callable, jacobian_action: Callable, y0: np.ndarray,
                    tangent0: np.ndarray, *, config: PACConfig, stop_on_turn: bool = False,
                    event: Callable | None = None) -> dict:
    """Continue a qualified seed with bounded attempts and a fixed step ladder.

    tangent0 sets orientation/bordering; it is normalized once and a tangent is
    freshly solved at the seed. Each accepted point receives its own bordered
    tangent and fresh normalized-null check. No failed tangent is reused.
    A finite rejection tries the next declared step size from the SAME base;
    success resets that ladder. Global attempted, accepted, rejected and actual
    residual/action limits all apply; callback errors abort without retry.
    One tangent solve costs at most max_linear_actions + one normalized action.
    Every attempted point and all solver calls are retained. Turn brackets use
    signs of the last tangent coordinate; no interpolated fold is returned.
    stop_on_turn terminates at the first qualified sign crossing or exact zero.
    Point admission also bounds the correction from the predictor by
    max_correction_to_step_ratio*step and requires the new tangent's dot with
    its predecessor to meet minimum_tangent_dot. These are local numerical
    safeguards, not a certificate that the branch is unique or connected.
    """
    if not isinstance(config, PACConfig):
        raise TypeError('config must be PACConfig')
    if type(stop_on_turn) is not bool:
        raise ValueError('stop_on_turn must be boolean')
    y0 = _input(y0, 'y0')
    if y0.size < 2:
        raise ValueError('continuation needs at least two coordinates')
    direction = _vector(tangent0, y0.size, 'tangent0')
    norm = _norm(direction)
    if norm == 0:
        raise ValueError('tangent0 must be nonzero')
    direction /= norm
    result = dict(kind='pseudoarclength_continuation', points=[], attempts=[], turn_brackets=[],
                  initial_tangent={}, accepted_steps=0, rejected_steps=0,
                  stop_on_turn=stop_on_turn,
                  scope='A numerical turn bracket is not an exact or certified fold, simple null mode, or stability result.')
    blocks = {'equation': (0, y0.size - 1, config.equation_tolerance),
              'arc': (y0.size - 1, y0.size, config.arc_tolerance)}
    def run(ledger):
        seed_r, seed_call = ledger.invoke('residual', residual, y0, None, y0.size - 1, 'seed_residual')
        result.update(seed_residual=seed_r.copy(), seed_residual_call_id=seed_call)
        if _norm(seed_r) > min(config.equation_tolerance, config.residual_tolerance):
            result.update(status='unresolved', reason='seed_equation_unqualified')
            return
        tangent = _tangent(jacobian_action, y0, direction, config, ledger, result['initial_tangent'])
        if tangent is None:
            result.update(status='unresolved', reason='seed_tangent_unqualified')
            return
        y = y0.copy()
        result['points'].append(dict(index=0, y=y.copy(), residual=seed_r.copy(), tangent=tangent.copy()))
        ladder_index = 0
        for index in range(config.max_attempted_steps):
            step = config.step_sizes[ladder_index]
            predictor = _vector(y + step * tangent, y.size, 'predictor')
            attempt = dict(index=index, base_point=len(result['points']) - 1, step_size=step,
                           predictor=predictor.copy(), tangent=tangent.copy(), corrector={}, tangent_update={}, status='running')
            result['attempts'].append(attempt)
            ledger.emit('attempt_begin', attempt=attempt)
            p = _Problem(residual, jacobian_action, y.size, ledger, predictor, tangent)
            corrected = _newton(p, predictor, blocks, config, attempt['corrector'])
            new_tangent = None
            rejection_reason = corrected['reason']
            if corrected['status'] == 'converged':
                distance = _norm(corrected['y'] - predictor)
                limit = config.max_correction_to_step_ratio * step
                attempt.update(correction_distance=distance, correction_limit=limit,
                               correction_distance_passed=distance <= limit,
                               step_distance=_norm(corrected['y'] - y))
                rejection_reason = 'correction_distance_exceeded'
                if distance <= limit:
                    new_tangent = _tangent(jacobian_action, corrected['y'], tangent, config, ledger, attempt['tangent_update'])
                    rejection_reason = 'tangent_unqualified'
                    if new_tangent is not None:
                        alignment = float(np.dot(tangent, new_tangent))
                        attempt.update(tangent_dot=alignment, minimum_tangent_dot=config.minimum_tangent_dot,
                                       tangent_alignment_passed=alignment >= config.minimum_tangent_dot)
                        if alignment < config.minimum_tangent_dot:
                            new_tangent = None
                            rejection_reason = 'tangent_alignment_failed'
            if new_tangent is None:
                result['rejected_steps'] += 1
                attempt.update(status='rejected', reason=rejection_reason)
                ledger.emit('attempt_end', attempt=attempt)
                ladder_index += 1
                if result['rejected_steps'] >= config.max_rejected_steps or ladder_index == len(config.step_sizes):
                    result.update(status='unresolved', reason='rejection_budget' if result['rejected_steps'] >= config.max_rejected_steps else 'step_size_ladder_exhausted')
                    return
                continue
            y = corrected['y'].copy()
            point_index = len(result['points'])
            result['points'].append(dict(index=point_index, y=y.copy(), residual=corrected['residual'][:-1].copy(),
                                         tangent=new_tangent.copy(), attempt=index))
            if tangent[-1] * new_tangent[-1] < 0 or (new_tangent[-1] == 0 and tangent[-1] != 0):
                result['turn_brackets'].append(dict(point_indices=[point_index - 1, point_index],
                                                     parameter_tangents=[float(tangent[-1]), float(new_tangent[-1])],
                                                     status='numerical_turn_bracket'))
            tangent = new_tangent
            result['accepted_steps'] += 1
            attempt.update(status='accepted', point_index=point_index)
            ledger.emit('attempt_end', attempt=attempt)
            ladder_index = 0
            if stop_on_turn and result['turn_brackets']:
                result.update(status='completed', reason='qualified_tangent_turn')
                return
            if result['accepted_steps'] == config.max_accepted_steps:
                result.update(status='completed', reason='accepted_step_budget')
                return
        result.update(status='unresolved', reason='attempted_step_budget')
    return _execute(config, event, result, run)
