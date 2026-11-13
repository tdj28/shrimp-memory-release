"""A bounded, model-free Newton--GMRES corrector with retained evidence.

The caller supplies G(x) and DG(x)v in float64.  This module neither constructs
a dynamical map nor infers attraction from convergence.  A converged result is
only a numerical residual observation, not an existence or stability proof.

No linear solve is accepted from its projected residual alone.  Every usable
GMRES correction receives a fresh DG(x)delta action, checked both against the
saved Arnoldi actions and against the full-space linear equation.  There are no
restarts, regularization, state normalization, or trust-ball recentering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class CorrectorConfig:
    """Explicit bounds; radius is measured from the fixed center, by default x0."""

    trust_radius: float
    max_newton_iterations: int = 4
    max_gmres_actions: int = 8
    max_residual_calls: int = 13
    max_jacobian_calls: int = 36
    root_tolerance: float = 1e-10
    linear_relative_tolerance: float = 0.1
    action_consistency_tolerance: float = 1e-10
    damping: tuple[float, ...] = (1.0, 0.5, 0.25)
    sufficient_decrease: float = 1e-4
    breakdown_tolerance: float = 1e-13

    def __post_init__(self) -> None:
        for name in ("max_newton_iterations", "max_gmres_actions",
                     "max_residual_calls", "max_jacobian_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("trust_radius", "root_tolerance", "linear_relative_tolerance",
                     "action_consistency_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.breakdown_tolerance) or self.breakdown_tolerance < 0:
            raise ValueError("breakdown_tolerance must be nonnegative and finite")
        if not math.isfinite(self.sufficient_decrease) or not 0 <= self.sufficient_decrease < 1:
            raise ValueError("sufficient_decrease must lie in [0, 1)")
        damping = tuple(self.damping)
        if (not damping or any(not math.isfinite(v) or not 0 < v <= 1 for v in damping)
                or any(a <= b for a, b in zip(damping, damping[1:]))):
            raise ValueError("damping must be nonempty, strictly decreasing, and in (0, 1]")
        object.__setattr__(self, "damping", damping)


def array_sha256(value: np.ndarray) -> str:
    """Hash C-order array bytes; callers must bind dtype and shape separately."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def serializable(value):
    """Make detached, strict-JSON evidence, including nonfinite failed outputs.

    Arrays carry dtype, shape and values.  Nonfinite floats are strings, never
    silently replaced by finite numbers.  The ordinary result retains arrays.
    """
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "sha256": array_sha256(value), "values": serializable(value.tolist())}
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"python_type": type(value).__name__, "repr": repr(value)}


def _norm(value: np.ndarray) -> float:
    """Scaled Euclidean norm without avoidable overflow from squaring."""
    scale = float(np.max(np.abs(value), initial=0.0))
    if scale == 0 or not math.isfinite(scale):
        return scale
    return float(scale * np.sqrt(np.sum((value / scale) ** 2, dtype=np.float64)))


def _vector(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must have float64 dtype")
    if array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array.copy()


class _Stop(Exception):
    def __init__(self, status: str, reason: str):
        self.status, self.reason = status, reason
        super().__init__(reason)


def correct(
    residual: Callable[[np.ndarray], np.ndarray],
    jacobian_action: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    config: CorrectorConfig,
    initial_residual: Mapping | None = None,
    trust_center: np.ndarray | None = None,
    event: Callable[[dict], None] | None = None,
) -> dict:
    """Correct G(x)=0 subject to explicit callback and iteration budgets.

    ``jacobian_action(x, v)`` must return DG(x)v.  Every callback receives fresh
    input copies.  Attempted calls count even if they raise or return bad data.
    A supplied initial residual requires a mapping with ``residual``,
    ``x_sha256`` and ``residual_sha256``; it is labelled supplied, not evaluated.
    The caller is responsible for the provenance of that binding.

    ``trust_center`` optionally fixes the trust ball independently of the
    starting state.  It must be finite float64 with x0's shape; the default
    is x0.  The initial state must lie inside the ball before any callbacks.
    The center is copied, saved and hash-bound, and never moves.  Trials retain
    both ``distance_from_seed`` (x0) and ``distance_from_trust_center``; only
    the latter controls trust admission.

    Result statuses are ``converged``, ``unresolved`` (finite solver/budget/trust
    exhaustion), and ``failed`` (callback or numerical instrument failure).
    ``reason`` distinguishes each stop.  Invalid configuration/input raises
    before any callbacks.  Observer exceptions propagate: failed evidence
    persistence must not be mistaken for a successful return.

    The optional observer receives strict-JSON snapshots of ``iteration_begin``
    and ``iteration_end``, with zero-based ``iteration``.  Every opened iteration
    has an end event, including callback failures.  Accepted trial residuals are
    carried exactly; there is no extra nonlinear evaluation at the next iterate.
    """
    if not isinstance(config, CorrectorConfig):
        raise TypeError("config must be a CorrectorConfig")
    seed = np.asarray(x0)
    if seed.ndim != 1 or seed.size == 0:
        raise ValueError("x0 must be a nonempty one-dimensional float64 vector")
    seed = _vector(seed, seed.shape, "x0")
    center = seed.copy() if trust_center is None else _vector(trust_center, seed.shape, "trust_center")
    with np.errstate(over="ignore", invalid="ignore"):
        initial_distance = _norm(seed - center)
    if not math.isfinite(initial_distance):
        raise ValueError("initial distance from trust_center must be finite")
    if initial_distance > config.trust_radius:
        raise ValueError("x0 is outside the fixed trust ball")
    x, r = seed.copy(), None
    counts = {"residual_calls": 0, "jacobian_calls": 0, "total_calls": 0}
    result = {"status": "running", "reason": None, "config": asdict(config),
              "trust_center": center.copy(), "trust_center_sha256": array_sha256(center),
              "initial_distance_from_trust_center": initial_distance,
              "counts": counts, "calls": [], "iterations": [], "iterates": [],
              "initial_residual_source": "callback" if initial_residual is None else "supplied"}

    def emit(payload: dict) -> None:
        if event is not None:
            event(serializable(payload))

    def invoke(kind: str, base: np.ndarray, direction: np.ndarray | None,
               purpose: str, iteration: int | None) -> tuple[np.ndarray, int]:
        key = "residual_calls" if kind == "residual" else "jacobian_calls"
        cap = config.max_residual_calls if kind == "residual" else config.max_jacobian_calls
        call = {"id": len(result["calls"]), "kind": kind, "purpose": purpose,
                "iteration": iteration, "x": base.copy(),
                "x_sha256": array_sha256(base), "called": False, "status": "pending"}
        if direction is not None:
            call["direction"] = direction.copy()
        result["calls"].append(call)
        if counts[key] >= cap:
            call["status"] = "not_called_budget"
            raise _Stop("unresolved", f"{kind}_call_budget")
        counts[key] += 1
        counts["total_calls"] += 1
        call["called"] = True
        try:
            raw = residual(base.copy()) if direction is None else jacobian_action(base.copy(), direction.copy())
            # Preserve the received value before validation or any next callback.
            call["output"] = np.asarray(raw).copy() if isinstance(raw, (np.ndarray, list, tuple)) else raw
            value = _vector(raw, seed.shape, f"{kind} output")
            if not math.isfinite(_norm(value)):
                raise ValueError(f"{kind} output has a nonfinite Euclidean norm")
        except Exception as exc:
            call.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise _Stop("failed", f"{kind}_callback_failure") from exc
        call.update(status="completed", output=value.copy(), output_sha256=array_sha256(value))
        return value, call["id"]

    def finite_arithmetic(*values) -> None:
        if any(not np.all(np.isfinite(value)) for value in values):
            raise _Stop("failed", "nonfinite_solver_arithmetic")

    def gmres(base: np.ndarray, rhs_residual: np.ndarray, iteration: int, record: dict) -> np.ndarray:
        beta = _norm(rhs_residual)
        m, n = config.max_gmres_actions, seed.size
        linear = {"rhs_norm": beta, "rhs": -rhs_residual.copy(), "steps": [],
                  "fresh_action": None, "stop_reason": None}
        record["linear"] = linear
        if m == 0:
            raise _Stop("unresolved", "gmres_action_budget")
        q = np.zeros((n, m + 1), dtype=np.float64)
        h = np.zeros((m + 1, m), dtype=np.float64)
        w = np.zeros((n, m), dtype=np.float64)
        triangular = np.zeros_like(h)
        rotated_rhs = np.zeros(m + 1, dtype=np.float64)
        rotated_rhs[0] = beta
        rotations: list[tuple[float, float]] = []
        q[:, 0] = -rhs_residual / beta
        k, basis_size = 0, 1
        y, delta = None, None
        try:
            for j in range(m):
                step = {"index": j, "basis_vector": q[:, j].copy()}
                linear["steps"].append(step)
                raw, call_id = invoke("jacobian", base, q[:, j], "arnoldi", iteration)
                step.update(call_id=call_id, raw_action=raw.copy())
                w[:, j] = raw
                k = j + 1
                v = raw.copy()
                passes = []
                # Two explicit modified Gram--Schmidt passes; no model calls here.
                for _ in range(2):
                    coefficients = np.zeros(j + 1, dtype=np.float64)
                    for i in range(j + 1):
                        coefficients[i] = np.dot(q[:, i], v)
                        h[i, j] += coefficients[i]
                        v -= coefficients[i] * q[:, i]
                    passes.append(coefficients)
                h[j + 1, j] = _norm(v)
                breakdown = h[j + 1, j] <= config.breakdown_tolerance
                step.update(orthogonalization_passes=passes, orthogonal_remainder=v.copy(),
                            subdiagonal=float(h[j + 1, j]), breakdown=bool(breakdown))
                finite_arithmetic(h[:, j], v)
                if not breakdown:
                    q[:, j + 1] = v / h[j + 1, j]
                    basis_size = j + 2
                try:
                    singular_values = np.linalg.svd(h[:k + 1, :k], compute_uv=False)
                except np.linalg.LinAlgError as exc:
                    raise _Stop("failed", "projected_diagnostic_failure") from exc
                # Diagnostic rank only: it never truncates the solve or changes a step.
                rank_threshold = np.finfo(np.float64).eps * (k + 1) * float(singular_values[0])
                step.update(projected_singular_values=singular_values.copy(),
                            projected_rank=int(np.count_nonzero(singular_values > rank_threshold)),
                            projected_rank_threshold=rank_threshold,
                            projected_rank_used_as_gate=False)
                column = h[:, j].copy()
                for i, (c, s) in enumerate(rotations):
                    a, b = column[i], column[i + 1]
                    column[i], column[i + 1] = c * a + s * b, -s * a + c * b
                rho = float(np.hypot(column[j], column[j + 1]))
                if rho == 0:
                    linear["stop_reason"] = "singular_projected_system"
                    raise _Stop("unresolved", "gmres_singular_projected_system")
                finite_arithmetic(rho)
                c, s = float(column[j] / rho), float(column[j + 1] / rho)
                rotations.append((c, s))
                column[j], column[j + 1] = rho, 0.0
                triangular[:, j] = column
                a, b = rotated_rhs[j], rotated_rhs[j + 1]
                rotated_rhs[j], rotated_rhs[j + 1] = c * a + s * b, -s * a + c * b
                # Nonsingular triangular solve: no pseudo-inverse or truncation.
                try:
                    y = np.linalg.solve(triangular[:k, :k], rotated_rhs[:k])
                except np.linalg.LinAlgError as exc:
                    linear["stop_reason"] = "singular_projected_system"
                    raise _Stop("unresolved", "gmres_singular_projected_system") from exc
                delta = q[:, :k] @ y
                reduced_rhs = np.zeros(k + 1, dtype=np.float64)
                reduced_rhs[0] = beta
                projected = h[:k + 1, :k] @ y - reduced_rhs
                projected_relative = _norm(projected) / beta
                step.update(coefficients=y.copy(), correction=delta.copy(),
                            projected_residual=projected.copy(), projected_relative=projected_relative,
                            rotation=(c, s))
                finite_arithmetic(y, delta, projected, projected_relative)
                if projected_relative <= config.linear_relative_tolerance:
                    linear["stop_reason"] = "projected_tolerance"
                    break
                if breakdown:
                    linear["stop_reason"] = "invariant_subspace_breakdown"
                    break
            else:
                linear["stop_reason"] = "gmres_action_budget"
            if delta is None or y is None:
                raise _Stop("unresolved", "gmres_missing_correction")
            combination = w[:, :k] @ y
            linear.update(coefficients=y.copy(), correction=delta.copy(),
                          action_combination=combination.copy())
            fresh, call_id = invoke("jacobian", base, delta, "fresh_linear_residual", iteration)
            full_residual = fresh + rhs_residual
            difference = fresh - combination
            full_norm, difference_norm = _norm(full_residual), _norm(difference)
            consistency_denominator = max(_norm(fresh), _norm(combination), 1e-14)
            full_relative = full_norm / beta
            consistency_relative = difference_norm / consistency_denominator
            linear["fresh_action"] = {"call_id": call_id, "value": fresh.copy(),
                                      "full_residual": full_residual.copy(),
                                      "full_residual_norm": full_norm,
                                      "full_relative": full_relative,
                                      "combination_difference": difference.copy(),
                                      "combination_difference_norm": difference_norm,
                                      "consistency_denominator": consistency_denominator,
                                      "consistency_relative": consistency_relative}
            finite_arithmetic(combination, full_residual, difference, full_relative, consistency_relative)
            if consistency_relative > config.action_consistency_tolerance:
                raise _Stop("failed", "fresh_action_inconsistent")
            if full_relative > config.linear_relative_tolerance:
                raise _Stop("unresolved", "fresh_linear_residual_insufficient")
            return delta
        finally:
            # Also retain partial systems if an action or a later gate failed.
            linear.update(basis=q[:, :basis_size].copy(), hessenberg=h[:k + 1, :k].copy(),
                          raw_actions=w[:, :k].copy(), triangular=triangular[:k + 1, :k].copy(),
                          rotated_rhs=rotated_rhs[:k + 1].copy(), rotations=rotations.copy(),
                          arnoldi_actions=k)

    try:
        if initial_residual is None:
            r, initial_call = invoke("residual", x, None, "initial_residual", None)
            result["initial_residual_call_id"] = initial_call
        else:
            if not isinstance(initial_residual, Mapping):
                raise ValueError("initial_residual requires an explicit state and residual hash binding")
            r = _vector(initial_residual.get("residual"), seed.shape, "supplied initial residual")
            if (initial_residual.get("x_sha256") != array_sha256(seed)
                    or initial_residual.get("residual_sha256") != array_sha256(r)):
                raise ValueError("supplied initial residual hash binding does not match x0 and residual")
            if not math.isfinite(_norm(r)):
                raise ValueError("supplied initial residual norm is nonfinite")
            result["initial_residual_binding"] = {"x_sha256": array_sha256(seed),
                                                  "residual_sha256": array_sha256(r),
                                                  "dtype": "float64", "shape": list(seed.shape)}
        result["iterates"].append({"iteration": -1, "x": x.copy(), "residual": r.copy(),
                                   "residual_norm": _norm(r)})
        if _norm(r) <= config.root_tolerance:
            raise _Stop("converged", "initial_residual_tolerance")
        for iteration in range(config.max_newton_iterations):
            before_counts = counts.copy()
            first_call = len(result["calls"])
            record = {"iteration": iteration, "x": x.copy(), "residual": r.copy(),
                      "residual_norm": _norm(r), "trials": [], "status": "running", "reason": None}
            result["iterations"].append(record)
            try:
                emit({"type": "iteration_begin", "iteration": iteration, "x": x.copy(),
                      "residual": r.copy(), "counts": counts.copy()})
                delta = gmres(x, r, iteration, record)
                accepted = False
                for damping in config.damping:
                    candidate = x + damping * delta
                    distance = _norm(candidate - center)
                    seed_distance = _norm(candidate - seed)
                    trial = {"damping": damping, "x": candidate.copy(), "distance_from_seed": seed_distance,
                             "distance_from_trust_center": distance,
                             "called": False, "accepted": False, "status": "pending"}
                    record["trials"].append(trial)
                    finite_arithmetic(candidate, distance, seed_distance)
                    if distance > config.trust_radius:
                        trial["status"] = "skipped_outside_trust_ball"
                        continue
                    trial["status"] = "residual_requested"
                    try:
                        trial_r, call_id = invoke("residual", candidate, None, "damping_trial", iteration)
                    except _Stop:
                        trial["status"] = "residual_unavailable"
                        raise
                    finally:
                        trial["call_id"] = len(result["calls"]) - 1
                        trial["called"] = result["calls"][-1]["called"]
                        trial["callback_status"] = result["calls"][-1]["status"]
                    norm = _norm(trial_r)
                    bound = (1 - config.sufficient_decrease * damping) * _norm(r)
                    accepted = norm <= config.root_tolerance or norm <= bound
                    trial.update(call_id=call_id, residual=trial_r.copy(), residual_norm=norm,
                                 sufficient_decrease_bound=bound, accepted=bool(accepted),
                                 status="accepted" if accepted else "rejected_insufficient_decrease")
                    if accepted:
                        x, r = candidate.copy(), trial_r.copy()
                        result["iterates"].append({"iteration": iteration, "x": x.copy(),
                                                   "residual": r.copy(), "residual_norm": norm,
                                                   "call_id": call_id, "damping": damping})
                        break
                if not accepted:
                    reason = ("trust_ball_exhausted" if all(not t["called"] for t in record["trials"])
                              else "line_search_exhausted")
                    raise _Stop("unresolved", reason)
                if _norm(r) <= config.root_tolerance:
                    raise _Stop("converged", "root_residual_tolerance")
                if iteration + 1 == config.max_newton_iterations:
                    raise _Stop("unresolved", "newton_iteration_budget")
                record.update(status="accepted", reason="continue_with_retained_residual")
            except _Stop as stop:
                record.update(status=stop.status, reason=stop.reason)
                raise
            except BaseException as exc:
                record.update(status="failed", reason="observer_or_execution_exception",
                              error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                record.update(x_after=x.copy(), residual_after=r.copy(), residual_norm_after=_norm(r),
                              counts_before=before_counts, counts_after=counts.copy(),
                              call_ids=list(range(first_call, len(result["calls"]))))
                emit({"type": "iteration_end", "iteration": iteration,
                      "status": record["status"], "reason": record["reason"],
                      "x": x.copy(), "residual": r.copy(), "counts": counts.copy(),
                      "evidence": record, "calls": result["calls"][first_call:]})
        raise _Stop("unresolved", "newton_iteration_budget")
    except _Stop as stop:
        result.update(status=stop.status, reason=stop.reason)
    result.update(x=x.copy(), residual=None if r is None else r.copy(),
                  residual_norm=None if r is None else _norm(r))
    return result
