"""Period-4 readout for the second register (docs/public-p4-register.md). Pure numpy, no model.

A cycle is an array (4, 896) indexed by phase; reference cycles are (2, 4, 896): 0 = sheet L, 1 = sheet U. A lane's
last four states are aligned cyclically to each reference cycle; it reads the cycle it lies within SETTLE of.
"""
from __future__ import annotations

import numpy as np

PERIOD = 4
SETTLE = 1e-6


def _finite_float64(x: np.ndarray, ndim: int, what: str) -> np.ndarray:
    """Malformed input is an error, never an unresolved readout (Codex Task 9, D3)."""
    if not isinstance(x, np.ndarray) or x.dtype != np.float64 or x.ndim != ndim or not np.isfinite(x).all():
        raise ValueError(f'{what}: expected a finite float64 array with {ndim} dimensions')
    return x


def lag_residual(states: np.ndarray, lag: int, last: int = PERIOD) -> float:
    """Largest distance between each of the last `last` states and the state `lag` updates earlier. Requires at
    least lag + last states, so no comparison wraps around to the same state."""
    _finite_float64(states, 2, 'states')
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in (lag, last)):
        raise ValueError('lag and last must be positive integers')
    n = states.shape[0]
    if n < lag + last:
        raise ValueError(f'history of {n} states is shorter than lag + last = {lag + last}')
    return float(np.max([np.linalg.norm(states[n - 1 - j] - states[n - 1 - j - lag]) for j in range(last)]))


def align(tail: np.ndarray, cycle: np.ndarray) -> tuple[int, float]:
    """Best cyclic shift k with tail[j] ~ cycle[(j + k) % P]; returns (k, largest distance at that shift)."""
    _finite_float64(tail, 2, 'tail'); _finite_float64(cycle, 2, 'cycle')
    p = cycle.shape[0]
    if tail.shape != cycle.shape:
        raise ValueError("tail and cycle shapes differ")
    vals = [float(np.max([np.linalg.norm(tail[j] - cycle[(j + k) % p]) for j in range(p)])) for k in range(p)]
    k = int(np.argmin(vals))
    return k, vals[k]


def read(trajectory: np.ndarray, refs: np.ndarray) -> dict:
    """Read one lane from its trajectory (T+1, 896): nearest reference cycle over the last period, and settle time."""
    _finite_float64(trajectory, 2, 'trajectory'); _finite_float64(refs, 3, 'refs')
    p = refs.shape[1]
    if trajectory.shape[0] < p or trajectory.shape[1] != refs.shape[2]:
        raise ValueError('trajectory is shorter than one period or its width differs from the references')
    tail = trajectory[-p:]
    per = [align(tail, refs[c]) for c in range(refs.shape[0])]
    c = int(np.argmin([d for _, d in per]))
    shift, dist = per[c]
    qualified = [k for k in range(len(per)) if per[k][1] < SETTLE]
    cycle = c if qualified == [c] else None              # a label needs a unique qualified reference
    settled_from = None
    if cycle is not None:
        t_end = trajectory.shape[0] - 1
        d = [float(np.linalg.norm(trajectory[m] - refs[c][(m - (t_end - p + 1) + shift) % p]))
             for m in range(trajectory.shape[0])]
        above = [m for m, x in enumerate(d) if x >= SETTLE]
        settled_from = (above[-1] + 1) if above else 0
    return {"cycle": cycle, "nearest": c, "shift": shift, "distance": dist,
            "other_distance": per[1 - c][1] if refs.shape[0] == 2 else None, "settled_from_update": settled_from}


def references_from_relaxation(path: np.ndarray) -> np.ndarray:
    """Reference cycles from a relaxation path (T+1, 2, 896) with T divisible by 4: phase j = the last state at an
    update congruent to j modulo 4."""
    _finite_float64(path, 3, 'path')
    t_end = path.shape[0] - 1
    if t_end < PERIOD:
        raise ValueError('relaxation shorter than one period')
    if t_end % PERIOD:
        raise ValueError("relaxation length must be a multiple of the period")
    idx = [t_end - ((t_end - j) % PERIOD) for j in range(PERIOD)]           # phase j at update idx[j]
    return np.stack([path[idx, lane] for lane in range(path.shape[1])])     # (lanes, 4, 896)
