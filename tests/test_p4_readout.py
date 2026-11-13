import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import p4_readout as r  # noqa: E402


def refs():
    rng = np.random.default_rng(4)
    return rng.normal(size=(2, 4, 896)) * 0.01


def test_reads_each_cycle_at_every_shift():
    R = refs()
    for c in (0, 1):
        for k in range(4):
            traj = np.stack([R[c][(m + k) % 4] for m in range(41)])
            out = r.read(traj, R)
            assert out["cycle"] == c and out["settled_from_update"] == 0
            # the last four states are updates 37..40, i.e. phases (37+k)..(40+k)
            assert out["shift"] == (37 + k) % 4


def test_unresolved_when_off_both_cycles():
    R = refs()
    traj = np.stack([R[0][m % 4] for m in range(41)])
    traj[-4:] += 1e-3
    assert r.read(traj, R)["cycle"] is None


def test_settle_time():
    R = refs()
    traj = np.stack([R[1][m % 4] for m in range(41)])
    traj[:10] += 1e-3
    assert r.read(traj, R)["settled_from_update"] == 10


def test_lag_residuals_and_references():
    R = refs()
    path = np.stack([np.stack([R[0][t % 4], R[1][t % 4]]) for t in range(401)])
    assert r.lag_residual(path[:, 0], 4) == 0.0
    assert r.lag_residual(path[:, 0], 2) > 1e-3
    built = r.references_from_relaxation(path)
    assert built.shape == (2, 4, 896)
    assert np.array_equal(built, R)


def test_references_need_whole_periods():
    with pytest.raises(ValueError):
        r.references_from_relaxation(np.zeros((402, 2, 896)))


# Codex Task 9, D3: malformed input is an error; a valid but unresolved trajectory is not.
def cycle_path(R, c=0, n=41):
    return np.stack([R[c][m % 4] for m in range(n)])


@pytest.mark.parametrize("j", range(4))
def test_nan_in_any_tail_phase_is_an_error(j):
    R = refs()
    traj = cycle_path(R)
    traj[-4 + j, 7] = np.nan
    with pytest.raises(ValueError):
        r.read(traj, R)
    with pytest.raises(ValueError):
        r.lag_residual(traj, 4)
    with pytest.raises(ValueError):
        r.align(traj[-4:], R[0])


def test_nonfinite_references_are_an_error():
    R = refs()
    bad = R.copy()
    bad[1, 2, 3] = np.inf
    with pytest.raises(ValueError):
        r.read(cycle_path(R), bad)
    with pytest.raises(ValueError):
        r.references_from_relaxation(np.stack([np.stack([bad[0][t % 4], bad[1][t % 4]]) for t in range(9)]))


def test_short_histories_are_errors():
    R = refs()
    path = cycle_path(R, n=8)
    assert r.lag_residual(path, 4) == 0.0           # exactly lag + last states: one real earlier period
    with pytest.raises(ValueError):
        r.lag_residual(path[:7], 4)
    with pytest.raises(ValueError):
        r.lag_residual(path[:4], 4)                 # the wraparound case: no earlier period supplied
    with pytest.raises(ValueError):
        r.read(path[:3], R)
    with pytest.raises(ValueError):
        r.references_from_relaxation(np.zeros((1, 2, 896)))


def test_wrong_dtype_shape_or_parameters_are_errors():
    R = refs()
    traj = cycle_path(R)
    with pytest.raises(ValueError):
        r.read(traj.astype(np.float32), R)
    with pytest.raises(ValueError):
        r.read(traj[:, :895], R)
    with pytest.raises(ValueError):
        r.read(traj, R.astype(np.float32))
    for lag, last in ((0, 4), (4, 0), (-1, 4), (4.0, 4), (True, 4)):
        with pytest.raises(ValueError):
            r.lag_residual(traj, lag, last)


def test_label_needs_a_unique_qualified_reference():
    R = refs()
    twin = np.stack([R[0], R[0]])                   # both references qualify: no label, and no error
    out = r.read(cycle_path(R), twin)
    assert out["cycle"] is None and out["distance"] == 0.0 and out["settled_from_update"] is None
