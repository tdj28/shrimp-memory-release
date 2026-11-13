"""Model-free correctness, budget, and evidence tests for bounded correction."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bounded_periodic_corrector import CorrectorConfig, array_sha256, correct, serializable


def cfg(**changes):
    return replace(CorrectorConfig(trust_radius=100.0), **changes)


def affine(matrix, target, *, x0=None, config=None, event=None):
    matrix, target = np.asarray(matrix, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if x0 is None:
        x0 = np.zeros(len(target), dtype=np.float64)
    return correct(lambda x: matrix @ (x - target), lambda x, v: matrix @ v,
                   x0, config=config or cfg(), event=event)


def assert_counts(result):
    called = [c for c in result["calls"] if c["called"]]
    assert result["counts"]["residual_calls"] == sum(c["kind"] == "residual" for c in called)
    assert result["counts"]["jacobian_calls"] == sum(c["kind"] == "jacobian" for c in called)
    assert result["counts"]["total_calls"] == len(called)
    assert result["counts"]["residual_calls"] <= result["config"]["max_residual_calls"]
    assert result["counts"]["jacobian_calls"] <= result["config"]["max_jacobian_calls"]
    for call in result["calls"]:
        assert array_sha256(call["x"]) == call["x_sha256"]
    json.dumps(serializable(result), allow_nan=False)


def test_known_affine_root_and_full_evidence():
    matrix = np.array([[3., 1.], [0., 2.]])
    target = np.array([.5, -.2])
    events = []
    result = affine(matrix, target, event=events.append,
                    config=cfg(linear_relative_tolerance=1e-12))
    assert result["status"] == "converged"
    np.testing.assert_allclose(result["x"], target, atol=1e-14)
    assert result["residual_norm"] <= 1e-10
    assert result["counts"] == dict(residual_calls=2, jacobian_calls=3, total_calls=5)
    linear = result["iterations"][0]["linear"]
    q, h, w, y = (linear[name] for name in ("basis", "hessenberg", "raw_actions", "coefficients"))
    k = len(y)
    np.testing.assert_allclose(w, matrix @ q[:, :k], atol=1e-14)
    # Happy breakdown leaves only the existing independent basis vectors.
    np.testing.assert_allclose(q @ h[:q.shape[1]], w, atol=1e-14)
    np.testing.assert_allclose(linear["correction"], q[:, :k] @ y, atol=1e-14)
    np.testing.assert_allclose(linear["action_combination"], w @ y, atol=1e-14)
    np.testing.assert_allclose(linear["fresh_action"]["full_residual"],
                               matrix @ linear["correction"] + result["iterates"][0]["residual"], atol=1e-14)
    assert [e["type"] for e in events] == ["iteration_begin", "iteration_end"]
    assert events[-1]["status"] == "converged" and events[-1]["iteration"] == 0
    assert events[-1]["evidence"]["trials"][0]["accepted"] is True
    assert_counts(result)


@pytest.mark.parametrize("multiplier", [.5, 2., -2.])
def test_stable_and_unstable_map_roots_both_correct(multiplier):
    # Six-step map Bx+c has the target as a root for stable and unstable B.
    # A root result contains no attraction or stability verdict.
    matrix = (multiplier ** 6 - 1) * np.eye(2)
    result = affine(matrix, [.3, -.4])
    assert result["status"] == "converged"
    np.testing.assert_allclose(result["x"], [.3, -.4], atol=1e-14)
    assert "stable" not in result and "attracting" not in result
    assert_counts(result)


def test_strongly_nonnormal_affine_system_uses_unmodified_actions():
    matrix = np.array([[1., 100.], [0., 2.]])
    result = affine(matrix, [.1, .2], config=cfg(linear_relative_tolerance=1e-12))
    assert result["status"] == "converged"
    np.testing.assert_allclose(result["x"], [.1, .2], atol=1e-11)
    linear = result["iterations"][0]["linear"]
    assert linear["arnoldi_actions"] == 2
    assert linear["steps"][-1]["projected_rank"] == 2
    assert linear["steps"][-1]["projected_rank_used_as_gate"] is False
    assert len(linear["steps"][-1]["projected_singular_values"]) == 2
    assert max(np.linalg.norm(s["raw_action"]) for s in linear["steps"]) > 10
    for call in result["calls"]:
        if call["kind"] == "jacobian":
            np.testing.assert_array_equal(call["x"], np.zeros(2))
            np.testing.assert_allclose(call["output"], matrix @ call["direction"], atol=1e-14)
    assert_counts(result)


@pytest.mark.parametrize("matrix", [np.zeros((2, 2)), np.diag([1., 0.])])
def test_singular_no_root_never_defaults_to_a_solution(matrix):
    result = correct(lambda x: matrix @ x - np.array([1., 1.]),
                     lambda x, v: matrix @ v, np.zeros(2), config=cfg())
    assert result["status"] == "unresolved"
    assert result["reason"] in {"gmres_singular_projected_system", "fresh_linear_residual_insufficient"}
    assert result["residual_norm"] > 1
    assert not result["iterations"][0]["trials"]
    np.testing.assert_array_equal(result["x"], np.zeros(2))
    assert_counts(result)


def test_happy_breakdown_requires_fresh_action_without_dividing_by_zero():
    result = affine(np.eye(2), [1., 0.])
    linear = result["iterations"][0]["linear"]
    assert result["status"] == "converged"
    assert linear["steps"][0]["breakdown"] is True
    assert linear["steps"][0]["subdiagonal"] == 0
    assert linear["arnoldi_actions"] == 1
    assert linear["fresh_action"]["full_relative"] == 0
    assert result["counts"]["jacobian_calls"] == 2
    assert_counts(result)


def test_projected_zero_does_not_accept_inconsistent_fresh_action():
    actions = []

    def action(x, v):
        actions.append(v.copy())
        return v if len(actions) == 1 else 2 * v

    result = correct(lambda x: x - 1, action, np.zeros(1), config=cfg())
    linear = result["iterations"][0]["linear"]
    assert linear["steps"][-1]["projected_relative"] == 0
    assert linear["fresh_action"]["full_relative"] == 1
    assert linear["fresh_action"]["consistency_relative"] == .5
    assert result["status"] == "failed" and result["reason"] == "fresh_action_inconsistent"
    assert not result["iterations"][0]["trials"]
    assert result["counts"]["residual_calls"] == 1
    assert_counts(result)


def test_fresh_full_space_residual_can_fail_with_consistent_actions():
    result = affine(np.diag([1., 3.]), [1., 1.], config=cfg(max_gmres_actions=1))
    linear = result["iterations"][0]["linear"]
    assert linear["stop_reason"] == "gmres_action_budget"
    assert linear["fresh_action"]["consistency_relative"] <= 1e-14
    assert linear["fresh_action"]["full_relative"] > .1
    assert result["status"] == "unresolved"
    assert result["reason"] == "fresh_linear_residual_insufficient"
    assert result["counts"]["jacobian_calls"] == 2
    assert not result["iterations"][0]["trials"]


def test_nonlinear_damping_keeps_both_rejections_and_accepted_residual():
    result = correct(lambda x: x * x - 1, lambda x, v: 2 * x * v,
                     np.array([.1]), config=cfg(max_newton_iterations=1))
    trials = result["iterations"][0]["trials"]
    assert [t["damping"] for t in trials] == [1, .5, .25]
    assert [t["accepted"] for t in trials] == [False, False, True]
    np.testing.assert_allclose([t["x"][0] for t in trials], [5.05, 2.575, 1.3375])
    assert result["status"] == "unresolved" and result["reason"] == "newton_iteration_budget"
    np.testing.assert_array_equal(result["residual"], trials[-1]["residual"])
    assert result["counts"]["residual_calls"] == 4
    assert_counts(result)


def test_no_damping_trial_accepted_is_unresolved():
    result = correct(lambda x: np.ones_like(x), lambda x, v: v,
                     np.zeros(1), config=cfg())
    assert result["status"] == "unresolved" and result["reason"] == "line_search_exhausted"
    assert len(result["iterations"][0]["trials"]) == 3
    assert all(t["called"] and not t["accepted"] for t in result["iterations"][0]["trials"])
    assert len(result["iterates"]) == 1
    assert_counts(result)


def test_outside_trust_ball_is_recorded_without_residual_calls():
    result = affine(np.eye(1), [4.], config=cfg(trust_radius=.5))
    assert result["status"] == "unresolved" and result["reason"] == "trust_ball_exhausted"
    trials = result["iterations"][0]["trials"]
    assert [t["distance_from_seed"] for t in trials] == [4., 2., 1.]
    assert all(t["status"] == "skipped_outside_trust_ball" and not t["called"] for t in trials)
    assert result["counts"]["residual_calls"] == 1
    assert_counts(result)


def test_trust_ball_never_recenters_or_clips():
    result = correct(lambda x: x * x - 4, lambda x, v: 2 * x * v,
                     np.array([1.]), config=cfg(trust_radius=.6))
    assert result["status"] == "unresolved" and result["reason"] == "trust_ball_exhausted"
    np.testing.assert_array_equal(result["trust_center"], [1.])
    for record in result["iterations"]:
        for trial in record["trials"]:
            assert trial["distance_from_seed"] == abs(trial["x"][0] - 1)
            np.testing.assert_array_equal(trial["x"], record["x"] + trial["damping"] * record["linear"]["correction"])
    assert all(abs(i["x"][0] - 1) <= .6 for i in result["iterates"])
    assert_counts(result)


def test_explicit_original_center_rejects_trial_that_recentering_would_accept():
    x0 = np.array([.75])
    options = dict(config=cfg(trust_radius=1., damping=(1.,), max_newton_iterations=1))
    residual, action = lambda x: x - 1.5, lambda x, v: v
    recentered = correct(residual, action, x0, **options)
    anchored = correct(residual, action, x0, trust_center=np.zeros(1), **options)
    assert recentered['status'] == 'converged'
    assert anchored['status'] == 'unresolved' and anchored['reason'] == 'trust_ball_exhausted'
    trial = anchored['iterations'][0]['trials'][0]
    assert trial['distance_from_seed'] == .75
    assert trial['distance_from_trust_center'] == 1.5
    assert trial['status'] == 'skipped_outside_trust_ball' and trial['called'] is False
    assert anchored['initial_distance_from_trust_center'] == .75
    assert anchored['trust_center_sha256'] == array_sha256(np.zeros(1))
    assert anchored['counts']['residual_calls'] == 1
    np.testing.assert_array_equal(anchored['x'], x0)
    assert_counts(anchored)


def test_original_center_distance_controls_each_trial_across_iterations():
    center, x0 = np.array([0.]), np.array([.75])
    result = correct(lambda x: x*x - 4, lambda x, v: 2*x*v, x0,
                     trust_center=center, config=cfg(trust_radius=1.5))
    assert len(result['iterations']) > 1
    assert result['reason'] == 'trust_ball_exhausted'
    for record in result['iterations']:
        for trial in record['trials']:
            assert trial['distance_from_trust_center'] == abs(trial['x'][0])
            assert trial['distance_from_seed'] == abs(trial['x'][0] - .75)
            if trial['distance_from_trust_center'] > 1.5:
                assert not trial['called'] and not trial['accepted']
    assert all(abs(item['x'][0]) <= 1.5 for item in result['iterates'])
    assert_counts(result)


def test_default_and_explicit_seed_center_are_identical():
    x0 = np.array([.1])
    residual, action = lambda x:x*x - 1, lambda x,v:2*x*v
    default = correct(residual, action, x0, config=cfg())
    explicit = correct(residual, action, x0, config=cfg(), trust_center=x0)
    assert serializable(default) == serializable(explicit)
    for record in default['iterations']:
        for trial in record['trials']:
            assert trial['distance_from_seed'] == trial['distance_from_trust_center']


@pytest.mark.parametrize('center', [np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.int64),
    np.zeros(1, dtype=np.complex128), np.array([np.nan]), np.array([np.inf]), np.zeros(2), np.zeros((1,1))])
def test_invalid_explicit_center_never_calls_callback(center):
    with pytest.raises(ValueError, match='trust_center'):
        correct(lambda x:pytest.fail('invalid center called residual'),
                lambda x,v:pytest.fail('invalid center called action'), np.zeros(1),
                config=cfg(), trust_center=center)


@pytest.mark.parametrize('supplied', [False, True])
def test_outside_initial_point_is_rejected_even_with_zero_supplied_residual(supplied):
    x0, residual = np.array([1.]), np.zeros(1)
    bound = dict(residual=residual, x_sha256=array_sha256(x0), residual_sha256=array_sha256(residual))
    with pytest.raises(ValueError, match='outside the fixed trust ball'):
        correct(lambda x:pytest.fail('outside initial point called residual'),
                lambda x,v:pytest.fail('outside initial point called action'), x0,
                config=cfg(trust_radius=np.nextafter(1., 0.)), trust_center=np.zeros(1),
                initial_residual=bound if supplied else None)


def test_exact_initial_boundary_is_admitted_without_clipping():
    x0 = np.array([1.])
    result = correct(lambda x:np.zeros_like(x), lambda x,v:pytest.fail('already root'),
                     x0, config=cfg(trust_radius=1.), trust_center=np.zeros(1))
    assert result['status'] == 'converged' and result['initial_distance_from_trust_center'] == 1.
    np.testing.assert_array_equal(result['x'], x0)
    assert_counts(result)


def test_overflowing_initial_center_distance_is_rejected_without_callbacks():
    high = np.finfo(np.float64).max
    with pytest.raises(ValueError, match='distance from trust_center must be finite'):
        correct(lambda x:pytest.fail('overflow distance called residual'),
                lambda x,v:pytest.fail('overflow distance called action'), np.array([high]),
                config=cfg(trust_radius=high), trust_center=np.array([-high]))


def test_center_and_seed_are_copied_before_callbacks_and_evidence_is_detached():
    x0, center = np.array([.25]), np.zeros(1)
    def residual(x):
        x0[:] = 999
        center[:] = 999
        return x - .5
    result = correct(residual, lambda x,v:v, x0, trust_center=center, config=cfg(trust_radius=1.))
    assert result['status'] == 'converged'
    np.testing.assert_array_equal(result['trust_center'], [0.])
    np.testing.assert_array_equal(result['iterates'][0]['x'], [.25])
    assert result['trust_center_sha256'] == array_sha256(np.zeros(1))
    assert result['iterations'][0]['trials'][0]['distance_from_trust_center'] == .5
    encoded = serializable(result)
    encoded['trust_center']['values'][0] = 888
    np.testing.assert_array_equal(result['trust_center'], [0.])
    assert_counts(result)


def test_explicit_readonly_inputs_stay_unchanged():
    x0, center = np.array([.25, 0.]), np.zeros(2)
    before_x, before_center = x0.tobytes(), center.tobytes()
    x0.setflags(write=False); center.setflags(write=False)
    result = correct(lambda x:x - np.array([.5, 0.]), lambda x,v:v,
                     x0, trust_center=center, config=cfg(trust_radius=1.))
    assert result['status'] == 'converged'
    assert x0.tobytes() == before_x and center.tobytes() == before_center
    assert not np.shares_memory(result['trust_center'], center)
    assert not np.shares_memory(result['iterates'][0]['x'], x0)


@pytest.mark.parametrize("changes,reason,residual_calls,jacobian_calls", [
    ({"max_residual_calls": 0}, "residual_call_budget", 0, 0),
    ({"max_newton_iterations": 0}, "newton_iteration_budget", 1, 0),
    ({"max_gmres_actions": 0}, "gmres_action_budget", 1, 0),
    ({"max_jacobian_calls": 0}, "jacobian_call_budget", 1, 0),
    ({"max_jacobian_calls": 1}, "jacobian_call_budget", 1, 1),
    ({"max_residual_calls": 1}, "residual_call_budget", 1, 2),
])
def test_global_budgets_stop_before_call(changes, reason, residual_calls, jacobian_calls):
    result = affine(np.eye(1), [1.], config=cfg(**changes))
    assert result["status"] == "unresolved" and result["reason"] == reason
    assert result["counts"]["residual_calls"] == residual_calls
    assert result["counts"]["jacobian_calls"] == jacobian_calls
    assert not any(t["accepted"] for row in result["iterations"] for t in row["trials"])
    assert_counts(result)


@pytest.mark.parametrize("kind", ["residual", "jacobian"])
@pytest.mark.parametrize("fault", ["raises", "none", "nan", "inf", "shape", "float32"])
def test_callback_failures_keep_attempts_and_close_iteration(kind, fault):
    events = []

    def invalid(x):
        if fault == "raises":
            raise RuntimeError("synthetic mid-composition failure")
        return {"none": None, "nan": np.array([np.nan]), "inf": np.array([np.inf]),
                "shape": np.ones(2), "float32": np.ones(1, dtype=np.float32)}[fault]

    residual = invalid if kind == "residual" else lambda x: x - 1
    action = (lambda x, v: invalid(x)) if kind == "jacobian" else lambda x, v: v
    result = correct(residual, action, np.zeros(1), config=cfg(), event=events.append)
    assert result["status"] == "failed" and result["reason"] == f"{kind}_callback_failure"
    call = result["calls"][-1]
    assert call["called"] and call["status"] == "failed"
    np.testing.assert_array_equal(call["x"], np.zeros(1))
    if kind == "jacobian":
        assert [e["type"] for e in events] == ["iteration_begin", "iteration_end"]
        assert events[-1]["status"] == "failed"
        assert events[-1]["calls"][-1]["status"] == "failed"
        assert "direction" in call
    else:
        assert events == []
    assert_counts(result)


def test_trial_callback_exception_keeps_candidate_and_old_accepted_state():
    def residual(x):
        if x[0] != 0:
            raise RuntimeError("failed trial composition")
        return x - 1

    result = correct(residual, lambda x, v: v, np.zeros(1), config=cfg())
    assert result["status"] == "failed"
    trial = result["iterations"][0]["trials"][0]
    assert trial["called"] and trial["callback_status"] == "failed"
    np.testing.assert_array_equal(trial["x"], [1.])
    np.testing.assert_array_equal(result["x"], [0.])
    assert result["counts"]["residual_calls"] == 2
    assert_counts(result)


def test_exact_zero_initial_residual_stops_without_actions_or_iterations():
    def action(x, v):
        pytest.fail("zero residual must not invoke the Jacobian")

    result = correct(lambda x: np.zeros_like(x), action, np.array([2., 3.]), config=cfg())
    assert result["status"] == "converged" and result["reason"] == "initial_residual_tolerance"
    assert result["residual_norm"] == 0
    assert not result["iterations"]
    assert result["counts"] == dict(residual_calls=1, jacobian_calls=0, total_calls=1)


def test_hash_bound_supplied_initial_residual_is_explicit_and_not_a_call():
    x0, residual = np.array([2., 3.]), np.zeros(2)
    supplied = {"x_sha256": array_sha256(x0), "residual": residual,
                "residual_sha256": array_sha256(residual)}

    def no_call(*args):
        pytest.fail("the bound supplied zero residual needs no callback")

    result = correct(no_call, no_call, x0, config=cfg(), initial_residual=supplied)
    assert result["status"] == "converged"
    assert result["initial_residual_source"] == "supplied"
    assert result["counts"]["total_calls"] == 0 and result["calls"] == []
    residual[:] = 9
    np.testing.assert_array_equal(result["residual"], np.zeros(2))
    assert_counts(result)


@pytest.mark.parametrize("fault", ["bare", "wrong_state", "wrong_residual"])
def test_initial_residual_cannot_silently_bind_another_state(fault):
    x0, residual = np.ones(2), np.zeros(2)
    supplied = {"x_sha256": array_sha256(x0), "residual": residual,
                "residual_sha256": array_sha256(residual)}
    if fault == "bare":
        supplied = residual
    elif fault == "wrong_state":
        supplied["x_sha256"] = array_sha256(np.zeros(2))
    else:
        supplied["residual_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding"):
        correct(lambda x: pytest.fail("no callback"), lambda x, v: pytest.fail("no callback"),
                x0, config=cfg(), initial_residual=supplied)


def test_callback_mutation_and_reused_output_buffer_do_not_corrupt_records():
    output = np.zeros(2)

    def residual(x):
        output[:] = x - np.array([1., 2.])
        x[:] = 999
        return output

    def action(x, v):
        output[:] = v
        x[:] = 888
        v[:] = 777
        return output

    x0 = np.zeros(2)
    result = correct(residual, action, x0, config=cfg())
    assert result["status"] == "converged"
    np.testing.assert_array_equal(x0, np.zeros(2))
    np.testing.assert_array_equal(result["calls"][0]["output"], [-1., -2.])
    for call in result["calls"]:
        if call["kind"] == "jacobian":
            np.testing.assert_array_equal(call["x"], [0., 0.])
            np.testing.assert_array_equal(call["output"], call["direction"])
    output[:] = -1000
    np.testing.assert_allclose(result["x"], [1., 2.], atol=1e-14)
    assert_counts(result)


def test_observer_gets_detached_strict_json_and_cannot_change_state():
    events = []

    def event(value):
        json.dumps(value, allow_nan=False)
        events.append(value)
        value["x"]["values"][:] = [500.]
        value["counts"]["total_calls"] = 999

    result = affine(np.eye(1), [1.], event=event)
    assert result["status"] == "converged"
    np.testing.assert_array_equal(result["x"], [1.])
    assert result["counts"]["total_calls"] == 4
    assert len(events) == 2


def test_observer_failure_propagates_and_attempts_end_event():
    events = []

    def event(value):
        events.append(value)
        if value["type"] == "iteration_begin":
            raise OSError("synthetic evidence sink unavailable")

    with pytest.raises(OSError, match="evidence sink"):
        affine(np.eye(1), [1.], event=event)
    assert [e["type"] for e in events] == ["iteration_begin", "iteration_end"]
    assert events[-1]["status"] == "failed"
    assert events[-1]["counts"]["jacobian_calls"] == 0


def test_fourth_accepted_newton_trial_converges_before_budget_check():
    # Newton for x^2=2 from 1 has residual >1e-10 after three steps,
    # and below the threshold after exactly four.
    calls = []

    def residual(x):
        calls.append(x.copy())
        return x * x - 2

    result = correct(residual, lambda x, v: 2 * x * v, np.ones(1), config=cfg())
    assert result["status"] == "converged" and result["reason"] == "root_residual_tolerance"
    assert len(result["iterations"]) == 4
    assert result["iterates"][-2]["residual_norm"] > 1e-10
    assert result["residual_norm"] <= 1e-10
    assert result["counts"] == dict(residual_calls=5, jacobian_calls=8, total_calls=13)
    # Each accepted trial supplies the next base without another residual call.
    assert len(calls) == len(result["iterates"])
    for previous, following in zip(result["iterations"], result["iterations"][1:]):
        np.testing.assert_array_equal(previous["residual_after"], following["residual"])
        np.testing.assert_array_equal(previous["x_after"], following["x"])
    assert_counts(result)


def test_four_accepted_steps_without_root_remain_unresolved():
    result = correct(lambda x: x * x - 2, lambda x, v: 2 * x * v,
                     np.ones(1), config=cfg(root_tolerance=1e-15))
    assert result["status"] == "unresolved" and result["reason"] == "newton_iteration_budget"
    assert len(result["iterations"]) == 4 and len(result["iterates"]) == 5
    assert result["residual_norm"] > 1e-15
    assert all(t["status"] in {"accepted", "unresolved"} for t in result["iterations"])
    assert_counts(result)


def test_full_callback_envelope_is_thirteen_residuals_and_thirty_six_actions():
    # Scripted residuals exercise accounting, not derivative qualification:
    # every third trial decreases and each eight-dimensional solve needs all
    # eight actions at this tolerance, followed by its mandatory fresh action.
    matrix = np.diag(np.arange(1., 9.))
    residual_calls = 0

    def residual(x):
        nonlocal residual_calls
        count = residual_calls
        residual_calls += 1
        if count == 0:
            return np.ones(8)
        iteration, trial = divmod(count - 1, 3)
        factor = .5 ** iteration
        return factor * (2 if trial < 2 else .5) * np.ones(8)

    result = correct(residual, lambda x, v: matrix @ v, np.zeros(8),
                     config=cfg(linear_relative_tolerance=1e-12))
    assert result["status"] == "unresolved" and result["reason"] == "newton_iteration_budget"
    assert result["counts"] == dict(residual_calls=13, jacobian_calls=36, total_calls=49)
    assert all(row["linear"]["arnoldi_actions"] == 8 for row in result["iterations"])
    assert all([t["accepted"] for t in row["trials"]] == [False, False, True]
               for row in result["iterations"])
    assert_counts(result)


def test_nine_choice_damping_can_accept_one_sixteenth_after_four_rejections():
    damping = tuple(2.**-k for k in range(9))
    result = correct(lambda x:x*x - 1., lambda x,v:2*x*v, np.array([.025]),
                     config=cfg(damping=damping, max_residual_calls=37, max_newton_iterations=1))
    trials = result['iterations'][0]['trials']
    assert [row['damping'] for row in trials] == list(damping[:5])
    assert [row['accepted'] for row in trials] == [False]*4 + [True]
    assert result['reason'] == 'newton_iteration_budget'
    assert result['counts']['residual_calls'] == 6
    np.testing.assert_array_equal(result['residual'], trials[-1]['residual'])
    assert_counts(result)


def test_nine_choice_line_search_failure_keeps_every_evaluated_rejection():
    damping = tuple(2.**-k for k in range(9))
    result = correct(lambda x:np.ones_like(x), lambda x,v:v, np.zeros(1),
                     config=cfg(damping=damping, max_residual_calls=37))
    trials = result['iterations'][0]['trials']
    assert result['reason'] == 'line_search_exhausted'
    assert [row['damping'] for row in trials] == list(damping)
    assert all(row['called'] and not row['accepted'] for row in trials)
    assert result['counts'] == dict(residual_calls=10, jacobian_calls=2, total_calls=12)
    assert_counts(result)


def test_four_iterations_nine_damping_full_dimension_budget_and_serialization(tmp_path):
    # Callback-accounting fixture only: scripted nonlinear values do not claim
    # a genuine derivative. Every ninth trial is accepted and every solve uses
    # eight actions plus a fresh action, with all 896 components retained.
    size = 896
    diagonal = np.tile(np.arange(1., 9.), size//8)
    count = 0
    def residual(x):
        nonlocal count
        current = count; count += 1
        if current == 0:
            return np.ones(size)
        iteration, trial = divmod(current - 1, 9)
        return .5**iteration * (2. if trial < 8 else .5) * np.ones(size)
    config = cfg(damping=tuple(2.**-k for k in range(9)), max_residual_calls=37,
                 max_jacobian_calls=36, linear_relative_tolerance=1e-12)
    result = correct(residual, lambda x,v:diagonal*v, np.zeros(size), config=config,
                     trust_center=np.full(size, .01))
    assert result['status'] == 'unresolved' and result['reason'] == 'newton_iteration_budget'
    assert result['counts'] == dict(residual_calls=37, jacobian_calls=36, total_calls=73)
    assert len(result['iterations']) == 4
    assert all(row['linear']['arnoldi_actions'] == 8 for row in result['iterations'])
    assert all([t['accepted'] for t in row['trials']] == [False]*8 + [True] for row in result['iterations'])
    for previous, following in zip(result['iterations'], result['iterations'][1:]):
        np.testing.assert_array_equal(previous['residual_after'], following['residual'])
    encoded = json.dumps(serializable(result), indent=2, allow_nan=False) + '\n'
    raw = encoded.encode('utf-8')
    (tmp_path/'max-nine-damping-evidence.json').write_bytes(raw)
    (tmp_path/'serialization-size.json').write_text(json.dumps(dict(bytes=len(raw),
        counts=result['counts'], dimension=size, iterations=4, damping_count=9,
        scope='Measured scripted fixture, not a universal upper bound.'))+'\n')
    decoded = json.loads(encoded)
    assert decoded['trust_center']['sha256'] == result['trust_center_sha256']
    assert len(decoded['calls']) == 73 and len(decoded['iterations'][-1]['trials']) == 9
    assert_counts(result)


@pytest.mark.parametrize("parameter,seed,expectation", [
    (1., 1.2, "converged"),
    (0., 0., "converged"),
    (-1., 0., "unresolved"),
])
def test_saddle_node_examples_do_not_infer_absence_from_unresolved(parameter, seed, expectation):
    result = correct(lambda x: x * x - parameter, lambda x, v: 2 * x * v,
                     np.array([seed]), config=cfg())
    assert result["status"] == expectation
    if expectation == "converged":
        assert np.linalg.norm(result["x"] ** 2 - parameter) <= 1e-10
    else:
        assert result["reason"] == "gmres_singular_projected_system"
        assert result["residual_norm"] == 1
    assert "no_root_exists" not in result
    assert_counts(result)


def test_unstable_distinct_root_in_bistable_map_is_valid_numerical_root():
    # F(x)=x+0.1(x-x^3) has attracting fixed points at +/-1 and an unstable
    # fixed point at zero. Newton from 0.1 reaches zero although iteration of
    # F itself moves toward the positive attracting fixed point.
    def residual(x):
        return .1 * (x - x ** 3)

    result = correct(residual, lambda x, v: .1 * (1 - 3 * x * x) * v,
                     np.array([.1]), config=cfg())
    assert result["status"] == "converged"
    assert abs(result["x"][0]) < 1e-8
    forward = np.array([.1])
    for _ in range(240):
        forward += residual(forward)
    assert abs(forward[0] - 1) < 1e-10
    assert abs(forward[0] - result["x"][0]) > .9
    assert "stable" not in result
    assert_counts(result)


def test_insufficient_eight_vector_coverage_is_a_finite_unresolved_result():
    # Well-separated eigenvalues make an eight-direction residual visibly
    # insufficient under the stated tolerance. There is still a unique root.
    matrix = np.diag(np.geomspace(1., 1e6, 20))
    result = correct(lambda x: matrix @ x - np.ones(20), lambda x, v: matrix @ v,
                     np.zeros(20), config=cfg(linear_relative_tolerance=1e-6))
    assert result["status"] == "unresolved"
    assert result["reason"] == "fresh_linear_residual_insufficient"
    linear = result["iterations"][0]["linear"]
    assert linear["arnoldi_actions"] == 8
    assert linear["fresh_action"]["full_relative"] > 1e-6
    assert linear["fresh_action"]["consistency_relative"] <= 1e-10
    assert result["counts"]["jacobian_calls"] == 9
    assert_counts(result)


def test_tiny_residual_below_root_threshold_avoids_rhs_normalization():
    result = correct(lambda x: np.array([1e-300]),
                     lambda x, v: pytest.fail("already converged"), np.zeros(1), config=cfg())
    assert result["status"] == "converged"
    assert result["residual_norm"] == 1e-300
    assert result["counts"]["jacobian_calls"] == 0


def test_fresh_action_exception_retains_the_attempted_correction():
    calls = 0

    def action(x, v):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fresh check failed part way through")
        return v

    result = correct(lambda x: x - 1, action, np.zeros(1), config=cfg())
    assert result["status"] == "failed"
    assert result["counts"]["jacobian_calls"] == 2
    call = result["calls"][-1]
    assert call["purpose"] == "fresh_linear_residual" and call["status"] == "failed"
    np.testing.assert_array_equal(call["direction"], result["iterations"][0]["linear"]["correction"])
    assert_counts(result)


def test_finite_callback_with_overflowing_norm_fails_closed():
    with np.errstate(over="ignore", invalid="ignore"):
        result = correct(lambda x: np.full(2, np.finfo(np.float64).max),
                         lambda x, v: v, np.zeros(2), config=cfg())
    assert result["status"] == "failed" and result["reason"] == "residual_callback_failure"
    assert np.isfinite(result["calls"][0]["output"]).all()
    assert result["counts"]["residual_calls"] == 1
    assert_counts(result)


def test_overflowing_projected_solve_retains_raw_system_and_never_calls_trial():
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = correct(lambda x: x - 1, lambda x, v: 1e-320 * v,
                         np.zeros(1), config=cfg())
    assert result["status"] == "failed" and result["reason"] == "nonfinite_solver_arithmetic"
    linear = result["iterations"][0]["linear"]
    assert linear["arnoldi_actions"] == 1
    np.testing.assert_array_equal(linear["raw_actions"], [[1e-320]])
    assert not np.isfinite(linear["steps"][0]["coefficients"]).all()
    assert not result["iterations"][0]["trials"]
    assert result["counts"]["residual_calls"] == 1
    assert_counts(result)


@pytest.mark.parametrize("changes", [
    {"trust_radius": 0}, {"trust_radius": np.inf}, {"root_tolerance": 0},
    {"max_newton_iterations": -1}, {"max_gmres_actions": 1.5}, {"max_residual_calls": True},
    {"damping": ()}, {"damping": (1., 1.)}, {"damping": (1., 0.)},
    {"sufficient_decrease": 1}, {"breakdown_tolerance": -1},
])
def test_invalid_config_is_rejected(changes):
    with pytest.raises(ValueError):
        cfg(**changes)


@pytest.mark.parametrize("seed", [np.ones(1, dtype=np.float32), np.array([np.nan]), np.ones((1, 2)), np.zeros(0)])
def test_invalid_seed_never_calls_callback(seed):
    with pytest.raises(ValueError):
        correct(lambda x: pytest.fail("invalid input callback"),
                lambda x, v: pytest.fail("invalid input callback"), seed, config=cfg())
