"""Offline evidence for bounded, real, fixed-reference spectral probes.

The caller supplies ``action(v) = M_x v`` with the *same* reference x on every
call. This module neither constructs nor advances that reference. Callbacks
must return finite real float64-compatible vectors and should record each call
before returning, including failures. No model, network or retry logic lives
here. Checks and bands are operational heuristics, not stability certificates.

All result dictionaries are flat and can be saved with ``save_evidence``.
``nested_ritz`` returns a list of such dictionaries. Complex arrays are saved
as complex arrays, never reduced to their real parts.
"""

from pathlib import Path

import numpy as np


ORTHOGONALITY_TOL = 1e-10
DECOMPOSITION_TOL = 1e-10
RESIDUAL_TOL = 1e-6
NEAR_UNIT_BAND = (0.98, 1.02)
CROSS_START_TOL = 0.005
REFERENCE_SENSITIVITY_TOL = 1e-5
NORM_FLOOR = 1e-14
SCOPE = (
    'Heuristic finite-subspace diagnostics at a fixed reference; '
    'small residuals and start agreement do not establish spectral coverage '
    'or full orbit stability.'
)


def _real_array(value, ndim, name):
    if np.iscomplexobj(value):
        raise ValueError(f'{name} must be real')
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite {ndim}-dimensional array')
    return array.copy()


def _norm(value):
    # Scale first so a finite small vector is not mistaken for the zero vector
    # by squaring underflow, and representable large norms do not overflow.
    scale = float(np.max(np.abs(value), initial=0.0))
    if scale == 0:
        return 0.0
    with np.errstate(over='ignore', invalid='ignore'):
        result = float(scale * np.linalg.norm(value / scale))
    if not np.isfinite(result):
        raise ValueError('nonfinite norm in spectral calculation')
    return result


def _complex_vector(value, name, *, nonzero=False):
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite nonempty vector')
    norm = _norm(array)
    if nonzero and norm == 0:
        raise ValueError(f'{name} must be nonzero')
    return array.copy()


def _complex_scalar(value, name):
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim != 0 or not np.isfinite(array):
        raise ValueError(f'{name} must be a finite scalar')
    return complex(array)


def _tolerance(value, name):
    if not np.isscalar(value) or not np.isreal(value):
        raise ValueError(f'{name} must be a finite nonnegative scalar')
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite nonnegative scalar')
    return value


def _real_action(action, vector):
    # A callback may reuse or mutate its working buffer. Neither may corrupt Q
    # or an already recorded output. The caller's actual component is unscaled.
    output = _real_array(action(vector.copy()), 1, 'action output')
    if output.shape != vector.shape:
        raise ValueError('action output shape differs from input')
    return output


def arnoldi(action, start, steps, *, breakdown_tol=1e-13):
    """Two-pass MGS with at most ``steps`` real fixed-reference actions.

    Returns q:(n,m+1), h:(m+1,m), matvecs:(n,m), where m=actions. Every
    ``matvecs[:,j]`` is the unmodified actual action on ``q[:,j]``. At happy
    breakdown q's terminal column is zero and the small discarded residual is
    included in the measured decomposition error. Orthogonality uses Q_m only.
    Errors are Frobenius norms; decomposition error is relative to
    max(||matvecs||_F, 1e-14). Absolute breakdown_tol is not a spectral bound.

    Validation/callback errors raise, without retries. The callback is
    responsible for retaining failed-call evidence in its own run record.
    """
    v = _real_array(start, 1, 'start')
    norm = _norm(v)
    if (v.size == 0 or norm == 0 or type(steps) is not int
            or not 1 <= steps <= v.size):
        raise ValueError('nonzero start and integer steps in [1, dimension] required')
    breakdown_tol = _tolerance(breakdown_tol, 'breakdown_tol')
    q = np.zeros((v.size, steps + 1), dtype=np.float64)
    h = np.zeros((steps + 1, steps), dtype=np.float64)
    outputs = np.zeros((v.size, steps), dtype=np.float64)
    q[:, 0] = v / norm
    happy = False
    for j in range(steps):
        outputs[:, j] = _real_action(action, q[:, j])
        w = outputs[:, j].copy()
        with np.errstate(over='raise', invalid='raise'):
            for _ in range(2):
                for i in range(j + 1):
                    coefficient = float(q[:, i] @ w)
                    h[i, j] += coefficient
                    w -= coefficient * q[:, i]
        if not np.isfinite(h[:, j]).all() or not np.isfinite(w).all():
            raise ValueError('nonfinite orthogonalization')
        h[j + 1, j] = _norm(w)
        used = j + 1
        # <= also makes exactly zero a happy breakdown when the tolerance is 0.
        if h[j + 1, j] <= breakdown_tol:
            happy = True
            break
        q[:, j + 1] = w / h[j + 1, j]
    q = q[:, :used + 1].copy()
    h = h[:used + 1, :used].copy()
    outputs = outputs[:, :used].copy()
    orthogonality = _norm(q[:, :used].T @ q[:, :used] - np.eye(used))
    decomposition_absolute = _norm(outputs - q @ h)
    decomposition = decomposition_absolute / max(_norm(outputs), NORM_FLOOR)
    return dict(
        q=q, h=h, matvecs=outputs, actions=used, steps_requested=steps,
        happy_breakdown=happy, breakdown_tol=breakdown_tol,
        orthogonality_error=orthogonality,
        decomposition_absolute_error=decomposition_absolute,
        decomposition_error=decomposition,
        checks_passed=bool(orthogonality <= ORTHOGONALITY_TOL
                           and decomposition <= DECOMPOSITION_TOL),
        scope=SCOPE,
    )


def _basis(evidence):
    q = _real_array(evidence['q'], 2, 'q')
    h = _real_array(evidence['h'], 2, 'h')
    outputs = _real_array(evidence['matvecs'], 2, 'matvecs')
    count = np.asarray(evidence['actions'])
    if count.ndim != 0 or count.dtype.kind not in 'iu':
        raise ValueError('actions must be an integer scalar')
    used = int(count)
    if (used < 1 or q.shape[0] < used
            or q.shape[1] != used + 1 or h.shape != (used + 1, used)
            or outputs.shape != (q.shape[0], used)):
        raise ValueError('inconsistent Arnoldi evidence shapes')
    return q, h, outputs, used


def ritz_at(evidence, dimension=None):
    """All Ritz pairs and actual-output residuals at a saved nested dimension.

    No action is called. Full vectors have unit norm. ``residual_norms`` are
    ||W_k*y - lambda*Q_k*y||, using the actual saved action outputs W_k, not
    only the final Hessenberg entry. They are not fresh validation residuals.
    A dimension beyond happy breakdown raises rather than fabricating a basis.
    """
    q, h, outputs, used = _basis(evidence)
    dimension = used if dimension is None else dimension
    if type(dimension) is not int or not 1 <= dimension <= used:
        raise ValueError('Ritz dimension must be within the actual saved basis')
    values, projected = np.linalg.eig(h[:dimension, :dimension])
    values = np.asarray(values, dtype=np.complex128)
    projected = np.asarray(projected, dtype=np.complex128)
    vectors = q[:, :dimension] @ projected
    for i in range(dimension):
        scale = _norm(vectors[:, i])
        if scale == 0:
            raise ValueError('zero full-space Ritz vector')
        projected[:, i] /= scale
        vectors[:, i] /= scale
    residuals = outputs[:, :dimension] @ projected - vectors * values
    norms = np.asarray([_norm(residuals[:, i]) for i in range(dimension)])
    if not all(np.isfinite(a).all() for a in (values, projected, vectors, residuals)):
        raise ValueError('nonfinite Ritz evidence')
    return dict(
        status='available', dimension=dimension, values=values,
        projected_vectors=projected, vectors=vectors,
        residual_vectors=residuals, residual_norms=norms, scope=SCOPE,
    )


def nested_ritz(evidence, dimensions=(4, 6, 8)):
    """Return available/unavailable records in requested order, with no calls."""
    _, _, _, used = _basis(evidence)
    dimensions = tuple(dimensions)
    if (any(type(k) is not int or k < 1 for k in dimensions)
            or len(set(dimensions)) != len(dimensions)):
        raise ValueError('unique positive integer nested dimensions required')
    return [ritz_at(evidence, k) if k <= used else dict(
        status='unavailable', dimension=k, actual_dimension=used,
        reason='requested dimension exceeds the actual saved basis', scope=SCOPE,
    ) for k in dimensions]


def leading_ritz_group(ritz, *, conjugate_tol=1e-10, tie_tol=1e-12):
    """Choose one real mode or a whole conjugate pair by descending modulus.

    The representative of a complex pair has positive imaginary part. Ties
    use descending real part, then imaginary part, then original index. All
    other groups within tie_tol*max(1, leading modulus) are explicitly marked
    ambiguous. No imaginary component is dropped merely for being small.
    """
    values = _complex_vector(ritz['values'], 'Ritz values')
    vectors = np.asarray(ritz['vectors'], dtype=np.complex128)
    if (vectors.ndim != 2 or vectors.shape[1] != values.size
            or vectors.shape[0] == 0 or not np.isfinite(vectors).all()):
        raise ValueError('invalid full-space Ritz vectors')
    conjugate_tol = _tolerance(conjugate_tol, 'conjugate_tol')
    tie_tol = _tolerance(tie_tol, 'tie_tol')
    remaining = set(range(values.size))
    groups = []
    while remaining:
        i = min(remaining)
        remaining.remove(i)
        if values[i].imag == 0:
            representative, indices = i, (i,)
        else:
            candidates = [j for j in remaining if values[j].imag != 0
                          and np.signbit(values[i].imag) != np.signbit(values[j].imag)]
            if not candidates:
                raise ValueError('unpaired complex Ritz value')
            partner = min(candidates, key=lambda j: (abs(values[j] - values[i].conjugate()), j))
            gap = abs(values[partner] - values[i].conjugate())
            if gap > conjugate_tol * max(1.0, abs(values[i]), abs(values[partner])):
                raise ValueError('complex Ritz values fail conjugate matching')
            remaining.remove(partner)
            representative = i if values[i].imag > 0 else partner
            indices = (representative, partner if representative == i else i)
        groups.append((representative, indices))
    groups.sort(key=lambda group: (
        -abs(values[group[0]]), -values[group[0]].real,
        -values[group[0]].imag, group[0],
    ))
    representative, indices = groups[0]
    value = complex(values[representative])
    vector = _complex_vector(vectors[:, representative], 'leading Ritz vector', nonzero=True)
    tied = [group[0] for group in groups[1:]
            if abs(abs(values[group[0]]) - abs(value)) <= tie_tol * max(1.0, abs(value))]
    return dict(
        indices=np.asarray(indices, dtype=np.int64),
        values=values[list(indices)].copy(), value=value, vector=vector,
        modulus=float(abs(value)),
        ambiguous_leading_modulus=bool(tied),
        tied_group_representatives=np.asarray(tied, dtype=np.int64),
        conjugate_tol=conjugate_tol, tie_tol=tie_tol, scope=SCOPE,
    )


def _complex_action(action, vector):
    z = _complex_vector(vector, 'complex direction', nonzero=True)
    real_input, imag_input = z.real.copy(), z.imag.copy()
    # Only *exactly* zero components are skipped. Small nonzero components
    # retain their original scale and consume their reserved call.
    real_called = bool(np.any(real_input != 0))
    imag_called = bool(np.any(imag_input != 0))
    real_output = _real_action(action, real_input) if real_called else np.zeros(z.size)
    imag_output = _real_action(action, imag_input) if imag_called else np.zeros(z.size)
    combined = real_output.astype(np.complex128) + 1j * imag_output
    return dict(
        vector=z, real_input=real_input, imag_input=imag_input,
        real_output=real_output, imag_output=imag_output,
        real_called=real_called, imag_called=imag_called,
        combined_action=combined, actions=int(real_called) + int(imag_called),
    )


def fresh_ritz_residual(action, value, vector):
    """Fresh full-space residual using at most two real, unscaled components.

    Relative error is ||M*z-lambda*z|| /
    max(||M*z||, |lambda|*||z||, 1e-14). A zero action and zero multiplier are
    valid; a zero direction is not. Exactly zero components leave slots unused.
    """
    value = _complex_scalar(value, 'Ritz value')
    result = _complex_action(action, vector)
    z, output = result['vector'], result['combined_action']
    residual = output - value * z
    denominator = max(_norm(output), abs(value) * _norm(z), NORM_FLOOR)
    absolute_error = _norm(residual)
    relative_error = absolute_error / denominator
    if not np.isfinite(denominator) or not np.isfinite(relative_error):
        raise ValueError('nonfinite fresh residual')
    result.update(
        value=value, residual_vector=residual, absolute_error=absolute_error,
        denominator=denominator, relative_error=relative_error,
        threshold=RESIDUAL_TOL, passed=bool(relative_error <= RESIDUAL_TOL),
        scope=SCOPE,
    )
    return result


def reference_action_sensitivity(action_at_reference, vector, baseline_action):
    """Compare a new-reference action with a saved action on the identical z.

    At most two real calls are made. Neither vector component is renormalized.
    Relative error uses max(||new action||, ||baseline action||, 1e-14). The
    baseline must come from ``fresh_ritz_residual`` on this exact vector; the
    caller must bind that identity in its evidence. No eigenvalue bound follows.
    """
    z = _complex_vector(vector, 'complex direction', nonzero=True)
    baseline = _complex_vector(baseline_action, 'baseline action')
    if baseline.shape != z.shape:
        raise ValueError('baseline action shape differs from direction')
    result = _complex_action(action_at_reference, z)
    difference = result['combined_action'] - baseline
    absolute_error = _norm(difference)
    denominator = max(_norm(result['combined_action']), _norm(baseline), NORM_FLOOR)
    relative_error = absolute_error / denominator
    result.update(
        baseline_action=baseline, difference=difference,
        absolute_error=absolute_error, denominator=denominator,
        relative_error=relative_error, threshold=REFERENCE_SENSITIVITY_TOL,
        passed=bool(relative_error <= REFERENCE_SENSITIVITY_TOL), scope=SCOPE,
    )
    return result


def modulus_region(value):
    """Return below_band, near_unit or above_band; never a stability verdict."""
    modulus = abs(_complex_scalar(value, 'Ritz value'))
    if modulus < NEAR_UNIT_BAND[0]:
        return 'below_band'
    if modulus > NEAR_UNIT_BAND[1]:
        return 'above_band'
    return 'near_unit'


def compare_leading_groups(first, second):
    """Heuristic complex agreement, allowing conjugate orientation only.

    Equal moduli with different complex values need not agree. A tie/cluster
    ambiguity prevents ``passed`` even when the selected values are close.
    """
    a = _complex_scalar(first['value'], 'first Ritz value')
    b = _complex_scalar(second['value'], 'second Ritz value')
    discrepancy = float(min(abs(a - b), abs(a - b.conjugate())))
    if not np.isfinite(discrepancy):
        raise ValueError('nonfinite complex eigenvalue discrepancy')
    ambiguous = bool(first['ambiguous_leading_modulus'] or second['ambiguous_leading_modulus'])
    within = discrepancy <= CROSS_START_TOL
    return dict(
        first_value=a, second_value=b, discrepancy=discrepancy,
        threshold=CROSS_START_TOL, within_tolerance=bool(within),
        ambiguous=ambiguous, passed=bool(within and not ambiguous), scope=SCOPE,
    )


def save_evidence(path, evidence):
    """Create an NPZ containing every field of one flat result, without pickle.

    Never overwrite. Save nested-dimension records individually. Numeric,
    complex, boolean and string fields round-trip with allow_pickle=False.
    """
    arrays = {}
    for key, value in evidence.items():
        if not isinstance(key, str) or key == 'file':
            raise ValueError('evidence keys must be strings other than file')
        array = np.asarray(value)
        if array.dtype.kind not in 'biufcUS':
            raise ValueError(f'evidence field {key} is not a flat numeric/string value')
        arrays[key] = array
    with Path(path).open('xb') as handle:
        np.savez(handle, **arrays)
