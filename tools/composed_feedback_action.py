"""Compose recorded forward-mode actions without resetting state or tangent.

This is measurement plumbing, not an orbit-stability estimator. The supplied
FeedbackMap owns precision checks, raw recording and the hard call budget.
"""


def composed_action(loop, state, tangent=None, *, steps=6, label="composed",
                    cached=True):
    if type(steps) is not int or not 1 <= steps <= 6:
        raise ValueError("composition requires one through six updates")
    if not isinstance(label, str) or not label or not all(
            c.isalnum() or c in "_-" for c in label):
        raise ValueError("safe nonempty recorder label required")
    differentiated = tangent is not None
    for index in range(steps):
        state, tangent = loop.evaluate(
            f"{label}-{index}", state, tangent=tangent, cached=cached)
        if differentiated and tangent is None:
            raise ValueError("composed action lost its tangent")
    return state, tangent
