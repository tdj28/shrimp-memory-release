"""Bounded-memory implementation of the fixed readout proposal; not a stop command."""
from collections import deque
import numpy as np
import audit_p6_recovery as gates


class StreamingReadout:
    """Consume post-update states only; retain the first three-checkpoint decision.

    This observer never stops its caller. Continued observation permits detecting
    later contradictions. The initial pre-update state must not be pushed.
    """
    def __init__(self, references):
        refs=np.asarray(references)
        if refs.shape!=(2,6,896) or refs.dtype!=np.float64 or not np.isfinite(refs).all():
            raise ValueError('require finite float64 references')
        self.references=refs.copy()
        self.buffer=deque(maxlen=12)
        self.steps=0
        self.streak=0
        self.previous=None
        self.decision=None

    def push(self, state):
        x=np.asarray(state)
        if x.shape!=(896,) or x.dtype!=np.float64 or not np.isfinite(x).all():
            raise ValueError('require finite float64 state')
        self.buffer.append(x.copy())
        self.steps+=1
        if self.steps%12:return None
        outcome=gates.recompute(np.stack(self.buffer),self.references,0)
        matches=outcome['reference_matches']
        label=matches[0] if outcome['cycle_gate_passed'] and len(matches)==1 else None
        self.streak=self.streak+1 if label is not None and label==self.previous else (1 if label else 0)
        self.previous=label
        if self.streak==3 and self.decision is None:
            self.decision={'steps':self.steps,'label':label}
        return {'steps':self.steps,'qualified_label':label,
                'maximum_closure_residual':outcome['maximum_l2_residual'],
                'reference_matches':matches}
