"""Small shared numerical helpers (ported from top-causal-effect utils)."""

from __future__ import annotations

import numpy as np


def numerical_integration(f, tseq):
    """Trapezoidal integral of samples ``f`` over grid ``tseq``.

    Works for ``f`` of shape ``[..., len(tseq)]``; integrates the last axis.
    """
    f = np.asarray(f)
    tseq = np.asarray(tseq)
    delta_t = tseq[1:] - tseq[:-1]
    f_right = f[..., 1:]
    f_left = f[..., :-1]
    return np.sum((f_right + f_left) / 2 * delta_t, axis=-1)
