"""``Band`` -- a standardised functional band over a grid ``tseq``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Band:
    """A simultaneous functional band ``[lower(t), upper(t)]`` over ``tseq``.

    Attributes:
        tseq:   grid ``[resolution]``.
        lower:  lower boundary ``[resolution]``.
        upper:  upper boundary ``[resolution]``.
        center: point estimate ``[resolution]`` (optional).
        level:  target coverage 1 - alpha (optional, for bookkeeping).
        kind:   ``"confidence"`` or ``"prediction"`` (the estimand/tool pairing).
    """

    tseq: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    center: Optional[np.ndarray] = None
    level: Optional[float] = None
    kind: Optional[str] = None

    def __post_init__(self):
        self.tseq = np.asarray(self.tseq, dtype=float)
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        if self.center is not None:
            self.center = np.asarray(self.center, dtype=float)

    @property
    def width(self):
        """Pointwise width ``upper - lower`` (array ``[resolution]``)."""
        return self.upper - self.lower

    def mean_width(self):
        from .coverage import mean_width

        return mean_width(self.lower, self.upper)

    def covers(self, target):
        """Simultaneous coverage of ``target`` (``[res]`` -> bool, ``[n,res]`` -> ``[n]``)."""
        from .coverage import covers_simultaneous

        return covers_simultaneous(self.lower, self.upper, target)
