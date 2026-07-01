"""Shared UQ metrics + band container (Phase 0.7).

Used by every UQ phase: ``Band`` standardises a functional band; the coverage
helpers evaluate both *simultaneous* (all-t) coverage -- the right notion for
functional confidence/prediction bands -- and *pointwise* / *interval-wise*
coverage (for Pini-Vantini-style interval-wise control).
"""

from .bands import Band
from .coverage import (
    covers_simultaneous,
    simultaneous_coverage,
    pointwise_coverage,
    mean_width,
    interval_wise_error,
)
from .plotting import plot_band

__all__ = [
    "Band",
    "covers_simultaneous",
    "simultaneous_coverage",
    "pointwise_coverage",
    "mean_width",
    "interval_wise_error",
    "plot_band",
]
