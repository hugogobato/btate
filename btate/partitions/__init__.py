"""Step 2 — Bayesian random-partition signal/noise separation (Martínez 2024).

Exposes the per-feature signal probability API ``signal_probability`` and the
log-normal + restricted-random-partition model. The MCMC sampler is implemented
in :mod:`.lifetime_model` (Phase 1, Task 1.3).
"""
from __future__ import annotations

from .lifetime_model import (
    PartitionPosterior,
    RestrictedPartitionModel,
    signal_probability,
)
from .diagnostics import effective_sample_size, potential_scale_reduction

__all__ = [
    "RestrictedPartitionModel",
    "PartitionPosterior",
    "signal_probability",
    "effective_sample_size",
    "potential_scale_reduction",
]
