"""Step 4 - Bayesian causal model for the TATE curve ``psi_d(t)``.

* :class:`FunctionalGPEstimator` - finite-rank functional GP over ``(X, t)``.
* :func:`nested_posterior_tate` - two-stage topological-to-causal propagation.
* :class:`TSBCFAdapter` - optional R bridge for targeted smooth BCF.
"""
from __future__ import annotations

from .fgp import (
    CausalEffectPosterior,
    FunctionalGPEstimator,
    bayesian_no_effect_test,
    summarize_causal_effect,
)
from .propagation import (
    PropagationComparison,
    compare_propagation,
    nested_posterior_tate,
    plugin_posterior_tate,
)
from .tsbcf import TSBCFAdapter, TSBCFLongData, make_tsbcf_long_data

__all__ = [
    "CausalEffectPosterior",
    "FunctionalGPEstimator",
    "bayesian_no_effect_test",
    "summarize_causal_effect",
    "PropagationComparison",
    "nested_posterior_tate",
    "plugin_posterior_tate",
    "compare_propagation",
    "TSBCFAdapter",
    "TSBCFLongData",
    "make_tsbcf_long_data",
]
