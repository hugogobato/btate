"""Step 1 — Bayesian posterior over persistence diagrams (Maroulas et al. 2020).

Public, dependency-light members (diagram-format adapters) are re-exported here.
The sampler and prior/clutter elicitation utilities live in :mod:`.sampler` and
:mod:`.elicitation` and import ``bayes_tda`` lazily.
"""
from __future__ import annotations

from .adapters import bd_to_bp, bp_to_bd, lifetimes
from .sampler import PosteriorDiagramSampler
from .elicitation import elicit_prior_clutter, sensitivity_analysis

__all__ = [
    "bd_to_bp",
    "bp_to_bd",
    "lifetimes",
    "PosteriorDiagramSampler",
    "elicit_prior_clutter",
    "sensitivity_analysis",
]
