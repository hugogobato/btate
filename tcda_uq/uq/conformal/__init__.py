"""Conformal prediction bands for the individual effect (C2, Phases 4-6).

Planned modules:

  * ``functional_cp.py``     -- Layer 1: functional split-CP (sup-norm score + modulation).
  * ``weighted_cp.py``       -- Layer 2: propensity-weighted causal CP (binary treatment).
  * ``composition.py``       -- Layer 3: arm composition phi^1, phi^0 -> delta band.
  * ``adaptive_cp.py``       -- covariate-adaptive / conditional CP (Phase 5).
  * ``diagram_score.py``     -- diagram-space (Wasserstein) scores (Phase 6.2).
  * ``stabilized_weights.py``-- positivity-stabilised weighting (Phase 6.5, headline).

Each will expose a uniform ``*_band(...) -> tcda_uq.metrics.Band`` interface.
"""
