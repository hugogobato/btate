"""Uncertainty quantification.

  * :mod:`.asymptotic` -- C1: confidence bands for the *mean* estimands (TATE, CTATE
    mean) via multiplier bootstrap / Liebl-Reimherr / Pini-Vantini (Phase 2/3).
  * :mod:`.conformal`  -- C2: finite-sample *prediction* bands for the individual
    effect (ITTE), adaptive/stabilised variants (Phases 4-6).

The organising distinction: expectations -> confidence (asymptotic, ~1/sqrt(n));
an individual draw -> prediction (conformal, non-vanishing aleatoric width).
"""
