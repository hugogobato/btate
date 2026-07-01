"""Pointwise-in-x confidence bands for the CTATE tau_d(., x) (Phase 3.3).

The *simplest defensible* confidence UQ for the conditional-mean silhouette
effect: a band that is **simultaneous over t** at a **fixed x** (pointwise in x).
Honest *uniform* inference over ``(t, x)`` -- simultaneous bands over the whole
covariate space at nonparametric rate in ``x`` -- is deliberately deferred as a
topology-agnostic spin-off (``RESEARCH_PLAN.md`` Phase 3, docs/phase3_ctate.md).

Construction (reuses the Phase-2 multiplier bootstrap "conditionally"). The
functional DR-learner is a linear smoother of the pseudo-outcome curves,

    tau_hat_d(., x) = sum_i a_i(x) * P psi_{i,d},

so at a fixed x its sampling fluctuation is the linear functional
``sum_i a_i(x) (psi_{i,d} - tau_d(., X_i))`` of the (conditionally) mean-zero
pseudo-outcome residuals. We estimate the residuals by the second-stage
residual curves ``r_{i,d}`` and calibrate a simultaneous-in-t band with a wild
(multiplier) bootstrap of

    G_d(t) = sum_i a_i(x) xi_i r_{i,d}(t),    xi_i iid mean 0 var 1,

a heteroskedasticity-robust (HC0) construction. Let ``s(t) = sqrt(sum_i
a_i(x)^2 r_{i,d}(t)^2)`` be its pointwise sd and ``c_{1-alpha}`` the conditional
``1-alpha`` quantile of ``sup_t |G_d(t)| / s(t)`` (studentized; the default) or
of ``sup_t |G_d(t)|`` (unstudentized, constant width). The band is

    tau_hat_d(., x)  +/-  c_{1-alpha} * s(t)   (studentized)   or   +/- c_{1-alpha}.

Because ``a_i`` already carries the ``1/n`` smoother scale, the half-width
vanishes at the ``1/sqrt(n)`` rate: this is a **confidence** band for the mean
curve tau_d(., x), not a prediction band for an individual (that is Phase 5).
In the marginal case (intercept-only design, ``a_i = 1/n``, ``r_i`` the centered
EIF) it reduces to the Phase-2 TATE multiplier band.

Documented approximations (kept honest): the band treats the linear-in-features
second stage as the working conditional-mean model and uses the HC0 residual
variance; it therefore ignores (i) second-stage smoothing bias if tau_d(., x) is
not linear in the chosen features and (ii) the smaller-order stage-1 nuisance-
estimation error. Under a well-specified second stage (e.g. the tri-oracle,
whose tau is linear in x) these vanish and coverage is ~nominal.
"""

from __future__ import annotations

import numpy as np

from ...metrics import Band
from .multiplier_bootstrap import _draw_multipliers


def ctate_confidence_band(
    learner,
    x,
    d: int = 0,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    multiplier: str = "rademacher",
    standardize: bool = True,
    rng=None,
) -> Band:
    """Simultaneous-in-t, pointwise-in-x confidence band for tau_d(., x).

    Args:
        learner: a fitted :class:`~tcda_uq.estimators.CTATEDRLearner`.
        x: the covariate value at which to band tau_d(., x) (1-D, length ``d_cov``).
        d: homology dimension.
        alpha: 1 - target simultaneous-in-t coverage.
        n_boot: number of bootstrap draws.
        multiplier: ``"rademacher"`` (default), ``"gaussian"`` or ``"mammen"``.
        standardize: studentize by the pointwise sd (variance-adaptive width).
            ``False`` gives the unstudentized constant-width band.
        rng: seed or ``np.random.Generator``.

    Returns:
        :class:`~tcda_uq.metrics.Band` with ``kind="confidence"``, centered at
        ``tau_hat_d(., x)``.
    """
    rng = np.random.default_rng(rng)
    a = learner.weights(x, d=d)                 # [n]  smoother weights
    R = learner.residuals(d)                    # [n, res]  second-stage residuals
    center = learner.predict_dim(np.atleast_2d(x), d)[0]   # [res]

    contrib = a[:, None] * R                    # [n, res]  a_i r_i(t)
    var = np.einsum("it,it->t", contrib, contrib)          # sum_i (a_i r_i(t))^2
    sd = np.sqrt(np.maximum(var, 1e-24))
    scale = sd if standardize else np.ones_like(sd)

    xi = _draw_multipliers(multiplier, (n_boot, a.shape[0]), rng)   # [B, n]
    boot = (xi @ contrib) / scale               # [B, res]
    sup = np.max(np.abs(boot), axis=1)          # [B]
    c = float(np.quantile(sup, 1.0 - alpha))

    half = c * scale
    return Band(
        tseq=learner.tseq,
        lower=center - half,
        upper=center + half,
        center=center,
        level=1.0 - alpha,
        kind="confidence",
    )


def ctate_confidence_bands(learner, x, *, alpha: float = 0.05, **kwargs):
    """Convenience: one CTATE confidence band per homology dim at a fixed ``x``.

    Extra keyword args are forwarded to :func:`ctate_confidence_band`.
    """
    return [
        ctate_confidence_band(learner, x, d, alpha=alpha, **kwargs)
        for d in range(learner.n_hom_dim)
    ]


def ctate_pointwise_sd(learner, x, d: int = 0):
    """Analytic HC0 pointwise sd of ``tau_hat_d(., x)``: ``sqrt(sum_i a_i^2 r_i(t)^2)``.

    The studentizing scale of :func:`ctate_confidence_band`; also usable for a
    Gaussian pointwise (non-simultaneous) interval ``tau_hat +/- z_{1-alpha/2} sd``.
    """
    a = learner.weights(x, d=d)
    R = learner.residuals(d)
    contrib = a[:, None] * R
    return np.sqrt(np.maximum(np.einsum("it,it->t", contrib, contrib), 1e-24))
