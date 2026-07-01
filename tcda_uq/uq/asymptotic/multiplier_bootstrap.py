"""Multiplier-bootstrap simultaneous confidence band for the TATE (Phase 2.1).

Implements the band underlying TATE Corollary 5.4 / Theorem 5.2. By Theorem 5.2,

    sqrt(n) { psi_hat_AIPW,d(t) - psi_d(t) }  ==>  G_d(t)   in l^inf(T),

a mean-zero Gaussian process with covariance kernel ``cov{phi_d(s), phi_d(t)}``.
A simultaneous 1-alpha band inverts the sup-statistic ``sup_t |sqrt(n)(psi_hat-psi)|``
(optionally studentized by the pointwise sd), whose law is estimated by the
Gaussian/Rademacher **multiplier bootstrap** of the estimated influence process:

    G_hat_n,d(t) = n^{-1/2} sum_i xi_i * phi_hat_{i,d}(t),   xi_i iid mean 0 var 1,

with ``phi_hat_{i,d}(t)`` the *centered* cross-fitted EIF (``cross_fit(...).influence()``).
Let ``c_{1-alpha}`` be the conditional (1-alpha) quantile of the (studentized)
sup-norm of ``G_hat_n,d``. The band is

    psi_hat_d(t)  +/-  c_{1-alpha} * s(t) / sqrt(n),

with ``s(t) = sigma_hat_d(t)`` (studentized, variance-adaptive width; the default)
or ``s(t) = 1`` (raw sup-statistic, literal Corollary 5.4). Both are
asymptotically valid; studentization gives shorter bands when the pointwise
variance varies over ``t`` and is the standard multiplier-bootstrap construction
(Chernozhukov-Chetverikov-Kato). The band is a **confidence** band: it covers the
mean curve ``psi_d`` and its half-width vanishes at the ``1/sqrt(n)`` rate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...metrics import Band
from .covariance import eif_pointwise_sd

_MULTIPLIERS = ("gaussian", "rademacher", "mammen")


def _draw_multipliers(kind, shape, rng):
    """iid mean-0 variance-1 multipliers ``xi`` of the requested family."""
    kind = kind.lower()
    if kind == "gaussian":
        return rng.standard_normal(shape)
    if kind == "rademacher":
        return rng.choice(np.array([-1.0, 1.0]), size=shape)
    if kind == "mammen":  # skewness-correcting two-point law (golden-ratio)
        s5 = np.sqrt(5.0)
        a, b = (1 - s5) / 2, (1 + s5) / 2
        p = (s5 + 1) / (2 * s5)  # P(xi = a)
        u = rng.random(shape)
        return np.where(u < p, a, b)
    raise ValueError(f"multiplier must be one of {_MULTIPLIERS}, got {kind!r}")


def multiplier_bootstrap_band(
    influence,
    tseq,
    estimate,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    multiplier: str = "rademacher",
    standardize: bool = True,
    variance=None,
    rng=None,
) -> Band:
    """Simultaneous 1-alpha multiplier-bootstrap band for one homology dim.

    Args:
        influence: centered EIF process ``[n, resolution]`` -- one element of
            ``cross_fit(...).influence()``. Its column mean should be ~0.
        tseq: silhouette grid ``[resolution]``.
        estimate: point estimate ``psi_hat_d(t)`` ``[resolution]`` (band center),
            i.e. ``cross_fit(...).aipw[d]``.
        alpha: 1 - target simultaneous coverage.
        n_boot: number of bootstrap draws.
        multiplier: ``"rademacher"`` (default), ``"gaussian"`` or ``"mammen"``.
        standardize: studentize by the pointwise sd (variance-adaptive width).
            ``False`` reproduces the literal Corollary 5.4 sup-statistic
            (constant width ``c/sqrt(n)``).
        variance: optional pointwise variance ``sigma_hat^2(t)`` ``[resolution]``;
            if ``None`` it is estimated from ``influence``. Supplying a shared
            variance (e.g. from :func:`~.covariance.eif_pointwise_variance`) keeps
            the studentization identical across bands in a comparison.
        rng: seed or ``np.random.Generator``.

    Returns:
        :class:`~tcda_uq.metrics.Band` with ``kind="confidence"``.
    """
    rng = np.random.default_rng(rng)
    phi = np.asarray(influence, dtype=float)
    phi = phi - phi.mean(axis=0, keepdims=True)
    n, res = phi.shape
    estimate = np.asarray(estimate, dtype=float)

    if standardize:
        if variance is None:
            sd = eif_pointwise_sd(phi)
        else:
            sd = np.sqrt(np.maximum(np.asarray(variance, dtype=float), 1e-12))
    else:
        sd = np.ones(res)

    # bootstrap sup-statistic:  sup_t | n^{-1/2} sum_i xi_i phi_i(t) | / s(t)
    xi = _draw_multipliers(multiplier, (n_boot, n), rng)      # [B, n]
    boot = (xi @ phi) / np.sqrt(n)                            # [B, res]
    boot /= sd                                                # studentize
    sup = np.max(np.abs(boot), axis=1)                        # [B]
    c = float(np.quantile(sup, 1.0 - alpha))

    half = c * sd / np.sqrt(n)
    return Band(
        tseq=tseq,
        lower=estimate - half,
        upper=estimate + half,
        center=estimate,
        level=1.0 - alpha,
        kind="confidence",
    )


def multiplier_bootstrap_bands(cross_fit_result, *, alpha: float = 0.05, **kwargs):
    """Convenience: one band per homology dim from a :class:`CrossFitResult`.

    Uses ``cross_fit_result.influence()`` and ``.aipw`` as the influence process
    and center. Extra keyword args are forwarded to :func:`multiplier_bootstrap_band`.
    """
    influence = cross_fit_result.influence()
    tseq = cross_fit_result.tseq
    return [
        multiplier_bootstrap_band(
            influence[d], tseq, cross_fit_result.aipw[d], alpha=alpha, **kwargs
        )
        for d in range(len(influence))
    ]


def topological_effect_test(
    influence,
    estimate,
    *,
    n_boot: int = 2000,
    multiplier: str = "rademacher",
    standardize: bool = False,
    variance=None,
    rng=None,
):
    """Corollary 5.4 test of ``H0: psi_d == 0`` (no topological effect).

    Statistic ``T_n = sqrt(n) * sup_t |psi_hat_d(t)| / s(t)`` compared to the
    multiplier-bootstrap null law of ``sup_t |G_hat_n,d(t)| / s(t)``. With
    ``standardize=False`` this is the literal Corollary 5.4 statistic
    ``T_n = sqrt(n) ||psi_hat_AIPW,d||_inf``.

    Returns:
        dict with ``statistic`` ``T_n``, bootstrap ``pvalue`` = mean(sup_boot >= T_n),
        and the critical values ``crit`` at the 90/95/99% levels.
    """
    rng = np.random.default_rng(rng)
    phi = np.asarray(influence, dtype=float)
    phi = phi - phi.mean(axis=0, keepdims=True)
    n, res = phi.shape
    estimate = np.asarray(estimate, dtype=float)

    if standardize:
        sd = (
            eif_pointwise_sd(phi)
            if variance is None
            else np.sqrt(np.maximum(np.asarray(variance, dtype=float), 1e-12))
        )
    else:
        sd = np.ones(res)

    T_n = float(np.max(np.abs(estimate) / sd) * np.sqrt(n))
    xi = _draw_multipliers(multiplier, (n_boot, n), rng)
    boot = (xi @ phi) / np.sqrt(n) / sd
    sup = np.max(np.abs(boot), axis=1)
    pvalue = float(np.mean(sup >= T_n))
    crit = {lvl: float(np.quantile(sup, lvl)) for lvl in (0.90, 0.95, 0.99)}
    return {"statistic": T_n, "pvalue": pvalue, "crit": crit}
