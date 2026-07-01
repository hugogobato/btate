"""Pini-Vantini interval-wise testing (IWT) band for the TATE (Phase 2.3).

Pini & Vantini (2017), *Interval-wise testing for functional data*, build two
p-value functions from a family of interval statistics and control error at two
resolutions:

  * the **unadjusted** ``p(t) = limsup_{I -> t} p_I`` controls the *point-wise*
    error rate (their eq. 6);
  * the **adjusted** ``p_tilde(t) = sup_{I ∋ t} p_I`` controls the *interval-wise*
    error rate (their eq. 7): for every interval ``I`` on which the null holds,
    ``P[for all t in I, p_tilde(t) <= alpha] <= alpha``.

Here the null is ``H0: psi_d = ref`` (default ``ref = 0`` -- the "no topological
effect" null that mirrors Corollary 5.4, but now *localised* to intervals of the
filtration axis). We replace PV's permutation test (they assume two i.i.d.
samples) by the **parametric bootstrap of the Gaussian limit** that the TATE
paper attributes to Pini & Vantini: by Theorem 5.2 ``sqrt(n)(psi_hat - psi)``
converges to a mean-zero Gaussian process whose law is resampled by the
multiplier bootstrap of the cross-fitted EIF. The interval statistic is the
``|I|``-normalised studentised L^2 statistic (their eq. 3),

    T_I = (1/|I|) * sum_{s in I} [ sqrt(n) (psi_hat(s) - ref(s)) / sigma_hat(s) ]^2 ,

whose ``|I|`` normalisation is exactly what keeps long intervals from dominating
(so the adjusted band lies *between* the pointwise and the simultaneous band,
rather than collapsing onto the simultaneous one).

Two entry points:
  * :func:`iwt_pvalues`    -- the faithful IWT object: unadjusted + adjusted
    p-value functions (and the alpha-thresholded significant regions).
  * :func:`pini_vantini_band` -- the induced interval-wise confidence band
    ``psi_hat(t) +/- (sigma_hat(t)/sqrt(n)) * k_tilde(t)``, whose guarantee is the
    interval-wise error control above (documented, and validated in Phase 2.6);
    it is *not* a simultaneous band -- that is :mod:`.multiplier_bootstrap`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...metrics import Band
from .covariance import eif_pointwise_sd
from .multiplier_bootstrap import _draw_multipliers


def _interval_means(square_cumsum):
    """Given prefix sums ``P`` of a per-point array over ``[..., res]``, return the
    per-interval means ``mean_{s in [j,k]} x[s]`` as a dict-free upper-triangular
    representation.

    ``square_cumsum`` has shape ``[..., res + 1]`` (with a leading zero). Returns
    ``means`` of shape ``[..., n_intervals]`` and index arrays ``j, k`` (0-based,
    inclusive) of length ``n_intervals``, ordered by ``(j, k)``.
    """
    res = square_cumsum.shape[-1] - 1
    j_idx, k_idx = np.triu_indices(res)               # all j <= k
    length = (k_idx - j_idx + 1).astype(float)        # |I| in grid points
    total = square_cumsum[..., k_idx + 1] - square_cumsum[..., j_idx]
    return total / length, j_idx, k_idx


@dataclass
class IWTResult:
    """Output of :func:`iwt_pvalues`."""

    tseq: np.ndarray
    p_unadjusted: np.ndarray   # [res]  point-wise error control (eq. 6)
    p_adjusted: np.ndarray     # [res]  interval-wise error control (eq. 7)
    alpha: float

    def significant(self, adjusted: bool = True):
        """Boolean mask of grid points where the null is rejected at ``alpha``."""
        p = self.p_adjusted if adjusted else self.p_unadjusted
        return p <= self.alpha


def iwt_pvalues(
    influence,
    tseq,
    estimate,
    *,
    ref=0.0,
    alpha: float = 0.05,
    n_boot: int = 2000,
    multiplier: str = "rademacher",
    variance=None,
    rng=None,
) -> IWTResult:
    """Interval-wise unadjusted + adjusted p-value functions for ``H0: psi_d = ref``.

    Args:
        influence: centered EIF process ``[n, res]`` (``cross_fit(...).influence()[d]``).
        tseq: grid ``[res]``.
        estimate: ``psi_hat_d(t)`` ``[res]``.
        ref: null curve (scalar broadcast or ``[res]``); ``0`` = no-effect null.
        alpha: error level for the significance masks.
        n_boot, multiplier, variance, rng: parametric-bootstrap controls
            (shared with the multiplier bootstrap).

    Returns:
        :class:`IWTResult`.
    """
    rng = np.random.default_rng(rng)
    phi = np.asarray(influence, dtype=float)
    phi = phi - phi.mean(axis=0, keepdims=True)
    n, res = phi.shape
    estimate = np.asarray(estimate, dtype=float)
    ref = np.broadcast_to(np.asarray(ref, dtype=float), (res,))

    sd = (
        eif_pointwise_sd(phi)
        if variance is None
        else np.sqrt(np.maximum(np.asarray(variance, dtype=float), 1e-12))
    )

    # observed studentised deviation and its square
    d2 = (np.sqrt(n) * (estimate - ref) / sd) ** 2                 # [res]
    # bootstrap studentised process under H0
    xi = _draw_multipliers(multiplier, (n_boot, n), rng)          # [B, n]
    W = (xi @ phi) / np.sqrt(n) / sd                              # [B, res]
    W2 = W ** 2

    # interval means of the squared statistics via prefix sums
    d2_cum = np.concatenate([[0.0], np.cumsum(d2)])              # [res+1]
    W2_cum = np.concatenate(
        [np.zeros((n_boot, 1)), np.cumsum(W2, axis=1)], axis=1
    )                                                            # [B, res+1]
    T_obs, j_idx, k_idx = _interval_means(d2_cum)               # [n_int]
    T_boot, _, _ = _interval_means(W2_cum)                     # [B, n_int]

    # interval p-values  p_I = P_boot( T_I^boot >= T_I^obs )
    p_I = (T_boot >= T_obs[None, :]).mean(axis=0)              # [n_int]

    # unadjusted p(t) = p of the singleton interval {t};
    # adjusted p_tilde(t) = max over intervals containing t.
    p_unadj = np.empty(res)
    p_adj = np.full(res, -np.inf)
    singleton = j_idx == k_idx
    p_unadj[j_idx[singleton]] = p_I[singleton]
    # scatter-max: p_tilde(t) = max over every interval [j,k] that contains t
    # (np.maximum is order-independent; each interval lifts its whole segment).
    for m in range(p_I.shape[0]):
        seg = p_adj[j_idx[m] : k_idx[m] + 1]
        np.maximum(seg, p_I[m], out=seg)
    return IWTResult(
        tseq=np.asarray(tseq, dtype=float),
        p_unadjusted=p_unadj,
        p_adjusted=p_adj,
        alpha=alpha,
    )


def pini_vantini_band(
    influence,
    tseq,
    estimate,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    multiplier: str = "rademacher",
    variance=None,
    rng=None,
) -> Band:
    """Interval-wise confidence band induced by the IWT parametric bootstrap.

    Half-width ``h(t) = (sigma_hat(t)/sqrt(n)) * k_tilde(t)`` with the adjusted
    critical function ``k_tilde(t) = sqrt( max_{I ∋ t} c_I )``, where ``c_I`` is
    the ``(1-alpha)`` quantile of the ``|I|``-normalised studentised bootstrap
    statistic ``(1/|I|) sum_{s in I} W_b(s)^2``. Because the ``|I|`` normalisation
    dilutes long intervals, ``k_tilde`` is dominated by short intervals near ``t``
    and the band lies between the pointwise and simultaneous bands.

    **Guarantee.** This is the band dual to the IWT adjusted p-value function: it
    controls the *interval-wise* error rate (Pini-Vantini eq. 7), not the
    simultaneous coverage. Use :mod:`.multiplier_bootstrap` for a simultaneous
    band. The claim is validated empirically in Phase 2.6.
    """
    rng = np.random.default_rng(rng)
    phi = np.asarray(influence, dtype=float)
    phi = phi - phi.mean(axis=0, keepdims=True)
    n, res = phi.shape
    estimate = np.asarray(estimate, dtype=float)

    sd = (
        eif_pointwise_sd(phi)
        if variance is None
        else np.sqrt(np.maximum(np.asarray(variance, dtype=float), 1e-12))
    )

    xi = _draw_multipliers(multiplier, (n_boot, n), rng)
    W2 = ((xi @ phi) / np.sqrt(n) / sd) ** 2                     # [B, res]
    W2_cum = np.concatenate([np.zeros((n_boot, 1)), np.cumsum(W2, axis=1)], axis=1)
    T_boot, j_idx, k_idx = _interval_means(W2_cum)             # [B, n_int]
    c_I = np.quantile(T_boot, 1.0 - alpha, axis=0)             # [n_int]

    # k_tilde(t)^2 = max over intervals containing t of c_I  (scatter-max)
    k2 = np.zeros(res)
    for m in range(c_I.shape[0]):
        seg = k2[j_idx[m] : k_idx[m] + 1]
        np.maximum(seg, c_I[m], out=seg)
    half = np.sqrt(k2) * sd / np.sqrt(n)
    return Band(
        tseq=tseq,
        lower=estimate - half,
        upper=estimate + half,
        center=estimate,
        level=1.0 - alpha,
        kind="confidence",
    )


def pini_vantini_bands(cross_fit_result, *, alpha: float = 0.05, **kwargs):
    """One interval-wise band per homology dim from a :class:`CrossFitResult`."""
    influence = cross_fit_result.influence()
    tseq = cross_fit_result.tseq
    return [
        pini_vantini_band(
            influence[d], tseq, cross_fit_result.aipw[d], alpha=alpha, **kwargs
        )
        for d in range(len(influence))
    ]
