"""Pure-numpy evaluation metrics for functional causal estimands.

These mirror the metric conventions in ``top-causal-effect-main`` (trapezoidal
integration; L1 distance to the true effect curve) and add credible-band
coverage / width for the Bayesian evaluation (Research_Plan Phase 4).
"""
from __future__ import annotations

import numpy as np


def numerical_integration(f, tseq) -> float:
    """Trapezoidal integral of ``f`` over grid ``tseq``.

    Matches ``top-causal-effect-main/utils/utils.py::numerical_integration``.
    """
    f = np.asarray(f, dtype=float)
    tseq = np.asarray(tseq, dtype=float)
    delta_t = tseq[1:] - tseq[:-1]
    return float(np.sum((f[1:] + f[:-1]) / 2.0 * delta_t))


def l1_distance(f, g, tseq) -> float:
    """L1 (integrated absolute) distance between curves ``f`` and ``g``."""
    return numerical_integration(np.abs(np.asarray(f) - np.asarray(g)), tseq)


def rmse(estimate, truth) -> float:
    """Root-mean-squared error between two curves (pointwise, unweighted)."""
    e = np.asarray(estimate, dtype=float)
    t = np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean((e - t) ** 2)))


def bias(estimate, truth) -> float:
    """Mean signed error (pointwise) between an estimate and the truth."""
    e = np.asarray(estimate, dtype=float)
    t = np.asarray(truth, dtype=float)
    return float(np.mean(e - t))


def integrated_bias(estimate, truth, tseq) -> float:
    """Trapezoidal integral of the signed error, grid-averaged to [t0, t1]."""
    e = np.asarray(estimate, dtype=float)
    t = np.asarray(truth, dtype=float)
    tseq = np.asarray(tseq, dtype=float)
    span = tseq[-1] - tseq[0]
    return float(numerical_integration(e - t, tseq) / span)


def max_abs_error(estimate, truth) -> float:
    """Sup-norm (uniform) error between two curves."""
    e = np.asarray(estimate, dtype=float)
    t = np.asarray(truth, dtype=float)
    return float(np.max(np.abs(e - t)))


def pointwise_coverage(lower, upper, truth) -> float:
    """Fraction of grid points where ``truth`` lies within ``[lower, upper]``."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.mean((truth >= lower) & (truth <= upper)))


def simultaneous_coverage(lower, upper, truth) -> float:
    """1.0 if the entire ``truth`` curve is inside the band, else 0.0.

    Average this indicator over many simulated datasets to estimate the
    simultaneous (uniform) coverage of a credible band.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.all((truth >= lower) & (truth <= upper)))


def clopper_pearson(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper–Pearson) binomial confidence interval for a coverage rate.

    Returns ``(lo, hi)`` at confidence ``1 - alpha`` for ``k = successes`` out of
    ``n`` Bernoulli trials, using the Beta-quantile form.  Used to attach honest
    Monte-Carlo error bars to the simultaneous-coverage rates in the Phase-4.5
    decision grid (a coverage of ``k/n`` with small ``n`` is nearly
    uninformative; ≥50 reps are required for a ±0.06 bound near 0.95).
    """
    from scipy.stats import beta as _beta

    n = int(n)
    k = int(successes)
    if n <= 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(_beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


def interval_width(lower, upper, tseq=None) -> float:
    """Mean band width, or grid-averaged width if ``tseq`` is given."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    if tseq is None:
        return float(np.mean(width))
    tseq = np.asarray(tseq, dtype=float)
    span = tseq[-1] - tseq[0]
    return float(numerical_integration(width, tseq) / span)


# --------------------------------------------------------------------------- #
# Peak-localised metrics + the fundamental estimand floor (Research_Plan P5.1)
# --------------------------------------------------------------------------- #
# The 2026-07-06 probe (docs/phase5_probe.md) showed the whole ``cov_sim_clean=0``
# failure lives at the *silhouette apex*: the effect curve is a bump, and every
# estimator loses the peak while the tails are covered.  The L2 attenuation ratio
# the project tracked averages that away and reads a benign ~0.8.  These helpers
# score the apex directly.

def peak_index(truth) -> int:
    """Grid index of the clean-truth apex ``argmax_t |psi_d(t)|``.

    All peak-localised metrics are anchored to the *reference* apex (not the
    estimate's), so a method cannot look good by moving its own peak elsewhere.
    """
    truth = np.asarray(truth, dtype=float)
    if truth.size == 0:
        return 0
    return int(np.argmax(np.abs(truth)))


def peak_signed_bias(estimate, truth) -> float:
    """Signed error ``estimate - truth`` at the clean-truth apex.

    Negative = the estimator under-shoots the peak (the attenuation direction the
    probe found for every non-arm-aware Step-1).
    """
    e = np.asarray(estimate, dtype=float)
    t = np.asarray(truth, dtype=float)
    j = peak_index(t)
    return float(e[j] - t[j])


def peak_abs_error(estimate, truth) -> float:
    """``|estimate - truth|`` at the clean-truth apex (peak attenuation gap)."""
    return float(abs(peak_signed_bias(estimate, truth)))


def peak_retention(estimate, truth) -> float:
    """Fraction of the clean peak *height* the estimate keeps, ``e[j]/t[j]``.

    Directly comparable to the probe's ``peak(psi_x)/peak(psi_d)`` retentions
    (Table 2 of docs/phase5_probe.md); 1.0 means the apex is preserved, values
    below the frequentist floor ``1 - F`` mean the denoiser over-smooths.
    """
    e = np.asarray(estimate, dtype=float)
    t = np.asarray(truth, dtype=float)
    j = peak_index(t)
    if abs(t[j]) <= 1e-12:
        return float("nan")
    return float(e[j] / t[j])


def _peak_window(truth, window: int) -> slice:
    truth = np.asarray(truth, dtype=float)
    j = peak_index(truth)
    lo = max(0, j - int(window))
    hi = min(truth.size, j + int(window) + 1)
    return slice(lo, hi)


def peak_localized_coverage(lower, upper, truth, window: int = 6) -> float:
    """1.0 iff the band covers ``truth`` **simultaneously over a peak window**.

    ``window`` is a half-width in grid points either side of the reference apex.
    This is the strict, decision-relevant coverage in Phase 5: the tails are easy,
    so we ask whether the band contains the truth *where the effect actually is*.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    truth = np.asarray(truth, dtype=float)
    sl = _peak_window(truth, window)
    seg_t, seg_lo, seg_hi = truth[sl], lower[sl], upper[sl]
    return float(np.all((seg_t >= seg_lo) & (seg_t <= seg_hi)))


def peak_pointwise_coverage(lower, upper, truth, window: int = 6) -> float:
    """Fraction of the peak-window grid points where ``truth`` is inside the band."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    truth = np.asarray(truth, dtype=float)
    sl = _peak_window(truth, window)
    seg_t, seg_lo, seg_hi = truth[sl], lower[sl], upper[sl]
    if seg_t.size == 0:
        return float("nan")
    return float(np.mean((seg_t >= seg_lo) & (seg_t <= seg_hi)))


def fundamental_floor(ref_mc, ref_clean) -> float:
    """The fundamental estimand floor ``F = 1 - peak(psi*) / peak(psi_d)``.

    ``ref_mc`` is the self-consistent Monte-Carlo estimand ``psi*`` any
    noisy-silhouette estimator (the frequentist AIPW in particular) converges to;
    ``ref_clean`` is the injected-loop truth ``psi_d``.  ``F`` is the fraction of
    the clean peak **amplitude** that no ``psi*``-centred band can recover — the
    best the frequentist could possibly do (Research_Plan Theorem A / P5.1).  A
    Bayesian method is only interesting in a cell where ``F`` is large.

    Peaks are **sup-norms of each curve** (peak-to-peak), matching the probe's
    ``floor_frac`` and the Theorem-A statement ``peak(psi_d) - peak(psi*)``.  This
    is deliberately *not* pinned to a single grid point: the noisy silhouette peak
    can shift slightly off the clean apex, and a per-apex ratio would then read a
    spuriously large floor.  (Contrast :func:`peak_retention`, which *is* pinned to
    the clean apex, because an estimator must cover ``psi_d`` at that specific
    location.)
    """
    clean = np.asarray(ref_clean, dtype=float)
    mc = np.asarray(ref_mc, dtype=float)
    denom = float(np.max(np.abs(clean)))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - float(np.max(np.abs(mc))) / denom)


# --------------------------------------------------------------------------- #
# Displacement diagnostics (Research_Plan Phase 5.5, Task 5.5.1)
# --------------------------------------------------------------------------- #
# Phase 5 showed the whole `cov_sim_clean = 0` failure lives at the silhouette
# apex, and is driven by the signal loop's death-time collapse under interior
# clutter.  The L2 attenuation ratio the project tracked for months averages
# this away and reads a benign ~0.8.  These metrics score the displacement
# directly, in t-units (range-invariant) rather than grid indices.


def apex_location(psi, grid):
    """Filtration-parameter location of the absolute apex of ``psi(t)``.

    Returns ``grid[argmax |psi|]`` in *t*-units, not a grid index — so the
    metric is comparable across filtrations with different scales.
    """
    psi = np.asarray(psi, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if psi.size == 0:
        return float("nan")
    return float(grid[int(np.argmax(np.abs(psi)))])


def apex_shift(psi_a, psi_b, grid):
    """Signed apex shift ``t_apex(a) - t_apex(b)`` in filtration-parameter units.

    Negative when ``psi_a`` is displaced leftward relative to ``psi_b`` (the
    clutter direction found in Phase 5).  Always in *t*-units.
    """
    return apex_location(psi_a, grid) - apex_location(psi_b, grid)


def apex_floor(psi_noisy, psi_clean, grid):
    """Apex-anchored floor ``F_apex = 1 - psi_noisy[argmax|psi_clean|] / psi_clean[argmax|psi_clean|]``.

    Unlike :func:`fundamental_floor` (which uses the *sup-norm* of each curve,
    i.e. peak-to-peak), this is pinned to the *clean* apex location.
    ``F_apex ≈ 1`` means the noisy curve has essentially zero signal at the
    location where the clean effect peaks — the apex has been displaced, not
    just attenuated.  ``F_apex = 0`` means the noisy curve preserves the clean
    peak value at the correct location.

    This is the metric that Phase 5's probe showed captures the failure: while
    ``F`` (sup-norm) reads 0.61–0.87 (amplitude), ``F_apex`` reads 0.95–1.00
    (location) on the low-SNR DGP.  Score ``F_apex`` alongside ``F`` in every
    decision row.
    """
    psi_noisy = np.asarray(psi_noisy, dtype=float)
    psi_clean = np.asarray(psi_clean, dtype=float)
    grid = np.asarray(grid, dtype=float)
    j = int(np.argmax(np.abs(psi_clean)))
    denom = float(psi_clean[j])
    if abs(denom) <= 1e-12:
        return float("nan")
    return float(1.0 - float(psi_noisy[j]) / denom)


def death_recovery(d_post, d_obs, d_clean):
    """Normalized death-time recovery ratio.

    ``death_recovery = (d_post - d_clean) / (d_obs - d_clean)`` when
    ``d_obs != d_clean`` (so 1.0 = no correction, 0.0 = full correction).  An
    alternative convention (used in Task 5.5.3) is
    ``death_recovery_ratio = (d_post - d_obs) / (d_clean - d_obs)`` which is
    0 = no correction, 1 = full correction.  Both are provided.

    Parameters
    ----------
    d_post : float
        Death coordinate of the signal feature in the posterior / corrected diagram.
    d_obs : float
        Death coordinate of the signal feature in the observed diagram.
    d_clean : float
        Death coordinate of the signal feature in the clean diagram.

    Returns
    -------
    dict with ``death_recovery_frac`` (=(d_obs-d_clean)/d_clean, the Phase 5.5
    metric) and ``death_recovery_ratio`` (=(d_post-d_obs)/(d_clean-d_obs), the
    Task 5.5.3 correction metric).
    """
    d_post, d_obs, d_clean = float(d_post), float(d_obs), float(d_clean)
    frac = (d_obs - d_clean) / d_clean if abs(d_clean) > 1e-12 else float("nan")
    gap = d_clean - d_obs
    ratio = (d_post - d_obs) / gap if abs(gap) > 1e-12 else float("nan")
    return {"death_recovery_frac": frac, "death_recovery_ratio": ratio}
