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
