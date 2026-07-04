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
