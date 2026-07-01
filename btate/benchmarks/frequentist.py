"""Self-contained frequentist TATE estimators for head-to-head evaluation.

These mirror ``top-causal-effect-main/estimators.py`` (IPW / plug-in / AIPW) but
are implemented in pure numpy on single-homology-dimension curve arrays of shape
``(n, resolution)``, and add:

* an **efficient-influence-function** pointwise variance for AIPW, and
* a **Gaussian-multiplier-bootstrap uniform (simultaneous) band**,

so the frequentist estimator produces both pointwise and uniform confidence
bands that are directly comparable to the Bayesian credible bands (Task 4.4).

The function-on-scalar regression uses a Fourier-basis ridge (``function_on_scalar_ridge``)
so plug-in / AIPW run without ``scikit-fda``; when ``scikit-fda`` is available the
same interface can be swapped in, but it is not required.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrequentistEffect:
    """Frequentist ``psi(t)`` estimate with pointwise and uniform bands."""

    grid: np.ndarray
    estimate: np.ndarray
    pointwise_lower: np.ndarray
    pointwise_upper: np.ndarray
    simultaneous_lower: np.ndarray
    simultaneous_upper: np.ndarray
    alpha: float
    estimator: str
    band_excludes_zero: bool
    influence: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


def _fourier_design(grid: np.ndarray, n_basis: int) -> np.ndarray:
    """Orthonormal-ish Fourier design matrix on ``grid`` with ``n_basis`` columns."""
    grid = np.asarray(grid, dtype=float)
    lo, hi = grid[0], grid[-1]
    span = hi - lo if hi > lo else 1.0
    u = (grid - lo) / span
    cols = [np.ones_like(u)]
    k = 1
    while len(cols) < n_basis:
        cols.append(np.sqrt(2.0) * np.sin(2.0 * np.pi * k * u))
        if len(cols) < n_basis:
            cols.append(np.sqrt(2.0) * np.cos(2.0 * np.pi * k * u))
        k += 1
    return np.column_stack(cols[:n_basis])


def function_on_scalar_ridge(phi, X, grid, n_basis: int = 5, ridge: float = 1e-3):
    """Fit E[phi(t) | X] by Fourier-basis-coefficient ridge regression.

    Returns a predictor ``predict(X_new) -> curves`` of shape ``(m, resolution)``.
    """
    phi = np.asarray(phi, dtype=float)
    X = np.asarray(X, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    B = _fourier_design(grid, n_basis)                       # (res, K)
    BtB_inv = np.linalg.inv(B.T @ B + 1e-10 * np.eye(B.shape[1]))
    coef = phi @ B @ BtB_inv                                 # (n, K)
    design = np.column_stack([np.ones(X.shape[0]), X])       # (n, 1 + p)
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0                                      # do not penalize intercept
    W = np.linalg.solve(design.T @ design + penalty, design.T @ coef)  # (1+p, K)

    def predict(X_new):
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new[:, None]
        d = np.column_stack([np.ones(X_new.shape[0]), X_new])
        return (d @ W) @ B.T

    return predict


def _clip_pi(pi_hat, A):
    pi = np.asarray(pi_hat, dtype=float).ravel()
    return np.clip(pi, 1e-2, 1.0 - 1e-2)


def _uniform_band(psi, influence, alpha, rng, n_boot=2000):
    """Gaussian-multiplier-bootstrap simultaneous band from influence functions.

    ``influence`` has shape ``(n, resolution)`` with mean (approximately) zero
    across subjects; ``psi = mean_i estimand_i`` and ``se(t) = std_i / sqrt(n)``.
    """
    n = influence.shape[0]
    se = influence.std(axis=0, ddof=1) / np.sqrt(n)
    se = np.maximum(se, 1e-12)
    centered = influence - influence.mean(axis=0, keepdims=True)
    # sup_t | (1/n) sum_i g_i IF_i(t) | / se(t)
    g = rng.standard_normal(size=(n_boot, n))
    boot = (g @ centered) / n                                # (n_boot, res)
    sup = np.max(np.abs(boot) / se[None, :], axis=1)
    crit = float(np.quantile(sup, 1.0 - alpha))
    z = 1.959963984540054
    return se, crit, z


def cross_fitted_scores(phi, A, X, grid, pi_hat=None, n_basis: int = 5,
                        cross_fit: bool = True, random_state=None) -> dict:
    """Cross-fitted per-unit AIPW score process (the efficient influence function).

    Returns a dict with the mean ``aipw``/``ipw``/``plugin`` curves, the per-unit
    DR ``scores`` (mean == ``aipw``), the centered ``influence`` process (mean ~ 0)
    that the ``tcda_uq`` simultaneous bands consume, and the clipped ``pi``.  Uses
    the skfda-free Fourier-basis ridge for the outcome regression, so it runs in a
    minimal environment while matching the DR structure of
    ``tcda_uq.estimators.cross_fit``.
    """
    phi = np.asarray(phi, dtype=float)
    A = np.asarray(A, dtype=int).ravel()
    X = np.asarray(X, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n = phi.shape[0]
    rng = np.random.default_rng(random_state)
    pi = np.full(n, float(np.mean(A))) if pi_hat is None else _clip_pi(pi_hat, A)

    mu1 = np.zeros_like(phi)
    mu0 = np.zeros_like(phi)
    if cross_fit and n >= 8:
        order = rng.permutation(n)
        folds = np.array_split(order, 2)
        for k in range(2):
            test = folds[k]
            train = folds[1 - k]
            tr1 = train[A[train] == 1]
            tr0 = train[A[train] == 0]
            if tr1.size >= 2 and tr0.size >= 2:
                p1 = function_on_scalar_ridge(phi[tr1], X[tr1], grid, n_basis)
                p0 = function_on_scalar_ridge(phi[tr0], X[tr0], grid, n_basis)
                mu1[test] = p1(X[test])
                mu0[test] = p0(X[test])
            else:  # degenerate fold: fall back to marginal means
                mu1[test] = phi[A == 1].mean(axis=0) if np.any(A == 1) else 0.0
                mu0[test] = phi[A == 0].mean(axis=0) if np.any(A == 0) else 0.0
    else:
        p1 = function_on_scalar_ridge(phi[A == 1], X[A == 1], grid, n_basis)
        p0 = function_on_scalar_ridge(phi[A == 0], X[A == 0], grid, n_basis)
        mu1 = p1(X)
        mu0 = p0(X)

    a = A[:, None].astype(float)
    pic = pi[:, None]
    resid1 = (a / pic) * (phi - mu1)
    resid0 = ((1.0 - a) / (1.0 - pic)) * (phi - mu0)
    scores = (mu1 - mu0) + resid1 - resid0                  # per-unit DR score
    aipw = scores.mean(axis=0)
    return {
        "grid": grid,
        "aipw": aipw,
        "ipw": ((a / pic - (1.0 - a) / (1.0 - pic)) * phi).mean(axis=0),
        "plugin": (mu1 - mu0).mean(axis=0),
        "scores": scores,
        "influence": scores - aipw,                          # centered EIF, mean ~ 0
        "pi": pi,
    }


def aipw_effect(phi, A, X, grid, pi_hat=None, alpha: float = 0.05,
                n_basis: int = 5, cross_fit: bool = True, n_boot: int = 2000,
                random_state=None, estimator: str = "aipw") -> FrequentistEffect:
    """Doubly-robust AIPW ``psi(t)`` with EIF pointwise and uniform bands.

    ``phi`` has shape ``(n, resolution)`` (one homology dimension).  ``estimator``
    selects ``{"aipw", "ipw", "plugin"}`` for the point estimate; the influence
    function (hence the bands) always uses the AIPW EIF.  The simultaneous band is
    the Gaussian-multiplier bootstrap (equivalent to
    ``tcda_uq``'s :func:`multiplier_bootstrap_band`).
    """
    grid = np.asarray(grid, dtype=float)
    rng = np.random.default_rng(random_state)
    cf = cross_fitted_scores(phi, A, X, grid, pi_hat=pi_hat, n_basis=n_basis,
                             cross_fit=cross_fit, random_state=random_state)
    pointwise_i = cf["scores"]
    aipw, ipw, plugin, influence = cf["aipw"], cf["ipw"], cf["plugin"], cf["influence"]

    if estimator == "aipw":
        psi = aipw
    elif estimator == "ipw":
        psi = ipw
    elif estimator == "plugin":
        psi = plugin
    else:
        raise ValueError("estimator must be one of {'aipw', 'ipw', 'plugin'}")

    n = pointwise_i.shape[0]
    se, crit, z = _uniform_band(psi, pointwise_i, alpha, rng, n_boot=n_boot)
    pointwise_lower = psi - z * se
    pointwise_upper = psi + z * se
    simultaneous_lower = psi - crit * se
    simultaneous_upper = psi + crit * se
    band_excludes_zero = bool(
        np.all(simultaneous_lower > 0.0) or np.all(simultaneous_upper < 0.0)
    )
    return FrequentistEffect(
        grid=grid, estimate=psi,
        pointwise_lower=pointwise_lower, pointwise_upper=pointwise_upper,
        simultaneous_lower=simultaneous_lower, simultaneous_upper=simultaneous_upper,
        alpha=alpha, estimator=estimator, band_excludes_zero=band_excludes_zero,
        influence=influence,
        metadata={
            "n": int(n), "n_basis": int(n_basis), "cross_fit": bool(cross_fit),
            "uniform_crit": crit, "n_boot": int(n_boot),
            "plugin": plugin, "ipw": ipw, "aipw": aipw,
        },
    )


# --------------------------------------------------------------------------- #
# Faithful TATE simultaneous bands via tcda_uq (Kim & Lee 2026, Thm 5.2)
# --------------------------------------------------------------------------- #
_TCDA_METHODS = ("multiplier_bootstrap", "liebl_reimherr", "pini_vantini")


def _tcda_band_fns():
    """Lazily import the three ``tcda_uq`` band constructors (skfda-free)."""
    from tcda_uq.uq.asymptotic.multiplier_bootstrap import multiplier_bootstrap_band
    from tcda_uq.uq.asymptotic.liebl_reimherr import liebl_reimherr_band
    from tcda_uq.uq.asymptotic.pini_vantini import pini_vantini_band
    return {
        "multiplier_bootstrap": multiplier_bootstrap_band,
        "liebl_reimherr": liebl_reimherr_band,
        "pini_vantini": pini_vantini_band,
    }


def frequentist_bands(phi, A, X, grid, pi_hat=None, alpha: float = 0.05,
                      methods=_TCDA_METHODS, n_basis: int = 5, cross_fit: bool = True,
                      random_state=None, n_boot: int = 2000,
                      liebl_backend: str = "python"):
    """Frequentist AIPW estimate + the faithful ``tcda_uq`` simultaneous bands.

    Computes the cross-fitted efficient-influence-function process with the
    skfda-free Fourier-ridge outcome regression, then constructs each requested
    band from ``tcda_uq.uq.asymptotic`` (Kim & Lee 2026 Corollary 5.4 multiplier
    bootstrap; Liebl-Reimherr fast-and-fair; Pini-Vantini interval-wise).

    ``liebl_backend`` selects the FFSCB backend: ``"python"`` (fast port; use in
    sweeps) or ``"R"``/``"auto"`` (the original ``ffscb`` R source, if available).

    Returns ``(estimate, {method: tcda_uq.metrics.Band}, cross_fit_dict)``.  If
    ``tcda_uq`` is unavailable, only the internal multiplier band is returned.
    """
    grid = np.asarray(grid, dtype=float)
    cf = cross_fitted_scores(phi, A, X, grid, pi_hat=pi_hat, n_basis=n_basis,
                             cross_fit=cross_fit, random_state=random_state)
    estimate = cf["aipw"]
    influence = cf["influence"]

    try:
        fns = _tcda_band_fns()
    except Exception:  # tcda_uq not importable -> internal multiplier fallback
        eff = aipw_effect(phi, A, X, grid, pi_hat=pi_hat, alpha=alpha,
                          n_basis=n_basis, cross_fit=cross_fit, n_boot=n_boot,
                          random_state=random_state)
        from tcda_uq.metrics.bands import Band  # optional; only if partially present
        band = None
        try:
            band = Band(tseq=grid, lower=eff.simultaneous_lower,
                        upper=eff.simultaneous_upper, center=eff.estimate,
                        level=1 - alpha, kind="confidence")
        except Exception:
            band = eff  # last resort: FrequentistEffect duck-types lower/upper
        return estimate, {"multiplier_bootstrap": band}, cf

    bands = {}
    for m in methods:
        if m == "multiplier_bootstrap":
            bands[m] = fns[m](influence, grid, estimate, alpha=alpha,
                              n_boot=n_boot, rng=random_state)
        elif m == "liebl_reimherr":
            bands[m] = fns[m](influence, grid, estimate, alpha=alpha,
                              backend=liebl_backend)
        elif m == "pini_vantini":
            bands[m] = fns[m](influence, grid, estimate, alpha=alpha,
                              n_boot=n_boot, rng=random_state)
        else:
            raise ValueError(f"unknown method {m!r}; choose from {_TCDA_METHODS}")
    return estimate, bands, cf
