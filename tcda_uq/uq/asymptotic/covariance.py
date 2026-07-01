"""Shared EIF covariance estimation (Phase 2.4).

Every asymptotic band in this phase is a functional of the *same* object: the
cross-fitted efficient-influence-function (EIF) process

    phi_hat_{i,d}(t) = phi_d(t, Z_i; eta_hat)          (TATE eq. 8, per unit)

centered at the AIPW estimate. By Theorem 5.2 the limit process ``G_d`` has

    cov{G_d(s), G_d(t)} = cov{phi_d(s, Z; eta), phi_d(t, Z; eta)},

so the empirical covariance of the centered EIF process is the plug-in estimate
of the limiting covariance kernel. Its diagonal is the pointwise variance that

  * studentizes the multiplier-bootstrap band (:mod:`.multiplier_bootstrap`),
  * is the ``diag.cov.x`` input to the Liebl-Reimherr band (:mod:`.liebl_reimherr`),
  * seeds the parametric bootstrap of the Pini-Vantini band (:mod:`.pini_vantini`).

Conventions
-----------
``influence`` is the centered EIF process for a single homology dimension, shape
``[n, resolution]`` -- exactly ``cross_fit(...).influence()[d]``. All estimators
scale as ``sqrt(n) (psi_hat - psi) -> G_d``, so the covariance of ``G_d`` is the
covariance of ``phi`` (an O(1) object), estimated by ``(1/n) sum_i phi_i phi_i^T``
when ``influence`` is already centered.
"""

from __future__ import annotations

import numpy as np


def eif_covariance(influence, ddof: int = 0):
    """Empirical covariance kernel of the centered EIF process.

    Args:
        influence: centered EIF process ``[n, resolution]``. Assumed (approximately)
            mean-zero over units, as returned by ``cross_fit(...).influence()``.
        ddof: divisor is ``n - ddof``. ``ddof=0`` gives the plug-in ``(1/n) sum
            phi_i phi_i^T`` consistent with Theorem 5.2's ``cov{G_d}``; ``ddof=1``
            gives the unbiased sample covariance.

    Returns:
        ``Sigma_hat`` of shape ``[resolution, resolution]`` estimating
        ``cov{G_d(s), G_d(t)}``.
    """
    phi = np.asarray(influence, dtype=float)
    n = phi.shape[0]
    # re-center defensively (cross-fitting can leave a tiny non-zero mean)
    phi = phi - phi.mean(axis=0, keepdims=True)
    return (phi.T @ phi) / (n - ddof)


def eif_pointwise_variance(influence, ddof: int = 0):
    """Pointwise variance ``sigma_hat_d(t)^2 = cov{G_d(t), G_d(t)}``, shape ``[res]``.

    This is ``diag(eif_covariance(...))`` computed directly (cheaper, no dense
    kernel). It is the pointwise asymptotic variance of ``sqrt(n) psi_hat_d(t)``.
    """
    phi = np.asarray(influence, dtype=float)
    n = phi.shape[0]
    phi = phi - phi.mean(axis=0, keepdims=True)
    return np.einsum("it,it->t", phi, phi) / (n - ddof)


def eif_pointwise_sd(influence, ddof: int = 0, floor: float = 1e-12):
    """Pointwise standard deviation ``sigma_hat_d(t)``, floored away from zero.

    The ``floor`` guards the studentization ``phi(t) / sigma(t)`` at grid points
    where the estimated variance collapses (e.g. a silhouette that is identically
    zero on part of the domain).
    """
    var = eif_pointwise_variance(influence, ddof=ddof)
    return np.sqrt(np.maximum(var, floor))


def eif_correlation(influence, ddof: int = 0, floor: float = 1e-12):
    """Correlation kernel of ``G_d`` -- the covariance standardized by its diagonal.

    Shape ``[resolution, resolution]``. Used by the parametric-bootstrap /
    interval-wise bands, which operate on the studentized process.
    """
    cov = eif_covariance(influence, ddof=ddof)
    sd = np.sqrt(np.maximum(np.diag(cov), floor))
    return cov / np.outer(sd, sd)
