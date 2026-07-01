"""PI / IPW / AIPW estimators for the functional (silhouette) treatment effect.

The *mean* estimators (:func:`ipw_estimator`, :func:`plugin_estimator`,
:func:`aipw_estimator`) are faithful ports of ``top-causal-effect-main``.
:func:`aipw_scores` / :func:`aipw_influence` return the *per-unit* doubly-robust
score process phi_hat_d(t, Z; eta_hat) -- the object Phase 2 turns into bands.

Conventions
-----------
``sample`` is a triplet ``(phi, A, X)`` with
  * ``phi``: silhouettes, shape ``[n, n_hom_dim, resolution]``
  * ``A``:   treatment, shape ``[n]``
  * ``X``:   covariates, shape ``[n, d]``
``mu_hats`` is a length-``n_hom_dim`` list of ``(mu0_hat, mu1_hat)`` predictions,
each of shape ``[n, resolution]`` (evaluated on the *estimation* sample).
``pi_hat`` is the estimated propensity, shape ``[n]``.
"""

from __future__ import annotations

import numpy as np

# clip estimated propensities away from {0, 1} (matches the original code)
EPS_PI = 1e-2


def _clip_pi(pi_hat):
    pi_hat = np.asarray(pi_hat, dtype=float).copy()
    pi_hat[pi_hat <= 0.0] = EPS_PI
    pi_hat[pi_hat >= 1.0] = 1.0 - EPS_PI
    return pi_hat


def _inv_weight(pi_hat, A):
    """IPW sign-weight  A/pi - (1-A)/(1-pi),  shape ``[n, 1]``."""
    A = np.asarray(A, dtype=float)
    return (A / pi_hat - (1 - A) / (1 - pi_hat))[:, np.newaxis]


def ipw_estimator(pi_hat, sample, return_inv_weight=False):
    """Mean IPW estimate per homology dim. Returns list of ``[resolution]`` arrays."""
    phi, A, _ = sample
    n_hom_dim = phi.shape[-2]
    pi_hat = _clip_pi(pi_hat)
    inv_weight = _inv_weight(pi_hat, A)

    ipw = [np.mean(inv_weight * phi[:, d, :], axis=0) for d in range(n_hom_dim)]
    if return_inv_weight:
        return ipw, inv_weight
    return ipw


def plugin_estimator(mu_hats, return_mu=False):
    """Mean plug-in estimate per homology dim. Returns list of ``[resolution]`` arrays."""
    plugin = [np.mean(mu1 - mu0, axis=0) for mu0, mu1 in mu_hats]
    if return_mu:
        mu0_list, mu1_list = zip(*mu_hats)
        return plugin, list(mu0_list), list(mu1_list)
    return plugin


def aipw_estimator(pi_hat, mu_hats, sample):
    """Mean AIPW (doubly-robust) estimate per homology dim. List of ``[resolution]``."""
    phi, A, _ = sample
    n_hom_dim = phi.shape[-2]

    ipw, inv_weight = ipw_estimator(pi_hat, sample, return_inv_weight=True)
    plugin, mu0_list, mu1_list = plugin_estimator(mu_hats, return_mu=True)

    A = np.asarray(A, dtype=float)[:, np.newaxis]
    dr = []
    for d in range(n_hom_dim):
        correction = ipw[d] - np.mean(
            inv_weight * (A * mu1_list[d] + (1 - A) * mu0_list[d]), axis=0
        )
        dr.append(plugin[d] + correction)
    return dr


def aipw_scores(pi_hat, mu_hats, sample):
    """Per-unit doubly-robust score process, one ``[n, resolution]`` array per hom dim.

    The mean over units equals :func:`aipw_estimator`. The *centered* version
    (see :func:`aipw_influence`) is the estimated efficient influence function
    process phi_hat_d(t, Z_i; eta_hat) that the multiplier bootstrap (Phase 2.1)
    reweights.
    """
    phi, A, _ = sample
    n_hom_dim = phi.shape[-2]
    pi_hat = _clip_pi(pi_hat)
    inv_weight = _inv_weight(pi_hat, A)  # [n, 1]
    A = np.asarray(A, dtype=float)[:, np.newaxis]

    scores = []
    for d in range(n_hom_dim):
        mu0, mu1 = mu_hats[d]
        plugin = mu1 - mu0                                   # [n, res]
        ipw = inv_weight * phi[:, d, :]                      # [n, res]
        model = inv_weight * (A * mu1 + (1 - A) * mu0)       # [n, res]
        scores.append(plugin + ipw - model)                 # [n, res]
    return scores


def aipw_influence(pi_hat, mu_hats, sample):
    """Centered EIF process ``scores - mean``, one ``[n, resolution]`` array per hom dim."""
    scores = aipw_scores(pi_hat, mu_hats, sample)
    return [s - s.mean(axis=0, keepdims=True) for s in scores]
