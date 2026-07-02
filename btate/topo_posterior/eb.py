r"""Empirical-Bayes (type-II ML) selection of the Maroulas ``sigma_DYO``.

Phase 4.25 replaced the fixed ``sigma_dyo`` with the adaptive rule
``c_sigma * median(prior.sigmas)``, but ``c_sigma`` itself was still tuned
against downstream coverage on benchmark cells — a procedure that does not
transfer across DGPs.  This module removes that knob: ``sigma_DYO`` is chosen
by maximizing the marked-PPP marginal likelihood of the *observed* diagrams
under the elicited prior/clutter model, the direct analogue of BART's
data-anchored ``sigma`` prior (Chipman, George & McCulloch 2010): the data
pick the noise scale, the model class stays fixed.

Model
-----
Under the Maroulas et al. (2020) model, an observed diagram is a Poisson point
process whose intensity at ``y`` (birth--persistence coordinates) is

.. math::

    \nu_\sigma(y) = c(y) + \alpha \sum_j w_j \,
        \mathcal N\!\big(y;\, \mu_j, (v_j + \sigma) I_2\big)
        \cdot \mathbf 1\{y \in W\},

where ``(mu_j, v_j, w_j)`` are the prior intensity's means / per-axis
variances / expected-count weights, ``c`` is the clutter intensity, ``sigma``
is the observation-noise variance (``sigma_DYO``), and ``W`` is the wedge
``{birth >= min_birth, persistence > 0}``.  For ``m`` observed diagrams the
PPP log-likelihood is (up to ``sigma``-independent clutter terms)

.. math::

    \ell(\sigma) = \sum_{y \in \text{DYO}} \log \nu_\sigma(y)
        - m\,\alpha \sum_j w_j Q_j(\sigma),

with ``Q_j(sigma)`` the Gaussian mass inside ``W``.  ``Q_j`` is computed here
directly from normal CDFs (per-axis independence of the isotropic Gaussian)
rather than via ``bayes_tda.RestrictedGaussian``, which squares its ``sigma``
argument and is therefore inconsistent with the variance convention used by
``RGaussianMixture.evaluate`` / ``Posterior``.

Because everything is closed-form, a 1-D grid search over multipliers of
``median(prior.sigmas)`` is cheap, scale-free, and per-dataset adaptive.
"""
from __future__ import annotations

import numpy as np


def _stack_points(diagrams_bp) -> tuple[np.ndarray, int]:
    """Stack diagram points; return ``(points, n_diagrams)`` counting empties."""
    diagrams = list(diagrams_bp)
    if not diagrams:
        raise ValueError("need at least one diagram to select sigma_dyo")
    pts = [np.atleast_2d(np.asarray(d, dtype=float)) for d in diagrams
           if np.asarray(d).size]
    if not pts:
        raise ValueError("all diagrams are empty; cannot select sigma_dyo")
    return np.vstack(pts), len(diagrams)


def _region_mass(mus: np.ndarray, variances: np.ndarray, min_birth: float) -> np.ndarray:
    """Gaussian mass inside the wedge ``{birth >= min_birth, persistence > 0}``."""
    from scipy.stats import norm

    std = np.sqrt(variances)
    mass_p = norm.cdf(mus[:, 1] / std)          # P(persistence > 0)
    if np.isneginf(min_birth):
        return mass_p
    mass_b = norm.cdf((mus[:, 0] - min_birth) / std)
    return mass_b * mass_p


def sigma_dyo_profile_loglik(diagrams_bp, prior, clutter, sigma_dyo: float,
                             alpha: float = 1.0) -> float:
    """Marked-PPP marginal log-likelihood of observed diagrams at one ``sigma_DYO``.

    Up to terms constant in ``sigma_dyo`` (the clutter compensator), so only
    *differences* across ``sigma_dyo`` values are meaningful.

    Parameters
    ----------
    diagrams_bp : list of (n_i, 2) arrays
        Observed diagrams in birth--persistence coordinates.
    prior, clutter : ``bayes_tda`` ``RGaussianMixture``
        Elicited intensities (``sigmas`` are per-axis variances, ``weights``
        expected counts — as produced by :func:`~btate.topo_posterior.elicitation.elicit_prior_clutter`).
    sigma_dyo : float
        Candidate observation-noise variance.
    alpha : float
        Maroulas mixing weight (matches ``Posterior``'s ``alpha``).
    """
    sigma = float(sigma_dyo)
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError("sigma_dyo must be positive and finite")
    Y, m = _stack_points(diagrams_bp)

    mus = np.atleast_2d(np.asarray(prior.mus, dtype=float))
    v = np.asarray(prior.sigmas, dtype=float).ravel() + sigma   # (k,) convolved variances
    w = np.asarray(prior.weights, dtype=float).ravel()

    # Convolved mixture intensity at the observed points.
    d2 = ((Y[:, None, :] - mus[None, :, :]) ** 2).sum(axis=2)   # (N, k)
    dens = np.exp(-0.5 * d2 / v[None, :]) / (2.0 * np.pi * v[None, :])
    signal = dens @ w                                            # (N,)

    clutter_dens = np.asarray(clutter.evaluate(Y), dtype=float).ravel()
    nu = clutter_dens + float(alpha) * signal
    point_term = float(np.sum(np.log(np.maximum(nu, 1e-300))))

    min_birth = float(getattr(prior, "min_birth", 0.0))
    compensator = float(np.sum(w * _region_mass(mus, v, min_birth)))
    return point_term - m * float(alpha) * compensator


def select_sigma_dyo(diagrams_bp, prior, clutter, alpha: float = 1.0,
                     multipliers=None, floor: float = 1e-8,
                     cap: float | None = None) -> dict:
    """Choose ``sigma_DYO`` by marginal-likelihood grid search (empirical Bayes).

    The grid is expressed in multiples of ``median(prior.sigmas)`` so the
    search is scale-free.  Returns a dict with the selected value, its implied
    multiplier, the full profile (for sensitivity plots), and a boundary flag
    (``True`` when the optimum sits on the grid edge — widen the grid then).
    """
    prior_sigmas = np.asarray(prior.sigmas, dtype=float).ravel()
    valid = prior_sigmas[np.isfinite(prior_sigmas) & (prior_sigmas > 0.0)]
    if valid.size == 0:
        raise ValueError("prior.sigmas has no positive finite values")
    med = float(np.median(valid))

    if multipliers is None:
        multipliers = np.geomspace(0.02, 20.0, 25)
    multipliers = np.asarray(multipliers, dtype=float).ravel()
    if multipliers.size < 2 or np.any(multipliers <= 0.0):
        raise ValueError("multipliers must be a grid of >= 2 positive values")

    sigmas = np.maximum(multipliers * med, float(floor))
    if cap is not None:
        sigmas = np.minimum(sigmas, float(cap))
    logliks = np.array([
        sigma_dyo_profile_loglik(diagrams_bp, prior, clutter, s, alpha=alpha)
        for s in sigmas
    ])
    best = int(np.argmax(logliks))
    return {
        "sigma_dyo": float(sigmas[best]),
        "sigma_dyo_multiplier": float(sigmas[best] / med),
        "prior_sigma_median": med,
        "profile_multipliers": multipliers,
        "profile_sigma_dyo": sigmas,
        "profile_loglik": logliks,
        "at_boundary": bool(best in (0, len(sigmas) - 1)),
    }
