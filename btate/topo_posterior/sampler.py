r"""Posterior persistence-diagram sampler — Research_Plan Task 1.1 (Phase 1).

Given a fitted ``bayes_tda.intensities.Posterior``, draw posterior persistence
diagrams :math:`\widetilde{\mathcal D}_i^a` whose *expected* intensity is the
closed-form posterior intensity ``posterior.evaluate`` (Maroulas et al. 2020).

Model / method
--------------
The posterior intensity is a (restricted) Gaussian mixture on the birth--
persistence wedge:

    posterior.evaluate(x) = 1[x in wedge] * sum_i coeff_i * N(x | mu_i, var_i I),

with two families of components (see ``bayes_tda.intensities.Posterior``):

* data term  ``coeff = (alpha / m) * C_j``   at ``posterior.posterior_means``,
  ``posterior.posterior_sigmas`` (these ``sigmas`` are **variances** in the
  ``RGaussianMixture.evaluate`` convention);
* prior term ``coeff = (1 - alpha) * w_k``   at ``posterior.prior.mus`` /
  ``.sigmas`` (only when ``alpha < 1``; at ``alpha = 1`` it vanishes; for
  ``alpha > 1`` it is negative and is dropped, with a warning).

A diagram is drawn as a finite point process:

1. cardinality ``N`` from the posterior cardinality model — a ``Poisson`` draw
   with mean ``posterior.lambd`` (the mean observed diagram size), the natural
   first cut suggested in the plan; ``count="fixed"`` forces a size;
2. ``N`` locations i.i.d. from the normalized spatial density.  Component
   selection is proportional to the (non-negative) ``coeff_i`` and each draw is
   **rejection-truncated** to the wedge (``b >= min_birth`` and ``p > 0``).
   Because rejection reproduces the mask in ``posterior.evaluate`` exactly, the
   accepted locations are distributed as the *normalized* posterior intensity —
   sidestepping the closed-form normalizing constants ``Q`` entirely and giving
   an exact match to ``posterior.evaluate`` for the moment / KDE validation
   (Research_Plan risk register).

Samples are returned in **birth--persistence** coordinates (``bayes_tda``
convention); use ``btate.topo_posterior.adapters.bp_to_bd`` for the TATE /
gudhi silhouette transform.
"""
from __future__ import annotations

import warnings

import numpy as np


class PosteriorDiagramSampler:
    """Draw posterior persistence diagrams from a Maroulas posterior intensity."""

    def __init__(self, posterior, min_birth: float | None = None,
                 include_prior: bool = True):
        # ``posterior`` is a bayes_tda.intensities.Posterior instance.
        self.posterior = posterior
        self.min_birth = posterior.min_birth if min_birth is None else float(min_birth)
        self.expected_cardinality = float(posterior.lambd)
        self._build_mixture(include_prior)

    # -- mixture assembly ----------------------------------------------------
    def _build_mixture(self, include_prior: bool) -> None:
        post = self.posterior
        alpha, m = post.alpha, post.num_obs_dgms

        means = [np.atleast_2d(post.posterior_means)]
        variances = [np.asarray(post.posterior_sigmas, dtype=float).ravel()]
        coeffs = [(alpha / m) * np.asarray(post.Cs, dtype=float).ravel()]

        prior_coeff = 1.0 - alpha
        if include_prior and prior_coeff > 0:
            means.append(np.atleast_2d(post.prior.mus))
            variances.append(np.asarray(post.prior.sigmas, dtype=float).ravel())
            coeffs.append(prior_coeff * np.asarray(post.prior.weights, dtype=float).ravel())
        elif include_prior and prior_coeff < 0:
            warnings.warn(
                "alpha > 1 makes the prior term of the posterior intensity "
                "negative (a signed measure); sampling from the data term only.",
                RuntimeWarning, stacklevel=2,
            )

        self.means = np.vstack(means)
        self.variances = np.concatenate(variances)
        coeff = np.concatenate(coeffs)
        coeff = np.clip(coeff, 0.0, None)
        total = coeff.sum()
        if total <= 0:
            raise ValueError("posterior intensity has non-positive total mass")
        self.component_probs = coeff / total
        self.mixture_mass = float(total)

    # -- spatial sampling ----------------------------------------------------
    def _sample_locations(self, n_points: int, rng, max_batches: int = 10000):
        """``n_points`` i.i.d. draws from the normalized posterior intensity."""
        if n_points == 0:
            return np.empty((0, 2), dtype=float)
        stds = np.sqrt(self.variances)
        out = np.empty((n_points, 2), dtype=float)
        filled = 0
        batches = 0
        while filled < n_points:
            batches += 1
            if batches > max_batches:
                raise RuntimeError(
                    "rejection sampling failed to fill the diagram; the wedge "
                    "mass may be negligible for this posterior."
                )
            need = n_points - filled
            draw = int(max(need * 2, 32))
            comp = rng.choice(self.component_probs.shape[0], size=draw,
                              p=self.component_probs)
            pts = rng.normal(self.means[comp], stds[comp][:, None])
            keep = (pts[:, 0] >= self.min_birth) & (pts[:, 1] > 0.0)
            acc = pts[keep]
            take = min(acc.shape[0], need)
            out[filled:filled + take] = acc[:take]
            filled += take
        return out

    def sample_diagrams(self, n_samples: int, random_state=None,
                        count: str = "poisson", cardinality: int | None = None):
        """Draw ``n_samples`` posterior persistence diagrams.

        Parameters
        ----------
        n_samples : int
            Number of posterior diagram draws.
        random_state : int | np.random.Generator, optional
        count : {"poisson", "fixed"}
            ``"poisson"`` draws ``N ~ Poisson(posterior.lambd)`` per diagram
            (posterior cardinality model); ``"fixed"`` uses ``cardinality``.
        cardinality : int, optional
            Diagram size when ``count="fixed"`` (defaults to
            ``round(posterior.lambd)``).

        Returns
        -------
        list of np.ndarray
            ``n_samples`` arrays of shape ``(N_s, 2)`` in birth--persistence
            coordinates.
        """
        rng = np.random.default_rng(random_state)
        diagrams = []
        for _ in range(n_samples):
            if count == "poisson":
                k = int(rng.poisson(self.expected_cardinality))
            elif count == "fixed":
                k = int(round(self.expected_cardinality) if cardinality is None
                        else cardinality)
            else:
                raise ValueError("count must be 'poisson' or 'fixed'")
            diagrams.append(self._sample_locations(k, rng))
        return diagrams
