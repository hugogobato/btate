r"""Prior & clutter intensity elicitation — Research_Plan Task 1.2 (Phase 1).

Builds default prior and clutter intensities (``bayes_tda`` restricted Gaussian
mixtures) for the Maroulas posterior from a set of *training* persistence
diagrams, and provides a sensitivity harness over ``sigma_DYO``, ``alpha`` and
the prior component count — mirroring the train/test split logic of
``bayes_tda.classifiers.EmpBayesFactorClassifier``.

Conventions
-----------
* Diagrams are in **birth--persistence** coordinates (``bayes_tda`` convention;
  use ``btate.topo_posterior.adapters.bd_to_bp`` on TATE/gudhi diagrams first).
* Mixture ``sigmas`` are **variances** (as consumed by ``RGaussianMixture`` /
  ``Posterior`` internally).

The prior places one Gaussian per k-means centroid of the pooled training
points, weighted by the *expected number of features per diagram* falling in
that cluster (so the prior intensity integrates to roughly the mean diagram
cardinality).  The clutter is a small, deliberately diffuse mixture that spreads
mass across the diagram to absorb spurious low-persistence features.

``bayes_tda`` (and ``matplotlib``, its import-time dependency) is imported
lazily so ``import btate`` stays light.
"""
from __future__ import annotations

import numpy as np


def _pool_points(diagrams):
    pts = [np.atleast_2d(np.asarray(d, dtype=float)) for d in diagrams
           if np.asarray(d).size]
    if not pts:
        raise ValueError("no points found in the training diagrams")
    return np.vstack(pts)


def _kmeans(points, k, random_state):
    from scipy.cluster.vq import kmeans2

    k = int(max(1, min(k, np.unique(points, axis=0).shape[0])))
    seed = None if random_state is None else int(
        np.random.default_rng(random_state).integers(0, 2**31 - 1))
    centroids, labels = kmeans2(points, k, minit="++", seed=seed, missing="raise")
    return centroids, labels, k


def elicit_prior_clutter(train_diagrams, n_components: int | None = None,
                         sigma: float | None = None,
                         clutter_n_components: int = 1,
                         clutter_sigma_scale: float = 4.0,
                         clutter_weight_scale: float = 1.0,
                         min_birth: float = 0.0, weight_floor: float = 1e-6,
                         random_state=None):
    """Elicit ``(prior, clutter)`` as ``bayes_tda`` ``RGaussianMixture`` objects.

    Parameters
    ----------
    train_diagrams : list of (n_i, 2) arrays
        Training persistence diagrams in birth--persistence coordinates.
    n_components : int, optional
        Prior mixture size (default ``round(mean cardinality)``).
    sigma : float, optional
        Common component **variance**; default is the within-cluster mean
        squared radius (per axis).
    clutter_n_components : int
        Number of diffuse clutter components (default 1).
    clutter_sigma_scale : float
        Clutter variance = ``clutter_sigma_scale`` x pooled per-axis variance.
    clutter_weight_scale : float
        Scales the (diffuse) clutter mass relative to the mean cardinality.
    min_birth : float
        ``0`` for Rips, ``-inf`` for sublevel/cubical filtrations.
    weight_floor : float
        Lower bound on component weights (avoids empty-cluster zeros).
    random_state : int | np.random.Generator, optional

    Returns
    -------
    (prior, clutter) : tuple of RGaussianMixture
    """
    from bayes_tda.intensities import RGaussianMixture

    diagrams = list(train_diagrams)
    n_diagrams = len(diagrams)
    if n_diagrams == 0:
        raise ValueError("need at least one training diagram")
    points = _pool_points(diagrams)
    cards = np.array([np.atleast_2d(np.asarray(d)).shape[0] if np.asarray(d).size
                      else 0 for d in diagrams], dtype=float)
    mean_card = float(cards.mean())

    if n_components is None:
        n_components = max(1, int(round(mean_card)))

    centroids, labels, k = _kmeans(points, n_components, random_state)

    # prior weights: expected number of features per diagram from each component.
    counts = np.bincount(labels, minlength=k).astype(float)
    prior_weights = np.maximum(counts / n_diagrams, weight_floor)

    # component variance (per axis): within-cluster mean squared radius / 2.
    if sigma is None:
        sq = ((points - centroids[labels]) ** 2).sum(axis=1)
        within = np.zeros(k)
        for j in range(k):
            sel = labels == j
            within[j] = sq[sel].mean() / 2.0 if sel.any() else 0.0
        # floor to a small fraction of the data spread to avoid degenerate spikes.
        spread = points.var(axis=0).mean()
        prior_sigmas = np.maximum(within, 1e-3 * spread + 1e-12)
    else:
        prior_sigmas = np.full(k, float(sigma))

    prior = RGaussianMixture(
        mus=np.atleast_2d(centroids), sigmas=prior_sigmas, weights=prior_weights,
        normalize_weights=False, min_birth=min_birth,
    )

    # clutter: diffuse mixture spreading mass across the diagram.
    c_centroids, _, ck = _kmeans(points, clutter_n_components, random_state)
    data_var = points.var(axis=0).mean()
    clutter_sigmas = np.full(ck, clutter_sigma_scale * data_var + 1e-12)
    clutter_weights = np.full(ck, clutter_weight_scale * mean_card / ck)
    clutter = RGaussianMixture(
        mus=np.atleast_2d(c_centroids), sigmas=clutter_sigmas,
        weights=clutter_weights, normalize_weights=False, min_birth=min_birth,
    )
    return prior, clutter


def sensitivity_analysis(train_diagrams, test_diagrams, sigma_DYO_grid=None,
                         alpha_grid=None, n_components_grid=None,
                         min_birth: float = 0.0, random_state=None,
                         elicit_kwargs=None):
    """Grid sensitivity of the Maroulas posterior to ``sigma_DYO`` / ``alpha`` /
    prior component count.

    For each combination, elicit prior+clutter from ``train_diagrams``, build a
    ``Posterior``, and score held-out ``test_diagrams`` by mean log intensity
    (``Posterior.evaluate_dgms``).  Returns a list of result dicts.
    """
    from bayes_tda.intensities import Posterior

    sigma_DYO_grid = [0.05, 0.1, 0.2] if sigma_DYO_grid is None else sigma_DYO_grid
    alpha_grid = [1.0] if alpha_grid is None else alpha_grid
    n_components_grid = [None] if n_components_grid is None else n_components_grid
    elicit_kwargs = dict(elicit_kwargs or {})

    results = []
    for nc in n_components_grid:
        prior, clutter = elicit_prior_clutter(
            train_diagrams, n_components=nc, min_birth=min_birth,
            random_state=random_state, **elicit_kwargs,
        )
        for sig in sigma_DYO_grid:
            for al in alpha_grid:
                post = Posterior(DYO=list(train_diagrams), prior=prior,
                                 clutter=clutter, sigma_DYO=sig, alpha=al,
                                 min_birth=min_birth)
                scores = post.evaluate_dgms(list(test_diagrams), log=True)
                results.append({
                    "sigma_DYO": sig, "alpha": al,
                    "n_components": prior.mus.shape[0],
                    "mean_test_loglik": float(np.mean(scores)),
                    "expected_cardinality": float(post.lambd),
                })
    return results
