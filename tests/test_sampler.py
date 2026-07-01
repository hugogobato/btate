"""Tests for the posterior persistence-diagram sampler (Task 1.1).

These require the vendored ``bayes_tda`` (on the path via ``conftest.py``); they
skip cleanly where it (or its import-time deps) are unavailable.
"""
import numpy as np
import pytest

bayes_tda = pytest.importorskip("bayes_tda")
from bayes_tda.intensities import Posterior, RGaussianMixture  # noqa: E402
from scipy.stats import norm  # noqa: E402

from btate.topo_posterior import PosteriorDiagramSampler  # noqa: E402


def _simple_posterior(alpha=1.0, min_birth=0.0, seed=0):
    rng = np.random.default_rng(seed)
    mus = np.array([[0.1, 0.5], [0.3, 1.0]], dtype=float)
    sigmas = np.array([0.02, 0.02])
    weights = np.array([1.0, 1.0])
    prior = RGaussianMixture(mus, sigmas, weights, min_birth=min_birth)
    clutter = RGaussianMixture(mus, np.array([0.5, 0.5]), weights,
                               min_birth=min_birth)
    dgms = [np.array([[0.1, 0.5], [0.3, 1.0], [0.2, 0.05]])
            + 0.02 * rng.standard_normal((3, 2)) for _ in range(8)]
    # keep persistence strictly positive
    dgms = [np.column_stack([d[:, 0], np.abs(d[:, 1])]) for d in dgms]
    return Posterior(DYO=dgms, prior=prior, clutter=clutter, sigma_DYO=0.05,
                     alpha=alpha, min_birth=min_birth)


def test_mixture_reproduces_evaluate_pointwise():
    """The assembled sampling mixture equals posterior.evaluate on the wedge."""
    post = _simple_posterior()
    s = PosteriorDiagramSampler(post)
    coeff = s.component_probs * s.mixture_mass
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.uniform(0, 0.5, 200), rng.uniform(1e-3, 1.5, 200)])
    d2 = ((X[:, None, :] - s.means[None, :, :]) ** 2).sum(-1)
    g = np.exp(-0.5 * d2 / s.variances[None, :]) / (2 * np.pi * s.variances[None, :])
    mine = (g * coeff[None, :]).sum(1)
    ev = post.evaluate(X)
    np.testing.assert_allclose(mine, ev, rtol=1e-9, atol=1e-12)


def test_samples_lie_in_wedge():
    post = _simple_posterior(min_birth=0.0)
    s = PosteriorDiagramSampler(post)
    for d in s.sample_diagrams(50, random_state=2):
        if len(d):
            assert np.all(d[:, 0] >= 0.0)
            assert np.all(d[:, 1] > 0.0)


def test_cardinality_matches_lambd():
    post = _simple_posterior()
    s = PosteriorDiagramSampler(post)
    draws = s.sample_diagrams(2000, random_state=3, count="poisson")
    mean_card = np.mean([len(d) for d in draws])
    assert mean_card == pytest.approx(post.lambd, rel=0.08)


def test_fixed_count():
    post = _simple_posterior()
    s = PosteriorDiagramSampler(post)
    draws = s.sample_diagrams(5, random_state=4, count="fixed", cardinality=7)
    assert all(len(d) == 7 for d in draws)


def test_sample_moments_match_analytic_truncated_mixture():
    """Grid-free: sample means match the mixture's truncated-Gaussian moments."""
    post = _simple_posterior()
    s = PosteriorDiagramSampler(post)
    coeff = s.component_probs * s.mixture_mass
    mub, mup = s.means[:, 0], s.means[:, 1]
    sd = np.sqrt(s.variances)
    Pb = 1 - norm.cdf((s.min_birth - mub) / sd)
    Pp0 = 1 - norm.cdf((0.0 - mup) / sd)
    wmass = coeff * Pb * Pp0
    W = wmass.sum()

    def trunc_mean(mu, lo):
        a = (lo - mu) / sd
        return mu + sd * norm.pdf(a) / (1 - norm.cdf(a))

    mean_b = (wmass * trunc_mean(mub, s.min_birth)).sum() / W
    mean_p = (wmass * trunc_mean(mup, 0.0)).sum() / W

    pts = np.vstack([d for d in s.sample_diagrams(4000, random_state=5) if len(d)])
    assert pts[:, 0].mean() == pytest.approx(mean_b, abs=0.01)
    assert pts[:, 1].mean() == pytest.approx(mean_p, abs=0.01)


def test_alpha_gt_one_drops_prior_with_warning():
    post = _simple_posterior(alpha=1.5)
    with pytest.warns(RuntimeWarning):
        PosteriorDiagramSampler(post, include_prior=True)
