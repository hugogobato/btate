"""Tests for prior/clutter elicitation + sensitivity harness (Task 1.2)."""
import numpy as np
import pytest

bayes_tda = pytest.importorskip("bayes_tda")
from bayes_tda.intensities import Posterior, RGaussianMixture  # noqa: E402

from btate.topo_posterior import elicit_prior_clutter, sensitivity_analysis  # noqa: E402


def _toy_diagrams(n_diagrams=15, seed=0):
    """Diagrams (birth-persistence) with a noise cloud + one persistent loop."""
    rng = np.random.default_rng(seed)
    dgms = []
    for _ in range(n_diagrams):
        noise = np.column_stack([rng.uniform(0, 0.2, 12), rng.uniform(0.0, 0.1, 12)])
        loop = np.array([[0.15, 0.8]]) + 0.02 * rng.standard_normal((1, 2))
        dgms.append(np.vstack([noise, np.abs(loop)]))
    return dgms


def test_elicit_returns_valid_mixtures():
    dgms = _toy_diagrams()
    prior, clutter = elicit_prior_clutter(dgms, min_birth=0.0, random_state=0)
    assert isinstance(prior, RGaussianMixture)
    assert isinstance(clutter, RGaussianMixture)
    assert prior.mus.shape[1] == 2
    assert np.all(prior.weights > 0)
    assert np.all(prior.sigmas > 0)
    assert np.all(clutter.sigmas > 0)


def test_elicited_prior_feeds_posterior():
    dgms = _toy_diagrams()
    prior, clutter = elicit_prior_clutter(dgms, n_components=8, min_birth=0.0,
                                          random_state=0)
    post = Posterior(DYO=dgms, prior=prior, clutter=clutter, sigma_DYO=0.05,
                     alpha=1.0, min_birth=0.0)
    x = np.array([[0.15, 0.8], [0.05, 0.02]])
    dens = post.evaluate(x)
    assert dens.shape == (2,)
    assert np.all(dens >= 0)
    # the persistent-loop location should have higher intensity than the noise.
    assert dens[0] > dens[1]


def test_n_components_controls_prior_size():
    dgms = _toy_diagrams()
    prior, _ = elicit_prior_clutter(dgms, n_components=5, random_state=0)
    assert prior.mus.shape[0] <= 5


def test_sensitivity_analysis_grid():
    dgms = _toy_diagrams()
    train, test = dgms[:10], dgms[10:]
    res = sensitivity_analysis(
        train, test, sigma_DYO_grid=[0.05, 0.1], alpha_grid=[1.0],
        n_components_grid=[5, 8], min_birth=0.0, random_state=0,
    )
    assert len(res) == 4
    for row in res:
        for key in ("sigma_DYO", "alpha", "n_components", "mean_test_loglik",
                    "expected_cardinality"):
            assert key in row
        assert np.isfinite(row["mean_test_loglik"])
