"""Tests for Phase-2 functional embeddings and summaries."""
import numpy as np
import pytest
from gudhi.representations import Landscape, Silhouette

from btate.embeddings import (
    fit_fpca,
    landscape_distances,
    posterior_embedding_summary,
    posterior_landscape,
    project_fourier,
    summarize_posterior_functions,
    weighted_silhouette,
)


def test_power_silhouette_matches_gudhi_tate_weight():
    dgm = [np.array([[0.0, 1.0], [0.2, 0.8], [0.45, 0.55]])]
    ours = weighted_silhouette(
        dgm, weights="power", r=3, sample_range=(0, 1), resolution=25
    )
    gudhi = Silhouette(
        weight=lambda x: abs(x[1] - x[0]) ** 3,
        resolution=25,
        sample_range=[0, 1],
        keep_endpoints=True,
    ).fit_transform(dgm)
    np.testing.assert_allclose(ours, gudhi)


def test_pi_weighted_silhouette_downweights_noise_feature():
    signal = np.array([[0.0, 1.0]])
    noise = np.array([[0.45, 0.55]])
    dgm = np.vstack([signal, noise])

    pi_curve = weighted_silhouette(
        [dgm], weights="pi", pi=[np.array([1.0, 0.0])],
        sample_range=(0, 1), resolution=21,
    )
    signal_only = weighted_silhouette(
        [signal], weights="power", r=0, sample_range=(0, 1), resolution=21
    )
    np.testing.assert_allclose(pi_curve, signal_only)


def test_zero_pi_policy_returns_zero_curve():
    dgm = [np.array([[0.0, 1.0]])]
    curve = weighted_silhouette(
        dgm, weights="pi", pi=[np.array([0.0])], sample_range=(0, 1),
        resolution=5,
    )
    assert np.all(curve == 0.0)


def test_pi_length_must_match_diagram():
    with pytest.raises(ValueError):
        weighted_silhouette(
            [np.array([[0.0, 1.0], [0.2, 0.8]])],
            weights="pi",
            pi=[np.array([1.0])],
        )


def test_posterior_landscape_reshapes_gudhi_output():
    dgm = [np.array([[0.0, 1.0], [0.2, 0.8]])]
    ours, grid = posterior_landscape(
        dgm, num_landscapes=3, sample_range=(0, 1), resolution=11,
        return_grid=True,
    )
    gudhi_transformer = Landscape(
        num_landscapes=3, resolution=11, sample_range=[0, 1],
        keep_endpoints=True,
    )
    gudhi = gudhi_transformer.fit_transform(dgm).reshape(1, 3, 11)
    np.testing.assert_allclose(ours, gudhi)
    np.testing.assert_allclose(grid, gudhi_transformer.grid_)


def test_landscape_distance_is_zero_for_identical_diagrams():
    dgm = np.array([[0.0, 1.0], [0.2, 0.8]])
    dist = landscape_distances(
        dgm, [dgm.copy()], num_landscapes=2, sample_range=(0, 1),
        resolution=20,
    )
    assert dist[0] == pytest.approx(0.0)


def test_posterior_summary_shapes_and_simultaneous_band():
    draws = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 3.0, 0.0],
    ])
    summary = summarize_posterior_functions(draws, grid=np.array([0.0, 0.5, 1.0]))
    assert summary.mean.shape == (3,)
    assert summary.pointwise_lower.shape == (3,)
    assert summary.simultaneous_lower.shape == (3,)
    assert summary.simultaneous_upper[1] >= summary.pointwise_upper[1]


def test_posterior_embedding_summary_for_landscapes():
    diagrams = [
        np.array([[0.0, 1.0]]),
        np.array([[0.0, 1.0], [0.2, 0.7]]),
    ]
    summary = posterior_embedding_summary(
        diagrams, embedding="landscape", num_landscapes=2,
        sample_range=(0, 1), resolution=9,
    )
    assert summary.draws.shape == (2, 2, 9)
    assert summary.mean.shape == (2, 9)


def test_fpca_scores_and_reconstruction_shapes():
    grid = np.linspace(0, 1, 25)
    curves = np.vstack([np.sin(np.pi * grid), 2 * np.sin(np.pi * grid), grid])
    model = fit_fpca(curves, n_components=2)
    scores = model.transform(curves)
    rec = model.inverse_transform(scores)
    assert scores.shape == (3, 2)
    assert rec.shape == curves.shape
    assert np.sum(model.explained_variance_ratio_) <= 1.0 + 1e-12


def test_fourier_projection_shapes():
    grid = np.linspace(0, 1, 20)
    curves = np.vstack([np.sin(2 * np.pi * grid), np.cos(2 * np.pi * grid)])
    proj = project_fourier(curves, grid, n_basis=5)
    assert proj.coefficients_.shape == (2, 5)
    assert proj.reconstruct().shape == curves.shape
