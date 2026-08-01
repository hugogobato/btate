"""Phase-6.25 regression and mathematical-validity tests."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import Bounds, LinearConstraint, minimize

pytest.importorskip("bayes_tda")
from bayes_tda.intensities import Posterior, RGaussianMixture  # noqa: E402

from btate.benchmarks import controlled_comparisons as ctrl  # noqa: E402
from btate.benchmarks.measurement_error_uq import (  # noqa: E402
    MEUQConfig,
    aggregate_me_uq,
)
from btate.embeddings.silhouette import weighted_silhouette  # noqa: E402


def test_unconditional_constraints_accept_out_of_grid_death():
    grid = np.linspace(0.0, 0.848, 96)
    poly = ctrl.silhouette_polyhedron(grid)
    curve = weighted_silhouette(
        [np.array([[0.0, 2.0]])], weights="power", r=3.0,
        sample_range=(0.0, 0.848), resolution=96,
    )[0]
    assert curve[-1] > 1.0  # old height/right-endpoint rules rejected this.
    stats = ctrl.cone_violation_stats(curve, poly, tol=1e-12)
    assert stats["viol_any_rate"] == 0.0
    assert ctrl.derive_lipschitz_constant() == pytest.approx(np.sqrt(2.0))

    too_small = ctrl.silhouette_polyhedron(
        grid, lipschitz_constant=0.99 * np.sqrt(2.0))
    assert ctrl.cone_violation_stats(curve, too_small, tol=1e-12)[
        "viol_c2_rate"] == 1.0


def test_validity_diagnostic_detects_each_constraint():
    poly = ctrl.silhouette_polyhedron(np.linspace(0.0, 1.0, 11))
    curves = np.zeros((3, 11))
    curves[0, 4] = -0.1             # C1
    curves[1, 5:] = 1.0             # C2
    curves[2] = 0.1                 # C3 at t=0
    stats = ctrl.cone_violation_stats(curves, poly)
    assert stats["viol_c1_rate"] == pytest.approx(1 / 3)
    assert stats["viol_c2_rate"] == pytest.approx(1 / 3)
    assert stats["viol_c3_rate"] == pytest.approx(1 / 3)
    assert stats["viol_any_rate"] == 1.0
    with pytest.raises(ValueError, match="finite"):
        ctrl.cone_violation_stats(np.full((1, 11), np.nan), poly)
    with pytest.raises(ValueError, match="last draw dimension"):
        ctrl.cone_violation_stats(np.zeros((2, 5)), poly)
    with pytest.raises(ValueError, match="last curve dimension"):
        ctrl.project_to_cone(np.zeros((2, 5)), poly)
    with pytest.raises(ValueError, match="strictly increasing"):
        ctrl.silhouette_polyhedron(np.linspace(1.0, 0.0, 11))


def _slsqp_projection(y, poly):
    m = len(y)
    D = np.zeros((m - 1, m))
    D[np.arange(m - 1), np.arange(m - 1)] = -1.0
    D[np.arange(m - 1), np.arange(1, m)] = 1.0
    lower = np.zeros(m)
    upper = np.full(m, np.inf)
    upper[0] = 0.0
    result = minimize(
        lambda x: 0.5 * np.sum((x - y) ** 2),
        ctrl.project_to_cone(y, poly),
        jac=lambda x: x - y,
        method="SLSQP",
        bounds=Bounds(lower, upper),
        constraints=[LinearConstraint(D, -poly.slope, poly.slope)],
        options={"ftol": 1e-12, "maxiter": 5000},
    )
    assert result.success, result.message
    return result.x


def test_hildreth_projection_matches_slsqp_and_is_idempotent():
    rng = np.random.default_rng(20260731)
    poly = ctrl.silhouette_polyhedron(np.linspace(0.0, 0.848, 24))
    curves = rng.normal(0.0, 0.25, size=(8, 24))
    projected = ctrl.project_to_cone(
        curves, poly, viol_tol=1e-10, update_tol=1e-12)
    for y, got in zip(curves, projected):
        expected = _slsqp_projection(y, poly)
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-8)

    assert ctrl.cone_violation_stats(projected, poly)["viol_any_rate"] == 0.0
    assert np.array_equal(projected, ctrl.project_to_cone(projected, poly))
    feasible = 0.01 * np.sin(np.pi * poly.grid / poly.grid_upper) ** 2
    feasible[0] = 0.0
    assert np.array_equal(feasible, ctrl.project_to_cone(feasible, poly))


def _posterior_inputs():
    prior = RGaussianMixture(
        mus=np.array([[0.12, 0.25], [0.35, 0.50], [0.55, 0.08]]),
        sigmas=np.array([0.015, 0.03, 0.01]),
        weights=np.array([0.8, 1.2, 0.4]),
        normalize_weights=False, min_birth=0.0,
    )
    clutter = RGaussianMixture(
        mus=np.array([[0.25, 0.05]]), sigmas=np.array([0.12]),
        weights=np.array([1.0]), normalize_weights=False, min_birth=0.0,
    )
    diagram = np.array([[0.11, 0.30], [0.31, 0.42], [0.50, 0.06]])
    step = ctrl.TopoStep1(
        prior=prior, clutter=clutter, sigma_dyo=0.007, alpha=0.75,
        min_birth=0.0, sample_range=(0.0, 0.848), resolution=96, r=3.0,
        n_calibration_curves=1,
    )
    return step, diagram


def test_fast_subject_posterior_matches_vendored_posterior():
    step, diagram = _posterior_inputs()
    expected = Posterior(
        DYO=[diagram], prior=step.prior, clutter=step.clutter,
        sigma_DYO=step.sigma_dyo, alpha=step.alpha, min_birth=0.0,
    )
    got = ctrl.fast_subject_posterior(step, diagram)
    for name in ("posterior_means", "posterior_sigmas", "Cs"):
        np.testing.assert_allclose(
            getattr(got, name), getattr(expected, name), rtol=1e-14, atol=1e-14)
    assert got.lambd == expected.lambd


def test_paired_calibration_selects_on_clean_holdout_and_refits(monkeypatch):
    rng = np.random.default_rng(7)
    observed, clean, subject_ids = [], [], []
    for subject in range(8):
        latent = np.array([[0.10, 0.62]]) + rng.normal(0.0, 0.005, (1, 2))
        latent[:, 1] = np.maximum(latent[:, 1], latent[:, 0] + 0.1)
        noisy = np.vstack([
            latent + rng.normal(0.0, 0.015, (1, 2)),
            np.array([[0.30, 0.33]]) + rng.normal(0.0, 0.003, (1, 2)),
        ])
        noisy[:, 1] = np.maximum(noisy[:, 1], noisy[:, 0] + 1e-3)
        observed.append(noisy)
        clean.append(latent)
        subject_ids.append(subject)

    monkeypatch.setattr(
        ctrl, "_calibration_paired_diagrams",
        lambda cfg: (observed, clean, np.asarray(subject_ids)),
    )
    monkeypatch.setattr(ctrl, "_prior_component_grid", lambda diagrams: (1, 2))
    monkeypatch.setattr(ctrl, "M6_ALPHA_GRID", (0.5, 1.0))
    monkeypatch.setattr(ctrl, "M6_CLUTTER_SCALE_GRID", (0.5, 1.0))
    monkeypatch.setattr(ctrl, "M6_SIGMA_MULTIPLIER_GRID", (0.1, 1.0))

    cfg = MEUQConfig(
        n_subjects=8, resolution=16, n_calibration_subjects=8,
        truth_subjects=8, truth_datasets=1, n_measurement_draws=2,
        n_causal_draws=2, n_plugin_draws=4, n_boot=20,
    )
    grid = SimpleNamespace(sample_range=(0.0, 1.0), resolution=16, r=3.0)
    step = ctrl.fit_topo_step1(cfg, grid)
    sel = step.selection
    assert sel["uses_paired_clean_diagrams"] is True
    assert sel["arm_labels_used"] is False
    assert sel["refit_on_full_calibration_sample"] is True
    assert set(sel["fit_subject_ids"]).isdisjoint(sel["holdout_subject_ids"])
    assert sel["n_candidates"] == 16
    assert step.n_calibration_curves == len(observed)
    # The full clean sample has one feature per diagram; the prior intensity
    # must reflect that rather than the two-feature observed cardinality.
    assert np.sum(step.prior.weights) == pytest.approx(1.0)


def test_aggregate_reports_global_worst_cone_magnitude():
    raw = pd.DataFrame([
        {"noise_level": 0.1, "method": "m", "rep": 0, "failed": False,
         "cone_m4_viol_c1_rate": 0.1, "cone_m4_viol_c1_mag": 1.0},
        {"noise_level": 0.1, "method": "m", "rep": 1, "failed": False,
         "cone_m4_viol_c1_rate": 0.3, "cone_m4_viol_c1_mag": 4.0},
    ])
    summary = aggregate_me_uq(raw).iloc[0]
    assert summary["cone_m4_viol_c1_rate_mean"] == pytest.approx(0.2)
    assert summary["cone_m4_viol_c1_mag_max"] == 4.0
    assert "cone_m4_viol_c1_mag_mean" not in summary.index
