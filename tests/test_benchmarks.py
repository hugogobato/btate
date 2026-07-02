"""Phase-4 benchmark harness tests (Task 4.1 / 4.4).

Kept fast: tiny sample sizes, the dependency-light ``jitter`` Step-1 surrogate,
and small MCMC settings so the suite runs in a minimal environment.
"""
from __future__ import annotations

import numpy as np
import pytest

from btate.benchmarks import (
    bias, integrated_bias, max_abs_error, rmse, simultaneous_coverage,
)
from btate.benchmarks.synthetic import (
    SyntheticConfig, generate_synthetic_dataset, montecarlo_reference, reference_effect,
)
from btate.benchmarks.frequentist import aipw_effect, function_on_scalar_ridge
from btate.benchmarks.pipeline import (
    PipelineConfig, run_bayesian_pipeline, silhouette_embedding_fn, h1_diagram,
    resolve_sigma_dyo,
)
from btate.benchmarks.harness import SweepCell, evaluate_run, run_cell, sweep_to_rows
from btate.benchmarks.ablation import full_ablation_grid, weighting_ablation
from btate.benchmarks.maroulas_diagnostics import (
    maroulas_sigma_sensitivity,
    pre_fgp_maroulas_diagnostic,
    strip_diagnostic_arrays,
)
from btate.benchmarks.joint_calibration import (
    JointCalibrationCell,
    run_joint_calibration,
)


def _fast_pipe(**kw) -> PipelineConfig:
    base = dict(
        embedding="silhouette", weights="pi", topo_method="jitter",
        posterior_draws=3, resolution=24, propagation="nested",
        n_causal_draws=12, n_plugin_draws=40, n_inducing=14,
        partition_samples=20, partition_burn_in=20, seed=0,
    )
    base.update(kw)
    return PipelineConfig(**base)


# --------------------------------------------------------------- metrics
def test_metric_helpers():
    truth = np.array([0.0, 1.0, 2.0])
    est = np.array([0.5, 1.0, 1.0])
    tseq = np.array([0.0, 0.5, 1.0])
    assert bias(est, truth) == pytest.approx((0.5 + 0.0 - 1.0) / 3.0)
    assert max_abs_error(est, truth) == pytest.approx(1.0)
    assert rmse(est, truth) > 0
    assert isinstance(integrated_bias(est, truth, tseq), float)
    assert simultaneous_coverage([0.0, 0.9, 1.9], [0.6, 1.1, 2.1], truth) == 1.0
    assert simultaneous_coverage([0.0, 0.9, 1.5], [0.6, 1.1, 1.9], truth) == 0.0


# --------------------------------------------------------------- DGP
def test_synthetic_dataset_shapes_and_effect_preserved():
    cfg = SyntheticConfig(n=10, num_pts=80, effect_size=0.12, noise_level=1.0, seed=1)
    ds = generate_synthetic_dataset(cfg)
    assert ds.clouds.shape[0] == 10 and ds.clouds.shape[1] == 2
    assert ds.clean_clouds.shape[:2] == (10, 2)
    assert len(np.unique(ds.A)) == 2
    # Treated loop dies later than control on the clean clouds (effect present).
    d0 = h1_diagram(ds.clean_clouds[0, 0])
    d1 = h1_diagram(ds.clean_clouds[0, 1])
    assert d1[:, 1].max() > d0[:, 1].max()


def test_reference_and_montecarlo_reference():
    cfg = SyntheticConfig(n=8, num_pts=70, effect_size=0.12, noise_level=1.0, seed=2)
    pipe = _fast_pipe()
    ds = generate_synthetic_dataset(cfg)
    fn = silhouette_embedding_fn(pipe, (0.0, 0.6))
    clean = reference_effect(ds, fn, None)
    mc = montecarlo_reference(cfg, fn, n_realizations=4)
    assert clean.shape == mc.shape == (pipe.resolution,)
    # Determinism: same seeds -> identical MC reference.
    mc2 = montecarlo_reference(cfg, fn, n_realizations=4)
    assert np.allclose(mc, mc2)


def test_noise_seed_separates_structure_from_noise():
    from dataclasses import replace
    a = generate_synthetic_dataset(SyntheticConfig(n=6, seed=3, noise_seed=1))
    b = generate_synthetic_dataset(SyntheticConfig(n=6, seed=3, noise_seed=2))
    # Same structural stream -> identical covariates/treatment...
    assert np.allclose(a.X, b.X) and np.array_equal(a.A, b.A)
    # ...but different noise -> different observed clouds.
    assert not np.allclose(a.clouds, b.clouds)


# --------------------------------------------------------------- frequentist
def test_function_on_scalar_ridge_predicts():
    rng = np.random.default_rng(0)
    grid = np.linspace(0, 1, 20)
    X = rng.normal(size=(30, 2))
    phi = (X[:, [0]] * np.sin(np.pi * grid)[None, :]) + rng.normal(0, 0.01, (30, 20))
    pred = function_on_scalar_ridge(phi, X, grid, n_basis=5)
    out = pred(X[:3])
    assert out.shape == (3, 20)


def test_aipw_effect_bands_and_shapes():
    rng = np.random.default_rng(1)
    grid = np.linspace(0, 0.6, 24)
    n = 40
    X = rng.normal(size=(n, 2))
    A = rng.binomial(1, 0.5, size=n)
    tau = 0.1 * np.exp(-0.5 * ((grid - 0.3) / 0.1) ** 2)
    phi = (A[:, None] * tau[None, :]) + rng.normal(0, 0.02, (n, 24))
    eff = aipw_effect(phi, A, X, grid, pi_hat=np.full(n, 0.5), random_state=0)
    assert eff.estimate.shape == (24,)
    assert np.all(eff.simultaneous_upper >= eff.pointwise_upper - 1e-9)
    assert np.all(eff.simultaneous_lower <= eff.pointwise_lower + 1e-9)


# --------------------------------------------------------------- pipeline / harness
def test_run_bayesian_pipeline_end_to_end():
    cfg = SyntheticConfig(n=10, num_pts=80, effect_size=0.12, noise_level=1.0, seed=4)
    ds = generate_synthetic_dataset(cfg)
    res = run_bayesian_pipeline(ds.observed_clouds(), ds.A, ds.X, ds.pi, _fast_pipe())
    assert res.nested is not None and res.plugin is not None
    assert res.phi_draws.shape[1] == 10
    assert res.nested.draws.shape[1] == res.grid.shape[0]
    assert res.timing["total_s"] >= 0.0


def test_adaptive_sigma_dyo_uses_prior_variance_scale():
    pytest.importorskip("bayes_tda")
    from bayes_tda.intensities import RGaussianMixture

    prior = RGaussianMixture(
        mus=np.array([[0.1, 0.2], [0.2, 0.3]]),
        sigmas=np.array([1e-4, 2e-4]),
        weights=np.array([1.0, 1.0]),
        normalize_weights=False,
    )
    info = resolve_sigma_dyo(
        prior,
        PipelineConfig(sigma_dyo=None, sigma_dyo_multiplier=3.0),
    )
    assert info["sigma_dyo_mode"] == "adaptive_median_prior_sigma"
    assert info["sigma_dyo"] == pytest.approx(4.5e-4)

    fixed = resolve_sigma_dyo(prior, PipelineConfig(sigma_dyo=0.01))
    assert fixed["sigma_dyo_mode"] == "fixed"
    assert fixed["sigma_dyo"] == pytest.approx(0.01)


def test_pre_fgp_maroulas_diagnostic_runs_tiny():
    pytest.importorskip("bayes_tda")
    synth = SyntheticConfig(n=6, num_pts=45, effect_size=0.10, noise_level=0.8, seed=10)
    pipe = _fast_pipe(
        topo_method="maroulas",
        weights="power",
        posterior_draws=2,
        resolution=16,
        sigma_dyo=None,
        sigma_dyo_multiplier=3.0,
    )
    row = pre_fgp_maroulas_diagnostic(synth, pipe, sigma_multiplier=3.0)
    for key in (
        "observed_effect_l2",
        "maroulas_effect_l2",
        "l2_attenuation_ratio",
        "rmse_to_observed_effect",
        "sigma_dyo_median",
        "flag_attenuated",
    ):
        assert key in row
    assert row["_grid"].shape == row["_observed_effect"].shape
    assert np.isfinite(row["sigma_dyo_median"])


def test_maroulas_sigma_sensitivity_strips_arrays():
    pytest.importorskip("bayes_tda")
    synth = SyntheticConfig(n=6, num_pts=45, effect_size=0.10, noise_level=0.8, seed=11)
    pipe = _fast_pipe(
        topo_method="maroulas",
        weights="power",
        posterior_draws=1,
        resolution=12,
        sigma_dyo=None,
    )
    rows = maroulas_sigma_sensitivity(
        synth, pipe, sigma_multipliers=(1.0, 3.0), prior_variants=("pooled",),
    )
    compact = strip_diagnostic_arrays(rows)
    assert len(compact) == 2
    assert all("sigma_setting" in row for row in compact)
    assert all(not any(k.startswith("_") for k in row) for row in compact)


def test_joint_calibration_runs_tiny():
    pytest.importorskip("bayes_tda")
    synth = SyntheticConfig(n=6, num_pts=45, effect_size=0.10, noise_level=0.8, seed=12)
    pipe = _fast_pipe(
        topo_method="maroulas",
        weights="power",
        posterior_draws=1,
        resolution=12,
        propagation="nested",
        n_causal_draws=4,
        n_plugin_draws=8,
        n_inducing=6,
        sigma_dyo=None,
    )
    cell = JointCalibrationCell(
        "tiny",
        synth,
        pipe,
        n_reps=1,
        sigma_multipliers=(1.0,),
        fgp_scales=(2.0, 4.0),
        mc_realizations=2,
    )
    summary, raw = run_joint_calibration(cell, n_jobs=1)
    assert len(raw) == 2
    assert len(summary) == 2
    assert {row["fgp_posterior_scale"] for row in summary} == {2.0, 4.0}
    assert all("topo_l2_attenuation_ratio" in row for row in summary)


def test_plugin_only_propagation():
    cfg = SyntheticConfig(n=10, num_pts=80, seed=5)
    ds = generate_synthetic_dataset(cfg)
    res = run_bayesian_pipeline(ds.observed_clouds(), ds.A, ds.X, ds.pi,
                                _fast_pipe(propagation="plugin"))
    assert res.nested is None and res.plugin is not None


def test_evaluate_run_record_keys():
    synth = SyntheticConfig(n=10, num_pts=80, effect_size=0.12, seed=6)
    rec = evaluate_run(synth, _fast_pipe(), run_frequentist=True,
                       freq_methods=("multiplier_bootstrap", "liebl_reimherr"))
    for key in ("bayes_rmse", "bayes_cov_simultaneous", "bayes_width",
                "bayes_reject", "freq_rmse", "total_s",
                "freq_multiplier_bootstrap_reject",
                "freq_multiplier_bootstrap_cov_simultaneous",
                "freq_liebl_reimherr_width"):
        assert key in rec
    assert isinstance(rec["bayes_reject"], bool)


def test_run_cell_aggregates():
    synth = SyntheticConfig(n=8, num_pts=70, effect_size=0.12, seed=7)
    cell = SweepCell("t", synth, _fast_pipe(), n_reps=2, run_frequentist=False)
    agg = run_cell(cell)
    assert agg["n_reps"] == 2
    rows = sweep_to_rows([agg])
    assert "_records" not in rows[0]


def test_landscape_embedding_runs():
    synth = SyntheticConfig(n=8, num_pts=70, effect_size=0.12, seed=8)
    rec = evaluate_run(synth, _fast_pipe(embedding="landscape"), run_frequentist=False)
    assert np.isfinite(rec["bayes_rmse"])


def test_ablation_builders():
    synth = SyntheticConfig(n=8, seed=9)
    pipe = _fast_pipe()
    cells = weighting_ablation(synth, pipe, n_reps=1)
    assert {c.name for c in cells} == {"silhouette_pi", "silhouette_fixed_r"}
    grid = full_ablation_grid(synth, pipe, n_reps=1)
    names = {c.name for c in grid}
    assert {"silhouette_pi", "silhouette_fixed_r", "landscape", "nested", "plugin"} <= names


# --------------------------------------------------------------- FGP posterior_scale
def test_posterior_scale_widens_bands():
    from btate.causal import FunctionalGPEstimator
    rng = np.random.default_rng(0)
    grid = np.linspace(0, 1, 24)
    n = 24
    X = rng.normal(size=(n, 2))
    A = rng.binomial(1, 0.5, size=n)
    phi = (A[:, None] * 0.2) + rng.normal(0, 0.05, (n, 24))
    widths = []
    for ps in (1.0, 16.0):
        est = FunctionalGPEstimator(n_inducing=16, posterior_scale=ps)
        est.fit(phi, A, X=X, tseq=grid, random_state=0)
        eff = est.posterior_tate(n_draws=200, random_state=1)
        widths.append(np.mean(eff.simultaneous_upper - eff.simultaneous_lower))
    assert widths[1] > widths[0]
