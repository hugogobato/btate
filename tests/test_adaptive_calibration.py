"""Tests for the Phase-4.25 adaptive calibration machinery.

Covers the two knobs that were previously hand-tuned against benchmark
coverage and are now data-driven:

* ``sigma_DYO`` selected by marked-PPP marginal likelihood (empirical Bayes,
  ``btate.topo_posterior.eb``);
* ``fgp_posterior_scale`` replaced by the cluster-robust Godambe/sandwich
  coefficient covariance (``FunctionalGPEstimator(posterior_scale="godambe")``).
"""
import numpy as np
import pytest

from btate.causal import FunctionalGPEstimator


# ---------------------------------------------------------------------------
# Empirical-Bayes sigma_DYO selection
# ---------------------------------------------------------------------------

def _toy_prior_clutter():
    bayes_tda = pytest.importorskip("bayes_tda")
    from bayes_tda.intensities import RGaussianMixture

    prior = RGaussianMixture(
        mus=np.array([[0.30, 0.20], [0.55, 0.40]]),
        sigmas=np.array([0.002, 0.002]),          # per-axis variances
        weights=np.array([3.0, 3.0]),             # expected counts
        normalize_weights=False, min_birth=0.0,
    )
    clutter = RGaussianMixture(
        mus=np.array([[0.4, 0.2]]),
        sigmas=np.array([0.5]),
        weights=np.array([0.5]),
        normalize_weights=False, min_birth=0.0,
    )
    return prior, clutter


def _noisy_diagrams(prior, noise_var, n_diagrams=12, per_component=3, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_diagrams):
        pts = []
        for mu in prior.mus:
            pts.append(mu + rng.normal(0.0, np.sqrt(noise_var),
                                       size=(per_component, 2)))
        d = np.vstack(pts)
        d[:, 1] = np.abs(d[:, 1])                 # keep persistence positive
        out.append(d)
    return out


def test_eb_sigma_dyo_tracks_observation_noise():
    prior, clutter = _toy_prior_clutter()
    from btate.topo_posterior import select_sigma_dyo

    low = select_sigma_dyo(
        _noisy_diagrams(prior, noise_var=5e-4, seed=1), prior, clutter)
    high = select_sigma_dyo(
        _noisy_diagrams(prior, noise_var=1e-2, seed=2), prior, clutter)

    assert np.all(np.isfinite(low["profile_loglik"]))
    assert high["sigma_dyo"] > low["sigma_dyo"]
    assert low["sigma_dyo"] > 0.0
    # profile grid and selection are consistent
    best = np.argmax(low["profile_loglik"])
    assert low["sigma_dyo"] == pytest.approx(low["profile_sigma_dyo"][best])


def test_resolve_sigma_dyo_eb_mode():
    prior, clutter = _toy_prior_clutter()
    from btate.benchmarks.pipeline import PipelineConfig, resolve_sigma_dyo

    diagrams = _noisy_diagrams(prior, noise_var=2e-3, seed=3)
    cfg = PipelineConfig(sigma_dyo=None, sigma_dyo_multiplier="eb")
    info = resolve_sigma_dyo(prior, cfg, diagrams_bp=diagrams, clutter=clutter)
    assert info["sigma_dyo_mode"] == "empirical_bayes_marginal_likelihood"
    assert info["sigma_dyo"] > 0.0
    assert np.isfinite(info["sigma_dyo_multiplier"])
    assert "sigma_dyo_at_boundary" in info

    with pytest.raises(ValueError):
        resolve_sigma_dyo(prior, cfg)             # diagrams/clutter required


# ---------------------------------------------------------------------------
# Godambe / sandwich FGP posterior scale
# ---------------------------------------------------------------------------

def _curve_data(n=40, res=25, intercept_sd=0.0, noise_sd=0.05, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    A = rng.binomial(1, 0.5, size=n)
    if len(np.unique(A)) < 2:
        A[: n // 2], A[n // 2:] = 0, 1
    t = np.linspace(0.0, 1.0, res)
    base = np.sin(np.pi * t)[None, :]
    curves = (
        0.2 * base
        + 0.1 * A[:, None] * base
        + 0.05 * X[:, [0]] * base
        + rng.normal(0.0, intercept_sd, size=(n, 1))   # within-curve correlation
        + rng.normal(0.0, noise_sd, size=(n, res))
    )
    return curves, A, X, t


def _fit_scale_hat(intercept_sd, seed=11):
    curves, A, X, t = _curve_data(intercept_sd=intercept_sd, seed=seed)
    est = FunctionalGPEstimator(
        n_inducing=16, posterior_scale="godambe", prior_scale=5.0,
    )
    est.fit(curves, A, X=X, tseq=t, random_state=0)
    return est


def test_godambe_scale_grows_with_within_curve_correlation():
    est_iid = _fit_scale_hat(intercept_sd=0.0)
    est_corr = _fit_scale_hat(intercept_sd=0.3)
    assert np.isfinite(est_iid.posterior_scale_hat_)
    assert np.isfinite(est_corr.posterior_scale_hat_)
    assert est_corr.posterior_scale_hat_ > 2.0 * est_iid.posterior_scale_hat_


def test_godambe_widens_bands_versus_naive():
    curves, A, X, t = _curve_data(intercept_sd=0.3, seed=7)
    naive = FunctionalGPEstimator(n_inducing=16, posterior_scale=1.0,
                                  prior_scale=5.0)
    robust = FunctionalGPEstimator(n_inducing=16, posterior_scale="godambe",
                                   prior_scale=5.0)
    naive.fit(curves, A, X=X, tseq=t, random_state=0)
    robust.fit(curves, A, X=X, tseq=t, random_state=0)
    e_naive = naive.posterior_tate(n_draws=400, random_state=1)
    e_robust = robust.posterior_tate(n_draws=400, random_state=1)
    w_naive = float(np.mean(e_naive.simultaneous_upper - e_naive.simultaneous_lower))
    w_robust = float(np.mean(e_robust.simultaneous_upper - e_robust.simultaneous_lower))
    assert w_robust > w_naive
    assert e_robust.metadata["posterior_scale"] == "godambe"
    assert e_robust.metadata["posterior_scale_hat"] > 1.0


def test_fixed_scale_metadata_reports_nan_hat():
    curves, A, X, t = _curve_data(seed=5)
    est = FunctionalGPEstimator(n_inducing=16, posterior_scale=8.0,
                                prior_scale=5.0)
    est.fit(curves, A, X=X, tseq=t, random_state=0)
    effect = est.posterior_tate(n_draws=50, random_state=1)
    assert effect.metadata["posterior_scale"] == 8.0
    assert np.isnan(effect.metadata["posterior_scale_hat"])


def test_posterior_scale_validation():
    with pytest.raises(ValueError):
        FunctionalGPEstimator(posterior_scale="sandwich")
    with pytest.raises(ValueError):
        FunctionalGPEstimator(posterior_scale=-1.0)


def test_aggregate_joint_records_mixed_scale_types():
    from btate.benchmarks.joint_calibration import aggregate_joint_records

    base = {"cell": "c", "n": 10, "noise_level": 1.0, "effect_size": 0.1,
            "overlap_strength": 0.8, "sigma_setting": "eb"}
    records = [
        dict(base, rep=r, fgp_posterior_scale=s, bayes_rmse=0.1 * (r + 1))
        for r in range(2) for s in (16.0, "godambe")
    ]
    rows = aggregate_joint_records(records)
    assert len(rows) == 2
    assert {row["fgp_posterior_scale"] for row in rows} == {16.0, "godambe"}
    assert all(row["n_reps"] == 2 for row in rows)


# ---------------------------------------------------------------------------
# End-to-end pipeline with both adaptive modes
# ---------------------------------------------------------------------------

def test_pipeline_runs_with_eb_sigma_and_godambe_scale():
    pytest.importorskip("bayes_tda")
    gd = pytest.importorskip("gudhi")  # noqa: F841
    from btate.benchmarks.pipeline import PipelineConfig, run_bayesian_pipeline
    from btate.benchmarks.synthetic import SyntheticConfig, generate_synthetic_dataset

    synth = SyntheticConfig(n=8, num_pts=40, noise_level=1.0,
                            effect_size=0.12, seed=99)
    dataset = generate_synthetic_dataset(synth)
    cfg = PipelineConfig(
        embedding="silhouette", weights="power", topo_method="maroulas",
        posterior_draws=3, resolution=24, sigma_dyo=None,
        sigma_dyo_multiplier="eb", fgp_posterior_scale="godambe",
        propagation="nested", n_causal_draws=8, n_plugin_draws=40,
        n_inducing=12, seed=99,
    )
    result = run_bayesian_pipeline(
        dataset.observed_clouds(), dataset.A, dataset.X, dataset.pi, cfg,
    )
    meta = result.meta
    assert meta["sigma_dyo_mode"] == "empirical_bayes_marginal_likelihood"
    assert meta["sigma_dyo"] > 0.0
    assert meta["fgp_posterior_scale"] == "godambe"
    hat = result.nested.metadata["posterior_scale_hat_mean"]
    assert np.isfinite(hat) and hat > 0.0
