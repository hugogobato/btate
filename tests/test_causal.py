"""Tests for Phase-3 Bayesian causal modeling."""
import numpy as np

from btate.causal import (
    FunctionalGPEstimator,
    bayesian_no_effect_test,
    compare_propagation,
    make_tsbcf_long_data,
)


def _synthetic_sample(n=28, m=31, seed=7):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, m)
    x = np.linspace(-1.0, 1.0, n)[:, None]
    A = np.arange(n) % 2
    baseline = 0.15 * np.sin(np.pi * t)[None, :] + 0.08 * x * t[None, :]
    tau = 0.18 + 0.12 * np.sin(np.pi * t)
    y = baseline + A[:, None] * tau[None, :]
    y += rng.normal(0.0, 0.01, size=y.shape)
    return y, A, x, t, tau


def test_functional_gp_posterior_tate_shapes_and_signal():
    phi, A, X, t, _ = _synthetic_sample()
    est = FunctionalGPEstimator(
        n_inducing=18, prior_scale=1.5, noise_variance=0.002
    )
    est.fit(phi, A, X=X, tseq=t, pi_hat=np.full(A.shape[0], 0.5), random_state=1)
    effect = est.posterior_tate(n_draws=80, alpha=0.1, random_state=2)

    assert effect.draws.shape == (80, t.shape[0])
    assert effect.mean.shape == t.shape
    assert effect.pointwise_lower.shape == t.shape
    assert effect.simultaneous_upper.shape == t.shape
    assert float(np.mean(effect.mean)) > 0.08
    assert effect.metadata["mode"] == "joint"


def test_functional_gp_fpca_mode_returns_curve_draws():
    phi, A, X, t, _ = _synthetic_sample(n=24, m=25)
    est = FunctionalGPEstimator(
        n_inducing=8, use_fpca=True, fpca_components=3,
        prior_scale=1.0, noise_variance=0.002,
    )
    est.fit(phi, A, X=X, tseq=t, random_state=3)
    effect = est.posterior_tate(n_draws=40, random_state=4)
    assert effect.draws.shape == (40, t.shape[0])
    assert effect.metadata["mode"] == "fpca"


def test_potential_outcome_axis_is_selected_when_requested():
    control, A, X, t, tau = _synthetic_sample(n=20, m=21)
    potential = np.stack([control, control + tau[None, :]], axis=1)
    est = FunctionalGPEstimator(n_inducing=10, noise_variance=0.002)
    est.fit(potential, A, X=X, tseq=t, potential_outcomes=True, random_state=5)
    effect = est.posterior_tate(n_draws=30, random_state=6)
    assert effect.draws.shape == (30, t.shape[0])
    assert float(np.mean(effect.mean)) > 0.08


def test_bayesian_no_effect_test_positive_draws():
    draws = np.ones((20, 5)) * 0.2
    out = bayesian_no_effect_test(draws, rope=0.05)
    assert out["pr_all_positive"] == 1.0
    assert out["pr_same_sign_excluding_rope"] == 1.0
    assert out["pr_supnorm_gt_rope"] == 1.0


def test_nested_vs_plugin_propagation_contract():
    phi, A, X, t, _ = _synthetic_sample(n=18, m=19)
    rng = np.random.default_rng(9)
    topo_draws = np.stack([
        phi + rng.normal(0.0, 0.02, size=phi.shape)
        for _ in range(3)
    ])
    est = FunctionalGPEstimator(n_inducing=8, noise_variance=0.005)
    comp = compare_propagation(
        topo_draws, A, X, t, estimator=est,
        n_causal_draws=12, n_plugin_draws=20, random_state=10,
    )
    assert comp.nested.draws.shape == (36, t.shape[0])
    assert comp.plugin.draws.shape == (20, t.shape[0])
    assert comp.mean_width_nested >= 0.0
    assert comp.width_ratio > 0.0


def test_make_tsbcf_long_data_shapes():
    phi, A, X, t, _ = _synthetic_sample(n=6, m=7)
    long = make_tsbcf_long_data(phi, A, X, t, pi_hat=np.full(6, 0.5))
    assert long.y.shape == (42,)
    assert long.tgt.shape == (42,)
    assert long.x_control.shape == (42, 1)
    assert long.subject_index[0] == 0
    assert long.subject_index[-1] == 5
