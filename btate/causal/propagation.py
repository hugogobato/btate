"""Nested topological-to-causal posterior propagation for Phase 3."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fgp import CausalEffectPosterior, FunctionalGPEstimator, summarize_causal_effect


@dataclass
class PropagationComparison:
    """Nested vs. plug-in posterior comparison for ``psi_d(t)``."""

    nested: CausalEffectPosterior
    plugin: CausalEffectPosterior
    mean_width_nested: float
    mean_width_plugin: float
    width_ratio: float
    topological_width_increment: float


def _clone_estimator(base: FunctionalGPEstimator | None) -> FunctionalGPEstimator:
    if base is None:
        return FunctionalGPEstimator()
    return FunctionalGPEstimator(
        n_inducing=base.n_inducing,
        use_fpca=base.use_fpca,
        fpca_components=base.fpca_components,
        length_scale_x=base.length_scale_x,
        length_scale_t=base.length_scale_t,
        prior_scale=base.prior_scale,
        noise_variance=base.noise_variance,
        propensity_clip=base.propensity_clip,
        jitter=base.jitter,
        posterior_scale=base.posterior_scale,
    )


def _mean_width(effect: CausalEffectPosterior) -> float:
    return float(np.mean(effect.simultaneous_upper - effect.simultaneous_lower))


def nested_posterior_tate(
    phi_draws,
    A,
    X,
    tseq,
    pi_hat=None,
    estimator: FunctionalGPEstimator | None = None,
    n_causal_draws: int = 100,
    alpha: float = 0.05,
    random_state=None,
    potential_outcomes: bool = False,
) -> CausalEffectPosterior:
    """Pool causal posterior draws over topological posterior draws.

    ``phi_draws`` is expected to have shape ``(n_topological_draws, n, m)`` for
    observed curves, or ``(n_topological_draws, n, 2, m)`` with
    ``potential_outcomes=True`` for validation settings with known potential
    outcomes.
    """
    arr = np.asarray(phi_draws, dtype=float)
    if arr.ndim not in (3, 4):
        raise ValueError("phi_draws must have shape (S, n, m) or (S, n, 2, m)")
    if int(n_causal_draws) < 1:
        raise ValueError("n_causal_draws must be positive")
    rng = np.random.default_rng(random_state)
    pooled = []
    for draw_idx in range(arr.shape[0]):
        fit_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        post_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        model = _clone_estimator(estimator)
        model.fit(
            arr[draw_idx], A, X=X, tseq=tseq, pi_hat=pi_hat,
            random_state=fit_seed, potential_outcomes=potential_outcomes,
        )
        pooled.append(
            model.posterior_tate(
                n_draws=n_causal_draws, alpha=alpha, random_state=post_seed
            ).draws
        )
    draws = np.vstack(pooled)
    return summarize_causal_effect(
        draws,
        grid=tseq,
        alpha=alpha,
        metadata={
            "propagation": "nested",
            "n_topological_draws": int(arr.shape[0]),
            "n_causal_draws_per_topological_draw": int(n_causal_draws),
        },
    )


def plugin_posterior_tate(
    phi_draws_or_mean,
    A,
    X,
    tseq,
    pi_hat=None,
    estimator: FunctionalGPEstimator | None = None,
    n_draws: int = 1000,
    alpha: float = 0.05,
    random_state=None,
    potential_outcomes: bool = False,
) -> CausalEffectPosterior:
    """Fit the causal posterior to the posterior-mean functional summary."""
    arr = np.asarray(phi_draws_or_mean, dtype=float)
    if potential_outcomes:
        if arr.ndim == 4:
            phi_mean = arr.mean(axis=0)
        elif arr.ndim == 3:
            phi_mean = arr
        else:
            raise ValueError("potential-outcome phi must have shape (S, n, 2, m) or (n, 2, m)")
    else:
        if arr.ndim == 3:
            phi_mean = arr.mean(axis=0)
        elif arr.ndim == 2:
            phi_mean = arr
        else:
            raise ValueError("observed phi must have shape (S, n, m) or (n, m)")
    model = _clone_estimator(estimator)
    model.fit(
        phi_mean, A, X=X, tseq=tseq, pi_hat=pi_hat,
        random_state=random_state, potential_outcomes=potential_outcomes,
    )
    effect = model.posterior_tate(
        n_draws=n_draws, alpha=alpha, random_state=random_state
    )
    effect.metadata.update({"propagation": "plugin"})
    return effect


def compare_propagation(
    phi_draws,
    A,
    X,
    tseq,
    pi_hat=None,
    estimator: FunctionalGPEstimator | None = None,
    n_causal_draws: int = 100,
    n_plugin_draws: int = 1000,
    alpha: float = 0.05,
    random_state=None,
    potential_outcomes: bool = False,
) -> PropagationComparison:
    """Compare nested propagation with the plug-in posterior-mean shortcut."""
    nested = nested_posterior_tate(
        phi_draws, A, X, tseq, pi_hat=pi_hat, estimator=estimator,
        n_causal_draws=n_causal_draws, alpha=alpha, random_state=random_state,
        potential_outcomes=potential_outcomes,
    )
    plugin = plugin_posterior_tate(
        phi_draws, A, X, tseq, pi_hat=pi_hat, estimator=estimator,
        n_draws=n_plugin_draws, alpha=alpha, random_state=random_state,
        potential_outcomes=potential_outcomes,
    )
    nested_width = _mean_width(nested)
    plugin_width = _mean_width(plugin)
    ratio = float(nested_width / plugin_width) if plugin_width > 0 else float("inf")
    return PropagationComparison(
        nested=nested,
        plugin=plugin,
        mean_width_nested=nested_width,
        mean_width_plugin=plugin_width,
        width_ratio=ratio,
        topological_width_increment=nested_width - plugin_width,
    )
