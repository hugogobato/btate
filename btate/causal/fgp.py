"""Functional GP causal model for Phase 3.

The primary estimator here is a dependency-light finite-rank Gaussian-process
approximation.  It uses inducing locations in the joint covariate/filtration
space ``(X, t)`` and conjugate Bayesian linear regression on RBF features.  This
keeps the Phase-3 API runnable in a minimal environment while preserving the GP
posterior object needed for draws of the TATE curve.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from btate.embeddings.aggregation import summarize_posterior_functions
from btate.embeddings.reduction import FPCAModel, fit_fpca


def _as_rng(random_state=None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _as_binary_treatment(A) -> np.ndarray:
    arr = np.asarray(A, dtype=int).ravel()
    if arr.ndim != 1:
        raise ValueError("A must be one-dimensional")
    values = set(np.unique(arr).tolist())
    if not values.issubset({0, 1}):
        raise ValueError("A must contain only 0/1 treatment indicators")
    if len(values) < 2:
        raise ValueError("both treatment arms must be represented")
    return arr


def _as_grid(tseq) -> np.ndarray:
    grid = np.asarray(tseq, dtype=float).ravel()
    if grid.ndim != 1 or grid.shape[0] < 2:
        raise ValueError("tseq must be a one-dimensional grid with at least two points")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("tseq must be strictly increasing")
    return grid


def _as_covariates(X, n: int) -> np.ndarray:
    if X is None:
        return np.zeros((n, 0), dtype=float)
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("X must be a 1-D or 2-D covariate array")
    if arr.shape[0] != n:
        raise ValueError("X and phi must have the same number of subjects")
    return arr


def _observed_curves(phi, A: np.ndarray, potential_outcomes: bool) -> np.ndarray:
    arr = np.asarray(phi, dtype=float)
    if arr.ndim == 2:
        curves = arr
    elif arr.ndim == 3 and potential_outcomes:
        if arr.shape[1] != 2:
            raise ValueError("potential-outcome phi must have shape (n, 2, resolution)")
        curves = arr[np.arange(arr.shape[0]), A]
    elif arr.ndim == 3 and arr.shape[1] == 1:
        curves = arr[:, 0, :]
    else:
        raise ValueError(
            "phi must have shape (n, resolution). For arrays shaped "
            "(n, 2, resolution), set potential_outcomes=True when the second "
            "axis is control/treated potential outcomes."
        )
    if curves.ndim != 2:
        raise ValueError("observed curves must reduce to shape (n, resolution)")
    if curves.shape[0] != A.shape[0]:
        raise ValueError("phi and A must have the same number of subjects")
    return np.asarray(curves, dtype=float)


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X.shape[1] == 0:
        return X.copy(), np.zeros(0, dtype=float), np.ones(0, dtype=float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (X - mean) / scale, mean, scale


def _standardize_apply(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if mean.shape[0] == 0:
        return np.zeros((X.shape[0], 0), dtype=float)
    return (X - mean) / scale


def _scale_t_fit(grid: np.ndarray) -> tuple[np.ndarray, float, float]:
    lo = float(grid[0])
    span = float(grid[-1] - grid[0])
    return (grid - lo) / span, lo, span


def _scale_t_apply(grid: np.ndarray, lo: float, span: float) -> np.ndarray:
    return (grid - lo) / span


def _pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=float)
    A2 = np.sum(A * A, axis=1)[:, None]
    B2 = np.sum(B * B, axis=1)[None, :]
    return np.maximum(A2 + B2 - 2.0 * A @ B.T, 0.0)


def _default_x_length_scale(X_std: np.ndarray) -> float:
    if X_std.shape[1] == 0 or X_std.shape[0] < 2:
        return 1.0
    dist = np.sqrt(_pairwise_sqdist(X_std, X_std))
    vals = dist[np.triu_indices_from(dist, k=1)]
    vals = vals[vals > 1e-12]
    if vals.size == 0:
        return 1.0
    return float(np.median(vals))


def _rbf_joint_features(
    X_std: np.ndarray,
    t_std: np.ndarray,
    inducing_x: np.ndarray,
    inducing_t: np.ndarray,
    length_scale_x: float,
    length_scale_t: float,
) -> np.ndarray:
    xdist = _pairwise_sqdist(X_std, inducing_x) / max(length_scale_x, 1e-12) ** 2
    tdist = (t_std[:, None] - inducing_t[None, :]) ** 2 / max(length_scale_t, 1e-12) ** 2
    return np.exp(-0.5 * (xdist + tdist))


def _rbf_x_features(
    X_std: np.ndarray,
    inducing_x: np.ndarray,
    length_scale_x: float,
) -> np.ndarray:
    xdist = _pairwise_sqdist(X_std, inducing_x) / max(length_scale_x, 1e-12) ** 2
    return np.exp(-0.5 * xdist)


def _normalize_weights(weight: np.ndarray) -> np.ndarray:
    out = np.asarray(weight, dtype=float).ravel()
    if np.any(out < 0.0) or not np.all(np.isfinite(out)):
        raise ValueError("weights must be finite and non-negative")
    mean = float(np.mean(out))
    if mean <= 0.0:
        return np.ones_like(out)
    return out / mean


@dataclass
class _BayesianLinearFit:
    """Conjugate Bayesian linear regression posterior on kernel features."""

    coef_mean: np.ndarray
    coef_cov: np.ndarray
    noise_variance: float
    prior_scale: float
    jitter: float
    posterior_scale: float = 1.0

    def draw_coefficients(self, n_draws: int, rng: np.random.Generator) -> np.ndarray:
        cov = self.posterior_scale * self.coef_cov + self.jitter * np.eye(self.coef_cov.shape[0])
        return rng.multivariate_normal(self.coef_mean, cov, size=int(n_draws))

    def predict_mean(self, features: np.ndarray) -> np.ndarray:
        return features @ self.coef_mean


@dataclass
class _JointArmFit:
    model: _BayesianLinearFit
    inducing_x: np.ndarray
    inducing_t: np.ndarray
    length_scale_x: float
    length_scale_t: float

    def features(self, X_std: np.ndarray, t_std: np.ndarray) -> np.ndarray:
        return _rbf_joint_features(
            X_std, t_std, self.inducing_x, self.inducing_t,
            self.length_scale_x, self.length_scale_t,
        )


@dataclass
class _ScoreArmFit:
    models: list[_BayesianLinearFit]
    inducing_x: np.ndarray
    length_scale_x: float

    def features(self, X_std: np.ndarray) -> np.ndarray:
        return _rbf_x_features(X_std, self.inducing_x, self.length_scale_x)


@dataclass
class CausalEffectPosterior:
    """Posterior draws and credible bands for ``psi_d(t)``.

    ``draws`` has shape ``(n_draws, resolution)``.  The simultaneous band uses
    the same sup-norm standardized-deviation construction as Phase 2.
    """

    draws: np.ndarray
    grid: np.ndarray
    mean: np.ndarray
    pointwise_lower: np.ndarray
    pointwise_upper: np.ndarray
    simultaneous_lower: np.ndarray
    simultaneous_upper: np.ndarray
    alpha: float
    simultaneous_radius: float
    pr_all_positive: float
    pr_all_negative: float
    pr_excludes_zero: float
    band_excludes_zero: bool
    metadata: dict = field(default_factory=dict)


def summarize_causal_effect(draws, grid, alpha: float = 0.05,
                            metadata: dict | None = None) -> CausalEffectPosterior:
    """Summarize posterior draws of the TATE curve with Bayesian test metrics."""
    summary = summarize_posterior_functions(draws, grid=grid, alpha=alpha)
    arr = summary.draws
    pr_all_positive = float(np.mean(np.all(arr > 0.0, axis=1)))
    pr_all_negative = float(np.mean(np.all(arr < 0.0, axis=1)))
    pr_excludes_zero = pr_all_positive + pr_all_negative
    band_excludes_zero = bool(
        np.all(summary.simultaneous_lower > 0.0)
        or np.all(summary.simultaneous_upper < 0.0)
    )
    return CausalEffectPosterior(
        draws=arr,
        grid=summary.grid,
        mean=summary.mean,
        pointwise_lower=summary.pointwise_lower,
        pointwise_upper=summary.pointwise_upper,
        simultaneous_lower=summary.simultaneous_lower,
        simultaneous_upper=summary.simultaneous_upper,
        alpha=summary.alpha,
        simultaneous_radius=summary.simultaneous_radius,
        pr_all_positive=pr_all_positive,
        pr_all_negative=pr_all_negative,
        pr_excludes_zero=pr_excludes_zero,
        band_excludes_zero=band_excludes_zero,
        metadata=dict(metadata or {}),
    )


def bayesian_no_effect_test(effect: CausalEffectPosterior | np.ndarray,
                            rope: float = 0.0) -> dict:
    """Bayesian no-effect diagnostic for posterior TATE draws.

    ``rope`` is a region-of-practical-equivalence threshold in sup-norm units.
    With ``rope=0``, ``pr_supnorm_gt_rope`` is the posterior probability of a
    nonzero effect on the discretized grid.
    """
    if isinstance(effect, CausalEffectPosterior):
        draws = effect.draws
        band_excludes_zero = effect.band_excludes_zero
    else:
        draws = np.asarray(effect, dtype=float)
        band_excludes_zero = False
    if draws.ndim != 2:
        raise ValueError("effect draws must have shape (n_draws, resolution)")
    if rope < 0:
        raise ValueError("rope must be non-negative")
    sup_abs = np.max(np.abs(draws), axis=1)
    pr_within_rope = float(np.mean(sup_abs <= rope))
    pr_all_positive = float(np.mean(np.all(draws > rope, axis=1)))
    pr_all_negative = float(np.mean(np.all(draws < -rope, axis=1)))
    return {
        "rope": float(rope),
        "pr_supnorm_le_rope": pr_within_rope,
        "pr_supnorm_gt_rope": 1.0 - pr_within_rope,
        "pr_all_positive": pr_all_positive,
        "pr_all_negative": pr_all_negative,
        "pr_same_sign_excluding_rope": pr_all_positive + pr_all_negative,
        "band_excludes_zero": bool(band_excludes_zero),
    }


class FunctionalGPEstimator:
    """Finite-rank functional GP estimator of the posterior TATE curve.

    Parameters
    ----------
    n_inducing
        Maximum number of inducing features per treatment arm.
    use_fpca
        If false, fit a GP directly over the joint space ``(X, t)``.  If true,
        fit fPCA scores and a covariate GP for each score.  The direct joint GP
        is the default Phase-3 primary model; fPCA mode is the computational
        fallback from Research_Plan Task 3.1.
    fpca_components
        Number of fPCA components in score mode.
    prior_scale
        Prior standard deviation for finite-rank GP coefficients.
    noise_variance
        Observation noise variance.  If ``None``, it is estimated by a weighted
        ridge prefit separately for each arm.
    propensity_clip
        Lower/upper clipping applied to ``pi_hat`` before overlap weights are
        formed, matching the TATE estimator convention.
    posterior_scale
        Effective-sample-size inflation for the coefficient posterior covariance
        (default ``1.0`` — no change).  The finite-rank fit treats the ``res``
        grid points of each subject curve as conditionally independent given the
        coefficients, which overcounts the effective sample size and yields
        over-confident bands for the *curve-level* average ``psi_d(t)``.  Setting
        ``posterior_scale > 1`` widens the posterior draws to restore calibration
        (the Phase-4 benchmark selects it via ``PipelineConfig.fgp_posterior_scale``).
    """

    def __init__(
        self,
        n_inducing: int = 64,
        use_fpca: bool = False,
        fpca_components: int = 5,
        length_scale_x: float | None = None,
        length_scale_t: float | None = None,
        prior_scale: float = 1.0,
        noise_variance: float | None = None,
        propensity_clip: float = 0.01,
        jitter: float = 1e-8,
        posterior_scale: float = 1.0,
    ):
        if int(n_inducing) < 1:
            raise ValueError("n_inducing must be positive")
        if int(fpca_components) < 1:
            raise ValueError("fpca_components must be positive")
        if prior_scale <= 0:
            raise ValueError("prior_scale must be positive")
        if noise_variance is not None and noise_variance <= 0:
            raise ValueError("noise_variance must be positive")
        if not (0.0 < propensity_clip < 0.5):
            raise ValueError("propensity_clip must lie in (0, 0.5)")
        if posterior_scale <= 0:
            raise ValueError("posterior_scale must be positive")
        self.posterior_scale = float(posterior_scale)
        self.n_inducing = int(n_inducing)
        self.use_fpca = bool(use_fpca)
        self.fpca_components = int(fpca_components)
        self.length_scale_x = length_scale_x
        self.length_scale_t = length_scale_t
        self.prior_scale = float(prior_scale)
        self.noise_variance = None if noise_variance is None else float(noise_variance)
        self.propensity_clip = float(propensity_clip)
        self.jitter = float(jitter)

        self._is_fit = False
        self._mode = "fpca" if self.use_fpca else "joint"

    def fit(
        self,
        phi,
        A,
        X=None,
        tseq=None,
        pi_hat=None,
        random_state=None,
        potential_outcomes: bool = False,
    ) -> "FunctionalGPEstimator":
        """Fit the causal model from observed functional outcomes.

        ``phi`` should normally be the observed curve matrix with shape
        ``(n, resolution)``.  For validation settings with known potential
        outcomes, pass ``phi`` as ``(n, 2, resolution)`` and set
        ``potential_outcomes=True``; the observed arm is selected according to
        ``A`` before fitting.
        """
        A_arr = _as_binary_treatment(A)
        curves = _observed_curves(phi, A_arr, potential_outcomes=potential_outcomes)
        if tseq is None:
            grid = np.arange(curves.shape[1], dtype=float)
        else:
            grid = _as_grid(tseq)
        if curves.shape[1] != grid.shape[0]:
            raise ValueError("curve resolution must match tseq length")
        X_arr = _as_covariates(X, curves.shape[0])
        X_std, self.x_mean_, self.x_scale_ = _standardize_fit(X_arr)
        t_std, self.t_min_, self.t_span_ = _scale_t_fit(grid)

        self.grid_ = grid
        self.A_ = A_arr
        self.X_ = X_arr
        self.X_std_ = X_std
        self.observed_curves_ = curves
        self.pi_hat_ = self._propensity(pi_hat, A_arr)
        self.observation_weight_ = self._observation_weights(A_arr, self.pi_hat_)
        self.length_scale_x_ = (
            float(self.length_scale_x)
            if self.length_scale_x is not None
            else _default_x_length_scale(X_std)
        )
        self.length_scale_t_ = (
            float(self.length_scale_t) if self.length_scale_t is not None else 0.25
        )

        rng = _as_rng(random_state)
        if self.use_fpca:
            self._fit_fpca_mode(curves, A_arr, X_std, rng)
        else:
            self._fit_joint_mode(curves, A_arr, X_std, t_std, rng)
        self._is_fit = True
        return self

    def posterior_tate(
        self,
        X_grid=None,
        n_draws: int = 1000,
        alpha: float = 0.05,
        random_state=None,
    ) -> CausalEffectPosterior:
        """Return posterior draws and credible bands for ``psi_d(t)``."""
        if not self._is_fit:
            raise RuntimeError("fit must be called before posterior_tate")
        if int(n_draws) < 1:
            raise ValueError("n_draws must be positive")
        if X_grid is None:
            X_target = self.X_
        else:
            X_target = _as_covariates(X_grid, np.asarray(X_grid).shape[0])
        X_std = _standardize_apply(X_target, self.x_mean_, self.x_scale_)
        rng = _as_rng(random_state)
        if self.use_fpca:
            draws = self._draw_tate_fpca(X_std, int(n_draws), rng)
        else:
            draws = self._draw_tate_joint(X_std, int(n_draws), rng)
        return summarize_causal_effect(
            draws,
            grid=self.grid_,
            alpha=alpha,
            metadata={
                "model": "finite_rank_fgp",
                "mode": self._mode,
                "n_inducing": self.n_inducing,
                "n_subjects": int(self.A_.shape[0]),
                "propensity_clip": self.propensity_clip,
            },
        )

    def _propensity(self, pi_hat, A: np.ndarray) -> np.ndarray:
        if pi_hat is None:
            p = float(np.mean(A))
            pi = np.full(A.shape[0], p, dtype=float)
        else:
            pi = np.asarray(pi_hat, dtype=float).ravel()
            if pi.shape[0] != A.shape[0]:
                raise ValueError("pi_hat must have length n")
        return np.clip(pi, self.propensity_clip, 1.0 - self.propensity_clip)

    def _observation_weights(self, A: np.ndarray, pi: np.ndarray) -> np.ndarray:
        weight = A / pi + (1 - A) / (1.0 - pi)
        return _normalize_weights(weight)

    def _fit_joint_mode(
        self,
        curves: np.ndarray,
        A: np.ndarray,
        X_std: np.ndarray,
        t_std: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self.arm_fits_: dict[int, _JointArmFit] = {}
        for arm in (0, 1):
            mask = A == arm
            X_arm = X_std[mask]
            y = curves[mask].reshape(-1)
            subject_weight = self.observation_weight_[mask]
            row_weight = np.repeat(_normalize_weights(subject_weight), t_std.shape[0])
            X_rep = np.repeat(X_arm, t_std.shape[0], axis=0)
            t_rep = np.tile(t_std, X_arm.shape[0])

            n_feat = min(self.n_inducing, X_rep.shape[0])
            idx = rng.choice(X_rep.shape[0], size=n_feat, replace=False)
            inducing_x = X_rep[idx]
            inducing_t = t_rep[idx]
            features = _rbf_joint_features(
                X_rep, t_rep, inducing_x, inducing_t,
                self.length_scale_x_, self.length_scale_t_,
            )
            model = self._fit_blr(features, y, row_weight)
            self.arm_fits_[arm] = _JointArmFit(
                model=model,
                inducing_x=inducing_x,
                inducing_t=inducing_t,
                length_scale_x=self.length_scale_x_,
                length_scale_t=self.length_scale_t_,
            )

    def _fit_fpca_mode(
        self,
        curves: np.ndarray,
        A: np.ndarray,
        X_std: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self.fpca_model_: FPCAModel = fit_fpca(
            curves, n_components=min(self.fpca_components, curves.shape[0], curves.shape[1])
        )
        scores = self.fpca_model_.transform(curves)
        self.score_arm_fits_: dict[int, _ScoreArmFit] = {}
        for arm in (0, 1):
            mask = A == arm
            X_arm = X_std[mask]
            weight = _normalize_weights(self.observation_weight_[mask])
            n_feat = min(self.n_inducing, X_arm.shape[0])
            idx = rng.choice(X_arm.shape[0], size=n_feat, replace=False)
            inducing_x = X_arm[idx]
            features = _rbf_x_features(X_arm, inducing_x, self.length_scale_x_)
            models = [
                self._fit_blr(features, scores[mask, k], weight)
                for k in range(scores.shape[1])
            ]
            self.score_arm_fits_[arm] = _ScoreArmFit(
                models=models,
                inducing_x=inducing_x,
                length_scale_x=self.length_scale_x_,
            )

    def _fit_blr(
        self,
        features: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
    ) -> _BayesianLinearFit:
        weight = _normalize_weights(weight)
        prior_var = self.prior_scale ** 2
        noise = self.noise_variance
        if noise is None:
            precision0 = features.T @ (weight[:, None] * features)
            precision0 += (1.0 / prior_var + self.jitter) * np.eye(features.shape[1])
            rhs0 = features.T @ (weight * y)
            coef0 = np.linalg.solve(precision0, rhs0)
            resid = y - features @ coef0
            noise = float(np.sum(weight * resid * resid) / max(np.sum(weight), 1.0))
            noise = max(noise, 1e-8)

        scaled_weight = weight / noise
        precision = features.T @ (scaled_weight[:, None] * features)
        precision += (1.0 / prior_var + self.jitter) * np.eye(features.shape[1])
        rhs = features.T @ (scaled_weight * y)
        coef_mean = np.linalg.solve(precision, rhs)
        coef_cov = np.linalg.inv(precision)
        coef_cov = 0.5 * (coef_cov + coef_cov.T)
        return _BayesianLinearFit(
            coef_mean=coef_mean,
            coef_cov=coef_cov,
            noise_variance=float(noise),
            prior_scale=self.prior_scale,
            jitter=self.jitter,
            posterior_scale=self.posterior_scale,
        )

    def _draw_tate_joint(
        self,
        X_std: np.ndarray,
        n_draws: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        t_std = _scale_t_apply(self.grid_, self.t_min_, self.t_span_)
        n_x = X_std.shape[0]
        out = np.zeros((n_draws, self.grid_.shape[0]), dtype=float)
        for arm in (0, 1):
            fit = self.arm_fits_[arm]
            X_rep = np.repeat(X_std, self.grid_.shape[0], axis=0)
            t_rep = np.tile(t_std, n_x)
            features = fit.features(X_rep, t_rep).reshape(n_x, self.grid_.shape[0], -1)
            mean_features = features.mean(axis=0)
            coef_draws = fit.model.draw_coefficients(n_draws, rng)
            arm_draws = coef_draws @ mean_features.T
            out += arm_draws if arm == 1 else -arm_draws
        return out

    def _draw_tate_fpca(
        self,
        X_std: np.ndarray,
        n_draws: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        score_draws = []
        for arm in (0, 1):
            fit = self.score_arm_fits_[arm]
            features = fit.features(X_std)
            mean_features = features.mean(axis=0)
            comp_draws = []
            for model in fit.models:
                coef_draws = model.draw_coefficients(n_draws, rng)
                comp_draws.append(coef_draws @ mean_features)
            score_draws.append(np.column_stack(comp_draws))
        diff_scores = score_draws[1] - score_draws[0]
        reconstructed = diff_scores @ self.fpca_model_.components_
        return reconstructed.reshape(n_draws, self.grid_.shape[0])
