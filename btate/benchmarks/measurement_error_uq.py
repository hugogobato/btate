r"""Bayesian-vs-frequentist UQ under topological measurement error (Phase 6).

Scientific setting
------------------
The TATE estimator treats the observed persistence diagram as the outcome.  It
does not model the fact that the observed diagram is a noisy, finite-sample
proxy for a latent *clean* topological object.  This module runs the experiment
that asks whether modelling that measurement error buys anything, and it fixes
one common target so the comparison is a UQ comparison rather than a comparison
of different estimands.

**Primary target (every method is scored against this):**

.. math::

    \psi^{\mathrm{clean}}_{\alpha}(t)
      = \mathbb E\bigl[\phi\{D_\alpha(Y^{\mathrm{clean}}(1))\}(t)
                     - \phi\{D_\alpha(Y^{\mathrm{clean}}(0))\}(t)\bigr],

the population power-weighted-silhouette contrast of the **Alpha-filtration**
diagrams of the latent clean point clouds -- the effect a perfect measurement
error procedure would recover.  The Alpha filtration matches the ORBIT
experiment of ``TATE_Paper/appendix.tex``.

**Secondary target:** the same functional of the *observed* (contaminated)
clouds, :math:`\psi^{\mathrm{noisy}}_{\alpha}`.  Every accuracy and coverage
column emitted here carries a ``_clean`` or ``_noisy`` suffix; the helpers in
:mod:`btate.benchmarks.metrics` raise if a target label is omitted.

Why DTM cannot simply be swapped in
-----------------------------------
On this DGP the clean Alpha effect peaks at ~0.170 while the clean DTM-Rips
``k=15`` effect peaks at ~0.296, on a filtration axis roughly twice as long
(``docs/phase5_5_transition_sweep.md``).  A DTM effect curve is therefore a
*different estimand*, and a smaller DTM-within-DTM error is not evidence about
clean Alpha.  DTM enters here only as an **auxiliary predictor block inside an
explicitly estimated and validated DTM-to-clean-Alpha bridge**
(``use_dtm_feature=True``); the bridge's output is scored against clean Alpha
under ``clean_alpha_*`` names.  No apex normalization or rescaling is used
anywhere.

Methods compared (identical datasets, grid, and seed streams)
-------------------------------------------------------------
====  ======================  ==================================================
M1    ``oracle_clean_aipw``   AIPW on the **clean** Alpha silhouettes.  Not
                              available in practice; the upper benchmark.
M2    ``blind_freq_aipw``     AIPW on contaminated Alpha silhouettes treated as
                              if perfectly observed (EIF pointwise band +
                              Gaussian-multiplier simultaneous band).
M3    ``bayes_causal_only``   Contaminated Alpha silhouettes through the
                              Bayesian functional-GP causal model, with **no**
                              measurement-error propagation.  Isolates the
                              causal model from the measurement layer.
M4    ``me_bayes_bridge``     Validation-calibrated posterior draws of the
                              latent clean Alpha silhouettes, propagated through
                              the same causal model (nested propagation).
M5    ``me_freq_mi``          The same posterior draws consumed by AIPW with
                              multiple-imputation pooling (Rubin's rules plus a
                              curve-level sup-t multiplier band).  Separates
                              "measurement-error correction" from "Bayesian
                              causal modelling".
====  ======================  ==================================================

``M4``/``M5`` gain the suffix ``_dtm`` when the bridge also consumes the DTM
silhouette block.

Identification caveat (stated in every report)
----------------------------------------------
The bridge is **validation-calibrated**: it is fitted on an independent
calibration sample in which both the clean and the contaminated cloud are
observed.  Nothing here identifies the clean Alpha estimand from contaminated
data alone.  In an application the method requires replicate measurements,
external validation data, or a known corruption model, and the transportability
assumption (the calibration corruption mechanism equals the evaluation one) is
an assumption, not a result.

Numerical reproducibility
-------------------------
Fixing the seeds is necessary but not sufficient.  The functional-GP posterior
covariance (especially the ``"godambe"`` sandwich) is close to singular, so a
``1e-16`` difference in a BLAS matrix product can move a credible band by
several percent.  Runs are therefore only bit-reproducible at a **fixed BLAS
thread count**; call :func:`pin_blas_threads` (or set ``OMP_NUM_THREADS`` and
friends before importing numpy) in every runner, notebook and test process.
Under a pinned thread count, serial and parallel runs of the same seed agree
exactly -- that is what ``test_serial_and_parallel_agree`` checks.

Leakage discipline
------------------
:class:`ObservedBundle` holds everything a non-oracle estimator may see and
contains no clean arrays.  Clean evaluation curves live in a separate
:class:`OracleBundle` that only ``M1`` and the scoring code touch.  The bridge
never sees an evaluation subject.  ``tests/test_measurement_error_uq.py``
asserts that ``M2``--``M5`` are bit-identical when the oracle bundle is
replaced by garbage.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from btate.benchmarks.metrics import (
    band_coverage_columns,
    curve_error_columns,
    integrated_abs_error,
    max_abs_error,
    nrmse,
    peak_index,
    rmse,
)
from btate.benchmarks.synthetic import (
    SyntheticConfig,
    generate_synthetic_dataset,
    low_snr_config,
)

METHODS = (
    "oracle_clean_aipw",
    "blind_freq_aipw",
    "bayes_causal_only",
    "me_bayes_bridge",
    "me_freq_mi",
)
DTM_METHODS = ("me_bayes_bridge_dtm", "me_freq_mi_dtm")
ORACLE_METHODS = ("oracle_clean_aipw",)


# --------------------------------------------------------------------------- #
# Deterministic, disjoint seed namespaces
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SeedNamespaces:
    """Disjoint deterministic seed bases, one per source of randomness.

    Keeping these apart is what makes the calibration sample independent of the
    evaluation replicates and the population truth independent of both.  The
    bases are spaced by 1e6 and every draw is derived through
    ``np.random.SeedSequence([base, *parts])``, so two namespaces cannot collide
    unless their bases do.
    """

    truth: int = 61_000_000
    calibration: int = 62_000_000
    evaluation: int = 63_000_000
    measurement_posterior: int = 64_000_000
    causal: int = 65_000_000
    bootstrap: int = 66_000_000

    def bases(self) -> tuple[int, ...]:
        return (self.truth, self.calibration, self.evaluation,
                self.measurement_posterior, self.causal, self.bootstrap)

    def validate(self) -> None:
        bases = self.bases()
        if len(set(bases)) != len(bases):
            raise ValueError(f"seed namespaces must be disjoint, got {bases}")


_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def pin_blas_threads(n_threads: int = 1):
    """Pin BLAS threading so numerical results are reproducible.

    Returns a context manager when ``threadpoolctl`` is available (the reliable
    path, since it re-limits pools that are already loaded) and otherwise a
    no-op object, having set the standard environment variables -- which only
    bite if numpy has not been imported yet.

    Use it in every process that produces experiment output.  Without it, the
    same seed can give different bands in a worker process than in the parent
    (see *Numerical reproducibility* above).
    """
    for name in _BLAS_ENV_VARS:
        os.environ.setdefault(name, str(int(n_threads)))
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - environment-dependent
        class _NoOp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _NoOp()
    return threadpool_limits(limits=int(n_threads))


def _rng(base: int, *parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(base), *map(int, parts)]))


def _seed_int(base: int, *parts: int) -> int:
    """A reproducible 31-bit seed for callees that want an int, not a Generator."""
    return int(_rng(base, *parts).integers(0, np.iinfo(np.int32).max))


def _noise_key(noise_level: float) -> int:
    """Integer key for a noise level (avoids float keys in seed sequences)."""
    return int(round(10_000.0 * float(noise_level)))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MEUQConfig:
    """Frozen configuration for one measurement-error UQ cell.

    Nothing in here may be chosen by looking at evaluation coverage.  The grid
    comes from the oracle truth sample, the bridge ridge penalty from a held-out
    split of the *calibration* sample, and the causal hyper-parameters are the
    project's pre-existing Phase-4 defaults with the no-knob ``"godambe"``
    posterior scale.
    """

    noise_level: float = 0.125
    n_subjects: int = 40
    resolution: int = 96
    r: float = 3.0
    max_points: int | None = 200
    alpha: float = 0.05
    peak_window: int = 6

    # Frozen grid (filled by :func:`freeze_alpha_grid`; never derived from
    # contaminated evaluation outcomes).
    grid_upper: float | None = None
    dtm_grid_upper: float | None = None

    # Auxiliary DTM feature block for the bridge.  DTM-Rips costs ~250 ms per
    # cloud (vs ~1 ms for Alpha), so the *feature* range is frozen on a capped
    # subsample of the oracle clean clouds; it is a representation choice, not
    # an estimand, so a coarser sample is sufficient.
    use_dtm_feature: bool = False
    dtm_k: int = 15
    dtm_grid_max_clouds: int = 80

    # Population truth (oracle Monte-Carlo sample).
    truth_subjects: int = 200
    truth_datasets: int = 8

    # Validation calibration sample for the bridge.
    n_calibration_subjects: int = 240
    calibration_holdout_frac: float = 0.25
    bridge_k_alpha: int = 10
    bridge_k_dtm: int = 6
    bridge_ridge_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    bridge_scale_prior: float = 1e-6

    # Posterior / causal / bootstrap sizes.
    n_measurement_draws: int = 16
    n_causal_draws: int = 60
    n_plugin_draws: int = 480
    n_boot: int = 1000

    # Causal model (Phase-4 defaults; ``"godambe"`` is the no-knob ESS scale).
    fgp_n_inducing: int = 48
    fgp_prior_scale: float = 5.0
    fgp_length_scale_t: float | None = 0.06
    fgp_posterior_scale: float | str = "godambe"
    n_basis: int = 5

    seeds: SeedNamespaces = field(default_factory=SeedNamespaces)

    def __post_init__(self) -> None:
        self.seeds.validate()
        if self.n_subjects < 8:
            raise ValueError("n_subjects must be at least 8 for cross-fitting")
        if self.resolution < 16:
            raise ValueError("resolution must be at least 16")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if not (0.0 < self.calibration_holdout_frac < 0.9):
            raise ValueError("calibration_holdout_frac must be in (0, 0.9)")
        if self.n_measurement_draws < 2:
            raise ValueError("n_measurement_draws must be at least 2")

    @property
    def methods(self) -> tuple[str, ...]:
        return METHODS + (DTM_METHODS if self.use_dtm_feature else ())

    def to_dict(self) -> dict:
        out = asdict(self)
        out["bridge_ridge_grid"] = list(self.bridge_ridge_grid)
        return out


def smoke_config(**overrides) -> MEUQConfig:
    """A deliberately tiny configuration for the local smoke test.

    Exercises every method and every metric; it is **not** evidence about
    coverage (2 replicates cannot estimate a 0.95 rate).
    """
    base = dict(
        n_subjects=12, resolution=64, truth_subjects=40, truth_datasets=2,
        n_calibration_subjects=60, bridge_k_alpha=6, bridge_k_dtm=4,
        n_measurement_draws=4, n_causal_draws=12, n_plugin_draws=80,
        n_boot=200,
    )
    base.update(overrides)
    return MEUQConfig(**base)


def dgp_config(cfg: MEUQConfig, *, n: int, seed: int, noise_seed: int,
               noise_level: float | None = None) -> SyntheticConfig:
    """The Phase-5.5 low-SNR DGP, parameterized by explicit seed streams."""
    return low_snr_config(
        n=int(n),
        noise_level=float(cfg.noise_level if noise_level is None else noise_level),
        seed=int(seed),
        noise_seed=int(noise_seed),
    )


# --------------------------------------------------------------------------- #
# Representation helpers
# --------------------------------------------------------------------------- #
def _subsample(points, max_points: int | None, rng: np.random.Generator):
    points = np.asarray(points, dtype=float)
    if max_points is None or len(points) <= int(max_points):
        return points
    keep = rng.choice(len(points), int(max_points), replace=False)
    return points[keep]


def alpha_diagram(points) -> np.ndarray:
    """H1 Alpha-complex diagram in birth--death coordinates."""
    from btate.benchmarks.pipeline import h1_diagram

    return h1_diagram(points)


def dtm_diagram(points, k: int = 15) -> np.ndarray:
    """H1 DTM-Rips diagram (auxiliary bridge feature only, never an estimand)."""
    from btate.benchmarks.dtm import h1_diagram_dtm

    return h1_diagram_dtm(points, k=int(k))


def silhouette_on_grid(diagram, sample_range, resolution: int,
                       r: float) -> np.ndarray:
    """Power-weighted silhouette of one diagram on a *fixed* grid."""
    from btate.embeddings.silhouette import weighted_silhouette

    curves = weighted_silhouette(
        [np.asarray(diagram, dtype=float)], weights="power", r=float(r),
        sample_range=tuple(float(v) for v in sample_range),
        resolution=int(resolution),
    )
    return np.asarray(curves, dtype=float)[0]


@dataclass(frozen=True)
class GridSpec:
    """The frozen filtration grid shared by every method, arm and replicate."""

    sample_range: tuple[float, float]
    resolution: int
    r: float
    grid: np.ndarray
    dtm_sample_range: tuple[float, float] | None = None
    dtm_resolution: int | None = None
    provenance: dict = field(default_factory=dict)

    def alpha_curve(self, cloud) -> np.ndarray:
        return silhouette_on_grid(alpha_diagram(cloud), self.sample_range,
                                  self.resolution, self.r)

    def dtm_curve(self, cloud, k: int) -> np.ndarray:
        if self.dtm_sample_range is None:
            raise ValueError("this GridSpec has no frozen DTM feature range")
        return silhouette_on_grid(dtm_diagram(cloud, k=k), self.dtm_sample_range,
                                  int(self.dtm_resolution or self.resolution),
                                  self.r)

    def to_dict(self) -> dict:
        return {
            "sample_range": list(self.sample_range),
            "resolution": int(self.resolution),
            "r": float(self.r),
            "dtm_sample_range": (None if self.dtm_sample_range is None
                                 else list(self.dtm_sample_range)),
            "dtm_resolution": self.dtm_resolution,
            "provenance": self.provenance,
        }


def _truth_datasets(cfg: MEUQConfig, noise_level: float):
    """Independent oracle datasets from the *truth* seed namespace."""
    key = _noise_key(noise_level)
    for k in range(int(cfg.truth_datasets)):
        yield generate_synthetic_dataset(dgp_config(
            cfg, n=int(cfg.truth_subjects),
            seed=_seed_int(cfg.seeds.truth, 1, k),
            noise_seed=_seed_int(cfg.seeds.truth, 2, k, key),
            noise_level=noise_level,
        ))


def freeze_alpha_grid(cfg: MEUQConfig) -> GridSpec:
    """Freeze the Alpha (and optional DTM-feature) grid on the oracle sample.

    The upper end is ``1.5 x`` the largest finite clean H1 death observed in an
    **independent oracle calibration sample** drawn from the truth namespace,
    rounded to three decimals so the frozen value is stable under trivial
    resampling.  It is computed once, at ``noise_level = 0``, from *clean*
    clouds only, so it is identical across noise levels, methods, arms and
    replicates, and can never be moved by contaminated evaluation outcomes.
    """
    if cfg.grid_upper is not None:
        upper = float(cfg.grid_upper)
        dtm_upper = cfg.dtm_grid_upper
        provenance = {"source": "explicit_config"}
    else:
        deaths: list[float] = []
        dtm_deaths: list[float] = []
        n_dtm_clouds = 0
        rng = _rng(cfg.seeds.truth, 3)
        for ds in _truth_datasets(cfg, 0.0):
            for i in range(ds.clean_clouds.shape[0]):
                for arm in (0, 1):
                    cloud = _subsample(ds.clean_clouds[i, arm], cfg.max_points, rng)
                    dgm = alpha_diagram(cloud)
                    if dgm.size:
                        vals = dgm[:, 1]
                        deaths.extend(vals[np.isfinite(vals)].tolist())
                    if cfg.use_dtm_feature and n_dtm_clouds < cfg.dtm_grid_max_clouds:
                        n_dtm_clouds += 1
                        ddgm = dtm_diagram(cloud, k=cfg.dtm_k)
                        if ddgm.size:
                            vals = ddgm[:, 1]
                            dtm_deaths.extend(vals[np.isfinite(vals)].tolist())
        if not deaths:
            raise RuntimeError("oracle clean sample has no finite Alpha H1 deaths")
        upper = float(np.round(1.5 * max(deaths), 3))
        dtm_upper = (float(np.round(1.5 * max(dtm_deaths), 3))
                     if cfg.use_dtm_feature and dtm_deaths else None)
        provenance = {
            "source": "oracle_clean_alpha_deaths",
            "rule": "round(1.5 * max finite clean H1 death, 3)",
            "n_clean_curves": int(cfg.truth_datasets * cfg.truth_subjects * 2),
            "max_clean_death": float(max(deaths)),
            "n_dtm_feature_clouds": int(n_dtm_clouds),
        }
    if upper <= 0.0:
        raise RuntimeError("frozen grid upper bound must be positive")
    grid = np.linspace(0.0, upper, int(cfg.resolution))
    return GridSpec(
        sample_range=(0.0, upper), resolution=int(cfg.resolution), r=float(cfg.r),
        grid=grid,
        dtm_sample_range=None if dtm_upper is None else (0.0, float(dtm_upper)),
        dtm_resolution=None if dtm_upper is None else int(cfg.resolution),
        provenance=provenance,
    )


# --------------------------------------------------------------------------- #
# Population targets
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PopulationTargets:
    """Fixed population TATE curves, from an independent oracle MC sample."""

    psi_clean_alpha: np.ndarray
    psi_noisy_alpha: np.ndarray
    mc_se_clean: np.ndarray
    mc_se_noisy: np.ndarray
    n_oracle_subjects: int
    noise_level: float
    grid: np.ndarray

    def target(self, label: str) -> np.ndarray:
        if label == "clean":
            return self.psi_clean_alpha
        if label == "noisy":
            return self.psi_noisy_alpha
        raise ValueError(f"unknown target label {label!r}")

    def to_dict(self) -> dict:
        return {
            "psi_clean_alpha": self.psi_clean_alpha.tolist(),
            "psi_noisy_alpha": self.psi_noisy_alpha.tolist(),
            "mc_se_clean": self.mc_se_clean.tolist(),
            "mc_se_noisy": self.mc_se_noisy.tolist(),
            "n_oracle_subjects": int(self.n_oracle_subjects),
            "noise_level": float(self.noise_level),
            "grid": self.grid.tolist(),
        }


def population_targets(cfg: MEUQConfig, grid: GridSpec) -> PopulationTargets:
    """Fixed clean-Alpha and noisy-Alpha population TATE curves.

    Both are Monte-Carlo averages of the per-subject arm contrast over a large,
    **independent** oracle sample from the truth seed namespace.  No evaluation
    replicate is reused.  ``psi_clean_alpha`` is computed once from the
    ``noise_level = 0`` truth datasets: the clean potential-outcome clouds have
    a noise-invariant distribution, so this is the *same* frozen clean target
    for every noise cell -- which is the whole point of the experiment.
    """
    clean_contrasts: list[np.ndarray] = []
    noisy_contrasts: list[np.ndarray] = []
    rng = _rng(cfg.seeds.truth, 4, _noise_key(cfg.noise_level))

    for ds in _truth_datasets(cfg, 0.0):
        for i in range(ds.clean_clouds.shape[0]):
            arms = [grid.alpha_curve(_subsample(ds.clean_clouds[i, a],
                                                cfg.max_points, rng))
                    for a in (0, 1)]
            clean_contrasts.append(arms[1] - arms[0])
    for ds in _truth_datasets(cfg, cfg.noise_level):
        for i in range(ds.clouds.shape[0]):
            arms = [grid.alpha_curve(_subsample(ds.clouds[i, a],
                                                cfg.max_points, rng))
                    for a in (0, 1)]
            noisy_contrasts.append(arms[1] - arms[0])

    clean = np.stack(clean_contrasts)
    noisy = np.stack(noisy_contrasts)
    n_clean, n_noisy = clean.shape[0], noisy.shape[0]
    return PopulationTargets(
        psi_clean_alpha=clean.mean(axis=0),
        psi_noisy_alpha=noisy.mean(axis=0),
        mc_se_clean=clean.std(axis=0, ddof=1) / np.sqrt(n_clean),
        mc_se_noisy=noisy.std(axis=0, ddof=1) / np.sqrt(n_noisy),
        n_oracle_subjects=int(n_clean),
        noise_level=float(cfg.noise_level),
        grid=grid.grid,
    )


# --------------------------------------------------------------------------- #
# Validation-calibrated measurement-error bridge
# --------------------------------------------------------------------------- #
@dataclass
class MeasurementBridge:
    """Posterior of the latent clean Alpha silhouette given observed curves.

    Model (grid space, fitted on an independent paired calibration sample)::

        phi_clean = W' z + e,      e ~ N(0, Sigma),
        z         = [1, PCA scores of phi_obs_alpha (, of phi_obs_dtm)]

    with a conjugate matrix-normal / inverse-Wishart posterior for
    ``(W, Sigma)``.  Draws therefore carry **both** residual measurement
    uncertainty (``Sigma``) and calibration-parameter uncertainty (the posterior
    of ``W``), which is the honest accounting for a validation-calibrated
    correction.

    The DTM block, when present, is exactly the "explicitly estimated and
    validated DTM-to-clean-Alpha bridge" -- the only sanctioned use of DTM
    against an Alpha target.  Arm labels are never predictors, so the bridge is
    a pure measurement-error map and cannot encode the treatment effect.
    """

    mean_obs_alpha: np.ndarray
    basis_alpha: np.ndarray
    mean_obs_dtm: np.ndarray | None
    basis_dtm: np.ndarray | None
    W_hat: np.ndarray                 # (q, m) posterior mean coefficients
    lambda_chol: np.ndarray           # (q, q) chol of (Z'Z + ridge I)^{-1}
    scale_matrix: np.ndarray          # (m, m) inverse-Wishart scale S_n
    df: int                           # inverse-Wishart degrees of freedom
    ridge: float
    n_calibration_curves: int
    diagnostics: dict = field(default_factory=dict)

    @property
    def uses_dtm(self) -> bool:
        return self.basis_dtm is not None

    def features(self, phi_obs_alpha, phi_obs_dtm=None) -> np.ndarray:
        """Design matrix ``Z`` (n, q) from observed curves only."""
        a = np.atleast_2d(np.asarray(phi_obs_alpha, dtype=float))
        blocks = [np.ones((a.shape[0], 1)), (a - self.mean_obs_alpha) @ self.basis_alpha]
        if self.uses_dtm:
            if phi_obs_dtm is None:
                raise ValueError("this bridge needs the DTM feature block")
            d = np.atleast_2d(np.asarray(phi_obs_dtm, dtype=float))
            blocks.append((d - self.mean_obs_dtm) @ self.basis_dtm)
        elif phi_obs_dtm is not None:
            raise ValueError("this bridge was fitted without a DTM feature block")
        return np.hstack(blocks)

    def posterior_mean(self, phi_obs_alpha, phi_obs_dtm=None) -> np.ndarray:
        """Plug-in posterior-mean reconstruction of the clean Alpha curves."""
        return self.features(phi_obs_alpha, phi_obs_dtm) @ self.W_hat

    def draw_clean_curves(self, phi_obs_alpha, phi_obs_dtm=None, *,
                          n_draws: int = 16, random_state=None) -> np.ndarray:
        """Draw ``(n_draws, n, m)`` latent clean Alpha silhouettes.

        ``(W, Sigma)`` is resampled once per draw index and shared across
        subjects -- correct, because the calibration parameters are shared -- and
        the residual ``e`` is drawn independently per subject.
        """
        Z = self.features(phi_obs_alpha, phi_obs_dtm)
        n, q = Z.shape
        m = self.W_hat.shape[1]
        rng = np.random.default_rng(random_state)
        out = np.empty((int(n_draws), n, m), dtype=float)
        for s in range(int(n_draws)):
            factor = _inverse_wishart_factor(self.df, self.scale_matrix, rng)
            noise = rng.standard_normal(size=(q, m))
            W = self.W_hat + self.lambda_chol @ noise @ factor.T
            eps = rng.standard_normal(size=(n, m))
            out[s] = Z @ W + eps @ factor.T
        return out

    def to_dict(self) -> dict:
        return {
            "uses_dtm": bool(self.uses_dtm),
            "ridge": float(self.ridge),
            "df": int(self.df),
            "n_calibration_curves": int(self.n_calibration_curves),
            "n_features": int(self.W_hat.shape[0]),
            "resolution": int(self.W_hat.shape[1]),
            "diagnostics": self.diagnostics,
        }


def _inverse_wishart_factor(df: int, scale: np.ndarray,
                            rng: np.random.Generator) -> np.ndarray:
    r"""A factor ``F`` with ``F F^T ~ InverseWishart(df, scale)``.

    Uses the Bartlett decomposition directly instead of
    ``scipy.stats.invwishart`` + a Cholesky of the realized matrix.  With
    ``scale = L L^T`` and ``A`` the lower-triangular Bartlett factor of a
    ``Wishart(df, I)`` draw, ``Sigma = L A^{-T} A^{-1} L^T``, so
    ``F = (A^{-1} L^T)^T`` is a valid factor obtained from one Cholesky and one
    triangular solve -- both backward-stable and deterministic.

    This matters beyond speed: forming ``Sigma`` and re-factorizing it made the
    draws depend on BLAS rounding at the ``1e-13`` level, which the
    near-singular Godambe sandwich downstream amplified into visibly different
    credible bands between a serial and a parallel run of the *same* seed.
    """
    from scipy.linalg import solve_triangular

    scale = np.asarray(scale, dtype=float)
    m = scale.shape[0]
    if df <= m - 1:
        raise ValueError(f"inverse-Wishart needs df > dim - 1, got {df} <= {m - 1}")
    chol = _safe_cholesky(scale)
    bartlett = np.zeros((m, m), dtype=float)
    bartlett[np.diag_indices(m)] = np.sqrt(
        rng.chisquare(df - np.arange(m, dtype=float)))
    lower = np.tril_indices(m, -1)
    bartlett[lower] = rng.standard_normal(size=lower[0].size)
    return solve_triangular(bartlett, chol.T, lower=True).T


def _safe_cholesky(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    jitter = 0.0
    base = float(np.mean(np.diag(matrix)))
    for _ in range(8):
        try:
            return np.linalg.cholesky(matrix + jitter * np.eye(matrix.shape[0]))
        except np.linalg.LinAlgError:
            jitter = max(1e-12 * max(base, 1.0), jitter * 10.0)
    raise np.linalg.LinAlgError("covariance is not positive definite")


def _pca_basis(curves: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Mean and top-``k`` right singular vectors of centred ``curves``."""
    mean = curves.mean(axis=0)
    centred = curves - mean
    _, sv, vt = np.linalg.svd(centred, full_matrices=False)
    k = int(min(k, vt.shape[0]))
    total = float(np.sum(sv ** 2))
    explained = float(np.sum(sv[:k] ** 2) / total) if total > 0 else float("nan")
    return mean, vt[:k].T, explained


def _calibration_curves(cfg: MEUQConfig, grid: GridSpec) -> dict:
    """Paired (observed, clean) Alpha curves from the calibration namespace.

    Both arms of every calibration subject contribute one paired curve, pooled
    **without** the arm label, so the fitted map is a measurement-error
    correction and not a treatment-effect model.  These subjects are drawn from
    the calibration seed namespace and never appear in an evaluation replicate.
    """
    key = _noise_key(cfg.noise_level)
    ds = generate_synthetic_dataset(dgp_config(
        cfg, n=int(cfg.n_calibration_subjects),
        seed=_seed_int(cfg.seeds.calibration, 1, key),
        noise_seed=_seed_int(cfg.seeds.calibration, 2, key),
    ))
    rng = _rng(cfg.seeds.calibration, 3, key)
    obs_alpha, clean_alpha, obs_dtm, subject_ids = [], [], [], []
    for i in range(ds.clouds.shape[0]):
        for arm in (0, 1):
            noisy = _subsample(ds.clouds[i, arm], cfg.max_points, rng)
            clean = _subsample(ds.clean_clouds[i, arm], cfg.max_points, rng)
            obs_alpha.append(grid.alpha_curve(noisy))
            clean_alpha.append(grid.alpha_curve(clean))
            if cfg.use_dtm_feature:
                obs_dtm.append(grid.dtm_curve(noisy, cfg.dtm_k))
            subject_ids.append(i)
    return {
        "obs_alpha": np.stack(obs_alpha),
        "clean_alpha": np.stack(clean_alpha),
        "obs_dtm": np.stack(obs_dtm) if obs_dtm else None,
        "subject_ids": np.asarray(subject_ids, dtype=int),
    }


def _mniw_fit(Z: np.ndarray, Phi: np.ndarray, ridge: float,
              scale_prior: float) -> dict:
    q = Z.shape[1]
    m = Phi.shape[1]
    lam = Z.T @ Z + float(ridge) * np.eye(q)
    lam_inv = np.linalg.inv(lam)
    W_hat = lam_inv @ (Z.T @ Phi)
    resid_ss = Phi.T @ Phi - W_hat.T @ lam @ W_hat
    resid_ss = 0.5 * (resid_ss + resid_ss.T)
    scale = resid_ss + float(scale_prior) * np.eye(m)
    df = int(m + 2 + Z.shape[0])
    return {
        "W_hat": W_hat,
        "lambda_chol": _safe_cholesky(lam_inv),
        "scale_matrix": scale,
        "df": df,
        "sigma_hat": scale / max(df - m - 1, 1),
    }


def _gaussian_logpdf(residuals: np.ndarray, sigma: np.ndarray) -> float:
    chol = _safe_cholesky(sigma)
    m = sigma.shape[0]
    solved = np.linalg.solve(chol, residuals.T)
    quad = np.sum(solved ** 2, axis=0)
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    return float(np.mean(-0.5 * (quad + logdet + m * np.log(2.0 * np.pi))))


def fit_measurement_bridge(cfg: MEUQConfig, grid: GridSpec,
                           calibration: dict | None = None) -> MeasurementBridge:
    """Fit the validation-calibrated clean-Alpha bridge and validate it.

    The ridge penalty is chosen by **held-out predictive log-density on a
    subject-disjoint split of the calibration sample** -- never by evaluation
    coverage -- then the bridge is refitted on the full calibration sample with
    that frozen penalty.  Held-out ``clean_alpha_*`` accuracy and held-out
    pointwise/simultaneous calibration of the predictive distribution are
    recorded in ``diagnostics``.
    """
    data = calibration if calibration is not None else _calibration_curves(cfg, grid)
    obs_alpha = data["obs_alpha"]
    clean_alpha = data["clean_alpha"]
    obs_dtm = data["obs_dtm"]
    subject_ids = data["subject_ids"]
    if cfg.use_dtm_feature and obs_dtm is None:
        raise ValueError("use_dtm_feature=True but the calibration set has no DTM block")

    mean_a, basis_a, explained_a = _pca_basis(obs_alpha, cfg.bridge_k_alpha)
    mean_d = basis_d = None
    explained_d = float("nan")
    if cfg.use_dtm_feature:
        mean_d, basis_d, explained_d = _pca_basis(obs_dtm, cfg.bridge_k_dtm)

    def design(idx):
        blocks = [np.ones((idx.size, 1)),
                  (obs_alpha[idx] - mean_a) @ basis_a]
        if basis_d is not None:
            blocks.append((obs_dtm[idx] - mean_d) @ basis_d)
        return np.hstack(blocks)

    # Subject-disjoint split, so the two arms of one calibration subject never
    # straddle the fit/validation boundary.
    uniq = np.unique(subject_ids)
    rng = _rng(cfg.seeds.calibration, 9, _noise_key(cfg.noise_level))
    perm = rng.permutation(uniq)
    n_hold = max(1, int(round(cfg.calibration_holdout_frac * uniq.size)))
    hold_subjects = set(perm[:n_hold].tolist())
    hold_mask = np.array([sid in hold_subjects for sid in subject_ids])
    fit_idx = np.flatnonzero(~hold_mask)
    hold_idx = np.flatnonzero(hold_mask)
    if fit_idx.size < 4 or hold_idx.size < 2:
        raise ValueError("calibration sample is too small to validate the bridge")

    Z_fit, Z_hold = design(fit_idx), design(hold_idx)
    Y_fit, Y_hold = clean_alpha[fit_idx], clean_alpha[hold_idx]

    scores = []
    for ridge in cfg.bridge_ridge_grid:
        fit = _mniw_fit(Z_fit, Y_fit, ridge, cfg.bridge_scale_prior)
        resid = Y_hold - Z_hold @ fit["W_hat"]
        scores.append((_gaussian_logpdf(resid, fit["sigma_hat"]), float(ridge), fit))
    best_score, best_ridge, best_fit = max(scores, key=lambda item: item[0])

    # Held-out accuracy + calibration of the frozen predictive distribution.
    pred_hold = Z_hold @ best_fit["W_hat"]
    sd_hold = np.sqrt(np.clip(np.diag(best_fit["sigma_hat"]), 0.0, None))
    z975 = 1.959963984540054
    inside = np.abs(Y_hold - pred_hold) <= z975 * sd_hold
    diagnostics = {
        "holdout_ridge_logpdf": {f"{r:g}": s for s, r, _ in scores},
        "selected_ridge": best_ridge,
        "selected_holdout_logpdf": best_score,
        "n_fit_curves": int(fit_idx.size),
        "n_holdout_curves": int(hold_idx.size),
        "pca_explained_alpha": explained_a,
        "pca_explained_dtm": explained_d,
        "holdout_clean_alpha_rmse": float(np.mean([
            rmse(pred_hold[j], Y_hold[j]) for j in range(Y_hold.shape[0])])),
        "holdout_clean_alpha_nrmse": float(np.nanmean([
            nrmse(pred_hold[j], Y_hold[j]) for j in range(Y_hold.shape[0])])),
        "holdout_clean_alpha_max_abs_error": float(np.mean([
            max_abs_error(pred_hold[j], Y_hold[j]) for j in range(Y_hold.shape[0])])),
        "holdout_clean_alpha_integrated_abs_error": float(np.mean([
            integrated_abs_error(pred_hold[j], Y_hold[j], grid.grid)
            for j in range(Y_hold.shape[0])])),
        "holdout_naive_clean_alpha_rmse": float(np.mean([
            rmse(obs_alpha[hold_idx][j], Y_hold[j]) for j in range(Y_hold.shape[0])])),
        "holdout_pointwise_calibration_95": float(np.mean(inside)),
    }

    final = _mniw_fit(design(np.arange(clean_alpha.shape[0])), clean_alpha,
                      best_ridge, cfg.bridge_scale_prior)
    return MeasurementBridge(
        mean_obs_alpha=mean_a, basis_alpha=basis_a,
        mean_obs_dtm=mean_d, basis_dtm=basis_d,
        W_hat=final["W_hat"], lambda_chol=final["lambda_chol"],
        scale_matrix=final["scale_matrix"], df=final["df"],
        ridge=best_ridge, n_calibration_curves=int(clean_alpha.shape[0]),
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------- #
# Evaluation replicate: observed / oracle separation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObservedBundle:
    """Everything a non-oracle estimator is allowed to see.

    Deliberately contains **no** clean array of any kind.  A method that needs
    something absent from this bundle is, by construction, an oracle method.
    """

    phi_alpha: np.ndarray           # (n, m) contaminated Alpha silhouettes
    phi_dtm: np.ndarray | None      # (n, m) contaminated DTM silhouettes (feature)
    A: np.ndarray
    X: np.ndarray
    pi_hat: np.ndarray
    grid: np.ndarray


@dataclass(frozen=True)
class OracleBundle:
    """Simulation-truth arrays.  Only ``M1`` and the scoring code may read this."""

    phi_clean_alpha: np.ndarray          # (n, m) clean curve of the *observed* arm
    psi_clean_finite: np.ndarray         # finite-sample clean-Alpha TATE
    psi_noisy_finite: np.ndarray         # finite-sample noisy-Alpha TATE


def build_replicate(cfg: MEUQConfig, grid: GridSpec,
                    rep: int) -> tuple[ObservedBundle, OracleBundle]:
    """Materialize one evaluation replicate from the evaluation seed namespace."""
    key = _noise_key(cfg.noise_level)
    ds = generate_synthetic_dataset(dgp_config(
        cfg, n=int(cfg.n_subjects),
        seed=_seed_int(cfg.seeds.evaluation, 1, key, rep),
        noise_seed=_seed_int(cfg.seeds.evaluation, 2, key, rep),
    ))
    rng = _rng(cfg.seeds.evaluation, 3, key, rep)
    n = ds.clouds.shape[0]

    phi_alpha, phi_dtm, phi_clean = [], [], []
    clean_arms = np.zeros((n, 2, grid.resolution), dtype=float)
    noisy_arms = np.zeros((n, 2, grid.resolution), dtype=float)
    for i in range(n):
        # One subsample per (subject, arm), reused by every representation, so
        # the Alpha and DTM features of a subject always describe the same cloud.
        noisy_clouds = [_subsample(ds.clouds[i, arm], cfg.max_points, rng)
                        for arm in (0, 1)]
        for arm in (0, 1):
            clean_cloud = _subsample(ds.clean_clouds[i, arm], cfg.max_points, rng)
            noisy_arms[i, arm] = grid.alpha_curve(noisy_clouds[arm])
            clean_arms[i, arm] = grid.alpha_curve(clean_cloud)
        a = int(ds.A[i])
        phi_alpha.append(noisy_arms[i, a])
        phi_clean.append(clean_arms[i, a])
        if cfg.use_dtm_feature:
            phi_dtm.append(grid.dtm_curve(noisy_clouds[a], cfg.dtm_k))

    observed = ObservedBundle(
        phi_alpha=np.stack(phi_alpha),
        phi_dtm=np.stack(phi_dtm) if phi_dtm else None,
        A=np.asarray(ds.A, dtype=int), X=np.asarray(ds.X, dtype=float),
        pi_hat=np.asarray(ds.pi, dtype=float), grid=grid.grid,
    )
    oracle = OracleBundle(
        phi_clean_alpha=np.stack(phi_clean),
        psi_clean_finite=(clean_arms[:, 1] - clean_arms[:, 0]).mean(axis=0),
        psi_noisy_finite=(noisy_arms[:, 1] - noisy_arms[:, 0]).mean(axis=0),
    )
    return observed, oracle


# --------------------------------------------------------------------------- #
# Band containers and the five methods
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MethodResult:
    """A method's point estimate plus its pointwise and simultaneous bands."""

    estimate: np.ndarray
    pointwise_lower: np.ndarray
    pointwise_upper: np.ndarray
    simultaneous_lower: np.ndarray
    simultaneous_upper: np.ndarray
    metadata: dict = field(default_factory=dict)


def _fgp_estimator(cfg: MEUQConfig):
    from btate.causal import FunctionalGPEstimator

    return FunctionalGPEstimator(
        n_inducing=cfg.fgp_n_inducing, prior_scale=cfg.fgp_prior_scale,
        length_scale_t=cfg.fgp_length_scale_t,
        posterior_scale=cfg.fgp_posterior_scale,
    )


def _aipw_result(cfg: MEUQConfig, phi, A, X, grid, pi_hat, seed) -> MethodResult:
    from btate.benchmarks.frequentist import aipw_effect

    eff = aipw_effect(phi, A, X, grid, pi_hat=pi_hat, alpha=cfg.alpha,
                      n_basis=cfg.n_basis, cross_fit=True, n_boot=cfg.n_boot,
                      random_state=seed)
    return MethodResult(
        estimate=eff.estimate,
        pointwise_lower=eff.pointwise_lower, pointwise_upper=eff.pointwise_upper,
        simultaneous_lower=eff.simultaneous_lower,
        simultaneous_upper=eff.simultaneous_upper,
        metadata={"band": "eif_pointwise + gaussian_multiplier_simultaneous",
                  "uniform_crit": eff.metadata.get("uniform_crit")},
    )


def _posterior_result(effect, band: str) -> MethodResult:
    return MethodResult(
        estimate=effect.mean,
        pointwise_lower=effect.pointwise_lower,
        pointwise_upper=effect.pointwise_upper,
        simultaneous_lower=effect.simultaneous_lower,
        simultaneous_upper=effect.simultaneous_upper,
        metadata={"band": band,
                  "simultaneous_radius": float(effect.simultaneous_radius),
                  "pr_excludes_zero": float(effect.pr_excludes_zero)},
    )


def method_oracle_clean_aipw(cfg: MEUQConfig, observed: ObservedBundle,
                             oracle: OracleBundle, rep: int) -> MethodResult:
    """M1 -- AIPW on the clean Alpha silhouettes (upper benchmark, not practical)."""
    return _aipw_result(
        cfg, oracle.phi_clean_alpha, observed.A, observed.X, observed.grid,
        observed.pi_hat,
        _seed_int(cfg.seeds.bootstrap, 1, _noise_key(cfg.noise_level), rep),
    )


def method_blind_freq_aipw(cfg: MEUQConfig, observed: ObservedBundle,
                           rep: int) -> MethodResult:
    """M2 -- AIPW on contaminated Alpha silhouettes, treated as exact."""
    return _aipw_result(
        cfg, observed.phi_alpha, observed.A, observed.X, observed.grid,
        observed.pi_hat,
        _seed_int(cfg.seeds.bootstrap, 2, _noise_key(cfg.noise_level), rep),
    )


def method_bayes_causal_only(cfg: MEUQConfig, observed: ObservedBundle,
                             rep: int) -> MethodResult:
    """M3 -- Bayesian functional-GP causal model, no measurement-error layer."""
    from btate.causal.propagation import plugin_posterior_tate

    effect = plugin_posterior_tate(
        observed.phi_alpha, observed.A, observed.X, observed.grid,
        pi_hat=observed.pi_hat, estimator=_fgp_estimator(cfg),
        n_draws=cfg.n_plugin_draws, alpha=cfg.alpha,
        random_state=_seed_int(cfg.seeds.causal, 3, _noise_key(cfg.noise_level), rep),
    )
    return _posterior_result(effect, "posterior_supnorm_standardized")


def method_me_bayes_bridge(cfg: MEUQConfig, observed: ObservedBundle,
                           bridge: MeasurementBridge, rep: int) -> MethodResult:
    """M4 -- bridge posterior draws propagated through the causal model.

    The simultaneous band is the posterior quantile of the *curve-level*
    sup-norm standardized deviation of the pooled ``psi`` draws, not a stack of
    pointwise quantiles.
    """
    from btate.causal.propagation import nested_posterior_tate

    key = _noise_key(cfg.noise_level)
    draws = bridge.draw_clean_curves(
        observed.phi_alpha, observed.phi_dtm if bridge.uses_dtm else None,
        n_draws=cfg.n_measurement_draws,
        random_state=_seed_int(cfg.seeds.measurement_posterior, 4, key, rep),
    )
    effect = nested_posterior_tate(
        draws, observed.A, observed.X, observed.grid, pi_hat=observed.pi_hat,
        estimator=_fgp_estimator(cfg), n_causal_draws=cfg.n_causal_draws,
        alpha=cfg.alpha,
        random_state=_seed_int(cfg.seeds.causal, 4, key, rep),
    )
    result = _posterior_result(effect, "posterior_supnorm_standardized")
    result.metadata["n_measurement_draws"] = int(cfg.n_measurement_draws)
    result.metadata["bridge_uses_dtm"] = bool(bridge.uses_dtm)
    return result


def multiple_imputation_band(psis, influences, alpha: float,
                             n_boot: int, rng: np.random.Generator) -> MethodResult:
    r"""Rubin-pooled AIPW curve with a genuine curve-level sup-t band.

    Pointwise, this is Rubin's rule: with ``S`` imputations, within-imputation
    variance ``W(t) = mean_s se_s(t)^2``, between-imputation variance
    ``B(t) = var_s psi_s(t)``, total ``T(t) = W(t) + (1 + 1/S) B(t)``.

    The simultaneous band uses a multiplier bootstrap of the *pooled* process,
    so the critical value is a sup over the grid rather than a pointwise
    quantile::

        Z_b(t) = [ n^{-1} sum_i g_{b,i} IF_{s_b,i}(t)
                   + sqrt((1+1/S)/(S-1)) sum_s h_{b,s} (psi_s(t) - psibar(t)) ]
                 / sqrt(T(t))

    with ``g_b ~ N(0, I_n)``, ``h_b ~ N(0, I_S)`` and ``s_b`` uniform on the
    imputations.  By construction ``Var Z_b(t) = 1`` under the pooled variance,
    and ``crit = quantile_{1-alpha} sup_t |Z_b(t)|`` gives the sup-t band
    ``psibar +/- crit sqrt(T)``.
    """
    psis = np.asarray(psis, dtype=float)              # (S, m)
    influences = np.asarray(influences, dtype=float)  # (S, n, m)
    S, n, m = influences.shape
    if S < 2:
        raise ValueError("multiple-imputation pooling needs at least 2 imputations")

    psi_bar = psis.mean(axis=0)
    se_s = influences.std(axis=1, ddof=1) / np.sqrt(n)        # (S, m)
    within = np.mean(se_s ** 2, axis=0)
    between = psis.var(axis=0, ddof=1)
    total = within + (1.0 + 1.0 / S) * between
    total = np.maximum(total, 1e-24)
    sd = np.sqrt(total)

    centred = influences - influences.mean(axis=1, keepdims=True)
    dev_psi = psis - psi_bar
    pick = rng.integers(0, S, size=int(n_boot))
    g = rng.standard_normal(size=(int(n_boot), n))
    h = rng.standard_normal(size=(int(n_boot), S))
    within_part = np.einsum("bi,bim->bm", g, centred[pick]) / n
    between_part = np.sqrt((1.0 + 1.0 / S) / (S - 1.0)) * (h @ dev_psi)
    sup = np.max(np.abs(within_part + between_part) / sd[None, :], axis=1)
    crit = float(np.quantile(sup, 1.0 - alpha))
    z = _normal_quantile(1.0 - alpha / 2.0)

    return MethodResult(
        estimate=psi_bar,
        pointwise_lower=psi_bar - z * sd, pointwise_upper=psi_bar + z * sd,
        simultaneous_lower=psi_bar - crit * sd,
        simultaneous_upper=psi_bar + crit * sd,
        metadata={"band": "rubin_pooled + supt_multiplier_bootstrap",
                  "uniform_crit": crit, "n_imputations": int(S),
                  "mean_within_var": float(np.mean(within)),
                  "mean_between_var": float(np.mean(between))},
    )


def _normal_quantile(p: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(p))


def method_me_freq_mi(cfg: MEUQConfig, observed: ObservedBundle,
                      bridge: MeasurementBridge, rep: int) -> MethodResult:
    """M5 -- AIPW on bridge imputations with multiple-imputation pooling."""
    from btate.benchmarks.frequentist import cross_fitted_scores

    key = _noise_key(cfg.noise_level)
    draws = bridge.draw_clean_curves(
        observed.phi_alpha, observed.phi_dtm if bridge.uses_dtm else None,
        n_draws=cfg.n_measurement_draws,
        random_state=_seed_int(cfg.seeds.measurement_posterior, 5, key, rep),
    )
    psis, influences = [], []
    for s in range(draws.shape[0]):
        cf = cross_fitted_scores(
            draws[s], observed.A, observed.X, observed.grid,
            pi_hat=observed.pi_hat, n_basis=cfg.n_basis, cross_fit=True,
            random_state=_seed_int(cfg.seeds.causal, 5, key, rep, s),
        )
        psis.append(cf["aipw"])
        influences.append(cf["scores"])
    result = multiple_imputation_band(
        psis, influences, cfg.alpha, cfg.n_boot,
        _rng(cfg.seeds.bootstrap, 5, key, rep),
    )
    result.metadata["bridge_uses_dtm"] = bool(bridge.uses_dtm)
    return result


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_method(cfg: MEUQConfig, result: MethodResult, targets: PopulationTargets,
                 oracle: OracleBundle, grid: np.ndarray) -> dict:
    """All accuracy / coverage / width columns, each labelled by its target.

    Primary target is the **population** clean-Alpha TATE; the noisy-Alpha
    population TATE and both finite-sample analogues are reported alongside as
    separately labelled secondary results.
    """
    row: dict = {}
    for label in ("clean", "noisy"):
        truth = targets.target(label)
        row.update(curve_error_columns(result.estimate, truth, grid, target=label))
        row.update(band_coverage_columns(
            result.pointwise_lower, result.pointwise_upper,
            result.simultaneous_lower, result.simultaneous_upper,
            truth, grid, target=label, alpha=cfg.alpha,
            peak_window=cfg.peak_window,
        ))
    # Secondary, explicitly-labelled finite-sample estimands.
    for label, truth in (("clean", oracle.psi_clean_finite),
                         ("noisy", oracle.psi_noisy_finite)):
        row.update(curve_error_columns(result.estimate, truth, grid,
                                       target=label, prefix="finite"))
        row.update(band_coverage_columns(
            result.pointwise_lower, result.pointwise_upper,
            result.simultaneous_lower, result.simultaneous_upper,
            truth, grid, target=label, prefix="finite", alpha=cfg.alpha,
            peak_window=cfg.peak_window,
        ))
    row["estimand_primary"] = "psi_clean_alpha_population"
    row["estimand_secondary"] = "psi_noisy_alpha_population"
    row["apex_index_clean"] = int(peak_index(targets.psi_clean_alpha))
    row["apex_t_clean"] = float(grid[peak_index(targets.psi_clean_alpha)])
    return row


# --------------------------------------------------------------------------- #
# Replicate driver
# --------------------------------------------------------------------------- #
def evaluate_replicate(cfg: MEUQConfig, grid: GridSpec, targets: PopulationTargets,
                       bridges: dict[str, MeasurementBridge], rep: int) -> list[dict]:
    """Run every method on one evaluation replicate and score them identically.

    ``bridges`` maps ``"alpha"`` (and optionally ``"dtm"``) to a frozen fitted
    bridge.  Each method is wrapped so a failure yields a row with
    ``failed=True`` and NaN metrics rather than killing the shard.
    """
    observed, oracle = build_replicate(cfg, grid, rep)
    base = {
        "noise_level": float(cfg.noise_level),
        "rep": int(rep),
        "n_subjects": int(observed.A.shape[0]),
        "resolution": int(cfg.resolution),
        "grid_upper": float(grid.sample_range[1]),
        "alpha": float(cfg.alpha),
        "n_measurement_draws": int(cfg.n_measurement_draws),
        "n_oracle_subjects": int(targets.n_oracle_subjects),
        "treated_frac": float(np.mean(observed.A)),
    }

    runners = {
        "oracle_clean_aipw":
            lambda: method_oracle_clean_aipw(cfg, observed, oracle, rep),
        "blind_freq_aipw":
            lambda: method_blind_freq_aipw(cfg, observed, rep),
        "bayes_causal_only":
            lambda: method_bayes_causal_only(cfg, observed, rep),
        "me_bayes_bridge":
            lambda: method_me_bayes_bridge(cfg, observed, bridges["alpha"], rep),
        "me_freq_mi":
            lambda: method_me_freq_mi(cfg, observed, bridges["alpha"], rep),
    }
    if cfg.use_dtm_feature:
        runners["me_bayes_bridge_dtm"] = (
            lambda: method_me_bayes_bridge(cfg, observed, bridges["dtm"], rep))
        runners["me_freq_mi_dtm"] = (
            lambda: method_me_freq_mi(cfg, observed, bridges["dtm"], rep))

    rows: list[dict] = []
    for name, runner in runners.items():
        row = dict(base)
        row["method"] = name
        row["is_oracle"] = name in ORACLE_METHODS
        row["measurement_error_aware"] = name.startswith("me_")
        row["bridge_uses_dtm"] = name.endswith("_dtm")
        t0 = time.perf_counter()
        try:
            result = runner()
            row.update(score_method(cfg, result, targets, oracle, grid.grid))
            row["failed"] = False
            row["error"] = ""
            row["band_kind"] = str(result.metadata.get("band", ""))
        except Exception as exc:  # keep the shard alive; the row records the failure
            row["failed"] = True
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["band_kind"] = ""
        row["runtime_s"] = time.perf_counter() - t0
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Cell driver, caching and aggregation
# --------------------------------------------------------------------------- #
def prepare_cell(cfg: MEUQConfig, cache_dir: str | Path | None = None,
                 verbose: bool = False) -> dict:
    """Freeze the grid, the population targets and the bridge(s) for one cell.

    Cached to ``cache_dir`` when given.  Keep that directory **outside** the
    published package: these artifacts encode simulation truth.
    """
    grid = freeze_alpha_grid(cfg)
    if verbose:
        print(f"[prepare] frozen alpha grid {grid.sample_range} "
              f"res={grid.resolution}")
    targets = population_targets(cfg, grid)
    if verbose:
        print(f"[prepare] population targets from {targets.n_oracle_subjects} "
              f"oracle subjects; peak(clean)="
              f"{np.max(np.abs(targets.psi_clean_alpha)):.4f} "
              f"peak(noisy)={np.max(np.abs(targets.psi_noisy_alpha)):.4f}")

    calibration = _calibration_curves(cfg, grid)
    bridges = {"alpha": fit_measurement_bridge(
        replace(cfg, use_dtm_feature=False), grid,
        calibration={**calibration, "obs_dtm": None})}
    if cfg.use_dtm_feature:
        bridges["dtm"] = fit_measurement_bridge(cfg, grid, calibration=calibration)
    if verbose:
        for name, bridge in bridges.items():
            d = bridge.diagnostics
            print(f"[prepare] bridge[{name}] ridge={bridge.ridge:g} "
                  f"holdout clean_alpha_rmse={d['holdout_clean_alpha_rmse']:.5f} "
                  f"(naive {d['holdout_naive_clean_alpha_rmse']:.5f})")

    out = {"config": cfg, "grid": grid, "targets": targets, "bridges": bridges}
    if cache_dir is not None:
        path = Path(cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        tag = f"noise{_noise_key(cfg.noise_level):06d}_res{cfg.resolution}"
        (path / f"meuq_prepared_{tag}.json").write_text(json.dumps({
            "config": cfg.to_dict(),
            "grid": grid.to_dict(),
            "targets": targets.to_dict(),
            "bridges": {k: v.to_dict() for k, v in bridges.items()},
        }, indent=2))
    return out


def run_me_uq_cell(cfg: MEUQConfig, rep_indices, *, cache_dir=None, n_jobs: int = 1,
                   prepared: dict | None = None, verbose: bool = False):
    """Evaluate ``rep_indices`` for one cell and return a tidy DataFrame."""
    import pandas as pd

    prepared = prepared or prepare_cell(cfg, cache_dir=cache_dir, verbose=verbose)
    grid, targets, bridges = (prepared["grid"], prepared["targets"],
                              prepared["bridges"])
    reps = [int(r) for r in rep_indices]
    if not reps:
        return pd.DataFrame()

    if n_jobs == 1:
        chunks = [evaluate_replicate(cfg, grid, targets, bridges, rep)
                  for rep in reps]
    else:
        from joblib import Parallel, delayed

        workers = max(1, min(int(n_jobs), len(reps)))
        chunks = Parallel(n_jobs=workers, backend="loky",
                          verbose=5 if verbose else 0)(
            delayed(evaluate_replicate)(cfg, grid, targets, bridges, rep)
            for rep in reps
        )
    return pd.DataFrame([row for chunk in chunks for row in chunk])


def aggregate_me_uq(raw, alpha: float = 0.05):
    """Aggregate per-replicate rows into per ``(noise, method)`` summaries.

    Simultaneous coverage is a **repeated-sampling** rate: the proportion of
    replicates whose entire target curve lies inside the band, with an exact
    Clopper--Pearson interval attached.  The pointwise-coverage columns are
    per-replicate grid fractions and are averaged; they are *not* Monte-Carlo
    calibration and are named so they cannot be mistaken for it.
    """
    import pandas as pd

    from btate.benchmarks.metrics import coverage_rate_with_ci

    raw = pd.DataFrame(raw)
    if raw.empty:
        return pd.DataFrame()
    if not {"noise_level", "method", "rep"} <= set(raw.columns):
        raise ValueError("raw frame needs noise_level, method and rep columns")

    sim_columns = [c for c in raw.columns if c.startswith(("cov_sim", "finite_cov_sim"))]
    sim_columns += [c for c in raw.columns
                    if c.endswith(("cov_peak_localized_clean", "cov_peak_localized_noisy"))]
    sim_columns = sorted(set(sim_columns))
    mean_columns = [
        c for c in raw.columns
        if c not in sim_columns
        and c not in {"noise_level", "method", "rep", "error", "estimand_primary",
                      "estimand_secondary", "band_kind"}
        and pd.api.types.is_numeric_dtype(raw[c])
    ]

    rows = []
    for (noise, method), cell in raw.groupby(["noise_level", "method"]):
        ok = cell[~cell["failed"].astype(bool)] if "failed" in cell else cell
        row = {
            "noise_level": float(noise),
            "method": method,
            "n_reps": int(len(cell)),
            "n_failed": int(len(cell) - len(ok)),
            "estimand_primary": "psi_clean_alpha_population",
        }
        for column in sim_columns:
            stats = coverage_rate_with_ci(ok[column], alpha=alpha)
            row[f"{column}_rate"] = stats["coverage"]
            row[f"{column}_cp_lower"] = stats["cp_lower"]
            row[f"{column}_cp_upper"] = stats["cp_upper"]
            row[f"{column}_n"] = stats["n_replicates"]
        for column in mean_columns:
            values = ok[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{column}_mean"] = float(np.mean(values)) if values.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["noise_level", "method"]).reset_index(drop=True)


def combine_shards(frames, dedup_keys=("noise_level", "rep", "method")):
    """Concatenate shard frames, drop duplicate rows and sort deterministically.

    Result must not depend on shard order or on a shard being re-run after a
    restart, so duplicates are dropped on ``(noise_level, rep, method)`` keeping
    the first occurrence after a deterministic sort.
    """
    import pandas as pd

    frames = [pd.DataFrame(f) for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    keys = [k for k in dedup_keys if k in out.columns]
    if keys:
        out = out.sort_values(keys, kind="mergesort")
        out = out.drop_duplicates(subset=keys, keep="first")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Collapse / invariance self-checks
# --------------------------------------------------------------------------- #
def algorithmic_collapse_report(cfg: MEUQConfig, grid: GridSpec,
                                cloud) -> dict:
    """Exact-collapse check: identical clean inputs must give zero error.

    Feeding one cloud in as both the "observed" and the "clean" input must give
    a bit-identical curve and exactly zero error under every representation
    metric.  This is an algorithmic identity, independent of the DGP -- unlike
    the noise-0 control, where two *independently sampled* clean clouds are
    legitimately different.
    """
    curve_a = grid.alpha_curve(cloud)
    curve_b = grid.alpha_curve(cloud)
    return {
        "max_curve_discrepancy": float(np.max(np.abs(curve_a - curve_b))),
        "rmse": rmse(curve_a, curve_b),
        "nrmse": nrmse(curve_a, curve_b),
        "max_abs_error": max_abs_error(curve_a, curve_b),
        "integrated_abs_error": integrated_abs_error(curve_a, curve_b, grid.grid),
    }
