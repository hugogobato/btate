"""Ground-truth synthetic study with controlled loop injection (Task 4.1).

This module builds a fully synthetic causal-topological data-generating process
(DGP) in which the Topological Average Treatment Effect ``psi_d(t)`` is *known*
by construction, so bias, RMSE, coverage, interval width, and test power can be
measured against a reference curve.

Design
------
Each subject :math:`i` has a covariate vector :math:`X_i` and a binary treatment
:math:`A_i` drawn from a propensity model whose steepness controls *overlap*.
Both potential outcomes are 2-D point clouds:

* **control arm** — a noisy circle (a single :math:`H_1` loop) of radius
  ``base_radius`` (with a small covariate-driven perturbation for heterogeneity),
  plus uniform *clutter* points that create spurious short-lived features;
* **treated arm** — the same construction with the loop radius enlarged by
  ``effect_size`` (``effect_size=0`` gives a true null for type-I error studies).

Because a larger loop is born and dies later under the Alpha/Rips filtration, the
treatment deterministically shifts the persistence silhouette, so the population
silhouette effect is a smooth, known function of the filtration parameter.

Noise / sample-size / overlap knobs
-----------------------------------
* ``noise_level`` scales the radial jitter *and* the clutter count (topological
  noise the Bayesian ``pi_p`` weighting is meant to be robust to);
* ``n`` is the number of subjects;
* ``overlap_strength`` is the propensity-logit slope — larger values push
  propensities toward 0/1 (weaker overlap).

The *reference* ``psi_d(t)`` for a given embedding configuration is computed on
the clean, dense, clutter-free potential-outcome diagrams (see
:func:`reference_effect`), which is exactly the estimand the pipeline targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class SyntheticConfig:
    """Configuration for the loop-injection DGP."""

    n: int = 60
    num_pts: int = 120
    base_radius: float = 0.35
    effect_size: float = 0.10
    noise_level: float = 1.0
    overlap_strength: float = 0.8
    n_covariates: int = 3
    radius_covariate_gain: float = 0.03
    effect_covariate_gain: float = 0.02
    base_jitter: float = 0.02
    base_clutter: int = 12
    center: tuple[float, float] = (0.5, 0.5)
    seed: int = 20260701
    noise_seed: int | None = None   # separate stream for jitter/clutter (MC estimand)

    def rng(self, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)

    def noise_rng(self) -> np.random.Generator:
        base = self.seed + 777 if self.noise_seed is None else self.noise_seed
        return np.random.default_rng(base)


@dataclass
class SyntheticDataset:
    """One realized synthetic dataset.

    Attributes
    ----------
    clouds : np.ndarray
        Potential-outcome point clouds, shape ``(n, 2, num_pts + clutter, 2)``
        with axis 1 ordered ``(control, treated)``.
    X, A, pi : np.ndarray
        Covariates ``(n, p)``, treatment ``(n,)``, true propensity ``(n,)``.
    clean_clouds : np.ndarray
        Dense, clutter-free clouds used to define the reference estimand,
        shape ``(n, 2, num_clean, 2)``.
    config : SyntheticConfig
    """

    clouds: np.ndarray
    X: np.ndarray
    A: np.ndarray
    pi: np.ndarray
    clean_clouds: np.ndarray
    config: SyntheticConfig
    meta: dict = field(default_factory=dict)

    def observed_clouds(self) -> list[np.ndarray]:
        """Return the observed point cloud per subject (arm selected by ``A``)."""
        return [self.clouds[i, int(self.A[i])] for i in range(self.clouds.shape[0])]


def _logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _noisy_circle(radius: float, num_pts: int, jitter: float,
                  center, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=num_pts)
    r = radius + rng.normal(0.0, jitter, size=num_pts)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack([x, y])


def _clutter(n_clutter: int, center, radius: float,
             rng: np.random.Generator) -> np.ndarray:
    """Peripheral clutter in an outer annulus.

    Clutter is placed *outside* the signal loop (radii in ``[1.2R, 1.2R + 0.25]``)
    so it creates spurious short-lived H1 features without filling the central
    loop and destroying its persistence — the loop's death time (hence the
    treatment effect) is preserved while topological noise increases.
    """
    if n_clutter <= 0:
        return np.empty((0, 2), dtype=float)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_clutter)
    r = 1.2 * radius + rng.uniform(0.0, 0.25, size=n_clutter)
    return np.column_stack([
        center[0] + r * np.cos(theta),
        center[1] + r * np.sin(theta),
    ])


def _arm_cloud(radius: float, cfg: SyntheticConfig, jitter: float,
               n_clutter: int, rng: np.random.Generator) -> np.ndarray:
    loop = _noisy_circle(radius, cfg.num_pts, jitter, cfg.center, rng)
    clutter = _clutter(n_clutter, cfg.center, radius, rng)
    cloud = np.vstack([loop, clutter])
    # Pad to a fixed width so realizations stack into one array.
    return cloud


def generate_synthetic_dataset(config: SyntheticConfig) -> SyntheticDataset:
    """Realize one synthetic causal-topological dataset from ``config``.

    Structural randomness (covariates, treatment, per-subject radii) is drawn
    from ``config.seed``; the topological noise (jitter / clutter locations) is
    drawn from a *separate* ``config.noise_seed`` stream.  Holding ``seed`` fixed
    while varying ``noise_seed`` gives independent noise realizations of the same
    population — the basis for the Monte-Carlo estimand in
    :func:`montecarlo_reference`.
    """
    cfg = config
    rng = cfg.rng()          # structural stream
    nrng = cfg.noise_rng()   # topological-noise stream
    p = cfg.n_covariates
    X = rng.normal(size=(cfg.n, p))

    # Propensity: steeper slope -> weaker overlap.
    coef = np.zeros(p)
    coef[: min(2, p)] = np.array([0.9, -0.6])[: min(2, p)]
    eta = cfg.overlap_strength * (X @ coef)
    pi = np.clip(_logit(eta), 0.03, 0.97)
    A = rng.binomial(1, pi)
    if len(np.unique(A)) < 2:  # guarantee both arms present
        A[: cfg.n // 2] = 0
        A[cfg.n // 2:] = 1
    radii = np.array([cfg.base_radius + cfg.radius_covariate_gain * X[i, 0]
                      for i in range(cfg.n)])
    effs = np.array([cfg.effect_size + cfg.effect_covariate_gain * X[i, min(1, p - 1)]
                     for i in range(cfg.n)])

    jitter = cfg.base_jitter * cfg.noise_level
    n_clutter = int(round(cfg.base_clutter * cfg.noise_level))
    width = cfg.num_pts + n_clutter
    # The clean reference uses the *same* loop-sampling density as the observed
    # clouds (no clutter, negligible jitter), so the estimand isolates the
    # topological-noise / clutter effect rather than a point-count artifact.
    clean_pts = cfg.num_pts

    clouds = np.zeros((cfg.n, 2, width, 2), dtype=float)
    clean = np.zeros((cfg.n, 2, clean_pts, 2), dtype=float)
    for i in range(cfg.n):
        r0, r1 = radii[i], radii[i] + effs[i]
        for arm, radius in ((0, r0), (1, r1)):
            noisy = _arm_cloud(radius, cfg, jitter, n_clutter, nrng)
            clouds[i, arm, : noisy.shape[0]] = noisy
            if noisy.shape[0] < width:  # pad by repeating loop points
                pad = width - noisy.shape[0]
                clouds[i, arm, noisy.shape[0]:] = noisy[nrng.integers(0, noisy.shape[0], size=pad)]
            # Clean reference: same density, no clutter, negligible jitter.
            clean[i, arm] = _noisy_circle(radius, clean_pts, cfg.base_jitter * 0.1,
                                          cfg.center, nrng)

    return SyntheticDataset(
        clouds=clouds, X=X, A=A, pi=pi, clean_clouds=clean, config=cfg,
        meta={
            "jitter": jitter,
            "n_clutter": n_clutter,
            "loop_points": cfg.num_pts,
            "clean_points": clean_pts,
            "treated_frac": float(np.mean(A)),
            "propensity_range": [float(pi.min()), float(pi.max())],
        },
    )


def reference_effect(dataset: SyntheticDataset, embedding_fn, tseq) -> np.ndarray:
    """Clean-truth reference ``psi_d(t)`` from the noise-free potential outcomes.

    ``embedding_fn`` maps a single point cloud to a curve on ``tseq`` (e.g. a
    fixed-``r`` silhouette or a landscape level).  This is the *injected-loop*
    estimand — the target used to measure denoising **bias / RMSE**.  Because
    geometric noise reduces a loop's persistence, the noisy-data estimator is
    biased for this target under noise (that bias is the robustness signal).
    """
    clean = dataset.clean_clouds
    n = clean.shape[0]
    control = np.stack([embedding_fn(clean[i, 0]) for i in range(n)])
    treated = np.stack([embedding_fn(clean[i, 1]) for i in range(n)])
    return treated.mean(axis=0) - control.mean(axis=0)


def montecarlo_reference(config: SyntheticConfig, embedding_fn,
                         n_realizations: int = 40, noise_seed0: int = 100000) -> np.ndarray:
    """Self-consistent Monte-Carlo estimand ``psi*_d(t)`` the estimator targets.

    Averages the *observed* (noisy) treated-minus-control embedding over many
    independent noise realizations of the same population (fixed ``config.seed``,
    varying ``noise_seed``).  This is the conditional-mean functional effect the
    outcome-regression targets, so credible-band **coverage** measured against
    ``psi*`` isolates calibration from the (separate) denoising bias captured by
    :func:`reference_effect`.
    """
    acc = None
    for k in range(n_realizations):
        cfg_k = replace(config, noise_seed=noise_seed0 + k)
        ds = generate_synthetic_dataset(cfg_k)
        n = ds.clouds.shape[0]
        eff = np.mean(
            [embedding_fn(ds.clouds[i, 1]) - embedding_fn(ds.clouds[i, 0]) for i in range(n)],
            axis=0,
        )
        acc = eff if acc is None else acc + eff
    return acc / float(n_realizations)
