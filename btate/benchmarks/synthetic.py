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
    """Configuration for the loop-injection DGP.

    Two clutter regimes (``clutter_mode``) trade off how much *denoising* the
    problem actually needs:

    * ``"annulus"`` (default, the standard Phase-4 DGP) — peripheral clutter in
      an outer annulus creates only *short-lived* spurious H1 features that leave
      the signal loop's death time (hence the treatment effect) intact.  The raw
      fixed-``r`` silhouette is already near-optimal here, so Step-1 denoising has
      little to recover (the 2026-07-03 finding).
    * ``"structured_loops"`` (Task 4.5.2 low-SNR regime) — the clutter is a set
      of secondary *decoy circles* of persistence comparable to the signal loop,
      present only in the observed clouds (never in the clean estimand).  Because
      the power-weighted silhouette sums ``|d-b|^r`` over *all* features, the
      decoys contaminate the raw silhouette; a working Step-2 ``pi_p`` weighting /
      Maroulas denoising should down-weight them and recover the clean effect.
      This is the regime where the Bayesian pipeline can genuinely beat the
      raw-silhouette frequentist (or, if it cannot, that is the honest negative
      result of Phase 4.5).
    """

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
    # --- low-SNR / heavy-clutter regime (Task 4.5.2) ---
    clutter_mode: str = "annulus"          # "annulus" | "structured_loops"
    n_decoy_loops: int = 4                  # decoy circles per NOISE unit
    decoy_points: int = 22                  # points per decoy circle
    decoy_radius_frac: float = 0.55         # decoy radius / signal radius
    decoy_radius_jitter: float = 0.18       # rel. spread of decoy radii across decoys
    decoy_jitter_frac: float = 1.0          # decoy radial jitter (x base_jitter*noise)
    decoy_spread: float = 1.6               # decoy-center distance (x signal radius)
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


def _decoy_loops(n_loops: int, points_per: int, cfg: SyntheticConfig,
                 radius: float, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Structured clutter: secondary circles of persistence near the signal.

    Each decoy is a noisy circle whose radius is a jittered fraction of the
    signal radius (so its H1 feature has *moderate* persistence, comparable to
    the signal loop), centred at a random offset from the signal centre.  These
    are the features a working ``pi_p`` / Maroulas denoiser must suppress; the
    raw power-weighted silhouette cannot.
    """
    if n_loops <= 0 or points_per <= 0:
        return np.empty((0, 2), dtype=float)
    parts = []
    base_r = cfg.decoy_radius_frac * radius
    for _ in range(n_loops):
        rk = base_r * (1.0 + cfg.decoy_radius_jitter * rng.normal())
        rk = max(rk, 0.05 * radius)
        ang = rng.uniform(0.0, 2.0 * np.pi)
        dist = cfg.decoy_spread * radius * np.sqrt(rng.uniform(0.15, 1.0))
        cx = cfg.center[0] + dist * np.cos(ang)
        cy = cfg.center[1] + dist * np.sin(ang)
        parts.append(_noisy_circle(rk, points_per, jitter, (cx, cy), rng))
    return np.vstack(parts)


def _arm_cloud(radius: float, cfg: SyntheticConfig, signal_jitter: float,
               n_clutter: int, n_decoy: int, decoy_jitter: float,
               rng: np.random.Generator) -> np.ndarray:
    loop = _noisy_circle(radius, cfg.num_pts, signal_jitter, cfg.center, rng)
    parts = [loop, _clutter(n_clutter, cfg.center, radius, rng)]
    if cfg.clutter_mode == "structured_loops":
        parts.append(_decoy_loops(n_decoy, cfg.decoy_points, cfg, radius,
                                  decoy_jitter, rng))
    cloud = np.vstack(parts)
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
    if cfg.clutter_mode == "structured_loops":
        # Low-SNR regime: the *signal* loop stays clean and stably persistent
        # (jitter fixed at base_jitter, not amplified by noise), while
        # ``noise_level`` scales the number of competing moderate-persistence
        # decoys.  Decoys use a small, noise-independent jitter so their
        # persistence forms a tight band that sits *below* the signal with a
        # gap — separable in principle, so the DP partition can suppress them.
        signal_jitter = cfg.base_jitter
        n_decoy = int(round(cfg.n_decoy_loops * cfg.noise_level))
        decoy_jitter = cfg.decoy_jitter_frac * cfg.base_jitter
        decoy_total = n_decoy * cfg.decoy_points
    else:
        signal_jitter = jitter
        n_decoy = 0
        decoy_jitter = jitter
        decoy_total = 0
    width = cfg.num_pts + n_clutter + decoy_total
    # The clean reference uses the *same* loop-sampling density as the observed
    # clouds (no clutter, negligible jitter), so the estimand isolates the
    # topological-noise / clutter effect rather than a point-count artifact.
    clean_pts = cfg.num_pts

    clouds = np.zeros((cfg.n, 2, width, 2), dtype=float)
    clean = np.zeros((cfg.n, 2, clean_pts, 2), dtype=float)
    for i in range(cfg.n):
        r0, r1 = radii[i], radii[i] + effs[i]
        for arm, radius in ((0, r0), (1, r1)):
            noisy = _arm_cloud(radius, cfg, signal_jitter, n_clutter, n_decoy,
                               decoy_jitter, nrng)
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
            "clutter_mode": cfg.clutter_mode,
            "n_decoy_loops": n_decoy,
            "decoy_points_total": decoy_total,
            "loop_points": cfg.num_pts,
            "clean_points": clean_pts,
            "treated_frac": float(np.mean(A)),
            "propensity_range": [float(pi.min()), float(pi.max())],
        },
    )


def standard_config(n: int = 120, noise_level: float = 1.0,
                    effect_size: float = 0.12, overlap_strength: float = 0.8,
                    seed: int = 20260701, **overrides) -> SyntheticConfig:
    """The standard Phase-4 annulus-clutter DGP (raw silhouette near-optimal)."""
    return replace(
        SyntheticConfig(
            n=n, noise_level=noise_level, effect_size=effect_size,
            overlap_strength=overlap_strength, seed=seed, clutter_mode="annulus",
        ),
        **overrides,
    )


def low_snr_config(n: int = 120, noise_level: float = 2.0,
                   effect_size: float = 0.12, overlap_strength: float = 0.8,
                   seed: int = 20260701, **overrides) -> SyntheticConfig:
    """The Task-4.5.2 low-SNR / heavy-clutter DGP (raw silhouette contaminated).

    A clean, stably-persistent signal loop is surrounded by many competing
    moderate-persistence decoy circles (count scales with ``noise_level``), so
    the power-weighted silhouette sums spurious mass the clean estimand does not
    contain.  This is the regime where Step-1 / ``pi_p`` denoising *can* earn its
    keep — or, if it cannot, where the honest negative result of Phase 4.5 is
    established.  Defaults are locked here so the sweep and notebook share one
    definition.
    """
    return replace(
        SyntheticConfig(
            n=n, noise_level=noise_level, effect_size=effect_size,
            overlap_strength=overlap_strength, seed=seed,
            clutter_mode="structured_loops",
            base_radius=0.35, num_pts=140, base_clutter=12,
            n_decoy_loops=8, decoy_points=24, decoy_radius_frac=0.45,
            decoy_radius_jitter=0.18, decoy_jitter_frac=1.0, decoy_spread=1.6,
        ),
        **overrides,
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
