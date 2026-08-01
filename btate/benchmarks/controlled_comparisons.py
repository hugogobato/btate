r"""Phase 6.25 - controlled comparisons on the frozen synthetic mechanism.

Two pieces of cross-phase folklore are converted into scored, falsifiable
results here, on the **already frozen** Phase-6 DGP, target, grid, calibration
sample and evaluation harness:

1. *"The topological error model was tried and lost to ordinary regression
   calibration"* - never a single table.  This phase builds M6
   (``me_topo_posterior``), a fairly-calibrated **topological** measurement-error
   model, and scores it head-to-head against the ordinary bridge (M4/M5).  It
   also runs M6-point (``me_topo_point``), the Phase-4.25/5 style **point**
   denoiser, as a separate arm so the table can separate "topology loses" from
   "point correction loses".

2. *"The Gaussian bridge operates on ``R^96`` and knows nothing about persistence
   diagrams"* - currently a fair criticism with no measurement attached.  This
   phase measures, per posterior draw of every arm that emits latent clean
   curves, the fraction of draws that violate the **valid-silhouette
   polyhedron** (C1-C3, unconditional necessary conditions read off
   ``btate/embeddings/silhouette.py:117``), and scores the **polyhedron-projected**
   arms (``*_proj``) to decide whether enforcing validity helps, hurts, or does
   nothing.

Scope guardrail (from Research_Plan Task 6.25.4): this phase introduces **no
new estimand, no new DGP and no new target**.  Everything is scored against the
frozen population ``psi_clean_alpha`` on the frozen grid, with the frozen
calibration sample and the unchanged Phase-6 harness.

M6, stated precisely
--------------------
The Maroulas posterior (Maroulas, Nasrin & Oballe 2020) is a marked-PPP model
whose closed-form posterior intensity over the birth-persistence wedge is a
restricted Gaussian mixture.  M6 draws diagrams from the **per-subject
posterior** (prior/clutter/sigma calibrated once on the Phase-6 calibration
sample, pooled without the arm label; the subject's own observed diagram is the
likelihood), computes the power silhouette of **each draw** on the frozen grid,
and propagates the resulting spread through the same nested FGP band machinery
that M4 uses (``nested_posterior_tate``).  This carries an explicit ``D | D~``
uncertainty term - exactly the ingredient Phase 6 identified as necessary.

Fairness: M4's bridge is fitted on the independent paired calibration sample;
M6 uses the **same paired clean/observed information**, pooled across arms
without the treatment label.  Its prior is elicited from clean diagrams, its
clutter template from observed diagrams, and the prior complexity, clutter
scale, ``alpha`` and ``sigma_DYO`` are selected by the conditional predictive
log score of held-out clean diagrams given their paired observed diagrams.
Subjects, rather than curves, define the split.  The selected structure is then
refitted on the full calibration sample and frozen.  This is what makes M6 a
validation-calibrated clean-target model rather than a marginal model for the
contaminated diagrams.

The valid-silhouette polyhedron (C1-C3)
---------------------------------------
Under the frozen grid with spacing ``Delta``, every genuine power-weighted
silhouette (``weighted_silhouette``, ``btate/embeddings/silhouette.py:117``)
satisfies, with ``L = sqrt(2)`` the implementation's Lipschitz constant:

* **C1** non-negativity: ``phi(t) >= 0``;
* **C2** Lipschitz bound: ``|phi_{i+1} - phi_i| <= L * Delta``;
* **C3** left support: ``phi(0) = 0`` because Alpha births are non-negative.

C1-C3 are **necessary, not sufficient** (the achievable set is the non-convex
image of diagram space under the silhouette map; the polyhedron is a convex
outer approximation), so a violation is a sound certificate of invalidity
while satisfaction proves nothing.  A finite evaluation-grid endpoint does
not bound diagram persistence, so neither a grid-derived height cap nor zero at
the right endpoint is imposed.  Those formerly registered conditions rejected
genuine silhouettes of features whose deaths extend beyond the grid.

M6 draws are silhouettes of sampled diagrams and therefore satisfy C1-C3 for
every draw.  Any M6 violation falsifies the diagnostic or the implementation.

New seed streams
----------------
All randomness stays inside the Phase-6 seed namespaces.  Stream index ``6`` of
the ``measurement_posterior`` namespace drives M6's per-subject diagram draws;
stream ``6`` of ``causal`` drives M6's nested FGP; stream ``6`` of
``bootstrap`` drives M6-point's multiplier band.  The projection arms share the
draw, causal and bootstrap streams of their base arms bit-for-bit, so
projection is the *only* difference between an arm and its ``_proj`` version.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from btate.benchmarks.measurement_error_uq import (
    MEUQConfig,
    MethodResult,
    _aipw_result,
    _fgp_estimator,
    _noise_key,
    _posterior_result,
    _rng,
    _seed_int,
    _subsample,
    alpha_diagram,
    build_replicate,
    dgp_config,
    method_bayes_causal_only,
    method_blind_freq_aipw,
    method_me_bayes_bridge,
    method_me_freq_mi,
    method_oracle_clean_aipw,
    prepare_cell,
    score_method,
    silhouette_on_grid,
)
from btate.benchmarks.synthetic import generate_synthetic_dataset
from btate.embeddings.silhouette import weighted_silhouette
from btate.topo_posterior.adapters import bd_to_bp, bp_to_bd
from btate.topo_posterior.elicitation import elicit_prior_clutter
from btate.topo_posterior.sampler import PosteriorDiagramSampler

# --------------------------------------------------------------------------- #
# The six arms added by Phase 6.25
# --------------------------------------------------------------------------- #
P625_METHODS = (
    "oracle_clean_aipw",
    "blind_freq_aipw",
    "bayes_causal_only",
    "me_bayes_bridge",
    "me_freq_mi",
    "me_bayes_bridge_proj",       # M4-proj: projected onto the polyhedron
    "me_freq_mi_proj",            # M5-proj: projected onto the polyhedron
    "me_topo_posterior",          # M6: topological posterior over diagrams
    "me_topo_point",              # M6-point: AIPW on the posterior-mean curve
    "me_topo_posterior_proj",     # M6-proj: projected onto the polyhedron
)
DTM_P625_METHODS = ("me_bayes_bridge_dtm", "me_freq_mi_dtm")

P625_CACHE_VERSION = 3
M6_ALPHA_GRID = (0.50, 0.75, 1.00)
M6_CLUTTER_SCALE_GRID = (0.50, 1.00, 2.00)
M6_SIGMA_MULTIPLIER_GRID = tuple(np.geomspace(0.02, 20.0, 9))

# --------------------------------------------------------------------------- #
# The unconditional valid-silhouette polyhedron (Task 6.25.2)
# --------------------------------------------------------------------------- #
def derive_lipschitz_constant() -> float:
    """The silhouette Lipschitz constant, re-derived from the implementation.

    ``weighted_silhouette`` (``btate/embeddings/silhouette.py:117``) computes
    ``phi(t) = sqrt(2) * (tents @ (w / total))``.  Each tent
    ``max(heights - |t - midpoints|, 0)`` is 1-Lipschitz in ``t`` (slope ``+1``
    on the left arm, ``-1`` on the right), a normalised convex combination of
    1-Lipschitz functions is 1-Lipschitz, and the final scaling is ``sqrt(2)``.
    So ``L = sqrt(2)``.  The unit tests assert this constant empirically on
    random diagrams: the bound holds for every diagram, and a smaller constant
    is violated by some diagram (so the constant is not vacuous).
    """
    from btate.embeddings.silhouette import weighted_silhouette as _ws

    import inspect

    source = inspect.getsource(_ws)
    if "np.sqrt(2.0)" not in source:
        raise AssertionError("weighted_silhouette no longer scales by sqrt(2)")
    return float(np.sqrt(2.0))


@dataclass(frozen=True)
class SilhouettePolyhedron:
    """C1-C3 outer approximation of the valid-silhouette set on one grid."""

    grid: np.ndarray
    delta: float
    lipschitz_constant: float
    slope: float
    grid_upper: float

    def to_dict(self) -> dict:
        return {
            "delta": float(self.delta),
            "lipschitz_constant": float(self.lipschitz_constant),
            "slope": float(self.slope),
            "grid_upper": float(self.grid_upper),
            "constraints": [
                "C1_nonnegative",
                "C2_sqrt2_lipschitz",
                "C3_left_endpoint_zero",
            ],
        }


def silhouette_polyhedron(
    grid, lipschitz_constant: float | None = None,
) -> SilhouettePolyhedron:
    """Return the unconditional C1-C3 outer approximation on ``grid``.

    The right grid endpoint is an evaluation boundary, not a support bound on
    persistence diagrams.  The former height cap and right-endpoint pin are
    therefore deliberately absent.
    """
    grid = np.asarray(grid, dtype=float).ravel()
    if grid.size < 2:
        raise ValueError("a silhouette polyhedron needs at least two grid points")
    spacings = np.diff(grid)
    if np.any(spacings <= 0.0):
        raise ValueError("the silhouette polyhedron requires a strictly increasing grid")
    if not np.allclose(spacings, spacings[0]):
        raise ValueError("the silhouette polyhedron requires a uniform grid")
    if not np.isclose(grid[0], 0.0):
        raise ValueError("C3 requires a grid whose left endpoint is zero")
    delta = float(spacings[0])
    upper = float(grid[-1])
    L = float(np.sqrt(2.0) if lipschitz_constant is None else lipschitz_constant)
    if L <= 0.0 or not np.isfinite(L):
        raise ValueError("lipschitz_constant must be positive and finite")
    return SilhouettePolyhedron(
        grid=grid, delta=delta, lipschitz_constant=L, slope=L * delta,
        grid_upper=upper,
    )


# Compatibility aliases for the registered arm names and older notebooks.
SilhouetteCone = SilhouettePolyhedron


def silhouette_cone(
    grid, lipschitz_constant: float | None = None,
) -> SilhouettePolyhedron:
    return silhouette_polyhedron(grid, lipschitz_constant=lipschitz_constant)


def cone_violation_stats(
    draws, cone: SilhouettePolyhedron, tol: float = 1e-9,
) -> dict:
    """Per-curve violation rates of C1-C3 plus worst magnitudes."""
    x = np.asarray(draws, dtype=float)
    if x.size == 0:
        raise ValueError("need at least one curve to measure silhouette validity")
    if np.any(~np.isfinite(x)):
        raise ValueError("silhouette draws must be finite")
    if x.ndim == 0 or x.shape[-1] != cone.grid.size:
        raise ValueError("the last draw dimension must match the polyhedron grid")
    if x.ndim == 2:
        x = x[None, :, :]
    x = x.reshape(-1, cone.grid.size)
    diffs = np.diff(x, axis=1)

    c1 = np.any(x < -tol, axis=1)
    c2 = np.any(np.abs(diffs) > cone.slope + tol, axis=1)
    c3 = np.abs(x[:, 0]) > tol
    any_ = c1 | c2 | c3

    out = {
        "viol_c1_rate": float(c1.mean()),
        "viol_c2_rate": float(c2.mean()),
        "viol_c3_rate": float(c3.mean()),
        "viol_any_rate": float(any_.mean()),
        "viol_c1_mag": float(-np.min(x[c1])) if c1.any() else 0.0,
        "viol_c3_mag": float(np.max(np.abs(x[c3, 0]))) if c3.any() else 0.0,
    }
    out["viol_c2_mag_delta_units"] = (
        float(np.max(np.abs(diffs[c2]) - cone.slope) / cone.delta)
        if c2.any() else 0.0
    )
    return out


def project_to_cone(
    curves, cone: SilhouettePolyhedron, max_sweeps: int = 4000,
    viol_tol: float = 1e-9, update_tol: float = 1e-11,
) -> np.ndarray:
    """Euclidean projection onto the unconditional C1-C3 polyhedron.

    Hildreth dual-coordinate updates are vectorised over curves.  Stopping
    requires both primal feasibility and a small maximum dual-coordinate
    update over a complete sweep.  Primal feasibility alone is insufficient
    because a feasible intermediate iterate can still be nonoptimal.
    """
    arr = np.asarray(curves, dtype=float)
    shape = arr.shape
    m = cone.grid.size
    if arr.size == 0:
        return arr
    if np.any(~np.isfinite(arr)):
        raise ValueError("curves must be finite")
    if arr.ndim == 0 or arr.shape[-1] != m:
        raise ValueError("the last curve dimension must match the polyhedron grid")
    x = arr.reshape(-1, m).copy()
    B = x.shape[0]
    s = cone.slope

    initial_viol = max(
        float(np.max(np.maximum(-x, 0.0))),
        float(np.max(np.abs(x[:, 0]))),
        float(np.max(np.maximum(np.abs(np.diff(x, axis=1)) - s, 0.0))),
    )
    if initial_viol <= viol_tol:
        return arr.copy()

    lam_low = np.zeros((B, m), dtype=float)           # -x_i <= 0
    lam_left = np.zeros(B, dtype=float)               # x_0 <= 0
    lam_lip_pos = np.zeros((B, m - 1), dtype=float)   # x_{i+1} - x_i <= s
    lam_lip_neg = np.zeros((B, m - 1), dtype=float)   # x_i - x_{i+1} <= s
    even_p1 = np.arange(0, m - 1, 2)
    odd_p1 = np.arange(1, m - 1, 2)

    converged = False
    last_viol = float("inf")
    last_update = float("inf")
    for _ in range(int(max_sweeps)):
        sweep_update = 0.0

        resid = -x
        delta = np.maximum(resid, -lam_low)
        sweep_update = max(sweep_update, float(np.max(np.abs(delta))))
        lam_low += delta
        x += delta

        resid = x[:, 0]
        delta = np.maximum(resid, -lam_left)
        sweep_update = max(sweep_update, float(np.max(np.abs(delta))))
        lam_left += delta
        x[:, 0] -= delta

        for p1 in (even_p1, odd_p1):
            p2 = p1 + 1
            resid = (x[:, p2] - x[:, p1] - s) / 2.0
            delta = np.maximum(resid, -lam_lip_pos[:, p1])
            sweep_update = max(sweep_update, float(np.max(np.abs(delta))))
            lam_lip_pos[:, p1] += delta
            x[:, p2] -= delta
            x[:, p1] += delta

            resid = (x[:, p1] - x[:, p2] - s) / 2.0
            delta = np.maximum(resid, -lam_lip_neg[:, p1])
            sweep_update = max(sweep_update, float(np.max(np.abs(delta))))
            lam_lip_neg[:, p1] += delta
            x[:, p1] -= delta
            x[:, p2] += delta

        last_viol = max(
            float(np.max(np.maximum(-x, 0.0))),
            float(np.max(np.abs(x[:, 0]))),
            float(np.max(np.maximum(np.abs(np.diff(x, axis=1)) - s, 0.0))),
        )
        last_update = sweep_update
        if last_viol <= viol_tol and last_update <= update_tol:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            "Hildreth projection did not converge: "
            f"primal_violation={last_viol:.3e}, "
            f"max_dual_update={last_update:.3e}, sweeps={int(max_sweeps)}"
        )
    return x.reshape(shape)


# --------------------------------------------------------------------------- #
# M6: the topological measurement-error model (Task 6.25.1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TopoStep1:
    """Frozen Maroulas Step-1 model, calibrated on the Phase-6 calibration sample.

    ``prior`` and ``clutter`` are ``bayes_tda`` ``RGaussianMixture`` objects
    fitted from paired clean and observed calibration diagrams, respectively,
    with arm labels pooled out.  Prior complexity, clutter scale, ``alpha`` and
    ``sigma_dyo`` are selected by subject-disjoint held-out predictive log score
    for the clean diagram conditional on its paired observation.  The selected
    structure is refitted on the full calibration sample and frozen.
    """

    prior: object
    clutter: object
    sigma_dyo: float
    alpha: float
    min_birth: float
    sample_range: tuple[float, float]
    resolution: int
    r: float
    n_calibration_curves: int
    selection: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sigma_dyo": float(self.sigma_dyo),
            "alpha": float(self.alpha),
            "min_birth": float(self.min_birth),
            "sample_range": list(self.sample_range),
            "resolution": int(self.resolution),
            "r": float(self.r),
            "n_calibration_curves": int(self.n_calibration_curves),
            "selection": self.selection,
        }


def _calibration_paired_diagrams(cfg: MEUQConfig):
    """Paired observed/clean Alpha diagrams from the Phase-6 calibration sample."""
    key = _noise_key(cfg.noise_level)
    ds = generate_synthetic_dataset(dgp_config(
        cfg, n=int(cfg.n_calibration_subjects),
        seed=_seed_int(cfg.seeds.calibration, 1, key),
        noise_seed=_seed_int(cfg.seeds.calibration, 2, key),
    ))
    rng = _rng(cfg.seeds.calibration, 3, key)
    observed, clean, subject_ids = [], [], []
    for i in range(ds.clouds.shape[0]):
        for arm in (0, 1):
            noisy = _subsample(ds.clouds[i, arm], cfg.max_points, rng)
            latent = _subsample(ds.clean_clouds[i, arm], cfg.max_points, rng)
            observed.append(alpha_diagram(noisy))
            clean.append(alpha_diagram(latent))
            subject_ids.append(i)
    return observed, clean, np.asarray(subject_ids, dtype=int)


def _scaled_mixture(mixture, weight_scale: float):
    """Copy a restricted Gaussian mixture while scaling its intensity mass."""
    from bayes_tda.intensities import RGaussianMixture

    return RGaussianMixture(
        mus=np.asarray(mixture.mus, dtype=float).copy(),
        sigmas=np.asarray(mixture.sigmas, dtype=float).copy(),
        weights=float(weight_scale) * np.asarray(mixture.weights, dtype=float),
        normalize_weights=False,
        tilted=bool(getattr(mixture, "tilted", True)),
        min_birth=float(getattr(mixture, "min_birth", 0.0)),
        fastQ=bool(getattr(mixture, "fastQ", False)),
    )


def _prior_component_grid(clean_diagrams_bp) -> tuple[int, ...]:
    cards = np.asarray([len(np.asarray(d)) for d in clean_diagrams_bp], dtype=float)
    centre = max(1, int(round(float(cards.mean()))))
    return tuple(sorted({max(1, centre // 2), centre, max(1, 2 * centre)}))


def _sampler_spatial_log_density(step1: TopoStep1, observed_bp, points_bp):
    """Log spatial density induced by :class:`PosteriorDiagramSampler`.

    Component masses include their wedge-acceptance probabilities, matching the
    sampler's rejection step.  This deliberately scores the distribution that
    M6 actually draws from rather than the vendored intensity's inconsistent
    ``sigma`` convention.
    """
    from scipy.stats import norm

    post = fast_subject_posterior(step1, observed_bp)
    means = [np.atleast_2d(np.asarray(post.posterior_means, dtype=float))]
    variances = [np.asarray(post.posterior_sigmas, dtype=float).ravel()]
    coeffs = [step1.alpha * np.asarray(post.Cs, dtype=float).ravel()]
    if step1.alpha < 1.0:
        means.append(np.atleast_2d(np.asarray(step1.prior.mus, dtype=float)))
        variances.append(np.asarray(step1.prior.sigmas, dtype=float).ravel())
        coeffs.append((1.0 - step1.alpha) *
                      np.asarray(step1.prior.weights, dtype=float).ravel())

    means = np.vstack(means)
    variances = np.concatenate(variances)
    coeffs = np.concatenate(coeffs)
    std = np.sqrt(variances)
    wedge_mass = (
        norm.cdf((means[:, 0] - step1.min_birth) / std)
        * norm.cdf(means[:, 1] / std)
    )
    total_mass = float(np.sum(coeffs * wedge_mass))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        return np.full(len(points_bp), -1e12, dtype=float)

    pts = np.atleast_2d(np.asarray(points_bp, dtype=float))
    d2 = ((pts[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
    density = (
        np.exp(-0.5 * d2 / variances[None, :])
        / (2.0 * np.pi * variances[None, :])
    ) @ coeffs
    inside = (pts[:, 0] >= step1.min_birth) & (pts[:, 1] > 0.0)
    density = np.where(inside, density / total_mass, 0.0)
    return np.log(np.maximum(density, 1e-300))


def _conditional_clean_logscore(
    step1: TopoStep1, observed_bp, clean_bp,
) -> float:
    """Proper log score for a clean diagram conditional on its observation.

    It is the Poisson-cardinality plus normalized spatial log probability of
    the exact finite-point-process sampler used by M6.
    """
    from scipy.special import gammaln

    observed_bp = np.atleast_2d(np.asarray(observed_bp, dtype=float))
    clean_bp = np.atleast_2d(np.asarray(clean_bp, dtype=float))
    n_obs = 0 if np.asarray(observed_bp).size == 0 else int(observed_bp.shape[0])
    n_clean = 0 if np.asarray(clean_bp).size == 0 else int(clean_bp.shape[0])
    if n_obs == 0:
        return 0.0 if n_clean == 0 else -1e12
    log_cardinality = (
        -float(n_obs) + n_clean * np.log(float(n_obs)) - float(gammaln(n_clean + 1))
    )
    if n_clean == 0:
        return log_cardinality
    return log_cardinality + float(np.sum(
        _sampler_spatial_log_density(step1, observed_bp, clean_bp)
    ))


def fit_topo_step1(cfg: MEUQConfig, grid) -> TopoStep1:
    """Fit and freeze validation-calibrated M6 on paired diagrams.

    Registered selection rule (fixed before any evaluation replicate runs):

    1. split subjects, never curves, into fit and hold-out sets;
    2. elicit candidate latent priors from fit-clean diagrams and a clutter
       template from fit-observed diagrams, without treatment labels;
    3. select prior component count, clutter intensity scale, ``alpha`` and a
       scale-free ``sigma_dyo`` multiplier by the mean conditional predictive
       log score of hold-out-clean diagrams given hold-out-observed diagrams;
    4. refit prior and clutter on every calibration pair using the selected
       structure, transfer the selected sigma multiplier to the refitted prior
       scale, and freeze the result before evaluation.
    """
    if not 0.0 < cfg.calibration_holdout_frac < 0.9:
        raise ValueError("calibration_holdout_frac must be in (0, 0.9)")
    key = _noise_key(cfg.noise_level)
    observed_bd, clean_bd, subject_ids = _calibration_paired_diagrams(cfg)
    observed_bp = [bd_to_bp(d) for d in observed_bd]
    clean_bp = [bd_to_bp(d) for d in clean_bd]

    uniq = np.unique(subject_ids)
    rng9 = _rng(cfg.seeds.calibration, 9, key)
    perm = rng9.permutation(uniq)
    n_hold = max(1, int(round(cfg.calibration_holdout_frac * uniq.size)))
    hold_subjects = set(perm[:n_hold].tolist())
    fit_mask = np.array([sid not in hold_subjects for sid in subject_ids])
    obs_fit = [d for d, m in zip(observed_bp, fit_mask) if m]
    clean_fit = [d for d, m in zip(clean_bp, fit_mask) if m]
    obs_hold = [d for d, m in zip(observed_bp, fit_mask) if not m]
    clean_hold = [d for d, m in zip(clean_bp, fit_mask) if not m]
    if len(obs_fit) < 4 or len(obs_hold) < 2:
        raise ValueError("calibration sample is too small for paired selection")

    component_grid = _prior_component_grid(clean_fit)
    _, clutter_template = elicit_prior_clutter(
        obs_fit, n_components=1, clutter_n_components=1, min_birth=0.0,
        random_state=_seed_int(cfg.seeds.calibration, 10, key),
    )

    records: list[dict] = []
    for n_components in component_grid:
        prior_fit, _ = elicit_prior_clutter(
            clean_fit, n_components=int(n_components), min_birth=0.0,
            random_state=_seed_int(cfg.seeds.calibration, 11, key,
                                   int(n_components)),
        )
        prior_median = float(np.median(np.asarray(prior_fit.sigmas, dtype=float)))
        for clutter_scale in M6_CLUTTER_SCALE_GRID:
            clutter_fit = _scaled_mixture(clutter_template, clutter_scale)
            for alpha in M6_ALPHA_GRID:
                for sigma_multiplier in M6_SIGMA_MULTIPLIER_GRID:
                    sigma = max(1e-8, float(sigma_multiplier) * prior_median)
                    candidate = TopoStep1(
                        prior=prior_fit, clutter=clutter_fit, sigma_dyo=sigma,
                        alpha=float(alpha), min_birth=0.0,
                        sample_range=grid.sample_range,
                        resolution=int(grid.resolution), r=float(grid.r),
                        n_calibration_curves=int(len(observed_bp)),
                    )
                    pair_scores = [
                        _conditional_clean_logscore(candidate, obs, clean)
                        for obs, clean in zip(obs_hold, clean_hold)
                    ]
                    records.append({
                        "prior_components": int(n_components),
                        "clutter_scale": float(clutter_scale),
                        "alpha": float(alpha),
                        "sigma_multiplier": float(sigma_multiplier),
                        "sigma_dyo_fit": float(sigma),
                        "holdout_clean_conditional_logscore": float(
                            np.mean(pair_scores)),
                    })

    best = max(records, key=lambda row: row["holdout_clean_conditional_logscore"])

    prior_full, _ = elicit_prior_clutter(
        clean_bp, n_components=int(best["prior_components"]), min_birth=0.0,
        random_state=_seed_int(cfg.seeds.calibration, 12, key),
    )
    _, clutter_full_template = elicit_prior_clutter(
        observed_bp, n_components=1, clutter_n_components=1, min_birth=0.0,
        random_state=_seed_int(cfg.seeds.calibration, 13, key),
    )
    clutter_full = _scaled_mixture(clutter_full_template, best["clutter_scale"])
    prior_sigma_median = float(np.median(
        np.asarray(prior_full.sigmas, dtype=float)))
    sigma_full = max(1e-8, float(best["sigma_multiplier"]) * prior_sigma_median)

    selection = {
        "rule": "paired_clean_conditional_logscore_subject_holdout_then_full_refit",
        "uses_paired_clean_diagrams": True,
        "arm_labels_used": False,
        "n_fit_pairs": int(len(obs_fit)),
        "n_holdout_pairs": int(len(obs_hold)),
        "fit_subject_ids": sorted(int(v) for v in uniq if v not in hold_subjects),
        "holdout_subject_ids": sorted(int(v) for v in hold_subjects),
        "prior_component_grid": list(component_grid),
        "clutter_scale_grid": list(M6_CLUTTER_SCALE_GRID),
        "alpha_grid": list(M6_ALPHA_GRID),
        "sigma_multiplier_grid": list(M6_SIGMA_MULTIPLIER_GRID),
        "n_candidates": int(len(records)),
        "prior_components": int(best["prior_components"]),
        "selected_clutter_scale": float(best["clutter_scale"]),
        "alpha": float(best["alpha"]),
        "selected_sigma_dyo": float(sigma_full),
        "selected_sigma_dyo_multiplier": float(best["sigma_multiplier"]),
        "prior_sigma_median": prior_sigma_median,
        "holdout_clean_conditional_logscore": float(
            best["holdout_clean_conditional_logscore"]),
        "holdout_logscore_range": [
            float(min(r["holdout_clean_conditional_logscore"] for r in records)),
            float(max(r["holdout_clean_conditional_logscore"] for r in records)),
        ],
        "refit_on_full_calibration_sample": True,
        "candidate_scores": records,
    }
    return TopoStep1(
        prior=prior_full, clutter=clutter_full, sigma_dyo=sigma_full,
        alpha=float(best["alpha"]),
        min_birth=0.0, sample_range=grid.sample_range,
        resolution=int(grid.resolution), r=float(grid.r),
        n_calibration_curves=int(len(observed_bp)), selection=selection,
    )


class _PosteriorView:
    """Attribute-compatible replica of ``bayes_tda.intensities.Posterior``.

    Carries exactly the attributes :class:`PosteriorDiagramSampler` reads
    (``alpha``, ``num_obs_dgms``, ``posterior_means``, ``posterior_sigmas``,
    ``Cs``, ``lambd``, ``prior``, ``min_birth``).  The values are produced by
    :func:`fast_subject_posterior`, which replicates ``Posterior``'s arithmetic
    with vectorised wedge constants, so the sampled diagrams are the same point
    process the vendored class would produce (asserted in the tests).
    """

    def __init__(self, alpha, num_obs_dgms, posterior_means, posterior_sigmas,
                 Cs, lambd, prior, min_birth):
        self.alpha = alpha
        self.num_obs_dgms = num_obs_dgms
        self.posterior_means = posterior_means
        self.posterior_sigmas = posterior_sigmas
        self.Cs = Cs
        self.lambd = lambd
        self.prior = prior
        self.min_birth = min_birth


def fast_subject_posterior(step1: TopoStep1, diagram_bp) -> _PosteriorView:
    """The per-subject Maroulas posterior, computed with vectorised wedge mass.

    This replicates ``bayes_tda.intensities.Posterior`` exactly for a single
    observed diagram: the posterior mixture means and variances, the ``w``
    weights, the wedge normalising constants ``Q`` and the posterior
    coefficients ``C = w / (clutter + alpha * sum_j w Q)``.  The vendored
    implementation constructs one ``scipy.stats.multivariate_normal`` per
    mixture component (tens of thousands of slow CDF calls per subject); here
    the wedge mass

    .. math:: Q = \\Phi\\bigl((\\mu_b - \\min\\_birth)/v\\bigr)
        \\cdot \\Phi\\bigl(\\mu_p/v\\bigr)

    is evaluated in closed form with vectorised normal CDFs.  ``v`` is the
    posterior *variance* value, used exactly as ``RestrictedGaussian`` uses its
    ``sigma`` argument (the published package's convention), so the resulting
    coefficients match the vendored class to floating-point round-off.
    """
    from scipy.stats import norm

    prior = step1.prior
    mus = np.atleast_2d(np.asarray(prior.mus, dtype=float))
    k = mus.shape[0]
    Y = np.atleast_2d(np.asarray(diagram_bp, dtype=float))
    n = Y.shape[0]
    if n == 0:
        raise ValueError("fast_subject_posterior needs a non-empty diagram")

    u_exp = np.repeat(mus, n, axis=0)              # (k*n, 2), prior-major
    y_exp = np.tile(Y, (k, 1))
    p_sig = np.repeat(np.asarray(prior.sigmas, dtype=float).ravel(), n)
    p_w = np.repeat(np.asarray(prior.weights, dtype=float).ravel(), n)
    sdyo = float(step1.sigma_dyo)

    posterior_means = (sdyo * u_exp + p_sig[:, None] * y_exp) / (sdyo + p_sig[:, None])
    posterior_sigmas = sdyo * p_sig / (sdyo + p_sig)

    conv = p_sig + sdyo
    d_sq = ((y_exp - u_exp) ** 2).sum(axis=1)
    dens = np.exp(-0.5 * d_sq / conv) / (2.0 * np.pi * conv)
    w = p_w * dens

    # Wedge mass with the vendored sigma-as-SD convention (see docstring).
    q = (norm.cdf((posterior_means[:, 0] - step1.min_birth) / posterior_sigmas)
         * norm.cdf(posterior_means[:, 1] / posterior_sigmas))

    wq = (w * q).reshape(k, n)
    swq = wq.sum(axis=1).repeat(n)
    clutter = np.asarray(step1.clutter.evaluate(y_exp), dtype=float).ravel()
    cs = w / (clutter + step1.alpha * swq)

    return _PosteriorView(
        alpha=step1.alpha, num_obs_dgms=1,
        posterior_means=posterior_means, posterior_sigmas=posterior_sigmas,
        Cs=cs, lambd=float(n), prior=prior, min_birth=step1.min_birth,
    )


def subject_topo_phi_draws(step1: TopoStep1, diagram_bd, n_draws: int,
                           random_state=None) -> np.ndarray:
    """``(n_draws, m)`` silhouette draws from one subject's Maroulas posterior.

    The per-subject posterior conditions on the subject's own observed diagram
    (the likelihood) with the frozen calibrated prior/clutter/``sigma_DYO``.
    Cardinality is drawn from the posterior cardinality model (Poisson with the
    mean observed diagram size), locations by rejection-truncated mixture
    sampling, and each draw is mapped back to birth--death coordinates and
    embedded with the power-weighted silhouette on the frozen grid.
    """
    m = int(step1.resolution)
    if diagram_bd is None or np.asarray(diagram_bd).size == 0:
        return np.zeros((int(n_draws), m), dtype=float)
    diagram_bd = np.atleast_2d(np.asarray(diagram_bd, dtype=float))
    diagram_bp = bd_to_bp(diagram_bd)
    posterior = fast_subject_posterior(step1, diagram_bp)
    sampler = PosteriorDiagramSampler(posterior)
    draws = sampler.sample_diagrams(int(n_draws), random_state=random_state,
                                    count="poisson")
    out = np.zeros((int(n_draws), m), dtype=float)
    for s, d in enumerate(draws):
        if d.shape[0] > 0:
            out[s] = silhouette_on_grid(
                bp_to_bd(d), step1.sample_range, step1.resolution, step1.r)
    return out


def topo_draws_for_replicate(cfg: MEUQConfig, observed, step1: TopoStep1,
                             rep: int, stream_part: int = 6) -> np.ndarray:
    """``(S, n, m)`` M6 posterior curves for one evaluation replicate."""
    key = _noise_key(cfg.noise_level)
    n = int(observed.A.shape[0])
    S = int(cfg.n_measurement_draws)
    out = np.empty((S, n, int(step1.resolution)), dtype=float)
    rng = _rng(cfg.seeds.measurement_posterior, stream_part, key, rep)
    sizes = observed.diagram_alpha_sizes
    padded = observed.diagrams_alpha
    for i in range(n):
        k = int(sizes[i])
        dgm = padded[i, :k] if k else np.empty((0, 2), dtype=float)
        out[:, i, :] = subject_topo_phi_draws(
            step1, dgm, S, random_state=int(rng.integers(0, np.iinfo(np.int32).max)))
    return out


def method_me_topo_posterior(cfg: MEUQConfig, observed, step1: TopoStep1,
                             rep: int, draws=None) -> MethodResult:
    """M6 -- Maroulas posterior diagram draws propagated through the causal model.

    Identical band construction to M4 (nested propagation through the FGP with
    a curve-level sup-norm band); the only difference is the source of the
    ``D | D~`` draws (a diagram-space posterior instead of a Gaussian bridge).
    """
    from btate.causal.propagation import nested_posterior_tate

    key = _noise_key(cfg.noise_level)
    if draws is None:
        draws = topo_draws_for_replicate(cfg, observed, step1, rep)
    effect = nested_posterior_tate(
        draws, observed.A, observed.X, observed.grid, pi_hat=observed.pi_hat,
        estimator=_fgp_estimator(cfg), n_causal_draws=cfg.n_causal_draws,
        alpha=cfg.alpha,
        random_state=_seed_int(cfg.seeds.causal, 6, key, rep),
    )
    result = _posterior_result(effect, "posterior_supnorm_standardized")
    result.metadata["n_measurement_draws"] = int(cfg.n_measurement_draws)
    result.metadata["topo_step1"] = True
    return result


def method_me_topo_point(cfg: MEUQConfig, observed, step1: TopoStep1,
                         rep: int, draws=None) -> MethodResult:
    """M6-point -- AIPW on the posterior-**mean** curve (the Phase-5 point denoiser).

    The posterior-mean silhouette ``E[phi(D | D~)]`` (mean over the same M6
    draws) is treated as an exact observation and fed to the same AIPW band
    machinery as M2.  This is the Phase-4.25/5 Step-1 construction, whose zero
    clean-target coverage Lemma 1 already predicts; running it as a separate
    arm separates "point correction loses" from "topology loses".
    """
    key = _noise_key(cfg.noise_level)
    if draws is None:
        draws = topo_draws_for_replicate(cfg, observed, step1, rep)
    phi_mean = np.asarray(draws, dtype=float).mean(axis=0)
    return _aipw_result(
        cfg, phi_mean, observed.A, observed.X, observed.grid, observed.pi_hat,
        _seed_int(cfg.seeds.bootstrap, 6, key, rep),
    )


def method_me_topo_posterior_proj(cfg: MEUQConfig, observed, step1: TopoStep1,
                                  cone: SilhouetteCone, rep: int,
                                  draws=None) -> MethodResult:
    """M6-proj -- M6 projected onto the unconditional C1-C3 polyhedron."""
    from btate.causal.propagation import nested_posterior_tate

    key = _noise_key(cfg.noise_level)
    if draws is None:
        draws = topo_draws_for_replicate(cfg, observed, step1, rep)
    draws = project_to_cone(draws, cone)
    effect = nested_posterior_tate(
        draws, observed.A, observed.X, observed.grid, pi_hat=observed.pi_hat,
        estimator=_fgp_estimator(cfg), n_causal_draws=cfg.n_causal_draws,
        alpha=cfg.alpha,
        random_state=_seed_int(cfg.seeds.causal, 6, key, rep),
    )
    result = _posterior_result(effect, "posterior_supnorm_standardized")
    result.metadata["n_measurement_draws"] = int(cfg.n_measurement_draws)
    result.metadata["cone_projected"] = True
    return result


# --------------------------------------------------------------------------- #
# Polyhedron-projected bridge arms (Task 6.25.3)
# --------------------------------------------------------------------------- #
def _bridge_draws(cfg: MEUQConfig, observed, bridge, rep: int,
                  stream_part: int) -> np.ndarray:
    """The bridge posterior draws for one replicate (M4/M5 stream semantics)."""
    key = _noise_key(cfg.noise_level)
    return bridge.draw_clean_curves(
        observed.phi_alpha, observed.phi_dtm if bridge.uses_dtm else None,
        n_draws=cfg.n_measurement_draws,
        random_state=_seed_int(cfg.seeds.measurement_posterior, stream_part,
                               key, rep),
    )


def method_me_bayes_bridge_proj(cfg: MEUQConfig, observed, bridge,
                                cone: SilhouetteCone, rep: int) -> MethodResult:
    """M4-proj -- M4 projected onto the unconditional C1-C3 polyhedron.

    Shares the draw, causal and bootstrap streams of M4 bit-for-bit; the
    projection is the only difference.
    """
    from btate.causal.propagation import nested_posterior_tate

    key = _noise_key(cfg.noise_level)
    draws = project_to_cone(_bridge_draws(cfg, observed, bridge, rep, 4), cone)
    effect = nested_posterior_tate(
        draws, observed.A, observed.X, observed.grid, pi_hat=observed.pi_hat,
        estimator=_fgp_estimator(cfg), n_causal_draws=cfg.n_causal_draws,
        alpha=cfg.alpha,
        random_state=_seed_int(cfg.seeds.causal, 4, key, rep),
    )
    result = _posterior_result(effect, "posterior_supnorm_standardized")
    result.metadata["n_measurement_draws"] = int(cfg.n_measurement_draws)
    result.metadata["bridge_uses_dtm"] = bool(bridge.uses_dtm)
    result.metadata["cone_projected"] = True
    return result


def method_me_freq_mi_proj(cfg: MEUQConfig, observed, bridge,
                           cone: SilhouetteCone, rep: int) -> MethodResult:
    """M5-proj -- M5 projected onto the unconditional C1-C3 polyhedron."""
    from btate.benchmarks.frequentist import cross_fitted_scores
    from btate.benchmarks.measurement_error_uq import multiple_imputation_band

    key = _noise_key(cfg.noise_level)
    draws = project_to_cone(_bridge_draws(cfg, observed, bridge, rep, 5), cone)
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
    result.metadata["cone_projected"] = True
    return result


# --------------------------------------------------------------------------- #
# Cell preparation and replicate driver
# --------------------------------------------------------------------------- #
def p625_methods(cfg: MEUQConfig) -> tuple[str, ...]:
    return P625_METHODS + (DTM_P625_METHODS if cfg.use_dtm_feature else ())


def prepare_cell_p625(cfg: MEUQConfig, cache_dir: str | Path | None = None,
                      verbose: bool = False) -> dict:
    """Phase-6.25 preparation with a versioned, reloadable binary cache.

    Everything is computed from seed namespaces disjoint from the evaluation
    replicates.  ``grid``, ``targets`` and ``bridges`` come from
    ``prepare_cell`` unchanged; ``step1`` is the frozen M6 Step-1 model and
    ``cone`` the C1-C3 valid-silhouette polyhedron on the frozen grid.
    """
    cache_path = json_path = None
    if cache_dir is not None:
        path = Path(cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        signature = hashlib.sha256(json.dumps({
            "cache_version": P625_CACHE_VERSION,
            "config": cfg.to_dict(),
        }, sort_keys=True).encode()).hexdigest()[:16]
        tag = f"noise{_noise_key(cfg.noise_level):06d}_res{cfg.resolution}_{signature}"
        cache_path = path / f"p625_prepared_{tag}.pkl"
        json_path = path / f"p625_prepared_{tag}.json"
        if cache_path.exists():
            with cache_path.open("rb") as handle:
                out = pickle.load(handle)
            if verbose:
                print(f"[prepare] loaded Phase-6.25 cache: {cache_path}")
            return out

    prepared = prepare_cell(cfg, cache_dir=cache_dir, verbose=verbose)
    t0 = time.perf_counter()
    step1 = fit_topo_step1(cfg, prepared["grid"])
    cone = silhouette_cone(prepared["grid"].grid)
    if verbose:
        sel = step1.selection
        print(f"[step1] sigma_DYO={step1.sigma_dyo:.6g} "
              f"(mult {sel['selected_sigma_dyo_multiplier']:.4g}, "
              f"alpha {sel['alpha']:.3g}, clutter {sel['selected_clutter_scale']:.3g}) "
              f"prior_components={sel['prior_components']} "
              f"holdout_clean_logscore={sel['holdout_clean_conditional_logscore']:.4g} "
              f"({time.perf_counter() - t0:.0f}s)")
        print(f"[polyhedron] C2 slope={cone.slope:.6g}; C3 pins only t=0")

    out = {"config": cfg, "grid": prepared["grid"], "targets": prepared["targets"],
           "bridges": prepared["bridges"], "step1": step1, "cone": cone}
    if cache_path is not None and json_path is not None:
        tmp_path = cache_path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
        json_path.write_text(json.dumps({
            "cache_version": P625_CACHE_VERSION,
            "config": cfg.to_dict(),
            "grid": prepared["grid"].to_dict(),
            "targets": prepared["targets"].to_dict(),
            "bridges": {k: v.to_dict() for k, v in prepared["bridges"].items()},
            "step1": step1.to_dict(),
            "cone": cone.to_dict(),
        }, indent=2))
    return out


def _latent_draw_sources(cfg: MEUQConfig, observed, bridges, step1, rep,
                         cone: SilhouetteCone) -> dict:
    """Latent clean curves per arm, with the arms' exact seed streams.

    Used by the polyhedron diagnostic so it measures exactly the draws the arms
    consumed (the base arms recompute them internally with identical streams).
    """
    sources = {
        "m4": _bridge_draws(cfg, observed, bridges["alpha"], rep, 4),
        "m5": _bridge_draws(cfg, observed, bridges["alpha"], rep, 5),
    }
    if cfg.use_dtm_feature:
        sources["m4_dtm"] = _bridge_draws(cfg, observed, bridges["dtm"], rep, 4)
        sources["m5_dtm"] = _bridge_draws(cfg, observed, bridges["dtm"], rep, 5)
    topo = topo_draws_for_replicate(cfg, observed, step1, rep)
    sources["m6"] = topo
    sources["m6_proj"] = project_to_cone(topo, cone)
    return sources


def evaluate_replicate_p625(cfg: MEUQConfig, grid, targets, bridges, step1,
                            cone, rep: int) -> list[dict]:
    """Run every Phase-6.25 arm on one evaluation replicate, scored identically."""
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
        "step1_sigma_dyo": float(step1.sigma_dyo),
        "step1_sigma_dyo_multiplier": float(
            step1.selection["selected_sigma_dyo_multiplier"]),
        "step1_alpha": float(step1.alpha),
        "step1_clutter_scale": float(step1.selection["selected_clutter_scale"]),
        "step1_holdout_clean_logscore": float(
            step1.selection["holdout_clean_conditional_logscore"]),
        "step1_prior_components": int(step1.selection["prior_components"]),
    }

    topo = topo_draws_for_replicate(cfg, observed, step1, rep)
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
        "me_bayes_bridge_proj":
            lambda: method_me_bayes_bridge_proj(cfg, observed,
                                                bridges["alpha"], cone, rep),
        "me_freq_mi_proj":
            lambda: method_me_freq_mi_proj(cfg, observed,
                                           bridges["alpha"], cone, rep),
        "me_topo_posterior":
            lambda: method_me_topo_posterior(cfg, observed, step1, rep, topo),
        "me_topo_point":
            lambda: method_me_topo_point(cfg, observed, step1, rep, topo),
        "me_topo_posterior_proj":
            lambda: method_me_topo_posterior_proj(cfg, observed, step1, cone,
                                                  rep, topo),
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
        row["is_oracle"] = name in ("oracle_clean_aipw",)
        row["measurement_error_aware"] = name.startswith("me_")
        row["bridge_uses_dtm"] = name.endswith("_dtm")
        row["cone_projected"] = name.endswith("_proj")
        row["topo_measurement_model"] = name.startswith("me_topo")
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

    # Polyhedron diagnostics (Task 6.25.2), on the arms' own draw streams.
    sources = _latent_draw_sources(cfg, observed, bridges, step1, rep, cone)
    cone_columns: dict = {}
    for name, draws in sources.items():
        stats = cone_violation_stats(draws, cone)
        for k, v in stats.items():
            cone_columns[f"cone_{name}_{k}"] = float(v)
    cone_columns["cone_n_draws"] = int(sources["m6"].shape[0] *
                                       sources["m6"].shape[1])
    for row in rows:
        row.update(cone_columns)
    return rows


def run_p625_cell(cfg: MEUQConfig, rep_indices, *, cache_dir=None,
                  n_jobs: int = 1, prepared: dict | None = None,
                  verbose: bool = False):
    """Evaluate ``rep_indices`` for one cell; returns a tidy DataFrame."""
    import pandas as pd

    prepared = prepared or prepare_cell_p625(cfg, cache_dir=cache_dir,
                                             verbose=verbose)
    grid, targets = prepared["grid"], prepared["targets"]
    bridges, step1, cone = (prepared["bridges"], prepared["step1"],
                            prepared["cone"])
    reps = [int(r) for r in rep_indices]
    if not reps:
        return pd.DataFrame()

    if n_jobs == 1:
        chunks = [evaluate_replicate_p625(cfg, grid, targets, bridges, step1,
                                          cone, rep) for rep in reps]
    else:
        from joblib import Parallel, delayed

        workers = max(1, min(int(n_jobs), len(reps)))
        chunks = Parallel(n_jobs=workers, backend="loky",
                          verbose=5 if verbose else 0)(
            delayed(evaluate_replicate_p625)(cfg, grid, targets, bridges,
                                             step1, cone, rep)
            for rep in reps
        )
    return pd.DataFrame([row for chunk in chunks for row in chunk])


def p625_cone_report(sources: dict, cone: SilhouetteCone,
                     alpha: float = 0.05) -> dict:
    """Headline validity table for one replicate set (helper for tests)."""
    out: dict = {}
    for name, draws in sources.items():
        stats = cone_violation_stats(draws, cone)
        out[name] = {
            "any_rate": stats["viol_any_rate"],
            "c1_rate": stats["viol_c1_rate"],
            "c2_rate": stats["viol_c2_rate"],
            "c3_rate": stats["viol_c3_rate"],
            "c1_worst_abs": stats["viol_c1_mag"],
            "c2_worst_delta_units": stats["viol_c2_mag_delta_units"],
            "c3_worst_abs": stats["viol_c3_mag"],
        }
    return out
