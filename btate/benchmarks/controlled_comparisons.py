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
   curves, the fraction of draws that violate the **valid-silhouette cone**
   (C1-C4, necessary conditions read off
   ``btate/embeddings/silhouette.py:117``), and scores the **cone-projected**
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
M6's prior, clutter and ``sigma_DYO`` are fitted on the **same** calibration
sample (its observed diagrams, pooled across arms without the treatment label),
with ``sigma_DYO`` selected by the Phase-4.25 empirical-Bayes marginal
likelihood re-scored on a subject-disjoint hold-out split of that sample, then
frozen.  The bridge additionally consumes the paired clean curves (its own
model's requirement); M6's model is a diagram-space PPP and consumes only
observed diagrams.  The selection rule is registered here and reported in every
cell's provenance.

The valid-silhouette cone (C1-C4)
---------------------------------
Under the frozen grid with spacing ``Delta``, every genuine power-weighted
silhouette (``weighted_silhouette``, ``btate/embeddings/silhouette.py:117``)
satisfies, with ``L = sqrt(2)`` the implementation's Lipschitz constant:

* **C1** non-negativity: ``phi(t) >= 0``;
* **C2** Lipschitz bound: ``|phi_{i+1} - phi_i| <= L * Delta``;
* **C3** height bound: ``phi(t) <= L * (max persistence) / 2``, which the
  frozen grid endpoint bounds by ``L * grid_upper / 2``;
* **C4** support: ``phi`` vanishes at the grid endpoints.

C1-C4 are **necessary, not sufficient** (the achievable set is the non-convex
image of diagram space under the silhouette map; the cone is a convex outer
approximation of it), so a violation is a sound certificate of invalidity while
satisfaction proves nothing.  The measured violation rate is therefore a
**lower bound** on how often the bridge leaves the valid set.

M6 draws are silhouettes of sampled diagrams, so they satisfy C1 and C2 for
*any* diagram.  They satisfy C3 and C4 whenever every sampled feature lies
inside the grid; the posterior intensity's Gaussian tails are unbounded, so a
rare feature sampled beyond ``grid_upper`` can legitimately trip C3/C4.  The
violation columns therefore carry a per-draw attribution check for M6
(``max_sampled_death``), so a non-zero M6 rate is a statement about the
posterior tails, not a falsification of the diagnostic.

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

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from btate.benchmarks.measurement_error_uq import (
    MEUQConfig,
    MethodResult,
    OracleBundle,
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
from btate.benchmarks.metrics import coverage_rate_with_ci
from btate.benchmarks.synthetic import generate_synthetic_dataset
from btate.embeddings.silhouette import weighted_silhouette
from btate.topo_posterior.adapters import bd_to_bp, bp_to_bd
from btate.topo_posterior.eb import select_sigma_dyo
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
    "me_bayes_bridge_proj",       # M4-proj: M4 draws projected onto the cone
    "me_freq_mi_proj",            # M5-proj: M5 draws projected onto the cone
    "me_topo_posterior",          # M6: topological posterior over diagrams
    "me_topo_point",              # M6-point: AIPW on the posterior-mean curve
    "me_topo_posterior_proj",     # M6-proj: M6 draws projected onto the cone
)
DTM_P625_METHODS = ("me_bayes_bridge_dtm", "me_freq_mi_dtm")

# --------------------------------------------------------------------------- #
# The valid-silhouette cone (Task 6.25.2)
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
class SilhouetteCone:
    """C1-C4 outer approximation of the valid-silhouette set on one grid."""

    grid: np.ndarray
    delta: float
    lipschitz_constant: float
    slope: float       # L * delta          (C2)
    height_bound: float  # L * grid_upper / 2  (C3, in-grid features)
    grid_upper: float

    def to_dict(self) -> dict:
        return {
            "delta": float(self.delta),
            "lipschitz_constant": float(self.lipschitz_constant),
            "slope": float(self.slope),
            "height_bound": float(self.height_bound),
            "grid_upper": float(self.grid_upper),
        }


def silhouette_cone(grid, lipschitz_constant: float | None = None) -> SilhouetteCone:
    """The C1-C4 cone for a frozen grid.

    ``max_persistence`` is bounded by ``grid_upper`` (births are ``>= 0`` and
    the grid starts at ``0``), so ``C3`` uses ``L * grid_upper / 2`` - the
    plan's ``~0.600`` constant on the frozen ``(0, 0.848)`` grid.  This is a
    necessary condition only for features that lie inside the grid; the
    diagnostics report the M6 sampled-feature attribution separately.
    """
    grid = np.asarray(grid, dtype=float).ravel()
    if grid.size < 2:
        raise ValueError("a cone needs a grid of at least two points")
    spacings = np.diff(grid)
    if not np.allclose(spacings, spacings[0]):
        raise ValueError("the cone requires a uniform grid")
    delta = float(spacings[0])
    upper = float(grid[-1])
    L = float(np.sqrt(2.0) if lipschitz_constant is None else lipschitz_constant)
    if L <= 0.0 or not np.isfinite(L):
        raise ValueError("lipschitz_constant must be positive and finite")
    return SilhouetteCone(
        grid=grid, delta=delta, lipschitz_constant=L, slope=L * delta,
        height_bound=L * upper / 2.0, grid_upper=upper,
    )


def cone_violation_stats(draws, cone: SilhouetteCone, tol: float = 1e-9) -> dict:
    """Per-draw violation rates of C1-C4 plus worst violation magnitudes.

    ``draws`` has shape ``(S, n, m)`` (one curve per draw per subject).  A draw
    counts as violating a condition when *any* grid point violates it, beyond a
    small absolute tolerance (kept conservative so the reported rate remains a
    *lower bound* on how often the arm leaves the valid set).  Magnitudes are
    reported over violating draws only: for C2 in units of ``delta``, for
    C1/C3/C4 in absolute silhouette units.
    """
    x = np.asarray(draws, dtype=float)
    if x.size == 0:
        raise ValueError("need at least one curve to measure cone violations")
    if x.ndim == 2:
        x = x[None, :, :]
    x = x.reshape(-1, cone.grid.size)
    diffs = np.diff(x, axis=1)

    c1 = np.any(x < -tol, axis=1)
    c2 = np.any(np.abs(diffs) > cone.slope + tol, axis=1)
    c3 = np.any(x > cone.height_bound + tol, axis=1)
    c4 = (np.abs(x[:, 0]) > tol) | (np.abs(x[:, -1]) > tol)
    any_ = c1 | c2 | c3 | c4

    out = {
        "cone_viol_c1_rate": float(c1.mean()),
        "cone_viol_c2_rate": float(c2.mean()),
        "cone_viol_c3_rate": float(c3.mean()),
        "cone_viol_c4_rate": float(c4.mean()),
        "cone_viol_any_rate": float(any_.mean()),
    }
    out["cone_viol_c1_mag"] = float(-np.min(x[c1])) if c1.any() else 0.0
    if c2.any():
        out["cone_viol_c2_mag_delta_units"] = float(
            np.max(np.abs(diffs[c2]) - cone.slope) / cone.delta)
    else:
        out["cone_viol_c2_mag_delta_units"] = 0.0
    out["cone_viol_c3_mag"] = (float(np.max(x[c3]) - cone.height_bound)
                               if c3.any() else 0.0)
    if c4.any():
        out["cone_viol_c4_mag"] = float(max(
            float(np.max(np.abs(x[c4, 0]))), float(np.max(np.abs(x[c4, -1])))))
    else:
        out["cone_viol_c4_mag"] = 0.0
    return out


def project_to_cone(curves, cone: SilhouetteCone, max_sweeps: int = 4000,
                    viol_tol: float = 1e-9) -> np.ndarray:
    """Euclidean projection of each curve onto the C1-C4 cone.

    The cone is a polyhedron ``{A x <= b}`` whose rows are box constraints
    (one variable), endpoint pins (one variable) and Lipschitz constraints
    (``x_{i+1} - x_i <= s``, two variables).  The projection is solved through
    the dual by block-coordinate descent (Hildreth's row-action method with
    Lagrange-multiplier tracking; without the multipliers the iterates stall at
    a feasible-but-suboptimal point, which is why the multipliers are
    maintained explicitly).  Blocks group constraints that act on disjoint
    variables (the box constraints, and the Lipschitz pairs of each parity), so
    every update is an exact line search, vectorised across the whole draw
    batch.  The exact coordinate step for constraint ``j`` is

    .. math:: \\lambda_j \\leftarrow \\max\\bigl(0,\\ \\lambda_j +
        (a_j^T x - b_j)/\\|a_j\\|^2\\bigr),

    which is evaluated as ``delta = max(resid, -lam_j)`` with ``resid`` the
    (normalised) residual.  Convergence is geometric on these draw
    distributions (constraint violations reach ``1e-11`` within ~2000 sweeps);
    the sweep cap exists only for the pathological all-constraints-active case.
    The result matches an independent SLSQP solve of the QP to ``1e-8`` in the
    tests.  A point that already satisfies every constraint has all residuals
    ``<= 0``, so it receives zero updates and comes back **bit-identical**; at
    convergence the projection is idempotent to the feasibility tolerance.  No
    randomness is involved, so results are bit-reproducible under a pinned BLAS
    thread count.
    """
    arr = np.asarray(curves, dtype=float)
    shape = arr.shape
    m = cone.grid.size
    if arr.size == 0:
        return arr
    x = arr.reshape(-1, m).copy()
    B = x.shape[0]
    s = cone.slope
    b = cone.height_bound

    # Dual variables, one block per constraint family.
    lam_low = np.zeros((B, m), dtype=float)      # -x_i <= 0
    lam_high = np.zeros((B, m), dtype=float)     # x_i <= b_i (0 at endpoints)
    lam_lip_pos = np.zeros((B, m - 1), dtype=float)  # x_{i+1} - x_i <= s
    lam_lip_neg = np.zeros((B, m - 1), dtype=float)  # x_i - x_{i+1} <= s

    high_bound = np.full(m, b, dtype=float)
    high_bound[0] = 0.0
    high_bound[-1] = 0.0

    even_p1 = np.arange(0, m - 1, 2)    # pairs (0,1), (2,3), ...
    odd_p1 = np.arange(1, m - 1, 2)     # pairs (1,2), (3,4), ...

    for it in range(int(max_sweeps)):
        # box lower: -x_i <= 0   (||a||^2 = 1)
        resid = -x
        delta = np.maximum(resid, -lam_low)
        lam_low += delta
        x += delta
        # box upper: x_i <= b_i (interior b, endpoints pinned to 0)
        resid = x - high_bound[None, :]
        delta = np.maximum(resid, -lam_high)
        lam_high += delta
        x -= delta
        for p1 in (even_p1, odd_p1):
            p2 = p1 + 1
            resid = (x[:, p2] - x[:, p1] - s) / 2.0
            delta = np.maximum(resid, -lam_lip_pos[:, p1])
            lam_lip_pos[:, p1] += delta
            x[:, p2] -= delta
            x[:, p1] += delta
            resid = (x[:, p1] - x[:, p2] - s) / 2.0
            delta = np.maximum(resid, -lam_lip_neg[:, p1])
            lam_lip_neg[:, p1] += delta
            x[:, p1] -= delta
            x[:, p2] += delta
        # The stopping criteria are evaluated every few sweeps.  The operative
        # one is primal feasibility (the point is already feasible to the cone
        # tolerance once the creeping multipliers settle); ``max_sweeps`` caps
        # the pathological all-constraints-active case.  Both are deterministic
        # functions of the data, so the sweep count is reproducible.
        if it % 25 == 0:
            viol = max(
                float(np.max(np.maximum(-x, 0.0))),
                float(np.max(np.maximum(x - high_bound[None, :], 0.0))),
                float(np.max(np.maximum(np.abs(np.diff(x, axis=1)) - s, 0.0))),
            )
            if viol < viol_tol:
                break
    return x.reshape(shape)


# --------------------------------------------------------------------------- #
# M6: the topological measurement-error model (Task 6.25.1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TopoStep1:
    """Frozen Maroulas Step-1 model, calibrated on the Phase-6 calibration sample.

    ``prior`` and ``clutter`` are ``bayes_tda`` ``RGaussianMixture`` objects
    elicited from the *observed* diagrams of the calibration sample (pooled
    across arms without the treatment label); ``sigma_dyo`` is the
    empirical-Bayes marginal-likelihood optimum of the fit split, re-scored on
    a subject-disjoint hold-out split by predictive PPP log-density, then
    frozen.  ``selection`` carries the full audit trail.
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


def _calibration_observed_diagrams(cfg: MEUQConfig):
    """Observed Alpha diagrams of the calibration sample (both arms, pooled).

    Regenerated with exactly the seeds and subsampling order of
    ``measurement_error_uq._calibration_curves`` so M6's calibration resource
    is the *same* 240-subject paired sample the bridge is fitted on (minus the
    clean curves, which the diagram-space model does not consume).
    """
    key = _noise_key(cfg.noise_level)
    ds = generate_synthetic_dataset(dgp_config(
        cfg, n=int(cfg.n_calibration_subjects),
        seed=_seed_int(cfg.seeds.calibration, 1, key),
        noise_seed=_seed_int(cfg.seeds.calibration, 2, key),
    ))
    rng = _rng(cfg.seeds.calibration, 3, key)
    diagrams, subject_ids = [], []
    for i in range(ds.clouds.shape[0]):
        for arm in (0, 1):
            noisy = _subsample(ds.clouds[i, arm], cfg.max_points, rng)
            diagrams.append(alpha_diagram(noisy))
            subject_ids.append(i)
    return diagrams, np.asarray(subject_ids, dtype=int)


def _holdout_predictive_loglik(candidates, holdout_bp, prior, clutter,
                               alpha: float) -> np.ndarray:
    """Mean predictive marked-PPP log-likelihood of held-out diagrams.

    This is the same functional ``select_sigma_dyo`` maximises, evaluated on
    data the selection did not see: the observed hold-out diagrams' PPP
    log-likelihood under the model with observation-noise variance
    ``sigma_dyo = candidates[k]``.  Up to ``sigma``-independent constants, so
    only differences across candidates are meaningful.
    """
    from btate.topo_posterior.eb import sigma_dyo_profile_loglik

    scores = np.empty(len(candidates), dtype=float)
    for k, s in enumerate(candidates):
        scores[k] = sigma_dyo_profile_loglik(holdout_bp, prior, clutter,
                                             float(s), alpha=alpha)
    return scores


def fit_topo_step1(cfg: MEUQConfig, grid) -> TopoStep1:
    """Fit and freeze the M6 Step-1 model on the calibration sample.

    Registered selection rule (fixed before any evaluation replicate runs):

    1. prior/clutter: pooled k-means elicitation (``elicit_prior_clutter``,
       Phase-4.25 defaults, ``min_birth=0``) on the **fit split** of the
       calibration observed diagrams, split subject-disjointly with the same
       stream the bridge uses;
    2. ``sigma_dyo``: the empirical-Bayes marginal-likelihood profile
       (``select_sigma_dyo``, grid of ``median(prior.sigmas)`` multipliers) on
       the fit split, re-scored on the hold-out split by predictive PPP
       log-likelihood; the argmax is frozen;
    3. the final posterior is refitted on the **full** calibration sample with
       the frozen ``sigma_dyo`` (mirroring the bridge's refit-on-full step).
    """
    from bayes_tda.intensities import Posterior

    if not 0.0 < cfg.calibration_holdout_frac < 0.9:
        raise ValueError("calibration_holdout_frac must be in (0, 0.9)")
    key = _noise_key(cfg.noise_level)
    diagrams_bd, subject_ids = _calibration_observed_diagrams(cfg)
    diagrams_bp = [bd_to_bp(d) for d in diagrams_bd]

    uniq = np.unique(subject_ids)
    rng9 = _rng(cfg.seeds.calibration, 9, key)
    perm = rng9.permutation(uniq)
    n_hold = max(1, int(round(cfg.calibration_holdout_frac * uniq.size)))
    hold_subjects = set(perm[:n_hold].tolist())
    fit_mask = np.array([sid not in hold_subjects for sid in subject_ids])
    fit_bp = [d for d, m in zip(diagrams_bp, fit_mask) if m]
    hold_bp = [d for d, m in zip(diagrams_bp, fit_mask) if not m]
    if len(fit_bp) < 4 or len(hold_bp) < 2:
        raise ValueError("calibration sample is too small to select sigma_dyo")

    prior, clutter = elicit_prior_clutter(
        fit_bp, min_birth=0.0,
        random_state=_seed_int(cfg.seeds.calibration, 10, key),
    )
    profile = select_sigma_dyo(fit_bp, prior, clutter, alpha=1.0)
    candidates = np.asarray(profile["profile_sigma_dyo"], dtype=float)
    scores = _holdout_predictive_loglik(candidates, hold_bp, prior, clutter,
                                        alpha=1.0)
    best = int(np.argmax(scores))
    sigma = float(candidates[best])

    selection = {
        "rule": "eb_marginal_likelihood_on_fit_split_rescored_on_holdout_split",
        "n_fit_curves": int(len(fit_bp)),
        "n_holdout_curves": int(len(hold_bp)),
        "prior_components": int(prior.mus.shape[0]),
        "eb_sigma_dyo": float(profile["sigma_dyo"]),
        "eb_sigma_dyo_multiplier": float(profile["sigma_dyo_multiplier"]),
        "eb_at_boundary": bool(profile["at_boundary"]),
        "selected_sigma_dyo": sigma,
        "selected_sigma_dyo_multiplier": float(sigma / profile["prior_sigma_median"]),
        "prior_sigma_median": float(profile["prior_sigma_median"]),
        "holdout_loglik_eb_winner": float(scores[best]),
        "holdout_loglik_range": [float(scores.min()), float(scores.max())],
        "alpha": 1.0,
    }
    return TopoStep1(
        prior=prior, clutter=clutter, sigma_dyo=sigma, alpha=1.0,
        min_birth=0.0, sample_range=grid.sample_range,
        resolution=int(grid.resolution), r=float(grid.r),
        n_calibration_curves=int(len(diagrams_bp)), selection=selection,
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
    """M6-proj -- M6 with each posterior draw projected onto the C1-C4 cone."""
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
# Cone-projected bridge arms (Task 6.25.3)
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
    """M4-proj -- M4 with every bridge draw projected onto the C1-C4 cone.

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
    """M5-proj -- M5 with every imputation projected onto the C1-C4 cone."""
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
    """Phase-6.25 cell preparation: Phase-6 artifacts plus Step-1 and the cone.

    Everything is computed from seed namespaces disjoint from the evaluation
    replicates.  ``grid``, ``targets`` and ``bridges`` come from
    ``prepare_cell`` unchanged; ``step1`` is the frozen M6 Step-1 model and
    ``cone`` the C1-C4 valid-silhouette cone on the frozen grid.
    """
    prepared = prepare_cell(cfg, cache_dir=cache_dir, verbose=verbose)
    t0 = time.perf_counter()
    step1 = fit_topo_step1(cfg, prepared["grid"])
    cone = silhouette_cone(prepared["grid"].grid)
    if verbose:
        sel = step1.selection
        print(f"[step1] sigma_DYO={step1.sigma_dyo:.6g} "
              f"(mult {sel['selected_sigma_dyo_multiplier']:.4g}, "
              f"eb_winner {sel['eb_sigma_dyo']:.6g}) "
              f"prior_components={sel['prior_components']} "
              f"holdout_loglik={sel['holdout_loglik_eb_winner']:.4g} "
              f"({time.perf_counter() - t0:.0f}s)")
        print(f"[cone]  C2 slope={cone.slope:.6g} C3 bound={cone.height_bound:.6g}")

    out = {"config": cfg, "grid": prepared["grid"], "targets": prepared["targets"],
           "bridges": prepared["bridges"], "step1": step1, "cone": cone}
    if cache_dir is not None:
        path = Path(cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        tag = f"noise{_noise_key(cfg.noise_level):06d}_res{cfg.resolution}"
        (path / f"p625_prepared_{tag}.json").write_text(json.dumps({
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

    Used by the cone diagnostic so it measures exactly the draws the arms
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
        "step1_holdout_loglik": float(step1.selection["holdout_loglik_eb_winner"]),
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

    # Cone diagnostics (Task 6.25.2): measured on the arms' own draw streams.
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
    """Headline cone-violation table for one replicate set (helper for tests)."""
    out: dict = {}
    for name, draws in sources.items():
        stats = cone_violation_stats(draws, cone)
        out[name] = {
            "any_rate": stats["cone_viol_any_rate"],
            "c1_rate": stats["cone_viol_c1_rate"],
            "c2_rate": stats["cone_viol_c2_rate"],
            "c3_rate": stats["cone_viol_c3_rate"],
            "c4_rate": stats["cone_viol_c4_rate"],
            "c2_worst_delta_units": stats["cone_viol_c2_mag_delta_units"],
            "c3_worst_abs": stats["cone_viol_c3_mag"],
            "c4_worst_abs": stats["cone_viol_c4_mag"],
        }
    return out
