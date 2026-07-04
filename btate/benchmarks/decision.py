"""Phase-4.5 decision-grade sweep driver.

Turns the low-rep smoke evidence of Phase 4 into decision-grade findings
(Research_Plan Phase 4.5).  One :class:`DecisionCell` fixes a DGP regime and a
set of Step-4 model variants; :func:`run_decision_cell` evaluates every variant
on a *disjoint* seed base with ``>= n_reps`` replicates and scores every
credible/confidence band against **both** estimands (Task 4.5.1):

* ``clean``  — the injected-loop truth the denoising method actually targets;
* ``mc``     — the self-consistent raw-silhouette ``psi*`` the frequentist AIPW
               is unbiased for by construction.

Efficiency (so ``>= 50`` reps of the faithful Maroulas path are affordable):
per replicate the dataset, the two reference curves, and the Maroulas
prior/clutter are built **once**; Maroulas posterior embeddings are computed
**once per (weights, sigma_dyo_multiplier)** and reused across every FGP scale
and the functional-BCF fit (the joint-calibration pattern, extended to BCF and
the frequentist).  Aggregation attaches Clopper–Pearson error bars to the
simultaneous-coverage rates (Task 4.5.3 / higher-rep confirmation).

The four Step-4 competitors and both estimands share the *same* datasets and
Maroulas embeddings, so the head-to-head is apples-to-apples.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from btate.causal import FunctionalGPEstimator, compare_propagation
from btate.topo_posterior import bd_to_bp

from .frequentist import aipw_effect
from .metrics import (
    bias, clopper_pearson, integrated_bias, interval_width, max_abs_error,
    pointwise_coverage, rmse, simultaneous_coverage,
)
from .pipeline import (
    PipelineConfig, _auto_sample_range, _subject_embedding_draws, h1_diagram,
    resolve_sigma_dyo, silhouette_embedding_fn,
)
from .synthetic import (
    SyntheticConfig, generate_synthetic_dataset, montecarlo_reference,
    reference_effect,
)

# Seed base disjoint from the Phase-4 / joint-calibration tuning seeds
# (``20260701 + 1000*rep``): every decision replicate is fresh data.
DECISION_SEED_OFFSET = 900000


@dataclass
class FGPVariant:
    """One FGP Step-1/Step-4 calibration to evaluate.

    ``sigma_dyo_multiplier`` and ``fgp_posterior_scale`` accept the same values
    as :class:`PipelineConfig` (floats or the ``"eb"`` / ``"godambe"`` sentinels).
    """

    label: str
    sigma_dyo_multiplier: float | str = 3.0
    fgp_posterior_scale: float | str = 8.0


@dataclass
class DecisionCell:
    """A DGP regime + the Step-4 competitors to score on it."""

    name: str
    synth: SyntheticConfig
    pipeline: PipelineConfig
    n_reps: int = 50
    weights_variants: tuple[str, ...] = ("power",)     # ("power", "pi") for regime map
    fgp_variants: tuple[FGPVariant, ...] = (
        FGPVariant("fixed8", 3.0, 8.0),
        FGPVariant("godambe", 3.0, "godambe"),
        FGPVariant("eb_godambe", "eb", "godambe"),
    )
    run_frequentist: bool = True
    run_bcf: bool = False
    # BCF reuses the embedding of this (weights, sigma_dyo_multiplier) group.
    bcf_weights: str = "power"
    bcf_sigma_dyo_multiplier: float | str = "eb"
    bcf_kwargs: dict = field(default_factory=dict)
    mc_realizations: int = 24
    seed_offset: int = DECISION_SEED_OFFSET


# --------------------------------------------------------------------------- #
# Per-replicate evaluation
# --------------------------------------------------------------------------- #
def _fit_prior_clutter(diagrams, pipe: PipelineConfig):
    from btate.topo_posterior.elicitation import elicit_prior_clutter

    train_bp = [bd_to_bp(d) for d in diagrams if d.shape[0] > 0]
    if not train_bp:
        raise ValueError("cannot fit Maroulas prior: all diagrams are empty")
    mean_card = max(1, int(np.mean([len(d) for d in train_bp])))
    prior, clutter = elicit_prior_clutter(
        train_bp,
        n_components=min(pipe.prior_components, mean_card),
        clutter_n_components=pipe.clutter_components,
        random_state=pipe.seed,
    )
    return prior, clutter, train_bp


def _observed_fixed_silhouette(diagrams, pipe: PipelineConfig, sample_range):
    from btate.embeddings import posterior_embedding_summary

    curves = []
    for d in diagrams:
        s = posterior_embedding_summary(
            [d], embedding="silhouette", weights="power", r=pipe.r,
            sample_range=sample_range, resolution=pipe.resolution, alpha=pipe.alpha,
        )
        curves.append(s.mean)
    return np.stack(curves)


def _embedding_draws(diagrams, prior, clutter, pipe, sample_range):
    per_subject = []
    grid = None
    for i, d in enumerate(diagrams):
        draws, grid = _subject_embedding_draws(
            d, prior, clutter, pipe, sample_range, seed=pipe.seed + 1000 * i,
        )
        per_subject.append(draws)
    return np.transpose(np.stack(per_subject), (1, 0, 2)), grid


def _dr_effect(curves, A, pi, clip):
    """IPW/DR-weighted mean effect of a per-subject curve stack (attenuation aux)."""
    a = np.asarray(A, dtype=float)[:, None]
    p = np.clip(pi, clip, 1.0 - clip)[:, None]
    return np.mean((a / p - (1.0 - a) / (1.0 - p)) * curves, axis=0)


def _band_rejects(lower, upper) -> bool:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return bool(np.any(lower > 0.0) or np.any(upper < 0.0))


def _score(effect, ref_clean, ref_mc, grid, model, weights, sigma_label,
           scale_label, extra=None) -> dict:
    """Common metric row for any effect posterior (FGP / BCF / frequentist)."""
    rec = {
        "model": model, "weights": weights,
        "sigma_setting": sigma_label, "scale_setting": scale_label,
        "rmse": rmse(effect.mean, ref_clean),          # RMSE always vs clean truth
        "bias": bias(effect.mean, ref_clean),
        "int_bias": integrated_bias(effect.mean, ref_clean, grid),
        "max_abs_err": max_abs_error(effect.mean, ref_clean),
        "cov_pw_clean": pointwise_coverage(
            effect.pointwise_lower, effect.pointwise_upper, ref_clean),
        "cov_sim_clean": simultaneous_coverage(
            effect.simultaneous_lower, effect.simultaneous_upper, ref_clean),
        "cov_pw_mc": pointwise_coverage(
            effect.pointwise_lower, effect.pointwise_upper, ref_mc),
        "cov_sim_mc": simultaneous_coverage(
            effect.simultaneous_lower, effect.simultaneous_upper, ref_mc),
        "width": interval_width(
            effect.simultaneous_lower, effect.simultaneous_upper, grid),
        "reject": _band_rejects(effect.simultaneous_lower, effect.simultaneous_upper),
    }
    if extra:
        rec.update(extra)
    return rec


def evaluate_decision_rep(cell: DecisionCell, rep: int) -> list[dict]:
    """Evaluate every Step-4 competitor for one replicate of ``cell``."""
    synth = replace(cell.synth, seed=cell.synth.seed + cell.seed_offset + 1000 * rep)
    base = replace(
        cell.pipeline, topo_method="maroulas", embedding="silhouette",
        sigma_dyo=None, seed=cell.pipeline.seed + cell.seed_offset + 1000 * rep,
    )
    dataset = generate_synthetic_dataset(synth)
    A, X, pi = dataset.A, dataset.X, dataset.pi
    diagrams = [h1_diagram(c) for c in dataset.observed_clouds()]
    sample_range = base.sample_range or _auto_sample_range(diagrams)

    ref_fn = silhouette_embedding_fn(base, sample_range)
    ref_clean = reference_effect(dataset, ref_fn, tseq=None)
    ref_mc = montecarlo_reference(synth, ref_fn, n_realizations=cell.mc_realizations)

    # Fixed-r observed silhouette: frequentist input + attenuation baseline.
    phi_obs = _observed_fixed_silhouette(diagrams, base, sample_range)
    observed_effect = _dr_effect(phi_obs, A, pi, base.propensity_clip)
    observed_l2 = float(np.sqrt(np.mean(observed_effect ** 2)))

    prior, clutter, train_bp = _fit_prior_clutter(diagrams, base)
    rows: list[dict] = []
    id_fields = {
        "cell": cell.name, "rep": int(rep), "n": int(synth.n),
        "clutter_mode": synth.clutter_mode,
        "noise_level": float(synth.noise_level),
        "effect_size": float(synth.effect_size),
        "overlap_strength": float(synth.overlap_strength),
    }

    grid = None
    for weights in cell.weights_variants:
        # Every FGP variant is evaluated under each requested embedding weight;
        # its Maroulas embedding is shared by sigma_dyo_multiplier.
        sigma_settings = {v.sigma_dyo_multiplier for v in cell.fgp_variants}
        if cell.run_bcf and weights == cell.bcf_weights:
            sigma_settings.add(cell.bcf_sigma_dyo_multiplier)

        for sigma_mult in sigma_settings:
            pipe_s = replace(base, weights=weights, sigma_dyo=None,
                             sigma_dyo_multiplier=sigma_mult)
            info = resolve_sigma_dyo(prior, pipe_s, diagrams_bp=train_bp, clutter=clutter)
            pipe_s = replace(pipe_s, sigma_dyo=info["sigma_dyo"])
            phi_draws, grid = _embedding_draws(diagrams, prior, clutter, pipe_s, sample_range)
            topo_mean = phi_draws.mean(axis=0)
            topo_effect = _dr_effect(topo_mean, A, pi, base.propensity_clip)
            topo_l2 = float(np.sqrt(np.mean(topo_effect ** 2)))
            atten = topo_l2 / observed_l2 if observed_l2 > 1e-12 else float("nan")
            aux = {
                "sigma_dyo": float(info["sigma_dyo"]),
                "sigma_dyo_multiplier": float(info["sigma_dyo_multiplier"]),
                "topo_l2_attenuation_ratio": atten,
            }

            # --- FGP variants sharing this (weights, sigma) embedding ---
            for v in cell.fgp_variants:
                if v.sigma_dyo_multiplier != sigma_mult:
                    continue
                est = FunctionalGPEstimator(
                    n_inducing=pipe_s.n_inducing, prior_scale=pipe_s.prior_scale,
                    length_scale_x=pipe_s.length_scale_x,
                    length_scale_t=pipe_s.length_scale_t,
                    noise_variance=pipe_s.noise_variance,
                    propensity_clip=pipe_s.propensity_clip,
                    posterior_scale=v.fgp_posterior_scale,
                )
                cmp = compare_propagation(
                    phi_draws, A, X, grid, pi_hat=pi, estimator=est,
                    n_causal_draws=pipe_s.n_causal_draws,
                    n_plugin_draws=pipe_s.n_plugin_draws,
                    alpha=pipe_s.alpha, random_state=pipe_s.seed + 5,
                    potential_outcomes=False,
                )
                eff = cmp.nested
                scale_hat = eff.metadata.get("posterior_scale_hat_mean", float("nan"))
                extra = dict(aux, fgp_posterior_scale_hat=float(scale_hat),
                             width_ratio_nested_plugin=float(cmp.width_ratio),
                             pr_excludes_zero=float(eff.pr_excludes_zero))
                rows.append({**id_fields, **_score(
                    eff, ref_clean, ref_mc, grid, f"fgp_{v.label}", weights,
                    str(v.sigma_dyo_multiplier), str(v.fgp_posterior_scale), extra)})

            # --- functional BCF on the same embedding (plug-in) ---
            if (cell.run_bcf and weights == cell.bcf_weights
                    and sigma_mult == cell.bcf_sigma_dyo_multiplier):
                from btate.causal import fit_tsbcf_tate
                bcf = fit_tsbcf_tate(
                    phi_draws, A, X, grid, pi_hat=pi, alpha=pipe_s.alpha,
                    **dict(cell.bcf_kwargs))
                extra = dict(aux, pr_excludes_zero=float(bcf.pr_excludes_zero))
                rows.append({**id_fields, **_score(
                    bcf, ref_clean, ref_mc, grid, "bcf", weights,
                    str(sigma_mult), "bcf", extra)})

    # --- frequentist AIPW on the fixed-r observed silhouette (once) ---
    # Doubly-robust EIF AIPW with pointwise + multiplier-bootstrap uniform bands
    # (the Kim & Lee 2026 yardstick); scored on the same grid / estimands.
    if cell.run_frequentist:
        gref = grid if grid is not None else np.linspace(
            sample_range[0], sample_range[1], base.resolution)
        fe = aipw_effect(phi_obs, A, X, gref, pi_hat=pi, alpha=base.alpha,
                         random_state=synth.seed + 99)

        class _FreqEff:  # adapt to the _score interface (.mean, band bounds)
            mean = fe.estimate
            pointwise_lower = fe.pointwise_lower
            pointwise_upper = fe.pointwise_upper
            simultaneous_lower = fe.simultaneous_lower
            simultaneous_upper = fe.simultaneous_upper
        rows.append({**id_fields, **_score(
            _FreqEff(), ref_clean, ref_mc, gref, "frequentist", "power",
            "raw", "aipw")})
    return rows


# --------------------------------------------------------------------------- #
# Aggregation with Clopper–Pearson error bars
# --------------------------------------------------------------------------- #
_GROUP_KEYS = ("cell", "model", "weights", "sigma_setting", "scale_setting")
_ID_KEYS = ("n", "clutter_mode", "noise_level", "effect_size", "overlap_strength")
_COVERAGE_KEYS = ("cov_sim_clean", "cov_sim_mc")


def aggregate_decision(records: list[dict]) -> list[dict]:
    """Mean every metric per model group; add Clopper–Pearson coverage bars."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault(tuple(r[k] for k in _GROUP_KEYS), []).append(r)

    out = []
    for key, vals in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        first = vals[0]
        row = {k: first[k] for k in _GROUP_KEYS}
        row.update({k: first[k] for k in _ID_KEYS if k in first})
        row["n_reps"] = len(vals)
        metric_keys = set()
        for r in vals:
            for k, v in r.items():
                if k in _GROUP_KEYS or k in _ID_KEYS or k == "rep":
                    continue
                if isinstance(v, (int, float, bool, np.integer, np.floating)):
                    metric_keys.add(k)
        for k in sorted(metric_keys):
            nums = [float(r[k]) for r in vals if k in r]
            out_key = k[:-len("reject")] + "reject_rate" if k.endswith("reject") else k
            row[out_key] = float(np.mean(nums)) if nums else float("nan")
        for k in _COVERAGE_KEYS:
            hits = [r[k] for r in vals if k in r]
            if hits:
                succ = int(round(float(np.sum(hits))))
                lo, hi = clopper_pearson(succ, len(hits))
                row[f"{k}_cp_lo"] = lo
                row[f"{k}_cp_hi"] = hi
        out.append(row)
    return out


def run_decision_cell(cell: DecisionCell, n_jobs: int = 1,
                      verbose: bool = False) -> tuple[list[dict], list[dict]]:
    """Run and aggregate one decision cell. Returns ``(summary, raw_records)``."""
    if n_jobs == 1:
        raw: list[dict] = []
        for rep in range(cell.n_reps):
            if verbose:
                print(f"[{cell.name}] rep {rep + 1}/{cell.n_reps}", flush=True)
            raw.extend(evaluate_decision_rep(cell, rep))
    else:
        from joblib import Parallel, delayed
        if verbose:
            print(f"[{cell.name}] {cell.n_reps} reps, n_jobs={n_jobs}", flush=True)
        chunks = Parallel(n_jobs=n_jobs, backend="loky", verbose=5 if verbose else 0)(
            delayed(evaluate_decision_rep)(cell, rep) for rep in range(cell.n_reps)
        )
        raw = [r for chunk in chunks for r in chunk]
    return aggregate_decision(raw), raw


def run_decision_grid(cells: list[DecisionCell], n_jobs: int = 1,
                      verbose: bool = False) -> tuple[list[dict], list[dict]]:
    """Run several cells; flatten ``(cell, rep)`` tasks across ``n_jobs``."""
    if n_jobs == 1:
        summ, raw = [], []
        for c in cells:
            s, r = run_decision_cell(c, n_jobs=1, verbose=verbose)
            summ.extend(s)
            raw.extend(r)
        return summ, raw

    from joblib import Parallel, delayed
    tasks = [(ci, rep) for ci, c in enumerate(cells) for rep in range(c.n_reps)]
    if verbose:
        print(f"[grid] {len(cells)} cells, {len(tasks)} reps, n_jobs={n_jobs}", flush=True)
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5 if verbose else 0)(
        delayed(evaluate_decision_rep)(cells[ci], rep) for ci, rep in tasks
    )
    per_cell: dict[int, list[dict]] = {}
    for (ci, _rep), chunk in zip(tasks, results):
        per_cell.setdefault(ci, []).extend(chunk)
    summ, raw = [], []
    for ci in range(len(cells)):
        recs = per_cell.get(ci, [])
        raw.extend(recs)
        summ.extend(aggregate_decision(recs))
    return summ, raw
