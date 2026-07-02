"""Joint Step-1 / FGP calibration sweeps.

This module tunes ``sigma_dyo_multiplier`` and ``fgp_posterior_scale`` together
without recomputing Maroulas posterior embeddings for every FGP scale.  For each
synthetic replicate, it computes topological posterior embeddings once per
``sigma_dyo_multiplier`` and reuses those draws across all FGP scales.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from btate.causal import FunctionalGPEstimator, compare_propagation
from btate.embeddings import posterior_embedding_summary
from btate.topo_posterior import bd_to_bp

from .metrics import bias, integrated_bias, interval_width, max_abs_error, pointwise_coverage, rmse, simultaneous_coverage
from .pipeline import PipelineConfig, _auto_sample_range, _subject_embedding_draws, h1_diagram, resolve_sigma_dyo, silhouette_embedding_fn
from .synthetic import SyntheticConfig, generate_synthetic_dataset, montecarlo_reference, reference_effect


@dataclass
class JointCalibrationCell:
    """One DGP cell for joint ``sigma_dyo`` / FGP posterior-scale tuning."""

    name: str
    synth: SyntheticConfig
    pipeline: PipelineConfig
    n_reps: int = 8
    # Entries may be floats or the adaptive sentinels: "eb" for the
    # marginal-likelihood sigma_dyo multiplier, "godambe" for the estimated
    # FGP posterior scale.
    sigma_multipliers: tuple[float | str, ...] = (0.5, 1.0, 2.0, 3.0)
    fgp_scales: tuple[float | str, ...] = (8.0, 16.0, 32.0, 64.0)
    mc_realizations: int = 20


def _fixed_power_curves(diagrams, pipe: PipelineConfig, sample_range) -> np.ndarray:
    curves = []
    for d in diagrams:
        summary = posterior_embedding_summary(
            [d],
            embedding="silhouette",
            weights="power",
            r=pipe.r,
            sample_range=sample_range,
            resolution=pipe.resolution,
            alpha=pipe.alpha,
        )
        curves.append(summary.mean)
    return np.stack(curves)


def _fit_prior_clutter(diagrams, pipe: PipelineConfig):
    from btate.topo_posterior.elicitation import elicit_prior_clutter

    train_bp = [bd_to_bp(d) for d in diagrams if d.shape[0] > 0]
    if not train_bp:
        raise ValueError("cannot fit Maroulas prior: all diagrams are empty")
    mean_card = max(1, int(np.mean([len(d) for d in train_bp])))
    return elicit_prior_clutter(
        train_bp,
        n_components=min(pipe.prior_components, mean_card),
        clutter_n_components=pipe.clutter_components,
        random_state=pipe.seed,
    )


def _embedding_draws_for_sigma(diagrams, prior, clutter, pipe, sample_range):
    per_subject = []
    grid = None
    for i, d in enumerate(diagrams):
        draws, grid = _subject_embedding_draws(
            d,
            prior,
            clutter,
            pipe,
            sample_range,
            seed=pipe.seed + 1000 * i,
        )
        per_subject.append(draws)
    return np.transpose(np.stack(per_subject), (1, 0, 2)), grid


def _effect_metrics(effect, reference, cov_ref, grid) -> dict:
    return {
        "bayes_rmse": rmse(effect.mean, reference),
        "bayes_bias": bias(effect.mean, reference),
        "bayes_int_bias": integrated_bias(effect.mean, reference, grid),
        "bayes_max_abs_err": max_abs_error(effect.mean, reference),
        "bayes_cov_pointwise": pointwise_coverage(
            effect.pointwise_lower, effect.pointwise_upper, cov_ref,
        ),
        "bayes_cov_simultaneous": simultaneous_coverage(
            effect.simultaneous_lower, effect.simultaneous_upper, cov_ref,
        ),
        "bayes_width": interval_width(
            effect.simultaneous_lower, effect.simultaneous_upper, grid,
        ),
        "bayes_reject": bool(
            np.any(effect.simultaneous_lower > 0.0)
            or np.any(effect.simultaneous_upper < 0.0)
        ),
        "bayes_pr_excludes_zero": float(effect.pr_excludes_zero),
    }


def _run_joint_rep(cell: JointCalibrationCell, rep: int) -> list[dict]:
    synth = replace(cell.synth, seed=cell.synth.seed + 1000 * rep)
    base_pipe = replace(
        cell.pipeline,
        topo_method="maroulas",
        embedding="silhouette",
        weights="power",
        sigma_dyo=None,
        seed=cell.pipeline.seed + 1000 * rep,
    )
    dataset = generate_synthetic_dataset(synth)
    clouds = dataset.observed_clouds()
    diagrams = [h1_diagram(c) for c in clouds]
    sample_range = base_pipe.sample_range or _auto_sample_range(diagrams)
    prior, clutter = _fit_prior_clutter(diagrams, base_pipe)

    reference_fn = silhouette_embedding_fn(base_pipe, sample_range)
    reference = reference_effect(dataset, reference_fn, tseq=None)
    cov_ref = montecarlo_reference(
        synth,
        reference_fn,
        n_realizations=cell.mc_realizations,
    )

    observed_curves = _fixed_power_curves(diagrams, base_pipe, sample_range)
    a = dataset.A.astype(float)[:, None]
    pi = np.clip(dataset.pi, base_pipe.propensity_clip, 1.0 - base_pipe.propensity_clip)[:, None]
    observed_effect = np.mean((a / pi - (1.0 - a) / (1.0 - pi)) * observed_curves, axis=0)

    train_bp = [bd_to_bp(d) for d in diagrams if d.shape[0] > 0]
    rows = []
    for sigma_mult in cell.sigma_multipliers:
        pipe_sigma = replace(
            base_pipe,
            sigma_dyo=None,
            sigma_dyo_multiplier=(
                sigma_mult if isinstance(sigma_mult, str) else float(sigma_mult)
            ),
        )
        sigma_info = resolve_sigma_dyo(
            prior, pipe_sigma, diagrams_bp=train_bp, clutter=clutter,
        )
        pipe_sigma = replace(pipe_sigma, sigma_dyo=sigma_info["sigma_dyo"])
        phi_draws, grid = _embedding_draws_for_sigma(
            diagrams, prior, clutter, pipe_sigma, sample_range,
        )
        topo_mean = phi_draws.mean(axis=0)
        topo_effect = np.mean((a / pi - (1.0 - a) / (1.0 - pi)) * topo_mean, axis=0)
        topo_l2 = float(np.sqrt(np.mean(topo_effect * topo_effect)))
        observed_l2 = float(np.sqrt(np.mean(observed_effect * observed_effect)))
        topo_ratio = topo_l2 / observed_l2 if observed_l2 > 1e-12 else float("nan")

        for fgp_scale in cell.fgp_scales:
            estimator = FunctionalGPEstimator(
                n_inducing=pipe_sigma.n_inducing,
                prior_scale=pipe_sigma.prior_scale,
                length_scale_x=pipe_sigma.length_scale_x,
                length_scale_t=pipe_sigma.length_scale_t,
                noise_variance=pipe_sigma.noise_variance,
                propensity_clip=pipe_sigma.propensity_clip,
                posterior_scale=(
                    fgp_scale if isinstance(fgp_scale, str) else float(fgp_scale)
                ),
            )
            comparison = compare_propagation(
                phi_draws,
                dataset.A,
                dataset.X,
                grid,
                pi_hat=dataset.pi,
                estimator=estimator,
                n_causal_draws=pipe_sigma.n_causal_draws,
                n_plugin_draws=pipe_sigma.n_plugin_draws,
                alpha=pipe_sigma.alpha,
                random_state=pipe_sigma.seed + 5,
                potential_outcomes=False,
            )
            effect = comparison.nested
            record = {
                "cell": cell.name,
                "rep": int(rep),
                "n": int(synth.n),
                "noise_level": float(synth.noise_level),
                "effect_size": float(synth.effect_size),
                "overlap_strength": float(synth.overlap_strength),
                "sigma_setting": str(sigma_mult),
                "sigma_dyo_multiplier": float(sigma_info["sigma_dyo_multiplier"]),
                "sigma_dyo": float(sigma_info["sigma_dyo"]),
                "prior_sigma_median": float(sigma_info["prior_sigma_median"]),
                "fgp_posterior_scale": (
                    fgp_scale if isinstance(fgp_scale, str) else float(fgp_scale)
                ),
                "fgp_posterior_scale_hat": float(
                    effect.metadata.get("posterior_scale_hat_mean", float("nan"))
                ),
                "topo_l2_attenuation_ratio": topo_ratio,
                "topo_rmse_to_observed_effect": rmse(topo_effect, observed_effect),
                "width_ratio_nested_plugin": float(comparison.width_ratio),
                "bayes_width_plugin": interval_width(
                    comparison.plugin.simultaneous_lower,
                    comparison.plugin.simultaneous_upper,
                    grid,
                ),
            }
            record.update(_effect_metrics(effect, reference, cov_ref, grid))
            rows.append(record)
    return rows


_ID_KEYS = (
    "cell",
    "n",
    "noise_level",
    "effect_size",
    "overlap_strength",
    "sigma_setting",
    "fgp_posterior_scale",
)


def aggregate_joint_records(records: list[dict]) -> list[dict]:
    """Aggregate per-rep joint calibration records by tuning setting."""
    groups: dict[tuple, list[dict]] = {}
    for rec in records:
        key = tuple(rec[k] for k in _ID_KEYS)
        groups.setdefault(key, []).append(rec)

    out = []
    # Keys may mix floats and adaptive sentinels (e.g. 16.0 vs "godambe");
    # sort on the string form so mixed grids aggregate cleanly.
    for key, vals in sorted(groups.items(),
                            key=lambda kv: tuple(str(k) for k in kv[0])):
        first = vals[0]
        row = {k: first[k] for k in _ID_KEYS}
        row["n_reps"] = len(vals)
        metric_keys = set()
        for rec in vals:
            for k, v in rec.items():
                if k in _ID_KEYS or k == "rep":
                    continue
                if isinstance(v, (int, float, bool, np.integer, np.floating)):
                    metric_keys.add(k)
        for k in sorted(metric_keys):
            numbers = [float(rec[k]) for rec in vals if k in rec]
            out_key = k[:-len("reject")] + "reject_rate" if k.endswith("reject") else k
            row[out_key] = float(np.mean(numbers)) if numbers else float("nan")
        out.append(row)
    return out


def run_joint_calibration(cell: JointCalibrationCell, n_jobs: int = 1,
                          verbose: bool = False) -> tuple[list[dict], list[dict]]:
    """Run and aggregate a joint calibration cell.

    Returns ``(summary_rows, raw_records)``.
    """
    if n_jobs == 1:
        raw = []
        for rep in range(cell.n_reps):
            if verbose:
                print(f"[{cell.name}] rep {rep + 1}/{cell.n_reps}", flush=True)
            raw.extend(_run_joint_rep(cell, rep))
    else:
        from joblib import Parallel, delayed

        if verbose:
            print(f"[{cell.name}] {cell.n_reps} reps, n_jobs={n_jobs}", flush=True)
        chunks = Parallel(n_jobs=n_jobs, backend="loky", verbose=5 if verbose else 0)(
            delayed(_run_joint_rep)(cell, rep) for rep in range(cell.n_reps)
        )
        raw = [rec for chunk in chunks for rec in chunk]
    return aggregate_joint_records(raw), raw
