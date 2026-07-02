"""Maroulas Step-1 calibration diagnostics for Phase 4.25.

These helpers run before the causal FGP layer.  They check whether the faithful
Maroulas posterior preserves the fixed-r observed silhouette effect, or whether
Step 1 has already attenuated the topological contrast.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .metrics import integrated_bias, max_abs_error, rmse
from .pipeline import (
    PipelineConfig,
    _auto_sample_range,
    _subject_embedding_draws,
    h1_diagram,
    resolve_sigma_dyo,
)
from .synthetic import SyntheticConfig, generate_synthetic_dataset
from btate.embeddings import posterior_embedding_summary
from btate.topo_posterior import bd_to_bp


_PRIOR_VARIANTS = {"pooled", "diffuse_pooled", "arm_aware"}


def _ipw_effect(curves, A, pi_hat, clip: float) -> np.ndarray:
    curves = np.asarray(curves, dtype=float)
    A = np.asarray(A, dtype=int).ravel()
    pi = np.clip(np.asarray(pi_hat, dtype=float).ravel(), clip, 1.0 - clip)
    score = A / pi - (1 - A) / (1.0 - pi)
    return np.mean(score[:, None] * curves, axis=0)


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


def _inflate_mixture(mixture, factor: float):
    if factor == 1.0:
        return mixture
    if factor <= 0.0 or not np.isfinite(factor):
        raise ValueError("diffuse_sigma_multiplier must be positive and finite")
    from bayes_tda.intensities import RGaussianMixture

    return RGaussianMixture(
        mus=np.asarray(mixture.mus, dtype=float).copy(),
        sigmas=np.asarray(mixture.sigmas, dtype=float) * float(factor),
        weights=np.asarray(mixture.weights, dtype=float).copy(),
        normalize_weights=False,
        tilted=mixture.tilted,
        min_birth=mixture.min_birth,
        fastQ=mixture.fastQ,
    )


def _fit_prior_clutter(train_bp, pipe: PipelineConfig, random_state, diffuse_factor: float):
    from btate.topo_posterior.elicitation import elicit_prior_clutter

    if not train_bp:
        raise ValueError("cannot elicit Maroulas prior from empty diagram list")
    mean_card = max(1, int(np.mean([len(d) for d in train_bp])))
    prior, clutter = elicit_prior_clutter(
        train_bp,
        n_components=min(pipe.prior_components, mean_card),
        clutter_n_components=pipe.clutter_components,
        random_state=random_state,
    )
    if diffuse_factor != 1.0:
        prior = _inflate_mixture(prior, diffuse_factor)
        clutter = _inflate_mixture(clutter, diffuse_factor)
    return prior, clutter


def _prior_bundle(diagrams, A, pipe: PipelineConfig, prior_variant: str,
                  diffuse_sigma_multiplier: float):
    if prior_variant not in _PRIOR_VARIANTS:
        opts = ", ".join(sorted(_PRIOR_VARIANTS))
        raise ValueError(f"prior_variant must be one of {{{opts}}}")

    all_bp = [bd_to_bp(d) for d in diagrams if d.shape[0] > 0]
    if not all_bp:
        raise ValueError("all diagrams are empty; Maroulas diagnostics are undefined")

    pooled = _fit_prior_clutter(all_bp, pipe, pipe.seed, diffuse_factor=1.0)
    if prior_variant == "pooled":
        return {"kind": prior_variant, "pooled": pooled}
    if prior_variant == "diffuse_pooled":
        return {
            "kind": prior_variant,
            "pooled": _fit_prior_clutter(
                all_bp, pipe, pipe.seed, diffuse_factor=diffuse_sigma_multiplier,
            ),
        }

    out = {"kind": prior_variant, "pooled": pooled}
    A = np.asarray(A, dtype=int).ravel()
    for arm in (0, 1):
        arm_bp = [bd_to_bp(d) for d, a in zip(diagrams, A) if a == arm and d.shape[0] > 0]
        if arm_bp:
            out[arm] = _fit_prior_clutter(
                arm_bp, pipe, pipe.seed + 101 * (arm + 1), diffuse_factor=1.0,
            )
        else:
            out[arm] = pooled
    return out


def _bundle_for_subject(bundle, arm: int):
    if bundle["kind"] == "arm_aware":
        return bundle[int(arm)]
    return bundle["pooled"]


def _sigma_values_for_subjects(bundle, A, pipe: PipelineConfig) -> np.ndarray:
    vals = []
    for arm in np.asarray(A, dtype=int).ravel():
        prior, _ = _bundle_for_subject(bundle, int(arm))
        vals.append(resolve_sigma_dyo(prior, pipe)["sigma_dyo"])
    return np.asarray(vals, dtype=float)


def pre_fgp_maroulas_diagnostic(
    synth: SyntheticConfig,
    pipe: PipelineConfig | None = None,
    *,
    prior_variant: str = "pooled",
    fixed_sigma_dyo: float | None = None,
    sigma_multiplier: float | None = None,
    diffuse_sigma_multiplier: float = 10.0,
    attenuation_threshold: float = 0.75,
) -> dict:
    """Compare observed fixed-r silhouettes to Maroulas posterior means.

    The returned scalar fields are suitable for CSV rows.  Array-valued fields
    prefixed with ``_`` are included for plotting/debugging and can be stripped
    before serialization.
    """
    base = PipelineConfig() if pipe is None else pipe
    if fixed_sigma_dyo is not None and sigma_multiplier is not None:
        raise ValueError("choose either fixed_sigma_dyo or sigma_multiplier, not both")
    if fixed_sigma_dyo is not None:
        pipe_d = replace(
            base, topo_method="maroulas", weights="power", sigma_dyo=fixed_sigma_dyo,
        )
    elif sigma_multiplier is not None:
        pipe_d = replace(
            base,
            topo_method="maroulas",
            weights="power",
            sigma_dyo=None,
            sigma_dyo_multiplier=sigma_multiplier,
        )
    else:
        pipe_d = replace(base, topo_method="maroulas", weights="power")

    dataset = generate_synthetic_dataset(synth)
    clouds = dataset.observed_clouds()
    diagrams = [h1_diagram(c) for c in clouds]
    sample_range = pipe_d.sample_range or _auto_sample_range(diagrams)
    bundle = _prior_bundle(
        diagrams,
        dataset.A,
        pipe_d,
        prior_variant=prior_variant,
        diffuse_sigma_multiplier=diffuse_sigma_multiplier,
    )

    observed_curves = _fixed_power_curves(diagrams, pipe_d, sample_range)
    maroulas_curves = []
    grid = None
    for i, d in enumerate(diagrams):
        prior, clutter = _bundle_for_subject(bundle, int(dataset.A[i]))
        draws, grid = _subject_embedding_draws(
            d,
            prior,
            clutter,
            pipe_d,
            sample_range,
            seed=pipe_d.seed + 1000 * i,
        )
        maroulas_curves.append(draws.mean(axis=0))
    maroulas_curves = np.stack(maroulas_curves)

    observed_effect = _ipw_effect(
        observed_curves, dataset.A, dataset.pi, pipe_d.propensity_clip,
    )
    maroulas_effect = _ipw_effect(
        maroulas_curves, dataset.A, dataset.pi, pipe_d.propensity_clip,
    )
    grid = np.asarray(grid, dtype=float)
    sigma_values = _sigma_values_for_subjects(bundle, dataset.A, pipe_d)

    obs_norm = float(np.sqrt(np.mean(observed_effect * observed_effect)))
    mar_norm = float(np.sqrt(np.mean(maroulas_effect * maroulas_effect)))
    obs_peak = float(np.max(np.abs(observed_effect)))
    mar_peak = float(np.max(np.abs(maroulas_effect)))
    norm_ratio = mar_norm / obs_norm if obs_norm > 1e-12 else float("nan")
    peak_ratio = mar_peak / obs_peak if obs_peak > 1e-12 else float("nan")
    obs_mean = float(np.mean(observed_effect))
    mar_mean = float(np.mean(maroulas_effect))
    mean_ratio = mar_mean / obs_mean if abs(obs_mean) > 1e-12 else float("nan")

    return {
        "prior_variant": prior_variant,
        "n": int(synth.n),
        "noise_level": float(synth.noise_level),
        "effect_size": float(synth.effect_size),
        "posterior_draws": int(pipe_d.posterior_draws),
        "sigma_dyo_mode": resolve_sigma_dyo(_bundle_for_subject(bundle, 0)[0], pipe_d)[
            "sigma_dyo_mode"
        ],
        "sigma_dyo_min": float(np.min(sigma_values)),
        "sigma_dyo_median": float(np.median(sigma_values)),
        "sigma_dyo_max": float(np.max(sigma_values)),
        "sigma_dyo_multiplier": float(pipe_d.sigma_dyo_multiplier),
        "diffuse_sigma_multiplier": float(diffuse_sigma_multiplier),
        "mean_cardinality": float(np.mean([d.shape[0] for d in diagrams])),
        "observed_effect_mean": obs_mean,
        "maroulas_effect_mean": mar_mean,
        "mean_effect_ratio": mean_ratio,
        "observed_effect_peak_abs": obs_peak,
        "maroulas_effect_peak_abs": mar_peak,
        "peak_attenuation_ratio": peak_ratio,
        "observed_effect_l2": obs_norm,
        "maroulas_effect_l2": mar_norm,
        "l2_attenuation_ratio": norm_ratio,
        "rmse_to_observed_effect": rmse(maroulas_effect, observed_effect),
        "integrated_bias_to_observed_effect": integrated_bias(
            maroulas_effect, observed_effect, grid,
        ),
        "max_abs_error_to_observed_effect": max_abs_error(
            maroulas_effect, observed_effect,
        ),
        "flag_attenuated": bool(np.isfinite(norm_ratio) and norm_ratio < attenuation_threshold),
        "attenuation_threshold": float(attenuation_threshold),
        "sample_range_low": float(sample_range[0]),
        "sample_range_high": float(sample_range[1]),
        "_grid": grid,
        "_observed_effect": observed_effect,
        "_maroulas_effect": maroulas_effect,
    }


def maroulas_sigma_sensitivity(
    synth: SyntheticConfig,
    pipe: PipelineConfig | None = None,
    *,
    sigma_multipliers=(0.5, 1.0, 2.0, 3.0, 5.0, 10.0),
    fixed_sigma_dyos=(),
    prior_variants=("pooled",),
    diffuse_sigma_multiplier: float = 10.0,
    attenuation_threshold: float = 0.75,
) -> list[dict]:
    """Return pre-FGP attenuation rows over sigma and prior variants."""
    rows = []
    for variant in prior_variants:
        for mult in sigma_multipliers:
            row = pre_fgp_maroulas_diagnostic(
                synth,
                pipe,
                prior_variant=variant,
                sigma_multiplier=float(mult),
                diffuse_sigma_multiplier=diffuse_sigma_multiplier,
                attenuation_threshold=attenuation_threshold,
            )
            row["sigma_setting"] = f"adaptive_x{float(mult):g}"
            rows.append(row)
        for sig in fixed_sigma_dyos:
            row = pre_fgp_maroulas_diagnostic(
                synth,
                pipe,
                prior_variant=variant,
                fixed_sigma_dyo=float(sig),
                diffuse_sigma_multiplier=diffuse_sigma_multiplier,
                attenuation_threshold=attenuation_threshold,
            )
            row["sigma_setting"] = f"fixed_{float(sig):g}"
            rows.append(row)
    return rows


def strip_diagnostic_arrays(rows: list[dict]) -> list[dict]:
    """Drop plotting arrays from diagnostic rows before CSV/JSON serialization."""
    return [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
