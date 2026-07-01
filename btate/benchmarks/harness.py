"""Config-driven simulation harness for Phase 4 (Task 4.1 / 4.4).

Runs the Bayesian TATE pipeline (and, head-to-head, the self-contained
frequentist AIPW) over a grid of synthetic-DGP configurations and random seeds,
collecting the Phase-4 metrics: bias, RMSE, pointwise/simultaneous coverage,
interval width, no-effect test power (or type-I error under the null), and
wall-clock timing.

The harness is deliberately pure-Python/numpy so a *smoke* sweep runs in a
minimal environment; the same ``SweepCell`` objects scale up unchanged for the
full Colab runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .frequentist import frequentist_bands
from .metrics import (
    bias,
    integrated_bias,
    interval_width,
    max_abs_error,
    pointwise_coverage,
    rmse,
    simultaneous_coverage,
)
from .pipeline import PipelineConfig, run_bayesian_pipeline, silhouette_embedding_fn
from .synthetic import (
    SyntheticConfig, generate_synthetic_dataset, montecarlo_reference, reference_effect,
)


@dataclass
class SweepCell:
    """One evaluation cell: a DGP configuration + pipeline configuration.

    ``coverage_reference`` selects the estimand that credible-band coverage is
    measured against: ``"clean"`` (the injected-loop truth; fast) or
    ``"montecarlo"`` (the self-consistent estimand the estimator is unbiased
    for; averages ``mc_realizations`` noisy realizations — slower but the correct
    target for a calibration study).  Bias / RMSE are always reported against the
    clean injected truth.
    """

    name: str
    synth: SyntheticConfig
    pipeline: PipelineConfig
    n_reps: int = 5
    run_frequentist: bool = True
    coverage_reference: str = "clean"
    mc_realizations: int = 40
    freq_methods: tuple = ("multiplier_bootstrap", "liebl_reimherr", "pini_vantini")
    freq_liebl_backend: str = "python"


def _reference(dataset, pipe: PipelineConfig, sample_range):
    """Canonical clean-diagram reference psi(t) for this embedding."""
    fn = silhouette_embedding_fn(pipe, sample_range)
    return reference_effect(dataset, fn, tseq=None)


def _band_rejects(lower, upper) -> bool:
    """Uniform no-effect test: reject if the band excludes 0 at *any* grid point.

    The silhouette/landscape effect is identically 0 at the grid endpoints, so a
    whole-curve same-sign rule can never fire; the correct uniform test asks
    whether the simultaneous band lies strictly off 0 *somewhere*.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return bool(np.any(lower > 0.0) or np.any(upper < 0.0))


def _observed_fixed_silhouette(clouds, pipe: PipelineConfig, sample_range):
    """Deterministic fixed-r silhouette per subject — the frequentist input."""
    from .pipeline import h1_diagram
    from btate.embeddings import posterior_embedding_summary

    curves = []
    for c in clouds:
        d = h1_diagram(c)
        summary = posterior_embedding_summary(
            [d], embedding="silhouette", weights="power", r=pipe.r,
            sample_range=sample_range, resolution=pipe.resolution, alpha=pipe.alpha,
        )
        curves.append(summary.mean)
    return np.stack(curves)


def evaluate_run(synth: SyntheticConfig, pipe: PipelineConfig,
                 run_frequentist: bool = True,
                 coverage_reference: str = "clean",
                 mc_realizations: int = 40,
                 freq_methods=("multiplier_bootstrap", "liebl_reimherr", "pini_vantini"),
                 freq_liebl_backend: str = "python") -> dict:
    """Run one dataset end-to-end and return per-run metrics.

    Bias / RMSE are computed against the clean injected-loop truth; coverage and
    interval width are computed against ``coverage_reference`` (``"clean"`` or
    the self-consistent ``"montecarlo"`` estimand).
    """
    dataset = generate_synthetic_dataset(synth)
    result = run_bayesian_pipeline(
        dataset.observed_clouds(), dataset.A, dataset.X, dataset.pi, pipe,
    )
    grid = result.grid
    sample_range = tuple(result.meta["sample_range"])
    reference = _reference(dataset, pipe, sample_range)      # clean truth (bias/RMSE)
    if coverage_reference == "montecarlo":
        cov_ref = montecarlo_reference(
            synth, silhouette_embedding_fn(pipe, sample_range),
            n_realizations=mc_realizations,
        )
    else:
        cov_ref = reference

    effect = result.nested if result.nested is not None else result.plugin
    is_null = synth.effect_size == 0.0 and synth.effect_covariate_gain == 0.0

    record = {
        "config_tag": pipe.embedding + "/" + pipe.weights + "/" + pipe.propagation,
        "n": synth.n,
        "noise_level": synth.noise_level,
        "overlap_strength": synth.overlap_strength,
        "effect_size": synth.effect_size,
        "is_null": bool(is_null),
        # Bayesian metrics
        "bayes_rmse": rmse(effect.mean, reference),
        "bayes_bias": bias(effect.mean, reference),
        "bayes_int_bias": integrated_bias(effect.mean, reference, grid),
        "bayes_max_abs_err": max_abs_error(effect.mean, reference),
        "bayes_cov_pointwise": pointwise_coverage(
            effect.pointwise_lower, effect.pointwise_upper, cov_ref),
        "bayes_cov_simultaneous": simultaneous_coverage(
            effect.simultaneous_lower, effect.simultaneous_upper, cov_ref),
        "bayes_width": interval_width(
            effect.simultaneous_lower, effect.simultaneous_upper, grid),
        "bayes_reject": _band_rejects(
            effect.simultaneous_lower, effect.simultaneous_upper),
        "bayes_pr_excludes_zero": float(effect.pr_excludes_zero),
        "total_s": result.timing["total_s"],
        "embedding_s": result.timing["embedding_s"],
        "causal_s": result.timing["causal_s"],
    }
    if result.comparison is not None:
        record["width_ratio_nested_plugin"] = result.comparison.width_ratio
        record["bayes_width_plugin"] = interval_width(
            result.plugin.simultaneous_lower, result.plugin.simultaneous_upper, grid)

    if run_frequentist:
        phi_obs = _observed_fixed_silhouette(dataset.observed_clouds(), pipe, sample_range)
        est, bands, _ = frequentist_bands(
            phi_obs, dataset.A, dataset.X, grid, pi_hat=dataset.pi,
            alpha=pipe.alpha, methods=freq_methods, random_state=synth.seed + 99,
            liebl_backend=freq_liebl_backend,
        )
        record["freq_rmse"] = rmse(est, reference)
        record["freq_bias"] = bias(est, reference)
        for m, b in bands.items():
            record[f"freq_{m}_cov_simultaneous"] = simultaneous_coverage(
                b.lower, b.upper, cov_ref)
            record[f"freq_{m}_width"] = interval_width(b.lower, b.upper, grid)
            record[f"freq_{m}_reject"] = _band_rejects(b.lower, b.upper)
    return record


def _evaluate_cell_rep(cell: SweepCell, rep: int) -> dict:
    """Evaluate one repetition of a cell (picklable target for joblib)."""
    synth = replace(cell.synth, seed=cell.synth.seed + 1000 * rep)
    pipe = replace(cell.pipeline, seed=cell.pipeline.seed + 1000 * rep)
    rec = evaluate_run(
        synth, pipe, run_frequentist=cell.run_frequentist,
        coverage_reference=cell.coverage_reference,
        mc_realizations=cell.mc_realizations,
        freq_methods=cell.freq_methods,
        freq_liebl_backend=cell.freq_liebl_backend,
    )
    rec["rep"] = rep
    return rec


def run_cell(cell: SweepCell, verbose: bool = False, n_jobs: int = 1) -> dict:
    """Run all repetitions of a cell and aggregate the per-run metrics.

    ``n_jobs`` parallelizes the repetitions across processes (respect available
    RAM: each worker holds one dataset + pipeline).
    """
    if n_jobs == 1:
        records = []
        for rep in range(cell.n_reps):
            rec = _evaluate_cell_rep(cell, rep)
            records.append(rec)
            if verbose:
                print(f"  [{cell.name}] rep {rep}: bayes_rmse={rec['bayes_rmse']:.4f} "
                      f"cov={rec['bayes_cov_simultaneous']:.0f} "
                      f"reject={rec['bayes_reject']} t={rec['total_s']:.1f}s")
    else:
        from joblib import Parallel, delayed
        records = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_evaluate_cell_rep)(cell, rep) for rep in range(cell.n_reps)
        )
    return aggregate_cell(cell.name, records)


_IDENTIFIER_KEYS = ("n", "noise_level", "overlap_strength",
                    "effect_size", "is_null")


def aggregate_cell(name: str, records: list[dict]) -> dict:
    """Average per-run records into a single cell summary.

    Every numeric / boolean metric key present across the records is averaged
    (so booleans like ``bayes_reject`` become rates and arbitrary ``freq_<method>_*``
    keys aggregate automatically).  ``bayes_reject`` / ``freq_*_reject`` become
    the no-effect-test rejection rate — power under an effect, type-I under the
    null.  Identifier fields are taken from the first record.
    """
    first = records[0]
    agg = {"name": name, "n_reps": len(records)}
    for key in _IDENTIFIER_KEYS:
        if key in first:
            agg[key] = first[key]

    metric_keys = set()
    for r in records:
        for k, v in r.items():
            if k in _IDENTIFIER_KEYS or k == "rep":
                continue
            if isinstance(v, (int, float, bool, np.floating, np.integer)):
                metric_keys.add(k)
    for k in sorted(metric_keys):
        vals = [float(r[k]) for r in records if k in r]
        out_key = k[:-len("reject")] + "reject_rate" if k.endswith("reject") else k
        agg[out_key] = float(np.mean(vals)) if vals else float("nan")

    agg["_records"] = records
    return agg


def run_sweep(cells: list[SweepCell], verbose: bool = False,
              n_jobs: int = 1) -> list[dict]:
    """Run a list of cells and return their aggregated summaries.

    With ``n_jobs != 1`` every ``(cell, rep)`` task is flattened and dispatched
    across ``n_jobs`` processes for maximum core utilization, then regrouped and
    aggregated per cell.  Keep ``n_jobs`` <= physical cores and mind RAM.
    """
    if n_jobs == 1:
        out = []
        for cell in cells:
            if verbose:
                print(f"[cell] {cell.name} ({cell.n_reps} reps)")
            out.append(run_cell(cell, verbose=verbose))
        return out

    from joblib import Parallel, delayed
    tasks = [(ci, rep) for ci, c in enumerate(cells) for rep in range(c.n_reps)]
    if verbose:
        print(f"[sweep] {len(cells)} cells, {len(tasks)} runs, n_jobs={n_jobs}")
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5 if verbose else 0)(
        delayed(_evaluate_cell_rep)(cells[ci], rep) for ci, rep in tasks
    )
    grouped: dict[int, list] = {}
    for (ci, _rep), rec in zip(tasks, results):
        grouped.setdefault(ci, []).append(rec)
    return [aggregate_cell(cells[ci].name, grouped[ci]) for ci in range(len(cells))]


def sweep_to_rows(summaries: list[dict]) -> list[dict]:
    """Strip the raw ``_records`` for compact CSV/JSON serialization."""
    return [{k: v for k, v in s.items() if k != "_records"} for s in summaries]
