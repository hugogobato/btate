"""Representation-first transition sweep for the clean TATE estimand.

The Phase-5.5 low-SNR experiments only examined a few, already severe,
structured-contamination settings.  This module scans a finer noise grid before
running an expensive causal-UQ comparison.  It uses the known pair of potential
outcomes for every synthetic subject, so the diagnostic is about representation
error rather than treatment-assignment error.

For every ``(noise, replicate, filtration)`` cell it reports:

* the apex-anchored floor between the observed and clean causal-effect curves;
* the relative displacement of the observed effect apex;
* relative error in the death coordinate of the most-persistent H1 feature;
* whole-curve ``within_filtration_*`` error between the observed effect curve
  and *its own filtration's* clean effect curve; and
* the exact contaminating-point fraction implied by the synthetic DGP.

**The ``within_filtration_*`` columns are within-representation only.**  A DTM
row compares DTM-observed against DTM-clean, whose clean effect amplitude and
filtration coordinates differ from the Alpha ones (clean Alpha peak ~0.170 vs
clean DTM-k15 peak ~0.296 on this DGP).  A small DTM ``within_filtration_rmse``
therefore says nothing about recovery of the clean *Alpha* estimand; only the
explicitly-fitted bridge in :mod:`btate.benchmarks.measurement_error_uq` scores
that, under ``clean_alpha_*`` names.

``alpha`` is the naive representation and ``dtm_rips_k15`` is the
pre-registered repaired representation.  Other DTM neighbourhoods are retained
as sensitivity analyses, not used to select the transition cell.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from btate.benchmarks.dtm import h1_diagram_filtration, top_feature_death
from btate.benchmarks.metrics import (
    apex_floor,
    apex_location,
    clopper_pearson,
    fundamental_floor,
    integrated_abs_error,
    max_abs_error,
    nrmse,
    rmse,
)
from btate.benchmarks.synthetic import (
    SyntheticConfig,
    generate_synthetic_dataset,
    low_snr_config,
)
from btate.embeddings import posterior_embedding_summary


DEFAULT_NOISE_LEVELS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
DEFAULT_FILTRATIONS = (
    "alpha",
    "dtm_rips_k5",
    "dtm_rips_k15",
    "dtm_rips_k30",
)
NAIVE_FILTRATION = "alpha"
REPAIRED_FILTRATION = "dtm_rips_k15"


def contamination_profile(config: SyntheticConfig) -> dict[str, float | int]:
    """Return the exact point-count contamination profile for ``config``."""
    n_clutter = int(round(config.base_clutter * config.noise_level))
    if config.clutter_mode == "structured_loops":
        n_decoy_loops = int(round(
            config.n_decoy_loops * config.noise_level
        ))
    else:
        n_decoy_loops = 0
    n_decoy_points = n_decoy_loops * int(config.decoy_points)
    n_signal_points = int(config.num_pts)
    n_contaminating_points = n_clutter + n_decoy_points
    n_total_points = n_signal_points + n_contaminating_points
    contamination_fraction = (
        n_contaminating_points / n_total_points
        if n_total_points > 0 else float("nan")
    )
    return {
        "n_signal_points": n_signal_points,
        "n_decoy_loops": n_decoy_loops,
        "n_decoy_points": n_decoy_points,
        "n_peripheral_clutter": n_clutter,
        "n_contaminating_points": n_contaminating_points,
        "n_total_points": n_total_points,
        "contamination_fraction": float(contamination_fraction),
    }


def _seeded_subsample(
    points: np.ndarray,
    max_points: int | None,
    seed_parts: Sequence[int],
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if max_points is None or len(points) <= max_points:
        return points
    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
    keep = rng.choice(len(points), int(max_points), replace=False)
    return points[keep]


def _silhouette_curve(
    diagram: np.ndarray,
    sample_range: tuple[float, float],
    resolution: int,
    r: float,
) -> tuple[np.ndarray, np.ndarray]:
    summary = posterior_embedding_summary(
        [diagram],
        embedding="silhouette",
        weights="power",
        r=r,
        sample_range=sample_range,
        resolution=resolution,
        alpha=0.05,
    )
    return np.asarray(summary.mean, dtype=float), np.asarray(
        summary.grid, dtype=float
    )


def _finite_deaths(diagrams: Iterable[np.ndarray]) -> list[float]:
    out: list[float] = []
    for diagram in diagrams:
        diagram = np.asarray(diagram, dtype=float)
        if diagram.size:
            vals = diagram[:, 1]
            out.extend(vals[np.isfinite(vals)].tolist())
    return out


def _safe_mean(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def evaluate_transition_rep(
    noise_level: float,
    rep: int,
    *,
    filtrations: Sequence[str] = DEFAULT_FILTRATIONS,
    n_subjects: int = 40,
    max_points: int | None = 200,
    seed_base: int = 20260701,
    resolution: int = 100,
    r: float = 3.0,
) -> list[dict]:
    """Evaluate one paired representation replicate over all filtrations.

    A dedicated no-clutter dataset supplies one clean reference that is
    invariant across noise levels within a replicate.  The noisy datasets share
    the structural seed, hence covariates and subject-specific radii, but use a
    separate topological-noise stream.
    """
    if n_subjects < 1:
        raise ValueError("n_subjects must be positive")
    if resolution < 8:
        raise ValueError("resolution must be at least 8")
    noise_level = float(noise_level)
    rep = int(rep)
    structural_seed = int(seed_base + 137 * rep)
    noise_key = int(round(1000.0 * noise_level))

    reference_cfg = low_snr_config(
        n=max(n_subjects, 20),
        noise_level=0.0,
        seed=structural_seed,
        noise_seed=seed_base + 8_000_000 + 137 * rep,
    )
    observed_cfg = low_snr_config(
        n=max(n_subjects, 20),
        noise_level=noise_level,
        seed=structural_seed,
        noise_seed=seed_base + 9_000_000 + 10_000 * rep + noise_key,
    )
    reference_ds = generate_synthetic_dataset(reference_cfg)
    observed_ds = generate_synthetic_dataset(observed_cfg)
    if not np.allclose(reference_ds.X, observed_ds.X):
        raise RuntimeError("structural pairing failed: covariates differ")

    n_use = min(n_subjects, observed_ds.clouds.shape[0])
    profile = contamination_profile(observed_cfg)
    rows: list[dict] = []

    for filtration in filtrations:
        clean_diagrams: list[list[np.ndarray]] = []
        observed_diagrams: list[list[np.ndarray]] = []
        for subject in range(n_use):
            clean_arms: list[np.ndarray] = []
            observed_arms: list[np.ndarray] = []
            for arm in (0, 1):
                clean_cloud = _seeded_subsample(
                    reference_ds.clean_clouds[subject, arm],
                    max_points,
                    (seed_base, rep, subject, arm, 0),
                )
                observed_cloud = _seeded_subsample(
                    observed_ds.clouds[subject, arm],
                    max_points,
                    (seed_base, rep, noise_key, subject, arm, 1),
                )
                clean_arms.append(
                    h1_diagram_filtration(clean_cloud, filtration)
                )
                observed_arms.append(
                    h1_diagram_filtration(observed_cloud, filtration)
                )
            clean_diagrams.append(clean_arms)
            observed_diagrams.append(observed_arms)

        # Keep the filtration grid invariant across noise levels.  Deriving the
        # range from observed deaths would let contamination change grid
        # resolution and recreate the Phase-5 range confound.
        clean_flat = [
            diagram for paired in clean_diagrams for diagram in paired
        ]
        deaths = _finite_deaths(clean_flat)
        if not deaths:
            raise RuntimeError(
                f"clean reference has no finite H1 deaths for {filtration}"
            )
        upper = max(1e-6, 1.5 * max(deaths))
        sample_range = (0.0, float(upper))

        clean_curves = np.zeros((n_use, 2, resolution), dtype=float)
        observed_curves = np.zeros((n_use, 2, resolution), dtype=float)
        grid: np.ndarray | None = None
        clean_deaths: list[float] = []
        observed_deaths: list[float] = []
        relative_death_errors: list[float] = []

        for subject in range(n_use):
            for arm in (0, 1):
                clean_curve, grid = _silhouette_curve(
                    clean_diagrams[subject][arm],
                    sample_range,
                    resolution,
                    r,
                )
                observed_curve, _ = _silhouette_curve(
                    observed_diagrams[subject][arm],
                    sample_range,
                    resolution,
                    r,
                )
                clean_curves[subject, arm] = clean_curve
                observed_curves[subject, arm] = observed_curve

                death_clean = top_feature_death(
                    clean_diagrams[subject][arm]
                )
                death_observed = top_feature_death(
                    observed_diagrams[subject][arm]
                )
                if (
                    np.isfinite(death_clean)
                    and np.isfinite(death_observed)
                    and death_clean > 1e-12
                ):
                    clean_deaths.append(death_clean)
                    observed_deaths.append(death_observed)
                    relative_death_errors.append(
                        (death_observed - death_clean) / death_clean
                    )

        if grid is None:
            continue
        psi_clean = np.mean(
            clean_curves[:, 1] - clean_curves[:, 0], axis=0
        )
        psi_observed = np.mean(
            observed_curves[:, 1] - observed_curves[:, 0], axis=0
        )
        clean_effect_peak = float(np.max(np.abs(psi_clean)))
        if not np.isfinite(clean_effect_peak) or clean_effect_peak <= 1e-10:
            raise RuntimeError(
                "clean causal effect is numerically zero for "
                f"rep={rep}, filtration={filtration}"
            )
        t_clean = apex_location(psi_clean, grid)
        t_observed = apex_location(psi_observed, grid)
        relative_apex_shift = (
            (t_observed - t_clean) / t_clean
            if abs(t_clean) > 1e-12 else float("nan")
        )
        f_apex = apex_floor(psi_observed, psi_clean, grid)
        clean_death_mean = _safe_mean(clean_deaths)
        observed_death_mean = _safe_mean(observed_deaths)

        row = {
            "noise_level": noise_level,
            "rep": rep,
            "filtration": filtration,
            "n_subjects": n_use,
            "max_points": max_points,
            "resolution": resolution,
            "r": float(r),
            **profile,
            "sample_range_upper": upper,
            "clean_effect_peak": clean_effect_peak,
            "observed_effect_peak": float(np.max(np.abs(psi_observed))),
            "F": fundamental_floor(psi_observed, psi_clean),
            "F_apex": f_apex,
            "clean_apex": t_clean,
            "observed_apex": t_observed,
            "apex_shift": float(t_observed - t_clean),
            "relative_apex_shift": float(relative_apex_shift),
            "death_clean_mean": clean_death_mean,
            "death_observed_mean": observed_death_mean,
            "death_relative_error": _safe_mean(relative_death_errors),
            "death_abs_relative_error": _safe_mean(
                np.abs(relative_death_errors).tolist()
            ),
            "n_valid_death_pairs": len(relative_death_errors),
            # Whole-curve error, observed vs clean *within this filtration*.
            # Never a clean-Alpha score for a DTM row (see the module docstring).
            "within_filtration_rmse": rmse(psi_observed, psi_clean),
            "within_filtration_nrmse": nrmse(psi_observed, psi_clean),
            "within_filtration_max_abs_error": max_abs_error(
                psi_observed, psi_clean
            ),
            "within_filtration_integrated_abs_error": integrated_abs_error(
                psi_observed, psi_clean, grid
            ),
            # Metric collapse checks.  These must be exactly zero up to
            # floating-point error, independently of the DGP.
            "collapse_F_apex": apex_floor(psi_clean, psi_clean, grid),
            "collapse_apex_shift": float(
                apex_location(psi_clean, grid)
                - apex_location(psi_clean, grid)
            ),
            "collapse_within_filtration_rmse": rmse(psi_clean, psi_clean),
            "collapse_within_filtration_max_abs_error": max_abs_error(
                psi_clean, psi_clean
            ),
            "collapse_within_filtration_integrated_abs_error": (
                integrated_abs_error(psi_clean, psi_clean, grid)
            ),
        }
        rows.append(row)
    return rows


def run_transition_sweep(
    *,
    noise_levels: Sequence[float] = DEFAULT_NOISE_LEVELS,
    filtrations: Sequence[str] = DEFAULT_FILTRATIONS,
    rep_indices: Sequence[int] = (0,),
    n_subjects: int = 40,
    max_points: int | None = 200,
    n_jobs: int = 8,
    seed_base: int = 20260701,
    resolution: int = 100,
    r: float = 3.0,
    verbose: int = 5,
) -> pd.DataFrame:
    """Run the representation sweep in bounded parallel jobs."""
    from joblib import Parallel, delayed

    jobs = [
        (float(noise), int(rep))
        for noise in noise_levels
        for rep in rep_indices
    ]
    if not jobs:
        return pd.DataFrame()
    workers = max(1, min(int(n_jobs), 12, len(jobs)))
    chunks = Parallel(n_jobs=workers, verbose=verbose)(
        delayed(evaluate_transition_rep)(
            noise,
            rep,
            filtrations=filtrations,
            n_subjects=n_subjects,
            max_points=max_points,
            seed_base=seed_base,
            resolution=resolution,
            r=r,
        )
        for noise, rep in jobs
    )
    raw = pd.DataFrame([row for chunk in chunks for row in chunk])
    if raw.empty:
        return raw

    collapse_columns = [
        column for column in raw.columns if column.startswith("collapse_")
    ]
    collapse_max = raw[collapse_columns].abs().to_numpy().max()
    if not np.isfinite(collapse_max) or collapse_max > 1e-10:
        raise RuntimeError(
            "same-curve metric collapse check failed: "
            f"maximum absolute discrepancy={collapse_max:.3g}"
        )

    for (rep, filtration), cell in raw.groupby(["rep", "filtration"]):
        grid_bounds = cell["sample_range_upper"].to_numpy(dtype=float)
        if not np.allclose(
            grid_bounds, grid_bounds[0], rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(
                "clean-derived grid changed across noise for "
                f"rep={rep}, filtration={filtration}"
            )
    return raw


def summarize_transition_sweep(
    raw: pd.DataFrame,
    *,
    naive_filtration: str = NAIVE_FILTRATION,
    repaired_filtration: str = REPAIRED_FILTRATION,
    naive_floor_interval: tuple[float, float] = (0.4, 0.8),
    repaired_floor_max: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate replicates and flag pre-registered transition cells."""
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    required = {"noise_level", "rep", "filtration", "F_apex"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"raw transition frame is missing {sorted(missing)}")

    metric_columns = [
        "F",
        "F_apex",
        "relative_apex_shift",
        "death_relative_error",
        "death_abs_relative_error",
        "contamination_fraction",
        "clean_effect_peak",
        "observed_effect_peak",
        "within_filtration_rmse",
        "within_filtration_nrmse",
        "within_filtration_max_abs_error",
        "within_filtration_integrated_abs_error",
        "collapse_F_apex",
        "collapse_apex_shift",
    ]
    metrics = [column for column in metric_columns if column in raw.columns]
    grouped = raw.groupby(
        ["noise_level", "filtration"], as_index=False
    )[metrics].agg(["mean", "std", "count"])
    grouped.columns = [
        "_".join(part for part in column if part)
        for column in grouped.columns.to_flat_index()
    ]
    grouped = grouped.rename(
        columns={
            "noise_level_": "noise_level",
            "filtration_": "filtration",
        }
    )

    f_table = grouped.pivot(
        index="noise_level",
        columns="filtration",
        values="F_apex_mean",
    )
    transition = pd.DataFrame(index=f_table.index)
    transition["F_apex_naive"] = f_table.get(naive_filtration)
    transition["F_apex_repaired"] = f_table.get(repaired_filtration)
    transition["naive_filtration"] = naive_filtration
    transition["repaired_filtration"] = repaired_filtration
    lo, hi = map(float, naive_floor_interval)
    transition["naive_in_transition_band"] = (
        transition["F_apex_naive"].between(lo, hi, inclusive="both")
    )
    transition["repaired_passes_gate"] = (
        transition["F_apex_repaired"] < float(repaired_floor_max)
    )
    transition["transition_candidate"] = (
        transition["naive_in_transition_band"]
        & transition["repaired_passes_gate"]
    )
    transition = transition.reset_index()
    return grouped.sort_values(
        ["noise_level", "filtration"]
    ).reset_index(drop=True), transition


def replicate_gate_table(
    raw: pd.DataFrame,
    *,
    naive_filtration: str = NAIVE_FILTRATION,
    repaired_filtration: str = REPAIRED_FILTRATION,
    naive_floor_interval: tuple[float, float] = (0.4, 0.8),
    repaired_floor_max: float = 0.3,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-replicate pass proportions for the pre-registered transition gates.

    :func:`summarize_transition_sweep` applies the gates to the *mean* ``F_apex``
    across replicates, which hides how often an individual replicate actually
    satisfies them.  This table applies both gates *within each replicate*, then
    reports the pass count, proportion and an exact Clopper--Pearson interval, so
    a cell that passes on the mean but only in half the replicates cannot be read
    as a stable transition point.

    The registered gates are kept visible and unchanged:
    ``F_apex_alpha in [0.4, 0.8]`` and ``F_apex_repaired < 0.3``.
    """
    if raw.empty:
        return pd.DataFrame()
    required = {"noise_level", "rep", "filtration", "F_apex"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"raw transition frame is missing {sorted(missing)}")

    wide = raw.pivot_table(
        index=["noise_level", "rep"], columns="filtration", values="F_apex"
    )
    for name in (naive_filtration, repaired_filtration):
        if name not in wide.columns:
            raise ValueError(
                f"filtration {name!r} is absent from the sweep; cannot score the "
                "registered gate"
            )
    lo, hi = map(float, naive_floor_interval)
    naive_pass = wide[naive_filtration].between(lo, hi, inclusive="both")
    repaired_pass = wide[repaired_filtration] < float(repaired_floor_max)
    flags = pd.DataFrame(
        {
            "naive_pass": naive_pass,
            "repaired_pass": repaired_pass,
            "joint_pass": naive_pass & repaired_pass,
        }
    )

    rows: list[dict] = []
    for noise, cell in flags.groupby(level="noise_level"):
        n = int(len(cell))
        row: dict[str, float | int | str] = {
            "noise_level": float(noise),
            "n_reps": n,
            "naive_filtration": naive_filtration,
            "repaired_filtration": repaired_filtration,
            "naive_floor_lo": lo,
            "naive_floor_hi": hi,
            "repaired_floor_max": float(repaired_floor_max),
        }
        for column in ("naive_pass", "repaired_pass", "joint_pass"):
            k = int(cell[column].sum())
            cp_lo, cp_hi = clopper_pearson(k, n, alpha=alpha)
            stem = column.replace("_pass", "")
            row[f"{stem}_n_pass"] = k
            row[f"{stem}_prop"] = float(k / n) if n else float("nan")
            row[f"{stem}_cp_lower"] = cp_lo
            row[f"{stem}_cp_upper"] = cp_hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values("noise_level").reset_index(drop=True)
