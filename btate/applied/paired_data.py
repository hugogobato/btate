"""Paired calibration resource by binomial thinning (Phase-6.5, Task 6.5.2).

Builds the registered calibration / evaluation split at the ``(compound,
plate)`` level, thins raw counts ``U' ~ Binom(U, p)`` with a fixed per-cell
seed, computes full-depth and thinned representation curves through the frozen
frame, and reports the registered corruption-geometry diagnostics.

See ``docs/phase6_5_registration.md`` Sections 2, 6 for the registered
definitions (threshold 60, fixed n = 60, no-leak normalisation, per-cell
seeds, the pair-disjoint split rule, and the diagnostics formulas).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .representation import (
    FrozenFrame,
    alpha_curve_on_grid,
    dtm_curve_on_grid,
)

# --------------------------------------------------------------------------- #
# Registered constants (docs/phase6_5_registration.md)
# --------------------------------------------------------------------------- #
RETENTION_LADDER: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
PRIMARY_P: float = 0.25
DTM_K_LADDER: tuple[int, ...] = (5, 10, 15, 25)
THRESHOLD_CELLS: int = 60
SUBSAMPLE_SEED: int = 65_002
THINNING_SALT: int = 0xB71E5A11

#: Pinned evaluation units (registration Section 2), keyed by (plate, well).
#: The ``well`` column is a site code that repeats across plates (52 plates x
#: 96 positions = 4992 units in total), so the unit identity is the pair.
EVAL_TREATED_PAIRS: tuple[tuple[str, str], ...] = (
    ("plate34", "plate4_E9"),
    ("plate49", "plate2_E3"),
    ("plate52", "plate10_E9"),
)
EVAL_CONTROL_PAIRS: tuple[tuple[str, str], ...] = (
    ("plate33", "plate3_C12"), ("plate33", "plate3_C6"),
    ("plate34", "plate4_C12"), ("plate34", "plate4_C6"),
    ("plate35", "plate5_G11"), ("plate35", "plate5_G5"),
    ("plate36", "plate6_G11"), ("plate36", "plate6_G5"),
    ("plate37", "plate7_D12"), ("plate37", "plate7_D6"),
    ("plate38", "plate8_D12"), ("plate38", "plate8_D6"),
    ("plate39", "plate9_D10"), ("plate39", "plate9_D4"),
    ("plate40", "plate10_D10"), ("plate40", "plate10_D4"),
    ("plate49", "plate2_C12"), ("plate49", "plate2_C6"),
    ("plate50", "plate8_C12"), ("plate50", "plate8_C6"),
    ("plate51", "plate9_C12"), ("plate51", "plate9_C6"),
    ("plate52", "plate10_C12"), ("plate52", "plate10_C6"),
)
EVAL_ALISERTIB_PLATES: tuple[str, ...] = ("plate9", "plate34", "plate49", "plate52")
EVAL_CONTROL_PLATES: tuple[str, ...] = (
    "plate33", "plate34", "plate35", "plate36", "plate37", "plate38",
    "plate39", "plate40", "plate49", "plate50", "plate51", "plate52",
)
EVAL_CONTROL_PERTURBATION: str = "control"
EVAL_PRIMARY_COMPOUND: str = "Alisertib (MLN8237)"


def cell_thinning_seed(barcode: str, p: float) -> int:
    """Deterministic 32-bit per-cell thinning seed (BLAS-independent PRF).

    ``blake2b(cell_barcode | p)`` truncated to 32 bits; the barcode list is
    stored in the artifact so the map is reproducible for any cell.
    """
    h = hashlib.blake2b(
        digest_size=8,
        key=str(SUBSAMPLE_SEED).encode("ascii"),
    )
    h.update(barcode.encode("utf-8"))
    h.update(b"\x00")
    h.update(f"{p:g}".encode("ascii"))
    return int.from_bytes(h.digest()[:4], "little")


def thin_cell_row(nonzero_values, barcode: str, p: float):
    """One cell's ``U' ~ Binom(U, p)`` with its fixed per-cell seed."""
    rng = np.random.default_rng(cell_thinning_seed(barcode, p))
    v = np.asarray(nonzero_values, dtype=np.int64)
    return rng.binomial(v, p) if v.size else v


@dataclass(frozen=True)
class UnitCurves:
    """All curves of one unit, computed once and reused by every replicate.

    ``full_alpha`` is the full-depth silhouette; ``thinned_alpha[p]`` is the
    thinned silhouette at retention ``p``; ``thinned_dtm[(p, k)]`` is the
    DTM feature on the observed (thinned) side.  ``cardinality_drop[p]`` is
    the number of cells that died (zero total) under thinning at ``p``.
    """

    unit_id: str
    n_cells_recovered: int
    full_alpha: np.ndarray
    thinned_alpha: dict[float, np.ndarray]
    thinned_dtm: dict[tuple[float, int], np.ndarray]
    cardinality_drop: dict[float, int] = field(default_factory=dict)

    def curves_for(self, p: float, use_dtm: bool = False,
                   dtm_k: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        alpha = self.thinned_alpha[p]
        if not use_dtm:
            return alpha, None
        return alpha, self.thinned_dtm[(p, int(dtm_k))]


def _unit_table(adata) -> "pd.DataFrame":
    import pandas as pd

    obs = adata.obs
    obs = obs[obs["cell_line"].notna() & obs["well"].notna()].copy()
    if "plate" not in obs:
        # sci-Plex 2 has no plate column.  Its pilot is mechanics-only, so all
        # wells are placed on a synthetic single plate and never enter the
        # registered sci-Plex 3 evaluation split.
        obs["plate"] = "__single_plate__"
    if "time" not in obs:
        obs["time"] = 24.0
    agg = obs.groupby(
        ["well", "plate", "cell_line", "perturbation"], observed=True
    ).agg(n_cells=("ncounts", "size"), time=("time", "first"))
    return agg.reset_index()


def build_split(adata, *, threshold: int = THRESHOLD_CELLS) -> dict:
    """Build the registered calibration / evaluation split.

    Returns a dict with ``units`` (DataFrame of well, plate, cell_line,
    perturbation, n_cells), boolean arrays ``evaluation``, ``calibration``,
    ``treated``, ``control`` aligned to ``units`` rows, and ``protocol``.

    Rules (registration Sections 2, 6): evaluation = the pinned treated and
    control wells with ``n_cells >= threshold``; calibration = every other
    unit with ``n_cells >= threshold`` whose ``(compound, plate)`` is not an
    evaluation pair -- ``(Alisertib, EVAL_ALISERTIB_PLATES)`` or
    ``(control, EVAL_CONTROL_PLATES)``.  Units matching an evaluation pair are
    in **neither** set.  No evaluation unit ever enters the bridge.
    """
    t = _unit_table(adata)
    t = t[t["n_cells"] >= threshold].reset_index(drop=True)

    # The ``well`` string is a site code that repeats across plates, so the
    # unit identity is the (plate, well) pair.
    treated_pairs_set = set(EVAL_TREATED_PAIRS)
    control_pairs_set = set(EVAL_CONTROL_PAIRS)
    pinned = treated_pairs_set | control_pairs_set
    eval_comp_plate = {
        (EVAL_PRIMARY_COMPOUND, pl) for pl in EVAL_ALISERTIB_PLATES
    } | {(EVAL_CONTROL_PERTURBATION, pl) for pl in EVAL_CONTROL_PLATES}

    pairs = list(zip(t["plate"], t["well"]))
    is_pinned = np.array([k in pinned for k in pairs])
    is_pair = np.array([(p, pl) in eval_comp_plate for p, pl in zip(t["perturbation"], t["plate"])])
    is_treated = np.array([k in treated_pairs_set for k in pairs])
    is_control = np.array([k in control_pairs_set for k in pairs])

    # A unit whose (compound, plate) matches an evaluation pair is excluded
    # from BOTH sets unless it is a pinned evaluation well (registration 6).
    is_eval = is_pinned
    is_pair_excluded = is_pair & ~is_pinned

    # Sanity: pinned wells exist above the threshold and never overlap.
    assert np.all(is_eval[is_treated | is_control]), "pinned eval well below threshold"
    assert not np.any(is_treated & is_control)
    assert np.all(is_pinned[is_treated | is_control])

    n_eval = int(is_eval.sum())
    n_treated = int(is_treated.sum())
    n_control = int(is_control.sum())
    n_calib = int((~is_eval & ~is_pair_excluded).sum())
    return {
        "units": t,
        "evaluation": is_eval,
        "calibration": ~is_eval & ~is_pair_excluded,
        "excluded": is_pair_excluded,
        "treated": is_treated,
        "control": is_control,
        "protocol": {
            "threshold": int(threshold),
            "n_eval_units": n_eval,
            "n_treated_units": n_treated,
            "n_control_units": n_control,
            "n_calibration_units": n_calib,
            "retention_ladder": list(RETENTION_LADDER),
            "primary_p": PRIMARY_P,
            "dtm_k_ladder": list(DTM_K_LADDER),
        },
    }


def thin_csr_rows(csr, barcodes, p: float):
    """Binomial-thin every row of a csr matrix with the per-cell fixed seeds."""
    import scipy.sparse as sp

    out = csr.copy()
    data = out.data
    indptr = out.indptr
    if data.size:
        new_data = np.empty(data.size, dtype=np.int64)
        for i in range(csr.shape[0]):
            start, stop = indptr[i], indptr[i + 1]
            new_data[start:stop] = thin_cell_row(data[start:stop], barcodes[i], p)
        out.data = new_data
        out.eliminate_zeros()
    return out


def unit_full_cloud(adata, row, frame: FrozenFrame, *, n_points: int = 60,
                    subsample_seed: int = SUBSAMPLE_SEED) -> tuple[np.ndarray, list, np.ndarray]:
    """The seeded ``n_points``-cell full-depth cloud of one unit.

    Shared by :func:`freeze_grid` and :func:`unit_curves` so both consume the
    identical (plate, well)-keyed, deterministically subsampled cloud.
    Returns ``(cloud, barcodes, counts)`` where ``counts`` is the sparse
    (n_points, n_genes) full-depth slice.
    """
    import scipy.sparse as sp

    well = str(row["well"])
    plate = str(row["plate"])
    obs = adata.obs
    cell_mask = obs["well"].astype(str).to_numpy() == well
    if "plate" in obs:
        cell_mask &= obs["plate"].astype(str).to_numpy() == plate
    barcodes = list(obs.index[cell_mask])
    counts = adata[cell_mask, :].X
    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)
    counts = counts.tocsr()

    totals = np.asarray(counts.sum(axis=1)).ravel()
    alive = np.flatnonzero(totals > 0)
    unit_key = f"{plate}:{well}"
    rng = np.random.default_rng(
        np.random.SeedSequence([int(subsample_seed),
                                int(hashlib.blake2b(unit_key.encode(),
                                                    digest_size=4)
                                    .hexdigest(), 16)]))
    keep = rng.choice(alive.size, size=int(n_points), replace=False) \
        if alive.size > int(n_points) else np.arange(alive.size)
    keep = alive[keep]
    if keep.size < int(n_points):
        raise ValueError(
            f"unit {unit_key} has only {keep.size} alive cells < n_points={n_points}")

    cloud = frame.unit_cloud(counts[keep])
    barcodes = [barcodes[i] for i in keep]
    return cloud, barcodes, counts[keep]


def unit_curves(adata, row, frame: FrozenFrame, grid, *,
                p_ladder=RETENTION_LADDER,
                dtm_k_ladder=DTM_K_LADDER,
                n_points: int = 60,
                subsample_seed: int = SUBSAMPLE_SEED) -> UnitCurves:
    """Compute the full-depth and thinned curves of one unit.

    ``row`` is a (well, plate, cell_line, perturbation, n_cells) record.  The
    ``n_points``-cell subsample is drawn once (seeded) from the full-depth
    profile; the same cells are thinned at every ``p`` (paired design), cells
    that die (zero total counts) under thinning are dropped and the drop is
    recorded.  All normalisation is per-cell by own totals (no-leak rule).
    """
    well = str(row["well"])
    plate = str(row["plate"])
    unit_key = f"{plate}:{well}"
    full_cloud, full_barcodes, full_counts = unit_full_cloud(
        adata, row, frame, n_points=n_points, subsample_seed=subsample_seed)
    full_alpha = alpha_curve_on_grid(full_cloud, grid)

    thinned_alpha: dict[float, np.ndarray] = {}
    thinned_dtm: dict[tuple[float, int], np.ndarray] = {}
    cardinality_drop: dict[float, int] = {}
    for p in p_ladder:
        thin = thin_csr_rows(full_counts, full_barcodes, float(p))
        t_totals = np.asarray(thin.sum(axis=1)).ravel()
        alive_t = np.flatnonzero(t_totals > 0)
        cardinality_drop[float(p)] = int(full_counts.shape[0] - alive_t.size)
        t_cloud = frame.unit_cloud(thin[alive_t])
        thinned_alpha[float(p)] = alpha_curve_on_grid(t_cloud, grid)
        for k in dtm_k_ladder:
            thinned_dtm[(float(p), int(k))] = dtm_curve_on_grid(t_cloud, grid, int(k))

    return UnitCurves(
        unit_id=unit_key,
        n_cells_recovered=int(row["n_cells"]),
        full_alpha=full_alpha,
        thinned_alpha=thinned_alpha,
        thinned_dtm=thinned_dtm,
        cardinality_drop=cardinality_drop,
    )


def corruption_geometry_diagnostics(full_cloud, thinned_cloud,
                                    full_diagram, thinned_diagram,
                                    grid_upper: float) -> dict:
    """Registered corruption-geometry diagnostics (registration Section 6).

    1. effective cardinality change (cells alive in each cloud),
    2. Hausdorff distance and mean per-coordinate Wasserstein-1 (PC1-3),
    3. H1 feature-count ratio and the fraction of thinned features matched
       within ``eps = max(0.05 * grid_upper, Hausdorff)`` to a full feature,
    4. bottleneck distance vs the registered stability bound ``2 * d_H``.
    """
    from scipy.spatial.distance import directed_hausdorff

    full = np.asarray(full_cloud, dtype=float)
    thin = np.asarray(thinned_cloud, dtype=float)
    d_h = max(float(directed_hausdorff(full, thin)[0]),
              float(directed_hausdorff(thin, full)[0]))

    w1 = []
    for c in range(full.shape[1]):
        a = np.sort(full[:, c])
        b = np.sort(thin[:, c])
        w1.append(float(np.mean(np.abs(a - b))))
    w1_mean = float(np.mean(w1)) if w1 else float("nan")

    n_full = full_diagram.shape[0]
    n_thin = thinned_diagram.shape[0]
    count_ratio = float(n_thin / n_full) if n_full else float("nan")

    matched = 0.0
    if n_full and n_thin:
        eps = max(0.05 * float(grid_upper), d_h)
        for f in thinned_diagram:
            d = np.min(np.sqrt(np.sum((full_diagram - f) ** 2, axis=1)))
            matched += float(d <= eps)
    match_frac = matched / n_thin if n_thin else float("nan")

    db = float("nan")
    if n_full and n_thin:
        from gudhi import bottleneck_distance

        db = float(bottleneck_distance(full_diagram, thinned_diagram, 1e-9))

    return {
        "n_full_points": int(full.shape[0]),
        "n_thinned_points": int(thin.shape[0]),
        "cardinality_change": int(thin.shape[0] - full.shape[0]),
        "hausdorff": d_h,
        "wasserstein1_mean_pc": w1_mean,
        "h1_count_full": int(n_full),
        "h1_count_thinned": int(n_thin),
        "h1_count_ratio": count_ratio,
        "h1_thinned_matched_frac": match_frac,
        "bottleneck": db,
        "stability_bound": 2.0 * d_h,
        "bottleneck_violates_bound": bool(db > 2.0 * d_h) if np.isfinite(db) else None,
    }
