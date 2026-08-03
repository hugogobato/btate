"""Frozen representation pipeline for the Phase-6.5 applied study (Task 6.5.3).

Everything that maps raw counts to ``phi`` curves lives here, in one file, so
that the local runs and the Colab shards provably share it.  The source hash
(:data:`REPRESENTATION_HASH`) is recorded in every artifact and must be
verified before any run consumes curves computed elsewhere.

The registered design (``docs/phase6_5_registration.md``):

* per-cell normalisation by the cell's **own** total counts (median total
  frozen at frame-fit time) followed by ``log1p``; a thinned profile is never
  normalised by a full-depth size factor (no-leak rule, asserted in
  :func:`project_unit`);
* PCA to ``n_pca = 50`` components, fit **once** on a capped, seeded subsample
  of calibration full-depth cells (``pca_calibration_cells = 50_000``) via
  streaming (Incremental) PCA so the fit never materialises the full matrix;
  every other cloud is projected through the stored loadings and frozen column
  mean, and frame identity is asserted in a test;
* the alpha filtration (primary) and the DTM-Rips filtration (bridge feature)
  are computed on the first ``d_alpha = 3`` principal components
  (``docs/phase6_5_registration.md``, deviation D2);
* every unit is subsampled to a fixed ``n_points = 60`` cells;
* the filtration grid is frozen by the registered rule applied to
  **calibration full-depth clouds only**: ``upper = round(1.5 * max finite
  H1 death, 3)`` at ``resolution = 96``, mirroring the Phase-6 rule.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..benchmarks.measurement_error_uq import (
    GridSpec,
    alpha_diagram,
    dtm_diagram,
    silhouette_on_grid,
)

_MODULE_FILE = Path(__file__).resolve()

#: sha256 of this module's source text, the frozen-representation fingerprint.
REPRESENTATION_HASH: str = ""

_BATCH_CELLS = 1_000


def _compute_hash() -> str:
    try:
        text = _MODULE_FILE.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - environment-dependent
        return "unavailable"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


REPRESENTATION_HASH = _compute_hash()


@dataclass(frozen=True)
class FrameConfig:
    """Registered representation choices (Section 4 of the registration)."""

    n_pca: int = 50
    d_alpha: int = 3
    n_points: int = 60
    resolution: int = 96
    r: float = 3.0
    pca_calibration_cells: int = 50_000
    frame_seed: int = 65_001
    dtm_grid_max_clouds: int = 80

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class FrozenFrame:
    """The frozen embedding frame: loadings, frozen column mean and median."""

    loadings: np.ndarray          # (n_genes, n_pca)
    column_mean: np.ndarray       # (n_genes,) mean of log1p(normalised) fit cells
    median_total: float           # frozen median library size (fit cells)
    explained_variance: np.ndarray  # (n_pca,)
    n_fit_cells: int
    config: FrameConfig
    hash: str = field(default_factory=lambda: REPRESENTATION_HASH)

    def project(self, counts) -> np.ndarray:
        """Project a cell x gene count matrix through the frozen frame.

        ``counts`` may be sparse (csr) or dense.  Each cell is normalised by
        its **own** row total with the frozen median; nothing from the frame's
        fit data is used except the frozen median, column mean and loadings.
        Returns ``(n_cells, n_pca)`` float64.
        """
        x = counts_to_normalised(counts, self.median_total)
        if hasattr(x, "toarray"):
            dense = np.asarray(x.toarray(), dtype=np.float64)
        else:
            dense = np.asarray(x, dtype=np.float64)
        return (dense - self.column_mean) @ self.loadings

    def unit_cloud(self, counts) -> np.ndarray:
        """The registered filtration cloud: ``(n_points, d_alpha)`` on PC1..d."""
        proj = self.project(counts)
        return proj[:, : self.config.d_alpha]

    def to_dict(self) -> dict:
        return {
            "n_pca": int(self.loadings.shape[1]),
            "n_genes": int(self.loadings.shape[0]),
            "d_alpha": int(self.config.d_alpha),
            "median_total": float(self.median_total),
            "n_fit_cells": int(self.n_fit_cells),
            "explained_variance": self.explained_variance.tolist(),
            "hash": self.hash,
            "config": self.config.to_dict(),
        }


def counts_to_normalised(counts, median_total: float):
    """Size-factor normalise a cell x gene count matrix by its own row totals.

    ``x_ij / (total_i / median_total)`` then ``log1p``, keeping sparsity
    (nonzero entries stay nonzero).  The median is frozen at frame-fit time and
    passed in, so thinned profiles are normalised by **their own** totals only.
    """
    import scipy.sparse as sp

    csr = sp.csr_matrix(counts, dtype=np.float64)
    totals = np.asarray(csr.sum(axis=1)).ravel()
    safe = np.where(totals > 0, totals, 1.0)
    factor = (median_total / safe).astype(np.float64)
    scaled = csr.multiply(factor[:, None])
    data = scaled.data
    if data.size:
        scaled.data = np.log1p(data)
    return scaled


def _fit_cells(counts, cell_indices: np.ndarray, median_total: float) -> np.ndarray:
    """Log1p-normalised rows of ``counts`` at ``cell_indices`` (dense batch)."""
    block = counts[cell_indices]
    block = counts_to_normalised(block, median_total)
    return np.asarray(block.toarray(), dtype=np.float64)


def fit_frame(counts, cell_indices: np.ndarray | None = None,
              config: FrameConfig | None = None) -> FrozenFrame:
    """Fit the frozen PCA frame on a capped subsample of calibration cells.

    ``counts`` is the calibration full-depth cell x gene matrix (sparse); when
    ``cell_indices`` is None a seeded subsample of at most
    ``pca_calibration_cells`` rows is drawn deterministically.  Uses
    streaming (Incremental) PCA over dense batches of ``_BATCH_CELLS`` rows so
    the fit never materialises the full matrix (RAM-bounded, < 5 GB).  The
    median library size is frozen **once** over the fit cells and every batch
    (and later every projection) is normalised against it.
    """
    config = config or FrameConfig()
    n = counts.shape[0]
    rng = np.random.default_rng(np.random.SeedSequence([config.frame_seed, 1]))
    if cell_indices is None:
        cap = int(min(config.pca_calibration_cells, n))
        cell_indices = rng.choice(n, size=cap, replace=False)
    cell_indices = np.asarray(cell_indices, dtype=int)

    totals = np.asarray(counts[cell_indices].sum(axis=1)).ravel()
    positive = totals[totals > 0]
    median_total = float(np.median(positive)) if positive.size else 1.0

    from sklearn.decomposition import IncrementalPCA

    ipca = IncrementalPCA(n_components=config.n_pca, batch_size=_BATCH_CELLS)
    order = rng.permutation(cell_indices.size)
    for start in range(0, cell_indices.size, _BATCH_CELLS):
        idx = cell_indices[order[start:start + _BATCH_CELLS]]
        batch = _fit_cells(counts, idx, median_total)
        ipca.partial_fit(batch)

    return FrozenFrame(
        loadings=np.asarray(ipca.components_, dtype=np.float64).T,
        column_mean=np.asarray(ipca.mean_, dtype=np.float64),
        median_total=median_total,
        explained_variance=np.asarray(ipca.explained_variance_, dtype=np.float64),
        n_fit_cells=int(cell_indices.size),
        config=config,
    )


def freeze_grid(frame: FrozenFrame, calibration_full_clouds,
                config: FrameConfig | None = None,
                dtm_k: int = 15) -> GridSpec:
    """Freeze the filtration grids by the registered rule (calibration only).

    ``calibration_full_clouds`` is a sequence of ``(n_points, d_alpha)``
    clouds of calibration full-depth units.  The rule is
    ``upper = round(1.5 * max finite H1 death, 3)``; the DTM feature grid uses
    the same rule on the DTM H1 deaths of a capped sample of
    ``dtm_grid_max_clouds`` clouds at ``dtm_k``.
    """
    config = config or FrameConfig()
    clouds = [np.asarray(c, dtype=float) for c in calibration_full_clouds]
    deaths: list[float] = []
    for cloud in clouds:
        dgm = alpha_diagram(cloud)
        if dgm.size:
            vals = dgm[:, 1]
            deaths.extend(vals[np.isfinite(vals)].tolist())
    if not deaths:
        raise RuntimeError("calibration full-depth sample has no finite H1 deaths")
    upper = float(np.round(1.5 * max(deaths), 3))

    dtm_deaths: list[float] = []
    n_seen = 0
    rng = np.random.default_rng(np.random.SeedSequence([config.frame_seed, 2]))
    for cloud in clouds:
        if n_seen >= config.dtm_grid_max_clouds:
            break
        n_seen += 1
        dgm = dtm_diagram(cloud, k=int(dtm_k))
        if dgm.size:
            vals = dgm[:, 1]
            dtm_deaths.extend(vals[np.isfinite(vals)].tolist())
    if not dtm_deaths:
        raise RuntimeError("no finite DTM H1 deaths in the capped calibration sample")
    dtm_upper = float(np.round(1.5 * max(dtm_deaths), 3))

    grid = np.linspace(0.0, upper, int(config.resolution))
    return GridSpec(
        sample_range=(0.0, upper), resolution=int(config.resolution),
        r=float(config.r), grid=grid,
        dtm_sample_range=(0.0, dtm_upper), dtm_resolution=int(config.resolution),
        provenance={
            "rule": "round(1.5 * max finite calibration full-depth H1 death, 3)",
            "n_clouds": int(len(clouds)),
            "max_clean_death": float(max(deaths)),
            "dtm_rule": "same rule, capped sample, k=%d" % int(dtm_k),
            "dtm_n_clouds": int(n_seen),
            "dtm_max_clean_death": float(max(dtm_deaths)),
            "hash": REPRESENTATION_HASH,
        },
    )


def alpha_curve_on_grid(cloud, grid: GridSpec) -> np.ndarray:
    """Silhouette (r = 3) of one cloud's alpha H1 diagram on the frozen grid."""
    return silhouette_on_grid(alpha_diagram(np.asarray(cloud, dtype=float)),
                              grid.sample_range, grid.resolution, grid.r)


def dtm_curve_on_grid(cloud, grid: GridSpec, k: int) -> np.ndarray:
    """Silhouette (r = 3) of one cloud's DTM-Rips H1 diagram (feature only)."""
    return silhouette_on_grid(dtm_diagram(np.asarray(cloud, dtype=float), k=int(k)),
                              grid.dtm_sample_range, grid.dtm_resolution, grid.r)
