"""Regression tests for the registered Phase-6.5 applied pipeline."""

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from btate.applied.paired_data import build_split, cell_thinning_seed
from btate.applied.representation import (
    FrameConfig,
    counts_to_normalised,
    fit_frame,
)


def test_frozen_frame_identity_and_projection_reuse():
    rng = np.random.default_rng(65001)
    counts = sp.csr_matrix(rng.poisson(3, size=(12, 10)))
    frame = fit_frame(
        counts,
        cell_indices=np.arange(counts.shape[0]),
        config=FrameConfig(
            n_pca=3,
            d_alpha=2,
            pca_calibration_cells=12,
            frame_seed=65001,
        ),
    )

    full = counts[2:3]
    thinned = full.copy()
    thinned.data = np.maximum(thinned.data // 2, 1)
    full_projection_1 = frame.project(full)
    full_projection_2 = frame.project(full)
    thinned_projection = frame.project(thinned)

    np.testing.assert_array_equal(full_projection_1, full_projection_2)
    assert full_projection_1.shape == (1, 3)
    assert thinned_projection.shape == (1, 3)
    # Both profiles were projected by the same stored frame, not separate fits.
    assert frame.loadings.shape == (counts.shape[1], 3)
    assert frame.hash


def test_thinned_normalisation_uses_only_thinned_row_totals():
    # The frozen median is 10.  A thinned row with total 1 therefore receives
    # the factor 10, independently of the full-depth row from which it came.
    thinned = sp.csr_matrix([[1, 0]], dtype=np.int64)
    normalised = counts_to_normalised(thinned, median_total=10.0).toarray()
    expected = np.log1p([[10.0, 0.0]])
    np.testing.assert_array_equal(normalised, expected)

    # The per-cell seed is stable and changes with the registered p value.
    assert cell_thinning_seed("cell-1", 0.25) == cell_thinning_seed(
        "cell-1", 0.25)
    assert cell_thinning_seed("cell-1", 0.25) != cell_thinning_seed(
        "cell-1", 0.5)


def test_registered_sciplex3_split_counts_and_pair_keying():
    ad = pytest.importorskip("anndata")
    h5ad = Path(__file__).resolve().parents[3] / "data" / \
        "srivatsan_2020_sciplex3.h5ad"
    if not h5ad.exists():
        pytest.skip("registered sci-Plex 3 data are not present")

    adata = ad.read_h5ad(h5ad, backed="r")
    split = build_split(adata)
    protocol = split["protocol"]
    assert protocol["n_eval_units"] == 27
    assert protocol["n_treated_units"] == 3
    assert protocol["n_control_units"] == 24
    assert protocol["n_calibration_units"] == 4614
    assert int(split["excluded"].sum()) == 3

    units = split["units"]
    keys = list(zip(units["plate"].astype(str), units["well"].astype(str)))
    assert len(keys) == len(set(keys))
    assert not np.any(split["evaluation"] & split["calibration"])
    assert not np.any(split["evaluation"] & split["excluded"])
