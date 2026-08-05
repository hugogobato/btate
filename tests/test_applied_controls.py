"""Tests for the Task-6.5.5 control machinery (``btate.applied.controls``)."""

import numpy as np
import pytest

from btate.applied.controls import (
    CORRECTED_ARMS,
    aggregate_null_thinning,
    aggregate_zero,
    draw_null_assignments,
    estimate_curves,
    negative_control,
    run_arms,
)
from btate.applied.evaluate import RealEvaluationConfig, fit_all_bridges
from btate.applied.paired_data import RETENTION_LADDER, UnitCurves


class _Grid:
    """Minimal stand-in for the frozen grid object the arms consume."""

    def __init__(self, resolution=12, upper=1.0):
        self.resolution = resolution
        self.grid = np.linspace(0.0, upper, resolution)
        self.sample_range = (0.0, upper)
        self.dtm_sample_range = (0.0, upper)


def _curves(n_units=24, resolution=12, seed=0):
    """Synthetic unit curves with a depth-dependent distortion."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, resolution)
    out = []
    for i in range(n_units):
        clean = np.exp(-((t - 0.4) ** 2) / 0.02) * (1.0 + 0.1 * rng.standard_normal())
        thinned, dtm = {}, {}
        for p in RETENTION_LADDER:
            # p = 1.0 is exactly the clean curve: thinning at full retention is
            # the identity, which is what the null-thinning control checks.
            bias = 0.0 if p == 1.0 else (1.0 - p) * 0.2
            noise = 0.0 if p == 1.0 else (1.0 - p) * 0.05
            thinned[float(p)] = clean * (1 - bias) + noise * rng.standard_normal(resolution)
            for k in (5, 10, 15, 25):
                dtm[(float(p), k)] = thinned[float(p)] * 0.9
        out.append(UnitCurves(
            unit_id=f"plate{i % 6}:well{i}", n_cells_recovered=100,
            full_alpha=clean, thinned_alpha=thinned, thinned_dtm=dtm,
            cardinality_drop={float(p): 0 for p in RETENTION_LADDER},
        ))
    return out


def test_null_assignment_is_plate_disjoint_and_correctly_sized():
    plates = np.array([f"plate{i % 6}" for i in range(24)])
    positions = np.arange(24)
    draws = draw_null_assignments(RealEvaluationConfig(), plates, positions,
                                  n_draws=50, n_pseudo_treated=3, seed=7)
    assert len(draws) == 50
    for A in draws:
        assert A.shape == (24,)
        assert A.sum() == 3
        # the three pseudo-treated wells sit on three distinct plates
        assert len(set(plates[A.astype(bool)].tolist())) == 3
    # seeded and reproducible
    again = draw_null_assignments(RealEvaluationConfig(), plates, positions,
                                  n_draws=50, n_pseudo_treated=3, seed=7)
    for a, b in zip(draws, again):
        np.testing.assert_array_equal(a, b)


def test_null_assignment_rejects_too_few_plates():
    plates = np.array(["plateA"] * 10)
    with pytest.raises(ValueError):
        draw_null_assignments(RealEvaluationConfig(), plates, np.arange(10),
                              n_draws=1, n_pseudo_treated=3)


def test_negative_control_runs_all_arms_and_scores_the_zero_function():
    grid = _Grid()
    curves = _curves(n_units=24, resolution=grid.resolution)
    cfg = RealEvaluationConfig(n_rep=12, n_replicates=3, n_boot=64,
                               n_measurement_draws=3, n_causal_draws=5,
                               n_plugin_draws=20, pool_treated=3,
                               pool_units=24)
    bridges = fit_all_bridges(cfg, grid, curves)
    plates = np.array([c.unit_id.split(":")[0] for c in curves])
    X = np.column_stack([np.ones(24), np.arange(24, dtype=float) / 24.0])

    frame = negative_control(cfg, grid, eval_curves=curves,
                             control_positions=np.arange(24), X_control=X,
                             plates=plates, bridges=bridges, n_draws=3,
                             n_pseudo_treated=3)
    assert not frame.empty
    ok = frame[~frame["failed"].astype(bool)]
    assert set(ok["method"]) >= {"M1_oracle_full_aipw", "M2_blind_aipw",
                                 "M4_bridge_bayes", "M5_bridge_freq_mi"}
    assert set(ok["p"]) == {float(p) for p in RETENTION_LADDER}
    # cov_zero is an indicator; the interval score against zero is finite
    assert set(np.unique(ok["cov_zero"])) <= {0.0, 1.0}
    assert np.all(np.isfinite(ok["interval_score_zero"]))

    agg = aggregate_zero(frame)
    assert {"cov_zero_rate", "cov_zero_cp_lower", "cov_zero_cp_upper"} <= set(agg.columns)
    assert np.all((agg["cov_zero_rate"] >= 0) & (agg["cov_zero_rate"] <= 1))


def test_estimate_curves_expose_the_null_thinning_shift():
    grid = _Grid()
    curves = _curves(n_units=16, resolution=grid.resolution)
    cfg = RealEvaluationConfig(n_rep=10, n_replicates=2, n_boot=64,
                               n_measurement_draws=3, n_causal_draws=5,
                               n_plugin_draws=20, pool_treated=3,
                               pool_units=16)
    bridges = fit_all_bridges(cfg, grid, curves)
    A = np.zeros(16, dtype=int)
    A[[1, 5, 9]] = 1
    X = np.column_stack([np.ones(16), np.arange(16, dtype=float) / 16.0])
    psi_full = np.zeros(grid.resolution)
    replicates = [np.arange(16), np.arange(16)[::-1]]

    rows, curve_stack = estimate_curves(cfg, grid, eval_curves=curves, A=A, X=X,
                                        psi_full=psi_full, bridges=bridges,
                                        replicates=replicates)
    assert len(rows) == curve_stack.shape[0]
    assert curve_stack.shape[1] == grid.resolution
    # every corrected arm reports a shift away from blind on the same units
    for method, _key, _dtm in CORRECTED_ARMS:
        cell = rows[rows["method"] == method]
        assert not cell.empty
        assert np.all(np.isfinite(cell["sup_shift_vs_blind"]))

    agg = aggregate_null_thinning(rows)
    assert "sup_shift_vs_blind_mean" in agg.columns
    # the blind arm is trivially zero distance from itself
    blind = agg[agg["method"] == "M2_blind_aipw"]
    np.testing.assert_allclose(blind["sup_shift_vs_blind_mean"], 0.0, atol=1e-12)


def test_run_arms_returns_bands_for_every_registered_arm():
    grid = _Grid()
    curves = _curves(n_units=12, resolution=grid.resolution)
    cfg = RealEvaluationConfig(n_rep=12, n_replicates=1, n_boot=32,
                               n_measurement_draws=2, n_causal_draws=4,
                               n_plugin_draws=16, pool_treated=3,
                               pool_units=12)
    bridges = fit_all_bridges(cfg, grid, curves)
    A = np.zeros(12, dtype=int)
    A[[0, 4, 8]] = 1
    X = np.column_stack([np.ones(12), np.arange(12, dtype=float) / 12.0])

    bands = run_arms(cfg, grid, rep=0, idx=np.arange(12), eval_curves=curves,
                     A_rep=A, X_rep=X, bridges=bridges, p=0.25)
    expected = {"M1_oracle_full_aipw", "M2_blind_aipw", "M3_bayes_causal_only",
                "M4_bridge_bayes", "M5_bridge_freq_mi",
                "M4_dtm_bridge_bayes", "M5_dtm_bridge_freq_mi"}
    assert expected <= set(bands)
    for method in expected:
        band = bands[method]
        assert band is not None, bands.get(f"__error__{method}")
        assert band.estimate.shape == (grid.resolution,)
        assert np.all(band.lower <= band.upper)
