"""Evaluation & benchmarking utilities (Phase 4).

Two layers:

* **metrics** — dependency-light, pure-numpy scalar metrics (numerical
  integration, L1, RMSE, bias, credible-band coverage / width).
* **harness** — the config-driven simulation machinery (Task 4.1 / 4.4): a
  ground-truth loop-injection DGP (:mod:`.synthetic`), the end-to-end Bayesian
  pipeline engine (:mod:`.pipeline`), a self-contained frequentist AIPW for the
  head-to-head (:mod:`.frequentist`), the sweep runner (:mod:`.harness`), and
  ablation-grid builders (:mod:`.ablation`).

The metric helpers import only numpy; the harness/pipeline import ``gudhi`` and
(only in ``maroulas`` mode) ``bayes_tda`` lazily, so ``import
btate.benchmarks`` always succeeds in a minimal environment.
"""
from __future__ import annotations

from .metrics import (
    apex_floor,
    apex_location,
    apex_shift,
    bias,
    death_recovery,
    integrated_bias,
    interval_width,
    l1_distance,
    max_abs_error,
    numerical_integration,
    pointwise_coverage,
    rmse,
    simultaneous_coverage,
)

# Harness/pipeline modules are imported lazily (PEP 562) to avoid importing gudhi
# at package-import time.  They can also be imported directly, e.g.::
#
#     from btate.benchmarks.harness import SweepCell, run_sweep
#     from btate.benchmarks.synthetic import SyntheticConfig
#
__all__ = [
    "numerical_integration",
    "l1_distance",
    "rmse",
    "bias",
    "integrated_bias",
    "max_abs_error",
    "pointwise_coverage",
    "simultaneous_coverage",
    "interval_width",
    "apex_location",
    "apex_shift",
    "apex_floor",
    "death_recovery",
]


def __getattr__(name):  # PEP 562 lazy access to the heavier submodules
    import importlib

    submodules = {
        "SyntheticConfig": "synthetic", "SyntheticDataset": "synthetic",
        "generate_synthetic_dataset": "synthetic", "reference_effect": "synthetic",
        "montecarlo_reference": "synthetic", "standard_config": "synthetic",
        "low_snr_config": "synthetic",
        "clopper_pearson": "metrics",
        "DecisionCell": "decision", "FGPVariant": "decision",
        "run_decision_cell": "decision", "run_decision_grid": "decision",
        "aggregate_decision": "decision", "evaluate_decision_rep": "decision",
        "PipelineConfig": "pipeline", "run_bayesian_pipeline": "pipeline",
        "SweepCell": "harness", "run_sweep": "harness", "run_cell": "harness",
        "evaluate_run": "harness", "sweep_to_rows": "harness",
        "aipw_effect": "frequentist", "FrequentistEffect": "frequentist",
        "full_ablation_grid": "ablation",
        "pre_fgp_maroulas_diagnostic": "maroulas_diagnostics",
        "maroulas_sigma_sensitivity": "maroulas_diagnostics",
        "strip_diagnostic_arrays": "maroulas_diagnostics",
        "JointCalibrationCell": "joint_calibration",
        "run_joint_calibration": "joint_calibration",
        "aggregate_joint_records": "joint_calibration",
        "h1_diagram_dtm": "dtm", "h1_diagram_filtration": "dtm",
        "top_feature_death": "dtm", "dtm_death_sweep": "dtm",
    }
    if name in submodules:
        mod = importlib.import_module(f"{__name__}.{submodules[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
