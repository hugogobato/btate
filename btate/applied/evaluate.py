"""Seven-arm bridge evaluation on held-out real data (Phase-6.5, Task 6.5.4).

Reproduces the Phase-6 arm table on the sci-Plex evaluation set with the
registered protocol (``docs/phase6_5_registration.md`` Sections 5, 7, 8):
fixed ``psi_full`` target, 100 seeded subsample replicates of ``n_rep = 20``
from the evaluation units, seven arms with simultaneous bands, headline
``cov_sim_full`` at the primary ``p = 0.25`` and at every ladder value, with
exact Clopper-Pearson intervals, interval scores and band widths alongside,
and the required DTM mass-parameter ablation.

The causal layer is identical in every arm (same AIPW estimator, covariate set
and design weights); only the treatment of measurement error varies.  ``M1``
sees the full-depth curves of the replicate's units (oracle arm) and is the
empirical reference level every other arm is read against, never against a
nominal 0.95 (registration Section 7).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..benchmarks.frequentist import aipw_effect, cross_fitted_scores
from ..benchmarks.measurement_error_uq import (
    MeasurementBridge,
    fit_measurement_bridge,
    multiple_imputation_band,
)
from ..causal import FunctionalGPEstimator
from ..causal.propagation import nested_posterior_tate, plugin_posterior_tate
from .paired_data import (
    DTM_K_LADDER,
    PRIMARY_P,
    RETENTION_LADDER,
    UnitCurves,
)

# --------------------------------------------------------------------------- #
# Registered evaluation hyper-parameters (registration Sections 5, 7, 8)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RealEvaluationConfig:
    n_rep: int = 20
    n_replicates: int = 100
    primary_p: float = PRIMARY_P
    retention_ladder: tuple[float, ...] = RETENTION_LADDER
    dtm_k_ladder: tuple[int, ...] = DTM_K_LADDER
    pool_treated: int = 3
    pool_units: int = 27
    cross_fit: bool = False
    n_basis: int = 5
    n_boot: int = 1000
    n_measurement_draws: int = 16
    n_causal_draws: int = 60
    n_plugin_draws: int = 480
    alpha: float = 0.05
    rep_seed: int = 65_006
    measurement_seed: int = 65_007
    causal_seed: int = 65_008
    bootstrap_seed: int = 65_009
    bridge_k_alpha: int = 10
    bridge_k_dtm: int = 6
    bridge_ridge_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    bridge_scale_prior: float = 1e-6
    calibration_holdout_frac: float = 0.25
    calibration_seed: int = 65_003

    @property
    def pi_design(self) -> float:
        return float(self.pool_treated) / float(self.pool_units)


# --------------------------------------------------------------------------- #
# Seed plumbing
# --------------------------------------------------------------------------- #
def _seed(cfg: RealEvaluationConfig, base: int, *parts: int) -> int:
    rng = np.random.default_rng(np.random.SeedSequence([int(base), *map(int, parts)]))
    return int(rng.integers(0, np.iinfo(np.int32).max))


def _rng(cfg: RealEvaluationConfig, base: int, *parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(base), *map(int, parts)]))


def _fgp_estimator(cfg: RealEvaluationConfig) -> FunctionalGPEstimator:
    """Phase-4 defaults; ``n_inducing`` clamps to n inside the FGP."""
    return FunctionalGPEstimator(
        n_inducing=48, prior_scale=5.0, length_scale_t=0.06,
        posterior_scale="godambe",
    )


# --------------------------------------------------------------------------- #
# Bridge fitting (alpha-only and DTM-augmented with the k ablation)
# --------------------------------------------------------------------------- #
class _BridgeCfg:
    """Duck-typed view of the bridge-fitting options for real data."""

    def __init__(self, cfg: RealEvaluationConfig, p: float, k: int | None):
        self.use_dtm_feature = k is not None
        self.dtm_k = int(k) if k is not None else 15
        self.bridge_k_alpha = int(cfg.bridge_k_alpha)
        self.bridge_k_dtm = int(cfg.bridge_k_dtm)
        self.bridge_ridge_grid = tuple(cfg.bridge_ridge_grid)
        self.bridge_scale_prior = float(cfg.bridge_scale_prior)
        self.calibration_holdout_frac = float(cfg.calibration_holdout_frac)
        self.noise_level = float(p)
        self.seeds = type("S", (), {
            "calibration": int(cfg.calibration_seed),
            "truth": 61_000_000, "evaluation": 63_000_000,
        })()
        self.dtm_grid_max_clouds = 80


def _calibration_pack(curves: list, p: float, k: int | None) -> dict:
    """Calibration dict for ``fit_measurement_bridge`` (unit ids as subjects)."""
    pack = {
        "obs_alpha": np.stack([c.thinned_alpha[p] for c in curves]),
        "clean_alpha": np.stack([c.full_alpha for c in curves]),
        "subject_ids": np.arange(len(curves)),
    }
    pack["obs_dtm"] = (
        None if k is None
        else np.stack([c.thinned_dtm[(p, int(k))] for c in curves]))
    return pack


def fit_bridge(cfg: RealEvaluationConfig, grid, calibration: list, p: float,
               k: int | None) -> MeasurementBridge:
    """Fit one bridge: alpha-only when ``k is None``, else DTM-augmented.

    The ridge is chosen internally by held-out predictive log-density on a
    unit-disjoint split of the calibration pool, then refit on the full pool.
    """
    return fit_measurement_bridge(
        _BridgeCfg(cfg, p, k), grid, calibration=_calibration_pack(calibration, p, k))


def select_dtm_k(cfg: RealEvaluationConfig, grid, calibration: list,
                 p: float) -> dict:
    """Registered DTM-mass selection: ``k*`` by held-out predictive log-density.

    For every ``k`` in the ladder the DTM bridge is fitted by the unit-disjoint
    calibration holdout (identical split for every ``k`` at this ``p``);
    ``k*`` maximises the held-out predictive log-density and is frozen for the
    headline DTM arms.  The per-``k`` record is retained for the ablation.
    Returns ``{"dtm": bridge_at_k*, "selected_k": k*, "k_selection": {...}}``.
    """
    record: dict = {}
    bridges_by_k: dict[int, MeasurementBridge] = {}
    best_score = -np.inf
    best = None
    for k in cfg.dtm_k_ladder:
        br = fit_bridge(cfg, grid, calibration, p, int(k))
        bridges_by_k[int(k)] = br
        score = float(br.diagnostics["selected_holdout_logpdf"])
        record[int(k)] = {
            "holdout_logpdf": score,
            "ridge": float(br.ridge),
            "holdout_clean_alpha_rmse": float(
                br.diagnostics["holdout_clean_alpha_rmse"]),
            "holdout_naive_clean_alpha_rmse": float(
                br.diagnostics["holdout_naive_clean_alpha_rmse"]),
        }
        if score > best_score:
            best_score, best = score, br
    selected_k = int(max(record, key=lambda k: record[k]["holdout_logpdf"]))
    return {"dtm": best, "selected_k": selected_k,
            "k_selection": record, "bridges_by_k": bridges_by_k}


def fit_all_bridges(cfg: RealEvaluationConfig, grid,
                    calibration: list) -> dict:
    """Fit the alpha-only and k-selected DTM bridge for every retention ``p``."""
    bridges = {}
    for p in cfg.retention_ladder:
        alpha_br = fit_bridge(cfg, grid, calibration, p, None)
        selection = select_dtm_k(cfg, grid, calibration, p)
        bridges[float(p)] = {
            "alpha": alpha_br,
            "dtm": selection["dtm"],
            "selected_k": selection["selected_k"],
            "k_selection": selection["k_selection"],
            "bridges_by_k": selection["bridges_by_k"],
        }
    return bridges


# --------------------------------------------------------------------------- #
# Target and scoring
# --------------------------------------------------------------------------- #
def psi_full_target(cfg: RealEvaluationConfig, eval_curves: list, A: np.ndarray,
                    X: np.ndarray, grid, seed: int) -> np.ndarray:
    """The fixed full-depth estimand: AIPW on the full-depth curves of the pool."""
    phi = np.stack([u.full_alpha for u in eval_curves])
    eff = aipw_effect(phi, A, X, np.asarray(grid.grid),
                      pi_hat=np.full(phi.shape[0], cfg.pi_design),
                      alpha=cfg.alpha, n_basis=cfg.n_basis,
                      cross_fit=cfg.cross_fit, n_boot=cfg.n_boot,
                      random_state=seed, estimator="aipw")
    return eff.estimate


def interval_score(lower, upper, truth, alpha: float) -> float:
    """Gneiting-Raftery interval score of a band against the fixed target."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    truth = np.asarray(truth, dtype=float)
    width = upper - lower
    below = np.clip(lower - truth, 0.0, None)
    above = np.clip(truth - upper, 0.0, None)
    return float(np.mean(width + (2.0 / alpha) * below + (2.0 / alpha) * above))


# --------------------------------------------------------------------------- #
# Per-arm bands
# --------------------------------------------------------------------------- #
@dataclass
class RepBand:
    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    metadata: dict = field(default_factory=dict)


def _aipw_band(cfg: RealEvaluationConfig, phi, A, X, grid, seed) -> RepBand:
    eff = aipw_effect(phi, A, X, grid, pi_hat=np.full(phi.shape[0], cfg.pi_design),
                      alpha=cfg.alpha, n_basis=cfg.n_basis,
                      cross_fit=cfg.cross_fit, n_boot=cfg.n_boot,
                      random_state=seed, estimator="aipw")
    return RepBand(eff.estimate, eff.simultaneous_lower, eff.simultaneous_upper,
                   {"band": "eif_pointwise + gaussian_multiplier_simultaneous",
                    "uniform_crit": eff.metadata.get("uniform_crit")})


def _plugin_band(cfg: RealEvaluationConfig, phi, A, X, grid, seed) -> RepBand:
    effect = plugin_posterior_tate(
        phi, A, X, grid, pi_hat=np.full(A.shape[0], cfg.pi_design),
        estimator=_fgp_estimator(cfg), n_draws=cfg.n_plugin_draws,
        alpha=cfg.alpha, random_state=seed)
    return RepBand(effect.mean, effect.simultaneous_lower,
                   effect.simultaneous_upper,
                   {"band": "posterior_supnorm_standardized",
                    "simultaneous_radius": float(effect.simultaneous_radius)})


def _bridge_draws(cfg: RealEvaluationConfig, bridge: MeasurementBridge,
                  phi_obs, phi_dtm, rep: int, k) -> np.ndarray:
    return bridge.draw_clean_curves(
        phi_obs, phi_dtm, n_draws=cfg.n_measurement_draws,
        random_state=_seed(cfg, cfg.measurement_seed, rep, k if k is not None else 0))


def _nested_band(cfg: RealEvaluationConfig, bridge: MeasurementBridge,
                 phi_obs, phi_dtm, A, X, grid, rep: int, k) -> RepBand:
    draws = _bridge_draws(cfg, bridge, phi_obs, phi_dtm, rep, k)
    effect = nested_posterior_tate(
        draws, A, X, grid, pi_hat=np.full(A.shape[0], cfg.pi_design),
        estimator=_fgp_estimator(cfg), n_causal_draws=cfg.n_causal_draws,
        alpha=cfg.alpha, random_state=_seed(cfg, cfg.causal_seed, rep, 3))
    return RepBand(effect.mean, effect.simultaneous_lower,
                   effect.simultaneous_upper,
                   {"band": "posterior_supnorm_standardized",
                    "simultaneous_radius": float(effect.simultaneous_radius)})


def _mi_band(cfg: RealEvaluationConfig, bridge: MeasurementBridge,
             phi_obs, phi_dtm, A, X, grid, rep: int, k) -> RepBand:
    draws = _bridge_draws(cfg, bridge, phi_obs, phi_dtm, rep, k)
    psis, influences = [], []
    for s in range(draws.shape[0]):
        cf = cross_fitted_scores(
            draws[s], A, X, grid, pi_hat=np.full(A.shape[0], cfg.pi_design),
            n_basis=cfg.n_basis, cross_fit=cfg.cross_fit,
            random_state=_seed(cfg, cfg.causal_seed, rep, 5, s))
        psis.append(cf["aipw"])
        influences.append(cf["influence"])
    result = multiple_imputation_band(psis, influences, cfg.alpha, cfg.n_boot,
                                      _rng(cfg, cfg.bootstrap_seed, rep, 6))
    return RepBand(result.estimate, result.simultaneous_lower,
                   result.simultaneous_upper,
                   {"band": result.metadata["band"],
                    "uniform_crit": result.metadata.get("uniform_crit")})


# --------------------------------------------------------------------------- #
# Replicate driver
# --------------------------------------------------------------------------- #
def _cov_row(cfg: RealEvaluationConfig, band: RepBand, psi_full, *, p, rep,
             method: str) -> dict:
    return {
        "p": float(p), "rep": int(rep), "method": method,
        "cov_sim_full": float(np.all((psi_full >= band.lower) &
                                      (psi_full <= band.upper))),
        "interval_score": interval_score(band.lower, band.upper, psi_full,
                                         cfg.alpha),
        "band_width": float(np.mean(band.upper - band.lower)),
        "failed": False, "error": "",
        "band_kind": str(band.metadata.get("band", "")),
    }


def evaluate_replicate(cfg: RealEvaluationConfig, grid, *, rep: int,
                       rep_idx: np.ndarray, eval_curves: list, A: np.ndarray,
                       X: np.ndarray, psi_full: np.ndarray,
                       bridges: dict) -> list[dict]:
    """Run the seven arms on one replicate and score them against ``psi_full``.

    ``rep_idx`` selects the replicate's units; every arm and every ``p`` uses
    the same unit subset (paired design).  ``bridges[p]`` holds the fitted
    alpha bridge, the k-selected DTM bridge and the selection record.
    """
    idx = np.asarray(rep_idx, dtype=int)
    A_rep = np.asarray(A)[idx]
    X_rep = np.asarray(X)[idx]
    phi_full = np.stack([eval_curves[i].full_alpha for i in idx])
    grid_arr = np.asarray(grid.grid)

    rows = []
    for p in cfg.retention_ladder:
        phi_p = np.stack([eval_curves[i].thinned_alpha[p] for i in idx])
        k_star = int(bridges[float(p)]["selected_k"])
        phi_dtm_p = np.stack(
            [eval_curves[i].thinned_dtm[(float(p), k_star)] for i in idx])

        arms = {
            "M1_oracle_full_aipw":
                lambda: _aipw_band(cfg, phi_full, A_rep, X_rep, grid_arr,
                                   _seed(cfg, cfg.bootstrap_seed, rep, 1)),
            "M2_blind_aipw":
                lambda: _aipw_band(cfg, phi_p, A_rep, X_rep, grid_arr,
                                   _seed(cfg, cfg.bootstrap_seed, rep, 2)),
            "M3_bayes_causal_only":
                lambda: _plugin_band(cfg, phi_p, A_rep, X_rep, grid_arr,
                                     _seed(cfg, cfg.causal_seed, rep, 3)),
            "M4_bridge_bayes":
                lambda: _nested_band(cfg, bridges[float(p)]["alpha"], phi_p,
                                     None, A_rep, X_rep, grid_arr, rep, None),
            "M5_bridge_freq_mi":
                lambda: _mi_band(cfg, bridges[float(p)]["alpha"], phi_p, None,
                                 A_rep, X_rep, grid_arr, rep, None),
            "M4_dtm_bridge_bayes":
                lambda: _nested_band(cfg, bridges[float(p)]["dtm"], phi_p,
                                     phi_dtm_p, A_rep, X_rep, grid_arr, rep,
                                     k_star),
            "M5_dtm_bridge_freq_mi":
                lambda: _mi_band(cfg, bridges[float(p)]["dtm"], phi_p,
                                 phi_dtm_p, A_rep, X_rep, grid_arr, rep, k_star),
        }
        if float(p) == float(cfg.primary_p):
            for k in cfg.dtm_k_ladder:
                k = int(k)
                phi_dtm_k = np.stack(
                    [eval_curves[i].thinned_dtm[(float(p), k)] for i in idx])
                bridge_k = bridges[float(p)]["bridges_by_k"][k]
                arms[f"M4_dtm_k{k}_bridge_bayes"] = (
                    lambda bridge_k=bridge_k, phi_dtm_k=phi_dtm_k, k=k:
                    _nested_band(cfg, bridge_k, phi_p, phi_dtm_k, A_rep,
                                 X_rep, grid_arr, rep, k))
                arms[f"M5_dtm_k{k}_bridge_freq_mi"] = (
                    lambda bridge_k=bridge_k, phi_dtm_k=phi_dtm_k, k=k:
                    _mi_band(cfg, bridge_k, phi_p, phi_dtm_k, A_rep, X_rep,
                             grid_arr, rep, k))
        for method, runner in arms.items():
            try:
                band = runner()
                rows.append(_cov_row(cfg, band, psi_full, p=p, rep=rep,
                                     method=method))
            except Exception as exc:  # keep the replicate alive; row records it
                rows.append({
                    "p": float(p), "rep": int(rep), "method": method,
                    "cov_sim_full": float("nan"), "interval_score": float("nan"),
                    "band_width": float("nan"), "failed": True,
                    "error": f"{type(exc).__name__}: {exc}", "band_kind": "",
                })
    return rows


def draw_replicates(cfg: RealEvaluationConfig, n_units: int,
                    treated: np.ndarray, control: np.ndarray) -> list[np.ndarray]:
    """Registered replicate draws: 100 seeded subsamples of ``n_rep`` units.

    A replicate must contain >= 1 treated and >= 1 control unit; invalid draws
    are skipped until 100 valid replicates are recorded.
    """
    rng = _rng(cfg, cfg.rep_seed, 1)
    reps: list[np.ndarray] = []
    attempts = 0
    while len(reps) < int(cfg.n_replicates) and attempts < 100_000:
        attempts += 1
        order = rng.permutation(int(n_units))
        idx = order[: int(cfg.n_rep)]
        if np.sum(treated[idx]) >= 1 and np.sum(control[idx]) >= 1:
            reps.append(idx)
    if len(reps) < int(cfg.n_replicates):
        raise RuntimeError("could not draw enough valid replicates")
    return reps


def run_replicates(cfg: RealEvaluationConfig, grid, *, eval_curves: list,
                   A: np.ndarray, X: np.ndarray, psi_full: np.ndarray,
                   bridges: dict, n_jobs: int = 1,
                   verbose: bool = False) -> tuple["pd.DataFrame", dict]:
    """Draw the replicates, evaluate every arm and return the tidy frame."""
    import pandas as pd

    treated = np.asarray(A, dtype=int)
    control = 1 - treated
    reps = draw_replicates(cfg, len(eval_curves), treated, control)

    if n_jobs == 1:
        chunks = [evaluate_replicate(cfg, grid, rep=r, rep_idx=idx,
                                     eval_curves=eval_curves, A=A, X=X,
                                     psi_full=psi_full, bridges=bridges)
                  for r, idx in enumerate(reps)]
    else:
        from joblib import Parallel, delayed

        workers = max(1, min(int(n_jobs), len(reps)))
        chunks = Parallel(n_jobs=workers, backend="loky",
                          verbose=5 if verbose else 0)(
            delayed(evaluate_replicate)(cfg, grid, rep=r, rep_idx=idx,
                                        eval_curves=eval_curves, A=A, X=X,
                                        psi_full=psi_full, bridges=bridges)
            for r, idx in enumerate(reps))
    frame = pd.DataFrame([row for chunk in chunks for row in chunk])
    return frame, {"n_replicates": int(len(reps)), "n_units": int(len(eval_curves)),
                   "n_rep": int(cfg.n_rep)}


def aggregate_cov(frame, alpha: float = 0.05) -> "pd.DataFrame":
    """Per ``(p, method)`` coverage rate with Clopper-Pearson, plus width/scores."""
    import pandas as pd

    from ..benchmarks.metrics import coverage_rate_with_ci

    frame = pd.DataFrame(frame)
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for (p, method), cell in frame.groupby(["p", "method"]):
        ok = cell[~cell["failed"].astype(bool)]
        stats = coverage_rate_with_ci(ok["cov_sim_full"], alpha=alpha)
        rows.append({
            "p": float(p), "method": method,
            "cov_sim_full_rate": stats["coverage"],
            "cov_sim_full_cp_lower": stats["cp_lower"],
            "cov_sim_full_cp_upper": stats["cp_upper"],
            "cov_sim_full_n": int(stats["n_replicates"]),
            "interval_score_mean": float(np.mean(ok["interval_score"])),
            "band_width_mean": float(np.mean(ok["band_width"])),
            "n_failed": int(len(cell) - len(ok)),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["p", "method"]).reset_index(drop=True)


def design_matrix(unit_rows) -> np.ndarray:
    """Registered design covariates: time indicator, log cell count, plate."""
    import pandas as pd

    df = pd.DataFrame(unit_rows).reset_index(drop=True)
    time = np.where(np.asarray(df["time"], dtype=float) >= 48.0, 1.0, 0.0)
    logcells = np.log1p(np.asarray(df["n_cells"], dtype=float))
    plates = sorted(set(df["plate"]))
    plate_oh = np.stack([(np.asarray(df["plate"]) == pl).astype(float)
                         for pl in plates], axis=1)
    return np.column_stack([time, logcells, plate_oh])
