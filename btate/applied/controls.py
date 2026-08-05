"""Task 6.5.5 controls, and the per-arm curves the registered run discarded.

The registered seven-arm run (``run_phase6_5.run``) scores every band against
the fixed full-depth target and keeps only three scalars per row
(``cov_sim_full``, ``interval_score``, ``band_width``).  Two things the plan
asks for cannot be recovered from that:

* **Task 6.5.5 controls.**  The negative control (vehicle versus vehicle,
  where the true contrast is exactly the zero function) was never run, and it
  gates the phase: "A failure here halts the phase."  The null-thinning
  control (``p = 1.0``, where the bridge must be approximately the identity)
  was only observable indirectly, through coverage.
* **The blind / corrected / DTM-augmented conclusion comparison** required as
  a deliverable by ``docs/phase6_analysis.md`` item 4.  That needs each arm's
  estimated ``psi`` curve, which ``_cov_row`` throws away.

This module supplies both, reusing the registered pipeline prefix
(``run_phase6_5.build_context``) so the bridges, grids, frame and seeds are
the ones the registered run used.  Nothing here re-fits or re-tunes anything:
the bridges are consumed exactly as fitted.

Three control experiments are run.

``negative`` (Task 6.5.5, "Negative control")
    Pseudo-treatment is assigned at random among the 24 vehicle evaluation
    wells, on distinct plates, mirroring the real contrast's 3-treated
    imbalance.  Because assignment is independent of the wells, the true
    contrast is the **zero function**, exactly and by construction -- this is
    the one place in the whole project where the target is known without an
    operational definition.  Every arm's band must contain zero at its
    nominal rate.

``null_thinning`` (Task 6.5.5, "Null-thinning control")
    At every retention ``p``, the sup-norm distance between each corrected
    arm's estimate and the blind estimate on the *same* replicate and units.
    At ``p = 1.0`` the thinned data is the full data, so a correction that is
    the identity must give ~0.  This is the direct measurement of the
    registered falsification condition ("the correction is non-trivial at
    ``p = 1.0``"), which coverage alone reports only as a symptom.

``estimates`` (deliverable, not a control)
    The registered treated contrast re-run with every arm's estimate curve
    persisted, so blind, corrected and DTM-augmented conclusions can finally
    be compared and plotted against ``psi_full``.

Positive control and dose monotonicity (the other two bullets of Task 6.5.5)
are **not** implemented here: both need contrasts outside the registered
evaluation split (other compounds, and sci-Plex 2's dose ladder), which is a
separate data build.  They are listed as open in ``docs/phase6_5_analysis.md``.
"""
from __future__ import annotations

import numpy as np

from .evaluate import (
    RealEvaluationConfig,
    _aipw_band,
    _mi_band,
    _nested_band,
    _plugin_band,
    _seed,
    interval_score,
)

#: Arms that apply a measurement-error correction, paired with the bridge key
#: they consume.  Order matches the registered table.
CORRECTED_ARMS: tuple[tuple[str, str, bool], ...] = (
    ("M4_bridge_bayes", "alpha", False),
    ("M5_bridge_freq_mi", "alpha", False),
    ("M4_dtm_bridge_bayes", "dtm", True),
    ("M5_dtm_bridge_freq_mi", "dtm", True),
)


# --------------------------------------------------------------------------- #
# Arm runner shared by every control (returns bands, not just scalars)
# --------------------------------------------------------------------------- #
def run_arms(cfg: RealEvaluationConfig, grid, *, rep: int, idx: np.ndarray,
             eval_curves: list, A_rep: np.ndarray, X_rep: np.ndarray,
             bridges: dict, p: float) -> dict:
    """Run the seven registered arms on one replicate at one ``p``.

    Returns ``{method: RepBand}``.  The arm definitions, seed derivation and
    band constructions are the ones in ``evaluate.evaluate_replicate``; only
    the return type differs (bands are kept rather than reduced to scalars).
    Arms that raise are recorded as ``None`` so one failure cannot lose the
    replicate.
    """
    idx = np.asarray(idx, dtype=int)
    grid_arr = np.asarray(grid.grid)
    phi_full = np.stack([eval_curves[i].full_alpha for i in idx])
    phi_p = np.stack([eval_curves[i].thinned_alpha[float(p)] for i in idx])
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
            lambda: _mi_band(cfg, bridges[float(p)]["dtm"], phi_p, phi_dtm_p,
                             A_rep, X_rep, grid_arr, rep, k_star),
    }
    out = {}
    for method, runner in arms.items():
        try:
            out[method] = runner()
        except Exception as exc:  # recorded, never fatal
            out[method] = None
            out[f"__error__{method}"] = f"{type(exc).__name__}: {exc}"
    return out


# --------------------------------------------------------------------------- #
# Negative control: vehicle versus vehicle, true contrast identically zero
# --------------------------------------------------------------------------- #
def draw_null_assignments(cfg: RealEvaluationConfig, plates: np.ndarray,
                          control_positions: np.ndarray, *,
                          n_draws: int, n_pseudo_treated: int = 3,
                          seed: int = 65_011) -> list[np.ndarray]:
    """Seeded pseudo-treatment draws among vehicle wells, on distinct plates.

    ``control_positions`` indexes the vehicle units inside ``eval_curves``;
    ``plates`` gives their plate labels.  Each draw picks
    ``n_pseudo_treated`` wells on **distinct** plates, mirroring the real
    contrast (3 Alisertib wells on 3 different plates) so the design imbalance
    and the plate structure of the covariate matrix are the same as in the
    registered analysis.  Assignment is independent of well content, so the
    true contrast is the zero function for every draw.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))
    plates = np.asarray(plates)
    positions = np.asarray(control_positions, dtype=int)
    unique_plates = sorted(set(plates.tolist()))
    if len(unique_plates) < int(n_pseudo_treated):
        raise ValueError("not enough distinct plates for a plate-disjoint draw")
    draws = []
    while len(draws) < int(n_draws):
        chosen_plates = rng.choice(len(unique_plates),
                                   size=int(n_pseudo_treated), replace=False)
        picked = []
        for pl in chosen_plates:
            on_plate = positions[plates == unique_plates[pl]]
            picked.append(int(rng.choice(on_plate)))
        A = np.zeros(positions.size, dtype=int)
        lookup = {int(pos): j for j, pos in enumerate(positions)}
        for pos in picked:
            A[lookup[pos]] = 1
        draws.append(A)
    return draws


def negative_control(cfg: RealEvaluationConfig, grid, *, eval_curves: list,
                     control_positions: np.ndarray, X_control: np.ndarray,
                     plates: np.ndarray, bridges: dict,
                     n_draws: int = 100, n_pseudo_treated: int = 3,
                     n_jobs: int = 1, verbose: bool = False):
    """Vehicle-versus-vehicle coverage of the exact zero function.

    Unlike ``cov_sim_full``, this target needs no operational definition and
    no oracle: under a random pseudo-assignment the contrast **is** zero.  A
    band that fails to contain it at its nominal rate is broken independently
    of any bridge claim (Task 6.5.5).
    """
    positions = np.asarray(control_positions, dtype=int)
    zero = np.zeros(int(grid.resolution), dtype=float)
    assignments = draw_null_assignments(
        cfg, plates, positions, n_draws=n_draws,
        n_pseudo_treated=n_pseudo_treated)

    def one(rep: int, A_rep: np.ndarray) -> list[dict]:
        rows = []
        for p in cfg.retention_ladder:
            bands = run_arms(cfg, grid, rep=rep, idx=positions,
                             eval_curves=eval_curves, A_rep=A_rep,
                             X_rep=np.asarray(X_control), bridges=bridges,
                             p=float(p))
            for method, band in bands.items():
                if method.startswith("__error__"):
                    continue
                if band is None:
                    rows.append({
                        "p": float(p), "rep": int(rep), "method": method,
                        "cov_zero": float("nan"),
                        "interval_score_zero": float("nan"),
                        "band_width": float("nan"),
                        "max_abs_estimate": float("nan"),
                        "failed": True,
                        "error": bands.get(f"__error__{method}", "unknown"),
                    })
                    continue
                rows.append({
                    "p": float(p), "rep": int(rep), "method": method,
                    "cov_zero": float(np.all((zero >= band.lower) &
                                             (zero <= band.upper))),
                    "interval_score_zero": interval_score(
                        band.lower, band.upper, zero, cfg.alpha),
                    "band_width": float(np.mean(band.upper - band.lower)),
                    "max_abs_estimate": float(np.max(np.abs(band.estimate))),
                    "failed": False, "error": "",
                })
        return rows

    if n_jobs > 1:
        from joblib import Parallel, delayed
        chunks = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(one)(r, A) for r, A in enumerate(assignments))
    else:
        chunks = [one(r, A) for r, A in enumerate(assignments)]

    import pandas as pd
    return pd.DataFrame([row for chunk in chunks for row in chunk])


def aggregate_zero(frame, alpha: float = 0.05):
    """Per ``(p, method)`` coverage of zero with exact Clopper-Pearson bounds."""
    import pandas as pd

    from ..benchmarks.metrics import coverage_rate_with_ci

    frame = pd.DataFrame(frame)
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for (p, method), cell in frame.groupby(["p", "method"]):
        ok = cell[~cell["failed"].astype(bool)]
        if ok.empty:
            continue
        stats = coverage_rate_with_ci(ok["cov_zero"], alpha=alpha)
        rows.append({
            "p": float(p), "method": method,
            "cov_zero_rate": stats["coverage"],
            "cov_zero_cp_lower": stats["cp_lower"],
            "cov_zero_cp_upper": stats["cp_upper"],
            "n_replicates": int(stats["n_replicates"]),
            "interval_score_zero_mean": float(np.mean(ok["interval_score_zero"])),
            "band_width_mean": float(np.mean(ok["band_width"])),
            "max_abs_estimate_mean": float(np.mean(ok["max_abs_estimate"])),
            "n_failed": int(len(cell) - len(ok)),
        })
    return pd.DataFrame(rows).sort_values(["p", "method"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Estimates + null-thinning magnitude on the registered treated contrast
# --------------------------------------------------------------------------- #
def estimate_curves(cfg: RealEvaluationConfig, grid, *, eval_curves: list,
                    A: np.ndarray, X: np.ndarray, psi_full: np.ndarray,
                    bridges: dict, replicates: list, n_jobs: int = 1,
                    verbose: bool = False):
    """Re-run the registered replicates, keeping every arm's estimate curve.

    Returns ``(index_frame, curves)`` where ``curves`` is
    ``(n_rows, resolution)`` aligned row-for-row with ``index_frame``, plus the
    null-thinning magnitudes: for each corrected arm, the sup-norm distance
    from that arm's estimate to the blind (M2) estimate on the same replicate
    and the same units.  At ``p = 1.0`` that distance is the registered
    falsification quantity.
    """
    def one(rep: int, idx: np.ndarray) -> tuple[list[dict], list[np.ndarray]]:
        rows, curves = [], []
        A_rep = np.asarray(A)[np.asarray(idx, dtype=int)]
        X_rep = np.asarray(X)[np.asarray(idx, dtype=int)]
        for p in cfg.retention_ladder:
            bands = run_arms(cfg, grid, rep=rep, idx=idx,
                             eval_curves=eval_curves, A_rep=A_rep,
                             X_rep=X_rep, bridges=bridges, p=float(p))
            blind = bands.get("M2_blind_aipw")
            oracle = bands.get("M1_oracle_full_aipw")
            for method, band in bands.items():
                if method.startswith("__error__") or band is None:
                    continue
                est = np.asarray(band.estimate, dtype=float)
                rows.append({
                    "p": float(p), "rep": int(rep), "method": method,
                    "cov_sim_full": float(np.all((psi_full >= band.lower) &
                                                 (psi_full <= band.upper))),
                    "interval_score": interval_score(band.lower, band.upper,
                                                     psi_full, cfg.alpha),
                    "band_width": float(np.mean(band.upper - band.lower)),
                    "sup_err_vs_psi_full": float(
                        np.max(np.abs(est - np.asarray(psi_full)))),
                    "sup_shift_vs_blind": (
                        float("nan") if blind is None else
                        float(np.max(np.abs(est - blind.estimate)))),
                    "sup_shift_vs_oracle": (
                        float("nan") if oracle is None else
                        float(np.max(np.abs(est - oracle.estimate)))),
                    "peak_abs_estimate": float(np.max(np.abs(est))),
                    "argmax_t": float(np.asarray(grid.grid)[
                        int(np.argmax(np.abs(est)))]),
                })
                curves.append(est)
        return rows, curves

    if n_jobs > 1:
        from joblib import Parallel, delayed
        chunks = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(one)(r, idx) for r, idx in enumerate(replicates))
    else:
        chunks = [one(r, idx) for r, idx in enumerate(replicates)]

    import pandas as pd
    rows = [row for chunk, _ in chunks for row in chunk]
    curves = [curve for _, chunk in chunks for curve in chunk]
    return pd.DataFrame(rows), np.stack(curves) if curves else np.zeros((0, 0))


def aggregate_null_thinning(frame):
    """Per ``(p, method)`` mean and max sup-norm shift away from blind.

    The registered falsification reads the ``p = 1.0`` block: a correction
    that is the identity on uncorrupted data has ``sup_shift_vs_blind`` ~ 0
    there.  ``sup_err_vs_psi_full`` at ``p = 1.0`` is the companion number --
    how far the arm's point estimate lands from the target it should
    reproduce exactly.
    """
    import pandas as pd

    frame = pd.DataFrame(frame)
    if frame.empty:
        return pd.DataFrame()
    out = frame.groupby(["p", "method"]).agg(
        sup_shift_vs_blind_mean=("sup_shift_vs_blind", "mean"),
        sup_shift_vs_blind_max=("sup_shift_vs_blind", "max"),
        sup_err_vs_psi_full_mean=("sup_err_vs_psi_full", "mean"),
        sup_shift_vs_oracle_mean=("sup_shift_vs_oracle", "mean"),
        peak_abs_estimate_mean=("peak_abs_estimate", "mean"),
        n=("rep", "size"),
    ).reset_index()
    return out.sort_values(["p", "method"]).reset_index(drop=True)
