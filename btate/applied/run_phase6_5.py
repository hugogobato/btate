"""Driver: Phase-6.5 applied evaluation on real sci-Plex data (Tasks 6.5.2-6.5.4).

Runs the registered protocol end to end (``docs/phase6_5_registration.md``):
build the pair-disjoint split, fit the frozen frame, freeze the grids, compute
all unit curves, fit the alpha and DTM bridges, compute the fixed full-depth
estimand ``psi_full``, then run the seven-arm replicate evaluation and emit the
aggregate table with Clopper-Pearson intervals and width/score diagnostics.

RAM-bounded: ``adata`` is read backed, the frame is fit in dense IncrementalPCA
batches of 1,000 cells, and per-unit work (clouds, curves) is spread across
``n_jobs`` worker processes that each pin BLAS to one thread for
reproducibility.  Peak local RAM stays far below the 5 GB cap.

Artifacts are written under ``out_dir`` together with the frozen
representation hash; local runs and Colab shards share these modules so the
registered source hash is verifiable for any artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from ..benchmarks.measurement_error_uq import pin_blas_threads
from .evaluate import (
    RealEvaluationConfig,
    aggregate_cov,
    design_matrix,
    fit_all_bridges,
    psi_full_target,
    run_replicates,
)
from .paired_data import (
    RETENTION_LADDER,
    SUBSAMPLE_SEED,
    THRESHOLD_CELLS,
    _unit_table,
    build_split,
    unit_full_cloud,
    unit_curves,
)
from .representation import (
REPRESENTATION_HASH,
    FrameConfig,
    fit_frame,
    freeze_grid,
)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _stratified_calibration_cells(obs, split, cap: int, seed: int) -> np.ndarray:
    """Seeded near-uniform subsample of calibration full-depth cells.

    Compliance units contribute with probability proportional to their size
    (larger wells dominate the PCA fit, per registration Section 4); the total
    is capped at ``cap`` cells.
    """
    cal_units = split["units"].loc[split["calibration"], ["plate", "well"]]
    cal_pairs = set(zip(cal_units["plate"].astype(str),
                        cal_units["well"].astype(str)))
    obs_plates = obs["plate"].astype(str).to_numpy()
    obs_wells = obs["well"].astype(str).to_numpy()
    cal_mask = np.fromiter(
        ((plate, well) in cal_pairs
         for plate, well in zip(obs_plates, obs_wells)),
        dtype=bool,
        count=len(obs),
    )
    lines_raw = np.asarray(
        obs.loc[cal_mask, "cell_line"].astype(object).fillna("nan")
        .astype(str).to_numpy())
    cell_lines = sorted({l for l in lines_raw.tolist() if l != "nan"})
    picked = []
    done = 0
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    for line in cell_lines:
        cal_positions = np.flatnonzero(cal_mask)
        idx = cal_positions[lines_raw == line]
        n = int(np.ceil((cap - done) / max(1, len(cell_lines))))
        n = min(n, max(0, cap - done), idx.size)
        if n <= 0:
            continue
        sub = rng.choice(idx, size=n, replace=False)
        picked.append(sub)
        done += sub.size
    if not picked:
        raise RuntimeError("no calibration cells for the frame fit")
    return np.sort(np.concatenate(picked))


def _clouds_worker(payload):
    """Top-level worker: one calibration unit's full-depth cloud."""
    import scipy.sparse as sp

    adata_path, plate, well, frame, n_points, subsample_seed = payload
    pin_blas_threads(1)
    import anndata as ad

    adata = ad.read_h5ad(adata_path, backed="r")
    row = {"plate": plate, "well": well}
    cloud, _b, _c = unit_full_cloud(adata, row, frame, n_points=n_points,
                                    subsample_seed=subsample_seed)
    return cloud


def _curves_worker(payload):
    """Top-level worker: one unit's full set of curves."""
    adata_path, row, frame, grid, n_points, subsample_seed = payload
    pin_blas_threads(1)
    import anndata as ad

    adata = ad.read_h5ad(adata_path, backed="r")
    return unit_curves(adata, row, frame, grid, n_points=n_points,
                       subsample_seed=subsample_seed)


def run_pilot(out_dir, *, h5ad_path: str, n_points: int = 60,
              frame_cap: int = 5_000, max_calib: int = 4,
              verbose: bool = True) -> dict:
    """Run the non-interpretive sci-Plex 2 mechanics/timing pilot.

    sci-Plex 2 does not carry the registered sci-Plex 3 plate identifiers, so
    it cannot be passed through the primary split or used for any treatment
    contrast.  This path only exercises the frozen frame, grid, thinning and
    curve machinery on a seeded handful of eligible wells.
    """
    pin_blas_threads(1)
    import anndata as ad

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    adata = ad.read_h5ad(h5ad_path, backed="r")
    units = _unit_table(adata)
    units = units[units["n_cells"] >= THRESHOLD_CELLS].reset_index(drop=True)
    if units.empty:
        raise RuntimeError("sci-Plex 2 pilot has no units above threshold")

    obs = adata.obs
    eligible = obs["cell_line"].notna() & obs["well"].notna()
    eligible_idx = np.flatnonzero(np.asarray(eligible, dtype=bool))
    rng = np.random.default_rng(np.random.SeedSequence([65_001, 7]))
    n_frame = min(int(frame_cap), eligible_idx.size)
    frame_idx = np.sort(rng.choice(eligible_idx, size=n_frame, replace=False))
    frame = fit_frame(adata.X, cell_indices=frame_idx,
                      config=FrameConfig(frame_seed=65_001))

    n_units = min(int(max_calib) if int(max_calib) > 0 else 4,
                  len(units))
    chosen = np.sort(rng.choice(len(units), size=n_units, replace=False))
    rows = [units.iloc[i] for i in chosen]
    clouds = []
    for row in rows:
        cloud, _barcodes, _counts = unit_full_cloud(
            adata, row, frame, n_points=n_points,
            subsample_seed=SUBSAMPLE_SEED)
        clouds.append(cloud)
    grid = freeze_grid(frame, clouds)
    curves = [unit_curves(adata, row, frame, grid, n_points=n_points,
                          subsample_seed=SUBSAMPLE_SEED) for row in rows]

    artifacts = {
        "pilot": True,
        "interpretive": False,
        "h5ad": str(h5ad_path),
        "shape": [int(adata.shape[0]), int(adata.shape[1])],
        "n_eligible_units": int(len(units)),
        "n_selected_units": int(len(curves)),
        "n_frame_cells": int(n_frame),
        "n_points": int(n_points),
        "representation_hash": REPRESENTATION_HASH,
        "alpha_upper": float(grid.sample_range[1]),
        "dtm_upper": float(grid.dtm_sample_range[1]),
        "cardinality_drop": {
            str(p): [int(c.cardinality_drop[p]) for c in curves]
            for p in RETENTION_LADDER
        },
        "elapsed_seconds": float(time.time() - t0),
    }
    (out_dir / "pilot_artifacts.json").write_text(
        json.dumps(artifacts, indent=2, default=str))
    if verbose:
        print(json.dumps(artifacts, indent=2))
    return artifacts


def build_context(out_dir, *, h5ad_path: str, n_jobs: int = 1,
                  n_points: int = 60, n_replicates: int = 100, n_rep: int = 20,
                  frame_cap: int = 50_000, max_calib: int = 0,
                  verbose: bool = True) -> dict:
    """Everything upstream of the replicate evaluation, in registered order.

    This is the expensive prefix (frozen frame, frozen grids, all unit curves,
    all bridges) shared by the registered seven-arm run and by the Task-6.5.5
    control run.  Seeds and call order are identical in both, so the numbers a
    control run reproduces are bit-comparable with the registered run.

    Returns the split, unit table, frame, grid, evaluation curves, calibration
    curves and fitted bridges, and writes ``frame.npz``, ``grid.json`` and
    ``bridge_diagnostics.json`` into ``out_dir``.
    """
    pin_blas_threads(1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    import anndata as ad

    if verbose:
        print(f"[phase6.5] opening {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    frame_cfg = FrameConfig(frame_seed=65_001)
    eval_cfg = RealEvaluationConfig(n_rep=n_rep, n_replicates=n_replicates)
    parallel = n_jobs > 1
    if parallel:
        from joblib import Parallel, delayed

    split = build_split(adata, threshold=THRESHOLD_CELLS)
    units = split["units"].reset_index(drop=True)
    if verbose:
        print("split in %.1fs" % (time.time() - t0), split["protocol"])

    # 1. frozen frame on calibration full-depth cells (registered 50 000 cap).
    cell_idx = _stratified_calibration_cells(
        adata.obs, split, cap=int(frame_cap), seed=int(frame_cfg.frame_seed))
    cell_idx = cell_idx[: int(frame_cap)]
    if verbose:
        print(f"frame fit on {cell_idx.size} calibration cells")
    frame = fit_frame(adata.X, cell_indices=cell_idx, config=frame_cfg)
    if verbose:
        print("frame in %.1fs, EV1=%.3f" % (time.time() - t0,
                                            frame.explained_variance[0]))
    np.savez(out_dir / "frame.npz", loadings=frame.loadings,
             column_mean=frame.column_mean, median_total=np.array(frame.median_total),
             explained_variance=frame.explained_variance,
             hash=np.array([frame.hash], dtype=object))

    # 2. calibration full-depth clouds -> frozen grids (registered rule).
    cal_idx = np.flatnonzero(split["calibration"])
    if int(max_calib) > 0:
        rng = np.random.default_rng(65003)
        cal_idx = rng.choice(cal_idx, size=min(int(max_calib), cal_idx.size),
                             replace=False)
        if verbose:
            print(f"smoke: calibration capped to {cal_idx.size} units")
    payloads = [(h5ad_path, units.at[i, "plate"], units.at[i, "well"],
                 frame, n_points, SUBSAMPLE_SEED) for i in cal_idx]
    if parallel:
        cal_clouds = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
            delayed(_clouds_worker)(p) for p in payloads)
    else:
        cal_clouds = [_clouds_worker(p) for p in payloads]
    grid = freeze_grid(frame, [np.asarray(c, dtype=float) for c in cal_clouds])
    if verbose:
        print("grid frozen: alpha=%.4f dtm=%.4f  in %.1fs"
              % (grid.sample_range[1], grid.dtm_sample_range[1],
                 time.time() - t0))
    (out_dir / "grid.json").write_text(json.dumps({
        "alpha_upper": float(grid.sample_range[1]),
        "dtm_upper": float(grid.dtm_sample_range[1]),
        "resolution": int(grid.resolution),
        "provenance": grid.provenance,
    }, indent=2))

    # 2. all unit curves (26 eval + calibration units).
    ev_idx = np.flatnonzero(split["evaluation"])
    all_rows = list(ev_idx) + list(cal_idx)
    payloads = [(h5ad_path, units.loc[i], frame, grid, n_points, SUBSAMPLE_SEED)
                for i in all_rows]
    if parallel:
        curves = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_curves_worker)(p) for p in payloads)
    else:
        curves = [_curves_worker(p) for p in payloads]
    if verbose:
        print("curves for %d units in %.1fs" % (len(curves), time.time() - t0))

    cal_curves = {curves[j].unit_id: curves[j]
                  for j in range(len(ev_idx), len(curves))}
    ev_curves = [curves[j] for j in range(len(ev_idx))]

    # 3. bridges on the calibration pool (alpha + DTM with k selection).
    if verbose:
        print("bridges ...")
    bridges = fit_all_bridges(eval_cfg, grid, list(cal_curves.values()))
    bridge_artifacts = {}
    for p, d in bridges.items():
        print(f"  p={p}: k*={d['selected_k']} "
              f"alpha logpdf={d['alpha'].diagnostics['selected_holdout_logpdf']:.3f} "
              f"rmse={d['alpha'].diagnostics['holdout_clean_alpha_rmse']:.4f}")
        bridge_artifacts[str(p)] = {
            "selected_k": int(d["selected_k"]),
            "alpha": d["alpha"].to_dict(),
            "dtm_selected": d["dtm"].to_dict(),
            "dtm_by_k": {str(k): br.to_dict()
                         for k, br in d["bridges_by_k"].items()},
            "k_selection": d["k_selection"],
        }
    (out_dir / "bridge_diagnostics.json").write_text(json.dumps(
        bridge_artifacts, indent=2, default=_json_default))

    return {
        "adata_path": h5ad_path,
        "split": split,
        "units": units,
        "ev_idx": ev_idx,
        "cal_idx": cal_idx,
        "frame": frame,
        "grid": grid,
        "ev_curves": ev_curves,
        "cal_curves": cal_curves,
        "bridges": bridges,
        "eval_cfg": eval_cfg,
        "frame_cells": int(cell_idx.size),
        "t0": t0,
    }


def run(out_dir, *, h5ad_path: str, n_jobs: int = 1, n_points: int = 60,
        n_replicates: int = 100, n_rep: int = 20, frame_cap: int = 50_000,
        max_calib: int = 0, verbose: bool = True) -> dict:
    out_dir = Path(out_dir)
    ctx = build_context(out_dir, h5ad_path=h5ad_path, n_jobs=n_jobs,
                        n_points=n_points, n_replicates=n_replicates,
                        n_rep=n_rep, frame_cap=frame_cap, max_calib=max_calib,
                        verbose=verbose)
    t0 = ctx["t0"]
    split, units, ev_idx = ctx["split"], ctx["units"], ctx["ev_idx"]
    frame, grid = ctx["frame"], ctx["grid"]
    ev_curves, cal_curves = ctx["ev_curves"], ctx["cal_curves"]
    bridges, eval_cfg = ctx["bridges"], ctx["eval_cfg"]

    # 4. fixed full-depth estimand and the replicate evaluation.
    A = split["treated"][split["evaluation"]].astype(int)
    ev_rows = units.loc[ev_idx]
    X = design_matrix(ev_rows[["time", "n_cells", "plate"]])
    psi = psi_full_target(eval_cfg, ev_curves, A, X, grid, seed=65_008)
    frame_df, meta = run_replicates(eval_cfg, grid, eval_curves=ev_curves,
                                    A=A, X=X, psi_full=psi, bridges=bridges,
                                    n_jobs=n_jobs, verbose=verbose)
    agg = aggregate_cov(frame_df)
    if verbose:
        print("replicates in %.1fs" % (time.time() - t0), meta)
        print(agg.to_string(index=False))

    artifacts = {
        "representation_hash": REPRESENTATION_HASH,
        "pca_ev1": float(frame.explained_variance[0]),
        "alpha_upper": float(grid.sample_range[1]),
        "dtm_upper": float(grid.dtm_sample_range[1]),
        "n_eval": int(split["evaluation"].sum()),
        "n_treated_eval": int(split["treated"].sum()),
        "n_control_eval": int(split["control"].sum()),
        "n_calibration": int(split["calibration"].sum()),
        "n_excluded": int(split["excluded"].sum()),
        "n_frame_cells": int(ctx["frame_cells"]),
        "n_replicates": meta["n_replicates"],
        "n_rep": meta["n_rep"],
    }
    row = units.loc[ev_idx]
    artifacts["eval_units"] = row[["plate", "well", "perturbation",
                                   "n_cells"]].to_dict("records")
    (out_dir / "artifacts.json").write_text(json.dumps(
        artifacts, indent=2, default=str))
    frame_df.to_csv(out_dir / "replicate_rows.csv", index=False)
    agg.to_csv(out_dir / "aggregate.csv", index=False)
    dtm_ablation = frame_df[
        frame_df["method"].astype(str).str.contains("_dtm_k")
    ].copy()
    dtm_ablation.to_csv(out_dir / "dtm_ablation_rows.csv", index=False)
    aggregate_dtm = aggregate_cov(dtm_ablation)
    aggregate_dtm.to_csv(out_dir / "dtm_ablation_aggregate.csv", index=False)

    geometry_rows = []
    for source, curves_to_report in (
            ("calibration", list(cal_curves.values())),
            ("evaluation", ev_curves)):
        for curve in curves_to_report:
            for p, diagnostic in curve.geometry.items():
                geometry_rows.append({
                    "source": source,
                    "unit_id": curve.unit_id,
                    "p": float(p),
                    **diagnostic,
                })
    import pandas as pd
    geometry_frame = pd.DataFrame(geometry_rows)
    geometry_frame.to_csv(out_dir / "corruption_geometry.csv", index=False)
    if not geometry_frame.empty:
        numeric = geometry_frame.select_dtypes(include=[np.number])
        geometry_summary = geometry_frame[["source", "p"]].copy()
        summary = numeric.groupby([geometry_frame["source"],
                                   geometry_frame["p"]]).mean()
        geometry_summary = summary.reset_index()
        geometry_summary.to_csv(out_dir / "corruption_geometry_summary.csv",
                                index=False)
    np.save(out_dir / "psi_full.npy", psi)
    return artifacts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", default="data/srivatsan_2020_sciplex3.h5ad")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--n-replicates", type=int, default=100)
    ap.add_argument("--n-rep", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=60)
    ap.add_argument("--frame-cap", type=int, default=50_000)
    ap.add_argument("--max-calib", type=int, default=0,
                    help="cap calibration units (smoke runs only; 0 = all)")
    ap.add_argument("--pilot", action="store_true",
                    help="run the non-interpretive sci-Plex 2 mechanics pilot")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pin_blas_threads(1)
    import anndata as ad

    if args.pilot:
        run_pilot(args.out, h5ad_path=args.h5, n_points=args.n_points,
                  frame_cap=args.frame_cap,
                  max_calib=args.max_calib or 4)
        return 0

    adata = ad.read_h5ad(args.h5, backed="r")
    split = build_split(adata, threshold=THRESHOLD_CELLS)
    if args.dry_run:
        print(json.dumps(split["protocol"], indent=2))
        print({"eval": int(split["evaluation"].sum()),
               "calib": int(split["calibration"].sum()),
               "excluded": int(split["excluded"].sum()),
               "treated": int(split["treated"].sum()),
               "control": int(split["control"].sum())})
        return 0
    n_jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 2) - 4)
    if args.max_calib:
        run(args.out, h5ad_path=args.h5, n_jobs=n_jobs, n_points=args.n_points,
            n_replicates=args.n_replicates, n_rep=args.n_rep,
            frame_cap=args.frame_cap, max_calib=args.max_calib)
    else:
        run(args.out, h5ad_path=args.h5, n_jobs=n_jobs, n_points=args.n_points,
            n_replicates=args.n_replicates, n_rep=args.n_rep,
            frame_cap=args.frame_cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
