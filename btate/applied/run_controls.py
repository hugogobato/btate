"""Driver: Task-6.5.5 controls and the per-arm curve deliverable.

Runs, on the registered sci-Plex 3 split and the registered bridges:

1. the **negative control** -- vehicle versus vehicle among the 24 evaluation
   control wells, where the true contrast is exactly zero, so band coverage
   needs no oracle and no operational reference;
2. the **null-thinning control** -- how far each corrected arm moves the
   estimate away from the blind estimate at every retention ``p``, read at
   ``p = 1.0`` where a correct correction must be the identity;
3. the **estimate curves** for every arm on the registered treated contrast,
   which the registered run computed and discarded, and without which the
   blind / corrected / DTM-augmented conclusion comparison required by
   ``docs/phase6_analysis.md`` item 4 cannot be produced.

It also persists the per-unit evaluation curves (``eval_curves.npz``), so any
future re-analysis of the evaluation set costs seconds rather than a full
bridge re-fit.

The expensive prefix is ``run_phase6_5.build_context``, the same function the
registered run uses, called with the same arguments and seeds.  Nothing is
re-tuned: bridges are consumed exactly as fitted.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from ..benchmarks.measurement_error_uq import pin_blas_threads
from .controls import (
    aggregate_null_thinning,
    aggregate_zero,
    estimate_curves,
    negative_control,
)
from .evaluate import design_matrix, draw_replicates, psi_full_target
from .paired_data import EVAL_CONTROL_PERTURBATION, RETENTION_LADDER
from .representation import REPRESENTATION_HASH
from .run_phase6_5 import build_context


def run(out_dir, *, h5ad_path: str, n_jobs: int = 1, n_points: int = 60,
        n_replicates: int = 100, n_rep: int = 20, frame_cap: int = 50_000,
        n_pseudo_treated: int = 3, max_calib: int = 0,
        verbose: bool = True) -> dict:
    pin_blas_threads(1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = build_context(out_dir, h5ad_path=h5ad_path, n_jobs=n_jobs,
                        n_points=n_points, n_replicates=n_replicates,
                        n_rep=n_rep, frame_cap=frame_cap, max_calib=max_calib,
                        verbose=verbose)
    t0 = ctx["t0"]
    split, units, ev_idx = ctx["split"], ctx["units"], ctx["ev_idx"]
    grid, ev_curves = ctx["grid"], ctx["ev_curves"]
    bridges, cfg = ctx["bridges"], ctx["eval_cfg"]

    ev_rows = units.loc[ev_idx].reset_index(drop=True)
    A = split["treated"][split["evaluation"]].astype(int)
    X = design_matrix(ev_rows[["time", "n_cells", "plate"]])

    # Persist the evaluation curves so re-analysis never re-fits the bridges.
    np.savez_compressed(
        out_dir / "eval_curves.npz",
        grid=np.asarray(grid.grid),
        unit_id=np.array([c.unit_id for c in ev_curves], dtype=object),
        A=np.asarray(A),
        X=np.asarray(X),
        full_alpha=np.stack([c.full_alpha for c in ev_curves]),
        **{f"thinned_alpha_{p}": np.stack([c.thinned_alpha[float(p)]
                                           for c in ev_curves])
           for p in RETENTION_LADDER},
        allow_pickle=True,
    )

    # ------------------------------------------------------------------ #
    # 1. Negative control: vehicle versus vehicle, truth = zero function
    # ------------------------------------------------------------------ #
    is_control = np.asarray(
        ev_rows["perturbation"].astype(str) == EVAL_CONTROL_PERTURBATION)
    control_positions = np.flatnonzero(is_control)
    if control_positions.size < n_pseudo_treated + 2:
        raise RuntimeError("too few vehicle units for a negative control")
    plates = np.asarray(ev_rows.loc[is_control, "plate"].astype(str))
    X_control = design_matrix(
        ev_rows.loc[is_control, ["time", "n_cells", "plate"]])

    if verbose:
        print(f"[controls] negative control on {control_positions.size} "
              f"vehicle wells across {len(set(plates.tolist()))} plates, "
              f"{n_pseudo_treated} pseudo-treated per draw")
    null_frame = negative_control(
        cfg, grid, eval_curves=ev_curves,
        control_positions=control_positions, X_control=X_control,
        plates=plates, bridges=bridges, n_draws=n_replicates,
        n_pseudo_treated=n_pseudo_treated, n_jobs=n_jobs, verbose=verbose)
    null_agg = aggregate_zero(null_frame)
    null_frame.to_csv(out_dir / "negative_control_rows.csv", index=False)
    null_agg.to_csv(out_dir / "negative_control_aggregate.csv", index=False)
    if verbose:
        print("negative control in %.1fs" % (time.time() - t0))
        print(null_agg.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 2 + 3. Estimate curves and null-thinning magnitude, treated contrast
    # ------------------------------------------------------------------ #
    psi = psi_full_target(cfg, ev_curves, A, X, grid, seed=65_008)
    replicates = draw_replicates(cfg, len(ev_curves),
                                 np.asarray(A).astype(bool),
                                 ~np.asarray(A).astype(bool))
    if verbose:
        print(f"[controls] estimate curves over {len(replicates)} registered "
              f"replicates")
    est_frame, curves = estimate_curves(
        cfg, grid, eval_curves=ev_curves, A=A, X=X, psi_full=psi,
        bridges=bridges, replicates=replicates, n_jobs=n_jobs, verbose=verbose)
    thin_agg = aggregate_null_thinning(est_frame)
    est_frame.to_csv(out_dir / "estimate_rows.csv", index=False)
    thin_agg.to_csv(out_dir / "null_thinning_aggregate.csv", index=False)
    np.savez_compressed(out_dir / "estimate_curves.npz",
                        curves=curves, grid=np.asarray(grid.grid), psi_full=psi)
    np.save(out_dir / "psi_full.npy", psi)
    if verbose:
        print("estimates in %.1fs" % (time.time() - t0))
        print(thin_agg.to_string(index=False))

    artifacts = {
        "representation_hash": REPRESENTATION_HASH,
        "n_eval": int(split["evaluation"].sum()),
        "n_control_eval": int(control_positions.size),
        "n_pseudo_treated": int(n_pseudo_treated),
        "n_null_draws": int(n_replicates),
        "n_estimate_replicates": int(len(replicates)),
        "control_plates": sorted(set(plates.tolist())),
        "grid_resolution": int(grid.resolution),
        "elapsed_seconds": float(time.time() - t0),
    }
    (out_dir / "controls_artifacts.json").write_text(
        json.dumps(artifacts, indent=2, default=str))
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
    ap.add_argument("--n-pseudo-treated", type=int, default=3)
    ap.add_argument("--max-calib", type=int, default=0,
                    help="cap calibration units (smoke runs only; 0 = all)")
    args = ap.parse_args()

    pin_blas_threads(1)
    n_jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 2) - 4)
    run(args.out, h5ad_path=args.h5, n_jobs=n_jobs, n_points=args.n_points,
        n_replicates=args.n_replicates, n_rep=args.n_rep,
        frame_cap=args.frame_cap, n_pseudo_treated=args.n_pseudo_treated,
        max_calib=args.max_calib)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
