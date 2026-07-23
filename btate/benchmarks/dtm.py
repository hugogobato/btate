"""DTM (Distance-to-Measure) filtration support for Phase 5.5.

The Phase-5 diagnosis (``docs/phase5_decision.md`` §7) showed that the alpha-complex
diagram of a noisy cloud has already lost the signal loop's death time: interior
clutter fills the loop and destroys the ``H1`` death coordinate.  The DTM filtration
(Chazal, Cohen-Steiner & Mérigot, 2011; Anai et al., 2019) replaces pointwise
distances with a distance-to-a-local-mass measure that is robust to a bounded
fraction of outliers.

Pre-check (``scratchpad/dtm_death_pilot.py``; ``results/phase5_decision/dtm_death_pilot.csv``):
alpha loses 41–58 % of the signal death time; DTM-Rips at ``k=15`` loses 6–19 %.

This module provides:

* :func:`h1_diagram_dtm` — DTM-Rips ``H1`` persistence diagram (birth–death).
* :func:`h1_diagram_filtration` — dispatcher: ``"alpha"`` or ``"dtm_rips_k{N}"``.
* :func:`dtm_death_sweep` — the Task-5.5.2 filtration sweep across cells.
"""
from __future__ import annotations

import numpy as np


def h1_diagram_dtm(points: np.ndarray, k: int = 15,
                   max_dimension: int = 2) -> np.ndarray:
    """DTM-Rips H1 persistence diagram in birth–death coordinates.

    Parameters
    ----------
    points : (m, d) array
        Point cloud in R^d.
    k : int
        Number of nearest neighbours for the DTM.  ``k=15`` is the operating
        point from the Phase-5.5 pre-check (best death recovery at noise 1–2).
    max_dimension : int
        Maximum simplex dimension for the Rips complex (default 2 gives the
        2-skeleton needed for H1).

    Returns
    -------
    (n_features, 2) array of (birth, death) pairs, finite only.
    """
    import gudhi as gd
    from gudhi.dtm_rips_complex import DTMRipsComplex

    points = np.asarray(points, dtype=float)
    if points.shape[0] < 3:
        return np.empty((0, 2), dtype=float)
    kk = max(2, min(int(k), points.shape[0] - 1))
    st = DTMRipsComplex(points=points, k=kk).create_simplex_tree(
        max_dimension=max_dimension)
    st.compute_persistence()
    dgm = np.asarray(st.persistence_intervals_in_dimension(1), dtype=float)
    if dgm.size == 0:
        return np.empty((0, 2), dtype=float)
    dgm = dgm[np.isfinite(dgm).all(axis=1)]
    return dgm


def h1_diagram_filtration(points: np.ndarray, filtration: str = "alpha",
                          **kwargs) -> np.ndarray:
    """Dispatch to the right H1 diagram computation.

    Parameters
    ----------
    filtration : str
        ``"alpha"`` (the Phase-5 default; calls :func:`pipeline.h1_diagram`) or
        ``"dtm_rips_k{N}"`` where ``N`` is the DTM neighbour count (e.g.
        ``"dtm_rips_k15"``; calls :func:`h1_diagram_dtm` with ``k=N``).
    """
    if filtration == "alpha":
        from .pipeline import h1_diagram
        return h1_diagram(points)
    if filtration.startswith("dtm_rips_k"):
        k = int(filtration.split("k", 1)[1])
        return h1_diagram_dtm(points, k=k, **kwargs)
    raise ValueError(f"unknown filtration: {filtration!r}; "
                     f"expected 'alpha' or 'dtm_rips_k{{N}}'")


def top_feature_death(diagram: np.ndarray) -> float:
    """Death coordinate of the most-persistent H1 feature (the signal loop).

    Returns ``nan`` if the diagram is empty.
    """
    d = np.asarray(diagram, dtype=float)
    if d.size == 0 or d.shape[0] == 0:
        return float("nan")
    idx = int(np.argmax(d[:, 1] - d[:, 0]))
    return float(d[idx, 1])


def dtm_death_sweep(cells, filtrations=("alpha", "dtm_rips_k5", "dtm_rips_k15",
                                        "dtm_rips_k30"),
                    n_reps: int = 3, n_subjects: int = 60,
                    max_pts: int = 200, n_jobs: int = 14,
                    seed_base: int = 20260701):
    """Task-5.5.2 sweep: death_recovery per filtration per cell.

    Parameters
    ----------
    cells : list of (mode, noise) tuples
        e.g. ``[("std", 1.0), ("lowsnr", 1.0), ("lowsnr", 2.0), ("lowsnr", 4.0)]``.
    filtrations : tuple of str
        Filtration names accepted by :func:`h1_diagram_filtration`.
    n_reps : int
        Number of independent dataset realizations.
    n_subjects : int
        Subjects per dataset (subsample if needed).
    max_pts : int
        Max points per cloud (subsample larger clouds).
    n_jobs : int
        Parallelism (one worker per (cell, rep) pair).
    seed_base : int

    Returns
    -------
    list of dicts, one per (cell, rep, subject, filtration) with columns
    ``cell``, ``rep``, ``subject``, ``filtration``, ``death_clean``,
    ``death_obs``, ``death_recovery``.
    """
    from joblib import Parallel, delayed
    from .synthetic import generate_synthetic_dataset, low_snr_config, standard_config

    def _subsample(pts, rng):
        pts = np.asarray(pts, dtype=float)
        if len(pts) <= max_pts:
            return pts
        return pts[rng.choice(len(pts), max_pts, replace=False)]

    def _one(mode, noise, rep):
        cfg_fn = standard_config if mode == "std" else low_snr_config
        cfg = cfg_fn(n=max(n_subjects, 20), noise_level=noise,
                     seed=seed_base + 137 * rep)
        ds = generate_synthetic_dataset(cfg)
        obs = ds.observed_clouds()
        rng = np.random.default_rng(1234 + rep)
        out = []
        for i in range(min(n_subjects, len(obs))):
            a = int(ds.A[i])
            pc = _subsample(ds.clean_clouds[i, a], rng)
            po = _subsample(obs[i], rng)
            for fname in filtrations:
                dc = top_feature_death(h1_diagram_filtration(pc, fname))
                do = top_feature_death(h1_diagram_filtration(po, fname))
                if not (np.isfinite(dc) and np.isfinite(do)) or dc <= 1e-9:
                    continue
                out.append({
                    "cell": f"{mode}_noise{noise}",
                    "rep": rep, "subject": i,
                    "filtration": fname,
                    "death_clean": dc, "death_obs": do,
                    "death_recovery": (do - dc) / dc,
                })
        return out

    jobs = [(m, n, r) for m, n in cells for r in range(n_reps)]
    results = Parallel(n_jobs=min(n_jobs, len(jobs)), verbose=5)(
        delayed(_one)(m, n, r) for m, n, r in jobs)
    return [row for chunk in results for row in chunk]
