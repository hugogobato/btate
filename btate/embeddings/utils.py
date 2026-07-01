"""Shared utilities for Phase-2 functional embeddings."""
from __future__ import annotations

import numpy as np


def as_diagram(diagram) -> np.ndarray:
    """Return ``diagram`` as a finite ``(n, 2)`` birth--death array."""
    d = np.asarray(diagram, dtype=float)
    if d.size == 0:
        return np.empty((0, 2), dtype=float)
    d = np.atleast_2d(d)
    if d.ndim != 2 or d.shape[1] != 2:
        raise ValueError(f"expected a (n, 2) diagram, got shape {d.shape}")
    if np.any(~np.isfinite(d)):
        raise ValueError("diagram coordinates must be finite")
    return d


def as_diagram_list(diagrams) -> list[np.ndarray]:
    """Normalize a single diagram or a list/tuple of diagrams."""
    if isinstance(diagrams, np.ndarray):
        return [as_diagram(diagrams)]
    if isinstance(diagrams, (list, tuple)):
        if len(diagrams) == 0:
            return []
        # A Python list ``[[b, d], ...]`` should be treated as one diagram.
        try:
            arr = np.asarray(diagrams, dtype=float)
        except (TypeError, ValueError):
            arr = None
        if arr is not None and arr.ndim == 2 and arr.shape[1] == 2:
            return [as_diagram(arr)]
        return [as_diagram(d) for d in diagrams]
    return [as_diagram(diagrams)]


def as_weight_list(weights, diagrams: list[np.ndarray]) -> list[np.ndarray]:
    """Normalize per-point weights to one array per diagram."""
    if len(diagrams) == 0:
        return []

    if isinstance(weights, np.ndarray):
        if len(diagrams) == 1:
            weight_list = [weights]
        elif weights.ndim == 1 and weights.shape[0] == len(diagrams):
            # Ambiguous but occasionally useful for one-point diagrams.
            weight_list = [np.asarray([w], dtype=float) for w in weights]
        else:
            weight_list = list(weights)
    elif isinstance(weights, (list, tuple)):
        if len(diagrams) == 1:
            w = np.asarray(weights, dtype=float)
            if w.ndim == 1 and w.shape[0] == diagrams[0].shape[0]:
                weight_list = [w]
            else:
                weight_list = list(weights)
        else:
            weight_list = list(weights)
    else:
        raise ValueError("weights must be an array or a list of arrays")

    if len(weight_list) != len(diagrams):
        raise ValueError("number of weight arrays must match number of diagrams")

    out = []
    for i, (w, d) in enumerate(zip(weight_list, diagrams)):
        arr = np.asarray(w, dtype=float).ravel()
        if arr.shape[0] != d.shape[0]:
            raise ValueError(
                f"weights for diagram {i} have length {arr.shape[0]}, "
                f"expected {d.shape[0]}"
            )
        out.append(arr)
    return out


def resolve_grid(diagrams: list[np.ndarray], sample_range, resolution: int,
                 keep_endpoints: bool = True) -> np.ndarray:
    """Resolve the common embedding grid.

    Explicit finite ranges match Gudhi's ``np.linspace`` behavior.  If one or
    both endpoints are ``np.nan``, infer them from the union of non-empty
    diagram supports ``[birth, death]``.  When ``keep_endpoints=False`` and an
    endpoint was inferred, the outer inferred endpoint is trimmed in the same
    spirit as Gudhi's plotting-oriented option.
    """
    if int(resolution) < 1:
        raise ValueError("resolution must be positive")
    res = int(resolution)
    sr = np.asarray(sample_range, dtype=float).ravel()
    if sr.shape[0] != 2:
        raise ValueError("sample_range must contain exactly two values")

    nan_mask = np.isnan(sr)
    if np.any(nan_mask):
        nonempty = [d for d in diagrams if d.shape[0] > 0]
        if not nonempty:
            raise ValueError("cannot infer sample_range from empty diagrams")
        all_pts = np.vstack(nonempty)
        inferred = np.array([np.min(all_pts[:, 0]), np.max(all_pts[:, 1])], dtype=float)
        sr[nan_mask] = inferred[nan_mask]

    if sr[1] < sr[0]:
        raise ValueError("sample_range upper endpoint must be >= lower endpoint")
    if sr[0] == sr[1]:
        return np.full(res, sr[0], dtype=float)

    grid_res = res + (int(np.sum(nan_mask)) if not keep_endpoints else 0)
    grid = np.linspace(sr[0], sr[1], grid_res)
    if not keep_endpoints and np.any(nan_mask):
        keep = np.ones(grid.shape[0], dtype=bool)
        if nan_mask[0]:
            keep[0] = False
        if nan_mask[1]:
            keep[-1] = False
        grid = grid[keep]
    return grid


def diagram_tents(diagram: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gudhi-compatible tent matrix with shape ``(resolution, n_points)``."""
    diagram = as_diagram(diagram)
    if diagram.shape[0] == 0:
        return np.zeros((grid.shape[0], 0), dtype=float)
    midpoints = (diagram[:, 0] + diagram[:, 1]) / 2.0
    heights = (diagram[:, 1] - diagram[:, 0]) / 2.0
    return np.maximum(heights[None, :] - np.abs(grid[:, None] - midpoints[None, :]), 0.0)
