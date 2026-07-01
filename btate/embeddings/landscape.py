r"""Persistence landscapes (Research_Plan Task 2.2).

Wraps ``gudhi.representations.Landscape`` and reshapes its concatenated output
into explicit landscape levels ``lambda_1(t), ..., lambda_k(t)``.  Diagrams are
birth--death arrays, matching the TATE/Gudhi convention.
"""
from __future__ import annotations

import numpy as np
from gudhi.representations import Landscape

from .utils import as_diagram_list


def posterior_landscape(diagrams, num_landscapes: int = 5,
                        sample_range=(0.0, 0.2), resolution: int = 100,
                        keep_endpoints: bool = True,
                        return_grid: bool = False,
                        flatten: bool = False):
    """Compute top-``k`` persistence landscapes for posterior diagram samples.

    Parameters
    ----------
    diagrams : array-like or list of array-like
        Persistence diagrams in birth--death coordinates.
    num_landscapes : int
        Number of landscape levels to expose.
    sample_range : pair of float
        Shared grid range passed to Gudhi.
    resolution : int
        Number of samples per landscape level.
    keep_endpoints : bool
        Forwarded to Gudhi.
    return_grid : bool
        If true, return ``(landscapes, grid)``.
    flatten : bool
        If true, keep Gudhi's flattened shape
        ``(n_diagrams, num_landscapes * resolution)``.  Otherwise return
        ``(n_diagrams, num_landscapes, resolution)``.
    """
    if int(num_landscapes) < 1:
        raise ValueError("num_landscapes must be positive")
    dgm_list = as_diagram_list(diagrams)
    transformer = Landscape(
        num_landscapes=int(num_landscapes),
        resolution=int(resolution),
        sample_range=list(sample_range),
        keep_endpoints=keep_endpoints,
    )
    flat = transformer.fit_transform(dgm_list)
    out = flat if flatten else flat.reshape(len(dgm_list), int(num_landscapes), int(resolution))
    if return_grid:
        return out, np.asarray(transformer.grid_, dtype=float)
    return out


def landscape_distances(reference, diagrams, num_landscapes: int = 5,
                        sample_range=(0.0, 0.2), resolution: int = 100,
                        p: float = 2.0) -> np.ndarray:
    """Empirical ``L^p`` distances from one reference diagram's landscape.

    This is the small diagnostic needed for Phase-2 stability checks: perturb a
    diagram, transform all perturbations on the same grid, then inspect how the
    landscape distance scales with perturbation/noise.
    """
    if p <= 0:
        raise ValueError("p must be positive")
    all_diagrams = [as_diagram_list(reference)[0], *as_diagram_list(diagrams)]
    landscapes, grid = posterior_landscape(
        all_diagrams, num_landscapes=num_landscapes, sample_range=sample_range,
        resolution=resolution, return_grid=True,
    )
    ref = landscapes[0]
    vals = landscapes[1:]
    dx = 1.0 if grid.shape[0] < 2 else float(np.mean(np.diff(grid)))
    diff = np.abs(vals - ref[None, :, :]) ** p
    return (np.sum(diff, axis=(1, 2)) * dx) ** (1.0 / p)
