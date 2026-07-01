r"""Probabilistically-weighted silhouettes (Research_Plan Task 2.1).

This module keeps parity with the TATE/Gudhi silhouette transform while allowing
the fixed power weight :math:`|d-b|^r` to be replaced by the posterior signal
probability :math:`\pi_p` from the Phase-1 partition model:

.. math::

   \phi(t; \widetilde D)
      = \frac{\sum_{p \in \widetilde D} \pi_p \Lambda_p(t)}
             {\sum_{p \in \widetilde D} \pi_p}.

Gudhi's vector-method implementation evaluates the tent function with midpoint
``(b+d)/2`` and half-lifetime ``(d-b)/2``, then multiplies the final curve by
``sqrt(2)``.  The implementation below follows that convention exactly so
``weights="power"`` matches ``top-causal-effect-main/utils/silhouette.py``.
"""
from __future__ import annotations

import numpy as np

from .utils import as_diagram_list, as_weight_list, diagram_tents, resolve_grid


def _power_weights(diagram: np.ndarray, r: float) -> np.ndarray:
    lifetimes = diagram[:, 1] - diagram[:, 0]
    return np.abs(lifetimes) ** float(r)


def weighted_silhouette(diagrams, weights: str | list | np.ndarray = "power",
                        r: float = 3.0, pi=None, sample_range=(0.0, 0.2),
                        resolution: int = 100, keep_endpoints: bool = True,
                        return_grid: bool = False,
                        zero_policy: str = "zero"):
    """Compute fixed-power or ``pi_p``-weighted silhouettes on one grid.

    Parameters
    ----------
    diagrams : array-like or list of array-like
        Persistence diagrams in birth--death coordinates, each with shape
        ``(n_points, 2)``.
    weights : {"power", "pi"} or array-like/list of array-like
        ``"power"`` uses the TATE baseline weight ``|d-b|^r``. ``"pi"`` uses
        the per-point probabilities supplied by ``pi``.  Passing arrays directly
        is treated as explicit per-point weights.
    r : float
        Power exponent for ``weights="power"``.
    pi : array-like or list of array-like, optional
        Per-point signal probabilities aligned to ``diagrams`` when
        ``weights="pi"``.
    sample_range : pair of float
        Shared filtration grid range.  ``np.nan`` endpoints are inferred from
        the diagrams using Gudhi's midpoint/height support convention.
    resolution : int
        Number of grid points.
    keep_endpoints : bool
        Kept for API parity with Gudhi.  With an explicit finite
        ``sample_range`` the grid includes both endpoints, matching Gudhi.
    return_grid : bool
        If true, return ``(curves, grid)``.
    zero_policy : {"zero", "uniform", "raise"}
        Behavior when a diagram has no positive total weight. ``"zero"``
        returns a zero curve; ``"uniform"`` averages all tents equally;
        ``"raise"`` raises ``ValueError``.

    Returns
    -------
    np.ndarray or tuple
        Silhouette curves with shape ``(n_diagrams, resolution)``; optionally
        paired with the grid.
    """
    dgm_list = as_diagram_list(diagrams)
    grid = resolve_grid(dgm_list, sample_range, resolution, keep_endpoints)

    mode = weights
    explicit_weights = None
    if isinstance(weights, str):
        if weights not in {"power", "pi"}:
            raise ValueError("weights must be 'power', 'pi', or per-point arrays")
    else:
        mode = "explicit"
        explicit_weights = as_weight_list(weights, dgm_list)

    if mode == "pi":
        if pi is None:
            raise ValueError("pi must be supplied when weights='pi'")
        explicit_weights = as_weight_list(pi, dgm_list)

    out = np.zeros((len(dgm_list), grid.shape[0]), dtype=float)
    for i, diagram in enumerate(dgm_list):
        if diagram.shape[0] == 0:
            continue

        if mode == "power":
            w = _power_weights(diagram, r)
        else:
            w = explicit_weights[i]

        if np.any(~np.isfinite(w)):
            raise ValueError("silhouette weights must be finite")
        if np.any(w < 0):
            raise ValueError("silhouette weights must be non-negative")

        total = float(np.sum(w))
        if total <= 0.0:
            if zero_policy == "zero":
                continue
            if zero_policy == "uniform":
                w = np.ones(diagram.shape[0], dtype=float)
                total = float(diagram.shape[0])
            elif zero_policy == "raise":
                raise ValueError("diagram has no positive silhouette weight")
            else:
                raise ValueError("zero_policy must be 'zero', 'uniform', or 'raise'")

        tents = diagram_tents(diagram, grid)
        out[i] = np.sqrt(2.0) * (tents @ (w / total))

    if return_grid:
        return out, grid
    return out
