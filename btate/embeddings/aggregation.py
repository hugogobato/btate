"""Posterior aggregation for Phase-2 functional embeddings."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .landscape import posterior_landscape
from .silhouette import weighted_silhouette


@dataclass
class PosteriorFunctionalSummary:
    """Posterior draws and credible bands for one functional summary.

    ``draws`` has shape ``(n_draws, ...)`` where the trailing dimensions are the
    functional object: ``(resolution,)`` for silhouettes or
    ``(num_landscapes, resolution)`` for landscapes.
    """

    draws: np.ndarray
    grid: np.ndarray
    mean: np.ndarray
    pointwise_lower: np.ndarray
    pointwise_upper: np.ndarray
    simultaneous_lower: np.ndarray
    simultaneous_upper: np.ndarray
    alpha: float
    simultaneous_radius: float


def summarize_posterior_functions(draws, grid=None, alpha: float = 0.05,
                                  eps: float = 1e-12) -> PosteriorFunctionalSummary:
    """Summarize posterior functional draws with pointwise and sup-norm bands.

    The simultaneous band is the posterior quantile of the maximum standardized
    absolute deviation across the discretized functional grid:
    ``mean +/- q_{1-alpha}(max |draw-mean|/sd) * sd``.
    """
    arr = np.asarray(draws, dtype=float)
    if arr.ndim < 2:
        raise ValueError("draws must have shape (n_draws, ...functional grid...)")
    if arr.shape[0] < 1:
        raise ValueError("need at least one posterior draw")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    mean = arr.mean(axis=0)
    lower = np.quantile(arr, alpha / 2.0, axis=0)
    upper = np.quantile(arr, 1.0 - alpha / 2.0, axis=0)

    if arr.shape[0] == 1:
        sd = np.zeros_like(mean)
        radius = 0.0
    else:
        sd = arr.std(axis=0, ddof=1)
        denom = np.where(sd > eps, sd, np.inf)
        max_std = np.max(np.abs(arr - mean) / denom, axis=tuple(range(1, arr.ndim)))
        if np.all(~np.isfinite(max_std)):
            radius = 0.0
        else:
            radius = float(np.quantile(max_std[np.isfinite(max_std)], 1.0 - alpha))

    sim_lower = mean - radius * sd
    sim_upper = mean + radius * sd
    if grid is None:
        grid = np.arange(arr.shape[-1], dtype=float)
    grid_arr = np.asarray(grid, dtype=float).ravel()
    if grid_arr.shape[0] != arr.shape[-1]:
        raise ValueError("grid length must match the last functional dimension")

    return PosteriorFunctionalSummary(
        draws=arr,
        grid=grid_arr,
        mean=mean,
        pointwise_lower=lower,
        pointwise_upper=upper,
        simultaneous_lower=sim_lower,
        simultaneous_upper=sim_upper,
        alpha=float(alpha),
        simultaneous_radius=radius,
    )


def posterior_embedding_summary(diagrams, embedding: str = "silhouette",
                                pi=None, sample_range=(0.0, 0.2),
                                resolution: int = 100, alpha: float = 0.05,
                                **kwargs) -> PosteriorFunctionalSummary:
    """Transform posterior diagram draws and summarize the functional posterior.

    Parameters
    ----------
    diagrams : list of array-like
        Posterior persistence-diagram draws for one subject/arm.
    embedding : {"silhouette", "landscape"}
        Functional embedding to compute.
    pi : array-like or list of array-like, optional
        Per-point signal probabilities for silhouette ``weights="pi"``.
    kwargs
        Forwarded to the selected embedding function.  For silhouettes, common
        options are ``weights`` and ``r``.  For landscapes, use
        ``num_landscapes``.
    """
    if embedding == "silhouette":
        weights = kwargs.pop("weights", "pi" if pi is not None else "power")
        curves, grid = weighted_silhouette(
            diagrams, weights=weights, pi=pi, sample_range=sample_range,
            resolution=resolution, return_grid=True, **kwargs,
        )
        return summarize_posterior_functions(curves, grid=grid, alpha=alpha)
    if embedding == "landscape":
        landscapes, grid = posterior_landscape(
            diagrams, sample_range=sample_range, resolution=resolution,
            return_grid=True, **kwargs,
        )
        return summarize_posterior_functions(landscapes, grid=grid, alpha=alpha)
    raise ValueError("embedding must be 'silhouette' or 'landscape'")
