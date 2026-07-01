"""Step 3 — posterior functional embeddings (silhouettes & landscapes).

* :func:`weighted_silhouette` — Task 2.1: silhouette with pi_p (or fixed-r) weights.
* :func:`posterior_landscape` — Task 2.2: persistence-landscape transform.
* :func:`posterior_embedding_summary` — Task 2.3: posterior draws and bands.
* :func:`fit_fpca` / :func:`project_fourier` — Task 2.4: dimension reduction.
"""
from __future__ import annotations

from .silhouette import weighted_silhouette
from .landscape import landscape_distances, posterior_landscape
from .aggregation import (
    PosteriorFunctionalSummary,
    posterior_embedding_summary,
    summarize_posterior_functions,
)
from .reduction import FPCAModel, FourierProjection, fit_fpca, fourier_basis, project_fourier

__all__ = [
    "weighted_silhouette",
    "posterior_landscape",
    "landscape_distances",
    "PosteriorFunctionalSummary",
    "posterior_embedding_summary",
    "summarize_posterior_functions",
    "FPCAModel",
    "FourierProjection",
    "fit_fpca",
    "fourier_basis",
    "project_fourier",
]
