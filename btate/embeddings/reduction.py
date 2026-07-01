"""Dimension reduction for posterior functional summaries (Task 2.4)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_curve_matrix(curves) -> np.ndarray:
    arr = np.asarray(curves, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim < 2:
        raise ValueError("curves must be at least 2-D")
    return arr.reshape(arr.shape[0], -1)


@dataclass
class FPCAModel:
    """Discrete fPCA model based on an SVD of centered grid curves."""

    mean_: np.ndarray
    components_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray
    original_shape_: tuple[int, ...]

    def transform(self, curves) -> np.ndarray:
        mat = _as_curve_matrix(curves)
        if mat.shape[1] != self.mean_.shape[0]:
            raise ValueError("curve grid size does not match fitted fPCA model")
        return (mat - self.mean_) @ self.components_.T

    def inverse_transform(self, scores) -> np.ndarray:
        score_mat = np.asarray(scores, dtype=float)
        if score_mat.ndim == 1:
            score_mat = score_mat[None, :]
        rec = score_mat @ self.components_ + self.mean_
        return rec.reshape((score_mat.shape[0],) + self.original_shape_)


def fit_fpca(curves, n_components: int = 5, center: bool = True) -> FPCAModel:
    """Fit a dependency-light discrete fPCA/SVD reduction.

    This is the Phase-2 object that can flow into Phase 3 when a full
    function-on-grid GP is too expensive.  It accepts silhouettes
    ``(n, resolution)`` or flattened/higher-order embeddings such as landscapes
    ``(n, k, resolution)``.
    """
    mat = _as_curve_matrix(curves)
    if mat.shape[0] < 1:
        raise ValueError("need at least one curve")
    if int(n_components) < 1:
        raise ValueError("n_components must be positive")
    n_comp = min(int(n_components), mat.shape[0], mat.shape[1])

    mean = mat.mean(axis=0) if center else np.zeros(mat.shape[1], dtype=float)
    centered = mat - mean
    _, svals, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_comp]
    denom = max(mat.shape[0] - 1, 1)
    explained = (svals[:n_comp] ** 2) / denom
    total = float(np.sum(svals ** 2) / denom)
    ratio = explained / total if total > 0.0 else np.zeros_like(explained)

    return FPCAModel(
        mean_=mean,
        components_=components,
        explained_variance_=explained,
        explained_variance_ratio_=ratio,
        original_shape_=np.asarray(curves).shape[1:],
    )


@dataclass
class FourierProjection:
    """Least-squares projection on a real Fourier basis over the grid domain."""

    coefficients_: np.ndarray
    basis_: np.ndarray
    grid_: np.ndarray
    original_shape_: tuple[int, ...]

    def reconstruct(self) -> np.ndarray:
        rec = self.coefficients_ @ self.basis_.T
        return rec.reshape((self.coefficients_.shape[0],) + self.original_shape_)


def fourier_basis(grid, n_basis: int) -> np.ndarray:
    """Build a real Fourier design matrix on ``grid`` with ``n_basis`` columns."""
    x = np.asarray(grid, dtype=float).ravel()
    if x.ndim != 1 or x.shape[0] < 2:
        raise ValueError("grid must contain at least two points")
    if int(n_basis) < 1:
        raise ValueError("n_basis must be positive")
    n_basis = int(n_basis)
    lo, hi = float(x[0]), float(x[-1])
    if hi <= lo:
        raise ValueError("grid upper endpoint must exceed lower endpoint")
    z = 2.0 * np.pi * (x - lo) / (hi - lo)
    cols = [np.ones_like(x)]
    harmonic = 1
    while len(cols) < n_basis:
        cols.append(np.sin(harmonic * z))
        if len(cols) < n_basis:
            cols.append(np.cos(harmonic * z))
        harmonic += 1
    return np.column_stack(cols[:n_basis])


def project_fourier(curves, grid, n_basis: int = 7) -> FourierProjection:
    """Project grid curves onto a low-dimensional Fourier basis."""
    arr = np.asarray(curves, dtype=float)
    mat = _as_curve_matrix(arr)
    basis = fourier_basis(grid, n_basis=n_basis)
    if mat.shape[1] != basis.shape[0]:
        raise ValueError("curve grid size must match Fourier grid length")
    coeff, _, _, _ = np.linalg.lstsq(basis, mat.T, rcond=None)
    return FourierProjection(
        coefficients_=coeff.T,
        basis_=basis,
        grid_=np.asarray(grid, dtype=float).ravel(),
        original_shape_=arr.shape[1:],
    )
