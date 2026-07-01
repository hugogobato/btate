"""Persistence silhouettes.

Ported from ``top-causal-effect-main/utils/silhouette.py`` and the two
notebook-local variants (ORBIT alpha-complex, SARS-CoV-2 cubical complex),
unified behind three functions.
"""

from __future__ import annotations

import numpy as np
from gudhi.representations import Silhouette


def power_weight(point, r: float = 3.0) -> float:
    """Power weighting of a persistence point: ``|death - birth|**r``."""
    birth, death = point
    return np.abs(death - birth) ** r


def compute_silhouette(diags, interval=(0.0, 0.2), r: float = 3.0, resolution: int = 100):
    """Power-weighted silhouette of a list of persistence diagrams.

    Args:
        diags: list of ``(k, 2)`` arrays, one persistence diagram per homology dim.
        interval: sample range ``[t_min, t_max]`` of the silhouette.
        r: power-weight exponent (Phase 6.3 studies width/coverage vs ``r``).
        resolution: number of grid points.

    Returns:
        ``(n_hom_dim, resolution)`` array of silhouette values.
    """
    silhouette = Silhouette(
        weight=lambda x: power_weight(x, r),
        resolution=resolution,
        sample_range=list(interval),
        keep_endpoints=True,
    )
    return silhouette.fit_transform(diags)


def silhouette_from_pointcloud(
    points,
    interval=(0.0, 0.2),
    r: float = 3.0,
    resolution: int = 100,
    homology_dims=(0, 1),
):
    """Alpha-complex silhouette of a point cloud (ORBIT modality).

    Builds an alpha complex, extracts persistence, converts squared alpha
    radii to radii (``sqrt``), and returns the power-weighted silhouette.

    Args:
        points: ``(m, d)`` point cloud.
        homology_dims: homology dimensions to keep (default H0, H1).
    """
    import gudhi as gd

    points = np.asarray(points, dtype=float)
    alpha = gd.AlphaComplex(points=points)
    st = alpha.create_simplex_tree()
    st.compute_persistence()

    diags = []
    for hom_dim in homology_dims:
        diag = st.persistence_intervals_in_dimension(hom_dim)
        if hom_dim == 0:  # drop the essential (infinite) class
            diag = diag[~np.isinf(diag).any(axis=1)]
        # alpha filtration values are squared radii -> use radius
        diag = np.sqrt(diag)
        diags.append(diag)
    return compute_silhouette(diags, interval=interval, r=r, resolution=resolution)


def silhouette_from_image(
    image,
    interval=(0.0, 1.0),
    r: float = 0.1,
    resolution: int = 100,
    homology_dims=(0,),
    max_filtration=None,
):
    """Cubical (sub-level / lower-star) silhouette of a grayscale image (SARS-CoV-2 CT).

    Args:
        image: 2-D array of pixel intensities (e.g. normalised to ``[0, 1]``).
        homology_dims: homology dimensions to keep (default H0 only).
        max_filtration: if given, infinite death times are replaced by this
            value (only relevant when keeping dims where classes never die).
    """
    import gudhi as gd

    image = np.asarray(image, dtype=float)
    cub = gd.CubicalComplex(vertices=image)
    cub.compute_persistence()

    diags = []
    for hom_dim in homology_dims:
        diag = cub.persistence_intervals_in_dimension(hom_dim)
        inf_mask = np.isinf(diag).any(axis=1) if diag.size else np.zeros(0, dtype=bool)
        if hom_dim == 0 or max_filtration is None:
            diag = diag[~inf_mask]  # drop essential class
        else:
            diag = diag.copy()
            diag[np.isinf(diag)] = max_filtration
        diags.append(diag)
    return compute_silhouette(diags, interval=interval, r=r, resolution=resolution)
