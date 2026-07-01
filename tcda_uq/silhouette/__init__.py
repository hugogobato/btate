"""Persistence -> silhouette functionals.

Three entry points, matching the three data modalities in ``top-causal-effect-main``:
  * :func:`compute_silhouette`          -- from precomputed persistence diagrams
  * :func:`silhouette_from_pointcloud`  -- Alpha-complex filtration (ORBIT)
  * :func:`silhouette_from_image`       -- cubical (lower-star) filtration (SARS-CoV-2 CT)
"""

from .core import (
    power_weight,
    compute_silhouette,
    silhouette_from_pointcloud,
    silhouette_from_image,
)

__all__ = [
    "power_weight",
    "compute_silhouette",
    "silhouette_from_pointcloud",
    "silhouette_from_image",
]
