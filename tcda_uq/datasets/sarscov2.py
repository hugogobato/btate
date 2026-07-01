"""SARS-CoV-2 CT-scan modality (0-dim cubical / lower-star filtration).

Requires the ``[data]`` extra (``kagglehub``, ``pillow``). Faithful to
``SARS_COV2/main.ipynb``: grayscale CT images -> H0 cubical silhouettes; the
causal dataset mixes infected / non-infected images to form potential outcomes.
"""

from __future__ import annotations

import os

import numpy as np

from ..silhouette import silhouette_from_image

_KAGGLE_SLUG = "plameneduardo/sarscov2-ctscan-dataset"


def download_ct(cache_dir="./") -> str:
    """Download the SARS-CoV-2 CT dataset via kagglehub; return its root path."""
    import kagglehub

    os.environ.setdefault("KAGGLEHUB_CACHE", cache_dir)
    return kagglehub.dataset_download(_KAGGLE_SLUG)


def load_ct_images(root_dir):
    """Load COVID / non-COVID grayscale images (first channel, normalised to [0,1])."""
    from PIL import Image

    def _load(folder):
        imgs = []
        for f in os.listdir(folder):
            arr = np.array(Image.open(os.path.join(folder, f))) / 255.0
            imgs.append(arr[:, :, 0] if arr.ndim == 3 else arr)
        return imgs

    infected = _load(os.path.join(root_dir, "COVID"))
    noninfected = _load(os.path.join(root_dir, "non-COVID"))
    return infected, noninfected


def images_to_silhouettes(images, interval=(0.0, 1.0), r=0.1, resolution=100):
    """H0 cubical silhouettes for a list of images: ``[n, 1, resolution]``."""
    return np.stack(
        [silhouette_from_image(im, interval, r, resolution, homology_dims=(0,)) for im in images]
    )


def make_sarscov2_causal(
    phi_infected,
    phi_noninfected,
    n=500,
    p=0.75,
    rng=None,
):
    """Build potential outcomes by mixing infected / non-infected silhouettes.

    Mirrors the notebook: control = infected; treated = a ``p`` / ``2-p`` mix of
    non-infected and infected, giving a non-trivial average effect.
    """
    rng = np.random.default_rng(rng)
    # treated arm: fraction p non-infected + (1-p) infected; control arm: infected.
    # (the notebook's int() truncation only balanced when n*p was integral).
    n_non = int(round(n * p))
    n_inf_trt = n - n_non
    idx_non = rng.choice(len(phi_noninfected), n_non)
    idx_inf = rng.choice(len(phi_infected), n + n_inf_trt)

    potential_trt = np.vstack([phi_noninfected[idx_non], phi_infected[idx_inf[n:]]])
    potential_ctrl = phi_infected[idx_inf[:n]]
    potential_outcomes = np.stack([potential_ctrl, potential_trt], axis=1)
    true_effect = potential_trt.mean(0) - potential_ctrl.mean(0)
    return {"potential_outcomes": potential_outcomes, "true_effect": true_effect}
