"""ORBIT synthetic testbed (linked-twist-map orbits), ported to numpy.

The original ``ORBIT/generate_data.py`` uses torch only trivially; this port is
pure numpy and vectorised across orbits, so the core install needs no torch.
RNG is numpy's ``Generator`` (not torch), so orbits are not bit-identical to the
paper's ``data.pt`` -- but the causal construction and downstream estimation
(and hence the AIPW-recovers-true-effect check) reproduce faithfully.
"""

from __future__ import annotations

import numpy as np

from ..silhouette import silhouette_from_pointcloud


def gen_orbits(rhos=(2.5, 3.5, 4.0, 4.1, 4.3), num_pts=1000, num_orbits_each=1000, rng=None):
    """Generate the ORBIT dataset, vectorised across all orbits.

    Returns:
        X: ``[len(rhos) * num_orbits_each, num_pts, 2]`` orbits.
        y: ``[len(rhos) * num_orbits_each]`` integer labels (index into ``rhos``).
    """
    rng = np.random.default_rng(rng)
    rhos = np.asarray(rhos, dtype=float)
    n_labels = len(rhos)
    M = n_labels * num_orbits_each

    rho = np.repeat(rhos, num_orbits_each)           # [M]
    x = rng.random(M)
    y = rng.random(M)
    X = np.empty((M, num_pts, 2), dtype=float)
    for i in range(num_pts):
        x = (x + rho * y * (1 - y)) % 1
        y = (y + rho * x * (1 - x)) % 1              # uses the updated x, as in original
        X[:, i, 0] = x
        X[:, i, 1] = y

    labels = np.repeat(np.arange(n_labels), num_orbits_each)
    return X, labels


def make_orbit_causal(
    n=1000,
    rho_labels=(1, 2, 3),        # indices into the default rhos -> r = 3.5, 4.0, 4.1
    p_reverse=0.3,
    interval=(0.0, 0.2),
    r=3.0,
    resolution=100,
    homology_dims=(0, 1),
    num_pts=1000,
    rng=None,
    orbits=None,
):
    """Build the ORBIT counterfactual dataset (notebook ``main.ipynb`` construction).

    Each unit picks two of the three orbit populations; the smaller-``r`` orbit
    is assigned to the control potential outcome, reversed with prob ``p_reverse``.
    Silhouettes of both potential outcomes are computed, giving the oracle
    counterfactual pair ``(phi^0, phi^1)``.

    Returns a dict with ``potential_outcomes`` ``[n, 2, n_hom, res]``,
    ``true_effect`` ``[n_hom, res]``, and ``tseq``.
    """
    rng = np.random.default_rng(rng)
    tseq = np.linspace(interval[0], interval[1], resolution)

    if orbits is None:
        X, labels = gen_orbits(num_pts=num_pts, num_orbits_each=n, rng=rng)
    else:
        X, labels = orbits

    # gather n orbits from each requested population
    pops = [X[labels == lab][:n] for lab in rho_labels]  # each [n, num_pts, 2]
    pops = np.stack(pops)                                 # [n_pop, n, num_pts, 2]

    potential_ctrl, potential_trt = [], []
    for i in range(n):
        idx = sorted(rng.choice(len(rho_labels), size=2, replace=False))
        if rng.random() <= p_reverse:
            idx = idx[::-1]
        x_ctrl = pops[idx[0], i]
        x_trt = pops[idx[1], i]
        potential_ctrl.append(
            silhouette_from_pointcloud(x_ctrl, interval, r, resolution, homology_dims)
        )
        potential_trt.append(
            silhouette_from_pointcloud(x_trt, interval, r, resolution, homology_dims)
        )

    potential_ctrl = np.stack(potential_ctrl)   # [n, n_hom, res]
    potential_trt = np.stack(potential_trt)
    potential_outcomes = np.stack([potential_ctrl, potential_trt], axis=1)  # [n,2,hom,res]
    true_effect = potential_trt.mean(0) - potential_ctrl.mean(0)            # [hom, res]

    return {
        "potential_outcomes": potential_outcomes,
        "true_effect": true_effect,
        "tseq": tseq,
    }
