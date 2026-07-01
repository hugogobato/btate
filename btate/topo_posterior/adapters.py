"""Persistence-diagram coordinate adapters.

Two conventions are used across the repository:

* **birth--death** ``(b, d)`` with ``d >= b`` — used by ``gudhi`` and the
  ``top-causal-effect`` (TATE) pipeline (see ``utils/pht.py``,
  ``utils/silhouette.py``).
* **birth--persistence** ``(b, p)`` with ``p = d - b >= 0`` (the "tilted"
  coordinates) — used by ``bayes_tda`` (see ``bayes_tda.intensities``,
  ``RestrictedGaussian(min_birth=...)``).

These helpers convert between the two so that TATE-format diagrams can be fed
to the Maroulas posterior and posterior samples can be fed back into the
silhouette / landscape transforms.  See ``docs/notation.md``.
"""
from __future__ import annotations

import numpy as np


def _as_2col(diagram) -> np.ndarray:
    d = np.asarray(diagram, dtype=float)
    if d.size == 0:
        return np.empty((0, 2), dtype=float)
    d = np.atleast_2d(d)
    if d.shape[1] != 2:
        raise ValueError(f"expected a (n, 2) diagram, got shape {d.shape}")
    return d


def bd_to_bp(diagram) -> np.ndarray:
    """Birth--death ``(b, d)`` -> birth--persistence ``(b, d - b)``.

    Converts a TATE/gudhi diagram into the ``bayes_tda`` tilted convention.
    """
    d = _as_2col(diagram)
    if d.shape[0] == 0:
        return d
    return np.column_stack([d[:, 0], d[:, 1] - d[:, 0]])


def bp_to_bd(diagram) -> np.ndarray:
    """Birth--persistence ``(b, p)`` -> birth--death ``(b, b + p)``.

    Inverse of :func:`bd_to_bp`; maps a ``bayes_tda`` posterior sample back to
    the TATE/gudhi birth--death convention consumed by the silhouette transform.
    """
    d = _as_2col(diagram)
    if d.shape[0] == 0:
        return d
    return np.column_stack([d[:, 0], d[:, 0] + d[:, 1]])


def lifetimes(diagram, convention: str = "bd") -> np.ndarray:
    """Return feature lifetimes ``l = d - b`` for a diagram.

    Parameters
    ----------
    diagram : array-like, shape (n, 2)
    convention : {"bd", "bp"}
        Whether ``diagram`` is in birth--death (``"bd"``) or birth--persistence
        (``"bp"``) coordinates.  In birth--persistence coordinates the second
        column *is* the lifetime.
    """
    d = _as_2col(diagram)
    if d.shape[0] == 0:
        return np.empty((0,), dtype=float)
    if convention == "bd":
        return d[:, 1] - d[:, 0]
    if convention == "bp":
        return d[:, 1].copy()
    raise ValueError("convention must be 'bd' or 'bp'")
