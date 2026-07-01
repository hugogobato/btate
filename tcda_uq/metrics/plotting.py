"""Band plotting helper."""

from __future__ import annotations

import numpy as np


def plot_band(band, truth=None, ax=None, label="band", color="C0", truth_label="truth"):
    """Plot a :class:`~tcda_uq.metrics.bands.Band` with an optional truth curve.

    Returns the matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    t = band.tseq
    ax.fill_between(t, band.lower, band.upper, alpha=0.25, color=color, label=f"{label} band")
    if band.center is not None:
        ax.plot(t, band.center, color=color, lw=2, label=f"{label} estimate")
    if truth is not None:
        ax.plot(t, np.asarray(truth), color="k", lw=2, ls="--", label=truth_label)
    ax.set_xlabel("t")
    ax.legend()
    return ax
