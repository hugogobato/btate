"""Coverage and width metrics for functional bands.

All functions take raw ``lower`` / ``upper`` arrays of shape ``[resolution]`` so
they work for any band source. ``target`` curves are ``[resolution]`` (single) or
``[n, resolution]`` (a batch, e.g. many ITTE draws).
"""

from __future__ import annotations

import numpy as np


def covers_simultaneous(lower, upper, target):
    """Whether ``target`` lies inside the band at **every** ``t``.

    Returns a bool for a single ``[res]`` target, or a bool array ``[n]`` for a
    batch ``[n, res]``. This is the correct notion for a simultaneous functional
    band (one miss anywhere = not covered).
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    target = np.asarray(target)
    inside = (target >= lower) & (target <= upper)
    return inside.all(axis=-1)


def simultaneous_coverage(lower, upper, targets):
    """Fraction of target curves fully contained in the band (simultaneous)."""
    covered = covers_simultaneous(lower, upper, np.atleast_2d(targets))
    return float(np.mean(covered))


def pointwise_coverage(lower, upper, targets):
    """Per-``t`` coverage fraction, array ``[resolution]`` (mean over target curves)."""
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    targets = np.atleast_2d(targets)
    inside = (targets >= lower) & (targets <= upper)
    return inside.mean(axis=0)


def mean_width(lower, upper):
    """Mean pointwise width of the band."""
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def interval_wise_error(lower, upper, target):
    """Fraction of the domain where a single ``target`` curve falls outside the band.

    The interval-wise error notion targeted by Pini-Vantini IWT (Phase 2.3):
    control of the *proportion* of the domain that is falsely excluded.
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    target = np.asarray(target)
    outside = (target < lower) | (target > upper)
    return float(np.mean(outside))
