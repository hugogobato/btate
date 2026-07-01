"""Tests for birth-death <-> birth-persistence diagram adapters."""
import numpy as np
import pytest

from btate.topo_posterior.adapters import bd_to_bp, bp_to_bd, lifetimes


def test_bd_to_bp_basic():
    bd = np.array([[0.1, 0.5], [0.2, 0.9]])
    bp = bd_to_bp(bd)
    np.testing.assert_allclose(bp, np.array([[0.1, 0.4], [0.2, 0.7]]))


def test_round_trip():
    rng = np.random.default_rng(0)
    b = rng.uniform(0, 1, size=20)
    p = rng.uniform(0, 1, size=20)
    bd = np.column_stack([b, b + p])
    np.testing.assert_allclose(bp_to_bd(bd_to_bp(bd)), bd)


def test_empty_diagram():
    for fn in (bd_to_bp, bp_to_bd):
        out = fn(np.empty((0, 2)))
        assert out.shape == (0, 2)
    assert lifetimes(np.empty((0, 2))).shape == (0,)


def test_lifetimes_conventions():
    bd = np.array([[0.1, 0.5], [0.2, 0.9]])
    np.testing.assert_allclose(lifetimes(bd, "bd"), np.array([0.4, 0.7]))
    bp = bd_to_bp(bd)
    np.testing.assert_allclose(lifetimes(bp, "bp"), np.array([0.4, 0.7]))


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        bd_to_bp(np.zeros((3, 3)))
