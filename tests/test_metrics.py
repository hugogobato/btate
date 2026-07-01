"""Tests for benchmark metric helpers."""
import numpy as np

from btate.benchmarks import (
    numerical_integration,
    l1_distance,
    rmse,
    pointwise_coverage,
    simultaneous_coverage,
    interval_width,
)


def test_numerical_integration_linear():
    t = np.linspace(0, 1, 101)
    # integral of f(t)=t over [0,1] is 0.5
    assert abs(numerical_integration(t, t) - 0.5) < 1e-6


def test_l1_distance_zero():
    t = np.linspace(0, 1, 50)
    f = np.sin(t)
    assert l1_distance(f, f, t) == 0.0


def test_rmse():
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert abs(rmse([0.0, 0.0], [1.0, 1.0]) - 1.0) < 1e-12


def test_coverage():
    truth = np.array([0.0, 0.5, 1.0])
    lower = np.array([-1.0, 0.0, 2.0])   # last point uncovered
    upper = np.array([1.0, 1.0, 3.0])
    assert abs(pointwise_coverage(lower, upper, truth) - 2 / 3) < 1e-12
    assert simultaneous_coverage(lower, upper, truth) == 0.0
    assert simultaneous_coverage(truth - 1, truth + 1, truth) == 1.0


def test_interval_width():
    lower = np.zeros(5)
    upper = np.ones(5) * 2.0
    assert interval_width(lower, upper) == 2.0
    t = np.linspace(0, 1, 5)
    assert abs(interval_width(lower, upper, t) - 2.0) < 1e-12
