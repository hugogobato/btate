"""Tests for MCMC convergence diagnostics (split-Rhat, ESS)."""
import numpy as np

from btate.partitions.diagnostics import (
    effective_sample_size,
    potential_scale_reduction,
)


def test_rhat_near_one_for_iid_chains():
    rng = np.random.default_rng(0)
    chains = rng.standard_normal((4, 2000))
    rhat = potential_scale_reduction(chains)
    assert 0.98 < rhat < 1.05


def test_rhat_detects_nonconvergence():
    rng = np.random.default_rng(1)
    # four chains stuck at very different locations -> large between-chain var.
    offsets = np.array([0.0, 5.0, 10.0, 15.0])[:, None]
    chains = 0.1 * rng.standard_normal((4, 500)) + offsets
    assert potential_scale_reduction(chains) > 2.0


def test_rhat_constant_is_nan():
    assert np.isnan(potential_scale_reduction(np.ones((3, 100))))


def test_ess_iid_close_to_n():
    rng = np.random.default_rng(2)
    n = 4000
    chains = rng.standard_normal((4, n))
    ess = effective_sample_size(chains)
    # iid: ESS should be a large fraction of the 16000 total draws.
    assert ess > 0.6 * 4 * n


def test_ess_autocorrelated_is_small():
    rng = np.random.default_rng(3)
    n = 4000
    # AR(1) with rho=0.9 -> ESS ~ N * (1-rho)/(1+rho) = N/19.
    x = np.zeros((2, n))
    for t in range(1, n):
        x[:, t] = 0.9 * x[:, t - 1] + rng.standard_normal(2)
    iid = effective_sample_size(rng.standard_normal((2, n)))
    ar = effective_sample_size(x)
    assert ar < iid
    assert ar < 0.3 * 2 * n


def test_ess_capped_at_total():
    rng = np.random.default_rng(4)
    chains = rng.standard_normal((2, 500))
    assert effective_sample_size(chains) <= 2 * 500 + 1e-6
