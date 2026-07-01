"""Tests for the restricted-random-partition signal/noise model (Task 1.3)."""
import numpy as np
import pytest

from btate.partitions import RestrictedPartitionModel, signal_probability
from btate.partitions.lifetime_model import _log_eppf, _signal_size, _edges_from_cuts


# --------------------------------------------------------------------------- #
# EPPF prior (Martínez Eq. 2) — closed-form checks.
# --------------------------------------------------------------------------- #
def test_eppf_matches_hand_computation_n3_theta1():
    # All 4 no-gaps partitions (compositions) of n=3, theta=1.
    # Pr: {123}=1/3, {1|23}=1/4, {12|3}=1/4, {1|2|3}=1/6.
    theta, n = 1.0, 3
    got = {
        (3,): np.exp(_log_eppf([3], theta, n)),
        (1, 2): np.exp(_log_eppf([1, 2], theta, n)),
        (2, 1): np.exp(_log_eppf([2, 1], theta, n)),
        (1, 1, 1): np.exp(_log_eppf([1, 1, 1], theta, n)),
    }
    assert got[(3,)] == pytest.approx(1 / 3)
    assert got[(1, 2)] == pytest.approx(1 / 4)
    assert got[(2, 1)] == pytest.approx(1 / 4)
    assert got[(1, 1, 1)] == pytest.approx(1 / 6)


def test_eppf_is_a_normalized_distribution():
    # Sum over all compositions of n must be 1 for any theta > 0.
    from itertools import product

    for theta in (0.5, 1.0, 3.0):
        for n in (3, 4, 5):
            total = 0.0
            for cuts in product([False, True], repeat=n - 1):
                sizes = np.diff(_edges_from_cuts(np.array(cuts), n))
                total += np.exp(_log_eppf(sizes, theta, n))
            assert total == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Signal recovery.
# --------------------------------------------------------------------------- #
def _signal_noise_lifetimes(seed=0, n_noise=60, signal=(0.8, 0.9, 1.0)):
    rng = np.random.default_rng(seed)
    noise = rng.lognormal(np.log(0.05), 0.3, size=n_noise)
    ell = np.concatenate([noise, np.asarray(signal)])
    rng.shuffle(ell)
    return ell


def test_recovers_known_signal_cardinality():
    ell = _signal_noise_lifetimes(seed=0)
    model = RestrictedPartitionModel().fit(
        ell, n_samples=800, burn_in=1500, n_chains=2, random_state=1
    )
    assert model.betti_number(q=0.1) == 3


def test_signal_probability_separates_signal_from_noise():
    ell = _signal_noise_lifetimes(seed=0)
    pi = signal_probability(ell, q=0.1, n_samples=800, burn_in=1500,
                            n_chains=2, random_state=1)
    order = np.argsort(ell)
    assert np.all(pi[order[-3:]] > 0.8)      # 3 largest -> signal
    assert np.all(pi[order[:40]] < 0.1)      # many smallest -> noise


def test_pi_p_monotone_in_lifetime():
    # signal is always a top-suffix -> pi_p is non-decreasing in lifetime.
    ell = _signal_noise_lifetimes(seed=2)
    model = RestrictedPartitionModel().fit(
        ell, n_samples=500, burn_in=1000, n_chains=2, random_state=3
    )
    pi = model.signal_probability(q=0.05)
    pi_sorted = pi[np.argsort(ell)]
    assert np.all(np.diff(pi_sorted) >= -1e-12)


def test_probabilities_in_unit_interval():
    ell = _signal_noise_lifetimes(seed=5)
    pi = signal_probability(ell, n_samples=400, burn_in=800, n_chains=1,
                            random_state=0)
    assert pi.min() >= 0.0 and pi.max() <= 1.0


# --------------------------------------------------------------------------- #
# Robustness: dropping / subsampling / diagram input.
# --------------------------------------------------------------------------- #
def test_non_finite_lifetimes_dropped_and_length_preserved():
    ell = np.array([np.inf, 0.05, 0.06, 0.9, -1.0, 0.04])
    with pytest.warns(RuntimeWarning):
        model = RestrictedPartitionModel().fit(
            ell, n_samples=300, burn_in=500, n_chains=1, random_state=0
        )
    pi = model.signal_probability(q=0.3)
    assert pi.shape == ell.shape
    assert pi[0] == 0.0 and pi[4] == 0.0     # inf and negative dropped -> 0


def test_max_points_subsampling():
    rng = np.random.default_rng(0)
    ell = np.concatenate([rng.lognormal(np.log(0.05), 0.3, 200), [1.0, 1.1]])
    model = RestrictedPartitionModel().fit(
        ell, n_samples=300, burn_in=600, n_chains=1, max_points=50, random_state=1
    )
    pi = model.signal_probability(q=0.1)
    assert pi.shape == ell.shape
    # the two big lifetimes survive subsampling and are flagged signal.
    assert pi[-1] > 0.5 and pi[-2] > 0.5


def test_signal_probability_from_diagram():
    # birth-death diagram: two long-lived loops + short-lived noise.
    rng = np.random.default_rng(0)
    b = rng.uniform(0, 0.1, 30)
    noise = np.column_stack([b, b + rng.uniform(0.01, 0.05, 30)])
    signal = np.array([[0.05, 0.9], [0.03, 1.0]])
    dgm = np.vstack([noise, signal])
    pi = signal_probability(dgm, convention="bd", q=0.1,
                            n_samples=400, burn_in=800, n_chains=1, random_state=0)
    assert pi.shape == (32,)
    assert pi[-1] > 0.5 and pi[-2] > 0.5


def test_diagnostics_report_keys():
    ell = _signal_noise_lifetimes(seed=0)
    model = RestrictedPartitionModel().fit(
        ell, n_samples=400, burn_in=800, n_chains=2, random_state=1
    )
    diag = model.diagnostics()
    for key in ("accept_rate", "rhat_k", "ess_k", "rhat_beta_signal",
                "ess_beta_signal", "rhat_theta"):
        assert key in diag
    assert 0.0 <= diag["accept_rate"] <= 1.0
