r"""MCMC convergence diagnostics — split-:math:`\widehat R` and effective sample
size (ESS) — for the Phase-1 partition sampler (Research_Plan Task 1.4).

Pure ``numpy``.  The estimators follow Vehtari, Gelman, Simpson, Carpenter &
Bürkner (2021), *Rank-normalization, folding, and localization: An improved
:math:`\widehat R` for assessing convergence of MCMC* — the same conventions
used by Stan / ArviZ — but without the (optional) rank normalization, which is
unnecessary for the roughly-Gaussian scalar summaries (block count ``k``,
signal size ``beta``, total mass ``theta``) tracked here.

Both functions take ``chains`` of shape ``(n_chains, n_draws)``.
"""
from __future__ import annotations

import numpy as np


def _as_chains(chains) -> np.ndarray:
    a = np.asarray(chains, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"chains must be (n_chains, n_draws); got shape {a.shape}")
    return a


def potential_scale_reduction(chains) -> float:
    r"""Split-:math:`\widehat R` (Gelman--Rubin) for ``chains`` shape
    ``(n_chains, n_draws)``.

    Each chain is split in half (doubling the chain count) to detect
    within-chain non-stationarity.  Returns ``nan`` if there are too few draws
    or the summary is constant (zero within-chain variance).
    """
    a = _as_chains(chains)
    n_chains, n_draws = a.shape
    half = n_draws // 2
    if half < 2:
        return float("nan")
    split = np.concatenate([a[:, :half], a[:, half:2 * half]], axis=0)
    m, n = split.shape
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    W = chain_vars.mean()
    if W <= 0:
        return float("nan")
    B = n * chain_means.var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def _autocorr_fft(x: np.ndarray) -> np.ndarray:
    """Biased autocorrelation of a 1-D series via FFT (lags ``0..n-1``)."""
    n = x.shape[0]
    x = x - x.mean()
    size = 1
    while size < 2 * n:
        size *= 2
    f = np.fft.rfft(x, n=size)
    acov = np.fft.irfft(f * np.conjugate(f), n=size)[:n]
    acov /= n
    if acov[0] == 0:
        return np.zeros_like(acov)
    return acov / acov[0]


def effective_sample_size(chains) -> float:
    r"""Multi-chain effective sample size for ``chains`` shape
    ``(n_chains, n_draws)``.

    Combines the within-chain autocorrelations with the between-chain variance
    (Stan / ArviZ ``ess_bulk`` without rank normalization) and truncates the
    autocorrelation sum with Geyer's initial monotone-sequence rule.  Returns
    ``nan`` for a constant summary and is capped at ``n_chains * n_draws``.
    """
    a = _as_chains(chains)
    n_chains, n_draws = a.shape
    if n_draws < 4:
        return float("nan")

    chain_means = a.mean(axis=1)
    chain_vars = a.var(axis=1, ddof=1)
    W = chain_vars.mean()
    if W <= 0:
        return float("nan")

    if n_chains > 1:
        B = n_draws * chain_means.var(ddof=1)
        var_plus = (n_draws - 1) / n_draws * W + B / n_draws
    else:
        var_plus = W

    # Mean (over chains) normalized autocorrelation at each lag.
    rho_hat = np.zeros(n_draws)
    acf = np.array([_autocorr_fft(a[c]) for c in range(n_chains)])  # (n_chains, n_draws)
    mean_var = chain_vars.mean()
    for t in range(n_draws):
        # per-chain autocovariance = acf * chain_var; combine as in Stan.
        acov_t = (acf[:, t] * chain_vars).mean()
        rho_hat[t] = 1.0 - (W - acov_t) / var_plus

    # Geyer initial monotone sequence: sum paired autocorrelations while positive.
    rho_hat[0] = 1.0
    tau = 1.0
    t = 1
    while t + 1 < n_draws:
        pair = rho_hat[t] + rho_hat[t + 1]
        if pair < 0:
            break
        tau += 2.0 * pair
        t += 2
    tau = max(tau, 1.0)

    ess = n_chains * n_draws / tau
    return float(min(ess, n_chains * n_draws))
