"""Covariate + treatment DGP (ported from top-causal-effect utils/generate.py).

Two-component Gaussian covariates and a logistic-with-interactions treatment
mechanism, shared by the ORBIT and SARS-CoV-2 experiments.
"""

from __future__ import annotations

import numpy as np


def gen_covariate(mu1, mu2, Sigma, n, rng=None):
    """Two Gaussian subgroups (balanced). Returns ``(cov1, cov2, covariate)``."""
    rng = np.random.default_rng(rng)
    size = n // 2
    cov1 = rng.multivariate_normal(mean=mu1, cov=Sigma, size=size)
    cov2 = rng.multivariate_normal(mean=mu2, cov=Sigma, size=n - size)
    return cov1, cov2, np.concatenate([cov1, cov2], axis=0)


def gen_trt_prob(covariate, beta, rng=None):
    """Logistic treatment with two interaction terms. Returns ``(prob, A)``."""
    rng = np.random.default_rng(rng)
    y = (
        covariate @ beta
        + 0.5 * covariate[:, 1] * covariate[:, 2]
        - 0.7 * covariate[:, 0] * covariate[:, 2]
    )
    prob = 1.0 / (1.0 + np.exp(-y))
    A = rng.binomial(n=1, p=prob)
    return prob, A
