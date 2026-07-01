"""Tri-oracle simulation harness (Phase 0.5).

A fully-specified generative model for silhouette treatment effects that exposes
ground truth for **all three** estimands, so both *confidence* coverage (TATE,
CTATE-mean) and *prediction* coverage (ITTE) can be checked:

  * TATE   psi_d(t)     = E[delta_{i,d}(t)]          -- marginal mean  (closed form)
  * CTATE  tau_d(t, x)  = E[delta_{i,d}(t) | X = x]  -- conditional mean (closed form)
  * ITTE   delta_{i,d}(t) = phi^1_{i,d}(t) - phi^0_{i,d}(t)  -- individual draw (realised)

Design (extends TATE Appendix D):
  * Covariates X in R^d from a two-component Gaussian mixture (subgroups).
  * Propensity pi(x) = expit(x @ beta + interactions).
  * Outcome means mu_{a,d}(t, x) are a Fourier-in-t function-on-scalar model with
    coefficient functions linear in x -- i.e. exactly the family that
    ``fit_functional_regression`` estimates, so nuisances are well specified.
  * phi^a_{i,d}(t) = mu_{a,d}(t, X_i) + eps^a_{i,d}(t), with eps a smooth mean-zero
    functional noise process. The eps carries the *aleatoric* spread that ITTE
    prediction bands must cover and that confidence bands must not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# defaults mirror ORBIT/main.ipynb's data-generating process
_MU1 = np.array([1.0, 0.6, -0.7, 2.2, -1.0])
_MU2 = np.array([0.4, -0.4, -0.6, 3.3, 3.0])
_BETA = np.array([-0.5, -0.1, 0.6, 0.1, 0.1])


def _fourier_basis(n_basis, tseq, interval):
    """Real Fourier basis ``[n_basis, resolution]``: [1, cos1, sin1, cos2, sin2, ...]."""
    t0, t1 = interval
    u = (np.asarray(tseq) - t0) / (t1 - t0)
    basis = [np.ones_like(u)]
    j = 1
    while len(basis) < n_basis:
        basis.append(np.cos(2 * np.pi * j * u))
        if len(basis) < n_basis:
            basis.append(np.sin(2 * np.pi * j * u))
        j += 1
    return np.stack(basis[:n_basis])


@dataclass
class SimulationSample:
    """One realised sample with all three oracle truth objects."""

    tseq: np.ndarray
    X: np.ndarray                 # [n, d]
    A: np.ndarray                 # [n]
    propensity: np.ndarray        # [n]  true pi(X)
    potential_outcomes: np.ndarray  # [n, 2, n_hom, res]  (phi^0, phi^1)
    oracle_itte: np.ndarray       # [n, n_hom, res]  delta_i = phi^1 - phi^0 (with noise)
    oracle_ctate: np.ndarray      # [n, n_hom, res]  tau(t, X_i)          (noiseless)
    oracle_tate: np.ndarray       # [n_hom, res]     psi(t)               (noiseless)

    @property
    def observed(self):
        """Observed triplet ``(phi, A, X)`` -- the factual silhouettes."""
        phi = self.potential_outcomes[np.arange(len(self.A)), self.A]
        return (phi, self.A, self.X)


class TriOracleSimulation:
    """Generative model exposing TATE / CTATE / ITTE oracles.

    Args:
        n_cov: covariate dimension.
        n_hom_dim: number of homology dimensions.
        resolution: silhouette grid size.
        interval: silhouette domain ``[t_min, t_max]``.
        n_basis: Fourier basis size of the outcome-mean model.
        mu1, mu2: subgroup mean vectors (length ``n_cov``).
        sigma2: isotropic within-subgroup variance.
        beta: propensity coefficients (length ``n_cov``).
        coef_scale, coef_decay: scale / spectral decay of the outcome coefficients.
        noise_scale: amplitude of the functional aleatoric noise.
        seed: seeds the *fixed* model coefficients (not the per-sample draws).
    """

    def __init__(
        self,
        n_cov: int = 5,
        n_hom_dim: int = 2,
        resolution: int = 100,
        interval=(0.0, 1.0),
        n_basis: int = 5,
        mu1=None,
        mu2=None,
        sigma2: float = 0.5,
        beta=None,
        coef_scale: float = 1.0,
        coef_decay: float = 0.5,
        noise_scale: float = 0.3,
        seed: int = 0,
    ):
        self.n_cov = n_cov
        self.n_hom_dim = n_hom_dim
        self.resolution = resolution
        self.interval = interval
        self.n_basis = n_basis
        self.noise_scale = noise_scale

        self.mu1 = np.asarray(mu1 if mu1 is not None else _MU1[:n_cov], dtype=float)
        self.mu2 = np.asarray(mu2 if mu2 is not None else _MU2[:n_cov], dtype=float)
        self.Sigma = np.eye(n_cov) * sigma2
        self.beta = np.asarray(beta if beta is not None else _BETA[:n_cov], dtype=float)

        self.tseq = np.linspace(interval[0], interval[1], resolution)
        self.Psi = _fourier_basis(n_basis, self.tseq, interval)      # [K, res]
        self.spectral = coef_decay ** np.arange(n_basis)            # smooth curves

        # fixed outcome-mean coefficients Gamma[a][d] of shape [K, n_cov + 1]
        init = np.random.default_rng(seed)
        self.Gamma = {0: [], 1: []}
        for a in (0, 1):
            for _ in range(n_hom_dim):
                C = init.normal(size=(n_basis, n_cov + 1)) * coef_scale
                C = C * self.spectral[:, None]
                self.Gamma[a].append(C)

        # E[X] under the balanced two-component mixture
        self.EX = 0.5 * (self.mu1 + self.mu2)

    # ------------------------------------------------------------------ means
    def _mean(self, a, X):
        """Outcome mean mu_{a,.}(t, X): ``[n, n_hom, res]``."""
        X = np.atleast_2d(X)
        design = np.hstack([np.ones((X.shape[0], 1)), X])           # [n, p+1]
        out = np.empty((X.shape[0], self.n_hom_dim, self.resolution))
        for d in range(self.n_hom_dim):
            out[:, d, :] = (design @ self.Gamma[a][d].T) @ self.Psi  # [n,K]@[K,res]
        return out

    def _propensity(self, X):
        y = (
            X @ self.beta
            + 0.5 * X[:, 1] * X[:, 2]
            - 0.7 * X[:, 0] * X[:, 2]
        )
        return 1.0 / (1.0 + np.exp(-y))

    # ------------------------------------------------------------- oracles
    def true_ctate(self, X):
        """CTATE tau_d(t, x) = mu_1 - mu_0 at each row of ``X``: ``[n, n_hom, res]``."""
        return self._mean(1, X) - self._mean(0, X)

    def true_tate(self):
        """TATE psi_d(t) = E_X[tau_d(t, X)]: ``[n_hom, res]`` (uses closed-form E[X])."""
        return self.true_ctate(self.EX[None, :])[0]

    # ------------------------------------------------------------- sampling
    def sample(self, n, rng=None) -> SimulationSample:
        rng = np.random.default_rng(rng)

        # covariates: balanced two-component Gaussian mixture
        n1 = n // 2
        X1 = rng.multivariate_normal(self.mu1, self.Sigma, size=n1)
        X2 = rng.multivariate_normal(self.mu2, self.Sigma, size=n - n1)
        X = np.vstack([X1, X2])

        # treatment
        pi = self._propensity(X)
        A = rng.binomial(1, pi)

        # potential-outcome silhouettes: mean + smooth functional noise
        mu0 = self._mean(0, X)
        mu1 = self._mean(1, X)
        eps0 = self._noise(n, rng)
        eps1 = self._noise(n, rng)
        phi0 = mu0 + eps0
        phi1 = mu1 + eps1
        potential_outcomes = np.stack([phi0, phi1], axis=1)         # [n, 2, hom, res]

        oracle_itte = phi1 - phi0                                   # realised individual
        oracle_ctate = mu1 - mu0                                    # E[delta | X_i]
        oracle_tate = self.true_tate()

        return SimulationSample(
            tseq=self.tseq,
            X=X,
            A=A,
            propensity=pi,
            potential_outcomes=potential_outcomes,
            oracle_itte=oracle_itte,
            oracle_ctate=oracle_ctate,
            oracle_tate=oracle_tate,
        )

    def _noise(self, n, rng):
        """Smooth mean-zero functional noise: ``[n, n_hom, res]``."""
        z = rng.normal(size=(n, self.n_hom_dim, self.n_basis))
        z = z * self.spectral[None, None, :]
        return self.noise_scale * (z @ self.Psi)                   # [n,hom,K]@[K,res]
