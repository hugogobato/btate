"""Functional DR-learner for the CTATE tau_d(t, x) (Phase 3.2).

The CTATE

    tau_d(t, x) = E[ delta_{i,d}(t) | X = x ]      (conditional-mean silhouette effect)

is the *bridge rung* of the estimand ladder TATE -> CTATE -> ITTE (see
``RESEARCH_PLAN.md`` Phase 3). Its point estimator is a deliberately cheap,
off-the-shelf composition -- the **DR-learner** of Kennedy (2023) applied to the
*functional* setting -- and both ingredients already live in this library:

  * Stage 1 (pseudo-outcome).  The DR-learner regresses the doubly-robust
    pseudo-outcome on X.  For this estimand the pseudo-outcome is *exactly* the
    per-unit TATE efficient-influence-function score (Kim & Lee eq. 8) whose
    sample mean is the AIPW TATE -- i.e. ``cross_fit(...).scores[d]``, already
    cross-fitted (each unit scored by nuisances fit on the complementary fold).

  * Stage 2 (regression on X).  Those pseudo-outcome *curves* are regressed on X
    with a Fourier function-on-scalar model (the same family as
    ``fit_functional_regression``), giving tau_hat_d(t, x) at any x.

This is intentionally **not** marketed as a new estimator; its value is
structural (its *mean* is confidence territory -> :mod:`.uq.asymptotic.ctate_bands`;
its *individual-at-x* is prediction territory -> Phase 5). Honest uniform
inference over ``(t, x)`` is deliberately out of scope (a topology-agnostic
spin-off); see ``docs/phase3_ctate.md``.

Linear-smoother representation
------------------------------
With a linear-in-features second stage the estimator is a **linear smoother** of
the pseudo-outcome curves: for design row ``z(x) = [1, f(x)]`` and design matrix
``Z`` (rows ``z(X_i)``),

    tau_hat_d(., x) = sum_i a_i(x) * P psi_{i,d},     a(x) = pinv(Z)^T z(x),

where ``psi_{i,d}`` is unit ``i``'s pseudo-outcome curve and ``P`` is the
Fourier projection (the response smoothing). The weights ``a(x)`` (summing to 1
because ``Z`` carries an intercept) and the second-stage residual curves
``r_{i,d} = P psi_{i,d} - tau_hat_d(., X_i)`` are exactly what the multiplier-
bootstrap confidence band consumes. In the marginal case ``a_i = 1/n`` and the
construction reduces to the Phase-2 TATE band.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .nuisance import cross_fit, CrossFitResult


def _fourier_design(tseq, n_basis):
    """Real Fourier basis evaluated on ``tseq``: ``[resolution, n_basis]``.

    Columns are ``[1, cos1, sin1, cos2, sin2, ...]`` on the rescaled domain --
    the same family used by the tri-oracle DGP and ``FourierBasis``, so a
    linear-in-x second stage is well specified against that truth.
    """
    tseq = np.asarray(tseq, dtype=float)
    t0, t1 = tseq[0], tseq[-1]
    u = (tseq - t0) / (t1 - t0)
    cols = [np.ones_like(u)]
    j = 1
    while len(cols) < n_basis:
        cols.append(np.cos(2 * np.pi * j * u))
        if len(cols) < n_basis:
            cols.append(np.sin(2 * np.pi * j * u))
        j += 1
    return np.stack(cols[:n_basis], axis=1)      # [res, K]


@dataclass
class SecondStageFit:
    """Fitted second-stage function-on-scalar regression for one homology dim.

    Attributes:
        B: Fourier basis ``[resolution, n_basis]``.
        M: ``pinv(Z)`` ``[p+1, n]`` -- maps responses to coefficients and, via
            ``M.T @ z(x)``, unit weights ``a(x)``.
        gamma: second-stage coefficients ``[p+1, n_basis]`` (``M @ C``).
        coef: per-unit smoothed pseudo-outcome coefficients ``C`` ``[n, n_basis]``.
        resid: per-unit second-stage residual curves ``[n, resolution]``
            (smoothed pseudo-outcome minus its fitted conditional mean).
    """

    B: np.ndarray
    M: np.ndarray
    gamma: np.ndarray
    coef: np.ndarray
    resid: np.ndarray

    def predict_coef(self, Z_eval):
        """Predicted basis coefficients at design rows ``Z_eval`` ``[m, p+1]``."""
        return np.asarray(Z_eval, dtype=float) @ self.gamma          # [m, K]

    def predict_curve(self, Z_eval):
        """Predicted curves ``tau_hat(., x)`` at design rows ``Z_eval``: ``[m, res]``."""
        return self.predict_coef(Z_eval) @ self.B.T                  # [m, res]

    def weights(self, z):
        """Smoother weights ``a(x) = pinv(Z)^T z(x)`` for a single design row ``z``.

        ``tau_hat(., x) = sum_i a_i(x) * (smoothed pseudo-outcome_i)`` and the
        band's bootstrap process is ``sum_i a_i(x) xi_i resid_i(.)``.
        """
        return self.M.T @ np.asarray(z, dtype=float)                 # [n]


class CTATEDRLearner:
    """Functional DR-learner estimator of the CTATE tau_d(t, x) (Phase 3.2).

    Usage::

        learner = CTATEDRLearner(n_basis=5).fit(sample, tseq)
        tau = learner.predict(X_eval)          # [m, n_hom, res]

    ``sample`` is the observed triplet ``(phi, A, X)``. Stage-1 nuisances are
    cross-fitted internally (or a precomputed :class:`CrossFitResult` may be
    supplied via ``cross_fit_result=`` to avoid recomputation); the second stage
    is a Fourier function-on-scalar regression of the cross-fitted pseudo-outcome
    curves on ``feature_fn(X)`` (default: identity, i.e. linear in X).
    """

    def __init__(
        self,
        n_basis: int = 5,
        *,
        feature_fn: Optional[Callable] = None,
        stage1_n_basis: Optional[int] = None,
    ):
        self.n_basis = n_basis
        self.feature_fn = feature_fn if feature_fn is not None else (lambda X: X)
        # stage-1 outcome-regression basis (nuisance); defaults to the stage-2 size
        self.stage1_n_basis = stage1_n_basis if stage1_n_basis is not None else n_basis

    # ------------------------------------------------------------------ design
    def _design(self, X):
        """Second-stage design ``[1, feature_fn(X)]``: ``[n, p+1]``."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        feat = np.atleast_2d(self.feature_fn(X))
        if feat.shape[0] != X.shape[0]:      # feature_fn returned a single row
            feat = feat.T
        return np.hstack([np.ones((X.shape[0], 1)), feat])

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        sample,
        tseq,
        *,
        cross_fit_result: Optional[CrossFitResult] = None,
        **cross_fit_kwargs,
    ):
        phi, A, X = sample
        X = np.asarray(X, dtype=float)
        self.tseq = np.asarray(tseq, dtype=float)
        self.n_hom_dim = phi.shape[-2]

        if cross_fit_result is None:
            cross_fit_result = cross_fit(
                sample, tseq, n_basis=self.stage1_n_basis, **cross_fit_kwargs
            )
        self.cross_fit_result_ = cross_fit_result

        # align pseudo-outcomes (cross-fit reorders units) with covariates
        order = cross_fit_result.order
        self.X_ = X[order]                                   # [n, d]
        self.pseudo_ = cross_fit_result.scores               # list of [n, res]

        B = _fourier_design(self.tseq, self.n_basis)          # [res, K]
        Bpinv = np.linalg.pinv(B)                             # [K, res]
        Z = self._design(self.X_)                             # [n, p+1]
        M = np.linalg.pinv(Z)                                 # [p+1, n]

        self.stages_ = []
        for d in range(self.n_hom_dim):
            psi = self.pseudo_[d]                             # [n, res]
            C = psi @ Bpinv.T                                # [n, K] smoothed coeffs
            gamma = M @ C                                    # [p+1, K]
            Chat = Z @ gamma                                 # [n, K] fitted coeffs
            resid = (C - Chat) @ B.T                         # [n, res] residual curves
            self.stages_.append(
                SecondStageFit(B=B, M=M, gamma=gamma, coef=C, resid=resid)
            )
        return self

    # --------------------------------------------------------------- predict
    def predict(self, X_eval):
        """CTATE curves ``tau_hat_d(t, x)`` at rows of ``X_eval``.

        Returns ``[m, n_hom_dim, resolution]`` (drops to ``[n_hom_dim, res]`` for
        a single ``x`` passed as a 1-D array).
        """
        X_eval = np.asarray(X_eval, dtype=float)
        single = X_eval.ndim == 1
        Z = self._design(X_eval)
        out = np.stack([ss.predict_curve(Z) for ss in self.stages_], axis=1)
        return out[0] if single else out

    def predict_dim(self, X_eval, d):
        """CTATE curves for a single homology dim ``d``: ``[m, resolution]``."""
        Z = self._design(np.atleast_2d(np.asarray(X_eval, dtype=float)))
        return self.stages_[d].predict_curve(Z)

    def weights(self, x, d=0):
        """Smoother weights ``a(x)`` over the training units for homology dim ``d``."""
        z = self._design(np.atleast_2d(np.asarray(x, dtype=float)))[0]
        return self.stages_[d].weights(z)

    def residuals(self, d=0):
        """Second-stage residual curves ``[n, resolution]`` for homology dim ``d``."""
        return self.stages_[d].resid
