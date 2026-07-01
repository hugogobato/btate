r"""Log-normal + restricted-random-partition lifetime model — Task 1.3 (Phase 1).

Implements Martínez (2024), *Bayesian Estimation of Topological Features of
Persistence Diagrams*, Bayesian Analysis 19(1) (``22-BA1341.pdf`` §3, MCMC in
``ba1341supp.pdf`` App. A).

Model (Eq. 1 of the paper), with lifetimes ``l_i = d_i - b_i`` **sorted
ascending** (``l_i <= l_{i+1}``):

    l_i | pi, phi ~ g(l_i | phi_j) 1(l_i in pi_j)   [ind]   i = 1..n
    phi_j | pi    ~ nu_0                             [iid]
    pi            ~ rho_0

* ``g`` is the **log-normal** density with parameter ``phi_j = (mu_j, sigma_j^2)``
  (equivalently ``y_i = ln l_i ~ N(mu_j, sigma_j^2)``);
* ``nu_0`` is the conjugate **normal-gamma** prior, hyperparameters ``(m, c, a, b)``:
  ``mu | tau ~ N(m, c / tau)``, ``tau ~ Ga(a, b)`` (rate ``b``);
* ``rho_0`` is a **restricted** (no-gaps) random partition — blocks are
  *consecutive* index ranges with ``max pi_j < min pi_{j+1}`` — with EPPF from a
  Dirichlet process of total mass ``theta`` (Eq. 2):

      Pr(pi) = C(n; n_1..n_k) theta^k / (k! (theta)_{n up}) prod_j Gamma(n_j).

Kernel parameters ``phi`` are integrated out analytically (normal-gamma
conjugacy) giving the block marginal likelihood (supplement Eq. 1); the
partition posterior ``p(pi | l) ~ rho_0(pi) L(l | pi)`` (Eq. 3) is explored with
a **split--merge** MCMC over no-gaps partitions.  Because a no-gaps partition of
the sorted lifetimes is exactly a *composition* of ``n`` — i.e. a choice of
"cut" gaps between consecutive items — the split/merge moves are cuts and
un-cuts of single gaps.  The total-mass ``theta`` is updated by the
Escobar & West (1995) auxiliary-variable step (supplement App. A).

The **topological signal** is the right-most (largest-lifetime) block(s); with
``kappa = min{ s : sum_{j>=s} #pi_j / n <= q }`` the Betti estimate is
``beta_h = #pi_kappa + ... + #pi_k`` (Eq. 4; add 1 for ``h = 0``).  The
project's estimand is the per-feature posterior **signal probability**
``pi_p in [0, 1]`` — the posterior probability that feature ``p`` belongs to the
signal, marginalized over the partition posterior.

Default hyperparameters are the paper's simulation settings:
``nu_0 = (m, c, a, b) = (0, 0.5, 1.1, 0.1)`` and ``theta`` prior ``(1.1, 0.1)``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln

from .diagnostics import effective_sample_size, potential_scale_reduction

_LOG_2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------- #
# Core densities (all in log space, all on log-lifetimes y_i = ln l_i).
# --------------------------------------------------------------------------- #
def _block_log_marginal_likelihood(size, s1, s2, m, c, a, b):
    """Log marginal likelihood of one block (supplement Eq. 1).

    ``size`` = ``n_j``; ``s1`` = ``sum y``; ``s2`` = ``sum y^2`` over the block.
    Kernel params integrated out under the normal-gamma prior ``(m, c, a, b)``.
    """
    mean = s1 / size
    ss = s2 - s1 * s1 / size  # sum of squared deviations S_{pi_j}
    denom = size * c + 1.0
    brace = 0.5 * ss + size * (mean - m) ** 2 / (2.0 * denom) + b
    return (
        a * np.log(b)
        + gammaln(size / 2.0 + a)
        - 0.5 * size * _LOG_2PI
        - 0.5 * np.log(denom)
        - gammaln(a)
        - (size / 2.0 + a) * np.log(brace)
    )


def _log_eppf(block_sizes, theta, n):
    """Log EPPF prior ``log rho_0(pi)`` (Eq. 2), for a no-gaps partition."""
    block_sizes = np.asarray(block_sizes, dtype=float)
    k = block_sizes.shape[0]
    # log[ n! / prod n_j! ] + k log theta - log k! - log (theta)_{n up} + sum log Gamma(n_j)
    #   with  sum log Gamma(n_j) - sum log n_j! = - sum log n_j.
    return (
        gammaln(n + 1.0)
        + k * np.log(theta)
        - gammaln(k + 1.0)
        - (gammaln(theta + n) - gammaln(theta))
        - np.sum(np.log(block_sizes))
    )


# --------------------------------------------------------------------------- #
# MCMC state helpers.  Partition == boolean ``cuts`` of length n-1 (cuts[i] True
# means a block boundary between sorted items i and i+1).  Sufficient statistics
# come from prefix sums so every block evaluation is O(1).
# --------------------------------------------------------------------------- #
def _edges_from_cuts(cuts, n):
    idx = np.nonzero(cuts)[0] + 1
    return np.concatenate(([0], idx, [n])).astype(int)


def _full_log_posterior(cuts, csum, csum2, theta, n, hp):
    edges = _edges_from_cuts(cuts, n)
    sizes = np.diff(edges)
    ll = 0.0
    for j in range(sizes.shape[0]):
        lo, hi = edges[j], edges[j + 1]
        ll += _block_log_marginal_likelihood(
            hi - lo, csum[hi] - csum[lo], csum2[hi] - csum2[lo], *hp
        )
    return _log_eppf(sizes, theta, n) + ll


def _block_ll(lo, hi, csum, csum2, hp):
    return _block_log_marginal_likelihood(
        hi - lo, csum[hi] - csum[lo], csum2[hi] - csum2[lo], *hp
    )


def _update_theta(theta, k, n, alpha_theta, beta_theta, rng):
    """Escobar & West (1995) auxiliary-variable update of DP total mass."""
    eta = rng.beta(theta + 1.0, n)
    log_eta = np.log(eta)
    w = alpha_theta + k - 1.0
    pi_mix = w / (w + n * (beta_theta - log_eta))
    rate = beta_theta - log_eta
    shape = alpha_theta + k if rng.random() < pi_mix else alpha_theta + k - 1.0
    return rng.gamma(shape, 1.0 / rate)


def _run_chain(y, hp, theta0, theta_prior, update_theta,
               n_samples, burn_in, thin, rng):
    """One split--merge chain.  Returns ``(cuts_samples, theta_samples,
    k_samples, logpost_samples, accept_rate)``."""
    n = y.shape[0]
    csum = np.concatenate(([0.0], np.cumsum(y)))
    csum2 = np.concatenate(([0.0], np.cumsum(y * y)))
    alpha_theta, beta_theta = theta_prior

    cuts = np.zeros(max(n - 1, 0), dtype=bool)  # start: single block (all noise)
    theta = float(theta0)

    total_iters = burn_in + n_samples * thin
    n_gaps = n - 1
    cuts_out = np.zeros((n_samples, max(n_gaps, 0)), dtype=bool)
    theta_out = np.zeros(n_samples)
    k_out = np.zeros(n_samples, dtype=int)
    lp_out = np.zeros(n_samples)
    n_accept = 0
    n_moves = 0
    store = 0

    for it in range(total_iters):
        if n_gaps > 0:
            k = 1 + int(cuts.sum())
            propose_split = rng.random() < 0.5
            if propose_split and (n - k) > 0:
                g = int(rng.choice(np.nonzero(~cuts)[0]))
                edges = _edges_from_cuts(cuts, n)
                bi = int(np.searchsorted(edges, g, side="right") - 1)
                lo, hi = edges[bi], edges[bi + 1]
                d_ll = (_block_ll(lo, g + 1, csum, csum2, hp)
                        + _block_ll(g + 1, hi, csum, csum2, hp)
                        - _block_ll(lo, hi, csum, csum2, hp))
                n_r, n_a, n_b = hi - lo, (g + 1) - lo, hi - (g + 1)
                d_eppf = (np.log(theta) - np.log(k + 1.0)
                          + np.log(n_r) - np.log(n_a) - np.log(n_b))
                log_ratio = d_ll + d_eppf + np.log((n - k) / k)
                n_moves += 1
                if np.log(rng.random()) < log_ratio:
                    cuts[g] = True
                    n_accept += 1
            elif (not propose_split) and k > 1:
                g = int(rng.choice(np.nonzero(cuts)[0]))
                edges = _edges_from_cuts(cuts, n)
                bi = int(np.searchsorted(edges, g, side="right") - 1)
                lo = edges[bi]
                hi = edges[bi + 2]           # edges[bi+1] == g+1 (the cut)
                d_ll = (_block_ll(lo, hi, csum, csum2, hp)
                        - _block_ll(lo, g + 1, csum, csum2, hp)
                        - _block_ll(g + 1, hi, csum, csum2, hp))
                n_a, n_b, n_r = (g + 1) - lo, hi - (g + 1), hi - lo
                d_eppf = (-np.log(theta) + np.log(k)
                          - np.log(n_r) + np.log(n_a) + np.log(n_b))
                log_ratio = d_ll + d_eppf + np.log((k - 1.0) / (n - k + 1.0))
                n_moves += 1
                if np.log(rng.random()) < log_ratio:
                    cuts[g] = False
                    n_accept += 1

        if update_theta:
            k = 1 + int(cuts.sum())
            theta = _update_theta(theta, k, n, alpha_theta, beta_theta, rng)

        if it >= burn_in and (it - burn_in) % thin == 0 and store < n_samples:
            cuts_out[store] = cuts
            theta_out[store] = theta
            k_out[store] = 1 + int(cuts.sum())
            lp_out[store] = _full_log_posterior(cuts, csum, csum2, theta, n, hp)
            store += 1

    accept_rate = n_accept / n_moves if n_moves else float("nan")
    return cuts_out, theta_out, k_out, lp_out, accept_rate


def _signal_size(cuts_row, n, q):
    """Number of largest-lifetime features classified as signal for one sample.

    ``kappa = min{ s : sum_{j>=s} n_j / n <= q }``; signal = blocks kappa..k, a
    top-suffix of ``q*n`` features made of *whole* blocks."""
    edges = _edges_from_cuts(cuts_row, n)
    sizes = np.diff(edges)
    thresh = q * n
    cum = 0
    for s in sizes[::-1]:          # accumulate whole blocks from the right
        if cum + s <= thresh:
            cum += int(s)
        else:
            break
    return cum


# --------------------------------------------------------------------------- #
# Public model.
# --------------------------------------------------------------------------- #
@dataclass
class PartitionPosterior:
    """Stored MCMC draws (partitions as ``cuts``) + diagnostics."""

    cuts: np.ndarray           # (n_total, n-1) bool, sorted-lifetime order
    theta: np.ndarray          # (n_total,)
    k: np.ndarray              # (n_total,) block counts
    logpost: np.ndarray        # (n_total,)
    order: np.ndarray          # argsort mapping original -> sorted
    sorted_lifetimes: np.ndarray
    n: int
    chain_ids: np.ndarray
    accept_rate: float
    dropped: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    n_original: int = 0


class RestrictedPartitionModel:
    """Restricted-random-partition outlier model for persistence lifetimes."""

    def __init__(self, m: float = 0.0, c: float = 0.5, a: float = 1.1,
                 b: float = 0.1, theta_prior=(1.1, 0.1)):
        # Normal-gamma prior nu_0 = (m, c, a, b); Dirichlet-process total mass theta.
        self.m, self.c, self.a, self.b = float(m), float(c), float(a), float(b)
        self.theta_prior = (float(theta_prior[0]), float(theta_prior[1]))
        self.posterior_: PartitionPosterior | None = None

    # -- fitting -------------------------------------------------------------
    def fit(self, lifetimes, n_samples: int = 5000, burn_in: int = 10000,
            thin: int = 1, n_chains: int = 4, update_theta: bool = True,
            theta_init: float | None = None, max_points: int | None = None,
            random_state=None, diag_q: float = 0.03):
        """Run split--merge MCMC over the restricted-partition posterior.

        Parameters
        ----------
        lifetimes : array-like, shape (n_original,)
            Feature lifetimes ``l = d - b > 0`` (any order).  Non-finite or
            non-positive values are dropped (a warning is issued); their signal
            probability is reported as 0.
        n_samples, burn_in, thin : int
            Retained draws, warm-up, and thinning *per chain*.
        n_chains : int
            Independent chains (for split-Rhat / ESS).
        update_theta : bool
            Update the DP total mass via Escobar & West; else hold it fixed.
        theta_init : float, optional
            Initial / fixed total mass (defaults to the prior mean
            ``alpha/beta``).
        max_points : int, optional
            If set and ``n > max_points``, keep only the ``max_points`` largest
            lifetimes (the smallest are certainly noise) and assign the dropped
            small ones ``pi_p = 0``.  Subsampling for dense diagrams
            (Research_Plan §4 mitigation).
        random_state : int | np.random.Generator, optional
        diag_q : float
            Outlier proportion used for the stored convergence diagnostics.
        """
        rng = np.random.default_rng(random_state)
        raw = np.asarray(lifetimes, dtype=float).ravel()
        n_original = raw.shape[0]

        good = np.isfinite(raw) & (raw > 0)
        dropped = np.nonzero(~good)[0]
        if dropped.size:
            warnings.warn(
                f"dropping {dropped.size} non-finite/non-positive lifetime(s) "
                "before fitting; their signal probability is set to 0.",
                RuntimeWarning, stacklevel=2,
            )
        kept_idx = np.nonzero(good)[0]
        vals = raw[kept_idx]

        # optional subsampling: keep the largest lifetimes (signal candidates).
        if max_points is not None and vals.shape[0] > max_points:
            keep_local = np.argsort(vals)[-max_points:]
            sub_dropped = np.setdiff1d(np.arange(vals.shape[0]), keep_local)
            dropped = np.concatenate([dropped, kept_idx[sub_dropped]])
            kept_idx = kept_idx[keep_local]
            vals = raw[kept_idx]

        n = vals.shape[0]
        if n < 1:
            raise ValueError("need at least one positive, finite lifetime")

        # sort ascending; keep mapping back to (kept) original positions.
        order_local = np.argsort(vals, kind="stable")
        sorted_vals = vals[order_local]
        order = kept_idx[order_local]      # sorted-position -> original index
        y = np.log(sorted_vals)

        hp = (self.m, self.c, self.a, self.b)
        theta0 = (theta_init if theta_init is not None
                  else self.theta_prior[0] / self.theta_prior[1])

        cuts_all, theta_all, k_all, lp_all, chain_ids = [], [], [], [], []
        accept_rates = []
        n_gaps = max(n - 1, 0)
        for ci in range(n_chains):
            child = np.random.default_rng(rng.integers(0, 2**63 - 1))
            cuts_c, theta_c, k_c, lp_c, acc = _run_chain(
                y, hp, theta0, self.theta_prior, update_theta,
                n_samples, burn_in, thin, child,
            )
            cuts_all.append(cuts_c.reshape(n_samples, n_gaps))
            theta_all.append(theta_c)
            k_all.append(k_c)
            lp_all.append(lp_c)
            chain_ids.append(np.full(n_samples, ci, dtype=int))
            accept_rates.append(acc)

        self.posterior_ = PartitionPosterior(
            cuts=np.vstack(cuts_all) if n_gaps else np.zeros((n_samples * n_chains, 0), bool),
            theta=np.concatenate(theta_all),
            k=np.concatenate(k_all),
            logpost=np.concatenate(lp_all),
            order=order,
            sorted_lifetimes=sorted_vals,
            n=n,
            chain_ids=np.concatenate(chain_ids),
            accept_rate=float(np.mean(accept_rates)),
            dropped=dropped,
            n_original=n_original,
        )
        self._n_chains = n_chains
        self._n_samples = n_samples
        self._diag_q = diag_q
        return self

    # -- estimands -----------------------------------------------------------
    def _require_fit(self) -> PartitionPosterior:
        if self.posterior_ is None:
            raise RuntimeError("call fit(...) before querying the posterior")
        return self.posterior_

    def signal_probability(self, q: float = 0.03) -> np.ndarray:
        """Per-feature posterior signal probability ``pi_p`` (marginal over pi).

        Returned in the **original** feature order (length ``n_original``);
        dropped features get ``pi_p = 0``.  ``q`` is the expected outlier
        proportion used to cut signal blocks (Eq. 4).
        """
        post = self._require_fit()
        n = post.n
        n_total = post.cuts.shape[0]
        # signal is a top-suffix of the sorted lifetimes -> accumulate a
        # per-sorted-position hit count, then map to original order.
        hits_sorted = np.zeros(n)
        for r in range(n_total):
            msize = _signal_size(post.cuts[r], n, q)
            if msize:
                hits_sorted[n - msize:] += 1.0
        pi_sorted = hits_sorted / n_total

        out = np.zeros(post.n_original if post.n_original else n)
        out[post.order] = pi_sorted
        return out

    def betti_distribution(self, q: float = 0.03, homology_degree: int | None = None):
        """Posterior draws of the Betti-number estimate ``beta_h`` (Eq. 4).

        Add 1 for ``homology_degree == 0`` (one connected component; assumes the
        infinite-lifetime feature was already dropped)."""
        post = self._require_fit()
        n = post.n
        draws = np.array([_signal_size(post.cuts[r], n, q)
                          for r in range(post.cuts.shape[0])], dtype=float)
        if homology_degree == 0:
            draws = draws + 1.0
        return draws

    def betti_number(self, q: float = 0.03, homology_degree: int | None = None) -> int:
        """Point estimate of ``beta_h`` = posterior mode of the signal size."""
        draws = self.betti_distribution(q, homology_degree)
        vals, counts = np.unique(draws.astype(int), return_counts=True)
        return int(vals[np.argmax(counts)])

    def modal_partition(self):
        """Most frequent partition among the draws, as block sizes ``(n_1..n_k)``."""
        post = self._require_fit()
        keys, first = {}, None
        for r in range(post.cuts.shape[0]):
            key = post.cuts[r].tobytes()
            keys[key] = keys.get(key, 0) + 1
        best = max(keys, key=keys.get)
        cuts_row = np.frombuffer(best, dtype=bool)
        sizes = np.diff(_edges_from_cuts(cuts_row, post.n))
        return sizes, keys[best] / post.cuts.shape[0]

    def diagnostics(self, q: float | None = None) -> dict:
        """Split-Rhat / ESS (on ``k``, ``theta``, and signal size) + acceptance."""
        post = self._require_fit()
        q = self._diag_q if q is None else q
        nc, ns = self._n_chains, self._n_samples

        def _reshape(x):
            return x.reshape(nc, ns)

        beta = np.array([_signal_size(post.cuts[r], post.n, q)
                         for r in range(post.cuts.shape[0])], dtype=float)
        out = {"accept_rate": post.accept_rate, "n_features": post.n,
               "n_chains": nc, "n_samples_per_chain": ns, "q": q}
        for name, series in (("k", post.k.astype(float)),
                             ("theta", post.theta),
                             ("beta_signal", beta)):
            chains = _reshape(series)
            out[f"rhat_{name}"] = potential_scale_reduction(chains)
            out[f"ess_{name}"] = effective_sample_size(chains)
        return out


def signal_probability(diagram, convention: str = "bd", q: float = 0.03,
                       n_samples: int = 5000, burn_in: int = 10000,
                       random_state=None, **kwargs) -> np.ndarray:
    """One-shot convenience: fit the model and return ``pi_p`` per feature.

    ``diagram`` may be a persistence diagram ``(n, 2)`` (lifetimes extracted with
    ``convention`` in ``{"bd", "bp"}``) or a 1-D array of positive lifetimes.
    Extra ``kwargs`` (hyperparameters, ``n_chains``, ``max_points``, ...) are
    forwarded to :class:`RestrictedPartitionModel` / :meth:`fit`.
    """
    from btate.topo_posterior.adapters import lifetimes as _diagram_lifetimes

    arr = np.asarray(diagram, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        ell = _diagram_lifetimes(arr, convention=convention)
    else:
        ell = arr.ravel()

    model_keys = ("m", "c", "a", "b", "theta_prior")
    model_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in model_keys}
    model = RestrictedPartitionModel(**model_kwargs)
    model.fit(ell, n_samples=n_samples, burn_in=burn_in,
              random_state=random_state, **kwargs)
    return model.signal_probability(q=q)
