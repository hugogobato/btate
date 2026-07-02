"""End-to-end Bayesian TATE pipeline engine for benchmarking (Phase 4).

Composes the four hierarchical steps into a single, config-driven callable that
maps observed point clouds to a posterior of ``psi_d(t)`` with credible bands:

    point cloud  --(gudhi PH)-->  H1 diagram
                 --(Step 1)-->    posterior diagram draws
                 --(Step 2)-->    per-point signal probabilities pi_p
                 --(Step 3)-->    posterior functional summaries (silhouette / landscape)
                 --(Step 4)-->    FGP nested / plug-in posterior of psi_d(t)

Two topological-posterior modes trade fidelity for speed:

* ``topo_method="maroulas"`` — the faithful Step-1 posterior: fit a
  ``bayes_tda`` restricted-Gaussian-mixture posterior intensity and draw
  diagrams from it (needs ``bayes_tda`` on the path).
* ``topo_method="jitter"`` — a fast bootstrap that perturbs the observed diagram
  in birth--persistence coordinates; pure ``numpy`` + ``gudhi``, suitable for
  large sweeps and minimal environments.

Both modes flow the resulting posterior draws through the *same* Step-2/3/4
machinery, so uncertainty propagates end to end in either case.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from btate.causal import compare_propagation, FunctionalGPEstimator
from btate.embeddings import posterior_embedding_summary
from btate.partitions import signal_probability
from btate.topo_posterior import bd_to_bp, bp_to_bd


@dataclass
class PipelineConfig:
    """Configuration for the end-to-end Bayesian TATE pipeline."""

    # --- Step 3 embedding ---
    embedding: str = "silhouette"          # "silhouette" | "landscape"
    weights: str = "pi"                    # "pi" | "power" (fixed-r baseline)
    r: float = 3.0
    num_landscapes: int = 3
    sample_range: tuple[float, float] | None = None   # auto from data if None
    resolution: int = 80

    # --- Step 1 posterior diagram draws ---
    topo_method: str = "jitter"            # "jitter" | "maroulas"
    posterior_draws: int = 8               # S topological posterior draws
    jitter_sigma: float = 0.30             # jitter as a fraction of median lifetime
    sigma_dyo: float = 0.03                # maroulas: DYO kernel sd
    posterior_alpha: float = 1.0
    prior_components: int = 8
    clutter_components: int = 2

    # --- Step 2 signal probabilities (pi_p) ---
    pi_per_draw: bool = False              # recompute pi per topological draw
    q: float = 0.08
    partition_samples: int = 120
    partition_burn_in: int = 200
    max_points: int = 60

    # --- Step 4 causal model ---
    propagation: str = "nested"            # "nested" | "plugin" | "both"
    n_causal_draws: int = 60
    n_plugin_draws: int = 480
    n_inducing: int = 48
    prior_scale: float = 5.0
    length_scale_x: float | None = None
    length_scale_t: float | None = 0.06   # smaller than the FGP default: the
    #   silhouette/landscape peak is sharp, so less temporal smoothing reduces
    #   peak-attenuation bias of the causal posterior mean (calibrated in Phase 4).
    noise_variance: float | None = None
    propensity_clip: float = 0.02
    # Effective-sample-size inflation for calibrated curve-level bands.  The
    # finite-rank FGP treats the ``resolution`` grid points per subject as
    # independent, overcounting precision by ~ resolution / (temporal dof).
    # Calibrated in Phase 4 (near-nominal coverage of the self-consistent estimand).
    fgp_posterior_scale: float = 8.0
    alpha: float = 0.05

    # --- Notebook ergonomics ---
    progress: bool = False                # show tqdm progress bars if available

    seed: int = 20260701


@dataclass
class PipelineResult:
    grid: np.ndarray
    phi_draws: np.ndarray            # (S, n, resolution) observed-arm draws
    nested: object | None
    plugin: object | None
    comparison: object | None
    timing: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def h1_diagram(points: np.ndarray) -> np.ndarray:
    """Alpha-complex H1 persistence diagram in birth--death coordinates."""
    import gudhi as gd

    points = np.asarray(points, dtype=float)
    st = gd.AlphaComplex(points=points).create_simplex_tree()
    st.compute_persistence()
    dgm = st.persistence_intervals_in_dimension(1)
    if dgm.size == 0:
        return np.empty((0, 2), dtype=float)
    dgm = dgm[~np.isinf(dgm).any(axis=1)]
    return np.sqrt(dgm)


def _auto_sample_range(diagrams) -> tuple[float, float]:
    deaths = [d[:, 1].max() for d in diagrams if d.shape[0] > 0]
    if not deaths:
        return (0.0, 1.0)
    return (0.0, float(1.05 * max(deaths)))


def _jitter_draws(diagram_bd, n_draws, sigma, rng):
    """Fast Step-1 surrogate: perturb the diagram in birth--persistence space.

    The perturbation scale is tied to the *signal* scale (an upper quantile of
    persistence) rather than the median, so the long-lived loop feature — not
    just the short-lived clutter — is genuinely perturbed and topological
    uncertainty propagates into the downstream bands.
    """
    if diagram_bd.shape[0] == 0:
        return [np.empty((0, 2), dtype=float) for _ in range(n_draws)]
    bp = bd_to_bp(diagram_bd)                       # columns (birth, persistence)
    signal_scale = np.percentile(bp[:, 1], 85) if bp.shape[0] else 1.0
    scale = max(sigma * signal_scale, 1e-4)
    out = []
    for _ in range(n_draws):
        noise = rng.normal(0.0, scale, size=bp.shape)
        pert = bp + noise
        pert[:, 1] = np.maximum(pert[:, 1], 1e-6)   # persistence stays positive
        pert[:, 0] = np.maximum(pert[:, 0], 0.0)
        out.append(bp_to_bd(pert))
    return out


def _maroulas_draws(diagram_bd, prior, clutter, cfg, rng_seed):
    from bayes_tda.intensities import Posterior

    from btate.topo_posterior import PosteriorDiagramSampler

    if diagram_bd.shape[0] == 0:
        return [np.empty((0, 2), dtype=float) for _ in range(cfg.posterior_draws)]
    diagram_bp = bd_to_bp(diagram_bd)
    posterior = Posterior(
        DYO=[diagram_bp], prior=prior, clutter=clutter,
        sigma_DYO=cfg.sigma_dyo, alpha=cfg.posterior_alpha, min_birth=0.0,
    )
    sampler = PosteriorDiagramSampler(posterior)
    cardinality = max(1, int(diagram_bd.shape[0]))
    return [
        bp_to_bd(d)
        for d in sampler.sample_diagrams(
            cfg.posterior_draws, random_state=rng_seed,
            count="fixed", cardinality=cardinality,
        )
    ]


def _pi_for(diagram_bd, cfg, seed):
    if diagram_bd.shape[0] == 0:
        return np.empty(0)
    return signal_probability(
        diagram_bd, convention="bd", q=cfg.q,
        n_samples=cfg.partition_samples, burn_in=cfg.partition_burn_in,
        n_chains=1, max_points=cfg.max_points, random_state=seed,
    )


def _subject_embedding_draws(diagram_bd, prior, clutter, cfg, sample_range, seed):
    """Return ``(S, resolution)`` posterior functional draws for one arm."""
    rng = np.random.default_rng(seed)
    if cfg.topo_method == "maroulas":
        draws_bd = _maroulas_draws(diagram_bd, prior, clutter, cfg, seed)
    elif cfg.topo_method == "jitter":
        draws_bd = _jitter_draws(diagram_bd, cfg.posterior_draws, cfg.jitter_sigma, rng)
    else:
        raise ValueError("topo_method must be 'jitter' or 'maroulas'")

    kwargs = dict(
        embedding=cfg.embedding, sample_range=sample_range,
        resolution=cfg.resolution, alpha=cfg.alpha,
    )
    if cfg.embedding == "silhouette":
        if cfg.weights == "pi":
            base_pi = None if cfg.pi_per_draw else _pi_for(diagram_bd, cfg, seed + 7)
            pi_draws = []
            for j, d in enumerate(draws_bd):
                if d.shape[0] == 0:
                    pi_draws.append(np.empty(0))
                elif cfg.pi_per_draw:
                    pi_draws.append(_pi_for(d, cfg, seed + 101 * (j + 1)))
                else:
                    pi_draws.append(base_pi if base_pi is not None else np.empty(0))
            summary = posterior_embedding_summary(
                draws_bd, weights="pi", pi=pi_draws, **kwargs,
            )
        else:
            summary = posterior_embedding_summary(
                draws_bd, weights="power", r=cfg.r, **kwargs,
            )
    else:  # landscape
        summary = posterior_embedding_summary(
            draws_bd, num_landscapes=cfg.num_landscapes, **kwargs,
        )
    draws = summary.draws
    if draws.ndim == 3:  # landscape returns (S, num_landscapes, res); use level 0
        draws = draws[:, 0, :]
    return draws, summary.grid


def _progress_iter(iterable, enabled: bool, **kwargs):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, **kwargs)


def run_bayesian_pipeline(clouds, A, X, pi_hat, cfg: PipelineConfig) -> PipelineResult:
    """Run the full Bayesian TATE pipeline on observed point clouds.

    Parameters
    ----------
    clouds : list of (m_i, 2) arrays
        Observed point cloud per subject (already arm-selected).
    A, X, pi_hat : arrays
        Treatment ``(n,)``, covariates ``(n, p)``, propensity ``(n,)``.
    cfg : PipelineConfig
    """
    A = np.asarray(A, dtype=int).ravel()
    X = np.asarray(X, dtype=float)
    n = len(clouds)
    t0 = time.perf_counter()

    diagrams = [
        h1_diagram(c)
        for c in _progress_iter(
            clouds, cfg.progress, total=n, desc="H1 diagrams", unit="subject",
        )
    ]
    sample_range = cfg.sample_range or _auto_sample_range(diagrams)

    prior = clutter = None
    if cfg.topo_method == "maroulas":
        from btate.topo_posterior.elicitation import elicit_prior_clutter

        train_bp = [bd_to_bp(d) for d in diagrams if d.shape[0] > 0]
        if train_bp:
            mean_card = max(1, int(np.mean([len(d) for d in train_bp])))
            prior, clutter = elicit_prior_clutter(
                train_bp,
                n_components=min(cfg.prior_components, mean_card),
                clutter_n_components=cfg.clutter_components,
                random_state=cfg.seed,
            )
    t_topo0 = time.perf_counter()

    per_subject = []
    grid = None
    for i in _progress_iter(
        range(n), cfg.progress, total=n, desc="posterior embeddings", unit="subject",
    ):
        draws, grid = _subject_embedding_draws(
            diagrams[i], prior, clutter, cfg, sample_range,
            seed=cfg.seed + 1000 * i,
        )
        per_subject.append(draws)                        # (S, res)
    # phi_draws[s, i, :] = draw s of subject i
    phi_draws = np.transpose(np.stack(per_subject), (1, 0, 2))   # (S, n, res)
    t_embed = time.perf_counter()

    estimator = FunctionalGPEstimator(
        n_inducing=cfg.n_inducing, prior_scale=cfg.prior_scale,
        length_scale_x=cfg.length_scale_x, length_scale_t=cfg.length_scale_t,
        noise_variance=cfg.noise_variance, propensity_clip=cfg.propensity_clip,
        posterior_scale=cfg.fgp_posterior_scale,
    )
    nested = plugin = comparison = None
    if cfg.propagation in ("nested", "both"):
        comparison = compare_propagation(
            phi_draws, A, X, grid, pi_hat=pi_hat, estimator=estimator,
            n_causal_draws=cfg.n_causal_draws, n_plugin_draws=cfg.n_plugin_draws,
            alpha=cfg.alpha, random_state=cfg.seed + 5, potential_outcomes=False,
        )
        nested = comparison.nested
        plugin = comparison.plugin
    else:  # plugin only
        from btate.causal import plugin_posterior_tate

        plugin = plugin_posterior_tate(
            phi_draws, A, X, grid, pi_hat=pi_hat, estimator=estimator,
            n_draws=cfg.n_plugin_draws, alpha=cfg.alpha,
            random_state=cfg.seed + 5, potential_outcomes=False,
        )
    t_causal = time.perf_counter()

    return PipelineResult(
        grid=grid, phi_draws=phi_draws, nested=nested, plugin=plugin,
        comparison=comparison,
        timing={
            "elicitation_s": t_topo0 - t0,
            "embedding_s": t_embed - t_topo0,
            "causal_s": t_causal - t_embed,
            "total_s": t_causal - t0,
        },
        meta={
            "n": int(n), "resolution": int(grid.shape[0]),
            "sample_range": list(sample_range),
            "topo_method": cfg.topo_method, "embedding": cfg.embedding,
            "weights": cfg.weights, "propagation": cfg.propagation,
            "posterior_draws": int(cfg.posterior_draws),
            "empty_diagrams": int(sum(d.shape[0] == 0 for d in diagrams)),
        },
    )


def silhouette_embedding_fn(cfg: PipelineConfig, sample_range):
    """Deterministic single-cloud embedding for the reference estimand.

    Returns a function mapping a point cloud to its (fixed-``r`` or landscape-0)
    curve on the pipeline grid, used by :func:`btate.benchmarks.synthetic.reference_effect`.
    """
    def _fn(cloud):
        d = h1_diagram(cloud)
        if cfg.embedding == "landscape":
            summary = posterior_embedding_summary(
                [d], embedding="landscape", num_landscapes=cfg.num_landscapes,
                sample_range=sample_range, resolution=cfg.resolution, alpha=cfg.alpha,
            )
            mean = summary.mean
            return mean[0] if mean.ndim == 2 else mean   # top landscape level
        summary = posterior_embedding_summary(
            [d], embedding="silhouette", weights="power", r=cfg.r,
            sample_range=sample_range, resolution=cfg.resolution, alpha=cfg.alpha,
        )
        return summary.mean

    return _fn
