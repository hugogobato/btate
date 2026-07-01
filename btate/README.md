# btate — End-to-End Bayesian Topological Causal Inference

`btate` implements a fully Bayesian pipeline for the **Topological Average
Treatment Effect (TATE)** curve `ψ_d(t)`, propagating topological / structural
uncertainty *and* causal / confounding uncertainty into calibrated posterior
credible bands. It is the software artifact of `Research_Plan.md`.

## The four hierarchical steps

| Submodule | Step | Method | Reference |
|---|---|---|---|
| `btate.topo_posterior` | 1. Posterior over persistence diagrams | Marked-PPP posterior intensity (closed-form restricted Gaussian mixture) | Maroulas et al. 2020 |
| `btate.partitions` | 2. Signal/noise separation → `π_p` | Log-normal lifetimes + restricted random partition (EPPF/DP), split–merge MCMC | Martínez 2024 |
| `btate.embeddings` | 3. Functional embeddings | `π_p`-weighted silhouettes / persistence landscapes | Bubenik 2015; Kim & Lee 2026 |
| `btate.causal` | 4. Bayesian causal model | Functional GP / functional BCF for `ψ_d(t)` | Kim & Lee 2026; Starling 2020 |
| `btate.benchmarks` | Evaluation | Coverage, RMSE, power vs. frequentist AIPW | — |

## Status

Phase 1 core topology is implemented and tested:

- `topo_posterior`: diagram adapters, Maroulas posterior-diagram sampler, and
  prior/clutter elicitation plus sensitivity grid;
- `partitions`: Martinez restricted random-partition lifetime model,
  split-merge MCMC, signal probabilities `pi_p`, Betti summaries, R-hat, and ESS;
- `benchmarks.metrics`: baseline curve metrics.

Phase 2 embeddings are implemented and tested:

- `embeddings.weighted_silhouette`: fixed-`r` TATE parity plus `pi_p` weights;
- `embeddings.posterior_landscape`: Gudhi landscape wrapper with explicit
  `(draw, level, grid)` output;
- `embeddings.posterior_embedding_summary`: posterior draws, means, pointwise
  bands, and simultaneous bands;
- `embeddings.fit_fpca` / `project_fourier`: dimension reduction for Phase 3.

Phase 3 causal modeling is implemented and tested:

- `causal.FunctionalGPEstimator`: finite-rank inducing-point functional GP over
  `(X, t)` with propensity-weighted likelihoods;
- `causal.compare_propagation`: nested topological-to-causal posterior
  propagation vs. the plug-in posterior-mean shortcut;
- `causal.CausalEffectPosterior`: posterior `psi_d(t)` draws with pointwise and
  simultaneous credible bands;
- `causal.TSBCFAdapter`: optional long-format bridge to `tsbcf-master`.

See `../docs/phase1_status.md`, `../docs/phase2_status.md`,
`../docs/phase3_status.md`, and `../results/bayesian/README.md` for validation
outputs.

## Install / run

The scientific stack is heavy; use a dedicated environment (see
`../ENVIRONMENT.md` and `../environment.yml`). The system Python may be
PEP-668 "externally managed", so prefer conda or a venv:

```bash
conda env create -f environment.yml    # creates the `btate` env
conda activate btate
pip install -e .                        # editable install of this package
pytest                                  # run the package tests
```

Without an install, run tests from the repository root (which puts `btate/` on
the path): `python -m pytest`.
