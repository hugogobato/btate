# btate — End-to-End Bayesian Topological Causal Inference (Bayesian TATE)

**Authors:** Hugo G. Souto (Dell Technologies, hugo.souto@dell.com) ·
Ioannis Diamantis (Department of Data Analytics and Digitalisation, Maastricht
University, i.diamantis@maastrichtuniversity.nl)

`btate` implements a fully Bayesian version of the **Topological Average
Treatment Effect (TATE)** pipeline of Kim & Lee (2026). Raw outcomes are turned
into *posterior distributions* over persistence diagrams, per-feature signal
probabilities, posterior functional summaries, and finally a posterior of the
treatment-effect curve `ψ_d(t)` with **simultaneous credible bands** that
propagate topological, structural, and causal/confounding uncertainty end to
end.

The four hierarchical steps:

1. **`btate.topo_posterior`** — marked-Poisson-point-process posterior over
   persistence diagrams (Maroulas et al. 2020), sampled from the closed-form
   restricted-Gaussian-mixture posterior intensity (vendored `bayes_tda`).
2. **`btate.partitions`** — Bayesian random-partition signal/noise separation,
   yielding per-feature signal probabilities `π_p` (Martínez 2024).
3. **`btate.embeddings`** — posterior-weighted silhouettes and persistence
   landscapes (Bubenik 2015; Kim & Lee 2026).
4. **`btate.causal`** — a functional Gaussian-process (and optional functional
   BCF) posterior for `ψ_d(t)`, with nested topological→causal propagation.

`btate.benchmarks` holds the Phase-4 evaluation harness: a ground-truth
loop-injection DGP, the end-to-end pipeline engine, a self-contained
frequentist AIPW comparator, and config-driven sweep / ablation drivers.

## Install

```bash
pip install "git+https://github.com/hugogobato/btate.git"
```

This installs `btate` **and** the vendored `bayes_tda` (Maroulas posterior) and
`tcda_uq` (faithful frequentist TATE bands: multiplier bootstrap, Liebl-Reimherr,
Pini-Vantini). For the functional-regression / semi-synthetic pipelines and the
brain-connectivity application add the `colab` extra:

```bash
pip install "git+https://github.com/hugogobato/btate.git#egg=btate[colab]"
```

The head-to-head frequentist bands run without R via a validated numpy port; the
original `ffscb` R backend is vendored under `ffscb-master/` and used
automatically if `Rscript` is on `PATH`.

## Quickstart — one synthetic benchmark run

```python
from btate.benchmarks.synthetic import SyntheticConfig
from btate.benchmarks.pipeline import PipelineConfig
from btate.benchmarks.harness import evaluate_run

synth = SyntheticConfig(n=60, effect_size=0.12, noise_level=1.0, seed=1)
pipe  = PipelineConfig(embedding="silhouette", weights="pi",
                       topo_method="jitter", propagation="nested")
record = evaluate_run(synth, pipe, run_frequentist=True)
print(record["bayes_rmse"], record["bayes_cov_simultaneous"], record["bayes_reject"])
```

Use `topo_method="maroulas"` for the faithful Step-1 posterior (slower);
`topo_method="jitter"` is a fast, dependency-light surrogate for large sweeps.

## Phase-4 benchmark notebooks (Google Colab)

The heavy / GPU-scale experiments are packaged as Colab notebooks that clone
this repository:

| Notebook | Task | What it runs |
|---|---|---|
| `P4_synthetic_sweep_colab.ipynb` | 4.1 + 4.4 | Full ground-truth sweep (noise × n × overlap) + head-to-head vs. frequentist AIPW + ablation grid |
| `P4_semisynthetic_colab.ipynb` | 4.2 | GEOM-Drugs & SARS-CoV-2 through the Bayesian pipeline vs. frozen frequentist baselines |
| `P4_brain_connectivity_colab.ipynb` | 4.3 | Real-application demo on brain functional-connectivity networks |

Each notebook begins with a bootstrap cell that installs `btate` from this repo.

## Testing

```bash
pip install "git+https://github.com/hugogobato/btate.git#egg=btate[dev]"
pytest -q
```

## Citation

If you use `btate`, please cite Kim & Lee (2026, TATE), Maroulas et al. (2020),
Martínez (2024), and Bubenik (2015), together with this package.

## License

MIT (see `LICENSE`). Vendored `bayes_tda` is redistributed under its original
MIT license.
