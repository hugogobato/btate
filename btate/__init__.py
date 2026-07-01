"""btate — End-to-End Bayesian Topological Causal Inference (Bayesian TATE).

Implements the four-step hierarchical pipeline from ``Research_Idea.md`` /
``Research_Plan.md``:

1. :mod:`btate.topo_posterior` — marked-Poisson-point-process posterior over
   persistence diagrams (Maroulas et al. 2020).  Wraps the closed-form
   restricted-Gaussian-mixture posterior intensity in ``bayes_tda``.
2. :mod:`btate.partitions` — Bayesian random-partition signal/noise separation
   yielding per-feature signal probabilities ``pi_p`` (Martínez 2024).
3. :mod:`btate.embeddings` — posterior-weighted silhouettes and persistence
   landscapes (Bubenik 2015; Kim & Lee 2026 silhouette).
4. :mod:`btate.causal` — functional Gaussian-process / BCF posterior for the
   Topological Average Treatment Effect curve ``psi_d(t)`` (Kim & Lee 2026).

:mod:`btate.benchmarks` holds evaluation utilities (coverage, RMSE, power).

See ``docs/notation.md`` for the cross-paper notation crosswalk and
``docs/literature_synthesis.md`` for method summaries and assumptions.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Submodules import only dependency-light members at package-import time.
# Heavy / optional dependencies (bayes_tda, gpytorch, skfda) are imported
# lazily inside the functions/classes that need them, so ``import btate``
# always succeeds in a minimal environment.
from . import topo_posterior, partitions, embeddings, causal, benchmarks

__all__ = [
    "topo_posterior",
    "partitions",
    "embeddings",
    "causal",
    "benchmarks",
    "__version__",
]
