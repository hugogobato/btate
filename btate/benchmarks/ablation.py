"""Ablation-grid construction for Phase 4 (Task 4.4).

Builds :class:`~btate.benchmarks.harness.SweepCell` lists that vary one design
axis at a time from a shared base configuration:

* ``pi_p`` weighting vs. fixed-``r`` power weighting (silhouette);
* silhouette vs. persistence landscape embedding;
* nested topological propagation vs. plug-in posterior-mean shortcut;
* (optionally) the faithful ``maroulas`` Step-1 posterior vs. the fast
  ``jitter`` surrogate.

The functional-BCF (``tsbcf``) alternative to the FGP is treated as a secondary
R-bridge comparison (see ``btate/causal/tsbcf.py``) and is not part of the
pure-numpy ablation grid.
"""
from __future__ import annotations

from dataclasses import replace

from .harness import SweepCell
from .pipeline import PipelineConfig
from .synthetic import SyntheticConfig


def weighting_ablation(base_synth: SyntheticConfig, base_pipe: PipelineConfig,
                       n_reps: int = 5) -> list[SweepCell]:
    """pi_p weighting vs. fixed-r power weighting (robustness to clutter)."""
    pi_pipe = replace(base_pipe, embedding="silhouette", weights="pi")
    r_pipe = replace(base_pipe, embedding="silhouette", weights="power")
    return [
        SweepCell("silhouette_pi", base_synth, pi_pipe, n_reps=n_reps),
        SweepCell("silhouette_fixed_r", base_synth, r_pipe, n_reps=n_reps),
    ]


def embedding_ablation(base_synth: SyntheticConfig, base_pipe: PipelineConfig,
                       n_reps: int = 5) -> list[SweepCell]:
    """Silhouette (pi) vs. persistence landscape."""
    sil = replace(base_pipe, embedding="silhouette", weights="pi")
    land = replace(base_pipe, embedding="landscape")
    return [
        SweepCell("silhouette_pi", base_synth, sil, n_reps=n_reps),
        SweepCell("landscape", base_synth, land, n_reps=n_reps),
    ]


def propagation_ablation(base_synth: SyntheticConfig, base_pipe: PipelineConfig,
                         n_reps: int = 5) -> list[SweepCell]:
    """Nested two-stage propagation vs. plug-in posterior-mean shortcut."""
    nested = replace(base_pipe, propagation="nested")
    plugin = replace(base_pipe, propagation="plugin")
    return [
        SweepCell("nested", base_synth, nested, n_reps=n_reps),
        SweepCell("plugin", base_synth, plugin, n_reps=n_reps),
    ]


def topo_fidelity_ablation(base_synth: SyntheticConfig, base_pipe: PipelineConfig,
                           n_reps: int = 3) -> list[SweepCell]:
    """Faithful Maroulas Step-1 posterior vs. fast jitter surrogate."""
    maroulas = replace(base_pipe, topo_method="maroulas")
    jitter = replace(base_pipe, topo_method="jitter")
    return [
        SweepCell("maroulas", base_synth, maroulas, n_reps=n_reps, run_frequentist=False),
        SweepCell("jitter", base_synth, jitter, n_reps=n_reps, run_frequentist=False),
    ]


def full_ablation_grid(base_synth: SyntheticConfig, base_pipe: PipelineConfig,
                       n_reps: int = 5) -> list[SweepCell]:
    """Concatenate the weighting / embedding / propagation ablations.

    Duplicate ``silhouette_pi`` cells (shared baseline) are de-duplicated by name.
    """
    cells: list[SweepCell] = []
    seen: set[str] = set()
    for group in (
        weighting_ablation(base_synth, base_pipe, n_reps),
        embedding_ablation(base_synth, base_pipe, n_reps),
        propagation_ablation(base_synth, base_pipe, n_reps),
    ):
        for cell in group:
            if cell.name in seen:
                continue
            seen.add(cell.name)
            cells.append(cell)
    return cells
