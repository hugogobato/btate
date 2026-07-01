"""Smoke tests: the package and all submodules import in a minimal environment."""
import importlib

import btate


def test_version():
    assert isinstance(btate.__version__, str)
    assert btate.__version__


def test_submodules_present():
    for name in ["topo_posterior", "partitions", "embeddings", "causal", "benchmarks"]:
        assert hasattr(btate, name), f"missing submodule: {name}"


def test_submodules_importable():
    for name in [
        "btate.topo_posterior",
        "btate.topo_posterior.adapters",
        "btate.topo_posterior.sampler",
        "btate.topo_posterior.elicitation",
        "btate.partitions",
        "btate.partitions.lifetime_model",
        "btate.embeddings",
        "btate.embeddings.silhouette",
        "btate.embeddings.landscape",
        "btate.causal",
        "btate.causal.fgp",
        "btate.causal.propagation",
        "btate.causal.tsbcf",
        "btate.benchmarks",
        "btate.benchmarks.metrics",
    ]:
        importlib.import_module(name)
