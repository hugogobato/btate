"""Tests for the completed functional-BCF (tsbcf) bridge (Task 3.2)."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from btate.causal import TSBCFAdapter, fit_tsbcf_tate

_REPO = Path(__file__).resolve().parents[1]
_PKG = _REPO / "tsbcf-master"

_HAS_R = shutil.which("Rscript") is not None


def _toy_curves(n=10, m=8, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    A = np.array([0, 1] * (n // 2))
    t = np.linspace(0.1, 0.9, m)
    base = np.sin(np.pi * t)[None, :]
    phi = 0.3 * base + 0.15 * A[:, None] * base + 0.05 * rng.normal(size=(n, m))
    return phi, A, X, t


def test_script_exports_psi_draws():
    adapter = TSBCFAdapter(package_dir=_PKG)
    script = adapter.script(nburn=10, nsim=20)
    assert "rowMeans(tau, dims = 2)" in script
    assert "tsbcf_psi_draws.csv" in script
    assert ".libPaths" in script            # source-install into local lib
    assert "devtools" not in script
    assert "saveRDS" not in script          # RDS export off by default
    assert 'saveRDS(fit, "fit.rds")' in adapter.script(output_rds="fit.rds")
    assert "ecross_moderate = 1.0" in script            # tsbcf default
    smooth = adapter.script(ecross_moderate=5, ecross_control="tune")
    assert "ecross_moderate = 5.0" in smooth
    assert 'ecross_control = "tune"' in smooth
    with pytest.raises(ValueError):
        adapter.script(ecross_moderate="waic")
    with pytest.raises(ValueError):
        adapter.script(ecross_control=-1)


@pytest.mark.skipif(not _HAS_R, reason="Rscript not on PATH")
def test_fit_tsbcf_tate_end_to_end(tmp_path):
    phi, A, X, t = _toy_curves()
    effect = fit_tsbcf_tate(
        phi, A, X, t, pi_hat=np.full(len(A), 0.5),
        package_dir=_PKG, nburn=20, nsim=40,
        ntree_control=20, ntree_moderate=10,
        workdir=tmp_path / "run",
    )
    assert effect.draws.shape == (40, len(t))
    assert effect.metadata["model"] == "tsbcf"
    assert np.all(effect.simultaneous_upper >= effect.simultaneous_lower)
    assert np.isfinite(effect.metadata["sigma_mean"])
    # injected effect is positive mid-grid; the posterior mean should be too
    assert effect.mean[len(t) // 2] > 0.0


@pytest.mark.skipif(not _HAS_R, reason="Rscript not on PATH")
def test_fit_tsbcf_tate_nested_pools_draws(tmp_path):
    phi, A, X, t = _toy_curves()
    rng = np.random.default_rng(0)
    phi_draws = phi[None, :, :] + 0.01 * rng.normal(size=(3,) + phi.shape)
    effect = fit_tsbcf_tate(
        phi_draws, A, X, t, pi_hat=np.full(len(A), 0.5),
        package_dir=_PKG, nburn=10, nsim=20,
        ntree_control=20, ntree_moderate=10, nested_draws=2,
        workdir=tmp_path / "nested",
    )
    assert effect.draws.shape == (40, len(t))          # 2 fits x 20 draws
    assert effect.metadata["propagation"] == "nested"
    assert effect.metadata["n_topological_draws"] == 2
