"""Functional-BCF alternative (Research_Plan Task 3.2): ``tsbcf`` bridge.

Targeted Smooth Bayesian Causal Forests (Starling et al. 2020; vendored R
package in ``tsbcf-master/``) treat the filtration parameter ``t`` as the
"targeted" smoothing covariate: the prognostic and treatment-effect forests
vary smoothly in ``t``, so the posterior draws of the effect surface
``tau(t, x)`` yield draws of the TATE curve ``psi_d(t)`` by averaging over
subjects.  This gives a tree-ensemble counterpart to the finite-rank FGP with
a *fully Bayesian* uncertainty treatment (including BART's data-anchored
``sigma`` prior: ``nu``, ``sigq``, ``lambda`` from a naive overestimate).

The bridge is subprocess-based (``Rscript``), not rpy2: the generated script
installs the vendored package from source into a persistent local library on
first use, fits ``tsbcf()``, averages ``tau`` over subjects per grid point,
and writes the ``(nsim, resolution)`` matrix of ``psi_d(t)`` draws to CSV.
:func:`fit_tsbcf_tate` reads the draws back and wraps them in the same
:class:`~btate.causal.fgp.CausalEffectPosterior` object the FGP produces, so
every downstream metric (coverage, width, no-effect test) applies unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .fgp import CausalEffectPosterior, summarize_causal_effect


def _r_scalar(value: float | str, name: str) -> str:
    """Render a numeric-or-``"tune"`` tsbcf argument as R source."""
    if isinstance(value, str):
        if value != "tune":
            raise ValueError(f"{name} must be a positive number or 'tune'")
        return '"tune"'
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive number or 'tune'")
    return repr(value)


@dataclass
class TSBCFLongData:
    """Long-format arrays consumed by the R ``tsbcf()`` function."""

    y: np.ndarray
    pihat: np.ndarray
    z: np.ndarray
    tgt: np.ndarray
    x_control: np.ndarray
    x_moderate: np.ndarray
    subject_index: np.ndarray
    grid: np.ndarray


def make_tsbcf_long_data(phi, A, X, tseq, pi_hat=None) -> TSBCFLongData:
    """Flatten functional outcomes for targeted smooth BCF.

    ``tseq`` becomes the targeted smoothing covariate ``tgt``.  Rows are
    subject-major (subject ``i``'s ``resolution`` grid points are contiguous),
    which the psi-extraction in :meth:`TSBCFAdapter.script` relies on.  The
    same baseline covariates are used for prognostic and treatment-effect
    trees by default; callers can post-process the returned arrays if they
    want different covariate sets.
    """
    curves = np.asarray(phi, dtype=float)
    if curves.ndim != 2:
        raise ValueError("phi must have shape (n, resolution) for tsbcf long data")
    A_arr = np.asarray(A, dtype=int).ravel()
    if curves.shape[0] != A_arr.shape[0]:
        raise ValueError("phi and A must have the same number of subjects")
    if not set(np.unique(A_arr).tolist()).issubset({0, 1}):
        raise ValueError("A must contain only 0/1 values")
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    if X_arr.ndim != 2 or X_arr.shape[0] != curves.shape[0]:
        raise ValueError("X must have shape (n, p)")
    grid = np.asarray(tseq, dtype=float).ravel()
    if grid.shape[0] != curves.shape[1]:
        raise ValueError("tseq length must match curve resolution")
    if pi_hat is None:
        pi = np.full(A_arr.shape[0], float(np.mean(A_arr)))
    else:
        pi = np.asarray(pi_hat, dtype=float).ravel()
        if pi.shape[0] != A_arr.shape[0]:
            raise ValueError("pi_hat must have length n")
    n, m = curves.shape
    subject_index = np.repeat(np.arange(n), m)
    return TSBCFLongData(
        y=curves.reshape(-1),
        pihat=np.repeat(np.clip(pi, 1e-3, 1.0 - 1e-3), m),
        z=np.repeat(A_arr, m),
        tgt=np.tile(grid, n),
        x_control=np.repeat(X_arr, m, axis=0),
        x_moderate=np.repeat(X_arr, m, axis=0),
        subject_index=subject_index,
        grid=grid,
    )


class TSBCFAdapter:
    """Bridge to the vendored R package in ``tsbcf-master``.

    The data contract (:meth:`make_long_data`) and the generated R script are
    testable without R; :meth:`run_rscript` and :func:`fit_tsbcf_tate` require
    ``Rscript`` plus the ``Rcpp``/``RcppArmadillo``/``data.table`` R packages
    (the vendored package is compiled from source into ``lib_dir`` on first
    use and reused afterwards).
    """

    def __init__(self, package_dir: str | Path = "tsbcf-master",
                 rscript: str = "Rscript",
                 lib_dir: str | Path | None = None):
        self.package_dir = Path(package_dir).resolve()
        self.rscript = rscript
        self.lib_dir = (
            Path(lib_dir).resolve() if lib_dir is not None
            else self.package_dir.parent / ".tsbcf-lib"
        )

    def available(self) -> bool:
        """Return true when the configured Rscript binary is on PATH."""
        return shutil.which(self.rscript) is not None

    def make_long_data(self, phi, A, X, tseq, pi_hat=None) -> TSBCFLongData:
        return make_tsbcf_long_data(phi, A, X, tseq, pi_hat=pi_hat)

    def script(self, input_csv: str = "tsbcf_input.csv",
               psi_csv: str = "tsbcf_psi_draws.csv",
               sigma_csv: str = "tsbcf_sigma_draws.csv",
               output_rds: str | None = None,
               nburn: int = 100, nsim: int = 1000,
               ntree_control: int = 200, ntree_moderate: int = 50,
               ecross_control: float | str = 1.0,
               ecross_moderate: float | str = 1.0) -> str:
        """Return an R script that fits ``tsbcf`` and exports psi_d(t) draws.

        ``tau`` from ``tsbcf()`` is ``(nsim, n*m)`` in input-row order; with
        subject-major rows, reshaping to ``(nsim, m, n)`` (column-major) and
        averaging over the subject dimension gives the ``(nsim, m)`` matrix of
        sample-average treatment-effect-curve draws — the same estimand the
        FGP's ``posterior_tate`` targets.

        ``ecross_control`` / ``ecross_moderate`` set the expected number of
        crossings of the leaf-level GPs over the ``t`` range (tsbcf's
        smoothness prior).  The tsbcf default of 1 makes ``tau(x, t)`` nearly
        linear in ``t``; raise it (or pass ``"tune"`` for tsbcf's WAIC-based
        selection) when the effect curve is expected to have local structure.
        """
        pkg = self.package_dir.as_posix()
        lib = self.lib_dir.as_posix()
        rds_line = (
            f'saveRDS(fit, "{output_rds}")' if output_rds is not None else ""
        )
        ecross_c = _r_scalar(ecross_control, "ecross_control")
        ecross_m = _r_scalar(ecross_moderate, "ecross_moderate")
        return f"""
lib <- Sys.getenv("TSBCF_LIB", unset = "{lib}")
dir.create(lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(lib, .libPaths()))
if (!requireNamespace("tsbcf", quietly = TRUE)) {{
  for (dep in c("Rcpp", "RcppArmadillo", "data.table")) {{
    if (!requireNamespace(dep, quietly = TRUE)) {{
      install.packages(dep, lib = lib, repos = "https://cloud.r-project.org")
    }}
  }}
  install.packages("{pkg}", lib = lib, repos = NULL, type = "source")
}}
library(tsbcf)
if (!requireNamespace("data.table", quietly = TRUE)) stop("data.table is required")
dat <- data.table::fread("{input_csv}")
y <- dat$y
pihat <- dat$pihat
z <- dat$z
tgt <- dat$tgt
xcols <- grep("^x", names(dat), value = TRUE)
x <- as.data.frame(dat[, ..xcols])   # tsbcf() appends pihat via df indexing
m <- length(unique(tgt))
n <- length(y) / m
stopifnot(n == round(n))
fit <- tsbcf(
  y = y, pihat = pihat, z = z, tgt = tgt,
  x_control = x, x_moderate = x,
  nburn = {int(nburn)}, nsim = {int(nsim)},
  ntree_control = {int(ntree_control)}, ntree_moderate = {int(ntree_moderate)},
  ecross_control = {ecross_c}, ecross_moderate = {ecross_m},
  verbose = FALSE
)
tau <- fit$tau                               # (nsim, n*m), input-row order
dim(tau) <- c(nrow(tau), m, as.integer(n))   # column-major: (draw, grid, subject)
psi <- rowMeans(tau, dims = 2)               # average over subjects -> (nsim, m)
write.table(psi, "{psi_csv}", sep = ",", row.names = FALSE, col.names = FALSE)
write.table(fit$sigma, "{sigma_csv}", sep = ",", row.names = FALSE, col.names = FALSE)
{rds_line}
""".lstrip()

    def _write_long_data(self, long_data: TSBCFLongData, data_path: Path) -> None:
        cols = [
            long_data.y,
            long_data.pihat,
            long_data.z,
            long_data.tgt,
            long_data.subject_index,
        ]
        header = ["y", "pihat", "z", "tgt", "subject_index"]
        for j in range(long_data.x_control.shape[1]):
            cols.append(long_data.x_control[:, j])
            header.append(f"x{j}")
        matrix = np.column_stack(cols)
        np.savetxt(data_path, matrix, delimiter=",",
                   header=",".join(header), comments="")

    def run_rscript(self, long_data: TSBCFLongData, workdir: str | Path | None = None,
                    nburn: int = 100, nsim: int = 1000,
                    ntree_control: int = 200, ntree_moderate: int = 50,
                    ecross_control: float | str = 1.0,
                    ecross_moderate: float | str = 1.0,
                    output_rds: str | None = None) -> subprocess.CompletedProcess:
        """Write long data and execute the generated R script.

        Requires local R dependencies; the returned process object contains
        stdout and stderr for reproducibility logs.
        """
        if not self.available():
            raise RuntimeError(f"{self.rscript!r} was not found on PATH")
        if workdir is None:
            tmp = tempfile.TemporaryDirectory()
            root = Path(tmp.name)
        else:
            tmp = None
            root = Path(workdir)
            root.mkdir(parents=True, exist_ok=True)
        try:
            self._write_long_data(long_data, root / "tsbcf_input.csv")
            script_path = root / "run_tsbcf.R"
            script_path.write_text(
                self.script(nburn=nburn, nsim=nsim,
                            ntree_control=ntree_control,
                            ntree_moderate=ntree_moderate,
                            ecross_control=ecross_control,
                            ecross_moderate=ecross_moderate,
                            output_rds=output_rds),
                encoding="utf-8",
            )
            return subprocess.run(
                [self.rscript, str(script_path)],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        finally:
            if tmp is not None:
                tmp.cleanup()

    def fit_posterior_tate(self, phi, A, X, tseq, pi_hat=None,
                           nburn: int = 100, nsim: int = 1000,
                           ntree_control: int = 200, ntree_moderate: int = 50,
                           ecross_control: float | str = 1.0,
                           ecross_moderate: float | str = 1.0,
                           alpha: float = 0.05,
                           workdir: str | Path | None = None) -> CausalEffectPosterior:
        """Fit ``tsbcf`` on observed curves and return the psi_d(t) posterior.

        ``phi`` has shape ``(n, resolution)`` — e.g. the Maroulas posterior-mean
        embeddings (plug-in propagation, mirroring ``plugin_posterior_tate``).
        """
        long_data = self.make_long_data(phi, A, X, tseq, pi_hat=pi_hat)
        if workdir is None:
            tmp = tempfile.TemporaryDirectory()
            root = Path(tmp.name)
        else:
            tmp = None
            root = Path(workdir)
            root.mkdir(parents=True, exist_ok=True)
        try:
            self._write_long_data(long_data, root / "tsbcf_input.csv")
            script_path = root / "run_tsbcf.R"
            script_path.write_text(
                self.script(nburn=nburn, nsim=nsim,
                            ntree_control=ntree_control,
                            ntree_moderate=ntree_moderate,
                            ecross_control=ecross_control,
                            ecross_moderate=ecross_moderate),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [self.rscript, str(script_path)],
                cwd=root, check=False, capture_output=True, text=True,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
            psi_path = root / "tsbcf_psi_draws.csv"
            if proc.returncode != 0 or not psi_path.exists():
                raise RuntimeError(
                    "tsbcf R fit failed (exit "
                    f"{proc.returncode}).\nstdout tail:\n{proc.stdout[-2000:]}\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}"
                )
            draws = np.loadtxt(psi_path, delimiter=",", ndmin=2)
            sigma_path = root / "tsbcf_sigma_draws.csv"
            sigma_mean = (
                float(np.mean(np.loadtxt(sigma_path, delimiter=",")))
                if sigma_path.exists() else float("nan")
            )
        finally:
            if tmp is not None:
                tmp.cleanup()
        if draws.shape[1] != long_data.grid.shape[0]:
            raise RuntimeError(
                f"tsbcf psi draws have {draws.shape[1]} columns; expected "
                f"{long_data.grid.shape[0]} grid points"
            )
        return summarize_causal_effect(
            draws, grid=long_data.grid, alpha=alpha,
            metadata={
                "model": "tsbcf",
                "propagation": "plugin",
                "nburn": int(nburn), "nsim": int(nsim),
                "ntree_control": int(ntree_control),
                "ntree_moderate": int(ntree_moderate),
                "ecross_control": ecross_control,
                "ecross_moderate": ecross_moderate,
                "sigma_mean": sigma_mean,
                "n_subjects": int(np.asarray(phi).shape[0]),
            },
        )


def fit_tsbcf_tate(phi_draws_or_mean, A, X, tseq, pi_hat=None,
                   package_dir: str | Path = "tsbcf-master",
                   rscript: str = "Rscript",
                   lib_dir: str | Path | None = None,
                   nburn: int = 100, nsim: int = 1000,
                   ntree_control: int = 200, ntree_moderate: int = 50,
                   ecross_control: float | str = 1.0,
                   ecross_moderate: float | str = 1.0,
                   alpha: float = 0.05, nested_draws: int | None = None,
                   workdir: str | Path | None = None) -> CausalEffectPosterior:
    """Functional-BCF posterior of ``psi_d(t)`` (drop-in FGP alternative).

    ``phi_draws_or_mean`` is either ``(n, resolution)`` observed curves or
    ``(S, n, resolution)`` topological posterior draws.  For 3-D input the
    default is plug-in propagation (average the draws, one MCMC fit).  Set
    ``nested_draws=k`` to instead refit ``tsbcf`` on ``k`` evenly-spaced
    topological draws and pool the psi draws — the (expensive) nested
    propagation, at MCMC cost multiplied by ``k``.
    """
    adapter = TSBCFAdapter(package_dir=package_dir, rscript=rscript,
                           lib_dir=lib_dir)
    arr = np.asarray(phi_draws_or_mean, dtype=float)
    kwargs = dict(nburn=nburn, nsim=nsim, ntree_control=ntree_control,
                  ntree_moderate=ntree_moderate, ecross_control=ecross_control,
                  ecross_moderate=ecross_moderate, alpha=alpha, workdir=workdir)
    if arr.ndim == 2 or nested_draws is None or nested_draws <= 1:
        phi = arr.mean(axis=0) if arr.ndim == 3 else arr
        return adapter.fit_posterior_tate(phi, A, X, tseq, pi_hat=pi_hat, **kwargs)
    if arr.ndim != 3:
        raise ValueError("phi must have shape (n, m) or (S, n, m)")
    idx = np.unique(np.linspace(0, arr.shape[0] - 1,
                                min(int(nested_draws), arr.shape[0])).astype(int))
    pooled = []
    effect = None
    for i in idx:
        effect = adapter.fit_posterior_tate(arr[i], A, X, tseq,
                                            pi_hat=pi_hat, **kwargs)
        pooled.append(effect.draws)
    draws = np.vstack(pooled)
    return summarize_causal_effect(
        draws, grid=effect.grid, alpha=alpha,
        metadata=dict(effect.metadata, propagation="nested",
                      n_topological_draws=int(len(idx))),
    )
