"""Adapter for the optional ``tsbcf`` functional-BCF alternative."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


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

    ``tseq`` becomes the targeted smoothing covariate ``tgt``.  The same
    baseline covariates are used for prognostic and treatment-effect trees by
    default; callers can post-process the returned arrays if they want different
    covariate sets.
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
    """Thin optional bridge to the R package in ``tsbcf-master``.

    The adapter is intentionally explicit: Python tests can validate the data
    contract without requiring R, while users with R installed can call
    :meth:`run_rscript` to execute a generated script in a temporary directory.
    """

    def __init__(self, package_dir: str | Path = "tsbcf-master",
                 rscript: str = "Rscript"):
        self.package_dir = Path(package_dir).resolve()
        self.rscript = rscript

    def available(self) -> bool:
        """Return true when the configured Rscript binary is on PATH."""
        return shutil.which(self.rscript) is not None

    def make_long_data(self, phi, A, X, tseq, pi_hat=None) -> TSBCFLongData:
        return make_tsbcf_long_data(phi, A, X, tseq, pi_hat=pi_hat)

    def script(self, input_csv: str = "tsbcf_input.csv",
               output_rds: str = "tsbcf_fit.rds",
               nburn: int = 100, nsim: int = 1000) -> str:
        """Return an R script that fits ``tsbcf`` from a prepared CSV file."""
        pkg = self.package_dir.as_posix()
        return f"""
if (!requireNamespace("data.table", quietly = TRUE)) stop("data.table is required")
if (!requireNamespace("devtools", quietly = TRUE)) stop("devtools is required")
devtools::load_all("{pkg}", quiet = TRUE)
dat <- data.table::fread("{input_csv}")
y <- dat$y
pihat <- dat$pihat
z <- dat$z
tgt <- dat$tgt
xcols <- grep("^x", names(dat), value = TRUE)
x <- as.matrix(dat[, ..xcols])
fit <- tsbcf(
  y = y, pihat = pihat, z = z, tgt = tgt,
  x_control = x, x_moderate = x,
  nburn = {int(nburn)}, nsim = {int(nsim)}, verbose = FALSE
)
saveRDS(fit, "{output_rds}")
""".lstrip()

    def run_rscript(self, long_data: TSBCFLongData, workdir: str | Path | None = None,
                    nburn: int = 100, nsim: int = 1000) -> subprocess.CompletedProcess:
        """Write long data and execute the generated R script.

        This method requires local R dependencies and is therefore not used by
        the Python test suite.  The returned process object contains stdout and
        stderr for reproducibility logs.
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
            data_path = root / "tsbcf_input.csv"
            script_path = root / "run_tsbcf.R"
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
            np.savetxt(data_path, matrix, delimiter=",", header=",".join(header), comments="")
            script_path.write_text(
                self.script(nburn=nburn, nsim=nsim),
                encoding="utf-8",
            )
            return subprocess.run(
                [self.rscript, str(script_path)],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
        finally:
            if tmp is not None:
                tmp.cleanup()
