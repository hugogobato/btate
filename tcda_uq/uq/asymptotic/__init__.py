"""Asymptotic confidence bands for the mean estimands (C1, Phase 2).

Each band consumes the cross-fitted EIF process from
:func:`tcda_uq.estimators.cross_fit` (Theorem 5.2: its covariance is the limiting
Gaussian covariance) and exposes a uniform ``*_band(...) -> tcda_uq.metrics.Band``:

  * :mod:`.covariance`           -- shared EIF covariance / pointwise variance (2.4).
  * :mod:`.multiplier_bootstrap` -- Gaussian/Rademacher multiplier band (TATE Cor 5.4).
  * :mod:`.pini_vantini`         -- interval-wise testing (IWT) band.
  * :mod:`.liebl_reimherr`       -- fair pivotal bands via the R ``ffscb`` package.
"""

from .covariance import (
    eif_covariance,
    eif_pointwise_variance,
    eif_pointwise_sd,
    eif_correlation,
)
from .multiplier_bootstrap import (
    multiplier_bootstrap_band,
    multiplier_bootstrap_bands,
    topological_effect_test,
)
from .pini_vantini import (
    iwt_pvalues,
    IWTResult,
    pini_vantini_band,
    pini_vantini_bands,
)
from .liebl_reimherr import (
    liebl_reimherr_band,
    liebl_reimherr_bands,
    cov2tau,
    r_backend_available,
)
from .ctate_bands import (
    ctate_confidence_band,
    ctate_confidence_bands,
    ctate_pointwise_sd,
)

__all__ = [
    "eif_covariance",
    "eif_pointwise_variance",
    "eif_pointwise_sd",
    "eif_correlation",
    "multiplier_bootstrap_band",
    "multiplier_bootstrap_bands",
    "topological_effect_test",
    "iwt_pvalues",
    "IWTResult",
    "pini_vantini_band",
    "pini_vantini_bands",
    "liebl_reimherr_band",
    "liebl_reimherr_bands",
    "cov2tau",
    "r_backend_available",
    "ctate_confidence_band",
    "ctate_confidence_bands",
    "ctate_pointwise_sd",
]
