"""Point estimators for topological treatment effects.

Ported from ``top-causal-effect-main/estimators.py`` (mean PI / IPW / AIPW) and
extended with per-unit efficient-influence-function (EIF) processes -- the hook
that Phase 2 UQ (multiplier bootstrap, functional bands) attaches to.
"""

from .aipw import (
    ipw_estimator,
    plugin_estimator,
    aipw_estimator,
    aipw_scores,
    aipw_influence,
    EPS_PI,
)
from .nuisance import (
    fit_functional_regression,
    predict_functional_regression,
    fit_propensity,
    cross_fit,
    NuisanceFit,
    CrossFitResult,
)
from .ctate_dr_learner import (
    CTATEDRLearner,
    SecondStageFit,
)

__all__ = [
    "ipw_estimator",
    "plugin_estimator",
    "aipw_estimator",
    "aipw_scores",
    "aipw_influence",
    "EPS_PI",
    "fit_functional_regression",
    "predict_functional_regression",
    "fit_propensity",
    "cross_fit",
    "NuisanceFit",
    "CrossFitResult",
    "CTATEDRLearner",
    "SecondStageFit",
]
