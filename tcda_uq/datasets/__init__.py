"""Datasets: synthetic generators + real-data loaders.

  * :mod:`.orbit`       -- ORBIT linked-twist-map testbed (numpy port).
  * :mod:`.simulation`  -- tri-oracle harness exposing TATE/CTATE/ITTE truth.
  * :mod:`.sarscov2`    -- SARS-CoV-2 CT loader (needs the ``[data]`` extra).
"""

from .orbit import gen_orbits, make_orbit_causal
from .simulation import TriOracleSimulation, SimulationSample
from .covariates import gen_covariate, gen_trt_prob

__all__ = [
    "gen_orbits",
    "make_orbit_causal",
    "TriOracleSimulation",
    "SimulationSample",
    "gen_covariate",
    "gen_trt_prob",
]
