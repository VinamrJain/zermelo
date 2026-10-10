"""A boat sails an unknown ocean current to a target, steering by heading, its speed through the water set by a known wind.

state = (position, target, field)
    position    (lat, lon) degrees of the grid point the boat is at
    target      (lat, lon) degrees of the grid point to reach, drawn once per episode
    field       W, the ocean current carrying it: W(position) = (u, v) in m/s
action in {0, ..., n_headings}      a heading 2 pi k / n_headings from east, or drifting

position' ~ the grid points around a step along (through_water + W), step_hours long
W = F + forecast_error              the truth is the forecast F the boat is given, plus the error it carries
reward = -1 until within arrival_km of target, 0 from then on; or the nats learned about W over a region

A plan and a belief are written over position alone
"""

from zermelo.problems.ocean.field import FixedError, ForecastPrior, GaussianError, OceanCurrent, OceanCurrentError
from zermelo.problems.ocean.grid import KM_PER_DEGREE, PlaneGrid
from zermelo.problems.ocean.objective import LearnRegion, ReachTarget
from zermelo.problems.ocean.readout import PointOceanCurrent
from zermelo.problems.ocean.transition import Act, OceanKernel, OceanTransition, Polar
from zermelo.problems.ocean.world import (
    OceanData,
    learn_objective,
    load_ocean,
    ocean_objective,
    ocean_states,
    ocean_transition,
    ocean_world,
)

__all__ = [
    "KM_PER_DEGREE",
    "Act",
    "OceanCurrentError",
    "OceanCurrent",
    "ReachTarget",
    "LearnRegion",
    "FixedError",
    "ForecastPrior",
    "GaussianError",
    "OceanData",
    "OceanKernel",
    "OceanTransition",
    "PlaneGrid",
    "PointOceanCurrent",
    "Polar",
    "learn_objective",
    "load_ocean",
    "ocean_objective",
    "ocean_states",
    "ocean_transition",
    "ocean_world",
]
