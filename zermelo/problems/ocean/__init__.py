"""A boat sails an unknown ocean current under an unknown wind, steering by heading, its speed through the water set by the wind.

state = (position, target, field)
    position    (lat, lon) degrees of the grid point the boat is at
    target      (lat, lon) degrees of the grid point to reach, drawn once per episode
    field       (W, U) the ocean current carrying it and the wind driving it: field(position) = (W_u, W_v, U_u, U_v) in m/s
action in {0, ..., n_headings}      a heading 2 pi k / n_headings from east, or drifting

position' ~ the grid points around a step along (through_water(U) + W), step_hours long
field = forecast + forecast_error   the truth is the forecast the boat is given, plus the error it carries
reward = -1 until within arrival_km of target, 0 from then on; or the nats learned about the field over a region

A plan and a belief are written over position alone
"""

from zermelo.problems.ocean.field import COMPONENTS, OCEAN_CURRENT, WIND, FixedError, ForecastError, ForecastPrior, GaussianError, GridField
from zermelo.problems.ocean.grid import KM_PER_DEGREE, PlaneGrid
from zermelo.problems.ocean.objective import LearnRegion, ReachTarget, SearchMagnitude, VisitTargets
from zermelo.problems.ocean.readout import PointField
from zermelo.problems.ocean.transition import Act, OceanKernel, OceanTransition, Polar
from zermelo.problems.ocean.world import (
    OceanData,
    learn_objective,
    load_ocean,
    ocean_objective,
    ocean_states,
    ocean_transition,
    ocean_world,
    search_objective,
    visit_objective,
)

__all__ = [
    "COMPONENTS",
    "KM_PER_DEGREE",
    "OCEAN_CURRENT",
    "WIND",
    "Act",
    "ForecastError",
    "GridField",
    "ReachTarget",
    "LearnRegion",
    "SearchMagnitude",
    "VisitTargets",
    "FixedError",
    "ForecastPrior",
    "GaussianError",
    "OceanData",
    "OceanKernel",
    "OceanTransition",
    "PlaneGrid",
    "PointField",
    "Polar",
    "learn_objective",
    "search_objective",
    "visit_objective",
    "load_ocean",
    "ocean_objective",
    "ocean_states",
    "ocean_transition",
    "ocean_world",
]
