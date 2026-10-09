"""A boat sails an unknown current, steering by heading, its speed through the water set by a known wind.

state = (position, field)
    position    (lat, lon) degrees of the grid point the boat is at
    field       W, the current carrying it: W(position) = (u, v) in m/s
action in {0, ..., n_headings}      a heading 2 pi k / n_headings from east, or drifting

position' ~ the grid points around a step along (through_water + W), step_hours long
W = F + forecast_error              the truth is the forecast F the boat is given, plus the error it carries
reward = ||W(state)||               counted on a candidate, a latitude-longitude box of grid points

A plan and a belief are written over position alone
"""

from zermelo.problems.ocean.field import CurrentError, CurrentField, FixedError, ForecastPrior, GaussianError
from zermelo.problems.ocean.grid import KM_PER_DEGREE, PlaneGrid
from zermelo.problems.ocean.objective import CurrentSearch
from zermelo.problems.ocean.readout import PointCurrent
from zermelo.problems.ocean.transition import Act, OceanKernel, OceanTransition, Polar
from zermelo.problems.ocean.world import OceanData, load_ocean, ocean_objective, ocean_states, ocean_transition, ocean_world

__all__ = [
    "KM_PER_DEGREE",
    "Act",
    "CurrentError",
    "CurrentField",
    "CurrentSearch",
    "FixedError",
    "ForecastPrior",
    "GaussianError",
    "OceanData",
    "OceanKernel",
    "OceanTransition",
    "PlaneGrid",
    "PointCurrent",
    "Polar",
    "load_ocean",
    "ocean_objective",
    "ocean_states",
    "ocean_transition",
    "ocean_world",
]
