"""A balloon drifts in an unknown wind around the globe, steering only by which altitude it flies at.

state = (position, altitude, balloon_resource, hours_elapsed, field)
    position            (lat, lon) degrees of the grid point the balloon is over
    altitude            which altitude, an index
    balloon_resource    altitude changes left
    hours_elapsed       hours since the episode began
    field               W, the wind carrying it: W(position, altitude, hours_elapsed) = (u, v) in m/s
action in {0, 1, 2}     down one altitude, hold, up one

position'         ~ the grid points around a great-circle step from position along W, step_hours long
altitude'         = altitude + action - 1, where balloon_resource > 0
balloon_resource' = balloon_resource - |altitude' - altitude|
hours_elapsed'    = hours_elapsed + step_hours
W = F + forecast_error          the truth is the forecast F the balloon is given, plus the error it carries
reward = ||W(state)||           counted on a candidate, a latitude-longitude box at every altitude

A plan and a belief are written over (position, altitude) alone
"""

from zermelo.problems.balloon.field import ForecastPrior, GaussianError, GEFSError, TimeVaryingWind, WindError, WindField
from zermelo.problems.balloon.grid import SphereGrid, Steps, balloon_states
from zermelo.problems.balloon.objective import ColumnPeakSpeed, PointSpeed, StormSearch, Target, balloon_candidates, candidate_box
from zermelo.problems.balloon.readout import PointWind
from zermelo.problems.balloon.transition import Act, Advection, Ascent, BalloonKernel, BalloonTransition, Factor
from zermelo.problems.balloon.world import Highest, WindData, balloon_objective, balloon_transition, balloon_world, load_wind

__all__ = [
    "Act",
    "Advection",
    "Ascent",
    "BalloonKernel",
    "BalloonTransition",
    "ColumnPeakSpeed",
    "Factor",
    "ForecastPrior",
    "GEFSError",
    "GaussianError",
    "Highest",
    "PointSpeed",
    "PointWind",
    "SphereGrid",
    "Steps",
    "StormSearch",
    "Target",
    "TimeVaryingWind",
    "WindData",
    "WindError",
    "WindField",
    "balloon_candidates",
    "balloon_objective",
    "balloon_states",
    "balloon_transition",
    "balloon_world",
    "candidate_box",
    "load_wind",
]
