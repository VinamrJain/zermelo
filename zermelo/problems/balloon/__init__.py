"""A balloon drifts in an unknown wind, steering only by which altitude it flies at.

s = (lat, lon, p, b, W)             where it is, which altitude, what ballast is left, the wind carrying it
a in {0, 1, 2}                      down one altitude, hold, up one
W(lat, lon, p) = (u, v)             metres per second, u eastward and v northward
(lat, lon)' = (lat, lon) + W(s) h   the wind carries it, h hours to a step
p' = p + a - 1, b' = b - |p' - p|   an altitude change costs one unit of ballast
W = F + e                           the truth is the forecast F the agent is given, plus a drawn error e
"""

from zermelo.problems.balloon.field import DriftingWind, ForecastPrior, GaussianError, GriddedWind, WindError, WindField
from zermelo.problems.balloon.grid import SphereGrid, Steps, balloon_states
from zermelo.problems.balloon.objective import ColumnPeakSpeed, PointSpeed, StormSearch, Target, balloon_candidates
from zermelo.problems.balloon.readout import BalloonReadout, ColumnWind, PointWind
from zermelo.problems.balloon.transition import Act, Advection, Ascent, BalloonKernel, BalloonTransition, Expenditure, Factor
from zermelo.problems.balloon.world import Highest, WindRecord, balloon_objective, balloon_transition, balloon_world, load_wind

__all__ = [
    "Act",
    "Advection",
    "Ascent",
    "BalloonKernel",
    "BalloonReadout",
    "BalloonTransition",
    "ColumnPeakSpeed",
    "ColumnWind",
    "DriftingWind",
    "Expenditure",
    "Factor",
    "ForecastPrior",
    "GaussianError",
    "GriddedWind",
    "Highest",
    "PointSpeed",
    "PointWind",
    "SphereGrid",
    "Steps",
    "StormSearch",
    "Target",
    "WindError",
    "WindField",
    "WindRecord",
    "balloon_candidates",
    "balloon_objective",
    "balloon_states",
    "balloon_transition",
    "balloon_world",
    "load_wind",
]
