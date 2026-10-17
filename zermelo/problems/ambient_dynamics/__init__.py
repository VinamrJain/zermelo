"""An unknown field displaces an actor on a grid that it also steers.

z = (y, q)                                                        a position: ambient part, controllable part
y' = y + clip(f*(z) + sigma_flow * eps, +-drift_bound)              the latent field displaces
q' = q + clip(control_step * choice + sigma_act * eta, +-control_bound)   the actor steers
r  = f*(z) + sigma_obs * xi                                       a reading; the dataset is (z_i, r_i)
"""

from zermelo.problems.ambient_dynamics.field import GP, GPField
from zermelo.problems.ambient_dynamics.grid import GridDomain, PaddedGridDomain, PeriodicGridDomain, ambient_candidates, ambient_positions
from zermelo.problems.ambient_dynamics.objective import AmbientObjective
from zermelo.problems.ambient_dynamics.readout import LocalReading
from zermelo.problems.ambient_dynamics.transition import Act, AmbientKernel, AmbientTransition
from zermelo.problems.ambient_dynamics.world import AmbientPrior, ambient_world

__all__ = [
    "GP",
    "Act",
    "AmbientKernel",
    "AmbientObjective",
    "AmbientPrior",
    "AmbientTransition",
    "GPField",
    "GridDomain",
    "LocalReading",
    "PaddedGridDomain",
    "PeriodicGridDomain",
    "ambient_candidates",
    "ambient_positions",
    "ambient_world",
]
