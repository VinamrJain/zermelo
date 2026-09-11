"""The world every study runs on, and the numbers every sweep of it holds equal"""

import dataclasses

from zermelo.experiments.balloon_waypoint.schema import BeliefConfig, ProblemConfig, Resources
from zermelo.experiments.balloon_waypoint.setup import column_peak_speed

N_ALTITUDES = 5
"""Altitudes the record holds, spanning 1.5 to 14.2 kilometres"""

IRMA_JOSE = ProblemConfig(
    wind_path="zermelo/problems/balloon/data/irma_jose.npz",
    frame=0,
    ballast_units=1000,
    step_hours=3.0,  # one cell per step at the record's mean speed of ten metres per second
    error_scale=0.0,
    error_lengthscale_km=400.0,  # about three and a half cells
    error_jitter=1e-4,
    readout="zermelo.problems.balloon.readout.ColumnWind",
    target=column_peak_speed(n_alt=N_ALTITUDES),
    margin_lat=2,
    margin_lon=2,
)
"""Two recorded hurricanes over the Atlantic, at the hour the forecast is issued, the forecast exact"""

PERTURBED = dataclasses.replace(IRMA_JOSE, error_scale=1.0)
"""The same world under a forecast wrong by a small error"""


def matched_belief(world: ProblemConfig) -> BeliefConfig:
    """A belief holding the world's own error numbers, starting from the forecast it was handed"""
    return BeliefConfig(
        oracle=False,
        kernel="gpjax.kernels.Matern52",
        lengthscale_km=world.error_lengthscale_km,
        lengthscale_altitude_km=3.0,  # the record's altitudes sit 1.8 to 4.3 kilometres apart
        lengthscale_ballast=1e6,  # ballast is observed, so set large enough that it doesn't matter
        amplitude=world.error_scale,
        noise=1e-2,
        forecast_prior=True,
        n_features=256,
        refit=False,
        refit_steps=100,  # unread while refitting is off, and stated anyway
    )


def oracle_belief(world: ProblemConfig) -> BeliefConfig:
    """The true field itself, the fitted settings stated and unread"""
    return dataclasses.replace(matched_belief(world), oracle=True)


PLANNING_BUDGET = 25
"""`L`: a replan fires every `L` moves, costs `L` backups, and truncates a hitting time at `L` steps"""

ARRIVAL_RADIUS_KM = 60.0
"""`rho`: how near a waypoint counts as arrived, inside cells about a hundred kilometres apart"""

OPENING_LEGS = 1
"""Waypoint legs of uniform random walking taken before the rule starts"""

HORIZON_LENGTH = 1000
"""`T`: moves an episode makes, at three hours a move"""

CLAIM_EVERY_STEPS = 1
"""Moves between re-readings of the claim off a belief"""

N_CANDIDATES = None
"""Candidates scored per move (None scores every one the grid and its altitudes hold)"""

N_SEEDS = 1
"""Instances every arm is run on, seeded `0` to `N_SEEDS - 1`"""

RESOURCES = Resources(cpus=2, mem_gb=16, timeout_min=360, gres="gpu:1", partition="gpu", constraint="avx2&gpu-high", array_parallelism=64)
"""What one cell of a sweep is given. A sweep wanting more states its own with `dataclasses.replace`.

`avx2` and `gpu-high` are node features. Dropping either lands cells on nodes where jaxlib fails to
import or a matrix multiply fails to launch. A cell wanting no device states `avx2` alone"""
