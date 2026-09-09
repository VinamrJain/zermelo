"""The world every study runs on, and the numbers every sweep of it holds equal"""

from zermelo.experiments.balloon_waypoint.schema import BeliefConfig, ProblemConfig, Resources
from zermelo.experiments.balloon_waypoint.setup import speed_at

IRMA_JOSE = ProblemConfig(
    wind_path="zermelo/problems/balloon/data/irma_jose.npz",
    frame=0,
    ballast_units=8,
    step_hours=3.0,
    error_scale=3.0,  # metres per second, against winds that reach tens
    error_lengthscale_km=400.0,
    error_jitter=1e-4,
    readout="zermelo.problems.balloon.readout.PointWind",
    target=speed_at(altitude=0),
    margin_lat=2,
    margin_lon=2,
)
"""Two recorded hurricanes over the Atlantic, at the hour the forecast is issued"""


def matched_belief(world: ProblemConfig) -> BeliefConfig:
    """A belief holding the world's own error numbers, starting from the forecast it was handed"""
    return BeliefConfig(
        oracle=False,
        kernel="gpjax.kernels.Matern52",
        lengthscale_km=world.error_lengthscale_km,
        lengthscale_altitude_km=3.0,  # the record's altitudes span about thirteen kilometres in five
        lengthscale_ballast=1e6,  # ballast is observed, so set large enough that it doesn't matter
        amplitude=world.error_scale,
        noise=1e-2,
        forecast_prior=True,
        n_features=256,
        refit=False,
        refit_steps=100,  # unread while refitting is off, and stated anyway
    )


PLANNING_BUDGET = 20
"""`L`: a replan fires every `L` moves, costs `L` backups, and truncates a hitting time at `L` steps"""

ARRIVAL_RADIUS_KM = 60.0
"""`rho`: how near a waypoint counts as arrived, against cells about a hundred kilometres apart"""

OPENING_LEGS = 1
"""Waypoint legs of uniform random walking taken before the rule starts"""

HORIZON_LENGTH = 120
"""`T`: moves an episode makes, at three hours a move"""

CLAIM_EVERY_STEPS = 10
"""Moves between re-readings of the claim off a belief"""

CANDIDATE_CAP = 256
"""Candidates scored per move, drawn from the tens of thousands the grid and its ladder hold"""

N_SEEDS = 10
"""Instances every arm is run on, seeded `0` to `N_SEEDS - 1`"""

RESOURCES = Resources(cpus=2, mem_gb=16, timeout_min=360, gres="gpu:1", partition="gpu", constraint="avx2&gpu-high", array_parallelism=64)
"""What one cell of a sweep is given. A sweep wanting more states its own with `dataclasses.replace`.

`avx2` and `gpu-high` are node features. Dropping either lands cells on nodes where jaxlib fails to
import or a matrix multiply fails to launch. A cell wanting no device states `avx2` alone"""
