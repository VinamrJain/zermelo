"""The world every study runs on, and the numbers every sweep of it holds equal"""

import dataclasses

from zermelo.experiments.balloon_waypoint.schema import BeliefConfig, BeliefKernelConfig, ProblemConfig, Resources
from zermelo.experiments.balloon_waypoint.setup import point_speed

ATLANTIC_2017SEP = ProblemConfig(
    wind_path="zermelo/problems/balloon/data/global_2017sep.npz",
    grid_stride=1,
    hour_stride=1,
    start_hour=3.0,  # the data's first frame, 2017-09-01 03Z
    time_varying=True,
    forecast="GEFS",
    resource_units=80,
    step_hours=3.0,
    error_scale=0.0,
    error_lengthscale_km=400.0,
    error_jitter=1e-4,
    target=point_speed(),
    lat_min=14.0,
    lat_max=30.0,
    lon_min=-85.0,
    lon_max=-40.0,
)
"""The whole globe in September 2017, scored over the Atlantic box two hurricanes cross"""


def matched_belief() -> BeliefConfig:
    """A belief the size and reach of the forecast's own error"""
    return BeliefConfig(
        oracle=False,
        # three horizontal coordinates in km at the error's own correlation, then altitude in km
        kernel_factors=[
            BeliefKernelConfig(kernel="gpjax.kernels.Matern52", parts=["position", "altitude"], lengthscale=[300.0] * 3 + [3.0])
        ],
        amplitude=4.8,  # m/s, the error's spread
        noise=1e-2,
        n_features=256,
        refit=False,
        refit_steps=100,  # unread while refitting is off, and stated anyway
    )


def oracle_belief() -> BeliefConfig:
    """The true field itself, the fitted settings stated and unread"""
    return dataclasses.replace(matched_belief(), oracle=True)


PLANNING_BUDGET = 25
"""`L`: a replan fires every `L` moves, costs `L` backups, and truncates a hitting time at `L` steps"""

ARRIVAL_RADIUS_KM = 60.0
"""`rho`: how near a waypoint counts as arrived, inside positions about 100 km apart"""

OPENING_LEGS = 1
"""Waypoint legs of uniform random walking taken before the rule starts"""

HORIZON_LENGTH = 238
"""time_steps in an episode: 714 hours at 3 hours each, from the data's first frame to its last"""

CLAIM_EVERY_STEPS = PLANNING_BUDGET
"""Moves between re-readings of the claim off a belief"""

RECORDED = {
    "state/position": 1,
    "state/altitude": 1,
    "state/balloon_resource": 1,
    "state/hours_elapsed": 1,
    "objective_state/best_possible_speed": 1,
    "objective_state/speed": 1,
    "objective_state/truth": 1,
    "objective_state/claim": CLAIM_EVERY_STEPS,
    "agent_state/time_steps_since_waypoint": 1,
    "agent_state/scores.predicted_step_cost": 1,
    "agent_state/scores.frac_zero_value_candidates": 1,
    "agent_state/scores.frac_reachable_candidates": 1,
    "agent_state/scores.waypoint": 1,
    "agent_state/scores.imagined_walk_to_waypoint": 1,
    "agent_state/scores.candidate_indices": 1,
    "agent_state/scores.acquisition_value": 1,
}
"""What a record stores, and the moves between the snapshots it keeps of each. The forecast is read from `wind_path` instead"""

N_CANDIDATES = None
"""Candidates scored per move (None scores every one the grid and its altitudes hold)"""

N_SEEDS = 10
"""Instances every arm is run on, seeded `0` to `N_SEEDS - 1`"""

RESOURCES = Resources(cpus=2, mem_gb=12, timeout_min=30, gres="gpu:1", partition="gpu", constraint="avx2&gpu-high", array_parallelism=64)
"""What one cell of a sweep is given. A sweep wanting more states its own with `dataclasses.replace`.

`avx2` and `gpu-high` are node features. Dropping either lands cells on nodes where jaxlib fails to
import or a matrix multiply fails to launch. A cell wanting no device states `avx2` alone"""
