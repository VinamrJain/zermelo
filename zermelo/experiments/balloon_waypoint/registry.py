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

IRMA_2017SEP06 = dataclasses.replace(ATLANTIC_2017SEP, time_varying=False, start_hour=132.0)
"""The one frame at 2017-09-06 12Z, Irma inside the box, held still for the whole episode"""


STILL_FRAMES = {"sep06": 132.0, "sep09": 204.0, "sep19": 444.0, "sep24": 564.0}
"""`start_hour` of the 12Z frames of September 2017 the paper's still-wind studies run on, keyed by date"""

TUNING_FRAME = 324.0
"""`start_hour` of 2017-09-14 12Z, the still frame settings are picked on, apart from the ones reported"""

STILL_HORIZON = 240
"""time_steps in an episode on a still frame"""


def still(start_hour: float, resource_units: int) -> ProblemConfig:
    """The frame at `start_hour` held still for the whole episode, with `resource_units` altitude changes to spend"""
    return dataclasses.replace(ATLANTIC_2017SEP, time_varying=False, start_hour=start_hour, resource_units=resource_units)


def _belief(forecast_as_prior_mean: bool, place_km: float, altitude_km: float, hours: float, amplitude: float) -> BeliefConfig:
    """A belief at the correlation lengths and spread of what it learns, no refitting"""
    return BeliefConfig(
        oracle=False,
        forecast_as_prior_mean=forecast_as_prior_mean,
        # three sphere coordinates in km, altitude in km, then the hour a reading was taken at
        kernel_factors=[
            BeliefKernelConfig(kernel="gpjax.kernels.Matern52", parts=["position", "altitude"], lengthscale=[place_km] * 3 + [altitude_km]),
            BeliefKernelConfig(kernel="gpjax.kernels.Matern32", parts=["hours_elapsed"], lengthscale=[hours]),
        ],
        amplitude=amplitude,
        noise=1e-2,
        n_features=256,
        refit=False,
        refit_steps=100,  # unread while refitting is off, and stated anyway
    )


# Correlation lengths and standard deviations measured over the candidate box, every altitude, September 2017
def error_belief() -> BeliefConfig:
    """Learns W - F about the forecast: correlated over 175 km, 1.7 km of altitude and 4 hours, spread 2.64 m/s"""
    return _belief(True, place_km=175.0, altitude_km=1.7, hours=4.0, amplitude=2.64)


def wind_belief() -> BeliefConfig:
    """Learns W about zero: correlated over 560 km, 6.5 km of altitude and 27 hours, spread 8.75 m/s"""
    return _belief(False, place_km=560.0, altitude_km=6.5, hours=27.0, amplitude=8.75)


def static_error_belief() -> BeliefConfig:
    """The error belief over position and altitude alone, for a wind that does not move"""
    return dataclasses.replace(error_belief(), kernel_factors=error_belief().kernel_factors[:1])


def oracle_belief() -> BeliefConfig:
    """The true wind itself, the fitted settings stated and unread"""
    return dataclasses.replace(wind_belief(), oracle=True)


PLANNING_BUDGET = 50
"""`L`: a replan fires every `L` moves, costs `L` backups, and truncates a hitting time at `L` steps"""

TARGET_CHUNK = 500
"""Candidates one planner solve handles"""

ARRIVAL_RADIUS_KM = 60.0
"""`rho`: how near a waypoint counts as arrived, inside positions about 100 km apart"""

OPENING_STEPS = 0
"""time_steps of uniform random walking taken before the rule starts: none, the forecast being the prior"""

HORIZON_LENGTH = 238
"""time_steps in an episode: 714 hours at 3 hours each, from the data's first frame to its last"""

CLAIM_EVERY_STEPS = 1
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

RESOURCES = Resources(
    cpus=2,
    mem_gb=16,
    timeout_min=120,
    gres="gpu:1",
    partition="default_partition",
    constraint="avx2&(gpu-high|gpu-mid)",
    array_parallelism=64,
)
"""What one cell of a sweep is given; `avx2` and the GPU classes are node features. A sweep wanting more states its own with `dataclasses.replace`"""
