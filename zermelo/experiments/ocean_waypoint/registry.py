"""The world every study runs on, and the numbers every sweep of it holds equal"""

import dataclasses

from zermelo.experiments.ocean_waypoint.schema import BeliefConfig, BeliefKernelConfig, ProblemConfig, Resources

GULF_STREAM_2019JUN = ProblemConfig(
    data_path="zermelo/problems/ocean/data/gulf_stream_2019jun.npz",
    grid_stride=2,  # 1/6 degree: about 18 km between grid points
    frame_hour=360.0,  # 2019-06-16 00Z
    forecast_lag_hours=48.0,  # F is the ocean current two days before
    step_hours=3.0,
    n_headings=8,
    top_speed_ms=2.5,  # about 5 knots in a strong wind, on a broad reach
    no_go_degrees=45.0,
    peak_degrees=120.0,
    rise_exponent=0.4,
    downwind_factor=0.8,
    wind_scale_ms=3.5,  # 7 knots: 0.58 of top speed at 6 knots, 0.94 at 20
    noise_ms=0.1,
    error_scale=0.0,
    error_lengthscale_km=100.0,
    error_jitter=1e-4,
    lat_min=35.0,
    lat_max=42.0,
    lon_min=-75.0,
    lon_max=-55.0,
    target_lat_min=36.4,  # the middle three fifths of the box, so no target sits on its rim
    target_lat_max=40.6,
    target_lon_min=-71.0,
    target_lon_max=-59.0,
    start_lat=37.0,
    start_lon=-65.0,
    arrival_km=20.0,
    objective="reach",
)
"""The Gulf Stream after Cape Hatteras, one frame of June 2019 held still, a sailboat of Saildrone speed"""


def _belief(forecast_as_prior_mean: bool, place_km: float, amplitude: float) -> BeliefConfig:
    """A belief at the correlation length and spread of what it learns, no refitting"""
    return BeliefConfig(
        oracle=False,
        forecast_as_prior_mean=forecast_as_prior_mean,
        kernel_factors=[BeliefKernelConfig(kernel="gpjax.kernels.Matern52", parts=["position"], lengthscale=[place_km] * 2)],
        amplitude=amplitude,
        noise=0.1,
        n_features=256,
        refit=False,
        refit_steps=100,  # unread while refitting is off, and stated anyway
    )


# Correlation lengths and standard deviations measured over the box on the 2019-06-16 00Z frame
def error_belief() -> BeliefConfig:
    """Learns W - F about the forecast: correlated over 40 km, spread 0.38 m/s"""
    return _belief(True, place_km=40.0, amplitude=0.38)


def ocean_current_belief() -> BeliefConfig:
    """Learns W about zero: correlated over 55 km, spread 0.48 m/s"""
    return _belief(False, place_km=55.0, amplitude=0.48)


def oracle_belief() -> BeliefConfig:
    """The true ocean current itself, the fitted settings stated and unread"""
    return dataclasses.replace(ocean_current_belief(), oracle=True)


PLANNING_BUDGET = 25
"""`L`: a waypoint is chosen anew after this many moves, and an imagined walk is rolled over at most as many"""

ARRIVAL_RADIUS_KM = 20.0
"""`rho`: how near a waypoint counts as arrived, inside positions about 18 km apart"""

HORIZON_LENGTH = 160
"""time_steps in an episode: 20 days at 3 hours each"""

CLAIM_EVERY_STEPS = 1
"""Moves between re-readings of the claim off a belief"""

RECORDED = {
    "state/position": 1,
    "state/target": 1,
    "objective_state/arrived": 1,
    "objective_state/steps_to_arrival": 1,
    "objective_state/truth": 1,
    "objective_state/claim": CLAIM_EVERY_STEPS,
    "agent_state/time_steps_since_waypoint": 1,
    "agent_state/scores.waypoint": 1,
    "agent_state/scores.imagined_walk_to_waypoint": 1,
    "agent_state/scores.candidate_indices": 1,
    "agent_state/scores.acquisition_value": 1,
}
"""What a record stores, and the moves between the snapshots it keeps of each. The forecast is read from `data_path` instead"""

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
