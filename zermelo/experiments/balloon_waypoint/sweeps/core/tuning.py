import dataclasses

from zermelo.experiments.balloon_waypoint.registry import (
    ARRIVAL_RADIUS_KM,
    ATLANTIC_2017SEP,
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    N_CANDIDATES,
    RECORDED,
    RESOURCES,
    TARGET_CHUNK,
    error_belief,
    wind_belief,
)
from zermelo.experiments.balloon_waypoint.schema import MethodConfig
from zermelo.experiments.balloon_waypoint.setup import Implementation, arm, sum_magnitude, sweep, value_iteration

BELIEFS = {"error": error_belief(), "wind": wind_belief()}
"""Learning the forecast's error about the forecast, or the wind about zero"""


def _method(
    utility: Implementation,
    *,
    subtract_dataset_utility: bool,
    n_field_samples: int,
    n_paths_per_field: int,
    lambda_: float,
    replan_length: int,
) -> MethodConfig:
    """One acquisition on a value-iteration planner, at the settings the acquisitions differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_length=replan_length, radius=ARRIVAL_RADIUS_KM, targets_per_solve=TARGET_CHUNK),
        subtract_dataset_utility=subtract_dataset_utility,
        n_field_samples=n_field_samples,
        n_paths_per_field=n_paths_per_field,
        lambda_=lambda_,
        n_candidates_scored=N_CANDIDATES,
    )


sweep(
    "tuning",
    problem=ATLANTIC_2017SEP,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=[0],
    resources=dataclasses.replace(RESOURCES, partition="default_partition"),
    arms=[
        arm(
            f"esi_L{replan_length}_wind_sr{str(lambda_).replace('.', 'p')}",
            _method(
                sum_magnitude(),
                subtract_dataset_utility=True,
                n_field_samples=16,
                n_paths_per_field=16,
                lambda_=lambda_,
                replan_length=replan_length,
            ),
            BELIEFS["wind"],
        )
        for replan_length in (10, 25, 50)
        for lambda_ in (0.0, 0.25, 0.5, 0.75)
    ],
)
"""Expected sum-of-speed improvement under the wind belief, over the leg length and the price of an altitude change"""
