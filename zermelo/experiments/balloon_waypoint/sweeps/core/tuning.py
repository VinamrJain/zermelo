import dataclasses

from zermelo.experiments.balloon_waypoint.registry import (
    ARRIVAL_RADIUS_KM,
    ATLANTIC_2017SEP,
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    N_CANDIDATES,
    OPENING_STEPS,
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
    penalty_rate: float,
    replan_length: int,
    cost_weight: float,
) -> MethodConfig:
    """One acquisition on a value-iteration planner, at the settings the acquisitions differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(
            replan_length=replan_length, radius=ARRIVAL_RADIUS_KM, targets_per_solve=TARGET_CHUNK, cost_weight=cost_weight
        ),
        subtract_dataset_utility=subtract_dataset_utility,
        n_field_samples=n_field_samples,
        n_paths_per_field=n_paths_per_field,
        penalty_rate=penalty_rate,
        n_candidates_scored=N_CANDIDATES,
        random_opening_steps=OPENING_STEPS,
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
            f"esi_L{replan_length}_wind_cw{str(cost_weight).replace('.', 'p')}_sr{str(penalty_rate).replace('.', 'p')}",
            _method(
                sum_magnitude(),
                subtract_dataset_utility=True,
                n_field_samples=16,
                n_paths_per_field=16,
                penalty_rate=penalty_rate,
                replan_length=replan_length,
                cost_weight=cost_weight,
            ),
            BELIEFS["wind"],
        )
        for replan_length in (10, 25, 50)
        for cost_weight in (0.0, 0.5, 1.0, 2.0)
        for penalty_rate in (0.0, 0.5, 1.0, 2.0)
    ],
)
"""Expected sum-of-speed improvement under the wind belief, over the leg length, the price of an altitude change and the price of travel"""
