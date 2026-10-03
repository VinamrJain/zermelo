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
    utility: Implementation, *, improvement: bool, n_fields: int, n_walks: int, step_rate: float, replan_every: int, cost_weight: float
) -> MethodConfig:
    """One acquisition on a value-iteration planner, at the settings the acquisitions differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_every=replan_every, radius=ARRIVAL_RADIUS_KM, target_chunk=TARGET_CHUNK, cost_weight=cost_weight),
        improvement=improvement,
        n_fields=n_fields,
        n_walks=n_walks,
        combination="linear",
        step_rate=step_rate,
        steps_from="predicted",
        n_candidates=N_CANDIDATES,
        opening_steps=OPENING_STEPS,
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
            f"esi_L{replan_every}_wind_cw{str(cost_weight).replace('.', 'p')}_sr{str(step_rate).replace('.', 'p')}",
            _method(
                sum_magnitude(),
                improvement=True,
                n_fields=16,
                n_walks=16,
                step_rate=step_rate,
                replan_every=replan_every,
                cost_weight=cost_weight,
            ),
            BELIEFS["wind"],
        )
        for replan_every in (10, 25, 50)
        for cost_weight in (0.0, 0.5, 1.0, 2.0)
        for step_rate in (0.0, 0.5, 1.0, 2.0)
    ],
)
"""Expected sum-of-speed improvement under the wind belief, over the leg length, the price of an altitude change and the price of travel"""
