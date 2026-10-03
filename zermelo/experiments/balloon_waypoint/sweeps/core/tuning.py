import dataclasses

from zermelo.experiments.balloon_waypoint.registry import (
    ARRIVAL_RADIUS_KM,
    ATLANTIC_2017SEP,
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    N_CANDIDATES,
    N_SEEDS,
    OPENING_LEGS,
    PLANNING_BUDGET,
    RECORDED,
    RESOURCES,
    error_belief,
    wind_belief,
)
from zermelo.experiments.balloon_waypoint.schema import MethodConfig
from zermelo.experiments.balloon_waypoint.setup import Implementation, arm, sum_magnitude, sweep, total_variance, value_iteration

BELIEFS = {"error": error_belief(), "wind": wind_belief()}
"""Learning the forecast's error about the forecast, or the wind about zero"""


def _method(
    utility: Implementation, *, improvement: bool, n_fields: int, n_walks: int, step_rate: float, replan_every: int, cost_weight: float
) -> MethodConfig:
    """One acquisition on a value-iteration planner, at the settings the acquisitions differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_every=replan_every, radius=ARRIVAL_RADIUS_KM, target_chunk=None, cost_weight=cost_weight),
        improvement=improvement,
        n_fields=n_fields,
        n_walks=n_walks,
        combination="linear",
        step_rate=step_rate,
        steps_from="predicted",
        n_candidates=N_CANDIDATES,
        opening_legs=OPENING_LEGS,
    )


sweep(
    "tuning",
    problem=ATLANTIC_2017SEP,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=RESOURCES,
    arms=[
        arm(
            f"esi_L8_{name}_cw{str(cost_weight).replace('.', 'p')}_sr{str(step_rate).replace('.', 'p')}",
            _method(
                sum_magnitude(), improvement=True, n_fields=16, n_walks=16, step_rate=step_rate, replan_every=8, cost_weight=cost_weight
            ),
            belief,
        )
        for name, belief in BELIEFS.items()
        for cost_weight in (0.0, 0.25, 1.0)
        for step_rate in (0.25, 0.5, 1.0)
    ],
)
"""Expected sum-of-speed improvement at a leg of 8 time_steps, over the price of an altitude change and the price of travel"""
