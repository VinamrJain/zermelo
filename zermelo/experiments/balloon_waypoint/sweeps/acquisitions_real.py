import dataclasses

from zermelo.experiments.balloon_waypoint.registry import (
    ARRIVAL_RADIUS_KM,
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    N_CANDIDATES,
    N_SEEDS,
    OPENING_LEGS,
    PERTURBED,
    PLANNING_BUDGET,
    RESOURCES,
    matched_belief,
)
from zermelo.experiments.balloon_waypoint.schema import MethodConfig
from zermelo.experiments.balloon_waypoint.setup import (
    Implementation,
    arm,
    expected_improvement,
    max_magnitude,
    posterior_spread,
    sweep,
    uniform,
    upper_confidence,
    value_iteration,
)

BELIEF = matched_belief(PERTURBED)
"""The same model on every arm, matched to the truth"""


def _method(
    utility: Implementation, *, improvement: bool, n_fields: int, n_walks: int, step_rate: float, cost_weight: float
) -> MethodConfig:
    """This sweep's held values"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(max_steps=PLANNING_BUDGET, radius=ARRIVAL_RADIUS_KM, target_chunk=None, cost_weight=cost_weight),
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
    "acquisitions_real",
    problem=PERTURBED,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    seeds=range(N_SEEDS),
    resources=dataclasses.replace(RESOURCES, mem_gb=16, timeout_min=480, array_parallelism=40),
    arms=[
        arm("random_act", None, BELIEF),  # no rule at all: acts uniformly and claims the prior
        arm("random_search", _method(uniform(), improvement=False, n_fields=0, n_walks=0, step_rate=0.0, cost_weight=0.0), BELIEF),
        arm("max_variance", _method(posterior_spread(), improvement=False, n_fields=0, n_walks=0, step_rate=0.0, cost_weight=0.0), BELIEF),
        arm("ucb2", _method(upper_confidence(2.0), improvement=False, n_fields=0, n_walks=0, step_rate=0.0, cost_weight=0.0), BELIEF),
        arm("ei", _method(expected_improvement(), improvement=False, n_fields=0, n_walks=0, step_rate=0.0, cost_weight=0.0), BELIEF),
        arm("mc_ei16", _method(max_magnitude(), improvement=True, n_fields=16, n_walks=0, step_rate=0.0, cost_weight=0.0), BELIEF),
        arm("thompson", _method(max_magnitude(), improvement=False, n_fields=1, n_walks=0, step_rate=0.0, cost_weight=0.0), BELIEF),
        # the rule this study proposes: a step costs 1 + 0.25 * the resource it spends
        arm(
            "eui_fields16_walks16_rate0p5",
            _method(max_magnitude(), improvement=True, n_fields=16, n_walks=16, step_rate=0.5, cost_weight=0.25),
            BELIEF,
        ),
        # the same rule charging the step alone, isolating what the resource term buys
        arm(
            "eui_fields16_walks16_rate0p5_steps",
            _method(max_magnitude(), improvement=True, n_fields=16, n_walks=16, step_rate=0.5, cost_weight=0.0),
            BELIEF,
        ),
    ],
)
