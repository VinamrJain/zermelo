from typing import Any

from zermelo.experiments.ambient_waypoint.registry import (
    AMBIENT1_CONTROL1,
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    N_SEEDS,
    OPENING_STEPS,
    PLANNING_BUDGET,
    RECORDED,
    RESOURCES,
    matched_belief,
)
from zermelo.experiments.ambient_waypoint.schema import MethodConfig
from zermelo.experiments.ambient_waypoint.setup import Implementation, arm, greedy, max_magnitude, sweep, value_iteration

BELIEF = matched_belief(AMBIENT1_CONTROL1)


def _method(planner: Implementation) -> MethodConfig:
    """This sweep's held values"""
    return MethodConfig(
        utility=max_magnitude(),
        planner=planner,
        subtract_dataset_utility=True,
        n_field_samples=16,
        n_paths_per_field=16,
        penalty_rate=0.5,
        n_candidates_scored=None,
        random_opening_steps=OPENING_STEPS,
    )


PLANNERS: dict[str, Any] = {"value_iteration": value_iteration, "greedy": greedy}

sweep(
    "planners",
    problem=AMBIENT1_CONTROL1,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=RESOURCES,
    arms=[
        arm(name, _method(build(replan_length=PLANNING_BUDGET, radius=0.5, targets_per_solve=None)), BELIEF)
        for name, build in PLANNERS.items()
    ],
)
