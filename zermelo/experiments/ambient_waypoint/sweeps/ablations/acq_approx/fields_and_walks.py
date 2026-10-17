import dataclasses

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
from zermelo.experiments.ambient_waypoint.setup import arm, max_magnitude, sweep, value_iteration

BELIEF = matched_belief(AMBIENT1_CONTROL1)


def _method(*, n_field_samples: int, n_paths_per_field: int) -> MethodConfig:
    """This sweep's held values"""
    return MethodConfig(
        utility=max_magnitude(),
        planner=value_iteration(replan_length=PLANNING_BUDGET, radius=0.5, targets_per_solve=None),
        subtract_dataset_utility=True,
        n_field_samples=n_field_samples,
        n_paths_per_field=n_paths_per_field,
        lambda_=0.5,
        n_candidates_scored=None,
        random_opening_steps=OPENING_STEPS,
    )


sweep(
    "fields_and_walks",
    problem=AMBIENT1_CONTROL1,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=dataclasses.replace(RESOURCES, timeout_min=720),
    arms=[arm(f"fields{s}_walks{m}", _method(n_field_samples=s, n_paths_per_field=m), BELIEF) for s in (0, 16) for m in (0, 16)],
)
