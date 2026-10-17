import dataclasses

from zermelo.experiments.ambient_waypoint.registry import (
    AMBIENT1_CONTROL1,
    CLAIM_EVERY_STEPS,
    OPENING_STEPS,
    PLANNING_BUDGET,
    RECORDED,
    RESOURCES,
    matched_belief,
)
from zermelo.experiments.ambient_waypoint.schema import MethodConfig
from zermelo.experiments.ambient_waypoint.setup import Implementation, arm, max_magnitude, sweep, uniform, value_iteration

HORIZON = 500
"""Moves an episode makes here: ten planning budgets"""


def _method(
    utility: Implementation, *, subtract_dataset_utility: bool, n_field_samples: int, n_paths_per_field: int, lambda_: float
) -> MethodConfig:
    """This sweep's held values"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_length=PLANNING_BUDGET, radius=0.5, targets_per_solve=None),
        subtract_dataset_utility=subtract_dataset_utility,
        n_field_samples=n_field_samples,
        n_paths_per_field=n_paths_per_field,
        lambda_=lambda_,
        n_candidates_scored=None,
        random_opening_steps=OPENING_STEPS,
    )


for name, resources in (
    ("device_cpu", dataclasses.replace(RESOURCES, gres=None, partition="dean", constraint="avx2", timeout_min=60, array_parallelism=2)),
    ("device_gpu", dataclasses.replace(RESOURCES, gres="gpu:1", partition="gpu", timeout_min=60, array_parallelism=2)),
):
    sweep(
        name,
        problem=AMBIENT1_CONTROL1,
        horizon=HORIZON,
        claim_every=CLAIM_EVERY_STEPS,
        recorded=RECORDED,
        seeds=[0],
        resources=resources,
        arms=[
            arm(
                "eui_fields16_walks16_rate0p5",  # the pathwise sampler runs on this arm
                _method(max_magnitude(), subtract_dataset_utility=True, n_field_samples=16, n_paths_per_field=16, lambda_=0.5),
                matched_belief(AMBIENT1_CONTROL1),
            ),
            # no draws and no walks: this arm times the planner alone
            arm(
                "random_search",
                _method(uniform(), subtract_dataset_utility=False, n_field_samples=0, n_paths_per_field=0, lambda_=0.0),
                matched_belief(AMBIENT1_CONTROL1),
            ),
        ],
    )
