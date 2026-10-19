import dataclasses

from zermelo.experiments.ambient_waypoint.registry import (
    AMBIENT1_CONTROL1,
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    N_SEEDS,
    PLANNING_BUDGET,
    RECORDED,
    RESOURCES,
    matched_belief,
)
from zermelo.experiments.ambient_waypoint.schema import MethodConfig
from zermelo.experiments.ambient_waypoint.setup import Implementation, arm, max_magnitude, sweep, uniform, value_iteration

WIDE = dataclasses.replace(AMBIENT1_CONTROL1, ambient_cells=100, controllable_cells=100)
"""n_states 120 * 104 = 12480 against 70 * 54 = 3780, and 10000 candidates against 2500"""

TARGET_CHUNK = 2500
"""Targets a plan solves at once on the wide grid: n_states * TARGET_CHUNK * n_actions floats memory per plan"""


def _method(
    utility: Implementation,
    *,
    subtract_dataset_utility: bool,
    n_field_samples: int,
    n_paths_per_field: int,
    lambda_: float,
    targets_per_solve: int | None,
) -> MethodConfig:
    """This sweep's held values"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_length=PLANNING_BUDGET, radius=0.5, targets_per_solve=targets_per_solve),
        subtract_dataset_utility=subtract_dataset_utility,
        n_field_samples=n_field_samples,
        n_paths_per_field=n_paths_per_field,
        lambda_=lambda_,
        n_candidates_scored=None,
    )


for name, problem, chunk, resources in (
    ("grid50", AMBIENT1_CONTROL1, None, RESOURCES),
    (
        "grid100",
        WIDE,
        TARGET_CHUNK,
        dataclasses.replace(RESOURCES, mem_gb=32, timeout_min=480, gres="gpu:1", partition="gpu", array_parallelism=8),
    ),
):
    sweep(
        name,
        problem=problem,
        horizon=HORIZON_LENGTH,
        claim_every=CLAIM_EVERY_STEPS,
        recorded=RECORDED,
        seeds=range(N_SEEDS),
        resources=resources,
        arms=[
            arm(
                "eui_fields16_walks16_rate0p5",  # the pathwise sampler runs on this arm
                _method(
                    max_magnitude(),
                    subtract_dataset_utility=True,
                    n_field_samples=16,
                    n_paths_per_field=16,
                    lambda_=1 / 3,
                    targets_per_solve=chunk,
                ),
                matched_belief(problem),
            ),
            # a charge would bias a uniform draw, which spans [0, 1] whatever the field
            arm(
                "random_search",
                _method(
                    uniform(), subtract_dataset_utility=False, n_field_samples=0, n_paths_per_field=0, lambda_=0.0, targets_per_solve=chunk
                ),
                matched_belief(problem),
            ),
        ],
    )
