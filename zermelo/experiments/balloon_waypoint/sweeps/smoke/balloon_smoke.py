import dataclasses

from zermelo.experiments.balloon_waypoint.registry import ATLANTIC_2017SEP, RECORDED, error_belief
from zermelo.experiments.balloon_waypoint.schema import BeliefConfig, BeliefKernelConfig, MethodConfig, Resources
from zermelo.experiments.balloon_waypoint.setup import (
    Implementation,
    arm,
    max_magnitude,
    point_speed,
    sweep,
    total_variance,
    upper_confidence,
    value_iteration,
)

WORLD = dataclasses.replace(
    ATLANTIC_2017SEP,
    grid_stride=4,  # a 4 degree grid: 45 rows by 90 columns
    hour_stride=2,  # a frame every 6 hours, at hours 3, 9, 15, ...
    forecast="gaussian",  # the drawn error, so the smoke run exercises that path too
    resource_units=3,
    error_scale=1.0,
    error_lengthscale_km=600.0,  # wide, so a handful of readings say something about the whole box
    target=point_speed(),
)
"""The world on a coarse grid at a ballast budget of two, the wind moving under the balloon"""


def _belief(oracle: bool) -> BeliefConfig:
    """The model an arm holds: the drawn error's own numbers, or the true wind itself"""
    place, _ = error_belief().kernel_factors
    return dataclasses.replace(
        error_belief(),
        oracle=oracle,
        kernel_factors=[
            dataclasses.replace(place, lengthscale=[WORLD.error_lengthscale_km] * 3 + [3.0]),
            BeliefKernelConfig(kernel="gpjax.kernels.Matern32", parts=["hours_elapsed"], lengthscale=[12.0]),
        ],
        amplitude=WORLD.error_scale,
        n_features=32,
        refit_steps=20,
    )


def _method(utility: Implementation, *, subtract_dataset_utility: bool, n_field_samples: int, n_paths_per_field: int) -> MethodConfig:
    """One rule on a value-iteration planner, at the settings the arms differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_length=3, radius=120.0, targets_per_solve=None),
        subtract_dataset_utility=subtract_dataset_utility,
        n_field_samples=n_field_samples,
        n_paths_per_field=n_paths_per_field,
        lambda_=1 / 3,
        n_candidates_scored=16,
    )


# One entry per device the runs can run on; `smoke_gpu` asks for an accelerator and a gpu-high node
for name, resources in (
    ("smoke", Resources(cpus=1, mem_gb=8, timeout_min=20, gres=None, partition="dean", constraint="avx2", array_parallelism=10)),
    (
        "smoke_gpu",
        Resources(cpus=1, mem_gb=8, timeout_min=20, gres="gpu:1", partition="dean", constraint="avx2&gpu-high", array_parallelism=10),
    ),
):
    sweep(
        name,
        problem=WORLD,
        horizon=10,
        claim_every=1,
        recorded={**RECORDED, "objective_state/claim": 1},
        seeds=[0],
        resources=resources,
        arms=[
            # the priced arm, so a smoke run walks both the charged plan and the unpriced one
            arm(
                "mean",
                _method(max_magnitude(), subtract_dataset_utility=True, n_field_samples=0, n_paths_per_field=0),
                _belief(oracle=False),
            ),
            arm(
                "oracle",
                _method(max_magnitude(), subtract_dataset_utility=True, n_field_samples=0, n_paths_per_field=0),
                _belief(oracle=True),
            ),
            arm(
                "ucb",
                _method(upper_confidence(c=2.0), subtract_dataset_utility=False, n_field_samples=0, n_paths_per_field=0),
                _belief(oracle=False),
            ),
            arm(
                "evi",
                _method(total_variance(), subtract_dataset_utility=True, n_field_samples=2, n_paths_per_field=2),
                _belief(oracle=False),
            ),
            arm("random", None, _belief(oracle=False)),  # no rule at all
        ],
    )
