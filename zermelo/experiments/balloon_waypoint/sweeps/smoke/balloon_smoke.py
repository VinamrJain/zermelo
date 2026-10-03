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


def _method(utility: Implementation, *, improvement: bool, n_fields: int, n_walks: int, cost_weight: float) -> MethodConfig:
    """One rule on a value-iteration planner, at the settings the arms differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_every=3, radius=120.0, target_chunk=None, cost_weight=cost_weight),
        improvement=improvement,
        n_fields=n_fields,
        n_walks=n_walks,
        combination="linear",
        step_rate=0.5,
        steps_from="predicted",
        n_candidates=16,
        opening_steps=3,  # three uniform moves, leaving three the rule decides
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
            arm("mean", _method(max_magnitude(), improvement=True, n_fields=0, n_walks=0, cost_weight=0.25), _belief(oracle=False)),
            arm("oracle", _method(max_magnitude(), improvement=True, n_fields=0, n_walks=0, cost_weight=0.0), _belief(oracle=True)),
            arm("ucb", _method(upper_confidence(c=2.0), improvement=False, n_fields=0, n_walks=0, cost_weight=0.0), _belief(oracle=False)),
            arm("evi", _method(total_variance(), improvement=True, n_fields=2, n_walks=2, cost_weight=0.25), _belief(oracle=False)),
            arm("random", None, _belief(oracle=False)),  # no rule at all
        ],
    )
