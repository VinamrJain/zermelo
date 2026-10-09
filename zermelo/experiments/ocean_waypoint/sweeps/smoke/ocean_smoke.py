import dataclasses

from zermelo.experiments.ocean_waypoint.registry import GULF_STREAM_2019JUN, RECORDED, error_belief
from zermelo.experiments.ocean_waypoint.schema import BeliefConfig, MethodConfig, Resources
from zermelo.experiments.ocean_waypoint.setup import (
    Implementation,
    arm,
    max_magnitude,
    sweep,
    total_variance,
    upper_confidence,
    value_iteration,
)

WORLD = dataclasses.replace(
    GULF_STREAM_2019JUN,
    grid_stride=6,  # a half-degree grid: 15 rows by 41 columns
    forecast_lag_hours=0.0,  # the drawn error, so the smoke run exercises that path too
    error_scale=0.3,
    error_lengthscale_km=200.0,  # wide, so a handful of readings say something about the whole box
)
"""The world on a coarse grid, the forecast being the truth under a drawn error"""


def _belief(oracle: bool) -> BeliefConfig:
    """The model an arm holds: the drawn error's own numbers, or the true ocean current itself"""
    (place,) = error_belief().kernel_factors
    return dataclasses.replace(
        error_belief(),
        oracle=oracle,
        kernel_factors=[dataclasses.replace(place, lengthscale=[WORLD.error_lengthscale_km] * 2)],
        amplitude=WORLD.error_scale,
        n_features=32,
        refit_steps=20,
    )


def _method(utility: Implementation, *, improvement: bool, n_fields: int, n_walks: int, cost_weight: float) -> MethodConfig:
    """One rule on a value-iteration planner, at the settings the arms differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_every=3, radius=60.0, target_chunk=None, cost_weight=cost_weight),
        improvement=improvement,
        n_fields=n_fields,
        n_walks=n_walks,
        combination="linear",
        step_rate=0.5,
        steps_from="predicted",
        n_candidates=16,
        opening_steps=3,  # three uniform moves, leaving seven the rule decides
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
