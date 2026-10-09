import dataclasses

from zermelo.experiments.ocean_waypoint.registry import GULF_STREAM_2019JUN, RECORDED, error_belief, oracle_belief
from zermelo.experiments.ocean_waypoint.schema import BeliefConfig, MethodConfig, Resources
from zermelo.experiments.ocean_waypoint.setup import (
    Implementation,
    arm,
    information_gain,
    remaining_trip,
    replanned_trip,
    sweep,
    uniform,
    value_iteration,
)

WORLD = dataclasses.replace(
    GULF_STREAM_2019JUN,
    grid_stride=6,  # a half-degree grid: 15 rows by 41 columns
    forecast_lag_hours=0.0,  # the drawn error, so the smoke run exercises that path too
    error_scale=0.3,
    error_lengthscale_km=200.0,  # wide, so a handful of readings say something about the whole box
    arrival_km=60.0,
)
"""The world on a coarse grid, the forecast being the truth under a drawn error"""

BACKUPS = 30
"""`K`: rounds of backup, and the steps a hitting cost is truncated at"""


def _belief(oracle: bool) -> BeliefConfig:
    """The model an arm holds: the drawn error's own numbers, or the true ocean current itself"""
    (place,) = error_belief().kernel_factors
    return dataclasses.replace(
        oracle_belief() if oracle else error_belief(),
        kernel_factors=[dataclasses.replace(place, lengthscale=[WORLD.error_lengthscale_km] * 2)],
        amplitude=WORLD.error_scale,
        n_features=32,
        refit_steps=20,
    )


def _method(
    utility: Implementation, *, n_fields: int, n_walks: int, n_candidates: int | None, information: Implementation | None
) -> MethodConfig:
    """One rule on a value-iteration planner, travel priced in steps, at the settings the arms differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_every=3, backups=BACKUPS, radius=60.0, target_chunk=None, cost_weight=0.0),
        improvement=True,
        n_fields=n_fields,
        n_walks=n_walks,
        combination="steps",
        step_rate=None,
        steps_from="predicted",
        n_candidates=n_candidates,
        information=information,
        information_rate=None if information is None else 1.0,
        opening_steps=2,  # two uniform moves, leaving eight the rule decides
    )


REMAINING = remaining_trip(backups=BACKUPS, radius=60.0, target_chunk=None, cost_weight=0.0)
"""-H_mu(end of walk, target)"""


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
            arm("direct", _method(REMAINING, n_fields=0, n_walks=0, n_candidates=0, information=None), _belief(oracle=False)),
            arm(
                "info",
                _method(REMAINING, n_fields=0, n_walks=2, n_candidates=8, information=information_gain(WORLD.noise_ms)),
                _belief(oracle=False),
            ),
            arm(
                "eti",
                _method(
                    replanned_trip(backups=BACKUPS, radius=60.0, target_chunk=None, cost_weight=0.0, chunk=4),
                    n_fields=2,
                    n_walks=2,
                    n_candidates=3,
                    information=None,
                ),
                _belief(oracle=False),
            ),
            arm("rand_wp", _method(uniform(), n_fields=0, n_walks=0, n_candidates=8, information=None), _belief(oracle=False)),
            arm("rand_act", None, _belief(oracle=False)),  # no rule at all
            arm("oracle", _method(REMAINING, n_fields=0, n_walks=0, n_candidates=0, information=None), _belief(oracle=True)),
        ],
    )
