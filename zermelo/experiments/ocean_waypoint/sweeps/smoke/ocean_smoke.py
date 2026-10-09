import dataclasses

from zermelo.experiments.ocean_waypoint.registry import GULF_STREAM_2019JUN, RECORDED, error_belief, oracle_belief
from zermelo.experiments.ocean_waypoint.schema import BeliefConfig, MethodConfig, Resources
from zermelo.experiments.ocean_waypoint.setup import Implementation, arm, learned_trip, remaining_trip, sweep, uniform, value_iteration

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


def _method(utility: Implementation, *, improvement: bool, n_fields: int, n_walks: int) -> MethodConfig:
    """One rule on a value-iteration planner, travel priced in steps, at the settings the arms differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_every=3, backups=BACKUPS, radius=60.0, target_chunk=None, cost_weight=0.0),
        improvement=improvement,
        n_fields=n_fields,
        n_walks=n_walks,
        combination="steps",
        step_rate=None,
        steps_from="predicted",
        n_candidates=8,
        opening_steps=2,  # two uniform moves, leaving eight the rule decides
    )


def _trip(learned: bool) -> Implementation:
    """A trip utility solving with the planner's own numbers"""
    return (learned_trip if learned else remaining_trip)(backups=BACKUPS, radius=60.0, target_chunk=None, cost_weight=0.0)


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
            arm("ce", _method(_trip(False), improvement=False, n_fields=0, n_walks=0), _belief(oracle=False)),
            arm("draws", _method(_trip(False), improvement=False, n_fields=2, n_walks=0), _belief(oracle=False)),
            arm("learned", _method(_trip(True), improvement=True, n_fields=2, n_walks=2), _belief(oracle=False)),
            arm("rand_wp", _method(uniform(), improvement=False, n_fields=0, n_walks=0), _belief(oracle=False)),
            arm("rand_act", None, _belief(oracle=False)),  # no rule at all
            arm("oracle", _method(_trip(False), improvement=False, n_fields=0, n_walks=0), _belief(oracle=True)),
        ],
    )
