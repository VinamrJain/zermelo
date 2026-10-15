import dataclasses

from zermelo.experiments.ocean_waypoint.registry import GULF_STREAM_2019JUN, RECORDED, SCORED_COMPONENTS, error_belief, oracle_belief
from zermelo.experiments.ocean_waypoint.schema import BeliefConfig, MethodConfig, Resources
from zermelo.experiments.ocean_waypoint.setup import (
    Implementation,
    arm,
    goals_reached,
    max_magnitude,
    posterior_spread,
    region_information,
    sweep,
    total_variance,
    uniform,
    value_iteration,
)
from zermelo.problems.ocean import OCEAN_CURRENT

WORLD = dataclasses.replace(
    GULF_STREAM_2019JUN,
    grid_stride=6,  # a half-degree grid: 15 rows by 41 columns
    forecast_lag_hours=0.0,  # the drawn error, so the smoke run exercises that path too
    error_scale=0.3,
    error_lengthscale_km=200.0,  # wide, so a handful of readings say something about the whole box
    arrival_km=60.0,
)
"""The world on a coarse grid, the forecast being the truth under a drawn error"""


def _belief(oracle: bool) -> BeliefConfig:
    """The model an arm holds: the drawn error's own numbers in every group, or the true field itself"""
    belief = oracle_belief() if oracle else error_belief()
    return dataclasses.replace(
        belief,
        groups=[
            dataclasses.replace(
                group,
                kernel_factors=[dataclasses.replace(place, lengthscale=[WORLD.error_lengthscale_km] * 2) for place in group.kernel_factors],
                amplitude=WORLD.error_scale,
            )
            for group in belief.groups
        ],
        n_features=32,
        refit_steps=20,
    )


def _method(utility: Implementation, *, n_field_samples: int, n_paths_per_field: int, n_candidates_scored: int | None) -> MethodConfig:
    """One rule on a value-iteration planner, only the candidates within reach scored, at the settings the arms differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(
            value_iterations=10, replan_length=3, only_reachable=True, radius=60.0, targets_per_solve=None, cost_weight=0.0
        ),
        subtract_dataset_utility=True,
        n_field_samples=n_field_samples,
        n_paths_per_field=n_paths_per_field,
        penalty_rate=0.0,
        n_candidates_scored=n_candidates_scored,
        random_opening_steps=2,  # two uniform moves, leaving eight the rule decides
    )


# The visiting objective, two targets, once per device the runs can run on; `smoke_gpu` asks for an accelerator and a gpu-high node
for name, resources in (
    ("smoke", Resources(cpus=1, mem_gb=8, timeout_min=20, gres=None, partition="dean", constraint="avx2", array_parallelism=10)),
    (
        "smoke_gpu",
        Resources(cpus=1, mem_gb=8, timeout_min=20, gres="gpu:1", partition="dean", constraint="avx2&gpu-high", array_parallelism=10),
    ),
):
    sweep(
        name,
        problem=dataclasses.replace(WORLD, objective="visit", n_targets=2),
        horizon=10,
        claim_every=1,
        recorded={
            **RECORDED,
            "objective_state/claim": 1,
            "objective_state/targets": 1,
            "objective_state/visited": 1,
            "objective_state/steps_to_completion": 1,
        },
        seeds=[0],
        resources=resources,
        arms=[
            arm(
                "etr", _method(goals_reached(60.0), n_field_samples=2, n_paths_per_field=2, n_candidates_scored=None), _belief(oracle=False)
            ),
            arm(
                "nearest",
                _method(goals_reached(60.0), n_field_samples=0, n_paths_per_field=0, n_candidates_scored=None),
                _belief(oracle=False),
            ),
            arm("rand_act", None, _belief(oracle=False)),  # no rule at all
        ],
    )

# The learning objective on the same coarse grid, the walks the two path-aware rules imagine kept short
sweep(
    "smoke_learn",
    problem=dataclasses.replace(WORLD, objective="learn"),
    horizon=10,
    claim_every=1,
    recorded={**RECORDED, "objective_state/claim": 1, "objective_state/information": 1},
    seeds=[0],
    resources=Resources(cpus=1, mem_gb=8, timeout_min=20, gres=None, partition="dean", constraint="avx2", array_parallelism=10),
    arms=[
        arm(
            "evi",
            _method(total_variance(SCORED_COMPONENTS[WORLD.learned]), n_field_samples=0, n_paths_per_field=2, n_candidates_scored=8),
            _belief(oracle=False),
        ),
        arm(
            "eii",
            _method(region_information(SCORED_COMPONENTS[WORLD.learned]), n_field_samples=0, n_paths_per_field=2, n_candidates_scored=8),
            _belief(oracle=False),
        ),
        arm(
            "max_var",
            _method(posterior_spread(SCORED_COMPONENTS[WORLD.learned]), n_field_samples=0, n_paths_per_field=0, n_candidates_scored=8),
            _belief(oracle=False),
        ),
        arm("rand_act", None, _belief(oracle=False)),
    ],
)

# The search objective on the same coarse grid, the fields drawn and the walks kept few
sweep(
    "smoke_search",
    problem=dataclasses.replace(WORLD, objective="search"),
    horizon=10,
    claim_every=1,
    recorded={
        **RECORDED,
        "objective_state/claim": 1,
        "objective_state/best_observed_magnitude": 1,
        "objective_state/best_possible_magnitude": 1,
    },
    seeds=[0],
    resources=Resources(cpus=1, mem_gb=8, timeout_min=20, gres=None, partition="dean", constraint="avx2", array_parallelism=10),
    arms=[
        arm(
            "emi",
            _method(max_magnitude(OCEAN_CURRENT), n_field_samples=2, n_paths_per_field=2, n_candidates_scored=8),
            _belief(oracle=False),
        ),
        arm(
            "mc_ei",
            _method(max_magnitude(OCEAN_CURRENT), n_field_samples=2, n_paths_per_field=0, n_candidates_scored=8),
            _belief(oracle=False),
        ),
        arm("rand_act", None, _belief(oracle=False)),
    ],
)
