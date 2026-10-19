import dataclasses
from typing import Literal

from zermelo.experiments.ocean_waypoint.registry import (
    ARRIVAL_RADIUS_KM,
    CLAIM_EVERY_STEPS,
    GULF_STREAM_2019JUN,
    HORIZON_LENGTH,
    PLANNING_BUDGET,
    RECORDED,
    RESOURCES,
    STILL_FRAMES,
    VALUE_ITERATIONS,
    field_belief,
    oracle_belief,
)
from zermelo.experiments.ocean_waypoint.schema import MethodConfig
from zermelo.experiments.ocean_waypoint.setup import Implementation, arm, goals_reached, sweep, uniform, value_iteration

WORLD = dataclasses.replace(GULF_STREAM_2019JUN, objective="visit")
"""The same frame, scored on the moves until every one of the goals drawn from the target box has been visited"""

N_GOALS = 10
"""Goals drawn per episode"""

N_SEEDS = 3
"""Seeds per frame, each drawing its own start and goals"""

N_FIELDS = 16
"""`N` fields drawn from the posterior per decision"""

N_WALKS = 16
"""`M` walks the path-aware rule imagines per waypoint under each drawn field"""


def _method(utility: Implementation, *, candidates: Literal["box", "goals"], n_field_samples: int, n_paths_per_field: int) -> MethodConfig:
    """One rule on a value-iteration planner, travel not charged"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(
            value_iterations=VALUE_ITERATIONS, replan_length=PLANNING_BUDGET, radius=ARRIVAL_RADIUS_KM, targets_per_solve=None
        ),
        subtract_dataset_utility=True,
        n_field_samples=n_field_samples,
        n_paths_per_field=n_paths_per_field,
        lambda_=0.0,
        candidates=candidates,
        n_candidates_scored=None,
    )


# any cell of the box, scored by E over fields and walks of the goals a walk newly passes within `rho`
ETR = _method(goals_reached(ARRIVAL_RADIUS_KM), candidates="box", n_field_samples=N_FIELDS, n_paths_per_field=N_WALKS)

# the goals a cell of the box itself reaches, no walk imagined
ENDPOINT = _method(goals_reached(ARRIVAL_RADIUS_KM), candidates="box", n_field_samples=0, n_paths_per_field=0)
RAND_GOAL = _method(uniform(), candidates="goals", n_field_samples=0, n_paths_per_field=0)

GOAL_COUNTS = (5, 20, 30)
"""Goals drawn per episode in the goal-count study, the 10 of `visit_<date>` besides"""

SHORT_REPLAN = 50
"""`L = K` of the short-replan goal-count study"""

LONG_HORIZON = 720
"""time_steps in an episode of the short-replan goal-count study: 90 days at 3 hours each"""


def _short(method: MethodConfig) -> MethodConfig:
    """`method` replanning every `SHORT_REPLAN` moves on as many value iterations"""
    planner = value_iteration(value_iterations=SHORT_REPLAN, replan_length=SHORT_REPLAN, radius=ARRIVAL_RADIUS_KM, targets_per_solve=None)
    return dataclasses.replace(method, planner=planner)


RECORDED_VISIT = {**RECORDED, "objective_state/targets": 1, "objective_state/visited": 1, "objective_state/steps_to_completion": 1}

for date, frame_hour in STILL_FRAMES.items():
    sweep(
        f"visit_{date}",
        problem=dataclasses.replace(WORLD, frame_hour=frame_hour, n_goals=N_GOALS),
        horizon=HORIZON_LENGTH,
        claim_every=CLAIM_EVERY_STEPS,
        recorded=RECORDED_VISIT,
        seeds=list(range(N_SEEDS)),
        resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high"),  # 48 GB cards and up
        arms=[
            arm("etr", ETR, field_belief()),
            arm("oracle_etr", ETR, oracle_belief()),
            arm("etr_endpoint", ENDPOINT, field_belief()),
            arm("rand_goal", RAND_GOAL, field_belief()),
            arm("rand_box", _method(uniform(), candidates="box", n_field_samples=0, n_paths_per_field=0), field_belief()),
        ],
    )
    for n_goals in GOAL_COUNTS:
        sweep(
            f"visit_g{n_goals:02d}_{date}",
            problem=dataclasses.replace(WORLD, frame_hour=frame_hour, n_goals=n_goals),
            horizon=HORIZON_LENGTH,
            claim_every=CLAIM_EVERY_STEPS,
            recorded=RECORDED_VISIT,
            seeds=list(range(N_SEEDS)),
            resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high"),
            arms=[
                arm("etr", ETR, field_belief()),
                arm("etr_endpoint", ENDPOINT, field_belief()),
                arm("rand_goal", RAND_GOAL, field_belief()),
            ],
        )
    for n_goals in (5, 10, 20, 30):
        sweep(
            f"visit_L{SHORT_REPLAN}_g{n_goals:02d}_{date}",
            problem=dataclasses.replace(WORLD, frame_hour=frame_hour, n_goals=n_goals),
            horizon=LONG_HORIZON,
            claim_every=CLAIM_EVERY_STEPS,
            recorded=RECORDED_VISIT,
            seeds=list(range(N_SEEDS)),
            resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high"),
            arms=[
                arm(name, _short(method), field_belief())
                for name, method in (("etr", ETR), ("etr_endpoint", ENDPOINT), ("rand_goal", RAND_GOAL))
            ],
        )
