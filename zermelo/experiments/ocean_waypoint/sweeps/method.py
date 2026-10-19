"""The rules every visiting sweep runs, and the world and counts they share"""

import dataclasses
from typing import Literal

from zermelo.experiments.ocean_waypoint.registry import ARRIVAL_RADIUS_KM, GULF_STREAM_2019JUN, PLANNING_BUDGET, RECORDED, VALUE_ITERATIONS
from zermelo.experiments.ocean_waypoint.schema import MethodConfig
from zermelo.experiments.ocean_waypoint.setup import Implementation, goals_reached, uniform, value_iteration

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


def rule(utility: Implementation, *, candidates: Literal["box", "goals"], n_field_samples: int, n_paths_per_field: int) -> MethodConfig:
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
ETR = rule(goals_reached(ARRIVAL_RADIUS_KM), candidates="box", n_field_samples=N_FIELDS, n_paths_per_field=N_WALKS)

# the goals a cell of the box itself reaches, no walk imagined
ENDPOINT = rule(goals_reached(ARRIVAL_RADIUS_KM), candidates="box", n_field_samples=0, n_paths_per_field=0)
RAND_GOAL = rule(uniform(), candidates="goals", n_field_samples=0, n_paths_per_field=0)

RECORDED_VISIT = {**RECORDED, "objective_state/targets": 1, "objective_state/visited": 1, "objective_state/steps_to_completion": 1}
"""The record fields every visiting sweep keeps"""
