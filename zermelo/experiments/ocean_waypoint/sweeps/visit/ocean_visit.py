import dataclasses

from zermelo.experiments.ocean_waypoint.registry import (
    ARRIVAL_RADIUS_KM,
    CLAIM_EVERY_STEPS,
    GULF_STREAM_2019JUN,
    HORIZON_LENGTH,
    N_SEEDS,
    RECORDED,
    RESOURCES,
    ocean_current_belief,
    oracle_belief,
)
from zermelo.experiments.ocean_waypoint.schema import MethodConfig
from zermelo.experiments.ocean_waypoint.setup import Implementation, arm, sweep, targets_reached, uniform, value_iteration

WORLD = dataclasses.replace(GULF_STREAM_2019JUN, objective="visit")
"""The same frame, scored on the moves until every one of the targets drawn from the target box has been visited"""

PLANNING_BUDGET = 100
"""`L` moves a trip is planned for, and how often the target is reconsidered"""

N_FIELDS = 16
"""`N` fields drawn from the posterior per decision"""

N_WALKS = 16
"""`M` walks the path-aware rule imagines per target under each drawn field"""


def _method(utility: Implementation, *, n_fields: int, n_walks: int, penalty_rate: float) -> MethodConfig:
    """One rule on a value-iteration planner, a full-budget trip charged `penalty_rate` of the spread of the candidate values"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(
            backup_iterations=HORIZON_LENGTH, replan_every=PLANNING_BUDGET, radius=ARRIVAL_RADIUS_KM, target_chunk=None, cost_weight=0.0
        ),
        improvement=True,
        n_fields=n_fields,
        n_walks=n_walks,
        penalty_rate=penalty_rate,
        n_candidates=None,
        opening_steps=0,
    )


def _expected_targets_reached(penalty_rate: float) -> MethodConfig:
    """E over fields and walks of the targets a walk newly passes within `rho` of, less `penalty_rate` of the spread per full-budget trip"""
    return _method(targets_reached(ARRIVAL_RADIUS_KM), n_fields=N_FIELDS, n_walks=N_WALKS, penalty_rate=penalty_rate)


def _nearest_unvisited() -> MethodConfig:
    """1 at an unvisited target and 0 at a visited one, less half the spread per full-budget trip: the nearest unvisited target under the mean"""
    return _method(targets_reached(ARRIVAL_RADIUS_KM), n_fields=0, n_walks=0, penalty_rate=0.5)


sweep(
    "visit",
    problem=WORLD,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded={**RECORDED, "objective_state/targets": 1, "objective_state/visited": 1, "objective_state/steps_to_completion": 1},
    seeds=list(range(N_SEEDS)),
    resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high"),  # 48 GB cards and up
    arms=[
        arm("etr_l0", _expected_targets_reached(0.0), ocean_current_belief()),
        arm("etr_l05", _expected_targets_reached(0.5), ocean_current_belief()),
        arm("etr_l1", _expected_targets_reached(1.0), ocean_current_belief()),
        arm("nearest", _nearest_unvisited(), ocean_current_belief()),
        arm("oracle_etr", _expected_targets_reached(0.5), oracle_belief()),
        arm("rand_wp", _method(uniform(), n_fields=0, n_walks=0, penalty_rate=0.0), ocean_current_belief()),
    ],
)
