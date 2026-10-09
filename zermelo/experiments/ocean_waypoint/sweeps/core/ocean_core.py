import dataclasses

from zermelo.experiments.ocean_waypoint.registry import (
    ARRIVAL_RADIUS_KM,
    CLAIM_EVERY_STEPS,
    GULF_STREAM_2019JUN,
    HORIZON_LENGTH,
    PLANNING_BUDGET,
    RECORDED,
    RESOURCES,
    error_belief,
    oracle_belief,
)
from zermelo.experiments.ocean_waypoint.schema import MethodConfig
from zermelo.experiments.ocean_waypoint.setup import Implementation, arm, learned_trip, remaining_trip, sweep, uniform, value_iteration

N_CANDIDATES = 16
"""Candidate waypoints scored per decision, drawn uniformly over the box"""

N_DRAWS = 4
"""`S` fields and `M` walks per field the `learned` arm imagines"""


def _method(utility: Implementation, *, improvement: bool, n_fields: int, n_walks: int) -> MethodConfig:
    """One rule on a value-iteration planner, travel priced in steps, hitting costs truncated at the horizon"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(
            replan_every=PLANNING_BUDGET, backups=HORIZON_LENGTH, radius=ARRIVAL_RADIUS_KM, target_chunk=None, cost_weight=0.0
        ),
        improvement=improvement,
        n_fields=n_fields,
        n_walks=n_walks,
        combination="steps",
        step_rate=None,
        steps_from="predicted",
        n_candidates=N_CANDIDATES,
        opening_steps=0,
    )


def _trip(learned: bool) -> Implementation:
    """A trip utility solving with the planner's own numbers"""
    return (learned_trip if learned else remaining_trip)(
        backups=HORIZON_LENGTH, radius=ARRIVAL_RADIUS_KM, target_chunk=None, cost_weight=0.0
    )


sweep(
    "core",
    problem=GULF_STREAM_2019JUN,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=[0],  # one instance while the pipeline is proved; range(N_SEEDS) for the study
    resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high"),  # 48 GB cards and up
    arms=[
        arm("ce", _method(_trip(False), improvement=False, n_fields=0, n_walks=0), error_belief()),
        arm("draws", _method(_trip(False), improvement=False, n_fields=N_DRAWS, n_walks=0), error_belief()),
        arm("learned", _method(_trip(True), improvement=True, n_fields=N_DRAWS, n_walks=N_DRAWS), error_belief()),
        arm("rand_wp", _method(uniform(), improvement=False, n_fields=0, n_walks=0), error_belief()),
        arm("rand_act", None, error_belief()),  # no rule at all
        arm("oracle", _method(_trip(False), improvement=False, n_fields=0, n_walks=0), oracle_belief()),
    ],
)
