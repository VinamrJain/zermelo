import dataclasses
from typing import Literal

from zermelo.experiments.ocean_waypoint.registry import (
    ARRIVAL_RADIUS_KM,
    CLAIM_EVERY_STEPS,
    GULF_STREAM_2019JUN,
    HORIZON_LENGTH,
    N_SEEDS,
    ONLY_REACHABLE,
    PLANNING_BUDGET,
    RECORDED,
    RESOURCES,
    SCORED_COMPONENTS,
    STILL_FRAMES,
    VALUE_ITERATIONS,
    field_belief,
)
from zermelo.experiments.ocean_waypoint.schema import MethodConfig
from zermelo.experiments.ocean_waypoint.setup import Implementation, arm, posterior_spread, sweep, total_variance, uniform, value_iteration

LEARNED: Literal["ocean_current", "wind", "both"] = "both"
"""Which part of the field the episode is scored on, and the rules count: the ocean current, the wind, or both"""

WORLD = dataclasses.replace(GULF_STREAM_2019JUN, objective="learn", learned=LEARNED)
"""The same frame, scored on the ocean current learned over the target box, which is also where a target may be aimed"""

N_WALKS = 16
"""`M` walks the path-aware rules imagine per candidate, under the posterior mean"""

PENALTY_RATE = 1.0
"""`lambda`: a full-budget trip costs the path-aware rules this many spreads of the candidate values"""


def _method(utility: Implementation, *, n_paths_per_field: int, penalty_rate: float) -> MethodConfig:
    """One rule on a value-iteration planner, a full-budget trip charged `penalty_rate` of the spread of the candidate values"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(
            value_iterations=VALUE_ITERATIONS,
            replan_length=PLANNING_BUDGET,
            only_reachable=ONLY_REACHABLE,
            radius=ARRIVAL_RADIUS_KM,
            targets_per_solve=None,
            cost_weight=0.0,
        ),
        subtract_dataset_utility=True,
        n_field_samples=0,
        n_paths_per_field=n_paths_per_field,
        penalty_rate=penalty_rate,
        n_candidates_scored=None,
        random_opening_steps=0,
    )


for date, frame_hour in STILL_FRAMES.items():
    sweep(
        f"learn_{date}",
        problem=dataclasses.replace(WORLD, frame_hour=frame_hour),
        horizon=HORIZON_LENGTH,
        claim_every=CLAIM_EVERY_STEPS,
        recorded={**RECORDED, "objective_state/information": 1},
        seeds=list(range(N_SEEDS)),
        resources=RESOURCES,
        arms=[
            # E over walks tau of [ sum over c of Var(c | D) - Var(c | D + obs(tau)) ]
            arm(
                "evi",
                _method(total_variance(SCORED_COMPONENTS[LEARNED]), n_paths_per_field=N_WALKS, penalty_rate=PENALTY_RATE),
                field_belief(),
            ),
            # the posterior spread at the destination alone: the point information gain, since every destination is a candidate
            arm("max_var", _method(posterior_spread(SCORED_COMPONENTS[LEARNED]), n_paths_per_field=0, penalty_rate=0.0), field_belief()),
            arm("rand_wp", _method(uniform(), n_paths_per_field=0, penalty_rate=0.0), field_belief()),
            arm("rand_act", None, field_belief()),  # no rule at all
        ],
    )
