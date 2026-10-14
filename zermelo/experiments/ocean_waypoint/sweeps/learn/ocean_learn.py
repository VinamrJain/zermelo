import dataclasses

from zermelo.experiments.ocean_waypoint.registry import (
    ARRIVAL_RADIUS_KM,
    CLAIM_EVERY_STEPS,
    GULF_STREAM_2019JUN,
    HORIZON_LENGTH,
    N_SEEDS,
    PLANNING_BUDGET,
    RECORDED,
    RESOURCES,
    STILL_FRAMES,
    ocean_current_belief,
)
from zermelo.experiments.ocean_waypoint.schema import MethodConfig
from zermelo.experiments.ocean_waypoint.setup import (
    Implementation,
    arm,
    posterior_spread,
    region_information,
    sweep,
    total_variance,
    uniform,
    value_iteration,
)

WORLD = dataclasses.replace(GULF_STREAM_2019JUN, objective="learn")
"""The same frame, scored on the ocean current learned over the target box, which is also where a waypoint may be aimed"""

N_WALKS = 16
"""`M` walks the path-aware rules imagine per candidate, under the posterior mean"""


def _method(utility: Implementation, *, n_walks: int) -> MethodConfig:
    """One rule on a value-iteration planner, travel not priced"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(
            backup_iterations=HORIZON_LENGTH, replan_every=PLANNING_BUDGET, radius=ARRIVAL_RADIUS_KM, target_chunk=None, cost_weight=0.0
        ),
        improvement=True,
        n_fields=0,
        n_walks=n_walks,
        penalty_rate=0.0,
        n_candidates=None,
        opening_steps=0,
    )


for date, frame_hour in STILL_FRAMES.items():
    sweep(
        f"learn_{date}",
        problem=dataclasses.replace(WORLD, frame_hour=frame_hour),
        horizon=HORIZON_LENGTH,
        claim_every=CLAIM_EVERY_STEPS,
        recorded={**RECORDED, "objective_state/information": 1},
        seeds=list(range(N_SEEDS)),
        resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high"),  # 48 GB cards and up
        arms=[
            # E over walks tau of [ sum over c of Var(c | D) - Var(c | D + obs(tau)) ]
            arm("evi", _method(total_variance(), n_walks=N_WALKS), ocean_current_belief()),
            # E over walks tau of I( f(C) ; obs(tau) | D )
            arm("eii", _method(region_information(WORLD.noise_ms), n_walks=N_WALKS), ocean_current_belief()),
            # the posterior spread at the destination alone: the point information gain, since every destination is a candidate
            arm("max_var", _method(posterior_spread(), n_walks=0), ocean_current_belief()),
            arm("rand_wp", _method(uniform(), n_walks=0), ocean_current_belief()),
            arm("rand_act", None, ocean_current_belief()),  # no rule at all
        ],
    )
