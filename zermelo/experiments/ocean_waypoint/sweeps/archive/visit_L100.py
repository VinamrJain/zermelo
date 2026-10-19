"""Visiting at the registry's planning budget over its horizon: every rule on 10 goals, then three rules on each other goal count"""

import dataclasses

from zermelo.experiments.ocean_waypoint.registry import (
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    RESOURCES,
    STILL_FRAMES,
    field_belief,
    oracle_belief,
)
from zermelo.experiments.ocean_waypoint.setup import arm, sweep, uniform
from zermelo.experiments.ocean_waypoint.sweeps.method import ENDPOINT, ETR, N_GOALS, N_SEEDS, RAND_GOAL, RECORDED_VISIT, WORLD, rule

GOAL_COUNTS = (5, 20, 30)
"""Goals drawn per episode in the goal-count sweeps, the 10 of `visit_<date>` besides"""

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
            arm("rand_box", rule(uniform(), candidates="box", n_field_samples=0, n_paths_per_field=0), field_belief()),
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
