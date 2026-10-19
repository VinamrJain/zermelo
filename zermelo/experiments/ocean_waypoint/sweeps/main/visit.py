"""Visiting with a short replan over a long horizon: three rules on each goal count, the main-text ocean figure"""

import dataclasses

from zermelo.experiments.ocean_waypoint.registry import ARRIVAL_RADIUS_KM, CLAIM_EVERY_STEPS, RESOURCES, STILL_FRAMES, field_belief
from zermelo.experiments.ocean_waypoint.schema import MethodConfig
from zermelo.experiments.ocean_waypoint.setup import arm, sweep, value_iteration
from zermelo.experiments.ocean_waypoint.sweeps.method import ENDPOINT, ETR, N_SEEDS, RAND_GOAL, RECORDED_VISIT, WORLD

SHORT_REPLAN = 50
"""`L = K` of the short-replan goal-count study"""

LONG_HORIZON = 720
"""time_steps in an episode of the short-replan goal-count study: 90 days at 3 hours each"""


def _short(method: MethodConfig) -> MethodConfig:
    """`method` replanning every `SHORT_REPLAN` moves on as many value iterations"""
    planner = value_iteration(value_iterations=SHORT_REPLAN, replan_length=SHORT_REPLAN, radius=ARRIVAL_RADIUS_KM, targets_per_solve=None)
    return dataclasses.replace(method, planner=planner)


for date, frame_hour in STILL_FRAMES.items():
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
