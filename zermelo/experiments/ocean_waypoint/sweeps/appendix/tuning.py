"""ETR at every planning budget `L`, `L` value iterations, on a frame no visiting study reports"""

import dataclasses

from zermelo.experiments.ocean_waypoint.registry import (
    ARRIVAL_RADIUS_KM,
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    RECORDED,
    RESOURCES,
    field_belief,
)
from zermelo.experiments.ocean_waypoint.setup import arm, sweep, value_iteration
from zermelo.experiments.ocean_waypoint.sweeps.method import ETR, N_GOALS, N_SEEDS, WORLD

TUNING_FRAME_HOUR = 216.0
"""2019-06-10 00Z, between the jun04 and jun16 frames"""

sweep(
    "tuning_jun10",
    problem=dataclasses.replace(WORLD, frame_hour=TUNING_FRAME_HOUR, n_goals=N_GOALS),
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded={**RECORDED, "objective_state/targets": 1, "objective_state/visited": 1, "objective_state/steps_to_completion": 1},
    seeds=list(range(N_SEEDS)),
    resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high", timeout_min=360),
    arms=[
        arm(
            f"etr_L{budget}",
            dataclasses.replace(
                ETR,
                planner=value_iteration(value_iterations=budget, replan_length=budget, radius=ARRIVAL_RADIUS_KM, targets_per_solve=None),
            ),
            field_belief(),
        )
        for budget in (25, 50, 100, 200)
    ],
)
