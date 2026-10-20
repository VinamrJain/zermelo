"""Our rules and the baselines on each still frame as the altitude changes an episode may spend are cut"""

import dataclasses

from zermelo.experiments.balloon_waypoint.registry import (
    CLAIM_EVERY_STEPS,
    RECORDED,
    RESOURCES,
    STILL_FRAMES,
    STILL_HORIZON,
    static_error_belief,
    still,
)
from zermelo.experiments.balloon_waypoint.setup import arm, max_magnitude, sweep, uniform, upper_confidence
from zermelo.experiments.balloon_waypoint.sweeps.method import BUDGETED_LAMBDA, baseline, mc_ei, ours

BELIEF = static_error_belief()

BUDGETS = (20, 40, 60, 80, 100, 120, STILL_HORIZON)
"""Altitude changes an episode of STILL_HORIZON time_steps may spend; STILL_HORIZON never binds"""

for budget in BUDGETS:
    for date, start_hour in STILL_FRAMES.items():
        sweep(
            f"paper_resource_b{budget:03d}_{date}",
            problem=still(start_hour, budget),
            horizon=STILL_HORIZON,
            claim_every=CLAIM_EVERY_STEPS,
            recorded=RECORDED,
            seeds=range(5),
            resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high"),
            arms=[
                arm("rand_act", None, BELIEF),
                arm("rand_target", baseline(uniform(), 0), BELIEF),
                arm("ucb", baseline(upper_confidence(2.0), 0), BELIEF),
                arm("ts", baseline(max_magnitude(), 1), BELIEF),
                arm("mc_ei", mc_ei(), BELIEF),
                arm("emi_w0", ours("emi"), BELIEF),
                arm("emi", ours("emi", lambda_=BUDGETED_LAMBDA), BELIEF),
            ],
        )
