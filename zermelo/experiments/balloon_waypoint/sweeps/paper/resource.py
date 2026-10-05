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
from zermelo.experiments.balloon_waypoint.setup import arm, max_magnitude, posterior_spread, sweep, uniform, upper_confidence
from zermelo.experiments.balloon_waypoint.sweeps.paper.method import BUDGETED_STEP_RATE, COST_WEIGHT, N_SEEDS, baseline, mc_ei, ours

BELIEF = static_error_belief()

BUDGETS = (20, 40, 60, 80, 100)
"""Altitude changes an episode of 240 time_steps may spend"""

for budget in BUDGETS:
    for date, start_hour in STILL_FRAMES.items():
        sweep(
            f"paper_resource_b{budget:03d}_{date}",
            problem=still(start_hour, budget),
            horizon=STILL_HORIZON,
            claim_every=CLAIM_EVERY_STEPS,
            recorded=RECORDED,
            seeds=range(N_SEEDS),
            resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high"),
            arms=[
                arm("rand_act", None, BELIEF),
                arm("rand_target", baseline(uniform(), 0), BELIEF),
                arm("ucb", baseline(upper_confidence(2.0), 0), BELIEF),
                arm("mc_ei", mc_ei(), BELIEF),
                arm("emi_w0", ours("emi"), BELIEF),
                arm("emi", ours("emi", step_rate=BUDGETED_STEP_RATE, cost_weight=COST_WEIGHT), BELIEF),
                arm("evi_w0", ours("evi"), BELIEF),
                arm("evi", ours("evi", step_rate=BUDGETED_STEP_RATE, cost_weight=COST_WEIGHT), BELIEF),
            ],
        )
        sweep(
            f"paper_resource_extra_b{budget:03d}_{date}",
            problem=still(start_hour, budget),
            horizon=STILL_HORIZON,
            claim_every=CLAIM_EVERY_STEPS,
            recorded=RECORDED,
            seeds=range(N_SEEDS),
            resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high"),
            arms=[arm("max_var", baseline(posterior_spread(), 0), BELIEF), arm("ts", baseline(max_magnitude(), 1), BELIEF)],
        )
