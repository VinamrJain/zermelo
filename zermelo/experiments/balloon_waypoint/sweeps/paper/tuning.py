"""Our two rules on the held-out frame: every planning budget and step rate, then every price of an altitude change under a budget"""

import dataclasses

from zermelo.experiments.balloon_waypoint.registry import (
    CLAIM_EVERY_STEPS,
    RECORDED,
    RESOURCES,
    STILL_HORIZON,
    TUNING_FRAME,
    static_error_belief,
    still,
)
from zermelo.experiments.balloon_waypoint.setup import arm, sweep
from zermelo.experiments.balloon_waypoint.sweeps.paper.method import N_SEEDS, ours, rate_name

BELIEF = static_error_belief()

for swept, resource_units, arms in (
    (
        "paper_tuning",
        STILL_HORIZON,  # a change every time_step: the budget never binds
        [
            arm(f"{name}_L{budget}_c{rate_name(rate)}", ours(name, step_rate=rate, replan_every=budget), BELIEF)
            for name in ("emi", "evi")
            for budget in (25, 50, 100)
            for rate in (0.0, 0.5, 1.0, 2.0)
        ],
    ),
    (
        "paper_tuning_cost_weight",
        100,
        [
            arm(f"{name}_w{rate_name(weight)}", ours(name, cost_weight=weight), BELIEF)
            for name in ("emi", "evi")
            for weight in (0.0, 0.5, 1.0, 2.0, 4.0)
        ],
    ),
):
    sweep(
        swept,
        problem=still(TUNING_FRAME, resource_units),
        horizon=STILL_HORIZON,
        claim_every=CLAIM_EVERY_STEPS,
        recorded=RECORDED,
        seeds=range(N_SEEDS),
        resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high"),
        arms=arms,
    )
