"""Our rules on the held-out frame: every planning budget `L`, then every price of an altitude change under each altitude budget"""

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
from zermelo.experiments.balloon_waypoint.sweeps.method import N_SEEDS, UTILITIES, ours, rate_name

BELIEF = static_error_belief()

BUDGET_ARMS = [arm(f"{name}_L{budget}", ours(name, replan_length=budget), BELIEF) for name in UTILITIES for budget in (25, 50, 100, 200)]
"""Every planning budget, the altitude budget never binding"""

WEIGHT_ARMS = [
    arm(f"{name}_c{rate_name(rate)}", ours(name, lambda_=rate), BELIEF) for name in ("emi", "evi") for rate in (0.0, 0.25, 0.5, 0.75)
]
"""Every weight `lambda` on the altitude changes, 0 the uncharged reference"""

for swept, resource_units, arms, seeds in (
    ("paper_tuning", STILL_HORIZON, BUDGET_ARMS, 5),
    *((f"paper_tuning_cost_weight_b{budget:03d}", budget, WEIGHT_ARMS, N_SEEDS) for budget in (20, 40, 80)),
):
    sweep(
        swept,
        problem=still(TUNING_FRAME, resource_units),
        horizon=STILL_HORIZON,
        claim_every=CLAIM_EVERY_STEPS,
        recorded=RECORDED,
        seeds=range(seeds),
        resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high", timeout_min=360),
        arms=arms,
    )
