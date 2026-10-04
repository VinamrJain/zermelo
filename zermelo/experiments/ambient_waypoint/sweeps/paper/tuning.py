"""Our two rules at every planning budget and step rate, on held-out seeds: where the paper's settings are picked"""

import dataclasses

from zermelo.experiments.ambient_waypoint.registry import (
    AMBIENT1_CONTROL1,
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    RECORDED,
    RESOURCES,
    matched_belief,
)
from zermelo.experiments.ambient_waypoint.setup import arm, sweep
from zermelo.experiments.ambient_waypoint.sweeps.paper.method import TUNING_SEEDS, ours, rate_name

BELIEF = matched_belief(AMBIENT1_CONTROL1)

sweep(
    "paper_tuning",
    problem=AMBIENT1_CONTROL1,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=TUNING_SEEDS,
    resources=dataclasses.replace(RESOURCES, mem_gb=16, timeout_min=240),
    arms=[
        arm(f"{name}_L{budget}_c{rate_name(rate)}", ours(name, step_rate=rate, replan_every=budget), BELIEF)
        for name in ("emi", "evi")
        for budget in (25, 50, 100)
        for rate in (0.0, 0.5, 1.0, 2.0)
    ],
)
