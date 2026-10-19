"""Our rules at every planning budget `L`, on held-out seeds"""

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
from zermelo.experiments.ambient_waypoint.sweeps.method import TUNING_SEEDS, UTILITIES, ours

sweep(
    "paper_tuning",
    problem=AMBIENT1_CONTROL1,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=TUNING_SEEDS,
    resources=dataclasses.replace(RESOURCES, mem_gb=16, timeout_min=240),
    arms=[
        arm(f"{name}_L{budget}", ours(name, replan_length=budget), matched_belief(AMBIENT1_CONTROL1))
        for name in UTILITIES
        for budget in (10, 25, 50, 100)
    ],
)
