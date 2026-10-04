"""Two of our rules as the random time_steps an episode opens with move"""

import dataclasses

from zermelo.experiments.ambient_waypoint.registry import (
    AMBIENT1_CONTROL1,
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    N_SEEDS,
    RECORDED,
    RESOURCES,
    matched_belief,
)
from zermelo.experiments.ambient_waypoint.setup import arm, sweep
from zermelo.experiments.ambient_waypoint.sweeps.paper.method import ours

sweep(
    "paper_opening_steps",
    problem=AMBIENT1_CONTROL1,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=dataclasses.replace(RESOURCES, mem_gb=16, timeout_min=240),
    arms=[
        arm(f"{name}_o{steps}", dataclasses.replace(ours(name), opening_steps=steps), matched_belief(AMBIENT1_CONTROL1))
        for name in ("emi", "evi")
        for steps in (0, 25, 50, 100)
    ],
)
