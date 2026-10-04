"""Our two rules as the step rate moves and as the planning budget moves, the other held at its tuned value"""

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
from zermelo.experiments.ambient_waypoint.sweeps.paper.method import ours, rate_name

BELIEF = matched_belief(AMBIENT1_CONTROL1)

for swept, arms in (
    (
        "paper_cost_rate",
        [
            arm(f"{name}_c{rate_name(rate)}", ours(name, step_rate=rate), BELIEF)
            for name in ("emi", "evi")
            for rate in (0.0, 0.25, 0.5, 1.0, 2.0)
        ],
    ),
    (
        "paper_planning_budget",
        [arm(f"{name}_L{budget}", ours(name, replan_every=budget), BELIEF) for name in ("emi", "evi") for budget in (10, 25, 50, 100, 200)],
    ),
):
    sweep(
        swept,
        problem=AMBIENT1_CONTROL1,
        horizon=HORIZON_LENGTH,
        claim_every=CLAIM_EVERY_STEPS,
        recorded=RECORDED,
        seeds=range(N_SEEDS),
        resources=dataclasses.replace(RESOURCES, mem_gb=16, timeout_min=480),
        arms=arms,
    )
