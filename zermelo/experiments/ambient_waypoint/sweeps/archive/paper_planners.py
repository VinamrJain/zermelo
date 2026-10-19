"""The 2D grid's rules on each planner: value iteration against a greedy one-step planner"""

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
from zermelo.experiments.ambient_waypoint.setup import arm, greedy, sweep, uniform, value_iteration
from zermelo.experiments.ambient_waypoint.sweeps.method import PLANNING_BUDGET, baseline, mc_ei, ours

MATCHED = matched_belief(AMBIENT1_CONTROL1)
PLANNERS = {"value_iteration": value_iteration, "greedy": greedy}

sweep(
    "paper_planners",
    problem=AMBIENT1_CONTROL1,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=dataclasses.replace(RESOURCES, mem_gb=16, timeout_min=480),
    arms=[
        arm(
            f"{name}_{planner}",
            dataclasses.replace(method, planner=build(replan_length=PLANNING_BUDGET, radius=0.5, targets_per_solve=None)),
            MATCHED,
        )
        for planner, build in PLANNERS.items()
        for name, method in (("emi", ours("emi")), ("evi", ours("evi")), ("mc_ei", mc_ei()), ("rand_target", baseline(uniform(), 0)))
    ],
)
