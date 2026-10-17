"""Our rules on the held-out frame: every planning budget and cost weight, then every price of an altitude change under each budget"""

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

RATE_ARMS = {
    swept: [
        arm(f"{name}_L{budget}_c{rate_name(rate)}", ours(name, lambda_=rate, replan_length=budget), BELIEF)
        for name in names
        for budget in (25, 50, 100)
        for rate in (0.0, 0.5, 1.0, 2.0)
    ]
    for swept, names in (("paper_tuning", ("emi", "evi")), ("paper_tuning_esi", ("esi",)))
}
"""Every planning budget and cost weight, the altitude budget never binding"""

WEIGHT_ARMS = [
    arm(f"{name}_c{rate_name(rate)}", ours(name, lambda_=rate), BELIEF)
    for name in ("emi", "evi", "esi")
    for rate in (0.0, 0.5, 1.0, 2.0, 4.0)
]
"""Every price of an altitude change"""

for swept, resource_units, arms in (
    *((swept, STILL_HORIZON, arms) for swept, arms in RATE_ARMS.items()),
    *((f"paper_tuning_cost_weight_b{budget:03d}", budget, WEIGHT_ARMS) for budget in (20, 40, 80)),
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
