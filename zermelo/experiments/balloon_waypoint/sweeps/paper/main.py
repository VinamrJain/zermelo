"""Our rules against the baselines on each still frame, with the altitude budget never binding"""

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
from zermelo.experiments.balloon_waypoint.sweeps.paper.method import N_SEEDS, baseline, mc_ei, ours

BELIEF = static_error_belief()
ORACLE = dataclasses.replace(BELIEF, oracle=True)
"""The true wind in place of a belief"""

for date, start_hour in STILL_FRAMES.items():
    sweep(
        f"paper_main_{date}",
        problem=still(start_hour, STILL_HORIZON),
        horizon=STILL_HORIZON,
        claim_every=CLAIM_EVERY_STEPS,
        recorded=RECORDED,
        seeds=range(N_SEEDS),
        resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high"),
        arms=[
            arm("rand_act", None, BELIEF),  # no rule at all: acts uniformly and claims the prior
            arm("rand_target", baseline(uniform(), 0), BELIEF),
            arm("max_var", baseline(posterior_spread(), 0), BELIEF),
            arm("ucb", baseline(upper_confidence(2.0), 0), BELIEF),
            arm("ts", baseline(max_magnitude(), 1), BELIEF),
            arm("mc_ei", mc_ei(), BELIEF),
            arm("emi", ours("emi"), BELIEF),
            arm("evi", ours("evi"), BELIEF),
            arm("esi", ours("esi"), BELIEF),
            arm("oracle_emi", ours("emi"), ORACLE),
        ],
    )
