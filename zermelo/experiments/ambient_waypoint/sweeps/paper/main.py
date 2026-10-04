"""Our rules against the baselines on the grid, the parts our rule is made of, and the same planner on the true field"""

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
from zermelo.experiments.ambient_waypoint.setup import (
    arm,
    expected_improvement,
    max_magnitude,
    posterior_spread,
    sweep,
    uniform,
    upper_confidence,
)
from zermelo.experiments.ambient_waypoint.sweeps.paper.method import TUNED_STEP_RATE, baseline, mc_ei, ours

BELIEF = matched_belief(AMBIENT1_CONTROL1)
ORACLE = dataclasses.replace(BELIEF, oracle=True)
"""The true field in place of a belief"""

sweep(
    "paper_main",
    problem=AMBIENT1_CONTROL1,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=dataclasses.replace(RESOURCES, mem_gb=16, timeout_min=240),
    arms=[
        arm("rand_act", None, BELIEF),  # no rule at all: acts uniformly and claims the prior
        arm("rand_target", baseline(uniform(), 0), BELIEF),
        arm("max_var", baseline(posterior_spread(), 0), BELIEF),
        arm("ucb", baseline(upper_confidence(2.0), 0), BELIEF),
        arm("ei", baseline(expected_improvement(), 0), BELIEF),
        arm("ts", baseline(max_magnitude(), 1), BELIEF),
        arm("mc_ei", mc_ei(0.0), BELIEF),
        arm("mc_ei_charged", mc_ei(TUNED_STEP_RATE["emi"]), BELIEF),
        arm("emi_c0", ours("emi", step_rate=0.0), BELIEF),
        arm("emi", ours("emi"), BELIEF),
        arm("evi", ours("evi"), BELIEF),
        arm("esi", ours("esi"), BELIEF),
    ],
)

sweep(
    "paper_oracle",
    problem=AMBIENT1_CONTROL1,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=dataclasses.replace(RESOURCES, mem_gb=16, timeout_min=240),
    arms=[arm("rand_target", baseline(uniform(), 0), ORACLE), arm("mc_ei", mc_ei(0.0), ORACLE), arm("oracle_emi", ours("emi"), ORACLE)],
)
