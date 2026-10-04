import dataclasses

from zermelo.experiments.balloon_waypoint.registry import (
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    IRMA_2017SEP06,
    N_SEEDS,
    RECORDED,
    RESOURCES,
    static_error_belief,
)
from zermelo.experiments.balloon_waypoint.setup import arm, max_magnitude, posterior_spread, sum_magnitude, sweep, uniform, upper_confidence
from zermelo.experiments.balloon_waypoint.sweeps.core.core_cost0 import _baseline, _ours

BELIEF = static_error_belief()

sweep(
    "static",
    problem=IRMA_2017SEP06,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high"),
    arms=[
        arm("rand_act", None, BELIEF),
        arm("rand_wp", _baseline(uniform(), 0), BELIEF),
        arm("max_var", _baseline(posterior_spread(), 0), BELIEF),
        arm("ucb", _baseline(upper_confidence(2.0), 0), BELIEF),
        arm("ts", _baseline(max_magnitude(), 1), BELIEF),
        arm("emi_L50", _ours(max_magnitude()), BELIEF),
        arm("esi_L50", _ours(sum_magnitude()), BELIEF),
        arm("evi_L50", _ours(posterior_spread()), BELIEF),
    ],
)
