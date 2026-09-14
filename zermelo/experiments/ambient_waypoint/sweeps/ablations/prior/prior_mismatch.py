import dataclasses

from zermelo.experiments.ambient_waypoint.registry import (
    AMBIENT1_CONTROL1,
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    N_SEEDS,
    OPENING_LEGS,
    PLANNING_BUDGET,
    RECORDED,
    RESOURCES,
    matched_belief,
)
from zermelo.experiments.ambient_waypoint.schema import BeliefConfig, MethodConfig
from zermelo.experiments.ambient_waypoint.setup import arm, max_magnitude, sweep, value_iteration

MATCHED = matched_belief(AMBIENT1_CONTROL1)

BELIEFS: dict[str, BeliefConfig] = {
    "matched": MATCHED,
    "matern32": dataclasses.replace(MATCHED, kernel="gpjax.kernels.Matern32"),
    "lengthscale5": dataclasses.replace(MATCHED, lengthscale=5.0),  # the truth's is 2.5
    "amplitude2": dataclasses.replace(MATCHED, amplitude=2.0),  # the truth's is 3.0, as a standard deviation
}

METHOD = MethodConfig(
    utility=max_magnitude(),
    planner=value_iteration(max_steps=PLANNING_BUDGET, radius=0.5, target_chunk=None),
    improvement=True,
    n_fields=16,
    n_walks=16,
    combination="linear",
    step_rate=0.5,
    steps_from="predicted",
    n_candidates=None,
    opening_legs=OPENING_LEGS,
)
"""Held on every arm"""

sweep(
    "prior_mismatch",
    problem=AMBIENT1_CONTROL1,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=RESOURCES,
    arms=[arm(name, METHOD, belief) for name, belief in BELIEFS.items()],
)
