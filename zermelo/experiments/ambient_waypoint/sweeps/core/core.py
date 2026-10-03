import dataclasses

from zermelo.experiments.ambient_waypoint.registry import (
    AMBIENT1_CONTROL1,
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    N_SEEDS,
    OPENING_STEPS,
    PLANNING_BUDGET,
    RECORDED,
    RESOURCES,
    matched_belief,
)
from zermelo.experiments.ambient_waypoint.schema import MethodConfig
from zermelo.experiments.ambient_waypoint.setup import (
    Implementation,
    arm,
    expected_improvement,
    max_magnitude,
    posterior_spread,
    sweep,
    uniform,
    upper_confidence,
    value_iteration,
)

BELIEF = matched_belief(AMBIENT1_CONTROL1)
"""The same model on every acquisition"""


def _method(
    utility: Implementation, *, improvement: bool, n_fields: int, n_walks: int, step_rate: float, replan_every: int
) -> MethodConfig:
    """One acquisition on a value-iteration planner, at the settings the acquisitions differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_every=replan_every, radius=0.5, target_chunk=None),
        improvement=improvement,
        n_fields=n_fields,
        n_walks=n_walks,
        combination="linear",
        step_rate=step_rate,
        steps_from="predicted",
        n_candidates=None,
        opening_steps=OPENING_STEPS,
    )


def _baseline(utility: Implementation, n_fields: int) -> MethodConfig:
    """A rule reading the posterior (or `n_fields` draws of it) at the candidates, charging no travel"""
    return _method(utility, improvement=False, n_fields=n_fields, n_walks=0, step_rate=0.0, replan_every=PLANNING_BUDGET)


def _ours(utility: Implementation, *, step_rate: float = 0.5, replan_every: int = PLANNING_BUDGET) -> MethodConfig:
    """Expected improvement of `utility` over 16 imagined walks under 16 drawn fields, a full-budget trip charged `step_rate` of the spread"""
    return _method(utility, improvement=True, n_fields=16, n_walks=16, step_rate=step_rate, replan_every=replan_every)


sweep(
    "core",
    problem=AMBIENT1_CONTROL1,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=dataclasses.replace(RESOURCES, mem_gb=16, timeout_min=480, array_parallelism=40),
    arms=[
        arm("rand_act", None, BELIEF),  # no rule at all: acts uniformly and claims the prior
        arm("rand_wp", _baseline(uniform(), 0), BELIEF),
        arm("max_var", _baseline(posterior_spread(), 0), BELIEF),
        arm("ucb", _baseline(upper_confidence(2.0), 0), BELIEF),
        arm("ei", _baseline(expected_improvement(), 0), BELIEF),
        arm("ts", _baseline(max_magnitude(), 1), BELIEF),
        arm("emi_L25", _ours(max_magnitude()), BELIEF),
        arm("emi_L50", _ours(max_magnitude(), replan_every=50), BELIEF),
        arm("emi_c1", _ours(max_magnitude(), step_rate=1.0), BELIEF),
        arm("evi_L25", _ours(posterior_spread()), BELIEF),
        arm("evi_L50", _ours(posterior_spread(), replan_every=50), BELIEF),
        arm("evi_c1", _ours(posterior_spread(), step_rate=1.0), BELIEF),
    ],
)
