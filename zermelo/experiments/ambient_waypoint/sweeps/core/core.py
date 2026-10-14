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
    utility: Implementation,
    *,
    subtract_dataset_utility: bool,
    n_field_samples: int,
    n_paths_per_field: int,
    penalty_rate: float,
    replan_length: int,
) -> MethodConfig:
    """One acquisition on a value-iteration planner, at the settings the acquisitions differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_length=replan_length, radius=0.5, targets_per_solve=None),
        subtract_dataset_utility=subtract_dataset_utility,
        n_field_samples=n_field_samples,
        n_paths_per_field=n_paths_per_field,
        penalty_rate=penalty_rate,
        n_candidates_scored=None,
        random_opening_steps=OPENING_STEPS,
    )


def _baseline(utility: Implementation, n_field_samples: int) -> MethodConfig:
    """A rule reading the posterior (or `n_field_samples` draws of it) at the candidates, charging no travel"""
    return _method(
        utility,
        subtract_dataset_utility=False,
        n_field_samples=n_field_samples,
        n_paths_per_field=0,
        penalty_rate=0.0,
        replan_length=PLANNING_BUDGET,
    )


def _ours(utility: Implementation, *, penalty_rate: float = 0.5, replan_length: int = PLANNING_BUDGET) -> MethodConfig:
    """Expected improvement of `utility` over 16 imagined walks under 16 drawn fields, a full-budget trip charged `penalty_rate` of the spread"""
    return _method(
        utility,
        subtract_dataset_utility=True,
        n_field_samples=16,
        n_paths_per_field=16,
        penalty_rate=penalty_rate,
        replan_length=replan_length,
    )


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
        arm("emi_L50", _ours(max_magnitude(), replan_length=50), BELIEF),
        arm("emi_c1", _ours(max_magnitude(), penalty_rate=1.0), BELIEF),
        arm("evi_L25", _ours(posterior_spread()), BELIEF),
        arm("evi_L50", _ours(posterior_spread(), replan_length=50), BELIEF),
        arm("evi_c1", _ours(posterior_spread(), penalty_rate=1.0), BELIEF),
    ],
)
