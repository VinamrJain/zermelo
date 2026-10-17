import dataclasses

from zermelo.experiments.ambient_waypoint.registry import (
    AMBIENT1_CONTROL1,
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    N_SEEDS,
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
"""The same model on every arm, matched to the truth"""


def _method(
    utility: Implementation, *, subtract_dataset_utility: bool, n_field_samples: int, n_paths_per_field: int, lambda_: float
) -> MethodConfig:
    """This sweep's held values"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_length=PLANNING_BUDGET, radius=0.5, targets_per_solve=None),
        subtract_dataset_utility=subtract_dataset_utility,
        n_field_samples=n_field_samples,
        n_paths_per_field=n_paths_per_field,
        lambda_=lambda_,
        n_candidates_scored=None,
    )


sweep(
    "acquisitions",
    problem=AMBIENT1_CONTROL1,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=dataclasses.replace(RESOURCES, mem_gb=16, timeout_min=480, array_parallelism=40),
    arms=[
        arm("random_act", None, BELIEF),  # no rule at all: acts uniformly and claims the prior
        arm(
            "random_search", _method(uniform(), subtract_dataset_utility=False, n_field_samples=0, n_paths_per_field=0, lambda_=0.0), BELIEF
        ),
        arm(
            "max_variance",
            _method(posterior_spread(), subtract_dataset_utility=False, n_field_samples=0, n_paths_per_field=0, lambda_=0.0),
            BELIEF,
        ),
        arm(
            "ucb2",
            _method(upper_confidence(2.0), subtract_dataset_utility=False, n_field_samples=0, n_paths_per_field=0, lambda_=0.0),
            BELIEF,
        ),
        arm(
            "ei",
            _method(expected_improvement(), subtract_dataset_utility=False, n_field_samples=0, n_paths_per_field=0, lambda_=0.0),
            BELIEF,
        ),
        arm(
            "mc_ei16", _method(max_magnitude(), subtract_dataset_utility=True, n_field_samples=16, n_paths_per_field=0, lambda_=0.0), BELIEF
        ),
        arm(
            "thompson",
            _method(max_magnitude(), subtract_dataset_utility=False, n_field_samples=1, n_paths_per_field=0, lambda_=0.0),
            BELIEF,
        ),
        arm(
            "eui_fields16_walks16_rate0p5",
            _method(max_magnitude(), subtract_dataset_utility=True, n_field_samples=16, n_paths_per_field=16, lambda_=1 / 3),
            BELIEF,
        ),
    ],
)
