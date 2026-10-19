import dataclasses

from zermelo.experiments.balloon_waypoint.registry import (
    ARRIVAL_RADIUS_KM,
    ATLANTIC_2017SEP,
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    N_CANDIDATES,
    N_SEEDS,
    PLANNING_BUDGET,
    RECORDED,
    RESOURCES,
    oracle_belief,
)
from zermelo.experiments.balloon_waypoint.schema import MethodConfig
from zermelo.experiments.balloon_waypoint.setup import (
    Implementation,
    arm,
    max_magnitude,
    posterior_spread,
    sweep,
    uniform,
    upper_confidence,
    value_iteration,
)

BELIEF = oracle_belief()
"""The true field on every arm, so a rule reading a spread reads zero"""


def _method(
    utility: Implementation, *, subtract_dataset_utility: bool, n_field_samples: int, n_paths_per_field: int, lambda_: float
) -> MethodConfig:
    """This sweep's held values"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_length=PLANNING_BUDGET, radius=ARRIVAL_RADIUS_KM, targets_per_solve=None),
        subtract_dataset_utility=subtract_dataset_utility,
        n_field_samples=n_field_samples,
        n_paths_per_field=n_paths_per_field,
        lambda_=lambda_,
        n_candidates_scored=N_CANDIDATES,
    )


def _sampled(utility: Implementation, *, subtract_dataset_utility: bool) -> MethodConfig:
    """`utility` at this sweep's draw counts and charge"""
    return _method(utility, subtract_dataset_utility=subtract_dataset_utility, n_field_samples=16, n_paths_per_field=16, lambda_=1 / 3)


sweep(
    "utilities",
    problem=ATLANTIC_2017SEP,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=RESOURCES,
    # an increment differs from a level only for `max_magnitude`, the one rule reading the imagined readings
    arms=[
        arm("max_magnitude", _sampled(max_magnitude(), subtract_dataset_utility=True), BELIEF),
        arm("posterior_spread", _sampled(posterior_spread(), subtract_dataset_utility=False), BELIEF),
        arm("ucb2", _sampled(upper_confidence(2.0), subtract_dataset_utility=False), BELIEF),
        # a charge would bias a uniform draw, which spans [0, 1] whatever the field
        arm(
            "random_search", _method(uniform(), subtract_dataset_utility=False, n_field_samples=0, n_paths_per_field=0, lambda_=0.0), BELIEF
        ),
    ],
)
