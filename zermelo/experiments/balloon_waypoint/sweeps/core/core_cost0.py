import dataclasses

from zermelo.experiments.balloon_waypoint.registry import (
    ARRIVAL_RADIUS_KM,
    ATLANTIC_2017SEP,
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    N_CANDIDATES,
    N_SEEDS,
    OPENING_STEPS,
    PLANNING_BUDGET,
    RECORDED,
    RESOURCES,
    TARGET_CHUNK,
    error_belief,
    wind_belief,
)
from zermelo.experiments.balloon_waypoint.schema import MethodConfig
from zermelo.experiments.balloon_waypoint.setup import (
    Implementation,
    arm,
    max_magnitude,
    posterior_spread,
    sum_magnitude,
    sweep,
    uniform,
    upper_confidence,
    value_iteration,
)

BELIEFS = {"error": error_belief(), "wind": wind_belief()}
"""Learning the forecast's error about the forecast, or the wind about zero"""


def _method(
    utility: Implementation,
    *,
    subtract_dataset_utility: bool,
    n_field_samples: int,
    n_paths_per_field: int,
    penalty_rate: float,
    replan_length: int,
    cost_weight: float,
) -> MethodConfig:
    """One acquisition on a value-iteration planner, at the settings the acquisitions differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(
            replan_length=replan_length, radius=ARRIVAL_RADIUS_KM, targets_per_solve=TARGET_CHUNK, cost_weight=cost_weight
        ),
        subtract_dataset_utility=subtract_dataset_utility,
        n_field_samples=n_field_samples,
        n_paths_per_field=n_paths_per_field,
        penalty_rate=penalty_rate,
        n_candidates_scored=N_CANDIDATES,
        random_opening_steps=OPENING_STEPS,
    )


COST_WEIGHT = 0.0
"""Price of one altitude change in time_steps: none, the `tuning` sweep steering only at zero"""

PENALTY_RATE = 1.0
"""Share of the spread of worth a full-budget trip costs, the `tuning` sweep's best"""


def _baseline(utility: Implementation, n_field_samples: int) -> MethodConfig:
    """A rule reading the posterior (or `n_field_samples` draws of it) at the candidates, charging no travel"""
    return _method(
        utility,
        subtract_dataset_utility=False,
        n_field_samples=n_field_samples,
        n_paths_per_field=0,
        penalty_rate=0.0,
        replan_length=PLANNING_BUDGET,
        cost_weight=COST_WEIGHT,
    )


def _ours(utility: Implementation, *, penalty_rate: float = PENALTY_RATE, replan_length: int = PLANNING_BUDGET) -> MethodConfig:
    """Expected improvement of `utility` over 16 imagined walks under 16 drawn fields"""
    return _method(
        utility,
        subtract_dataset_utility=True,
        n_field_samples=16,
        n_paths_per_field=16,
        penalty_rate=penalty_rate,
        replan_length=replan_length,
        cost_weight=COST_WEIGHT,
    )


def _per_belief(name: str, method: MethodConfig | None) -> list:
    """One acquisition under each belief, the belief's name appended"""
    return [arm(f"{name}_{held}", method, belief) for held, belief in BELIEFS.items()]


sweep(
    "core_cost0",
    problem=ATLANTIC_2017SEP,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high"),  # 48 GB cards and up
    arms=[
        arm("rand_act", None, BELIEFS["error"]),  # no rule at all: acts uniformly and claims the prior
        # the forecast flown as if exact: a belief that cannot move off it, aimed at the fastest forecast wind
        arm(
            "forecast",
            _method(
                sum_magnitude(),
                subtract_dataset_utility=False,
                n_field_samples=0,
                n_paths_per_field=16,
                penalty_rate=PENALTY_RATE,
                replan_length=PLANNING_BUDGET,
                cost_weight=COST_WEIGHT,
            ),
            dataclasses.replace(BELIEFS["error"], amplitude=1e-3),
        ),
        *_per_belief("rand_wp", _baseline(uniform(), 0)),
        *_per_belief("max_var", _baseline(posterior_spread(), 0)),
        *_per_belief("ucb", _baseline(upper_confidence(2.0), 0)),
        *_per_belief("ts", _baseline(max_magnitude(), 1)),
        *_per_belief("esi_L50", _ours(sum_magnitude())),
        *_per_belief("evi_L50", _ours(posterior_spread())),
    ],
)
