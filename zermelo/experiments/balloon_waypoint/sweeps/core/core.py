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
    utility: Implementation, *, improvement: bool, n_fields: int, n_walks: int, step_rate: float, replan_every: int, cost_weight: float
) -> MethodConfig:
    """One acquisition on a value-iteration planner, at the settings the acquisitions differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_every=replan_every, radius=ARRIVAL_RADIUS_KM, target_chunk=None, cost_weight=cost_weight),
        improvement=improvement,
        n_fields=n_fields,
        n_walks=n_walks,
        combination="linear",
        step_rate=step_rate,
        steps_from="predicted",
        n_candidates=N_CANDIDATES,
        opening_steps=OPENING_STEPS,
    )


COST_WEIGHT = 0.25
"""Price of one altitude change in time_steps, to be set from the `tuning` sweep"""

STEP_RATE = 0.5
"""Share of the spread of worth a full-budget trip costs, to be set from the `tuning` sweep"""


def _baseline(utility: Implementation, n_fields: int) -> MethodConfig:
    """A rule reading the posterior (or `n_fields` draws of it) at the candidates, charging no travel"""
    return _method(
        utility, improvement=False, n_fields=n_fields, n_walks=0, step_rate=0.0, replan_every=PLANNING_BUDGET, cost_weight=COST_WEIGHT
    )


def _ours(utility: Implementation, *, step_rate: float = STEP_RATE, replan_every: int = PLANNING_BUDGET) -> MethodConfig:
    """Expected improvement of `utility` over 16 imagined walks under 16 drawn fields"""
    return _method(
        utility, improvement=True, n_fields=16, n_walks=16, step_rate=step_rate, replan_every=replan_every, cost_weight=COST_WEIGHT
    )


def _per_belief(name: str, method: MethodConfig | None) -> list:
    """One acquisition under each belief, the belief's name appended"""
    return [arm(f"{name}_{held}", method, belief) for held, belief in BELIEFS.items()]


sweep(
    "core",
    problem=ATLANTIC_2017SEP,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=RESOURCES,
    arms=[
        arm("rand_act", None, BELIEFS["error"]),  # no rule at all: acts uniformly and claims the prior
        # the forecast flown as if exact: a belief that cannot move off it, aimed at the fastest forecast wind
        arm(
            "forecast",
            _method(
                sum_magnitude(),
                improvement=False,
                n_fields=0,
                n_walks=16,
                step_rate=STEP_RATE,
                replan_every=PLANNING_BUDGET,
                cost_weight=COST_WEIGHT,
            ),
            dataclasses.replace(BELIEFS["error"], amplitude=1e-3),
        ),
        *_per_belief("rand_wp", _baseline(uniform(), 0)),
        *_per_belief("max_var", _baseline(posterior_spread(), 0)),
        *_per_belief("ucb", _baseline(upper_confidence(2.0), 0)),
        *_per_belief("ts", _baseline(max_magnitude(), 1)),
        *_per_belief("esi_L1", _ours(sum_magnitude(), replan_every=1)),
        *_per_belief("esi_L8", _ours(sum_magnitude(), replan_every=8)),
        *_per_belief("esi_L25", _ours(sum_magnitude())),
        *_per_belief("esi_c1", _ours(sum_magnitude(), step_rate=1.0)),
        *_per_belief("evi_L1", _ours(posterior_spread(), replan_every=1)),
        *_per_belief("evi_L8", _ours(posterior_spread(), replan_every=8)),
        *_per_belief("evi_L25", _ours(posterior_spread())),
        *_per_belief("evi_c1", _ours(posterior_spread(), step_rate=1.0)),
    ],
)
