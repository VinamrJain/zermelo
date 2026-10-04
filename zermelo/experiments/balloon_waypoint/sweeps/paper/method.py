"""The rules the paper compares on still wind, and the settings our own run at"""

from zermelo.experiments.balloon_waypoint.registry import ARRIVAL_RADIUS_KM, N_CANDIDATES, OPENING_STEPS, PLANNING_BUDGET, TARGET_CHUNK
from zermelo.experiments.balloon_waypoint.schema import MethodConfig
from zermelo.experiments.balloon_waypoint.setup import Implementation, max_magnitude, posterior_spread, sum_magnitude, value_iteration

UTILITIES = {"emi": max_magnitude, "evi": posterior_spread, "esi": sum_magnitude}
"""What each of our rules takes the expected improvement of"""

TUNED_PLANNING_BUDGET = {"emi": 100, "evi": 100, "esi": 100}
"""`L` each of our rules replans at, picked on the `paper_tuning` sweep"""

TUNED_STEP_RATE = {"emi": 0.0, "evi": 0.0, "esi": 0.0}
"""`c` each of our rules charges a full-budget trip, picked on the `paper_tuning` sweep"""

TUNED_COST_WEIGHT = {"emi": 0.0, "evi": 0.0, "esi": 0.0}
"""`w`, the time_steps the planner prices one altitude change at under a budget, picked on the `paper_tuning_cost_weight` sweep"""

N_SEEDS = 5
"""Instances every arm is run on at each frame"""


def method(
    utility: Implementation, *, improvement: bool, n_fields: int, n_walks: int, step_rate: float, replan_every: int, cost_weight: float
) -> MethodConfig:
    """One acquisition on a value-iteration planner, at the settings the acquisitions differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_every=replan_every, radius=ARRIVAL_RADIUS_KM, target_chunk=TARGET_CHUNK, cost_weight=cost_weight),
        improvement=improvement,
        n_fields=n_fields,
        n_walks=n_walks,
        combination="linear",
        step_rate=step_rate,
        steps_from="predicted",
        n_candidates=N_CANDIDATES,
        opening_steps=OPENING_STEPS,
    )


def baseline(utility: Implementation, n_fields: int) -> MethodConfig:
    """A rule reading the posterior (or `n_fields` draws of it) at the candidates, charging no travel and no altitude change"""
    return method(utility, improvement=False, n_fields=n_fields, n_walks=0, step_rate=0.0, replan_every=PLANNING_BUDGET, cost_weight=0.0)


def mc_ei() -> MethodConfig:
    """Expected improvement of the largest speed at the target alone, over 16 drawn winds and no walk"""
    return method(max_magnitude(), improvement=True, n_fields=16, n_walks=0, step_rate=0.0, replan_every=PLANNING_BUDGET, cost_weight=0.0)


def ours(name: str, *, step_rate: float | None = None, replan_every: int | None = None, cost_weight: float = 0.0) -> MethodConfig:
    """Expected improvement of `name`'s utility over 16 imagined walks under 16 drawn winds, at its tuned `L` and `c` unless given"""
    return method(
        UTILITIES[name](),
        improvement=True,
        n_fields=16,
        n_walks=16,
        step_rate=TUNED_STEP_RATE[name] if step_rate is None else step_rate,
        replan_every=TUNED_PLANNING_BUDGET[name] if replan_every is None else replan_every,
        cost_weight=cost_weight,
    )


def rate_name(value: float) -> str:
    """A number as an arm name carries it: 0.5 -> `0p5`, 2.0 -> `2`"""
    return f"{value:g}".replace(".", "p")
