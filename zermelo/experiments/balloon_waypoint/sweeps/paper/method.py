"""The rules the paper compares on still wind, and the settings they run at"""

from zermelo.experiments.balloon_waypoint.registry import ARRIVAL_RADIUS_KM, N_CANDIDATES, OPENING_STEPS, TARGET_CHUNK
from zermelo.experiments.balloon_waypoint.schema import MethodConfig
from zermelo.experiments.balloon_waypoint.setup import Implementation, max_magnitude, posterior_spread, sum_magnitude, value_iteration

UTILITIES = {"emi": max_magnitude, "evi": posterior_spread, "esi": sum_magnitude}
"""What each of our rules takes the expected improvement of"""

PLANNING_BUDGET = 100
"""`L` every rule replans at"""

STEP_RATE = 0.0
"""`c` each of our rules charges a full-budget trip"""

BUDGETED_STEP_RATE = 1.0
"""`c` each of our rules charges a full-budget trip where the altitude budget binds"""

COST_WEIGHT = 1.0
"""`w`, the time_steps the planner prices one altitude change at where the altitude budget binds"""

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
    """Expected improvement of `name`'s utility over 16 imagined walks under 16 drawn winds, at `PLANNING_BUDGET` and `STEP_RATE` unless given"""
    return method(
        UTILITIES[name](),
        improvement=True,
        n_fields=16,
        n_walks=16,
        step_rate=STEP_RATE if step_rate is None else step_rate,
        replan_every=PLANNING_BUDGET if replan_every is None else replan_every,
        cost_weight=cost_weight,
    )


def rate_name(value: float) -> str:
    """A number as an arm name carries it: 0.5 -> `0p5`, 2.0 -> `2`"""
    return f"{value:g}".replace(".", "p")
