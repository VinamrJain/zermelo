"""The rules the paper compares on still wind, and the settings they run at"""

from zermelo.experiments.balloon_waypoint.registry import ARRIVAL_RADIUS_KM, N_CANDIDATES, OPENING_STEPS, TARGET_CHUNK
from zermelo.experiments.balloon_waypoint.schema import MethodConfig
from zermelo.experiments.balloon_waypoint.setup import Implementation, max_magnitude, posterior_spread, sum_magnitude, value_iteration

UTILITIES = {"emi": max_magnitude, "evi": posterior_spread, "esi": sum_magnitude}
"""What each of our rules takes the expected improvement of"""

PLANNING_BUDGET = 100
"""`L` every rule replans at"""

LAMBDA = 0.0
"""`lambda` each of our rules charges the altitude changes of its walk"""

BUDGETED_LAMBDA = 0.5
"""`lambda` each of our rules charges the altitude changes of its walk where the altitude budget binds"""

N_SEEDS = 5
"""Instances every arm is run on at each frame"""


def method(
    utility: Implementation,
    *,
    subtract_dataset_utility: bool,
    n_field_samples: int,
    n_paths_per_field: int,
    lambda_: float,
    replan_length: int,
) -> MethodConfig:
    """One acquisition on a value-iteration planner, at the settings the acquisitions differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_length=replan_length, radius=ARRIVAL_RADIUS_KM, targets_per_solve=TARGET_CHUNK),
        subtract_dataset_utility=subtract_dataset_utility,
        n_field_samples=n_field_samples,
        n_paths_per_field=n_paths_per_field,
        lambda_=lambda_,
        n_candidates_scored=N_CANDIDATES,
        random_opening_steps=OPENING_STEPS,
    )


def baseline(utility: Implementation, n_field_samples: int) -> MethodConfig:
    """A rule reading the posterior (or `n_field_samples` draws of it) at the candidates, charging no travel and no altitude change"""
    return method(
        utility,
        subtract_dataset_utility=False,
        n_field_samples=n_field_samples,
        n_paths_per_field=0,
        lambda_=0.0,
        replan_length=PLANNING_BUDGET,
    )


def mc_ei() -> MethodConfig:
    """Expected improvement of the largest speed at the target alone, over 16 drawn winds and no walk"""
    return method(
        max_magnitude(), subtract_dataset_utility=True, n_field_samples=16, n_paths_per_field=0, lambda_=0.0, replan_length=PLANNING_BUDGET
    )


def ours(name: str, *, lambda_: float | None = None, replan_length: int | None = None) -> MethodConfig:
    """Expected improvement of `name`'s utility over 16 imagined walks under 16 drawn winds, at `PLANNING_BUDGET` and `LAMBDA` unless given"""
    return method(
        UTILITIES[name](),
        subtract_dataset_utility=True,
        n_field_samples=16,
        n_paths_per_field=16,
        lambda_=LAMBDA if lambda_ is None else lambda_,
        replan_length=PLANNING_BUDGET if replan_length is None else replan_length,
    )


def rate_name(value: float) -> str:
    """A number as an arm name carries it: 0.5 -> `0p5`, 2.0 -> `2`"""
    return f"{value:g}".replace(".", "p")
