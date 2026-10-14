"""The rules the paper compares on the 2D grid, and the settings they run at"""

from zermelo.experiments.ambient_waypoint.registry import OPENING_STEPS
from zermelo.experiments.ambient_waypoint.schema import MethodConfig
from zermelo.experiments.ambient_waypoint.setup import Implementation, max_magnitude, posterior_spread, sum_magnitude, value_iteration

UTILITIES = {"emi": max_magnitude, "evi": posterior_spread, "esi": sum_magnitude}
"""What each of our rules takes the expected improvement of"""

PLANNING_BUDGET = 25
"""`L` every rule replans at"""

PENALTY_RATE = 0.0
"""`c` each of our rules charges a full-budget trip"""

TUNING_SEEDS = range(100, 105)
"""Instances the settings above are picked on, disjoint from the ones reported"""


def method(
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


def baseline(utility: Implementation, n_field_samples: int) -> MethodConfig:
    """A rule reading the posterior (or `n_field_samples` draws of it) at the candidates, charging no travel"""
    return method(
        utility,
        subtract_dataset_utility=False,
        n_field_samples=n_field_samples,
        n_paths_per_field=0,
        penalty_rate=0.0,
        replan_length=PLANNING_BUDGET,
    )


def mc_ei() -> MethodConfig:
    """Expected improvement of the largest magnitude at the target alone, over 16 drawn fields and no walk"""
    return method(
        max_magnitude(),
        subtract_dataset_utility=True,
        n_field_samples=16,
        n_paths_per_field=0,
        penalty_rate=0.0,
        replan_length=PLANNING_BUDGET,
    )


def ours(name: str, *, penalty_rate: float | None = None, replan_length: int | None = None) -> MethodConfig:
    """Expected improvement of `name`'s utility over 16 imagined walks under 16 drawn fields, at `PENALTY_RATE` and `PLANNING_BUDGET` unless given"""
    return method(
        UTILITIES[name](),
        subtract_dataset_utility=True,
        n_field_samples=16,
        n_paths_per_field=16,
        penalty_rate=PENALTY_RATE if penalty_rate is None else penalty_rate,
        replan_length=PLANNING_BUDGET if replan_length is None else replan_length,
    )


def rate_name(value: float) -> str:
    """A number as an arm name carries it: 0.5 -> `0p5`, 2.0 -> `2`"""
    return f"{value:g}".replace(".", "p")
