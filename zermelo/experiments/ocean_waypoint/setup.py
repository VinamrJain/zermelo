"""What a sweep module is written with"""

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from zermelo.experiments.ocean_waypoint.schema import BeliefConfig, MethodConfig, ProblemConfig, Resources

cs = ConfigStore.instance()


@dataclass(frozen=True)
class Arm:
    """One fully specified rule and the model it holds, or no rule at all"""

    name: str
    method: MethodConfig | None
    belief: BeliefConfig


def _check_unspecified(block: Any) -> None:
    """Raises where any field was left unset"""
    unspecified = [f.name for f in fields(block) if getattr(block, f.name) == MISSING]
    if unspecified:
        raise ValueError(f"{type(block).__name__} was built without {', '.join(unspecified)}")


def arm(name: str, method: MethodConfig | None, belief: BeliefConfig) -> Arm:
    """One arm of a sweep, refused unless every setting of it was given"""
    if method is not None:
        _check_unspecified(method)
    _check_unspecified(belief)
    return Arm(name, method, belief)


def sweep(
    name: str,
    *,
    problem: ProblemConfig,
    horizon: int,
    claim_every: int,
    recorded: dict[str, int],
    seeds: Sequence[int],
    resources: Resources,
    arms: Sequence[Arm],
) -> None:
    """Register one sweep: its arms crossed with its seeds, the world they share, and what a cell is given"""
    _check_unspecified(problem)
    _check_unspecified(resources)
    for one in arms:
        cs.store(
            group="arm",
            name=f"{name}_{one.name}",
            package="_global_",
            node={
                "problem": problem,
                "belief": one.belief,
                "method": one.method,
                "horizon": horizon,
                "claim_every": claim_every,
                "recorded": recorded,
            },
        )
    cs.store(
        group="sweep",
        name=name,
        package="_global_",
        node={
            "resources": resources,  # read when the launcher is built, before any arm is composed
            "hydra": {
                "mode": "MULTIRUN",
                "sweeper": {
                    "params": {"+arm": ",".join(f"{name}_{one.name}" for one in arms), "seed": ",".join(str(seed) for seed in seeds)}
                },
                "sweep": {
                    "dir": f"results/ocean_waypoint/{name}/${{now:%Y-%m-%d_%H-%M-%S}}",  # sorts lexicographically, holds no colon
                    "subdir": "arm=${hydra:runtime.choices.arm},seed=${seed}",
                },
                "job": {"name": name},
            },
        },
    )


Implementation = dict[str, Any]
"""A dotted path and the arguments to build it with"""


def _built(path: str, **arguments: Any) -> Implementation:
    """One dotted path and its arguments, as the mapping hydra builds from"""
    return {"_target_": path, "_convert_": "all", **arguments}


# --- utilities: which quantity of a belief a rule maximizes --------------------------------------------------------


def information_gain(components: tuple[int, ...]) -> Implementation:
    """sum over j in components of (1/2) log det(I + K_n,g / noise_g^2) over a walk's readings, g the group of j"""
    return _built("zermelo.methods.path_eui.utility.InformationGain", components=components)


def region_information(components: tuple[int, ...]) -> Implementation:
    """sum over j in components of (1/2) [log det(I + K_n,g(w, w) / noise_g^2) - log det(I + S_g / noise_g^2)] over a walk's readings w, S_b its covariance given the candidates, g the group of j"""
    return _built("zermelo.methods.path_eui.utility.RegionInformation", components=components)


def max_magnitude(components: tuple[int, ...]) -> Implementation:
    """The largest reading magnitude over `components` among the cells scored"""
    return _built("zermelo.methods.path_eui.utility.MaxMagnitude", components=components)


def sum_magnitude(components: tuple[int, ...]) -> Implementation:
    """The reading magnitudes over `components` among the cells scored, summed"""
    return _built("zermelo.methods.path_eui.utility.SumMagnitude", components=components)


def total_variance(components: tuple[int, ...]) -> Implementation:
    """Minus the posterior variance of `components` summed over every candidate state, given where the dataset was read"""
    return _built("zermelo.methods.path_eui.utility.TotalVariance", components=components)


def posterior_spread(components: tuple[int, ...]) -> Implementation:
    """The posterior standard deviations of `components` at the scored cells, summed"""
    return _built("zermelo.methods.path_eui.utility.PosteriorSpread", components=components)


def upper_confidence(c: float, components: tuple[int, ...]) -> Implementation:
    """The largest optimistic magnitude `||mean|| + c * spread` over `components` at the scored cells"""
    return _built("zermelo.methods.path_eui.utility.UpperConfidence", c=c, components=components)


def expected_improvement(components: tuple[int, ...]) -> Implementation:
    """`E[(|f| - beta)^+]` at each scored cell in closed form over one component, `beta` the largest magnitude read so far"""
    return _built("zermelo.methods.path_eui.utility.ExpectedImprovement", components=components)


def goals_reached(radius: float) -> Implementation:
    """How many goals lie within `radius` of some reading's state; the goals are set when the episode is built"""
    return _built("zermelo.methods.path_eui.utility.GoalsReached", radius=radius, goals=None)


def uniform() -> Implementation:
    """A draw ignoring the data"""
    return _built("zermelo.methods.path_eui.utility.Uniform")


# --- planners: how the actor is carried to a target ----------------------------------


def value_iteration(value_iterations: int | None, replan_length: int, radius: float, targets_per_solve: int | None) -> Implementation:
    """`v_0 = 0`, `v_{k+1}(s) = 0 on arrival, else min over a of 1 + sum over s' of P(s' | s, a) v_k(s')`, `value_iterations` rounds, `None` the steps left"""
    return _built(
        "zermelo.methods.path_eui.planner.ValueIteration",
        value_iterations=value_iterations,
        replan_length=replan_length,
        goal_radius=radius,
        targets_per_solve=targets_per_solve,
    )


def greedy(value_iterations: int, replan_length: int, radius: float, targets_per_solve: int | None) -> Implementation:
    """One backup on the distance to the target: the act whose next cell is nearest it in expectation"""
    return _built(
        "zermelo.methods.path_eui.planner.Greedy",
        value_iterations=value_iterations,
        replan_length=replan_length,
        goal_radius=radius,
        targets_per_solve=targets_per_solve,
    )
