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


def information_gain(blocks: tuple[int, ...]) -> Implementation:
    """sum over the counted blocks b of (widths[b] / 2) log det(I + K_n,b / noise_b^2) over a walk's readings"""
    return _built("zermelo.methods.waypoint_bo.utility.InformationGain", blocks=blocks)


def region_information(blocks: tuple[int, ...]) -> Implementation:
    """sum over the counted blocks b of (widths[b] / 2) [log det(I + K_n,b(w, w) / noise_b^2) - log det(I + S_b / noise_b^2)] over a walk's readings w, S_b its covariance given the candidates"""
    return _built("zermelo.methods.waypoint_bo.utility.RegionInformation", blocks=blocks)


def max_magnitude(block: int) -> Implementation:
    """The largest reading magnitude over one block's channels among the cells scored"""
    return _built("zermelo.methods.waypoint_bo.utility.MaxMagnitude", block=block)


def sum_magnitude(block: int) -> Implementation:
    """The reading magnitudes over one block's channels among the cells scored, summed"""
    return _built("zermelo.methods.waypoint_bo.utility.SumMagnitude", block=block)


def total_variance(blocks: tuple[int, ...]) -> Implementation:
    """Minus the posterior variance of the counted blocks summed over every candidate state, given where the dataset was read"""
    return _built("zermelo.methods.waypoint_bo.utility.TotalVariance", blocks=blocks)


def posterior_spread(blocks: tuple[int, ...]) -> Implementation:
    """The posterior standard deviations of the counted blocks at the scored cells, summed"""
    return _built("zermelo.methods.waypoint_bo.utility.PosteriorSpread", blocks=blocks)


def upper_confidence(c: float, block: int) -> Implementation:
    """The largest optimistic magnitude `||mean|| + c * spread` over one block's channels at the scored cells"""
    return _built("zermelo.methods.waypoint_bo.utility.UpperConfidence", c=c, block=block)


def expected_improvement(block: int) -> Implementation:
    """`E[(|f| - beta)^+]` at each scored cell in closed form over a one-channel block, `beta` the largest magnitude read so far"""
    return _built("zermelo.methods.waypoint_bo.utility.ExpectedImprovement", block=block)


def targets_reached(radius: float) -> Implementation:
    """How many candidates lie within `radius` of some reading's state"""
    return _built("zermelo.methods.waypoint_bo.utility.TargetsReached", radius=radius)


def uniform() -> Implementation:
    """A draw ignoring the data"""
    return _built("zermelo.methods.waypoint_bo.utility.Uniform")


# --- planners: how the actor is carried to a waypoint, and how that walk is priced ----------------------------------


def value_iteration(
    backup_iterations: int, replan_every: int, radius: float, target_chunk: int | None, cost_weight: float
) -> Implementation:
    """`v_0 = 0`, `v_{k+1}(s) = 0 on arrival, else min over a of c(s, a) + sum over s' of P(s' | s, a) v_k(s')`, `backup_iterations` rounds"""
    return _built(
        "zermelo.methods.waypoint_bo.planner.ValueIteration",
        backup_iterations=backup_iterations,
        replan_every=replan_every,
        radius=radius,
        target_chunk=target_chunk,
        cost_weight=cost_weight,
    )


def greedy(backup_iterations: int, replan_every: int, radius: float, target_chunk: int | None, cost_weight: float) -> Implementation:
    """One backup on the distance to the target: the act whose next cell is nearest it in expectation"""
    return _built(
        "zermelo.methods.waypoint_bo.planner.Greedy",
        backup_iterations=backup_iterations,
        replan_every=replan_every,
        radius=radius,
        target_chunk=target_chunk,
        cost_weight=cost_weight,
    )
