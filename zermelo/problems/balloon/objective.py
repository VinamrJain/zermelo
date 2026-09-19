"""What a balloon is scored on: standing in the fastest wind the candidates offer at that hour

W(position, altitude, hours_elapsed) = (u, v)     the wind, m/s
candidates                  the states the balloon is scored on, a latitude-longitude box at every altitude
target                      the scalar read off W at a state; speed = ||W||
at every time_step, from the state arrived in:
    truth_t[c]            = target at candidate c, at this hours_elapsed `t`         (n_candidates,)
    best_possible_speed_t = max over c of truth_t[c]
    speed_t               = ||W(state)|| on a candidate, 0 otherwise
    reward_t              = speed
    regret_t              = best_possible_speed - speed                           derived
claim_t                       (u, v) predicted at every candidate, as means then log-variances
"""

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from zermelo.interface import BoxDomain, Decision, Domain, FunctionDomain, Objective, ProductDomain, Subset
from zermelo.problems.balloon.field import WindField
from zermelo.problems.balloon.grid import SphereGrid


class Target(ABC):
    """The scalar of W an episode is scored on"""

    @abstractmethod
    def __call__(self, field: WindField, candidates: dict[str, Any]) -> Float[Array, " n_candidates"]:
        """The scalar at every candidate state"""


@dataclass(frozen=True)
class PointSpeed(Target):
    """speed = ||W(state)||"""

    def __call__(self, field: WindField, candidates: dict[str, Any]) -> Float[Array, " n_candidates"]:
        """Speed at each candidate"""
        return jnp.linalg.norm(field(candidates), axis=-1)


@dataclass(frozen=True)
class ColumnPeakSpeed(Target):
    """max over altitudes of ||W|| above the candidate's grid point, one value shared down a column"""

    n_alt: int
    """How many altitudes the column spans"""

    def __call__(self, field: WindField, candidates: dict[str, Any]) -> Float[Array, " n_candidates"]:
        """The fastest wind anywhere in each candidate's column"""
        above = {name: part for name, part in candidates.items() if name != "altitude"}  # position, and the hour
        per_altitude = jnp.stack(
            [
                jnp.linalg.norm(field(above | {"altitude": jnp.full_like(candidates["altitude"], level)}), axis=-1)
                for level in range(self.n_alt)
            ]
        )  # (alt, n_candidates)
        return jnp.max(per_altitude, axis=0)


@dataclass(frozen=True)
class StormSearch(Objective[dict[str, Float[Array, "..."]]]):
    """Reward is the speed of the wind the balloon stands in, counted on a candidate, with the best on offer and the claim recorded beside it"""

    target: Target

    candidates: Subset[dict[str, Any]]
    """The states scored, over the state domain a belief is maintained on"""

    candidate_states: dict[str, Any] = dataclasses.field(init=False, repr=False)
    """The live candidates, gathered once"""

    candidate_indices: Int[Array, " n_candidates"] = dataclasses.field(init=False, repr=False)
    """Their indices into the state domain"""

    def __post_init__(self) -> None:
        """Gather the live candidates as concrete arrays"""
        live = jnp.flatnonzero(self.candidates.live)  # (n_candidates,) indices into the state domain
        object.__setattr__(self, "candidate_indices", live)
        object.__setattr__(self, "candidate_states", jax.tree.map(lambda a: a[live], self.candidates.elements()))

    @property
    def claim_domain(self) -> Domain:
        """A claim is a function: the mean of (u, v) at any candidate, then their log-variances"""
        return FunctionDomain(self.candidates, BoxDomain((4,)))

    def observe(self, state: dict[str, Any]) -> dict[str, Float[Array, "..."]]:
        """truth, best_possible_speed and speed at `state`, at the hour it carries"""
        truth = self.target(state["field"], self.candidate_states | {"hours_elapsed": state["hours_elapsed"]})  # (n_candidates,)
        on_candidate = self.candidates.live[self.candidates.index_of({part: state[part] for part in self.candidate_states})]
        speed = jnp.where(on_candidate, jnp.linalg.norm(state["field"](state)), 0.0)
        return {"truth": truth, "best_possible_speed": jnp.max(truth), "speed": speed}

    def reset(self, state: dict[str, Any]) -> dict[str, Float[Array, "..."]]:
        """The objective's memory before any time_step, nothing claimed yet"""
        observed = self.observe(state)
        return observed | {"claim": jnp.zeros((observed["truth"].shape[0], 4))}  # (n_candidates, 4)

    def score(
        self, objective_state: dict[str, Float[Array, "..."]], state: dict[str, Any], decision: Decision, next_state: dict[str, Any]
    ) -> tuple[dict[str, Float[Array, "..."]], Float[Array, ""]]:
        """The speed at the state arrived in, and the claim as it stood"""
        observed = self.observe(next_state)
        return observed | {"claim": decision.claim(self.candidate_states)}, observed["speed"]


def candidate_box(grid: SphereGrid, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> Bool[Array, " pos"]:
    """The grid points with lat_min <= lat <= lat_max and lon_min <= lon <= lon_max, a box that does not cross the 180 degree meridian"""
    if lat_min > lat_max or lon_min > lon_max:
        raise ValueError(
            f"a box runs from its smaller bound to its larger, and this one is lat {lat_min}..{lat_max}, lon {lon_min}..{lon_max}"
        )
    at = grid.elements()  # (pos, 2)
    return (at[:, 0] >= lat_min) & (at[:, 0] <= lat_max) & (at[:, 1] >= lon_min) & (at[:, 1] <= lon_max)


def balloon_candidates(states: ProductDomain, grid: SphereGrid, box: Bool[Array, " pos"]) -> Subset[dict[str, Any]]:
    """`states` above the grid points `box` marks, at every altitude"""
    return states.narrow(box[grid.flat_index(states.elements()["position"])])
