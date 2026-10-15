"""What a boat is scored on: learning the ocean current over a region, finding where it runs fastest, or visiting every target

field(position) = (W_u, W_v, U_u, U_v)    the ocean current W and the wind U, m/s
candidates                the grid points a waypoint may be aimed at and a claim is scored on
truth[c]                  field(c) at candidate c                                (n_candidates, components)
claim                     field predicted at every candidate, as means then log-variances   (n_candidates, 2 components)
learning the region:
    information           = (1/2) sum over candidates and the scored components of (prior_log_variance - claimed log-variance), nats
    reward                = information gained this time_step
or, searching the region for the fastest current:
    best_observed         = max over positions z_u stood on so far, z_u a candidate, of ||W(z_u)||
    best_possible         = max over candidates of ||W(c)||
    reward                = max(0, ||W(position)|| - best_observed before) where position is a candidate, else 0
or, visiting every target, the candidates being the targets:
    visited[i]            = visited[i] before, or ||position - candidate i|| <= arrival_km on the tangent plane
    reward                = 0 where every target is visited, else -1
    steps_to_completion   = time_steps rewarded -1 so far
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from zermelo.interface import BoxDomain, Decision, Domain, FunctionDomain, Objective, Subset
from zermelo.problems.ocean.field import COMPONENTS, OCEAN_CURRENT
from zermelo.problems.ocean.grid import PlaneGrid


@dataclass(frozen=True)
class LearnRegion(Objective[dict[str, Float[Array, "..."]]]):
    """Reward is the nats the claim's marginals shed over the candidates this time_step, with the truth recorded beside it"""

    candidates: Subset[dict[str, Any]]
    """The states a waypoint may be aimed at and a claim is scored on, over the state domain a belief is maintained on"""

    prior_log_variance: Float[Array, " components"]
    """log Var[field(c)] per component before any reading, the level information is counted from"""

    scored: tuple[int, ...]
    """Which components of the field the information is counted over"""

    candidate_states: dict[str, Any] = dataclasses.field(init=False, repr=False)
    """The live candidates, gathered once"""

    def __post_init__(self) -> None:
        """Gather the live candidates as concrete arrays"""
        live = jnp.flatnonzero(self.candidates.live)  # (n_candidates,) indices into the state domain
        object.__setattr__(self, "candidate_states", jax.tree.map(lambda a: a[live], self.candidates.elements()))

    @property
    def claim_domain(self) -> Domain:
        """A claim is a function: the mean of the field at any candidate, then its log-variances"""
        return FunctionDomain(self.candidates, BoxDomain((2 * COMPONENTS,)))

    def information(self, claim: Float[Array, "n_candidates claim_width"]) -> Float[Array, ""]:
        """(1/2) sum over candidates and the scored components of (prior_log_variance - claimed log-variance)"""
        scored = jnp.asarray(self.scored)
        return 0.5 * jnp.sum(self.prior_log_variance[scored] - claim[:, COMPONENTS:][:, scored])

    def reset(self, state: dict[str, Any]) -> dict[str, Float[Array, "..."]]:
        """The objective's memory before any time_step: the prior claimed, nothing learned"""
        truth = state["field"](self.candidate_states)  # (n_candidates, components)
        prior = jnp.concatenate([jnp.zeros_like(truth), jnp.broadcast_to(self.prior_log_variance, truth.shape)], axis=-1)
        return {"truth": truth, "information": jnp.zeros(()), "claim": prior}  # claim (n_candidates, 2 components)

    def score(
        self, objective_state: dict[str, Float[Array, "..."]], state: dict[str, Any], decision: Decision, next_state: dict[str, Any]
    ) -> tuple[dict[str, Float[Array, "..."]], Float[Array, ""]]:
        """The nats gained between the claim as it stood and the claim made"""
        claim = decision.claim(self.candidate_states)
        information = self.information(claim)
        return {"truth": objective_state["truth"], "information": information, "claim": claim}, information - objective_state["information"]


@dataclass(frozen=True)
class SearchMagnitude(Objective[dict[str, Float[Array, "..."]]]):
    """Reward is what standing on a candidate raised the largest current speed stood on so far, with the truth and the claim recorded beside it"""

    candidates: Subset[dict[str, Any]]
    """The states a waypoint may be aimed at and a claim is scored on, over the state domain a belief is maintained on"""

    candidate_states: dict[str, Any] = dataclasses.field(init=False, repr=False)
    """The live candidates, gathered once"""

    def __post_init__(self) -> None:
        """Gather the live candidates as concrete arrays"""
        live = jnp.flatnonzero(self.candidates.live)  # (n_candidates,) indices into the state domain
        object.__setattr__(self, "candidate_states", jax.tree.map(lambda a: a[live], self.candidates.elements()))

    @property
    def claim_domain(self) -> Domain:
        """A claim is a function: the mean of the field at any candidate, then its log-variances"""
        return FunctionDomain(self.candidates, BoxDomain((2 * COMPONENTS,)))

    def speed_at(self, state: dict[str, Any]) -> tuple[Float[Array, ""], Bool[Array, ""]]:
        """||W(position)|| where the boat stands, and whether that position is a candidate"""
        position = {"position": state["position"]}
        return jnp.linalg.norm(state["field"](position)[..., OCEAN_CURRENT]), self.candidates.live[self.candidates.index_of(position)]

    def reset(self, state: dict[str, Any]) -> dict[str, Float[Array, "..."]]:
        """The objective's memory before any time_step, nothing claimed yet"""
        truth = state["field"](self.candidate_states)  # (n_candidates, components)
        speed, on_candidate = self.speed_at(state)
        return {
            "truth": truth,
            "best_observed_magnitude": jnp.where(on_candidate, speed, 0.0),
            "best_possible_magnitude": jnp.max(jnp.linalg.norm(truth[:, OCEAN_CURRENT], axis=-1)),
            "claim": jnp.zeros((truth.shape[0], 2 * COMPONENTS)),  # (n_candidates, 2 components)
        }

    def score(
        self, objective_state: dict[str, Float[Array, "..."]], state: dict[str, Any], decision: Decision, next_state: dict[str, Any]
    ) -> tuple[dict[str, Float[Array, "..."]], Float[Array, ""]]:
        """What the position arrived in raised the best observed speed by, and the claim as it stood"""
        speed, on_candidate = self.speed_at(next_state)
        best = objective_state["best_observed_magnitude"]
        carried = objective_state | {
            "best_observed_magnitude": jnp.where(on_candidate, jnp.maximum(best, speed), best),
            "claim": decision.claim(self.candidate_states),
        }
        return carried, jnp.where(on_candidate, jnp.maximum(speed - best, 0.0), 0.0)


@dataclass(frozen=True)
class VisitTargets(Objective[dict[str, Float[Array, "..."]]]):
    """Reward is -1 every time_step until the boat has been within arrival_km of every candidate and 0 from then on, with the truth and the claim recorded beside it"""

    candidates: Subset[dict[str, Any]]
    """The targets, as the states a waypoint may be aimed at and a claim is scored on, over the state domain a belief is maintained on"""

    grid: PlaneGrid
    """What positions are measured on"""

    arrival_km: float
    """How near a target counts as visited"""

    candidate_states: dict[str, Any] = dataclasses.field(init=False, repr=False)
    """The targets, gathered once"""

    def __post_init__(self) -> None:
        """Gather the targets as concrete arrays"""
        live = jnp.flatnonzero(self.candidates.live)  # (n_targets,) indices into the state domain
        object.__setattr__(self, "candidate_states", jax.tree.map(lambda a: a[live], self.candidates.elements()))

    @property
    def claim_domain(self) -> Domain:
        """A claim is a function: the mean of the field at any target, then its log-variances"""
        return FunctionDomain(self.candidates, BoxDomain((2 * COMPONENTS,)))

    def near(self, state: dict[str, Any]) -> Float[Array, " n_targets"]:
        """1 at each target within arrival_km of `state`"""
        gap = self.grid.embed(state["position"]) - self.grid.embed(self.candidate_states["position"])  # (n_targets, 2) km
        return (jnp.linalg.norm(gap, axis=-1) <= self.arrival_km).astype(jnp.float32)

    def reset(self, state: dict[str, Any]) -> dict[str, Float[Array, "..."]]:
        """The objective's memory before any time_step, nothing claimed yet"""
        truth = state["field"](self.candidate_states)  # (n_targets, components)
        return {
            "targets": self.candidate_states["position"],  # (n_targets, 2) as (lat, lon)
            "truth": truth,
            "visited": self.near(state),
            "steps_to_completion": jnp.zeros(()),
            "claim": jnp.zeros((truth.shape[0], 2 * COMPONENTS)),  # (n_targets, 2 components)
        }

    def score(
        self, objective_state: dict[str, Float[Array, "..."]], state: dict[str, Any], decision: Decision, next_state: dict[str, Any]
    ) -> tuple[dict[str, Float[Array, "..."]], Float[Array, ""]]:
        """-1 unless every target is visited by the state arrived in, and the claim as it stood"""
        visited = jnp.maximum(objective_state["visited"], self.near(next_state))
        reward = jnp.min(visited) - 1.0
        return {
            "targets": objective_state["targets"],
            "truth": objective_state["truth"],
            "visited": visited,
            "steps_to_completion": objective_state["steps_to_completion"] - reward,
            "claim": decision.claim(self.candidate_states),
        }, reward
