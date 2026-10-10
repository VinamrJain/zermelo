"""What a boat is scored on: reaching a target, or learning the ocean current over a region

W(position) = (u, v)      the ocean current, m/s
target                    the grid point to reach, drawn once per episode
candidates                the grid points a waypoint may be aimed at and a claim is scored on
at every time_step, from the state arrived in:
    arrived               = arrived before, or ||position - target|| <= arrival_km on the tangent plane
    reward                = 0 where arrived, else -1
    steps_to_arrival      = time_steps rewarded -1 so far
    truth[c]              = W(c) at candidate c                                    (n_candidates, 2)
claim                     (u, v) predicted at every candidate, as means then log-variances
or, learning the region:
    information           = (1/2) sum over candidates and components of (prior_log_variance - claimed log-variance), nats
    reward                = information gained this time_step
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from zermelo.interface import BoxDomain, Decision, Domain, FunctionDomain, Objective, Subset
from zermelo.problems.ocean.grid import PlaneGrid


@dataclass(frozen=True)
class ReachTarget(Objective[dict[str, Float[Array, "..."]]]):
    """Reward is -1 every time_step until the boat is within arrival_km of the target and 0 from then on, with the truth and the claim recorded beside it"""

    candidates: Subset[dict[str, Any]]
    """The states a waypoint may be aimed at and a claim is scored on, over the state domain a belief is maintained on"""

    grid: PlaneGrid
    """What positions are measured on"""

    arrival_km: float
    """How near the target counts as arrived"""

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

    def at_target(self, state: dict[str, Any]) -> Bool[Array, ""]:
        """Whether `state` is within arrival_km of its target"""
        return jnp.linalg.norm(self.grid.embed(state["position"]) - self.grid.embed(state["target"])) <= self.arrival_km

    def reset(self, state: dict[str, Any]) -> dict[str, Float[Array, "..."]]:
        """The objective's memory before any time_step, nothing claimed yet"""
        truth = state["field"](self.candidate_states)  # (n_candidates, 2)
        return {
            "truth": truth,
            "arrived": self.at_target(state).astype(jnp.float32),
            "steps_to_arrival": jnp.zeros(()),
            "claim": jnp.zeros((truth.shape[0], 4)),  # (n_candidates, 4)
        }

    def score(
        self, objective_state: dict[str, Float[Array, "..."]], state: dict[str, Any], decision: Decision, next_state: dict[str, Any]
    ) -> tuple[dict[str, Float[Array, "..."]], Float[Array, ""]]:
        """-1 unless arrived by the state arrived in, and the claim as it stood"""
        arrived = jnp.maximum(objective_state["arrived"], self.at_target(next_state).astype(jnp.float32))
        reward = arrived - 1.0
        return {
            "truth": objective_state["truth"],
            "arrived": arrived,
            "steps_to_arrival": objective_state["steps_to_arrival"] - reward,
            "claim": decision.claim(self.candidate_states),
        }, reward


@dataclass(frozen=True)
class LearnRegion(Objective[dict[str, Float[Array, "..."]]]):
    """Reward is the nats the claim's marginals shed over the candidates this time_step, with the truth recorded beside it"""

    candidates: Subset[dict[str, Any]]
    """The states a waypoint may be aimed at and a claim is scored on, over the state domain a belief is maintained on"""

    prior_log_variance: float
    """log Var[f(c)] per component before any reading, the level information is counted from"""

    candidate_states: dict[str, Any] = dataclasses.field(init=False, repr=False)
    """The live candidates, gathered once"""

    def __post_init__(self) -> None:
        """Gather the live candidates as concrete arrays"""
        live = jnp.flatnonzero(self.candidates.live)  # (n_candidates,) indices into the state domain
        object.__setattr__(self, "candidate_states", jax.tree.map(lambda a: a[live], self.candidates.elements()))

    @property
    def claim_domain(self) -> Domain:
        """A claim is a function: the mean of (u, v) at any candidate, then their log-variances"""
        return FunctionDomain(self.candidates, BoxDomain((4,)))

    def information(self, claim: Float[Array, "n_candidates 4"]) -> Float[Array, ""]:
        """(1/2) sum over candidates and components of (prior_log_variance - claimed log-variance)"""
        return 0.5 * jnp.sum(self.prior_log_variance - claim[:, 2:4])

    def reset(self, state: dict[str, Any]) -> dict[str, Float[Array, "..."]]:
        """The objective's memory before any time_step: the prior claimed, nothing learned"""
        truth = state["field"](self.candidate_states)  # (n_candidates, 2)
        prior = jnp.concatenate([jnp.zeros((truth.shape[0], 2)), jnp.full((truth.shape[0], 2), self.prior_log_variance)], axis=-1)
        return {"truth": truth, "information": jnp.zeros(()), "claim": prior}  # claim (n_candidates, 4)

    def score(
        self, objective_state: dict[str, Float[Array, "..."]], state: dict[str, Any], decision: Decision, next_state: dict[str, Any]
    ) -> tuple[dict[str, Float[Array, "..."]], Float[Array, ""]]:
        """The nats gained between the claim as it stood and the claim made"""
        claim = decision.claim(self.candidate_states)
        information = self.information(claim)
        return {"truth": objective_state["truth"], "information": information, "claim": claim}, information - objective_state["information"]
