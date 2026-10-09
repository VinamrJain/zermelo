"""What a boat is scored on: standing in the fastest current the candidates offer

W(position) = (u, v)      the current, m/s
candidates                the grid points the boat is scored on
at every time_step, from the state arrived in:
    truth[c]            = ||W(c)|| at candidate c                 (n_candidates,)
    best_possible_speed = max over c of truth[c]
    speed               = ||W(state)|| on a candidate, 0 otherwise
    reward              = speed
claim                     (u, v) predicted at every candidate, as means then log-variances
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from zermelo.interface import BoxDomain, Decision, Domain, FunctionDomain, Objective, Subset


@dataclass(frozen=True)
class CurrentSearch(Objective[dict[str, Float[Array, "..."]]]):
    """Reward is the speed of the current the boat stands in, counted on a candidate, with the best on offer and the claim recorded beside it"""

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
        """truth, best_possible_speed and speed at `state`"""
        truth = jnp.linalg.norm(state["field"](self.candidate_states), axis=-1)  # (n_candidates,)
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
