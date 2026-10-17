"""The objective: what a move earns, and what is recorded beside it"""

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from zermelo.interface import BoxDomain, Decision, Domain, FunctionDomain, Objective, Subset


@dataclass(frozen=True)
class AmbientObjective(Objective[dict[str, Float[Array, "..."]]]):
    """Reward is the increment max(0, ||f*(z')|| - beta_n), with the truth and the claim recorded beside it"""

    candidates: Subset[dict[str, Any]]
    """Where a sample on_candidate, a waypoint may be aimed and a claim is scored"""

    scored: dict[str, Any] = dataclasses.field(init=False, repr=False)
    """The candidate positions, gathered once"""

    def __post_init__(self) -> None:
        """Gather the live candidates as concrete arrays"""
        keep = jnp.flatnonzero(self.candidates.live)  # (n_scored,) indices into the position domain
        object.__setattr__(self, "scored", jax.tree.map(lambda a: a[keep], self.candidates.elements()))

    @property
    def ambient_axes(self) -> int:
        """Axes of the displaced part"""
        return int(self.scored["ambient"].shape[-1])

    @property
    def claim_domain(self) -> Domain:
        """A claim is a function: a mean and a log-variance per displaced axis, at any candidate"""
        return FunctionDomain(self.candidates, BoxDomain((2 * self.ambient_axes,)))

    def _magnitude(self, state: dict[str, Any]) -> tuple[Float[Array, ""], Bool[Array, ""]]:
        """`||f*(z)||` where the actor stands, and whether that position is a candidate"""
        z = {"ambient": state["ambient"], "controllable": state["controllable"]}
        return jnp.linalg.norm(state["field"](z)), self.candidates.live[self.candidates.index_of(z)]

    def reset(self, state: dict[str, Any]) -> dict[str, Float[Array, "..."]]:
        """The objective's memory at step zero"""
        truth = state["field"](self.scored)  # (n_scored, ambient_axes)
        magnitude, on_candidate = self._magnitude(state)
        return {
            "best_observed_magnitude": jnp.where(on_candidate, magnitude, 0.0),  # the start's own norm where it is a candidate, else 0
            "best_possible_magnitude": jnp.max(jnp.linalg.norm(truth, axis=-1)),
            "truth": truth,
            "claim": jnp.zeros((truth.shape[0], 2 * self.ambient_axes)),  # (n_scored, 2 * ambient_axes): nothing claimed yet
        }

    def score(
        self, objective_state: dict[str, Float[Array, "..."]], state: dict[str, Any], decision: Decision, next_state: dict[str, Any]
    ) -> tuple[dict[str, Float[Array, "..."]], Float[Array, ""]]:
        """What the arrival improved on the best_observed_magnitude at a candidate, and the claim as it stood"""
        magnitude, on_candidate = self._magnitude(next_state)
        best_observed_magnitude = objective_state["best_observed_magnitude"]
        carried = objective_state | {
            "best_observed_magnitude": jnp.where(on_candidate, jnp.maximum(best_observed_magnitude, magnitude), best_observed_magnitude),
            "claim": decision.claim(self.scored),  # (n_scored, 2 * ambient_axes): the means, then the log-variances
        }
        return carried, jnp.where(on_candidate, jnp.maximum(magnitude - best_observed_magnitude, 0.0), 0.0)
