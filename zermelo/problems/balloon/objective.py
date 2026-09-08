"""What a balloon is scored on: predicting where a chosen reading of the wind is largest

W(lat, lon, p) = (u, v)     the wind, metres per second, u eastward and v northward
g(W; lat, lon)              the scalar g reads off W at one cell
claim                       g predicted at every candidate cell, as a mean and a log-variance
"""

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from zermelo.interface import BoxDomain, Decision, Domain, FunctionDomain, Objective, Subset
from zermelo.problems.balloon.field import WindField
from zermelo.problems.balloon.grid import SphereGrid


class Target(ABC):
    """The scalar g an episode is scored on predicting"""

    @abstractmethod
    def __call__(self, field: WindField, grid: SphereGrid) -> Float[Array, " pos"]:
        """g at every cell of `grid`"""


@dataclass(frozen=True)
class SpeedAt(Target):
    """g(W; lat, lon) = ||W(lat, lon, p_ref)||"""

    level: int
    """The p every cell is read at"""

    def __call__(self, field: WindField, grid: SphereGrid) -> Float[Array, " pos"]:
        """Wind speed at `level`, at every cell"""
        cells = grid.elements()  # (pos, 2)
        wind = field({"position": cells, "altitude": jnp.full(cells.shape[0], self.level)})
        return jnp.linalg.norm(wind, axis=-1)


@dataclass(frozen=True)
class ColumnPeakSpeed(Target):
    """g(W; lat, lon) = max over p of ||W(lat, lon, p)||"""

    n_alt: int
    """How many levels the column spans"""

    def __call__(self, field: WindField, grid: SphereGrid) -> Float[Array, " pos"]:
        """The fastest wind anywhere in each cell's column"""
        cells = grid.elements()  # (pos, 2)
        per_level = jnp.stack(
            [jnp.linalg.norm(field({"position": cells, "altitude": jnp.full(cells.shape[0], p)}), axis=-1) for p in range(self.n_alt)]
        )  # (alt, pos)
        return jnp.max(per_level, axis=0)


@dataclass(frozen=True)
class StormSearch(Objective[dict[str, Float[Array, "..."]]]):
    """Reward is the increment in the largest g the balloon has flown through, with the claim recorded beside it"""

    grid: SphereGrid
    target: Target

    candidates: Subset[Float[Array, " 2"]]
    """The cells g is predicted at, and the cells its peak is taken over"""

    scored: Float[Array, "n_scored 2"] = dataclasses.field(init=False, repr=False)
    """Those cells as degrees, gathered once"""

    keep: Int[Array, " n_scored"] = dataclasses.field(init=False, repr=False)
    """Their indices into the grid"""

    def __post_init__(self) -> None:
        """Gather the live candidates as concrete arrays"""
        keep = jnp.flatnonzero(self.candidates.live)  # (n_scored,) indices into the grid
        object.__setattr__(self, "keep", keep)
        object.__setattr__(self, "scored", self.candidates.elements()[keep])

    @property
    def claim_domain(self) -> Domain:
        """A claim is a function: a mean and a log-variance of g at any candidate"""
        return FunctionDomain(self.candidates, BoxDomain((2,)))

    def truth(self, state: dict[str, Any]) -> Float[Array, " n_scored"]:
        """g at every candidate"""
        return self.target(state["field"], self.grid)[self.keep]

    def flown(self, state: dict[str, Any]) -> Float[Array, ""]:
        """||W|| where the balloon stands"""
        return jnp.linalg.norm(state["field"](state))

    def reset(self, state: dict[str, Any]) -> dict[str, Float[Array, "..."]]:
        """The objective's memory at step zero"""
        truth = self.truth(state)  # (n_scored,)
        return {
            "incumbent": self.flown(state),
            "oracle": jnp.max(truth),
            "truth": truth,
            "claim": jnp.zeros((truth.shape[0], 2)),  # (n_scored, 2): the mean of g, then its log-variance
        }

    def score(
        self, objective_state: dict[str, Float[Array, "..."]], state: dict[str, Any], decision: Decision, next_state: dict[str, Any]
    ) -> tuple[dict[str, Float[Array, "..."]], Float[Array, ""]]:
        """What arriving improved on the largest wind flown through, and the claim as it stood"""
        magnitude, incumbent = self.flown(next_state), objective_state["incumbent"]
        carried = objective_state | {
            "incumbent": jnp.maximum(incumbent, magnitude),
            "claim": decision.claim(self.scored),  # (n_scored, 2)
        }
        return carried, jnp.maximum(magnitude - incumbent, 0.0)


def balloon_candidates(grid: SphereGrid, margin_lat: int, margin_lon: int) -> Subset[Float[Array, " 2"]]:
    """The grid without a band `margin_lat` rows and `margin_lon` columns deep on every side"""
    cells = grid.cell_of(grid.elements())  # (pos, 2)
    inside = (
        (cells[:, 0] >= margin_lat)
        & (cells[:, 0] < grid.n_lat - margin_lat)
        & (cells[:, 1] >= margin_lon)
        & (cells[:, 1] < grid.n_lon - margin_lon)
    )
    return grid.narrow(inside)
