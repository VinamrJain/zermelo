"""What a balloon is scored on: naming the cell where a chosen reading of the wind is largest

W(lat, lon, p) = (u, v)     the wind, metres per second, u eastward and v northward
T(lat, lon)                 the scalar T reads off W at one cell
claim                       the cell the agent names as T's largest
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from zermelo.interface import Decision, Domain, Objective, Subset
from zermelo.problems.balloon.field import WindField
from zermelo.problems.balloon.grid import SphereGrid


class Target(ABC):
    """The scalar an episode is scored on finding the largest value of"""

    @abstractmethod
    def __call__(self, field: WindField, grid: SphereGrid, n_alt: int) -> Float[Array, " pos"]:
        """T at every cell"""


@dataclass(frozen=True)
class SpeedAt(Target):
    """T(lat, lon) = |W(lat, lon, p_ref)|"""

    level: int
    """The p every cell is read at"""

    def __call__(self, field: WindField, grid: SphereGrid, n_alt: int) -> Float[Array, " pos"]:
        """Wind speed at `level`, at every cell"""
        cells = grid.elements()
        wind = field({"position": cells, "altitude": jnp.full(cells.shape[0], self.level)})
        return jnp.linalg.norm(wind, axis=-1)


@dataclass(frozen=True)
class ColumnPeakSpeed(Target):
    """T(lat, lon) = max over p of |W(lat, lon, p)|"""

    def __call__(self, field: WindField, grid: SphereGrid, n_alt: int) -> Float[Array, " pos"]:
        """The fastest wind anywhere in each cell's column"""
        cells = grid.elements()
        per_level = jnp.stack(
            [jnp.linalg.norm(field({"position": cells, "altitude": jnp.full(cells.shape[0], p)}), axis=-1) for p in range(n_alt)]
        )
        return jnp.max(per_level, axis=0)


@dataclass(frozen=True)
class StormSearch(Objective[Float[Array, ""]]):
    """Each step scores how much of T's peak the claimed cell holds, less what a step costs"""

    grid: SphereGrid
    n_alt: int
    target: Target

    candidates: Subset[Float[Array, " 2"]]
    """The cells a claim may name, and the cells T's peak is taken over"""

    step_cost: float
    """Subtracted every step"""

    @property
    def claim_domain(self) -> Domain:
        """The cell the agent names"""
        return self.candidates

    def field_target(self, state: dict[str, Any]) -> Float[Array, " pos"]:
        """T at every cell, minus infinity outside the candidates"""
        return jnp.where(self.candidates.live, self.target(state["field"], self.grid, self.n_alt), -jnp.inf)

    def reset(self, state: dict[str, Any]) -> Float[Array, ""]:
        """The largest T anywhere a claim may name"""
        return jnp.max(self.field_target(state))

    def score(
        self, objective_state: Float[Array, ""], state: dict[str, Any], decision: Decision, next_state: dict[str, Any]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """T(claim) / max T, less step_cost"""
        at_claim = self.field_target(state)[self.grid.flat_index(decision.claim)]
        return objective_state, at_claim / objective_state - self.step_cost

    def terminated(self, objective_state: Float[Array, ""], state: dict[str, Any]) -> Bool[Array, ""]:
        """Never: every episode runs its full horizon"""
        return jnp.bool_(False)


def balloon_candidates(grid: SphereGrid, margin_lat: int, margin_lon: int) -> Subset[Float[Array, " 2"]]:
    """The grid without a band `margin_lat` rows and `margin_lon` columns deep on every side"""
    cells = grid.cell_of(grid.elements())
    inside = (
        (cells[:, 0] >= margin_lat)
        & (cells[:, 0] < grid.n_lat - margin_lat)
        & (cells[:, 1] >= margin_lon)
        & (cells[:, 1] < grid.n_lon - margin_lon)
    )
    return grid.narrow(inside)
