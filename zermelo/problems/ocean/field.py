"""The ocean current a boat is carried by: how it is read, and the law an episode's own is drawn by

W(position) = (u, v)      u eastward, v northward, both m/s, at the grid point `position` names
W = F + forecast_error     F the forecast the boat is given
"""

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, PRNGKeyArray

from zermelo.interface import Domain, FunctionDomain, Prior
from zermelo.problems.ocean.grid import PlaneGrid


@register_dataclass
@dataclass(frozen=True)
class OceanCurrent:
    """W over the whole grid, read at the grid point nearest a state's position"""

    ocean_current_on_grid: Float[Array, "pos uv"]
    """(u, v) at every grid point"""

    grid: PlaneGrid = dataclasses.field(metadata=dict(static=True))
    """What `pos` is indexed by"""

    def __call__(self, state: dict[str, Any]) -> Float[Array, "*batch uv"]:
        """ocean_current_on_grid[flat index of position]"""
        return self.ocean_current_on_grid[self.grid.flat_index(state["position"])]


class OceanCurrentError(ABC):
    """Where forecast_error comes from, for a truth W = F + forecast_error"""

    @abstractmethod
    def sample(self, key: PRNGKeyArray, grid: PlaneGrid) -> Float[Array, "pos uv"]:
        """One draw of forecast_error at every grid point"""


@dataclass(frozen=True)
class GaussianError(OceanCurrentError):
    """forecast_error ~ N(0, K) over grid points, independently per component of (u, v), with

    K(a, b) = amplitude_ms^2 * exp(-||embed(a) - embed(b)||^2 / (2 * length_scale_km^2))
    """

    length_scale_km: float
    amplitude_ms: float

    jitter: float
    """Added to K's diagonal relative to amplitude_ms^2, so the factorisation succeeds"""

    def sample(self, key: PRNGKeyArray, grid: PlaneGrid) -> Float[Array, "pos uv"]:
        """L @ white, for L the Cholesky factor of K. Time O(pos^3), memory O(pos^2)"""
        points = grid.embed(grid.elements())  # (pos, 2)
        gap = jnp.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)  # (pos, pos)
        correlation = jnp.exp(-gap / (2.0 * self.length_scale_km**2)) + self.jitter * jnp.eye(grid.size())
        factor = self.amplitude_ms * jnp.linalg.cholesky(correlation)
        if jnp.isnan(factor).any():  # a singular K returns NaN rather than raising
            raise ValueError(f"K over {grid.size()} positions at {self.length_scale_km} km does not factorise at jitter {self.jitter}")
        return factor @ jax.random.normal(key, (grid.size(), 2))


@register_dataclass
@dataclass(frozen=True)
class FixedError(OceanCurrentError):
    """forecast_error = W - F as the data holds it, the same at every draw"""

    error: Float[Array, "pos uv"]
    """W - F at every grid point"""

    def sample(self, key: PRNGKeyArray, grid: PlaneGrid) -> Float[Array, "pos uv"]:
        """`error`, whatever the key"""
        return self.error


@dataclass(frozen=True)
class ForecastPrior(Prior[OceanCurrent]):
    """W = F + forecast_error, F fixed and forecast_error taken once per episode"""

    forecast: Float[Array, "pos uv"]
    """F at every grid point"""

    error: OceanCurrentError
    """Where forecast_error comes from"""

    grid: PlaneGrid
    """What `pos` is indexed by"""

    def sample(self, domain: Domain[OceanCurrent], key: PRNGKeyArray) -> OceanCurrent:
        """One episode's W"""
        if not isinstance(domain, FunctionDomain):
            raise TypeError(f"a {type(domain).__name__} holds no fields, so none can be drawn from it")
        return OceanCurrent(self.forecast + self.error.sample(key, self.grid), self.grid)
