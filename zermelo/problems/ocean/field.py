"""The field a boat sails in, the ocean current carrying it and the wind driving it: how it is read, and the law an episode's own is drawn by

field(position) = (W_u, W_v, U_u, U_v)    W the ocean current and U the wind, u eastward, v northward, m/s, at the grid point `position` names
field = forecast + forecast_error         forecast the frame the boat is given
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

OCEAN_CURRENT = slice(0, 2)
"""The channels of the field holding W"""

WIND = slice(2, 4)
"""The channels of the field holding U"""

CHANNELS = 4
"""How wide the field is: W then U"""


@register_dataclass
@dataclass(frozen=True)
class GridField:
    """The field over the whole grid, read at the grid point nearest a state's position"""

    field_on_grid: Float[Array, "pos channels"]
    """(W_u, W_v, U_u, U_v) at every grid point"""

    grid: PlaneGrid = dataclasses.field(metadata=dict(static=True))
    """What `pos` is indexed by"""

    def __call__(self, state: dict[str, Any]) -> Float[Array, "*batch channels"]:
        """field_on_grid[flat index of position]"""
        return self.field_on_grid[self.grid.flat_index(state["position"])]


class ForecastError(ABC):
    """Where forecast_error comes from, for a truth field = forecast + forecast_error"""

    @abstractmethod
    def sample(self, key: PRNGKeyArray, grid: PlaneGrid) -> Float[Array, "pos channels"]:
        """One draw of forecast_error at every grid point"""


@dataclass(frozen=True)
class GaussianError(ForecastError):
    """forecast_error ~ N(0, K) over grid points, independently per channel, with

    K(a, b) = amplitude_ms^2 * exp(-||embed(a) - embed(b)||^2 / (2 * length_scale_km^2))
    """

    length_scale_km: float
    amplitude_ms: float

    jitter: float
    """Added to K's diagonal relative to amplitude_ms^2, so the factorisation succeeds"""

    def sample(self, key: PRNGKeyArray, grid: PlaneGrid) -> Float[Array, "pos channels"]:
        """L @ white, for L the Cholesky factor of K. Time O(pos^3), memory O(pos^2)"""
        points = grid.embed(grid.elements())  # (pos, 2)
        gap = jnp.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)  # (pos, pos)
        correlation = jnp.exp(-gap / (2.0 * self.length_scale_km**2)) + self.jitter * jnp.eye(grid.size())
        factor = self.amplitude_ms * jnp.linalg.cholesky(correlation)
        if jnp.isnan(factor).any():  # a singular K returns NaN rather than raising
            raise ValueError(f"K over {grid.size()} positions at {self.length_scale_km} km does not factorise at jitter {self.jitter}")
        return factor @ jax.random.normal(key, (grid.size(), CHANNELS))


@register_dataclass
@dataclass(frozen=True)
class FixedError(ForecastError):
    """forecast_error = field - forecast as the data holds it, the same at every draw"""

    error: Float[Array, "pos channels"]
    """field - forecast at every grid point"""

    def sample(self, key: PRNGKeyArray, grid: PlaneGrid) -> Float[Array, "pos channels"]:
        """`error`, whatever the key"""
        return self.error


@dataclass(frozen=True)
class ForecastPrior(Prior[GridField]):
    """field = forecast + forecast_error, forecast fixed and forecast_error taken once per episode"""

    forecast: Float[Array, "pos channels"]
    """The forecast at every grid point"""

    error: ForecastError
    """Where forecast_error comes from"""

    grid: PlaneGrid
    """What `pos` is indexed by"""

    def sample(self, domain: Domain[GridField], key: PRNGKeyArray) -> GridField:
        """One episode's field"""
        if not isinstance(domain, FunctionDomain):
            raise TypeError(f"a {type(domain).__name__} holds no fields, so none can be drawn from it")
        return GridField(self.forecast + self.error.sample(key, self.grid), self.grid)
