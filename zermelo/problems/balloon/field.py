"""The wind a balloon is carried by: how it is read, and the law an episode's own is drawn by

W(lat, lon, p) = (u, v)     u: eastward, v: northward, both metres per second
lat, lon                    degrees, on the grid's own cells
p                           which pressure level, an index into the ladder
t                           hours since the episode began
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
from zermelo.problems.balloon.grid import SphereGrid


class WindField(ABC):
    """A read of W at one state"""

    @abstractmethod
    def __call__(self, state: dict[str, Any]) -> Float[Array, "*batch uv"]:
        """(u, v) where the state sits"""


@register_dataclass
@dataclass(frozen=True)
class GriddedWind(WindField):
    """W(lat, lon, p), held per level and cell, read at the nearest cell"""

    wind: Float[Array, "alt pos uv"]
    """(u, v) at every level and cell, in the grid's index order"""

    grid: SphereGrid = dataclasses.field(metadata=dict(static=True))
    """What `pos` is indexed by"""

    def __call__(self, state: dict[str, Any]) -> Float[Array, "*batch uv"]:
        """wind[p, cell]"""
        return self.wind[state["altitude"], self.grid.flat_index(state["position"])]


@register_dataclass
@dataclass(frozen=True)
class DriftingWind(WindField):
    """W(lat, lon, p, t), read at the nearest cell and the nearest recorded hour"""

    wind: Float[Array, "t alt pos uv"]
    """(u, v) at every recorded hour, level and cell"""

    hours: Float[Array, " t"]
    """The t each record stands for"""

    grid: SphereGrid = dataclasses.field(metadata=dict(static=True))
    """What `pos` is indexed by"""

    def __call__(self, state: dict[str, Any]) -> Float[Array, "*batch uv"]:
        """wind[argmin |hours - t|, p, cell]"""
        return self.wind[jnp.argmin(jnp.abs(self.hours - state["hours"])), state["altitude"], self.grid.flat_index(state["position"])]


class WindError(ABC):
    """The law e is drawn by, for a truth W = forecast + e"""

    @abstractmethod
    def sample(self, key: PRNGKeyArray, grid: SphereGrid, n_levels: int) -> Float[Array, "alt pos uv"]:
        """One draw of e at every level and cell"""


@dataclass(frozen=True)
class GaussianError(WindError):
    """e ~ N(0, K), independently per level and per component of (u, v), with

    K(a, b) = amplitude_ms^2 * exp(-||embed(a) - embed(b)||^2 / (2 * length_scale_km^2))
    """

    length_scale_km: float
    amplitude_ms: float

    jitter: float
    """Added to K's diagonal relative to amplitude_ms^2, so the factorisation succeeds"""

    def sample(self, key: PRNGKeyArray, grid: SphereGrid, n_levels: int) -> Float[Array, "alt pos uv"]:
        """L @ white, for L the Cholesky factor of K. Time O(pos^3), memory O(pos^2)"""
        points = grid.embed(grid.elements())  # (pos, 3)
        gap = jnp.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)  # (pos, pos)
        correlation = jnp.exp(-gap / (2.0 * self.length_scale_km**2)) + self.jitter * jnp.eye(grid.size())
        factor = self.amplitude_ms * jnp.linalg.cholesky(correlation)
        if jnp.isnan(factor).any():  # a singular K returns NaN rather than raising
            raise ValueError(f"K over {grid.size()} cells at {self.length_scale_km} km does not factorise at jitter {self.jitter}")
        white = jax.random.normal(key, (grid.size(), n_levels, 2))
        return jnp.moveaxis(jnp.tensordot(factor, white, axes=(1, 0)), 0, 1)  # (alt, pos, uv)


@dataclass(frozen=True)
class ForecastPrior(Prior[WindField]):
    """W = forecast + e, the forecast fixed and e drawn once per episode"""

    forecast: Float[Array, "alt pos uv"]
    """(u, v) predicted at every level and cell, and what the agent is told"""

    error: WindError
    """The law e is drawn by"""

    grid: SphereGrid
    """What `pos` is indexed by"""

    def sample(self, domain: Domain[WindField], key: PRNGKeyArray) -> WindField:
        """One episode's W"""
        if not isinstance(domain, FunctionDomain):
            raise TypeError(f"a {type(domain).__name__} holds no fields, so none can be drawn from it")
        return GriddedWind(self.forecast + self.error.sample(key, self.grid, self.forecast.shape[0]), self.grid)
