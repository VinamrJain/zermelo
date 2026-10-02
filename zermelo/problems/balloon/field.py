"""The wind a balloon is carried by: how it is read, and the law an episode's own is drawn by

W(position, altitude, hours_elapsed) = (u, v)     u eastward, v northward, both m/s
position            (lat, lon) degrees of a grid point
altitude            an index into the altitudes the data holds
hours_elapsed       hours since the episode began
W = F + forecast_error                            F the forecast the balloon is given
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
    """W over the whole grid, as a function: called at one state it gives the (u, v) there, called at a batch of states it gives one (u, v) per state"""

    @abstractmethod
    def __call__(self, state: dict[str, Any]) -> Float[Array, "*batch uv"]:
        """(u, v) at each state given"""


@register_dataclass
@dataclass(frozen=True)
class TimeVaryingWind(WindField):
    """W(position, altitude, hours_elapsed), read at the nearest grid point and the nearest hour the frames stand for"""

    wind_on_grid: Float[Array, "frames alt pos uv"]
    """(u, v) at every frame, altitude and grid point"""

    frame_hours_elapsed: Float[Array, " frames"]
    """The hours_elapsed each frame stands for; one frame is a wind that never changes"""

    grid: SphereGrid = dataclasses.field(metadata=dict(static=True))
    """What `pos` is indexed by"""

    def __call__(self, state: dict[str, Any]) -> Float[Array, "*batch uv"]:
        """wind_on_grid[argmin over frames of |frame_hours_elapsed - hours_elapsed|, altitude, flat index of position]"""
        gap = jnp.abs(jnp.asarray(state["hours_elapsed"])[..., None] - self.frame_hours_elapsed)  # (*batch, frames)
        frame = jnp.argmin(gap, axis=-1)  # (*batch,)
        return self.wind_on_grid[frame, state["altitude"], self.grid.flat_index(state["position"])]


class WindError(ABC):
    """Where forecast_error comes from, for a truth W = F + forecast_error"""

    @abstractmethod
    def sample(self, key: PRNGKeyArray, grid: SphereGrid, n_altitudes: int) -> Float[Array, "*frames alt pos uv"]:
        """One draw of forecast_error at every altitude and grid point, per frame or the same at every frame"""


@dataclass(frozen=True)
class GaussianError(WindError):
    """forecast_error ~ N(0, K) over grid points, independently per altitude and per component of (u, v), the same at every hour, with

    K(a, b) = amplitude_ms^2 * exp(-||embed(a) - embed(b)||^2 / (2 * length_scale_km^2))
    """

    length_scale_km: float
    amplitude_ms: float

    jitter: float
    """Added to K's diagonal relative to amplitude_ms^2, so the factorisation succeeds"""

    def sample(self, key: PRNGKeyArray, grid: SphereGrid, n_altitudes: int) -> Float[Array, "alt pos uv"]:
        """L @ white, for L the Cholesky factor of K. Time O(pos^3), memory O(pos^2)"""
        points = grid.embed(grid.elements())  # (pos, 3)
        gap = jnp.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)  # (pos, pos)
        correlation = jnp.exp(-gap / (2.0 * self.length_scale_km**2)) + self.jitter * jnp.eye(grid.size())
        factor = self.amplitude_ms * jnp.linalg.cholesky(correlation)
        if jnp.isnan(factor).any():  # a singular K returns NaN rather than raising
            raise ValueError(f"K over {grid.size()} positions at {self.length_scale_km} km does not factorise at jitter {self.jitter}")
        white = jax.random.normal(key, (grid.size(), n_altitudes, 2))
        return jnp.moveaxis(jnp.tensordot(factor, white, axes=(1, 0)), 0, 1)  # (alt, pos, uv)


@register_dataclass
@dataclass(frozen=True)
class GEFSError(WindError):
    """forecast_error = W - F as the data holds it, the same at every draw"""

    error: Float[Array, "frames alt pos uv"]
    """W - F at every frame, altitude and grid point"""

    def sample(self, key: PRNGKeyArray, grid: SphereGrid, n_altitudes: int) -> Float[Array, "frames alt pos uv"]:
        """`error`, whatever the key"""
        return self.error


@dataclass(frozen=True)
class ForecastPrior(Prior[WindField]):
    """W = F + forecast_error, F fixed and forecast_error taken once per episode"""

    forecast: Float[Array, "frames alt pos uv"]
    """F at every frame, altitude and grid point"""

    error: WindError
    """Where forecast_error comes from"""

    grid: SphereGrid
    """What `pos` is indexed by"""

    frame_hours_elapsed: Float[Array, " frames"]
    """The hours_elapsed each frame stands for"""

    def sample(self, domain: Domain[WindField], key: PRNGKeyArray) -> WindField:
        """One episode's W"""
        if not isinstance(domain, FunctionDomain):
            raise TypeError(f"a {type(domain).__name__} holds no fields, so none can be drawn from it")
        error = self.error.sample(key, self.grid, self.forecast.shape[1])  # (alt, pos, uv) or (frames, alt, pos, uv)
        return TimeVaryingWind(
            self.forecast + (error if error.ndim == self.forecast.ndim else error[None]), self.frame_hours_elapsed, self.grid
        )
