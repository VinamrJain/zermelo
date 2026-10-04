"""A balloon problem, assembled from a wind field and the forecast of it"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from zermelo.interface import (
    BoxDomain,
    DiscreteDomain,
    Domain,
    Enumerable,
    FunctionDomain,
    Prior,
    ProductDomain,
    ProductPrior,
    Uniform,
    World,
)
from zermelo.problems.balloon.field import ForecastPrior, TimeVaryingWind, WindError
from zermelo.problems.balloon.grid import SphereGrid
from zermelo.problems.balloon.objective import StormSearch, Target, balloon_candidates
from zermelo.problems.balloon.readout import PointWind
from zermelo.problems.balloon.transition import Advection, Ascent, BalloonTransition, Clock


class Highest(Prior[Any]):
    """The largest element of a finite domain, drawn with probability one"""

    def sample(self, domain: Domain[Any], key: PRNGKeyArray) -> Any:
        """The element at `size() - 1`"""
        if not isinstance(domain, Enumerable):
            raise TypeError(f"a {type(domain).__name__} has no largest element, so none can be drawn from it")
        return domain.from_index(jnp.asarray(domain.size() - 1, jnp.int32))


class Zero(Prior[Float[Array, ""]]):
    """The scalar zero, drawn with probability one"""

    def sample(self, domain: Domain[Float[Array, ""]], key: PRNGKeyArray) -> Float[Array, ""]:
        """0.0"""
        return jnp.zeros(())


@dataclass(frozen=True)
class WindData:
    """The wind data: W the truth and F the forecast

    hour                    hours since the data's own start, one per frame
    last_hour_in_dataset    the hour of the last frame
    """

    wind: Float[Array, "frames alt pos uv"]
    forecast: Float[Array, "frames alt pos uv"]
    hour: Float[Array, " frames"]
    altitude_km: Float[Array, " alt"]
    grid: SphereGrid

    @property
    def n_alt(self) -> int:
        """How many altitudes the data holds"""
        return self.wind.shape[1]

    def frames_from(
        self, start_hour: float, episode_hours: float, time_varying: bool
    ) -> tuple[Float[Array, "frames alt pos uv"], Float[Array, "frames alt pos uv"], Float[Array, " frames"]]:
        """(W, F, frame_hours_elapsed) over the frames an episode starting at `start_hour` reads:

            time_varying    every frame with start_hour <= hour <= start_hour + episode_hours
            otherwise       the one frame at start_hour

        frame_hours_elapsed = hour - start_hour.
        """
        last_hour_in_dataset = float(self.hour[-1])
        if not bool(jnp.any(self.hour == start_hour)):
            raise ValueError(
                f"the data holds no frame at hour {start_hour}; it runs from hour {float(self.hour[0])} to {last_hour_in_dataset}"
            )
        end_hour = start_hour + episode_hours if time_varying else start_hour
        if end_hour > last_hour_in_dataset:
            raise ValueError(
                f"an episode from hour {start_hour} runs to hour {end_hour}, and last_hour_in_dataset is {last_hour_in_dataset}"
            )
        frames = jnp.flatnonzero((self.hour >= start_hour) & (self.hour <= end_hour))
        return self.wind[frames], self.forecast[frames], self.hour[frames] - start_hour


def load_wind(path: Path, grid_stride: int, hour_stride: int) -> WindData:
    """The wind data stored at `path`, keeping every `grid_stride`-th row and column and every `hour_stride`-th frame"""
    raw = np.load(path)
    lat, lon = raw["latitude"][::grid_stride], raw["longitude"][::grid_stride]
    grid = SphereGrid(
        n_lat=len(lat),
        n_lon=len(lon),
        lat_first=float(lat[0]),
        lon_first=float(lon[0]),
        lat_step=float(lat[1] - lat[0]),
        lon_step=float(lon[1] - lon[0]),
    )

    def laid_out(u: str, v: str) -> Float[Array, "frames alt pos uv"]:
        """One wind as (frames, alt, pos, uv), strided as it is read, the whole data being larger than this process may hold"""
        kept = np.stack([raw[name][::hour_stride, :, ::grid_stride, ::grid_stride] for name in (u, v)], axis=-1)
        return jnp.asarray(kept.reshape(kept.shape[0], kept.shape[1], grid.size(), 2))

    return WindData(
        laid_out("u", "v"),
        laid_out("forecast_u", "forecast_v"),
        jnp.asarray(raw["hours"][::hour_stride]),
        jnp.asarray(raw["altitude_km"]),
        grid,
    )


def balloon_transition(wind_data: WindData, states: ProductDomain, step_hours: float) -> BalloonTransition:
    """One time_step of the balloon over `states`: the wind carries it, the action moves it an altitude, and the clock advances"""
    return BalloonTransition(Advection(wind_data.grid, step_hours), Ascent(wind_data.n_alt), Clock(step_hours), states)


def balloon_world(
    grid: SphereGrid,
    states: ProductDomain,
    forecast: Float[Array, "frames alt pos uv"],
    frame_hours_elapsed: Float[Array, " frames"],
    error: WindError,
    transition: BalloonTransition,
    resource_units: int,
    start_box: Bool[Array, " pos"],
) -> World:
    """The problem a forecast poses: the balloon starts over a grid point `start_box` marks, and the wind is that forecast plus an error"""
    wind_domain = FunctionDomain(ProductDomain({**states.parts, "hours_elapsed": BoxDomain(())}), BoxDomain((2,)))
    return World(
        state_domain=ProductDomain(
            {
                **states.parts,
                "position": grid.narrow(start_box),
                "balloon_resource": DiscreteDomain(resource_units + 1),
                "hours_elapsed": BoxDomain(()),
                "field": wind_domain,
            }
        ),
        prior=ProductPrior(
            {
                "position": Uniform(),
                "altitude": Uniform(),
                "balloon_resource": Highest(),
                "hours_elapsed": Zero(),
                "field": ForecastPrior(forecast, error, grid, frame_hours_elapsed),
            }
        ),
        transition=transition,
        readout=PointWind(TimeVaryingWind(forecast, frame_hours_elapsed, grid), states),
    )


def balloon_objective(states: ProductDomain, grid: SphereGrid, target: Target, box: Bool[Array, " pos"]) -> StormSearch:
    """What the episode is scored on: `target` over the states above the grid points `box` marks"""
    return StormSearch(target, balloon_candidates(states, grid, box))
