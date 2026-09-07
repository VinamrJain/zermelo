"""A balloon problem, assembled from a recorded wind field"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from zermelo.interface import BoxDomain, Domain, Enumerable, FunctionDomain, Prior, ProductDomain, ProductPrior, Uniform, World
from zermelo.problems.balloon.field import ForecastPrior, WindError
from zermelo.problems.balloon.grid import SphereGrid, balloon_states
from zermelo.problems.balloon.objective import StormSearch, Target, balloon_candidates
from zermelo.problems.balloon.readout import BalloonReadout
from zermelo.problems.balloon.transition import Advection, Ascent, BalloonTransition, Expenditure


class Highest(Prior[Any]):
    """The largest element of a finite domain, drawn with probability one"""

    def sample(self, domain: Domain[Any], key: PRNGKeyArray) -> Any:
        """The element at `size() - 1`"""
        if not isinstance(domain, Enumerable):
            raise TypeError(f"a {type(domain).__name__} has no largest element, so none can be drawn from it")
        return domain.from_index(jnp.asarray(domain.size() - 1))


@dataclass(frozen=True)
class Recording:
    """One wind record read off disk: W over time, level and cell, on the grid it was sampled on"""

    wind: Float[Array, "t alt pos uv"]
    hours: Float[Array, " t"]
    altitude_km: Float[Array, " alt"]
    grid: SphereGrid

    @property
    def n_alt(self) -> int:
        """How many levels the ladder has"""
        return self.wind.shape[1]

    def at(self, frame: int) -> Float[Array, "alt pos uv"]:
        """W at one recorded hour"""
        return self.wind[frame]


def load(path: Path) -> Recording:
    """The wind record stored at `path`, laid out for the grid it was sampled on"""
    raw = np.load(path)
    lat, lon = raw["latitude"], raw["longitude"]
    grid = SphereGrid(
        n_lat=len(lat),
        n_lon=len(lon),
        lat_first=float(lat[0]),
        lon_first=float(lon[0]),
        lat_step=float(lat[1] - lat[0]),
        lon_step=float(lon[1] - lon[0]),
    )
    wind = np.stack([raw["u"], raw["v"]], axis=-1).reshape(raw["u"].shape[0], raw["u"].shape[1], grid.size(), 2)
    return Recording(jnp.asarray(wind), jnp.asarray(raw["hours"]), jnp.asarray(raw["altitude_km"]), grid)


def balloon_world(
    recording: Recording, frame: int, ballast_units: int, step_hours: float, error: WindError, readout: type[BalloonReadout]
) -> World:
    """The problem a recorded field poses: the balloon starts anywhere, and the wind is the record plus an error"""
    grid, n_alt = recording.grid, recording.n_alt
    forecast = recording.at(frame)
    ascent = Ascent(n_alt)
    state_domain = balloon_states(grid, n_alt, ballast_units)
    return World(
        state_domain=ProductDomain({**state_domain.parts, "field": FunctionDomain(grid, BoxDomain((2,)))}),
        prior=ProductPrior(
            {"position": Uniform(), "altitude": Uniform(), "ballast": Highest(), "field": ForecastPrior(forecast, error, grid)}
        ),
        transition=BalloonTransition(Advection(grid, step_hours), ascent, Expenditure(ascent, ballast_units + 1), state_domain),
        readout=readout(grid, n_alt, forecast),
    )


def balloon_objective(recording: Recording, target: Target, margin_lat: int, margin_lon: int, step_cost: float) -> StormSearch:
    """What the episode is scored on: the peak of `target` over the grid's interior"""
    return StormSearch(recording.grid, recording.n_alt, target, balloon_candidates(recording.grid, margin_lat, margin_lon), step_cost)
