"""A boat problem, assembled from one frame of a ocean current and the wind over it"""

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from zermelo.interface import BoxDomain, FunctionDomain, ProductDomain, ProductPrior, Uniform, World
from zermelo.problems.ocean.field import ForecastPrior, OceanCurrent, OceanCurrentError
from zermelo.problems.ocean.grid import PlaneGrid
from zermelo.problems.ocean.objective import LearnRegion, ReachTarget
from zermelo.problems.ocean.readout import PointOceanCurrent
from zermelo.problems.ocean.transition import OceanTransition, Polar


@dataclass(frozen=True)
class OceanData:
    """The data: W the ocean current and U the wind at every frame, and which grid points are land

    hour    hours since the data's own start, one per frame
    """

    ocean_current: Float[Array, "frames pos uv"]
    wind: Float[Array, "frames pos uv"]
    hour: Float[Array, " frames"]
    land: Bool[Array, " pos"]
    grid: PlaneGrid

    def frame(self, hour: float) -> int:
        """The index of the frame at `hour`"""
        at = np.flatnonzero(np.asarray(self.hour) == hour)
        if at.size != 1:
            raise ValueError(f"the data holds no frame at hour {hour}; it runs from hour {float(self.hour[0])} to {float(self.hour[-1])}")
        return int(at[0])


def load_ocean(path: Path, grid_stride: int) -> OceanData:
    """The data stored at `path`, keeping every `grid_stride`-th row and column; land carries zero ocean current and zero wind"""
    raw = np.load(path)
    lat, lon = raw["latitude"][::grid_stride], raw["longitude"][::grid_stride]
    grid = PlaneGrid(
        n_lat=len(lat),
        n_lon=len(lon),
        lat_first=float(lat[0]),
        lon_first=float(lon[0]),
        lat_step=float(lat[1] - lat[0]),
        lon_step=float(lon[1] - lon[0]),
    )

    def laid_out(u: str, v: str) -> Float[Array, "frames pos uv"]:
        """One field as (frames, pos, uv), strided as it is read"""
        kept = np.stack([raw[name][:, ::grid_stride, ::grid_stride] for name in (u, v)], axis=-1)
        return jnp.asarray(kept.reshape(kept.shape[0], grid.size(), 2))

    return OceanData(
        laid_out("u", "v"),
        laid_out("wind_u", "wind_v"),
        jnp.asarray(raw["hours"]),
        jnp.asarray(raw["land"][::grid_stride, ::grid_stride].reshape(-1)),
        grid,
    )


def ocean_states(grid: PlaneGrid) -> ProductDomain:
    """The state domain a belief and a plan are written over: which grid point the boat is at"""
    return ProductDomain({"position": grid})


def ocean_transition(
    grid: PlaneGrid, wind: Float[Array, "pos uv"], polar: Polar, n_headings: int, step_hours: float, states: ProductDomain
) -> OceanTransition:
    """One time_step of the boat over `states` under the wind `wind`"""
    return OceanTransition(grid, polar, wind, n_headings, step_hours, states)


def ocean_world(
    grid: PlaneGrid,
    states: ProductDomain,
    forecast: Float[Array, "pos uv"],
    error: OceanCurrentError,
    transition: OceanTransition,
    noise_ms: float,
    start_box: Bool[Array, " pos"],
    target_box: Bool[Array, " pos"],
) -> World:
    """The problem a forecast poses: the boat starts at a grid point `start_box` marks, its target is drawn uniformly from `target_box`, and the ocean current is that forecast plus an error"""
    return World(
        state_domain=ProductDomain(
            {"position": grid.narrow(start_box), "target": grid.narrow(target_box), "field": FunctionDomain(states, BoxDomain((2,)))}
        ),
        prior=ProductPrior({"position": Uniform(), "target": Uniform(), "field": ForecastPrior(forecast, error, grid)}),
        transition=transition,
        readout=PointOceanCurrent(OceanCurrent(forecast, grid), states, noise_ms),
    )


def ocean_objective(states: ProductDomain, grid: PlaneGrid, box: Bool[Array, " pos"], arrival_km: float) -> ReachTarget:
    """What the episode is scored on: reaching the target, with waypoints and claims over the grid points `box` marks"""
    return ReachTarget(states.narrow(box), grid, arrival_km)


def learn_objective(states: ProductDomain, box: Bool[Array, " pos"], prior_log_variance: float) -> LearnRegion:
    """What the episode is scored on: information about the ocean current over the grid points `box` marks, waypoints aimed inside it"""
    return LearnRegion(states.narrow(box), prior_log_variance)
