"""A latitude-longitude grid over one box of sea, an element being degrees (latitude, longitude) of one grid point

    (lat, lon) = (lat_first, lon_first) + (row, col) * (lat_step, lon_step)      0 <= row < n_lat, 0 <= col < n_lon

Distance is measured on the plane tangent to the earth at the box's middle latitude:

    (north_km, east_km) = ((lat - lat_first) * KM_PER_DEGREE, (lon - lon_first) * KM_PER_DEGREE * cos(lat_mid))
"""

import math
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from zermelo.interface import Domain, Embeddable, Enumerable, Subset

KM_PER_DEGREE = 111.2
"""Kilometres in one degree of latitude"""


@dataclass(frozen=True)
class PlaneGrid(Domain[Float[Array, " 2"]], Enumerable[Float[Array, " 2"]], Embeddable[Float[Array, " 2"]]):
    """The grid points of one box, the box's edge holding whatever is pushed past it"""

    n_lat: int
    n_lon: int

    lat_first: float
    """The latitude of row 0, in degrees"""

    lon_first: float
    """The longitude of column 0, in degrees"""

    lat_step: float
    """Degrees from one row to the next"""

    lon_step: float
    """Degrees from one column to the next"""

    def __post_init__(self) -> None:
        """Refuse a grid with fewer than two rows or two columns"""
        if self.n_lat < 2 or self.n_lon < 2:
            raise ValueError(f"a grid of {self.n_lat} rows by {self.n_lon} columns has no cell to land inside")

    @property
    def steps(self) -> Float[Array, " 2"]:
        """The per-axis spacing, in degrees"""
        return jnp.asarray([self.lat_step, self.lon_step])

    @property
    def first(self) -> Float[Array, " 2"]:
        """The degrees of row 0, column 0"""
        return jnp.asarray([self.lat_first, self.lon_first])

    @property
    def shape(self) -> Float[Array, " 2"]:
        """(n_lat, n_lon)"""
        return jnp.asarray([self.n_lat, self.n_lon], jnp.float32)

    @property
    def km_per_degree(self) -> Float[Array, " 2"]:
        """Kilometres in one degree of (latitude, longitude) at the box's middle latitude"""
        lat_mid = self.lat_first + (self.n_lat - 1) * self.lat_step / 2
        return jnp.asarray([KM_PER_DEGREE, KM_PER_DEGREE * math.cos(math.radians(lat_mid))])

    def row_col_of(self, x: Float[Array, "*batch 2"]) -> Int[Array, "*batch 2"]:
        """The nearest grid point to degrees `x`, clipped into the box"""
        nearest = jnp.round((x - self.first) / self.steps)
        return jnp.clip(nearest, 0, self.shape - 1).astype(jnp.int32)

    def degrees_of(self, row_col: Int[Array, "*batch 2"]) -> Float[Array, "*batch 2"]:
        """first + (row, col) * step"""
        return self.first + row_col * self.steps

    def displaced(self, x: Float[Array, "*batch 2"], east_north_km: Float[Array, "*batch 2"]) -> Float[Array, "*batch 2"]:
        """Degrees `x` moved by (east_km, north_km) on the tangent plane"""
        return x + east_north_km[..., ::-1] / self.km_per_degree

    def landing_grid_points(self, x: Float[Array, "*batch 2"]) -> tuple[Int[Array, "*batch 4"], Float[Array, "*batch 4"]]:
        """The four grid points around degrees `x` as flat indices, and the probability of landing on each, their mean position being `x`:

            row_exact, col_exact = (x - first) / step                       where x sits, counted in rows and columns
            row_low  = clip(floor(row_exact), 0, n_lat - 2),  row_high = row_low + 1,  P(row_high) = clip(row_exact - row_low, 0, 1)
            col_low  = clip(floor(col_exact), 0, n_lon - 2),  col_high = col_low + 1,  P(col_high) = clip(col_exact - col_low, 0, 1)

        A point past the box's edge lands on the edge with probability one.
        Index order: (row_low, col_low), (row_low, col_high), (row_high, col_low), (row_high, col_high).
        """
        exact = (x - self.first) / self.steps
        low = jnp.clip(jnp.floor(exact), 0, self.shape - 2).astype(jnp.int32)  # (*batch, 2)
        prob_high = jnp.clip(exact - low, 0.0, 1.0)  # (*batch, 2)
        rows = jnp.stack([low[..., 0], low[..., 0], low[..., 0] + 1, low[..., 0] + 1], axis=-1)  # (*batch, 4)
        cols = jnp.stack([low[..., 1], low[..., 1] + 1, low[..., 1], low[..., 1] + 1], axis=-1)
        prob_row = jnp.stack([1.0 - prob_high[..., 0]] * 2 + [prob_high[..., 0]] * 2, axis=-1)
        prob_col = jnp.stack([1.0 - prob_high[..., 1], prob_high[..., 1]] * 2, axis=-1)
        return rows * self.n_lon + cols, prob_row * prob_col

    def inside(self, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> Bool[Array, " pos"]:
        """The grid points with lat_min <= lat <= lat_max and lon_min <= lon <= lon_max"""
        if lat_min > lat_max or lon_min > lon_max:
            raise ValueError(
                f"a box runs from its smaller bound to its larger, and this one is lat {lat_min}..{lat_max}, lon {lon_min}..{lon_max}"
            )
        at = self.elements()  # (pos, 2)
        return (at[:, 0] >= lat_min) & (at[:, 0] <= lat_max) & (at[:, 1] >= lon_min) & (at[:, 1] <= lon_max)

    def contains(self, x: Float[Array, " 2"]) -> Bool[Array, ""]:
        """True where `x` sits on a grid point"""
        return jnp.allclose(self.project(x), x)

    def project(self, x: Float[Array, "*batch 2"]) -> Float[Array, "*batch 2"]:
        """`x` snapped to the nearest grid point"""
        return self.degrees_of(self.row_col_of(x))

    def narrow(self, mask: Bool[Array, " pos"]) -> Subset[Float[Array, " 2"]]:
        """This grid restricted to the grid points `mask` marks, in this grid's index order"""
        return Subset(self, mask)

    def size(self) -> int:
        """n_lat * n_lon"""
        return self.n_lat * self.n_lon

    def index_of(self, x: Float[Array, " 2"]) -> Int[Array, ""]:
        """The flat index of `x`"""
        return self.flat_index(x)

    def flat_index(self, x: Float[Array, "*batch 2"]) -> Int[Array, "*batch"]:
        """row * n_lon + col, the longitude varying fastest. Batches, unlike `index_of`"""
        row_col = self.row_col_of(x)
        return row_col[..., 0] * self.n_lon + row_col[..., 1]

    def from_index(self, i: Int[Array, "*batch"]) -> Float[Array, "*batch 2"]:
        """The degrees of the grid point at flat index `i`"""
        return self.degrees_of(jnp.stack([i // self.n_lon, i % self.n_lon], axis=-1))

    def elements(self) -> Float[Array, "pos 2"]:
        """Every grid point, as (n_lat * n_lon, 2) degrees in flat index order"""
        return self.from_index(jnp.arange(self.size()))

    def dim(self) -> int:
        """2: (north_km, east_km)"""
        return 2

    def embed(self, x: Float[Array, "*batch 2"]) -> Float[Array, "*batch 2"]:
        """(north_km, east_km) of degrees `x` from row 0, column 0"""
        return (x - self.first) * self.km_per_degree

    def unembed(self, v: Float[Array, "*batch 2"]) -> Float[Array, "*batch 2"]:
        """The grid point nearest (north_km, east_km) `v`"""
        return self.project(self.first + v / self.km_per_degree)
