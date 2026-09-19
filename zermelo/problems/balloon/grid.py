"""The sphere a balloon's position lives on, and the state domain it is one part of"""

import math
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from zermelo.interface import Domain, Embeddable, Enumerable, ProductDomain, Subset

EARTH_RADIUS_KM = 6371.0
"""The sphere embedded coordinates are measured on"""


@dataclass(frozen=True)
class SphereGrid(Domain[Float[Array, " 2"]], Enumerable[Float[Array, " 2"]], Embeddable[Float[Array, " 2"]]):
    """A latitude-longitude grid around the whole globe, an element being degrees (latitude, longitude) of one grid point

        (lat, lon) = (lat_first, lon_first) + (row, col) * (lat_step, lon_step)      0 <= row < n_lat, 0 <= col < n_lon

    Longitude goes once round: n_lon * lon_step = 360, so column n_lon is column 0 again.

    Latitude stops short of both poles: the first row sits at lat_first, not at 90. Walk north from a grid
    point in that row, over the pole, and down the other side, and the next grid point met is in the same
    row, half way round the globe:

        facing(row, col)        = (row, col + n_lon / 2)                    for row the first or the last
        degrees walked          = 2 * (90 - |latitude of that row|)
        pole_gap_rows           = degrees walked / |lat_step|               the same walk counted in rows

    n_lon is even, so that half way round is a whole number of columns.
    """

    n_lat: int
    n_lon: int

    lat_first: float
    """The latitude of row 0, in degrees"""

    lon_first: float
    """The longitude of column 0, in degrees"""

    lat_step: float
    """Degrees from one row to the next, negative where latitude descends"""

    lon_step: float
    """Degrees from one column to the next"""

    def __post_init__(self) -> None:
        """Refuse a grid whose columns do not go once round the globe, or whose n_lon / 2 is not a whole column"""
        if not math.isclose(self.n_lon * self.lon_step, 360.0) or self.n_lon % 2:
            raise ValueError(f"{self.n_lon} columns {self.lon_step} degrees apart do not go once round the globe in an even count")

    @property
    def steps(self) -> Float[Array, " 2"]:
        """The per-axis spacing, in degrees"""
        return jnp.asarray([self.lat_step, self.lon_step])

    @property
    def first(self) -> Float[Array, " 2"]:
        """The degrees of row 0, column 0"""
        return jnp.asarray([self.lat_first, self.lon_first])

    @property
    def pole_gap_rows(self) -> tuple[float, float]:
        """pole_gap_rows for the first row, then for the last row: 2 * (90 - |latitude of that row|) / |lat_step|"""
        lat_last = self.lat_first + (self.n_lat - 1) * self.lat_step
        return 2.0 * (90.0 - abs(self.lat_first)) / abs(self.lat_step), 2.0 * (90.0 - abs(lat_last)) / abs(self.lat_step)

    def row_col_of(self, x: Float[Array, "*batch 2"]) -> Int[Array, "*batch 2"]:
        """The nearest grid point to degrees `x`: the row clipped into range, the column taken round the globe"""
        nearest = jnp.round((x - self.first) / self.steps).astype(jnp.int32)
        return jnp.stack([jnp.clip(nearest[..., 0], 0, self.n_lat - 1), nearest[..., 1] % self.n_lon], axis=-1)

    def degrees_of(self, row_col: Int[Array, "*batch 2"]) -> Float[Array, "*batch 2"]:
        """first + (row, col) * step"""
        return self.first + row_col * self.steps

    def landing_grid_points(self, x: Float[Array, "*batch 2"]) -> tuple[Int[Array, "*batch 4"], Float[Array, "*batch 4"]]:
        """The four grid points around degrees `x` as flat indices, and the probability of landing on each, their mean position being `x`:

            row_exact, col_exact = (x - first) / step               where x sits, counted in rows and columns
            col_low  = floor(col_exact),  col_high = col_low + 1    P(col_high) = col_exact - col_low
            row_low  = floor(row_exact),  row_high = row_low + 1    P(row_high) = row_exact - row_low

        and where x is inside a polar cap, `rows_past` rows beyond the first or last row, both rows are that row,
        one of them reached across the pole:

            P(facing(row, col)) = rows_past / pole_gap_rows,   else (row, col) itself

        Index order: (row_low, col_low), (row_low, col_high), (row_high, col_low), (row_high, col_high).
        """
        exact = (x - self.first) / self.steps
        row_exact, col_exact = exact[..., 0], exact[..., 1]
        gap_past_first_row, gap_past_last_row = self.pole_gap_rows
        last_row = self.n_lat - 1
        rows_past_first, rows_past_last = -row_exact, row_exact - last_row  # positive inside that pole's cap
        in_first_cap, in_last_cap = rows_past_first > 0, rows_past_last > 0
        floor_row = jnp.floor(row_exact)
        row_low = jnp.where(in_first_cap, 0, jnp.where(in_last_cap, last_row, floor_row)).astype(jnp.int32)
        row_high = jnp.where(in_first_cap, 0, jnp.where(in_last_cap, last_row, jnp.minimum(floor_row + 1, last_row))).astype(jnp.int32)
        # in the first row's cap the far side plays row_low, in the last row's cap it plays row_high
        low_is_across_pole, high_is_across_pole = in_first_cap, in_last_cap
        prob_row_high = jnp.where(
            in_first_cap,
            1.0 - rows_past_first / gap_past_first_row,
            jnp.where(in_last_cap, rows_past_last / gap_past_last_row, row_exact - floor_row),
        )
        col_low = jnp.floor(col_exact).astype(jnp.int32)
        prob_col_high = col_exact - col_low
        columns_half_way_round = self.n_lon // 2
        rows = jnp.stack([row_low, row_low, row_high, row_high], axis=-1)  # (*batch, 4)
        cols = jnp.stack([col_low, col_low + 1, col_low, col_low + 1], axis=-1)
        across_pole = jnp.stack([low_is_across_pole, low_is_across_pole, high_is_across_pole, high_is_across_pole], axis=-1)
        prob_row = jnp.stack([1.0 - prob_row_high, 1.0 - prob_row_high, prob_row_high, prob_row_high], axis=-1)
        prob_col = jnp.stack([1.0 - prob_col_high, prob_col_high, 1.0 - prob_col_high, prob_col_high], axis=-1)
        return rows * self.n_lon + (cols + across_pole * columns_half_way_round) % self.n_lon, prob_row * prob_col

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
        """3: a point on the sphere"""
        return 3

    def unit_vector(self, x: Float[Array, "*batch 2"]) -> Float[Array, "*batch 3"]:
        """(cos(lat) cos(lon), cos(lat) sin(lon), sin(lat)): degrees `x` as a point on the unit sphere"""
        lat, lon = jnp.radians(x[..., 0]), jnp.radians(x[..., 1])
        return jnp.stack([jnp.cos(lat) * jnp.cos(lon), jnp.cos(lat) * jnp.sin(lon), jnp.sin(lat)], axis=-1)

    def embed(self, x: Float[Array, "*batch 2"]) -> Float[Array, "*batch 3"]:
        """EARTH_RADIUS_KM * unit_vector(x), so a chord approximates an arc"""
        return EARTH_RADIUS_KM * self.unit_vector(x)

    def degrees_at(self, v: Float[Array, "*batch 3"]) -> Float[Array, "*batch 2"]:
        """The (latitude, longitude) a vector from the earth's centre points at, of whatever length"""
        lat = jnp.arcsin(jnp.clip(v[..., 2] / jnp.linalg.norm(v, axis=-1), -1.0, 1.0))
        return jnp.degrees(jnp.stack([lat, jnp.arctan2(v[..., 1], v[..., 0])], axis=-1))

    def unembed(self, v: Float[Array, "*batch 3"]) -> Float[Array, "*batch 2"]:
        """The degrees `v` points at, snapped onto the grid"""
        return self.project(self.degrees_at(v))


@dataclass(frozen=True)
class Steps(Domain[Int[Array, ""]], Enumerable[Int[Array, ""]], Embeddable[Int[Array, ""]]):
    """The integers `0` to `len(coordinate) - 1`, step `i` embedding to `coordinate[i]`"""

    coordinate: tuple[float, ...]
    """Where each step sits, in the units distance between states is measured in"""

    def contains(self, x: Int[Array, ""]) -> Bool[Array, ""]:
        return (0 <= x) & (x < len(self.coordinate))

    def project(self, x: Int[Array, ""]) -> Int[Array, ""]:
        return jnp.clip(x, 0, len(self.coordinate) - 1)

    def narrow(self, mask: Bool[Array, " n"]) -> Subset[Int[Array, ""]]:
        """`mask` marks which steps are live now"""
        return Subset(self, mask)

    def size(self) -> int:
        return len(self.coordinate)

    def index_of(self, x: Int[Array, ""]) -> Int[Array, ""]:
        return jnp.asarray(x, jnp.int32)

    def from_index(self, i: Int[Array, ""]) -> Int[Array, ""]:
        return i

    def elements(self) -> Int[Array, " n"]:
        return jnp.arange(len(self.coordinate))

    def dim(self) -> int:
        """1"""
        return 1

    def embed(self, x: Int[Array, "*batch"]) -> Float[Array, "*batch 1"]:
        """coordinate[x]"""
        return jnp.asarray(self.coordinate)[jnp.asarray(x, jnp.int32)][..., None]

    def unembed(self, v: Float[Array, "*batch 1"]) -> Int[Array, "*batch"]:
        """The step nearest `v`"""
        return jnp.argmin(jnp.abs(jnp.asarray(self.coordinate) - v[..., :1]), axis=-1)


def balloon_states(grid: SphereGrid, altitude_km: tuple[float, ...]) -> ProductDomain:
    """The state domain: which grid point the balloon is over, and which altitude it flies at"""
    return ProductDomain({"position": grid, "altitude": Steps(altitude_km)})
