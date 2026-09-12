"""One step of a balloon: the wind carries it, the action moves it between altitudes, and that spends resource

s = (lat, lon, p, r, W)     where it is, which altitude, what resource is left, the wind it moves in
a in {0, 1, 2}              down one, hold, up one
W(lat, lon, p) = (u, v)     the wind there, metres per second, u eastward and v northward
h                           hours in one step
(i, j)                      the grid cell of (lat, lon): i the row, j the column

Two factors write (lat, lon, p), and the resource follows from what they did:

    (i, j)' = round((i, j) + d),  d = the wind's displacement in cells      Advection
    p'      = p + a - 1           where r > 0 and 0 <= p' < n_altitudes     Ascent
    r'      = r - |p' - p|                                                  Resource spent per move
"""

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from zermelo.interface import DiscreteDomain, Domain, Function, ProductDomain, Transition, TransitionKernel
from zermelo.problems.balloon.grid import SphereGrid

type Act = Int[Array, ""]
"""0 down one altitude, 1 hold, 2 up one"""

HOLD = 1
"""The action that changes no altitude and spends no resource"""


class Factor[Action](ABC):
    """One part of s' sampled from the whole of s"""

    @property
    @abstractmethod
    def part(self) -> str:
        """Which part of s this writes"""

    @abstractmethod
    def sample(self, key: PRNGKeyArray, state: dict[str, Any], action: Action) -> Any:
        """One draw of part'"""


@dataclass(frozen=True)
class Advection(Factor[Act]):
    """The wind carries the balloon, each axis rounding down or up on its own.

        exact = (i, j) + d,   d the wind's step in cells
        f = floor(exact),     u = exact - f in [0, 1)

        P(i' = clip(f_lat + m), j' = clip(f_lon + n)) = (m ? u_lat : 1 - u_lat) * (n ? u_lon : 1 - u_lon)

    for m, n in {0, 1}, clipping onto [0, n_lat - 1] and [0, n_lon - 1]. Zero at every other cell.
    """

    grid: SphereGrid
    step_hours: float

    @property
    def part(self) -> str:
        return "position"

    def offset(self, state: dict[str, Any]) -> Float[Array, "*batch 2"]:  # states batch along a leading axis
        """The step in cells: (v, u) * h * 3.6 km / km_per_degree(lat) / degrees_per_cell"""
        wind = state["field"]({part: state[part] for part in state if part != "field"})  # (..., 2) as (u, v)
        km = jnp.stack([wind[..., 1], wind[..., 0]], axis=-1) * self.step_hours * 3.6  # (..., 2) as (lat, lon)
        return km / self.grid.km_per_degree(state["position"][..., 0]) / self.grid.steps

    def sample(self, key: PRNGKeyArray, state: dict[str, Any], action: Act) -> Float[Array, "*batch 2"]:
        """One draw of (i, j)': each axis takes f + 1 with probability u and f otherwise, then clips onto the grid"""
        exact = self.grid.cell_of(state["position"]) + self.offset(state)
        rounded = jnp.floor(exact) + (jax.random.uniform(key, exact.shape) < exact - jnp.floor(exact))
        return self.grid.coords_of(jnp.clip(rounded, 0, self.grid.shape - 1).astype(jnp.int32))


@dataclass(frozen=True)
class Ascent(Factor[Act]):
    """The action moves the balloon one altitude, if it can afford it and there is one that way:

        p' = p + (a - 1)   where r > 0 and 0 <= p + (a - 1) < n_altitudes
           = p             otherwise

    Deterministic given (p, r, a)
    """

    n_altitudes: int

    @property
    def part(self) -> str:
        return "altitude"

    def moved(self, state: dict[str, Any], action: Act) -> Int[Array, "*batch"]:
        """p' - p, one of -1, 0, +1: the wanted a - 1 where r > 0 and the altitude landed on is in range, else 0"""
        wanted = action - HOLD
        landing = state["altitude"] + wanted
        return jnp.where((state["balloon_resource"] > 0) & (landing >= 0) & (landing < self.n_altitudes), wanted, 0)

    def sample(self, key: PRNGKeyArray, state: dict[str, Any], action: Act) -> Int[Array, ""]:
        """p' = p + (p' - p)"""
        return state["altitude"] + self.moved(state, action)


@dataclass(frozen=True)
class BalloonTransition(Transition[Act]):
    """s' factor by factor: the wind moves (lat, lon), the action moves p, and a move spends resource"""

    advection: Advection
    ascent: Ascent

    state_domain: ProductDomain
    """Every state, in the index order a kernel's arrays follow"""

    @property
    def action_domain(self) -> Domain:
        """Three actions, legal at every state: one the balloon cannot afford moves it nowhere"""
        return DiscreteDomain(3)

    @property
    def factors(self) -> tuple[Factor[Act], ...]:
        """The factors in the order their parts are written"""
        return (self.advection, self.ascent)

    def __call__(self, key: PRNGKeyArray, state: dict[str, Any], action: Act) -> dict[str, Any]:
        """One sampled step, and the resource the altitude change spent"""
        keys = jax.random.split(key, len(self.factors))
        out = {f.part: f.sample(k, state, action) for f, k in zip(self.factors, keys, strict=True)}
        spent = jnp.abs(self.ascent.moved(state, action))
        return {**out, "balloon_resource": state["balloon_resource"] - spent, "field": state["field"]}

    def wind_step(self, hypothesis: Function) -> tuple[Int[Array, "pos_alt 4"], Float[Array, "pos_alt 4"]]:
        """The four cells the wind may carry (lat, lon) to from each (lat, lon, p), and their probabilities.

        (pos_alt, 4) each, pos_alt = pos * alt: W reads neither the resource nor a, so one row serves both.
        """
        grid, pos, n_alt = self.advection.grid, self.advection.grid.size(), self.ascent.n_altitudes
        cells = jnp.repeat(grid.elements(), n_alt, axis=0)  # (pos_alt, 2), altitude varying fastest
        under = {"position": cells, "altitude": jnp.tile(jnp.arange(n_alt), pos), "field": hypothesis}
        exact = grid.cell_of(cells) + self.advection.offset(under)  # (pos_alt, 2) in cells
        floor = jnp.floor(exact)  # both ends clip off this, so a landing beyond the edge puts all its mass on the edge
        low = jnp.clip(floor, 0, grid.shape - 1).astype(jnp.int32)
        high = jnp.clip(floor + 1, 0, grid.shape - 1).astype(jnp.int32)
        up = jnp.clip(exact - floor, 0.0, 1.0)  # (pos_alt, 2): u per axis, the probability of rounding up
        lat_at, lat_prob = jnp.stack([low[:, 0], high[:, 0]], -1), jnp.stack([1.0 - up[:, 0], up[:, 0]], -1)
        lon_at, lon_prob = jnp.stack([low[:, 1], high[:, 1]], -1), jnp.stack([1.0 - up[:, 1], up[:, 1]], -1)
        at = (lat_at[:, :, None] * grid.n_lon + lon_at[:, None, :]).reshape(-1, 4)  # (pos_alt, 4) flat cells
        return at, (lat_prob[:, :, None] * lon_prob[:, None, :]).reshape(-1, 4)

    def kernel(self, hypothesis: Function) -> "BalloonKernel":
        """The law at every state, with `hypothesis` in place of the wind an episode drew"""
        n_alt = self.ascent.n_altitudes
        next_pos, prob = self.wind_step(hypothesis)
        alt = jnp.arange(n_alt)[None, :]  # (1, alt), broadcasting over actions
        # the law is written with resource in hand; the world refuses the move once it runs out
        under = {"altitude": jnp.broadcast_to(alt, (3, n_alt)), "balloon_resource": jnp.ones((3, n_alt), jnp.int32)}
        moved = jnp.stack([self.ascent.moved(under, jnp.asarray(a))[a] for a in range(3)])  # (actions, alt)
        return BalloonKernel(self.state_domain, next_pos, prob, alt + moved)


@register_dataclass
@dataclass(frozen=True)
class BalloonKernel(TransitionKernel[Act]):
    """The one-step law at every state, as four reachable positions and the deterministic landing of p.

    s = (lat, lon, p)                       indexed pos * n_alt + p

    P(s' | s, a) = prob[s, k]               s' = (next_pos[s, k], altitude_at[a, p])
                 = 0                        otherwise
    """

    domain: Domain = dataclasses.field(metadata=dict(static=True))
    """What `values` is indexed by, in its own index order"""

    next_pos: Int[Array, "pos_alt 4"]
    """The four cells the wind may carry (lat, lon) to, as flat positions"""

    prob: Float[Array, "pos_alt 4"]
    """P of each, summing to one along the last axis"""

    altitude_at: Int[Array, "actions alt"]
    """p' = p + a - 1 where that is in range, else p"""

    def expectation(self, values: Float[Array, " states"], action: Act) -> Float[Array, " states"]:
        """out[pos, p] = sum over k in 0..3 of prob[s, k] * values[next_pos[s, k], altitude_at[a, p]].
        Time O(states * 4), memory O(states * 4)
        """
        n_alt = self.altitude_at.shape[1]
        table = values.reshape(-1, n_alt)  # (pos, alt)
        landed = table[:, self.altitude_at[action]]  # (pos, alt): p' applied
        at = self.next_pos.reshape(-1, n_alt, 4)  # (pos, alt, 4): the flat cells the wind may reach
        alt = jnp.arange(n_alt)[None, :, None]  # (1, alt, 1), pairing each cell with the altitude it was read at
        reached = landed[at, alt]  # (pos, alt, 4)
        return jnp.einsum("pak,pak->pa", self.prob.reshape(-1, n_alt, 4), reached).reshape(-1)

    def sample(self, key: PRNGKeyArray, index: Int[Array, ""], action: Act) -> Int[Array, ""]:
        """One draw from P(. | s, a) at `index`: one of the four cells by `prob`, then p' read off"""
        n_alt = self.altitude_at.shape[1]
        p = index % n_alt
        q = self.next_pos[index, jax.random.categorical(key, jnp.log(self.prob[index]))]
        return q * n_alt + self.altitude_at[action, p]

    def step_cost(self, action: Act) -> Float[Array, " states"]:
        """The resource one step spends: `c(s, a) = |altitude_at[a, p] - p|`"""
        n_alt = self.altitude_at.shape[1]
        per_altitude = jnp.abs(self.altitude_at[action] - jnp.arange(n_alt)).astype(jnp.float32)  # (alt,)
        return jnp.tile(per_altitude, self.next_pos.shape[0] // n_alt)  # (states,), altitude varying fastest
