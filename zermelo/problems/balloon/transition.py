"""One step of a balloon: the wind carries it, the action moves it between levels, and that costs ballast

s = (lat, lon, p, b, W)     where it is, which pressure level, what ballast is left, the wind it moves in
a in {0, 1, 2}              down a level, hold, up a level
W(lat, lon, p) = (u, v)     the wind there, metres per second, u eastward and v northward
h                           hours in one step
"""

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from zermelo.interface import DiscreteDomain, Domain, ProductDomain, Transition, TransitionKernel
from zermelo.problems.balloon.field import WindField
from zermelo.problems.balloon.grid import SphereGrid

type Act = Int[Array, ""]
"""0 down a level, 1 hold, 2 up a level"""

HOLD = 1
"""The action that changes no level and spends no ballast"""


class Factor[Action](ABC):
    """One part of s' given the whole of s, as a sampler and as a law over that part's values"""

    @property
    @abstractmethod
    def part(self) -> str:
        """Which part of s this writes"""

    @property
    @abstractmethod
    def values(self) -> int:
        """How many values that part takes"""

    @abstractmethod
    def sample(self, key: PRNGKeyArray, state: dict[str, Any], action: Action) -> Any:
        """One draw of part'"""

    @abstractmethod
    def law(self, states: dict[str, Any], action: Action) -> Float[Array, "states values"]:
        """P(part' | s, a) at every s in `states`, whose field is whatever those states carry"""


@dataclass(frozen=True)
class Advection(Factor[Act]):
    """(lat, lon)' = round((lat, lon) + W(s) * h), landing split between the cells it falls between"""

    grid: SphereGrid
    step_hours: float

    @property
    def part(self) -> str:
        return "position"

    @property
    def values(self) -> int:
        return self.grid.size()

    def offset(self, state: dict[str, Any]) -> Float[Array, "*batch 2"]:  # states batch along a leading axis
        """The step in cells: (v, u) * h * 3.6 km / km_per_degree(lat) / degrees_per_cell"""
        wind = state["field"](state)  # (..., 2) as (u, v)
        km = jnp.stack([wind[..., 1], wind[..., 0]], axis=-1) * self.step_hours * 3.6  # (..., 2) as (lat, lon)
        return km / self.grid.km_per_degree(state["position"][..., 0]) / self.grid.steps

    def sample(self, key: PRNGKeyArray, state: dict[str, Any], action: Act) -> Float[Array, "*batch 2"]:
        """Each axis lands on floor(c + d) + 1 with probability frac(d), else floor(c + d)"""
        exact = self.grid.cell_of(state["position"]) + self.offset(state)
        rounded = jnp.floor(exact) + (jax.random.uniform(key, exact.shape) < exact - jnp.floor(exact))
        return self.grid.coords_of(jnp.clip(rounded, 0, self.grid.shape - 1).astype(jnp.int32))

    def law(self, states: dict[str, Any], action: Act) -> Float[Array, "states pos"]:
        """Mass on the four cells the landing point falls between, one factor per axis"""
        exact = self.grid.cell_of(states["position"]) + self.offset(states)  # (states, 2)
        low = jnp.clip(jnp.floor(exact), 0, self.grid.shape - 1).astype(jnp.int32)
        high = jnp.clip(low + 1, 0, self.grid.shape - 1)
        up = jnp.clip(exact - jnp.floor(exact), 0.0, 1.0)  # (states, 2): the mass going to `high`
        rows = jnp.arange(exact.shape[0])
        out = jnp.zeros((exact.shape[0], self.grid.size()))
        for lat_at, lat_mass in ((low[:, 0], 1.0 - up[:, 0]), (high[:, 0], up[:, 0])):
            for lon_at, lon_mass in ((low[:, 1], 1.0 - up[:, 1]), (high[:, 1], up[:, 1])):
                out = out.at[rows, lat_at * self.grid.n_lon + lon_at].add(lat_mass * lon_mass)
        return out


@dataclass(frozen=True)
class Ascent(Factor[Act]):
    """p' = p + a - 1 where b > 0 and that stays in range, else p"""

    n_levels: int

    @property
    def part(self) -> str:
        return "altitude"

    @property
    def values(self) -> int:
        return self.n_levels

    def moved(self, state: dict[str, Any], action: Act) -> Int[Array, "*batch"]:
        """a - 1 where affordable and in range, else 0"""
        wanted = action - HOLD
        landing = state["altitude"] + wanted
        return jnp.where((state["ballast"] > 0) & (landing >= 0) & (landing < self.n_levels), wanted, 0)

    def sample(self, key: PRNGKeyArray, state: dict[str, Any], action: Act) -> Int[Array, ""]:
        """p + moved"""
        return state["altitude"] + self.moved(state, action)

    def law(self, states: dict[str, Any], action: Act) -> Float[Array, "states alt"]:
        """One at p + moved"""
        return jax.nn.one_hot(states["altitude"] + self.moved(states, action), self.n_levels)


@dataclass(frozen=True)
class Expenditure(Factor[Act]):
    """b' = b - 1 where the level moved, else b"""

    ascent: Ascent
    ballast_values: int

    @property
    def part(self) -> str:
        return "ballast"

    @property
    def values(self) -> int:
        return self.ballast_values

    def sample(self, key: PRNGKeyArray, state: dict[str, Any], action: Act) -> Int[Array, ""]:
        """b - |moved|"""
        return state["ballast"] - jnp.abs(self.ascent.moved(state, action))

    def law(self, states: dict[str, Any], action: Act) -> Float[Array, "states ballast"]:
        """One at b - |moved|"""
        return jax.nn.one_hot(states["ballast"] - jnp.abs(self.ascent.moved(states, action)), self.ballast_values)


@dataclass(frozen=True)
class BalloonTransition(Transition[Act]):
    """s' factor by factor: the wind moves (lat, lon), the action moves p, a move spends b"""

    advection: Advection
    ascent: Ascent
    expenditure: Expenditure

    state_domain: ProductDomain
    """Every state, in the index order a kernel's arrays follow"""

    @property
    def action_domain(self) -> Domain:
        """Three actions, always the same three"""
        return DiscreteDomain(3)

    @property
    def factors(self) -> tuple[Factor[Act], ...]:
        """The factors in the order their parts are written"""
        return (self.advection, self.ascent, self.expenditure)

    def legal_actions(self, state: dict[str, Any]) -> Domain:
        """Hold always; up and down only where b > 0 and the level stays in range"""
        p, spare = state["altitude"], state["ballast"] > 0
        live = jnp.stack([spare & (p > 0), jnp.bool_(True), spare & (p < self.ascent.n_levels - 1)])
        return self.action_domain.narrow(live)

    def __call__(self, key: PRNGKeyArray, state: dict[str, Any], action: Act) -> dict[str, Any]:
        """One sampled step, the field carried through unchanged"""
        keys = jax.random.split(key, len(self.factors))
        out = {f.part: f.sample(k, state, action) for f, k in zip(self.factors, keys, strict=True)}
        return {**out, "field": state["field"]}

    def kernel(self, hypothesis: WindField) -> "BalloonKernel":
        """The law at every enumerated state, with `hypothesis` in place of the wind an episode drew"""
        under = {**self.state_domain.elements(), "field": hypothesis}
        laws = tuple(
            jnp.stack([f.law(under, jnp.asarray(a)) for a in range(3)])  # (actions, states, values)
            for f in self.factors
        )
        return BalloonKernel(laws[0], laws[1], laws[2])


@register_dataclass
@dataclass(frozen=True)
class BalloonKernel(TransitionKernel[Act]):
    """The one-step law at every state, one dense factor per part of s"""

    position_law: Float[Array, "actions states pos"]
    """P((lat, lon)' | s, a)"""

    altitude_law: Float[Array, "actions states alt"]
    """P(p' | s, a)"""

    ballast_law: Float[Array, "actions states ballast"]
    """P(b' | s, a)"""

    def expectation(self, values: Float[Array, " states"], action: Act) -> Float[Array, " states"]:
        """out[s] = sum over (q, l, c) of position_law[s, q] altitude_law[s, l] ballast_law[s, c] values[q, l, c],
        contracted one part at a time: (states, pos, alt, ballast) -> (states, pos, alt) -> (states, pos) -> (states,)
        """
        n_pos, n_alt, n_ball = self.position_law.shape[2], self.altitude_law.shape[2], self.ballast_law.shape[2]
        table = values.reshape(n_pos, n_alt, n_ball)
        over_ballast = jnp.tensordot(self.ballast_law[action], table, axes=(1, 2))  # (states, pos, alt)
        over_altitude = jnp.einsum("sl,spl->sp", self.altitude_law[action], over_ballast)
        return jnp.einsum("sq,sq->s", self.position_law[action], over_altitude)

    def sample(self, key: PRNGKeyArray, index: Int[Array, ""], action: Act) -> Int[Array, ""]:
        """One draw from P(. | s, a) at `index`, as this domain's own flat index"""
        n_alt, n_ball = self.altitude_law.shape[2], self.ballast_law.shape[2]
        k_pos, k_alt, k_ball = jax.random.split(key, 3)
        q = jax.random.categorical(k_pos, jnp.log(self.position_law[action, index]))
        l = jax.random.categorical(k_alt, jnp.log(self.altitude_law[action, index]))
        c = jax.random.categorical(k_ball, jnp.log(self.ballast_law[action, index]))
        return (q * n_alt + l) * n_ball + c
