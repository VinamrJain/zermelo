"""One time_step of a balloon: the wind carries it, the action moves it between altitudes, and that spends resource

state = (position, altitude, balloon_resource, hours_elapsed, field)
    position            (lat, lon) degrees of the grid point the balloon is over
    altitude            which altitude, an index
    balloon_resource    altitude changes left
    hours_elapsed       hours since the episode began
    field               W, the wind: W(position, altitude, hours_elapsed) = (u, v) in m/s, u eastward and v northward
action in {0, 1, 2}     down one altitude, hold, up one
step_hours              hours in one time_step

Three factors write (position, altitude, hours_elapsed), and the resource follows from what they did:

    position'         ~ one of the four grid points around a great-circle step along W    Advection
    altitude'         = altitude + action - 1   where balloon_resource > 0 and in range   Ascent
    hours_elapsed'    = hours_elapsed + step_hours                                        Clock
    balloon_resource' = balloon_resource - |altitude' - altitude|
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
from zermelo.problems.balloon.grid import EARTH_RADIUS_KM, SphereGrid

type Act = Int[Array, ""]
"""0 down one altitude, 1 hold, 2 up one"""

HOLD = 1
"""The action that changes no altitude and spends no resource"""


class Factor[Action](ABC):
    """One part of state' sampled from the whole of state"""

    @property
    @abstractmethod
    def part(self) -> str:
        """Which part of the state this writes"""

    @abstractmethod
    def sample(self, key: PRNGKeyArray, state: dict[str, Any], action: Action) -> Any:
        """One draw of part'"""


@dataclass(frozen=True)
class Advection(Factor[Act]):
    """The wind carries the balloon along a great circle for one time_step, and it lands on a grid point nearby:

        here             = unit vector of position                                a point on the unit sphere
        east_from_here   = (-sin(lon), cos(lon), 0)                               unit vector pointing east at `here`
        north_from_here  = (-sin(lat) cos(lon), -sin(lat) sin(lon), cos(lat))     unit vector pointing north at `here`
        (u, v)           = W(state)                                               m/s
        speed            = ||(u, v)||
        heading_towards  = (u * east_from_here + v * north_from_here) / speed     unit vector the wind blows towards
        angle_travelled  = speed * step_hours * 3.6 / EARTH_RADIUS_KM             radians of arc; 3.6 turns m/s * hours into km
        carried_to       = here * cos(angle_travelled) + heading_towards * sin(angle_travelled)

    `carried_to` lies between grid points, so position' is drawn from the four around it, with probabilities
    whose mean position is `carried_to`. A pole is crossed like any other point.
    """

    grid: SphereGrid
    step_hours: float

    @property
    def part(self) -> str:
        return "position"

    def carried_to(self, state: dict[str, Any]) -> Float[Array, "*batch 2"]:  # states batch along a leading axis
        """`carried_to` as (lat, lon) degrees"""
        wind = state["field"]({part: state[part] for part in state if part != "field"})  # (..., 2) as (u, v)
        lat, lon = jnp.radians(state["position"][..., 0]), jnp.radians(state["position"][..., 1])
        here = self.grid.unit_vector(state["position"])
        east_from_here = jnp.stack([-jnp.sin(lon), jnp.cos(lon), jnp.zeros_like(lon)], axis=-1)
        north_from_here = jnp.stack([-jnp.sin(lat) * jnp.cos(lon), -jnp.sin(lat) * jnp.sin(lon), jnp.cos(lat)], axis=-1)
        speed = jnp.linalg.norm(wind, axis=-1, keepdims=True)  # (..., 1)
        # still air blows towards nowhere, and angle_travelled = 0 then drops the term
        heading_towards = (wind[..., :1] * east_from_here + wind[..., 1:] * north_from_here) / jnp.maximum(speed, 1e-12)
        angle_travelled = speed * self.step_hours * 3.6 / EARTH_RADIUS_KM
        return self.grid.degrees_at(here * jnp.cos(angle_travelled) + heading_towards * jnp.sin(angle_travelled))

    def sample(self, key: PRNGKeyArray, state: dict[str, Any], action: Act) -> Float[Array, "*batch 2"]:
        """One draw of position'"""
        grid_point_index, probability = self.grid.landing_grid_points(self.carried_to(state))  # (..., 4) each
        drawn = jax.random.categorical(key, jnp.log(probability), axis=-1)  # (...,): which of the four
        return self.grid.from_index(jnp.take_along_axis(grid_point_index, drawn[..., None], axis=-1)[..., 0])


@dataclass(frozen=True)
class Ascent(Factor[Act]):
    """The action moves the balloon one altitude, if it can afford it and there is one that way:

    altitude' = altitude + (action - 1)   where balloon_resource > 0 and 0 <= altitude + (action - 1) < n_altitudes
              = altitude                  otherwise
    """

    n_altitudes: int

    @property
    def part(self) -> str:
        return "altitude"

    def moved(self, state: dict[str, Any], action: Act) -> Int[Array, "*batch"]:
        """altitude' - altitude, one of -1, 0, +1"""
        wanted = action - HOLD
        landing = state["altitude"] + wanted
        return jnp.where((state["balloon_resource"] > 0) & (landing >= 0) & (landing < self.n_altitudes), wanted, 0)

    def sample(self, key: PRNGKeyArray, state: dict[str, Any], action: Act) -> Int[Array, ""]:
        """altitude'"""
        return state["altitude"] + self.moved(state, action)


@dataclass(frozen=True)
class Clock(Factor[Act]):
    """hours_elapsed' = hours_elapsed + step_hours, whatever the action"""

    step_hours: float

    @property
    def part(self) -> str:
        return "hours_elapsed"

    def sample(self, key: PRNGKeyArray, state: dict[str, Any], action: Act) -> Float[Array, ""]:
        """hours_elapsed + step_hours"""
        return state["hours_elapsed"] + self.step_hours


@dataclass(frozen=True)
class BalloonTransition(Transition[Act]):
    """state' factor by factor: the wind moves position, the action moves altitude, the clock advances, and an altitude change spends resource"""

    advection: Advection
    ascent: Ascent
    clock: Clock

    state_domain: ProductDomain
    """Every state, in the index order a kernel's arrays follow"""

    @property
    def action_domain(self) -> Domain:
        """Three actions, legal at every state: one the balloon cannot afford moves it nowhere"""
        return DiscreteDomain(3)

    @property
    def factors(self) -> tuple[Factor[Act], ...]:
        """The factors in the order their parts are written"""
        return (self.advection, self.ascent, self.clock)

    def __call__(self, key: PRNGKeyArray, state: dict[str, Any], action: Act) -> dict[str, Any]:
        """One sampled time_step, and the resource the altitude change spent"""
        keys = jax.random.split(key, len(self.factors))
        out = {f.part: f.sample(k, state, action) for f, k in zip(self.factors, keys, strict=True)}
        spent = jnp.abs(self.ascent.moved(state, action))
        return {**out, "balloon_resource": state["balloon_resource"] - spent, "field": state["field"]}

    def wind_step(self, hypothesis: Function) -> tuple[Int[Array, "pos_alt 4"], Float[Array, "pos_alt 4"]]:
        """From every state, the four grid points one time_step under `hypothesis` may land on as flat indices, and their probabilities.

        Both (pos_alt, 4), altitude varying fastest.
        """
        grid, n_alt = self.advection.grid, self.ascent.n_altitudes
        under = {
            "position": jnp.repeat(grid.elements(), n_alt, axis=0),  # (pos_alt, 2), altitude varying fastest
            "altitude": jnp.tile(jnp.arange(n_alt), grid.size()),
            "field": hypothesis,
        }
        return grid.landing_grid_points(self.advection.carried_to(under))

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
    """The law of one time_step at every state, as four reachable grid points and the altitude the action lands on.

    state = (position, altitude)                    indexed pos * n_alt + altitude

    P(state' | state, action) = prob[state, k]      state' = (next_pos[state, k], altitude_at[action, altitude])
                              = 0                   otherwise
    """

    domain: Domain = dataclasses.field(metadata=dict(static=True))
    """What `values` is indexed by, in its own index order"""

    next_pos: Int[Array, "pos_alt 4"]
    """The four grid points the wind may carry the balloon to, as flat indices"""

    prob: Float[Array, "pos_alt 4"]
    """P of each, summing to one along the last axis"""

    altitude_at: Int[Array, "actions alt"]
    """altitude' under each action from each altitude"""

    def expectation(self, values: Float[Array, " states"], action: Act) -> Float[Array, " states"]:
        """out[state] = sum over k in 0..3 of prob[state, k] * values[next_pos[state, k], altitude_at[action, altitude]].
        Time O(states * 4), memory O(states * 4)
        """
        n_alt = self.altitude_at.shape[1]
        table = values.reshape(-1, n_alt)  # (pos, alt)
        landed = table[:, self.altitude_at[action]]  # (pos, alt): altitude' applied
        at = self.next_pos.reshape(-1, n_alt, 4)  # (pos, alt, 4): the grid points the wind may reach
        alt = jnp.arange(n_alt)[None, :, None]  # (1, alt, 1), pairing each grid point with the altitude it was read at
        reached = landed[at, alt]  # (pos, alt, 4)
        return jnp.einsum("pak,pak->pa", self.prob.reshape(-1, n_alt, 4), reached).reshape(-1)

    def sample(self, key: PRNGKeyArray, index: Int[Array, ""], action: Act) -> Int[Array, ""]:
        """One draw from P(. | state, action) at `index`: one of the four grid points by `prob`, then altitude' read off"""
        n_alt = self.altitude_at.shape[1]
        altitude = index % n_alt
        landed = self.next_pos[index, jax.random.categorical(key, jnp.log(self.prob[index]))]
        return landed * n_alt + self.altitude_at[action, altitude]

    def step_cost(self, action: Act) -> Float[Array, " states"]:
        """The resource one time_step spends: |altitude' - altitude|"""
        n_alt = self.altitude_at.shape[1]
        per_altitude = jnp.abs(self.altitude_at[action] - jnp.arange(n_alt)).astype(jnp.float32)  # (alt,)
        return jnp.tile(per_altitude, self.next_pos.shape[0] // n_alt)  # (states,), altitude varying fastest
