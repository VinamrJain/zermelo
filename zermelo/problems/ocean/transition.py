"""One time_step of a boat: it sails a heading through the water, the ocean current carries the water, and it lands on a grid point nearby

state = (position, field)
    position    (lat, lon) degrees of the grid point the boat is at
    field       (W, U) the ocean current and the wind: field(position) = (W_u, W_v, U_u, U_v) in m/s, u eastward and v northward
action in {0, ..., n_headings}
    k < n_headings     sail at heading 2 pi k / n_headings, counted from east towards north
    k = n_headings     drift: no way through the water
step_hours             hours in one time_step

    through_water   = polar speed at the heading, under U(position), times the heading's unit vector     m/s
    carried_to      = position moved by step_hours * 3.6 * (through_water + W(position)) km on the tangent plane
    position'       ~ one of the four grid points around carried_to, with probabilities whose mean position is carried_to
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from zermelo.interface import DiscreteDomain, Domain, Function, ProductDomain, Transition, TransitionKernel
from zermelo.problems.ocean.field import OCEAN_CURRENT, WIND
from zermelo.problems.ocean.grid import PlaneGrid

type Act = Int[Array, ""]
"""A heading index below n_headings, or n_headings for drifting"""


@dataclass(frozen=True)
class Polar:
    """Speed through the water at a heading, from the wind blowing there:

        theta        = angle between the heading and the direction the wind comes from, 0 straight into it, in degrees
        g(theta)     = 0                                                                 theta < no_go_degrees
                     = ((theta - no_go_degrees) / (peak_degrees - no_go_degrees)) ** rise_exponent     no_go_degrees <= theta < peak_degrees
                     = 1 - (1 - downwind_factor) * ((theta - peak_degrees) / (180 - peak_degrees)) ** 2    theta >= peak_degrees
        wind_factor  = 1 - exp(-||U|| / wind_scale_ms)                                   1 where wind_scale_ms = 0
        speed        = top_speed_ms * g(theta) * wind_factor                             m/s

    top_speed_ms with no_go_degrees = 0, rise_exponent = 0, downwind_factor = 1 and wind_scale_ms = 0 is a motor boat
    """

    top_speed_ms: float
    no_go_degrees: float
    peak_degrees: float
    rise_exponent: float
    downwind_factor: float
    wind_scale_ms: float

    def speed(self, heading: Float[Array, "*batch 2"], wind: Float[Array, "*batch 2"]) -> Float[Array, "*batch"]:
        """`speed` at each (heading, wind), heading a unit vector (east, north) and wind (u, v) in m/s"""
        wind_speed = jnp.linalg.norm(wind, axis=-1)
        # the direction the wind comes from is -U; still air comes from nowhere, and the cosine is set to 0 then
        cosine = -jnp.sum(heading * wind, axis=-1) / jnp.maximum(wind_speed, 1e-12)
        theta = jnp.degrees(jnp.arccos(jnp.clip(cosine, -1.0, 1.0)))
        rise = jnp.clip((theta - self.no_go_degrees) / max(self.peak_degrees - self.no_go_degrees, 1e-12), 0.0, 1.0) ** self.rise_exponent
        fall = 1.0 - (1.0 - self.downwind_factor) * ((theta - self.peak_degrees) / max(180.0 - self.peak_degrees, 1e-12)) ** 2
        wind_factor = 1.0 - jnp.exp(-wind_speed / self.wind_scale_ms) if self.wind_scale_ms > 0 else jnp.ones_like(wind_speed)
        return self.top_speed_ms * jnp.where(theta < self.peak_degrees, rise, fall) * wind_factor


@dataclass(frozen=True)
class OceanTransition(Transition[Act]):
    """The ocean current carries the boat, the heading moves it through the water under the wind, and it lands on a grid point nearby"""

    grid: PlaneGrid
    polar: Polar
    n_headings: int
    step_hours: float

    state_domain: ProductDomain
    """Every state, in the index order a kernel's arrays follow"""

    @property
    def action_domain(self) -> Domain:
        """n_headings headings and drifting, legal at every state"""
        return DiscreteDomain(self.n_headings + 1)

    @property
    def headings(self) -> Float[Array, "actions 2"]:
        """The unit vector (east, north) of each heading, and the zero vector for drifting"""
        angle = 2 * math.pi * jnp.arange(self.n_headings) / self.n_headings
        return jnp.concatenate([jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1), jnp.zeros((1, 2))])

    def through_water(self, wind: Float[Array, "*batch 2"], action: Int[Array, "*batch"]) -> Float[Array, "*batch 2"]:
        """The boat's velocity through the water under `action` in `wind`, (east, north) m/s"""
        heading = self.headings[action]  # (*batch, 2)
        return self.polar.speed(heading, wind)[..., None] * heading

    def carried_to(self, state: dict[str, Any], action: Int[Array, "*batch"]) -> Float[Array, "*batch 2"]:
        """`carried_to` as (lat, lon) degrees; 3.6 turns m/s * hours into km"""
        field = state["field"]({"position": state["position"]})  # (*batch, 4): the ocean current, then the wind
        velocity = self.through_water(field[..., WIND], action) + field[..., OCEAN_CURRENT]  # (*batch, 2)
        return self.grid.displaced(state["position"], velocity * self.step_hours * 3.6)

    def __call__(self, key: PRNGKeyArray, state: dict[str, Any], action: Act) -> dict[str, Any]:
        """One sampled time_step: position moves, every other part stays"""
        grid_point_index, probability = self.grid.landing_grid_points(self.carried_to(state, action))  # (4,) each
        drawn = jax.random.categorical(key, jnp.log(probability))
        return {**state, "position": self.grid.from_index(grid_point_index[drawn])}

    def kernel(self, hypothesis: Function) -> "OceanKernel":
        """The law at every state under every action, with `hypothesis` in place of the field an episode drew"""
        under = {"position": self.grid.elements(), "field": hypothesis}  # (pos, 4)
        actions = jnp.arange(self.n_headings + 1)
        next_pos, prob = jax.vmap(
            lambda a: self.grid.landing_grid_points(self.carried_to(under, jnp.broadcast_to(a, (self.grid.size(),))))
        )(actions)
        return OceanKernel(self.state_domain, next_pos, prob)


@register_dataclass
@dataclass(frozen=True)
class OceanKernel(TransitionKernel[Act]):
    """The law of one time_step at every state, as four reachable grid points per action

    P(state' | state, action) = prob[action, state, k]      state' = next_pos[action, state, k]
                              = 0                           otherwise
    """

    domain: Domain = dataclasses.field(metadata=dict(static=True))
    """What `values` is indexed by, in its own index order"""

    next_pos: Int[Array, "actions pos 4"]
    """The four grid points the boat may land on, as flat indices"""

    prob: Float[Array, "actions pos 4"]
    """P of each, summing to one along the last axis"""

    def expectation(self, values: Float[Array, " pos"], action: Act) -> Float[Array, " pos"]:
        """out[state] = sum over k in 0..3 of prob[action, state, k] * values[next_pos[action, state, k]]. Time O(pos * 4)"""
        return jnp.sum(self.prob[action] * values[self.next_pos[action]], axis=-1)

    def sample(self, key: PRNGKeyArray, index: Int[Array, ""], action: Act) -> Int[Array, ""]:
        """One draw from P(. | state, action) at `index`"""
        return self.next_pos[action, index, jax.random.categorical(key, jnp.log(self.prob[action, index]))]
