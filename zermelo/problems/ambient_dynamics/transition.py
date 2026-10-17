"""The one-step transition law: the field displaces the ambient part, the actor steers the controllable one"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from zermelo.interface import DiscreteDomain, Domain, Function, Transition, TransitionKernel
from zermelo.problems.ambient_dynamics.grid import PaddedGridDomain, ambient_positions

type Act = Int[Array, ""]
"""One flat index in [0, choices_per_axis ** controllable_axes): one choice per controllable axis"""


def _clipped_normal_cdf(d: Float[Array, "*batch"], mean: Float[Array, "*batch"], sigma: float, bound: float) -> Float[Array, "*batch"]:
    """P(D <= d) for D = clip(N(mean, sigma^2), +-bound): the normal CDF held at 0 below -bound and at 1 from +bound."""
    return jnp.where(d <= -bound, 0.0, jnp.where(d >= bound, 1.0, norm.cdf(d, loc=mean, scale=max(sigma, 1e-12))))


def _axis_law(
    grid: PaddedGridDomain,
    start: Int[Array, " n_starts"],
    mean: Float[Array, " n_starts"],
    sigma: float,
    bound: float,
    suspended: Bool[Array, " n_starts"],
) -> Float[Array, "n_starts cells"]:
    """Where one axis lands from each start cell, as a distribution over that axis' cells.

    suspended:  one cell toward [0, extent], with probability one
    otherwise:  start + o, with probability F((o + 0.5) * cell_spacing) - F((o - 0.5) * cell_spacing),
                for F the CDF of a displacement drawn N(mean, sigma^2) and clipped to +-bound
    """
    width = math.ceil(bound / grid.cell_spacing + 0.5)  # the furthest cell a displacement bounded by +-bound can reach
    offsets = jnp.arange(-width, width + 1)
    edges = (jnp.arange(-width, width + 2) - 0.5) * grid.cell_spacing  # offset o's one-cell bin runs edges[o] .. edges[o + 1]
    below = _clipped_normal_cdf(edges, mean[:, None], sigma, bound)  # (n_starts, 2 * width + 2)
    rows = jnp.arange(start.shape[0])[:, None]  # (n_starts, 1), pairing each start with its own landing cells
    land = grid.cell_of(grid.project(grid.coords_of(start[:, None] + offsets)))  # an offset off the grid folds back onto it
    displaced = jnp.zeros((start.shape[0], grid.cells)).at[rows, land].add(below[:, 1:] - below[:, :-1])
    at = grid.coords_of(start)  # inward_direction reads coordinates, not cells
    walked = grid.cell_of(grid.project(at + grid.cell_spacing * grid.inward_direction(at)))  # an axis already inside stays put
    return jnp.where(suspended[:, None], jax.nn.one_hot(walked, grid.cells), displaced)


def _inverse_cdf(key: PRNGKeyArray, cumulative: Float[Array, " cells"]) -> Int[Array, ""]:
    """One draw from the law whose running mass is `cumulative`: c = min(#{j : F_j < u}, cells - 1), u ~ U(0, 1)"""
    return jnp.minimum(jnp.sum(cumulative < jax.random.uniform(key), axis=-1), cumulative.shape[0] - 1)


def _successor_gap(values: Float[Array, "..."], axis: int) -> Float[Array, "..."]:
    """v_w - v_{w+1} along `axis`, taking v_W = 0. Contracts a running mass:
    sum over w of p_w v_w = sum over w of F_w (v_w - v_{w+1})
    """
    return values - jnp.take(values, jnp.arange(1, values.shape[axis] + 1), axis=axis, mode="fill", fill_value=0.0)


@dataclass(frozen=True)
class AmbientTransition(Transition[Act]):
    """The one-step law: two factors see the whole position z = (y, q) and each writes a disjoint part of it.

    The field is read at the whole z, so steering q navigates y
    """

    ambient: PaddedGridDomain
    controllable: PaddedGridDomain
    sigma_flow: float
    sigma_act: float

    choices_per_axis: int
    """Choices per controllable axis. Odd: the middle choice stands still"""

    drift_bound_cells: float
    """How far the field may carry the actor in one step, in cells and no more than the ambient pad"""

    control_step_cells: float
    """What one choice of steering moves, in cells"""

    control_bound_cells: float
    """Largest displacement the actor may steer in one step, in cells and no more than the controllable pad"""

    def __post_init__(self) -> None:
        if self.choices_per_axis % 2 != 1:
            raise ValueError(f"choices_per_axis must be odd so that standing still is a choice, got {self.choices_per_axis}")
        if self.drift_bound_cells > self.ambient.pad:
            raise ValueError(
                f"a {self.drift_bound_cells}-cell displacement overshoots a {self.ambient.pad}-cell pad, so the domain is not closed"
            )
        if self.control_bound_cells > self.controllable.pad:
            raise ValueError(
                f"a {self.control_bound_cells}-cell steering step overshoots a {self.controllable.pad}-cell pad, so the domain is not closed"
            )

    @property
    def drift_bound(self) -> float:
        """The clip on one step of field displacement, in coordinates: drift_bound_cells * cell_spacing"""
        return self.drift_bound_cells * self.ambient.cell_spacing

    @property
    def control_step(self) -> float:
        """What one choice of steering moves, in coordinates: control_step_cells * cell_spacing"""
        return self.control_step_cells * self.controllable.cell_spacing

    @property
    def control_bound(self) -> float:
        """The clip on one step of steering, in coordinates: control_bound_cells * cell_spacing"""
        return self.control_bound_cells * self.controllable.cell_spacing

    @property
    def action_domain(self) -> Domain:
        """The flat actions: choices_per_axis ** controllable_axes of them"""
        return DiscreteDomain(self.choices_per_axis**self.controllable.n_axes)

    @property
    def offsets(self) -> Int[Array, " choices_per_axis"]:
        """What each choice moves, in signed cells: -(choices_per_axis // 2) .. +(choices_per_axis // 2), the middle one standing still"""
        return jnp.arange(self.choices_per_axis) - self.choices_per_axis // 2

    def choices(self, action: Int[Array, "*batch"]) -> Int[Array, "*batch controllable_axes"]:
        """Which choice `action` gives axis j: (action // choices_per_axis ** (controllable_axes - 1 - j)) mod choices_per_axis"""
        radix = self.choices_per_axis ** jnp.arange(self.controllable.n_axes - 1, -1, -1)  # row-major: the last axis varies fastest
        return action[..., None] // radix % self.choices_per_axis

    def drift(self, key: PRNGKeyArray, z: dict[str, Any], field: Function) -> Float[Array, " ambient_axes"]:
        """One draw of `p^field`, the ambient displacement dy, with eps standard normal:

        dy = clip(field(z) + sigma_flow * eps, +-drift_bound)   every ambient axis inside [0, extent]
           = one cell toward [0, extent]                        any ambient axis outside it
        """
        y, eps = z["ambient"], jax.random.normal(key, (self.ambient.n_axes,))
        return jnp.where(
            self.ambient.in_pad(y)[..., None],  # one flag for the whole part: an axis still inside is frozen, not driven
            self.ambient.cell_spacing * self.ambient.inward_direction(y),  # one cell per step back
            jnp.clip(field(z) + self.sigma_flow * eps, -self.drift_bound, self.drift_bound),
        )

    def control(self, key: PRNGKeyArray, z: dict[str, Any], action: Act) -> Float[Array, " controllable_axes"]:
        """One draw of `p^actor`, the controllable displacement dq, with eta standard normal and `o` the
        signed cells `action` gives each axis:

        dq = clip(control_step * o + sigma_act * eta, +-control_bound)   an axis inside [0, extent]
           = one cell toward [0, extent]                                 an axis outside it
        """
        q, eta = z["controllable"], jax.random.normal(key, (self.controllable.n_axes,))
        inward = self.controllable.inward_direction(q)  # (controllable_axes,), zero on an axis still inside
        return jnp.where(
            inward != 0,
            self.controllable.cell_spacing * inward,
            jnp.clip(
                self.control_step * self.offsets[self.choices(action)] + self.sigma_act * eta, -self.control_bound, self.control_bound
            ),
        )

    def __call__(self, key: PRNGKeyArray, state: dict[str, Any], action: Act) -> dict[str, Any]:
        """One sampled step: y' = project(y + dy) and q' = project(q + dq), the field carried through unchanged"""
        k_flow, k_act = jax.random.split(key)
        z = {"ambient": state["ambient"], "controllable": state["controllable"]}
        return {
            "ambient": self.ambient.project(z["ambient"] + self.drift(k_flow, z, state["field"])),
            "controllable": self.controllable.project(z["controllable"] + self.control(k_act, z, action)),
            "field": state["field"],
        }

    def kernel(self, hypothesis: Function) -> "AmbientKernel":
        """The same law written down exactly at every position, with `hypothesis` in place of the true field"""
        positions = ambient_positions(self.ambient, self.controllable)  # z = (y, q)
        z = positions.elements()  # {"ambient": (n_states, ambient_axes), "controllable": (n_states, controllable_axes)}
        y, mean = z["ambient"], hypothesis(z)  # (n_states, ambient_axes): the unclipped displacement at every position
        start, frozen = self.ambient.cell_of(y), self.ambient.in_pad(y)  # (n_states,): one flag for the whole ambient part, not per axis
        ambient_law = jnp.stack(  # (ambient_axes, n_states, ambient_cells)
            [
                _axis_law(self.ambient, start[:, i], mean[:, i], self.sigma_flow, self.drift_bound, frozen)
                for i in range(self.ambient.n_axes)
            ]
        )
        cells = self.controllable.cells
        from_cell = jnp.repeat(jnp.arange(cells), self.choices_per_axis)  # (cells * choices,): one row per (cell, choice), choices fastest
        control_law = _axis_law(  # (cells * choices, cells): p(q'_j | q_j, a_j) for one steered axis
            self.controllable,
            from_cell,
            self.control_step * jnp.tile(self.offsets, cells),
            self.sigma_act,
            self.control_bound,
            self.controllable.inward_direction(self.controllable.coords_of(from_cell)) != 0,  # per axis, not per part
        )
        return AmbientKernel(
            positions,
            jnp.cumsum(ambient_law, axis=-1),  # F(y'_i | z), running along the landing axis
            jnp.cumsum(control_law.reshape(cells, self.choices_per_axis, cells), axis=-1),  # F(q'_j | q_j, a_j), likewise
            self.controllable.cell_of(z["controllable"]),
            self.choices(jnp.arange(self.choices_per_axis**self.controllable.n_axes)),
        )


@register_dataclass
@dataclass(frozen=True)
class AmbientKernel(TransitionKernel[Act]):
    """The law at every enumerated position, one factor per axis, published as one operator"""

    domain: Domain = dataclasses.field(metadata=dict(static=True))
    """Every position, in the index order this kernel's arrays follow"""

    f_ambient: Float[Array, "ambient_axes n_states ambient_cells"]
    """F(y'_i | z) per ambient axis: the running mass over where that axis lands from every position"""

    f_control: Float[Array, "controllable_cells choices_per_axis controllable_cells"]
    """F(q'_j | q_j, a_j): the running mass over where one controllable axis lands, per cell and choice"""

    control_cell: Int[Array, "n_states controllable_axes"]
    """Each position's own controllable cells"""

    choice_index: Int[Array, "n_actions controllable_axes"]
    """Every flat action decoded into one choice per controllable axis"""

    def expectation(self, values: Float[Array, " n_states"], action: Act) -> Float[Array, " n_states"]:
        """sum over z' of P(z' | z, a) * values(z'), at every position z. Exact, not sampled"""
        ambient_axes, controllable_axes = self.f_ambient.shape[0], self.control_cell.shape[1]
        ambient_cells, controllable_cells = self.f_ambient.shape[2], self.f_control.shape[0]
        controllable_positions = controllable_cells**controllable_axes  # z = (y, q), and the index counts q fastest
        out = values.reshape((ambient_cells,) * ambient_axes + (controllable_cells,) * controllable_axes)
        for j in range(controllable_axes):
            # out[.., q_j, ..] <- sum over q'_j of F(q'_j | q_j, a_j) * (out[.., q'_j, ..] - out[.., q'_j + 1, ..])
            # the control law is the same at every position
            law = self.f_control[:, self.choice_index[action, j], :]  # (controllable_cells, controllable_cells)
            gap = _successor_gap(out, ambient_axes + j)
            out = jnp.moveaxis(jnp.tensordot(law, gap, axes=(1, ambient_axes + j)), 0, ambient_axes + j)
        out = out.reshape((ambient_cells,) * ambient_axes + (controllable_positions,))
        for i in range(ambient_axes):
            # out[z, ..] <- sum over y'_i of F(y'_i | z) * (out[y'_i, .., q] - out[y'_i + 1, .., q])
            # f_ambient is indexed by the whole z = (y, q), so the einsum shares q
            law = self.f_ambient[i].reshape(-1, controllable_positions, ambient_cells)  # (y, q, y'_i)
            out = (
                jnp.einsum("yqw,w...q->yq...", law, _successor_gap(out, 0))
                if i == 0
                else jnp.einsum("yqw,yq...w->yq...", law, _successor_gap(out, -1))
            )
        return out.reshape(-1)  # (n_states,), in the domain's own index order

    def sample(self, key: PRNGKeyArray, index: Int[Array, ""], action: Act) -> Int[Array, ""]:
        """One draw z' from P(. | z, a) at `index`, as that domain's own flat index"""
        ambient_axes, controllable_axes = self.f_ambient.shape[0], self.control_cell.shape[1]
        ambient_cells, controllable_cells = self.f_ambient.shape[2], self.f_control.shape[0]
        keys = jax.random.split(key, ambient_axes + controllable_axes)
        landing = jnp.asarray(0)  # z' = (... * ambient_cells + y'_last) * controllable_cells + ... + q'_last, as `index_of` counts
        for axis in range(ambient_axes):
            landing = landing * ambient_cells + _inverse_cdf(keys[axis], self.f_ambient[axis, index])
        for axis in range(controllable_axes):
            row = self.f_control[self.control_cell[index, axis], self.choice_index[action, axis]]
            landing = landing * controllable_cells + _inverse_cdf(keys[ambient_axes + axis], row)
        return landing
