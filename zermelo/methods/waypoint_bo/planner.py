"""pi[f](z, x): the act at state z steering to target x under a field f, and what following it takes from z

P(z' | z, a) the transition under f, c(z, a) >= 0 a step's own cost, mu its weight, rho the arrival radius, L the budget:

    v_0 = 0,   v_{k+1}(z, x) = 0 if |z - x| <= rho else min over a of [ 1 + mu c(z, a) + sum over z' of P(z' | z, a) v_k(z', x) ]
    pi(z, x)  = argmin over a of [ 1 + mu c(z, a) + sum over z' of P(z' | z, a) v_L(z', x) ]
    S(z, x)   = expected steps from z to x, C(z, x) = expected cost spent, both under pi and truncated at L, so v_L = S + mu C
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from zermelo.interface import Domain, TransitionKernel
from zermelo.methods.waypoint_bo.belief import coordinates


def position_of(actions: Int[Array, " n_actions"], act: Int[Array, "*batch"]) -> Int[Array, "*batch"]:
    """Where each act's value sits in `actions`, the axis a table stacked over `actions` is indexed by"""
    return jnp.argmax(actions == act[..., None], axis=-1)


@register_dataclass
@dataclass(frozen=True)
class Policy:
    """`pi(., x)` for every target `x`, with `S(z, x)` and `C(z, x)` at the one state `z` planned at"""

    act: Int[Array, "n_states n_targets"]
    """`pi(z, x)`, the act's value"""

    steps_to_target: Float[Array, " n_targets"]
    """`S(z, x)`: 0 within `rho` of `x`, `L` where `pi` never arrives"""

    budget_spent_to_target: Float[Array, " n_targets"]
    """`C(z, x)`: 0 where the transition charges none"""


type PolicyTables = tuple[Int[Array, "n_states n_targets"], Float[Array, "n_states n_targets"], Float[Array, "n_states n_targets"]]
"""`(pi, S, C)` over every state"""


@dataclass(frozen=True)
class Planner(ABC):
    """`solve` gives `(pi, S, C)`, `plan` gives it `target_chunk` targets at a time, `roll` samples walks under `pi`"""

    replan_every: int
    """`L`: backups per solve, steps per walk, moves per waypoint"""

    radius: float
    """`rho`: `|z - x| <= rho` is arrival, on embedded coordinates"""

    target_chunk: int | None
    """Targets per solve, `None` all at once"""

    cost_weight: float
    """`mu`"""

    @property
    def cost_budget(self) -> float:
        """`B = L (1 + mu)`, the largest `S + mu C`"""
        return self.replan_every * (1.0 + self.cost_weight)

    def step_cost(self, kernel: TransitionKernel[Int[Array, ""]], action: Int[Array, ""]) -> Float[Array, " n_states"] | float:
        """`c(., a)` over every state, 0 where the kernel charges none"""
        charged = kernel.step_cost(action)
        return 0.0 if charged is None else charged

    @abstractmethod
    def solve(
        self, kernel: TransitionKernel[Int[Array, ""]], targets: Int[Array, " n_targets"], actions: Int[Array, " n_actions"]
    ) -> PolicyTables:
        """`(kernel, targets, actions) -> (pi, S, C)` over every state"""

    def arrived(self, domain: Domain, targets: Int[Array, " n_targets"]) -> Bool[Array, "n_states n_targets"]:
        """`|z - x| <= rho` for every state `z` against every target `x`"""
        return self._squared_distance(domain, targets) <= self.radius**2

    def _squared_distance(self, domain: Domain, targets: Int[Array, " n_targets"]) -> Float[Array, "n_states n_targets"]:
        """`|z - x|^2` on embedded coordinates, every state `z` against every target `x`"""
        state_coords = coordinates(domain)  # (n_states, k)
        squared_norm = jnp.sum(state_coords**2, axis=-1)  # (n_states,)
        return jnp.maximum(squared_norm[:, None] + squared_norm[targets][None, :] - 2 * state_coords @ state_coords[targets].T, 0.0)

    def _backup(
        self,
        kernel: TransitionKernel[Int[Array, ""]],
        values: Float[Array, "n_states n_targets"],
        actions: Int[Array, " n_actions"],
        per_step: Callable[[Int[Array, ""]], Float[Array, " n_states"] | float],
    ) -> Float[Array, "n_actions n_states n_targets"]:
        """out[a, z, x] = per_step(a)[z] + sum over z' of P(z' | z, a) values[z', x]"""
        expected_next = jax.vmap(kernel.expectation, in_axes=(1, None), out_axes=1)
        # (n_states, 1), or a scalar left as one, onto (n_states, n_targets)
        return jnp.stack([jnp.reshape(per_step(jnp.asarray(a)), (-1, 1)) + expected_next(values, jnp.asarray(a)) for a in actions])

    def plan(
        self,
        kernel: TransitionKernel[Int[Array, ""]],
        targets: Int[Array, " n_targets"],
        actions: Int[Array, " n_actions"],
        planned_at: Int[Array, ""],
    ) -> Policy:
        """`solve` over `target_chunk` targets at a time, `S` and `C` kept at `planned_at` only"""
        chunk = targets.shape[0] if self.target_chunk is None else self.target_chunk
        if targets.shape[0] <= chunk:
            return self._plan_block(kernel, targets, actions, planned_at)
        # every target is its own problem, so a block of columns is the whole answer for those columns
        blocks = [
            self._plan_block(kernel, targets[first : first + chunk], actions, planned_at) for first in range(0, targets.shape[0], chunk)
        ]
        return jax.tree.map(lambda *columns: jnp.concatenate(columns, axis=-1), *blocks)

    @partial(jax.jit, static_argnums=0)
    def _plan_block(
        self,
        kernel: TransitionKernel[Int[Array, ""]],
        targets: Int[Array, " n_targets"],
        actions: Int[Array, " n_actions"],
        planned_at: Int[Array, ""],
    ) -> Policy:
        """`solve`, jitted: `pi` as int8, `S` and `C` at `planned_at`"""
        act_table, steps_to_target, budget_spent_to_target = self.solve(kernel, targets, actions)
        return Policy(act_table.astype(jnp.int8), steps_to_target[planned_at], budget_spent_to_target[planned_at])

    def evaluate(
        self,
        kernel: TransitionKernel[Int[Array, ""]],
        act_table: Int[Array, "n_states n_targets"],
        targets: Int[Array, " n_targets"],
        actions: Int[Array, " n_actions"],
        per_step: Callable[[Int[Array, ""]], Float[Array, " n_states"] | float],
    ) -> Float[Array, "n_states n_targets"]:
        """h_L for h_0 = 0, h_{k+1}(z, x) = 0 if |z - x| <= rho else per_step(pi(z, x))[z] + sum over z' of P(z' | z, pi(z, x)) h_k(z', x)"""
        arrived = self.arrived(kernel.domain, targets)
        state_rows, target_columns = jnp.arange(arrived.shape[0])[:, None], jnp.arange(arrived.shape[1])[None, :]
        act_rows = position_of(actions, act_table)  # (n_states, n_targets): the act's position on the backup's action axis
        return lax.fori_loop(
            0,
            self.replan_every,
            lambda _, accumulated: jnp.where(
                arrived, 0.0, self._backup(kernel, accumulated, actions, per_step)[act_rows, state_rows, target_columns]
            ),
            jnp.zeros(arrived.shape),
        )

    @partial(jax.jit, static_argnums=0)
    def roll(
        self,
        walk_keys: PRNGKeyArray,
        kernel: TransitionKernel[Int[Array, ""]],
        policy: Policy,
        targets: Int[Array, " n_targets"],
        start: Int[Array, ""],
    ) -> tuple[Int[Array, "n_walks n_targets replan_every"], Bool[Array, "n_walks n_targets replan_every"]]:
        """z_0 = start, z_{t+1} ~ P(. | z_t, pi(z_t, x)) while |z_t - x| > rho, L steps, one walk per key per target: the states, and which steps moved"""
        state_coords = coordinates(kernel.domain)  # (n_states, k)
        squared_norm, target_coords = jnp.sum(state_coords**2, axis=-1), state_coords[targets]  # (n_states,), (n_targets, k)
        target_columns = jnp.arange(policy.act.shape[1])  # (n_targets,): which column of the act table each walk reads

        def arrived(walk_states: Int[Array, " n_targets"]) -> Bool[Array, " n_targets"]:
            """`|z_t - x| <= rho` per walk"""
            cross = jnp.sum(state_coords[walk_states] * target_coords, axis=-1)
            return jnp.maximum(squared_norm[walk_states] + squared_norm[targets] - 2 * cross, 0.0) <= self.radius**2

        sample_next = jax.vmap(kernel.sample, in_axes=(None, 0, 0))  # (key, state per target, act per target) -> next state per target

        def one_walk(key: PRNGKeyArray) -> tuple[Int[Array, "n_targets replan_every"], Bool[Array, "n_targets replan_every"]]:
            """One walk to every target"""
            walk_states = jnp.full(target_columns.shape, start)  # (n_targets,): z_0 = start
            travelling = jnp.ones(target_columns.shape, bool)  # (n_targets,): not arrived(z_t)
            path, moved = [], []
            for step_key in jax.random.split(key, self.replan_every):
                # z_{t+1} = P-draw where travelling, else z_t
                walk_states = jnp.where(
                    travelling, sample_next(step_key, walk_states, policy.act[walk_states, target_columns]), walk_states
                )
                path.append(walk_states)
                moved.append(travelling)
                travelling = travelling & ~arrived(walk_states)  # travelling and not arrived(z_{t+1})
            return jnp.stack(path, axis=-1), jnp.stack(moved, axis=-1)  # both (n_targets, replan_every)

        return jax.vmap(one_walk)(walk_keys)  # both (n_walks, n_targets, replan_every)


@dataclass(frozen=True)
class ValueIteration(Planner):
    """The module's recursion, L rounds, with a_k(z, x) the round's minimiser:

    S_{k+1}(z, x) = 1 + sum over z' of P(z' | z, a_k) S_k(z', x),   C_{k+1}(z, x) = c(z, a_k) + sum over z' of P(z' | z, a_k) C_k(z', x)
    """

    def solve(
        self, kernel: TransitionKernel[Int[Array, ""]], targets: Int[Array, " n_targets"], actions: Int[Array, " n_actions"]
    ) -> PolicyTables:
        arrived = self.arrived(kernel.domain, targets)
        state_rows, target_columns = jnp.arange(arrived.shape[0])[:, None], jnp.arange(arrived.shape[1])[None, :]
        charged = partial(self.step_cost, kernel)  # step_cost(s, a)
        planned_cost = lambda a: 1.0 + self.cost_weight * charged(a)  # noqa: E731

        def backup(
            _: Int[Array, ""], tables: tuple[Float[Array, "n_states n_targets"], ...]
        ) -> tuple[Float[Array, "n_states n_targets"], ...]:
            value, steps_to_target, budget_spent_to_target = tables
            backed_up = self._backup(kernel, value, actions, planned_cost)  # (n_actions, n_states, n_targets)
            minimising = jnp.argmin(backed_up, axis=0)  # (n_states, n_targets): the position of this round's minimising act
            return (
                jnp.where(arrived, 0.0, backed_up[minimising, state_rows, target_columns]),
                jnp.where(
                    arrived, 0.0, self._backup(kernel, steps_to_target, actions, lambda a: 1.0)[minimising, state_rows, target_columns]
                ),
                jnp.where(
                    arrived, 0.0, self._backup(kernel, budget_spent_to_target, actions, charged)[minimising, state_rows, target_columns]
                ),
            )

        zeros = jnp.zeros(arrived.shape)
        value, steps_to_target, budget_spent_to_target = lax.fori_loop(0, self.replan_every, backup, (zeros, zeros, zeros))
        act_table = actions[jnp.argmin(self._backup(kernel, value, actions, planned_cost), axis=0)]
        return act_table, steps_to_target, budget_spent_to_target


@dataclass(frozen=True)
class Greedy(Planner):
    """pi(z, x) = argmin over a of [ 1 + sum over z' of P(z' | z, a) |z' - x| ], then S and C by `evaluate`"""

    def solve(
        self, kernel: TransitionKernel[Int[Array, ""]], targets: Int[Array, " n_targets"], actions: Int[Array, " n_actions"]
    ) -> PolicyTables:
        distance = jnp.sqrt(self._squared_distance(kernel.domain, targets))
        act_table = actions[jnp.argmin(self._backup(kernel, distance, actions, lambda a: 1.0), axis=0)]
        steps_to_target = self.evaluate(kernel, act_table, targets, actions, lambda a: 1.0)
        budget_spent_to_target = self.evaluate(kernel, act_table, targets, actions, partial(self.step_cost, kernel))
        return act_table, steps_to_target, budget_spent_to_target
