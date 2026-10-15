"""pi[f](z, x): the action at state z steering to target x under a field f, and what following it takes from z

P(z' | z, a) the transition under f, c(z, a) >= 0 a step's own cost, mu its weight, rho the goal radius, K the value iterations, L the replan length:

    v_0 = 0,   v_{k+1}(z, x) = 0 if |z - x| <= rho else min over a of [ 1 + mu c(z, a) + sum over z' of P(z' | z, a) v_k(z', x) ]
    pi(z, x)  = argmin over a of [ 1 + mu c(z, a) + sum over z' of P(z' | z, a) v_K(z', x) ]
    S(z, x)   = expected steps from z to x, C(z, x) = expected cost spent, both under pi and truncated at K, so v_K = S + mu C
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
from zermelo.methods.path_eui.belief import coordinates


def index_in_actions(actions: Int[Array, " n_actions"], action: Int[Array, "*batch"]) -> Int[Array, "*batch"]:
    """Where each action's value sits in `actions`, the axis a table stacked over `actions` is indexed by"""
    return jnp.argmax(actions == action[..., None], axis=-1)


@register_dataclass
@dataclass(frozen=True)
class Policy:
    """`pi(., x)` for every target `x`, with `S(z, x)` and `C(z, x)` at the one state `z` planned at"""

    action: Int[Array, "n_states n_targets"]
    """`pi(z, x)`, the action's value"""

    expected_path_length: Float[Array, " n_targets"]
    """`S(z, x)`: 0 within `rho` of `x`, `K` where `pi` never arrives"""

    expected_cost: Float[Array, " n_targets"]
    """`C(z, x)`: 0 where the transition charges none"""


type PolicyTables = tuple[Int[Array, "n_states n_targets"], Float[Array, "n_states n_targets"], Float[Array, "n_states n_targets"]]
"""`(pi, S, C)` over every state"""


@dataclass(frozen=True)
class Planner(ABC):
    """`solve_shortest_paths` gives `(pi, S, C)`, `policy_to_targets` gives it `targets_per_solve` targets at a time, `roll_paths` samples paths under `pi`"""

    value_iterations: int | None
    """`K`: rounds of the recursion per solve_shortest_paths; `None` is the steps left in the episode, set by the agent"""

    replan_length: int
    """`L`: the replan length, steps per path and moves per target"""

    only_reachable: bool
    """Choose only among `x` with `H-hat(z_h, x) < L` when True"""

    goal_radius: float
    """`rho`: `|z - x| <= rho` is arrival, on embedded coordinates"""

    targets_per_solve: int | None
    """Targets per solve_shortest_paths, `None` all at once"""

    cost_weight: float
    """`mu`"""

    @property
    def rounds(self) -> int:
        """`K`, raising while it is still the steps left"""
        if self.value_iterations is None:
            raise ValueError("value_iterations is None: the agent sets it to the steps left before planning")
        return self.value_iterations

    @property
    def cost_budget(self) -> float:
        """`B = L (1 + mu)`, the largest `S + mu C`"""
        return self.replan_length * (1.0 + self.cost_weight)

    def step_cost(self, kernel: TransitionKernel[Int[Array, ""]], action: Int[Array, ""]) -> Float[Array, " n_states"] | float:
        """`c(., a)` over every state, 0 where the kernel charges none"""
        charged = kernel.step_cost(action)
        return 0.0 if charged is None else charged

    @abstractmethod
    def solve_shortest_paths(
        self, kernel: TransitionKernel[Int[Array, ""]], targets: Int[Array, " n_targets"], actions: Int[Array, " n_actions"]
    ) -> PolicyTables:
        """`(kernel, targets, actions) -> (pi, S, C)` over every state"""

    def in_goal_set(self, domain: Domain, targets: Int[Array, " n_targets"]) -> Bool[Array, "n_states n_targets"]:
        """`|z - x| <= rho` for every state `z` against every target `x`"""
        return self._squared_distance(domain, targets) <= self.goal_radius**2

    def _squared_distance(self, domain: Domain, targets: Int[Array, " n_targets"]) -> Float[Array, "n_states n_targets"]:
        """`|z - x|^2` on embedded coordinates, every state `z` against every target `x`"""
        state_coords = coordinates(domain)  # (n_states, k)
        squared_norm = jnp.sum(state_coords**2, axis=-1)  # (n_states,)
        return jnp.maximum(squared_norm[:, None] + squared_norm[targets][None, :] - 2 * state_coords @ state_coords[targets].T, 0.0)

    def _one_step_lookahead(
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

    def policy_to_targets(
        self,
        kernel: TransitionKernel[Int[Array, ""]],
        targets: Int[Array, " n_targets"],
        actions: Int[Array, " n_actions"],
        planned_at: Int[Array, ""],
    ) -> Policy:
        """`solve_shortest_paths` over `targets_per_solve` targets at a time, `S` and `C` kept at `planned_at` only"""
        chunk = targets.shape[0] if self.targets_per_solve is None else self.targets_per_solve
        if targets.shape[0] <= chunk:
            return self._solve_jitted(kernel, targets, actions, planned_at)
        # every target is its own problem, so a block of columns is the whole answer for those columns
        blocks = [
            self._solve_jitted(kernel, targets[first : first + chunk], actions, planned_at) for first in range(0, targets.shape[0], chunk)
        ]
        return jax.tree.map(lambda *columns: jnp.concatenate(columns, axis=-1), *blocks)

    @partial(jax.jit, static_argnums=0)
    def _solve_jitted(
        self,
        kernel: TransitionKernel[Int[Array, ""]],
        targets: Int[Array, " n_targets"],
        actions: Int[Array, " n_actions"],
        planned_at: Int[Array, ""],
    ) -> Policy:
        """`solve_shortest_paths`, jitted: `pi` as int8, `S` and `C` at `planned_at`"""
        action_table, expected_path_length, expected_cost = self.solve_shortest_paths(kernel, targets, actions)
        return Policy(action_table.astype(jnp.int8), expected_path_length[planned_at], expected_cost[planned_at])

    def expected_cost_of_policy(
        self,
        kernel: TransitionKernel[Int[Array, ""]],
        action_table: Int[Array, "n_states n_targets"],
        targets: Int[Array, " n_targets"],
        actions: Int[Array, " n_actions"],
        per_step: Callable[[Int[Array, ""]], Float[Array, " n_states"] | float],
    ) -> Float[Array, "n_states n_targets"]:
        """h_K for h_0 = 0, h_{k+1}(z, x) = 0 if |z - x| <= rho else per_step(pi(z, x))[z] + sum over z' of P(z' | z, pi(z, x)) h_k(z', x)"""
        in_goal_set = self.in_goal_set(kernel.domain, targets)
        state_rows, target_columns = jnp.arange(in_goal_set.shape[0])[:, None], jnp.arange(in_goal_set.shape[1])[None, :]
        action_rows = index_in_actions(actions, action_table)  # (n_states, n_targets): the action's position on the backup's action axis
        return lax.fori_loop(
            0,
            self.rounds,
            lambda _, accumulated: jnp.where(
                in_goal_set, 0.0, self._one_step_lookahead(kernel, accumulated, actions, per_step)[action_rows, state_rows, target_columns]
            ),
            jnp.zeros(in_goal_set.shape),
        )

    @partial(jax.jit, static_argnums=0)
    def roll_paths(
        self,
        path_keys: PRNGKeyArray,
        kernel: TransitionKernel[Int[Array, ""]],
        policy: Policy,
        targets: Int[Array, " n_targets"],
        start: Int[Array, ""],
    ) -> tuple[Int[Array, "n_paths_per_field n_targets replan_length"], Bool[Array, "n_paths_per_field n_targets replan_length"]]:
        """z_0 = start, z_{t+1} ~ P(. | z_t, pi(z_t, x)) while |z_t - x| > rho, L steps, one path per key per target: the states, and which steps were before arrival"""
        state_coords = coordinates(kernel.domain)  # (n_states, k)
        squared_norm, target_coords = jnp.sum(state_coords**2, axis=-1), state_coords[targets]  # (n_states,), (n_targets, k)
        target_columns = jnp.arange(policy.action.shape[1])  # (n_targets,): which column of the action table each path reads

        def in_goal_set(states_on_path: Int[Array, " n_targets"]) -> Bool[Array, " n_targets"]:
            """`|z_t - x| <= rho` per path"""
            cross = jnp.sum(state_coords[states_on_path] * target_coords, axis=-1)
            return jnp.maximum(squared_norm[states_on_path] + squared_norm[targets] - 2 * cross, 0.0) <= self.goal_radius**2

        sample_next = jax.vmap(kernel.sample, in_axes=(None, 0, 0))  # (key, state per target, action per target) -> next state per target

        def one_path(key: PRNGKeyArray) -> tuple[Int[Array, "n_targets replan_length"], Bool[Array, "n_targets replan_length"]]:
            """One path to every target"""
            states_on_path = jnp.full(target_columns.shape, start)  # (n_targets,): z_0 = start
            travelling = jnp.ones(target_columns.shape, bool)  # (n_targets,): not in_goal_set(z_t)
            states, before_arrival = [], []
            for step_key in jax.random.split(key, self.replan_length):
                # z_{t+1} = P-draw where travelling, else z_t
                states_on_path = jnp.where(
                    travelling, sample_next(step_key, states_on_path, policy.action[states_on_path, target_columns]), states_on_path
                )
                states.append(states_on_path)
                before_arrival.append(travelling)
                travelling = travelling & ~in_goal_set(states_on_path)  # travelling and not in_goal_set(z_{t+1})
            return jnp.stack(states, axis=-1), jnp.stack(before_arrival, axis=-1)  # both (n_targets, replan_length)

        return jax.vmap(one_path)(path_keys)  # both (n_paths_per_field, n_targets, replan_length)


@dataclass(frozen=True)
class ValueIteration(Planner):
    """The module's recursion, K rounds, with a_k(z, x) the round's minimiser:

    S_{k+1}(z, x) = 1 + sum over z' of P(z' | z, a_k) S_k(z', x),   C_{k+1}(z, x) = c(z, a_k) + sum over z' of P(z' | z, a_k) C_k(z', x)
    """

    def solve_shortest_paths(
        self, kernel: TransitionKernel[Int[Array, ""]], targets: Int[Array, " n_targets"], actions: Int[Array, " n_actions"]
    ) -> PolicyTables:
        in_goal_set = self.in_goal_set(kernel.domain, targets)
        state_rows, target_columns = jnp.arange(in_goal_set.shape[0])[:, None], jnp.arange(in_goal_set.shape[1])[None, :]
        charged = partial(self.step_cost, kernel)  # step_cost(s, a)
        planned_cost = lambda a: 1.0 + self.cost_weight * charged(a)  # noqa: E731

        def backup(
            _: Int[Array, ""], tables: tuple[Float[Array, "n_states n_targets"], ...]
        ) -> tuple[Float[Array, "n_states n_targets"], ...]:
            value, expected_path_length, expected_cost = tables
            backed_up = self._one_step_lookahead(kernel, value, actions, planned_cost)  # (n_actions, n_states, n_targets)
            minimising = jnp.argmin(backed_up, axis=0)  # (n_states, n_targets): the position of this round's minimising action
            return (
                jnp.where(in_goal_set, 0.0, backed_up[minimising, state_rows, target_columns]),
                jnp.where(
                    in_goal_set,
                    0.0,
                    self._one_step_lookahead(kernel, expected_path_length, actions, lambda a: 1.0)[minimising, state_rows, target_columns],
                ),
                jnp.where(
                    in_goal_set,
                    0.0,
                    self._one_step_lookahead(kernel, expected_cost, actions, charged)[minimising, state_rows, target_columns],
                ),
            )

        zeros = jnp.zeros(in_goal_set.shape)
        value, expected_path_length, expected_cost = lax.fori_loop(0, self.rounds, backup, (zeros, zeros, zeros))
        action_table = actions[jnp.argmin(self._one_step_lookahead(kernel, value, actions, planned_cost), axis=0)]
        return action_table, expected_path_length, expected_cost


@dataclass(frozen=True)
class Greedy(Planner):
    """pi(z, x) = argmin over a of [ 1 + sum over z' of P(z' | z, a) |z' - x| ], then S and C by `evaluate`"""

    def solve_shortest_paths(
        self, kernel: TransitionKernel[Int[Array, ""]], targets: Int[Array, " n_targets"], actions: Int[Array, " n_actions"]
    ) -> PolicyTables:
        distance = jnp.sqrt(self._squared_distance(kernel.domain, targets))
        action_table = actions[jnp.argmin(self._one_step_lookahead(kernel, distance, actions, lambda a: 1.0), axis=0)]
        expected_path_length = self.expected_cost_of_policy(kernel, action_table, targets, actions, lambda a: 1.0)
        expected_cost = self.expected_cost_of_policy(kernel, action_table, targets, actions, partial(self.step_cost, kernel))
        return action_table, expected_path_length, expected_cost
