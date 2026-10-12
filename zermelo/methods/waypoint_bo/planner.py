"""How the actor gets from where it stands to a waypoint, under a guessed field, and what that takes in steps and in budget"""

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
    """`pi_n`: the act to take at every state to reach each target, and what following it takes from the state planned at"""

    act: Int[Array, "n_states n_targets"]
    """The action's value, as the transition takes it"""

    steps_to_target: Float[Array, " n_targets"]
    """`S(z, x)`: expected time_steps from the state planned at to `x` under the policy, zero on arrival and truncated at `L`"""

    budget_spent_to_target: Float[Array, " n_targets"]
    """`C(z, x)`: expected step cost spent from the state planned at to `x` under the policy, zero where the transition charges none"""


type Tables = tuple[Int[Array, "n_states n_targets"], Float[Array, "n_states n_targets"], Float[Array, "n_states n_targets"]]
"""A solve's act, `S` and `C` over every state"""


@dataclass(frozen=True)
class Planner(ABC):
    """Turns a kernel and a set of targets into an act table with its step and budget tables, and rolls walks under it"""

    replan_every: int
    """`L`: a waypoint is chosen anew after this many time_steps, a plan is solved with as many backups, and a walk is rolled over at most as many"""

    radius: float
    """`rho`: how close to a target counts as arrived, on embedded coordinates"""

    target_chunk: int | None
    """How many targets one solve handles, `None` taking every target at once"""

    cost_weight: float
    """`mu`: what one unit of a step's own cost is worth in steps to the plan. Zero plans on steps alone"""

    @property
    def cost_budget(self) -> float:
        """`B = L * (1 + mu)`: the most a plan's cost-to-go `S + mu C` can read"""
        return self.replan_every * (1.0 + self.cost_weight)

    def step_cost(self, kernel: TransitionKernel[Int[Array, ""]], action: Int[Array, ""]) -> Float[Array, " n_states"] | float:
        """What one step under `action` costs beyond the step itself, zero where the transition charges none"""
        charged = kernel.step_cost(action)
        return 0.0 if charged is None else charged

    @abstractmethod
    def solve(
        self, kernel: TransitionKernel[Int[Array, ""]], targets: Int[Array, " n_targets"], actions: Int[Array, " n_actions"]
    ) -> Tables:
        """The act to take at each state when steering to each target, and what following it takes from each state"""

    def arrived(self, domain: Domain, targets: Int[Array, " n_targets"]) -> Bool[Array, "n_states n_targets"]:
        """Whether each state of `domain` is within `radius` of each target"""
        return self._squared_distance(domain, targets) <= self.radius**2

    def _squared_distance(self, domain: Domain, targets: Int[Array, " n_targets"]) -> Float[Array, "n_states n_targets"]:
        """Squared embedded distance from every state to every target"""
        coords = coordinates(domain)  # (n_states, k)
        square = jnp.sum(coords**2, axis=-1)
        return jnp.maximum(square[:, None] + square[targets][None, :] - 2 * coords @ coords[targets].T, 0.0)

    def _backup(
        self,
        kernel: TransitionKernel[Int[Array, ""]],
        values: Float[Array, "n_states n_targets"],
        actions: Int[Array, " n_actions"],
        per_step: Callable[[Int[Array, ""]], Float[Array, " n_states"] | float],
    ) -> Float[Array, "n_actions n_states n_targets"]:
        """One backup, all targets at once:

        out[a, s, x] = per_step(a)[s] + sum over s' of P(s' | s, a) values[s', x]
        """
        step = jax.vmap(kernel.expectation, in_axes=(1, None), out_axes=1)
        # (n_states, 1), or a scalar left as one, onto (n_states, n_targets)
        return jnp.stack([jnp.reshape(per_step(jnp.asarray(a)), (-1, 1)) + step(values, jnp.asarray(a)) for a in actions])

    def plan(
        self,
        kernel: TransitionKernel[Int[Array, ""]],
        targets: Int[Array, " n_targets"],
        actions: Int[Array, " n_actions"],
        at: Int[Array, ""],
    ) -> Policy:
        """`solve`, `target_chunk` targets at a time, `S` and `C` kept at `at` alone"""
        step = targets.shape[0] if self.target_chunk is None else self.target_chunk
        if targets.shape[0] <= step:
            return self._plan_block(kernel, targets, actions, at)
        # every target is its own problem, so a block of columns is the whole answer for those columns
        blocks = [self._plan_block(kernel, targets[start : start + step], actions, at) for start in range(0, targets.shape[0], step)]
        return jax.tree.map(lambda *columns: jnp.concatenate(columns, axis=-1), *blocks)

    @partial(jax.jit, static_argnums=0)
    def _plan_block(
        self,
        kernel: TransitionKernel[Int[Array, ""]],
        targets: Int[Array, " n_targets"],
        actions: Int[Array, " n_actions"],
        at: Int[Array, ""],
    ) -> Policy:
        """`solve`, jitted, the act stored as a byte"""
        act, steps, spent = self.solve(kernel, targets, actions)
        return Policy(act.astype(jnp.int8), steps[at], spent[at])

    def evaluate(
        self,
        kernel: TransitionKernel[Int[Array, ""]],
        act: Int[Array, "n_states n_targets"],
        targets: Int[Array, " n_targets"],
        actions: Int[Array, " n_actions"],
        per_step: Callable[[Int[Array, ""]], Float[Array, " n_states"] | float],
    ) -> Float[Array, "n_states n_targets"]:
        """What following `act` under `kernel` accumulates of `per_step` from every state to every target, `L` rounds of

        h_0(s)     = 0
        h_{k+1}(s) = 0                                                            on arrival
                   = per_step(act(s))[s] + sum over s' of P(s' | s, act(s)) h_k(s')
        """
        stop = self.arrived(kernel.domain, targets)
        rows, columns = jnp.arange(stop.shape[0])[:, None], jnp.arange(stop.shape[1])[None, :]
        return lax.fori_loop(
            0,
            self.replan_every,
            lambda _, h: jnp.where(stop, 0.0, self._backup(kernel, h, actions, per_step)[position_of(actions, act), rows, columns]),
            jnp.zeros(stop.shape),
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
        """One sampled walk per key per target from `start`: the states stepped onto, and which of those steps moved"""
        coords = coordinates(kernel.domain)  # (n_states, k)
        square, aimed = jnp.sum(coords**2, axis=-1), coords[targets]  # (n_states,), (n_targets, k)
        target_column = jnp.arange(policy.act.shape[1])  # (n_targets,): which column of the act table each walk reads

        def arrived(state: Int[Array, " n_targets"]) -> Bool[Array, " n_targets"]:
            """Whether each walk's state is within `radius` of its target"""
            return jnp.maximum(square[state] + square[targets] - 2 * jnp.sum(coords[state] * aimed, axis=-1), 0.0) <= self.radius**2

        step_to = jax.vmap(kernel.sample, in_axes=(None, 0, 0))  # (key, state per target, act per target) -> next state per target

        def one_walk(key: PRNGKeyArray) -> tuple[Int[Array, "n_targets replan_every"], Bool[Array, "n_targets replan_every"]]:
            """One walk to every target: z_0 = start, z_{t+1} ~ P(. | z_t, act(z_t)) until arrived(z_t)"""
            state = jnp.full(target_column.shape, start)  # (n_targets,): z_0 = start
            travelling = jnp.ones(target_column.shape, bool)  # (n_targets,): not arrived(z_t)
            path, moved = [], []
            for step_key in jax.random.split(key, self.replan_every):
                # z_{t+1} = P-draw where travelling, else z_t
                state = jnp.where(travelling, step_to(step_key, state, policy.act[state, target_column]), state)
                path.append(state)
                moved.append(travelling)
                travelling = travelling & ~arrived(state)  # travelling and not arrived(z_{t+1})
            return jnp.stack(path, axis=-1), jnp.stack(moved, axis=-1)  # both (n_targets, replan_every)

        return jax.vmap(one_walk)(walk_keys)  # both (n_walks, n_targets, replan_every)


@dataclass(frozen=True)
class ValueIteration(Planner):
    """`L` rounds of

        v_0(s)     = 0
        v_{k+1}(s) = 0                                                                        on arrival
                   = min over a of [ 1 + mu * step_cost(s, a) + sum over s' of P(s' | s, a) v_k(s') ]

    with the steps and the step cost accumulated along each round's minimising act, so `v_L = S + mu * C`,
    then the act attaining the minimum
    """

    def solve(
        self, kernel: TransitionKernel[Int[Array, ""]], targets: Int[Array, " n_targets"], actions: Int[Array, " n_actions"]
    ) -> Tables:
        stop = self.arrived(kernel.domain, targets)
        rows, columns = jnp.arange(stop.shape[0])[:, None], jnp.arange(stop.shape[1])[None, :]
        cost = partial(self.step_cost, kernel)
        plan_on = lambda a: 1.0 + self.cost_weight * cost(a)  # noqa: E731

        def backup(
            _: Int[Array, ""], tables: tuple[Float[Array, "n_states n_targets"], ...]
        ) -> tuple[Float[Array, "n_states n_targets"], ...]:
            value, steps, spent = tables
            backed = self._backup(kernel, value, actions, plan_on)  # (n_actions, n_states, n_targets)
            best = jnp.argmin(backed, axis=0)  # (n_states, n_targets): the position of this round's minimising act
            return (
                jnp.where(stop, 0.0, backed[best, rows, columns]),
                jnp.where(stop, 0.0, self._backup(kernel, steps, actions, lambda a: 1.0)[best, rows, columns]),
                jnp.where(stop, 0.0, self._backup(kernel, spent, actions, cost)[best, rows, columns]),
            )

        zeros = jnp.zeros(stop.shape)
        value, steps, spent = lax.fori_loop(0, self.replan_every, backup, (zeros, zeros, zeros))
        return actions[jnp.argmin(self._backup(kernel, value, actions, plan_on), axis=0)], steps, spent


@dataclass(frozen=True)
class Greedy(Planner):
    """One backup on the distance to the target:

        act(s) = argmin over a of [ 1 + sum over s' of P(s' | s, a) d(s', x) ]

    for `d` the embedded distance, then the steps and the step cost accumulated along it
    """

    def solve(
        self, kernel: TransitionKernel[Int[Array, ""]], targets: Int[Array, " n_targets"], actions: Int[Array, " n_actions"]
    ) -> Tables:
        distance = jnp.sqrt(self._squared_distance(kernel.domain, targets))
        act = actions[jnp.argmin(self._backup(kernel, distance, actions, lambda a: 1.0), axis=0)]
        steps = self.evaluate(kernel, act, targets, actions, lambda a: 1.0)
        return act, steps, self.evaluate(kernel, act, targets, actions, partial(self.step_cost, kernel))
