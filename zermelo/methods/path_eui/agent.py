"""Path-EUI: every L moves, or on arrival within rho of x*, x* <- choose_target(belief, z_h), and the move is pi[f-hat](z_h, x*)

z_h the state read at move h, D_h = D_{h-1} + (z_h, reading_h, context_h), f-hat = E[f | D_h]; a uniform legal act for h < random_opening_steps
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from zermelo.interface import Agent, Analytic, Decision, Domain, Enumerable, Function, Observation, Subset
from zermelo.methods.path_eui.acquisition import Acquisition, TargetChoice
from zermelo.methods.path_eui.belief import Belief, Dataset, coordinates, table_as_field
from zermelo.methods.path_eui.planner import Policy


def posterior_mean_and_log_variance(belief: Belief, context: Float[Array, " n_context"]) -> Function:
    """`(E[f | D], log Var[f | D])` at `context`, as a function over every state"""
    mean, var = belief.mean_and_variance(jnp.arange(coordinates(belief.positions).shape[0]), context)
    return table_as_field(belief.positions, jnp.concatenate([mean, jnp.log(var)], axis=-1))


@register_dataclass
@dataclass(frozen=True)
class PathEUIState:
    """What one move hands the next"""

    belief: Belief
    """`p(f | D_h)`"""

    choice: TargetChoice
    """The `choose_target` that set the standing `x*`"""

    steps_since_target: Int[Array, ""]
    """Moves since that `choose_target`"""

    time_step: Int[Array, ""]
    """`h`"""

    claim: Function = dataclasses.field(metadata=dict(static=True))
    """`posterior_mean_and_log_variance`, re-read every `claim_every` moves"""


@dataclass(frozen=True, kw_only=True)
class PathEUIAgent(Agent[PathEUIState]):
    """The module's rule: `reset` gives `D_0` empty, `decide` gives the move"""

    candidates: Subset[Any]
    """Which states are worth aiming at, published by the problem over the set the field lives on"""

    transition: Analytic[Int[Array, ""], Any]
    """The exact one-step law under a guessed field"""

    belief: Belief
    """An empty belief, its buffer sized to `horizon`"""

    acquisition: Acquisition

    position_key: str
    """Which part of the reading carries where the actor stands"""

    measurement_key: str
    """Which part of the reading carries what it read there"""

    context_key: str
    """Which part of the reading carries the coordinates beyond the state, absent where the readout publishes none"""

    prior_mean_key: str | None
    """Which part of the reading carries the field's prior mean as a function (None learns about zero)"""

    goal_key: str | None
    """Which part of the reading carries the state a trip utility steers to, in the form the position part takes (None steers nowhere)"""

    budget_column: int | None
    """Which column of the context row carries the budget left to spend on steps, None where the problem keeps none"""

    horizon: int
    """`T`: the episode's time_steps, one reading per time_step filling one buffer row"""

    random_opening_steps: int
    """time_steps walked at random before the rule starts"""

    claim_every: int
    """time_steps between re-readings of the claim off the belief"""

    @property
    def positions(self) -> Domain:
        """The states `f` is defined on"""
        return self.candidates.base

    def _candidate_indices(self) -> Int[Array, " n_live"]:
        """Every candidate `x`, as position domain indices"""
        return jnp.flatnonzero(self.candidates.live)

    def _legal_actions(self, obs: Observation) -> Int[Array, " n_actions"]:
        """The acts legal at the state acted from, as values"""
        if not isinstance(obs.legal_actions, Enumerable):
            raise TypeError(f"a planner enumerates acts, and a {type(obs.legal_actions).__name__} has no index")
        acts = jnp.asarray(obs.legal_actions.elements())
        return acts[obs.legal_actions.live] if isinstance(obs.legal_actions, Subset) else acts

    def _context_of(self, obs: Observation) -> Float[Array, " n_context"]:
        """`context_h`, empty where the readout publishes none"""
        return obs.reading[self.context_key] if self.context_key in obs.reading else jnp.zeros(0)

    def _belief_with_observation(self, state: PathEUIState, obs: Observation) -> tuple[Int[Array, ""], Belief]:
        """`z_h`, and `p(f | D_{h-1} + (z_h, reading_h, context_h))`"""
        z = self.candidates.index_of(obs.reading[self.position_key])
        row = int(jnp.sum(state.belief.dataset.is_observed))  # rows written so far; concrete, an episode being a Python loop
        written = state.belief.dataset.add_observation(row, z, obs.reading[self.measurement_key], self._context_of(obs))
        return z, state.belief.fit(written)

    def reset(self, key: PRNGKeyArray, obs: Observation) -> PathEUIState:
        """`D_0` empty, blank `TargetChoice` at their shape, moves since target = `L` so the first move plans"""
        if self.belief.positions != self.positions:
            raise ValueError("the belief and the candidate set were built over different position domains")
        if self.random_opening_steps >= self.horizon:
            raise ValueError("the random walk opening is longer than the episode horizon")
        n_states = coordinates(self.positions).shape[0]
        width = self.belief.dataset.measurement.shape[-1]
        n_scored = (
            len(self._candidate_indices()) if self.acquisition.n_candidates_scored is None else max(self.acquisition.n_candidates_scored, 1)
        )
        time_steps = 1 if self.acquisition.n_paths_per_field == 0 else self.acquisition.planner.replan_length
        blank = TargetChoice(
            candidate_indices=jnp.zeros(n_scored, jnp.int32),
            acquisition_value=jnp.zeros(n_scored),
            target=jnp.zeros((), jnp.int32),
            imagined_path_to_target=jnp.zeros(time_steps, jnp.int32),
            policy_to_target=Policy(jnp.zeros((n_states, 1), jnp.int8), jnp.zeros((1,)), jnp.zeros((1,))),
        )
        context = self._context_of(obs)
        prior_mean = obs.reading[self.prior_mean_key] if self.prior_mean_key is not None and self.prior_mean_key in obs.reading else None
        empty = self.belief.with_prior_mean(prior_mean).fit(Dataset.empty(self.horizon, width, context.shape[-1]))
        return PathEUIState(
            empty,
            blank,
            jnp.asarray(self.acquisition.planner.replan_length, jnp.int32),
            jnp.zeros((), jnp.int32),
            posterior_mean_and_log_variance(empty, context),
        )

    def decide(self, key: PRNGKeyArray, agent_state: PathEUIState, obs: Observation) -> tuple[PathEUIState, Decision]:
        """`(state, obs) -> (state', a)` by the module's rule"""
        k_draw, k_score = jax.random.split(key)
        planner = self.acquisition.planner
        z, belief = self._belief_with_observation(agent_state, obs)
        context = self._context_of(obs)  # the field is read, planned and scored at the context this reading carries
        claim = (
            posterior_mean_and_log_variance(belief, context) if int(agent_state.time_step) % self.claim_every == 0 else agent_state.claim
        )
        if int(agent_state.time_step) < self.random_opening_steps:
            # the opening: a uniform act, no target and no planning
            return PathEUIState(belief, agent_state.choice, agent_state.steps_since_target, agent_state.time_step + 1, claim), Decision(
                jax.random.choice(k_draw, self._legal_actions(obs)), claim
            )
        budget_spent = agent_state.steps_since_target >= planner.replan_length
        if bool(budget_spent | planner.in_goal_set(self.positions, agent_state.choice.target[None])[z, 0]):
            actions = self._legal_actions(obs)
            goal = None if self.goal_key is None else self.candidates.index_of(obs.reading[self.goal_key])
            budget_left = None if self.budget_column is None else context[self.budget_column]
            choice = self.acquisition.choose_target(
                k_score, belief, self.transition, z, actions, self.candidates, context, goal, budget_left
            )
            since = jnp.zeros((), jnp.int32)
        else:
            choice, since = agent_state.choice, agent_state.steps_since_target + 1
        return PathEUIState(belief, choice, since, agent_state.time_step + 1, claim), Decision(choice.policy_to_target.action[z, 0], claim)
