"""Path-EUI: every L moves, or on reaching the goal set of x*, x* <- choose_target(belief, z_h); the move is pi[phi-hat](z_h, x*)

z_h the state observed at move h, d_h = d_{h-1} + (z_h, w_h, context_h), phi-hat = E[phi | d_h]; a uniform legal action for
h < random_opening_steps.
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
    """`(E[phi | d], log Var[phi | d])` at `context`, as a function over every state: the claim the agent publishes"""
    mean, var = belief.mean_and_variance(jnp.arange(coordinates(belief.positions).shape[0]), context)
    return table_as_field(belief.positions, jnp.concatenate([mean, jnp.log(var)], axis=-1))


@register_dataclass
@dataclass(frozen=True)
class PathEUIState:
    """What one move hands the next"""

    belief: Belief
    """`P(phi | d_h)`"""

    choice: TargetChoice
    """The `choose_target` that set the standing `x*`"""

    steps_since_target: Int[Array, ""]
    """`ell`: moves since that `choose_target`"""

    time_step: Int[Array, ""]
    """`h`"""

    claim: Function = dataclasses.field(metadata=dict(static=True))
    """`posterior_mean_and_log_variance`, re-read every `claim_every` moves"""


@dataclass(frozen=True, kw_only=True)
class PathEUIAgent(Agent[PathEUIState]):
    """The module's rule: `reset` gives `d_0` empty, `decide` gives the move"""

    candidates: Subset[Any]
    """`X`: which states a target may be, published by the problem over the set the field lives on"""

    transition: Analytic[Int[Array, ""], Any]
    """`T`: the exact one-step law under a given field"""

    belief: Belief
    """An empty belief, its dataset sized to `horizon`"""

    acquisition: Acquisition

    position_key: str
    """Which part of `obs.reading` carries `z_h`, where the agent stands"""

    measurement_key: str
    """Which part of `obs.reading` carries `w_h`, what it measured there"""

    context_key: str
    """Which part of `obs.reading` carries the coordinates beyond the state, absent where the readout publishes none"""

    prior_mean_key: str | None
    """Which part of `obs.reading` carries the field's prior mean as a function (None learns about zero)"""

    horizon: int
    """`H`: the episode's time steps, one observation per time step filling one slot of the dataset"""

    random_opening_steps: int
    """Time steps moved uniformly at random before the rule starts"""

    claim_every: int
    """Time steps between re-readings of the claim off the belief"""

    @property
    def positions(self) -> Domain:
        """`Z`: the states the field is defined on"""
        return self.candidates.base

    def _candidate_indices(self) -> Int[Array, " n_candidates"]:
        """Every candidate `x`, as position domain indices"""
        return jnp.flatnonzero(self.candidates.live)

    def _legal_actions(self, obs: Observation) -> Int[Array, " n_actions"]:
        """The actions legal at the state acted from, as values"""
        if not isinstance(obs.legal_actions, Enumerable):
            raise TypeError(f"a planner enumerates actions, and a {type(obs.legal_actions).__name__} has no index")
        actions = jnp.asarray(obs.legal_actions.elements())
        return actions[obs.legal_actions.live] if isinstance(obs.legal_actions, Subset) else actions

    def _context_of(self, obs: Observation) -> Float[Array, " n_context"]:
        """`context_h`, empty where the readout publishes none"""
        return obs.reading[self.context_key] if self.context_key in obs.reading else jnp.zeros(0)

    def _belief_with_observation(self, state: PathEUIState, obs: Observation) -> tuple[Int[Array, ""], Belief]:
        """`z_h`, and `P(phi | d_{h-1} + (z_h, w_h, context_h))`"""
        z = self.candidates.index_of(obs.reading[self.position_key])
        slot = int(jnp.sum(state.belief.dataset.is_observed))  # observations so far; concrete, an episode being a Python loop
        dataset = state.belief.dataset.add_observation(slot, z, obs.reading[self.measurement_key], self._context_of(obs))
        return z, state.belief.fit(dataset)

    def reset(self, key: PRNGKeyArray, obs: Observation) -> PathEUIState:
        """`d_0` empty, a blank `TargetChoice` at its shape, `ell = L` so the first move chooses a target"""
        if self.belief.positions != self.positions:
            raise ValueError("the belief and the candidate set were built over different position domains")
        if self.random_opening_steps >= self.horizon:
            raise ValueError("the random opening is longer than the episode horizon")
        n_states = coordinates(self.positions).shape[0]
        n_components = self.belief.dataset.measurement.shape[-1]
        n_scored = len(self._candidate_indices()) if self.acquisition.n_candidates_scored is None else self.acquisition.n_candidates_scored
        path_length = 1 if self.acquisition.n_paths_per_field == 0 else self.acquisition.planner.replan_length
        blank = TargetChoice(
            candidate_indices=jnp.zeros(n_scored, jnp.int32),
            acquisition_value=jnp.zeros(n_scored),
            target=jnp.zeros((), jnp.int32),
            imagined_path_to_target=jnp.zeros(path_length, jnp.int32),
            policy_to_target=Policy(jnp.zeros((n_states, 1), jnp.int8), jnp.zeros((1,))),
        )
        context = self._context_of(obs)
        prior_mean = obs.reading[self.prior_mean_key] if self.prior_mean_key is not None and self.prior_mean_key in obs.reading else None
        empty = self.belief.with_prior_mean(prior_mean).fit(Dataset.empty(self.horizon, n_components, context.shape[-1]))
        return PathEUIState(
            empty,
            blank,
            jnp.asarray(self.acquisition.planner.replan_length, jnp.int32),
            jnp.zeros((), jnp.int32),
            posterior_mean_and_log_variance(empty, context),
        )

    def decide(self, key: PRNGKeyArray, agent_state: PathEUIState, obs: Observation) -> tuple[PathEUIState, Decision]:
        """`(state, obs) -> (state', a)` by the module's rule"""
        k_opening, k_choice = jax.random.split(key)
        planner = self.acquisition.planner
        z, belief = self._belief_with_observation(agent_state, obs)
        context = self._context_of(obs)  # the field is read, planned and scored at the context this observation carries
        claim = (
            posterior_mean_and_log_variance(belief, context) if int(agent_state.time_step) % self.claim_every == 0 else agent_state.claim
        )
        if int(agent_state.time_step) < self.random_opening_steps:  # h < random_opening_steps: a uniform action, no target
            return PathEUIState(belief, agent_state.choice, agent_state.steps_since_target, agent_state.time_step + 1, claim), Decision(
                jax.random.choice(k_opening, self._legal_actions(obs)), claim
            )
        # ell >= L, or z_h in G(x*)
        replan_is_due = (agent_state.steps_since_target >= planner.replan_length) | planner.in_goal_set(
            self.positions, agent_state.choice.target[None]
        )[z, 0]
        if bool(replan_is_due):
            acquisition = (  # K = H - h where the planner's K is the steps left
                self.acquisition
                if planner.value_iterations is not None
                else dataclasses.replace(
                    self.acquisition, planner=dataclasses.replace(planner, value_iterations=self.horizon - int(agent_state.time_step))
                )
            )
            choice = acquisition.choose_target(k_choice, belief, self.transition, z, self._legal_actions(obs), self.candidates, context)
            steps_since_target = jnp.zeros((), jnp.int32)
        else:
            choice, steps_since_target = agent_state.choice, agent_state.steps_since_target + 1
        return PathEUIState(belief, choice, steps_since_target, agent_state.time_step + 1, claim), Decision(
            choice.policy_to_target.action[z, 0], claim
        )
