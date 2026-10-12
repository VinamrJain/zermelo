"""Path-EUI: every L moves, or on arrival within rho of x*, x* <- choose(belief, z_h), and the move is pi[f-hat](z_h, x*)

z_h the state read at move h, D_h = D_{h-1} + (z_h, reading_h, context_h), f-hat = E[f | D_h]; a uniform legal act for h < opening_steps
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from zermelo.interface import Agent, Analytic, Decision, Domain, Enumerable, Function, Observation, Subset
from zermelo.methods.waypoint_bo.acquisition import Acquisition, Scores
from zermelo.methods.waypoint_bo.belief import Belief, Dataset, coordinates, lookup
from zermelo.methods.waypoint_bo.planner import Policy


def posterior_moments(belief: Belief, context: Float[Array, " n_context"]) -> Function:
    """`(E[f | D], log Var[f | D])` at `context`, as a function over every state"""
    mean, var = belief.predict(jnp.arange(coordinates(belief.positions).shape[0]), context)
    return lookup(belief.positions, jnp.concatenate([mean, jnp.log(var)], axis=-1))


@register_dataclass
@dataclass(frozen=True)
class WaypointAgentState:
    """What one move hands the next"""

    belief: Belief
    """`p(f | D_h)`"""

    scores: Scores
    """The `choose` that set the standing `x*`"""

    time_steps_since_waypoint: Int[Array, ""]
    """Moves since that `choose`"""

    time_steps_taken: Int[Array, ""]
    """`h`"""

    claim: Function = dataclasses.field(metadata=dict(static=True))
    """`posterior_moments`, re-read every `claim_every` moves"""


@dataclass(frozen=True, kw_only=True)
class WaypointAgent(Agent[WaypointAgentState]):
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

    reading_key: str
    """Which part of the reading carries what it read there"""

    context_key: str
    """Which part of the reading carries the coordinates beyond the state, absent where the readout publishes none"""

    prior_mean_key: str | None
    """Which part of the reading carries the field's prior mean as a function (None learns about zero)"""

    target_key: str | None
    """Which part of the reading carries the state a trip utility steers to, in the form the position part takes (None steers nowhere)"""

    budget_column: int | None
    """Which column of the context row carries the budget left to spend on steps, None where the problem keeps none"""

    horizon: int
    """`T`: the episode's time_steps, one reading per time_step filling one buffer row"""

    opening_steps: int
    """time_steps walked at random before the rule starts"""

    claim_every: int
    """time_steps between re-readings of the claim off the belief"""

    @property
    def positions(self) -> Domain:
        """The states `f` is defined on"""
        return self.candidates.base

    def _live_candidates(self) -> Int[Array, " n_live"]:
        """Every candidate `x`, as position domain indices"""
        return jnp.flatnonzero(self.candidates.live)

    def _legal_actions(self, obs: Observation) -> Int[Array, " n_actions"]:
        """The acts legal at the state acted from, as values"""
        if not isinstance(obs.legal_actions, Enumerable):
            raise TypeError(f"a planner enumerates acts, and a {type(obs.legal_actions).__name__} has no index")
        acts = jnp.asarray(obs.legal_actions.elements())
        return acts[obs.legal_actions.live] if isinstance(obs.legal_actions, Subset) else acts

    def _context(self, obs: Observation) -> Float[Array, " n_context"]:
        """`context_h`, empty where the readout publishes none"""
        return obs.reading[self.context_key] if self.context_key in obs.reading else jnp.zeros(0)

    def _fold_reading(self, state: WaypointAgentState, obs: Observation) -> tuple[Int[Array, ""], Belief]:
        """`z_h`, and `p(f | D_{h-1} + (z_h, reading_h, context_h))`"""
        z = self.candidates.index_of(obs.reading[self.position_key])
        row = int(jnp.sum(state.belief.data.row_is_written))  # rows written so far; concrete, an episode being a Python loop
        written = state.belief.data.write(row, z, obs.reading[self.reading_key], self._context(obs))
        return z, state.belief.fit(written)

    def reset(self, key: PRNGKeyArray, obs: Observation) -> WaypointAgentState:
        """`D_0` empty, blank `Scores` at their shape, moves since waypoint = `L` so the first move plans"""
        if self.belief.positions != self.positions:
            raise ValueError("the belief and the candidate set were built over different position domains")
        if self.opening_steps >= self.horizon:
            raise ValueError("the random walk opening is longer than the episode horizon")
        n_states = coordinates(self.positions).shape[0]
        width = self.belief.data.state_value.shape[-1]
        n_scored = len(self._live_candidates()) if self.acquisition.n_candidates is None else max(self.acquisition.n_candidates, 1)
        time_steps = 1 if self.acquisition.n_walks == 0 else self.acquisition.planner.replan_every
        blank = Scores(
            candidate_indices=jnp.zeros(n_scored, jnp.int32),
            acquisition_value=jnp.zeros(n_scored),
            waypoint=jnp.zeros((), jnp.int32),
            imagined_walk_to_waypoint=jnp.zeros(time_steps, jnp.int32),
            route=Policy(jnp.zeros((n_states, 1), jnp.int8), jnp.zeros((1,)), jnp.zeros((1,))),
        )
        context = self._context(obs)
        prior_mean = obs.reading[self.prior_mean_key] if self.prior_mean_key is not None and self.prior_mean_key in obs.reading else None
        empty = self.belief.with_prior_mean(prior_mean).condition(Dataset.empty(self.horizon, width, context.shape[-1]))
        return WaypointAgentState(
            empty,
            blank,
            jnp.asarray(self.acquisition.planner.replan_every, jnp.int32),
            jnp.zeros((), jnp.int32),
            posterior_moments(empty, context),
        )

    def decide(self, key: PRNGKeyArray, agent_state: WaypointAgentState, obs: Observation) -> tuple[WaypointAgentState, Decision]:
        """`(state, obs) -> (state', a)` by the module's rule"""
        k_draw, k_score = jax.random.split(key)
        planner = self.acquisition.planner
        z, belief = self._fold_reading(agent_state, obs)
        context = self._context(obs)  # the field is read, planned and scored at the context this reading carries
        claim = posterior_moments(belief, context) if int(agent_state.time_steps_taken) % self.claim_every == 0 else agent_state.claim
        if int(agent_state.time_steps_taken) < self.opening_steps:
            # the opening: a uniform act, no waypoint and no planning
            return WaypointAgentState(
                belief, agent_state.scores, agent_state.time_steps_since_waypoint, agent_state.time_steps_taken + 1, claim
            ), Decision(jax.random.choice(k_draw, self._legal_actions(obs)), claim)
        budget_spent = agent_state.time_steps_since_waypoint >= planner.replan_every
        if bool(budget_spent | planner.arrived(self.positions, agent_state.scores.waypoint[None])[z, 0]):
            actions = self._legal_actions(obs)
            target = None if self.target_key is None else self.candidates.index_of(obs.reading[self.target_key])
            budget_left = None if self.budget_column is None else context[self.budget_column]
            scores = self.acquisition.choose(k_score, belief, self.transition, z, actions, self.candidates, context, target, budget_left)
            since = jnp.zeros((), jnp.int32)
        else:
            scores, since = agent_state.scores, agent_state.time_steps_since_waypoint + 1
        return WaypointAgentState(belief, scores, since, agent_state.time_steps_taken + 1, claim), Decision(scores.route.act[z, 0], claim)
