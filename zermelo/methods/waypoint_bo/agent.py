"""The method itself: pick a waypoint, commit a route to it, walk that route, pick again"""

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
    """The belief's mean and log-variance per component at one context, readable at any state"""
    mean, var = belief.predict(jnp.arange(coordinates(belief.positions).shape[0]), context)
    return lookup(belief.positions, jnp.concatenate([mean, jnp.log(var)], axis=-1))


@register_dataclass
@dataclass(frozen=True)
class WaypointAgentState:
    """What the method carries between steps, and what a consumer plots it by"""

    belief: Belief
    """Holds the readings folded in so far, and the model's parameters as they stand"""

    scores: Scores
    """The acquisition pass that chose the standing waypoint, carried until the next one"""

    policy: Policy
    """The route committed to when the waypoint was chosen, one column wide"""

    time_steps_since_waypoint: Int[Array, ""]
    """time_steps taken since that waypoint was chosen"""

    time_steps_taken: Int[Array, ""]
    """time_steps taken this episode"""

    claim: Function = dataclasses.field(metadata=dict(static=True))
    """What the method asserts about the field, re-read off the belief every `claim_every` time_steps and unchanged in between"""


@dataclass(frozen=True, kw_only=True)
class WaypointAgent(Agent[WaypointAgentState]):
    """Bayesian optimization over waypoints, with the walk to each waypoint scored and charged"""

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

    horizon: int
    """`T`: the episode's time_steps, one reading per time_step filling one buffer row"""

    opening_legs: int
    """How many of the planner's step budgets are walked at random before the rule starts"""

    claim_every: int
    """time_steps between re-readings of the claim off the belief"""

    @property
    def positions(self) -> Domain:
        """The states the field is defined on"""
        return self.candidates.base

    def _live_candidates(self) -> Int[Array, " n_live"]:
        """Every candidate state, as position domain indices"""
        return jnp.flatnonzero(self.candidates.live)

    def _legal_actions(self, obs: Observation) -> Int[Array, " n_actions"]:
        """The values of the acts legal at the state being acted from"""
        if not isinstance(obs.legal_actions, Enumerable):
            raise TypeError(f"a planner enumerates acts, and a {type(obs.legal_actions).__name__} has no index")
        acts = jnp.asarray(obs.legal_actions.elements())
        return acts[obs.legal_actions.live] if isinstance(obs.legal_actions, Subset) else acts

    def _checked_kernel(self, field: Function) -> Any:
        """The law under `field`, checked to be written over the states this method aims at"""
        kernel = self.transition.kernel(field)
        if kernel.domain != self.positions:
            raise ValueError("the transition's kernel is written over a different set from the one this agent was given")
        return kernel

    def _context(self, obs: Observation) -> Float[Array, " n_context"]:
        """The coordinates this reading carries beyond its state, empty where the readout publishes none"""
        return obs.reading[self.context_key] if self.context_key in obs.reading else jnp.zeros(0)

    def _fold_reading(self, state: WaypointAgentState, obs: Observation) -> tuple[Int[Array, ""], Belief]:
        """Where the actor stands, and the belief with this step's reading folded in"""
        z = self.candidates.index_of(obs.reading[self.position_key])
        row = int(jnp.sum(state.belief.data.row_is_written))  # rows written so far; concrete, an episode being a Python loop
        written = state.belief.data.write(row, z, obs.reading[self.reading_key], self._context(obs))
        return z, state.belief.fit(written)

    def reset(self, key: PRNGKeyArray, obs: Observation) -> WaypointAgentState:
        """Nothing read yet: an empty buffer, blank scores at the shape they keep, and the step budget already spent"""
        if self.belief.positions != self.positions:
            raise ValueError("the belief and the candidate set were built over different position domains")
        if self.acquisition.planner.replan_every * self.opening_legs >= self.horizon:
            raise ValueError("the random walk opening is longer than the episode horizon")
        n_states = coordinates(self.positions).shape[0]
        width = self.belief.data.state_value.shape[-1]
        n_scored = len(self._live_candidates()) if self.acquisition.n_candidates is None else self.acquisition.n_candidates
        time_steps = 1 if self.acquisition.n_walks == 0 else self.acquisition.planner.replan_every
        blank = Scores(
            candidate_indices=jnp.zeros(n_scored, jnp.int32),
            acquisition_value=jnp.zeros(n_scored),
            waypoint=jnp.zeros((), jnp.int32),
            imagined_walk_to_waypoint=jnp.zeros(time_steps, jnp.int32),
            waypoint_value_spread=jnp.zeros(()),
            predicted_step_cost=jnp.zeros(()),
            rolled_step_cost=jnp.zeros(()),
            frac_zero_value_candidates=jnp.zeros(()),
            step_charge=jnp.zeros(()),
            frac_reachable_candidates=jnp.zeros(()),
        )
        policy = Policy(jnp.zeros((n_states, 1), jnp.int32), jnp.zeros((n_states, 1)))
        context = self._context(obs)
        prior_mean = obs.reading[self.prior_mean_key] if self.prior_mean_key is not None and self.prior_mean_key in obs.reading else None
        empty = self.belief.with_prior_mean(prior_mean).condition(Dataset.empty(self.horizon, width, context.shape[-1]))
        return WaypointAgentState(
            empty,
            blank,
            policy,
            jnp.asarray(self.acquisition.planner.replan_every, jnp.int32),
            jnp.zeros((), jnp.int32),
            posterior_moments(empty, context),
        )

    def decide(self, key: PRNGKeyArray, agent_state: WaypointAgentState, obs: Observation) -> tuple[WaypointAgentState, Decision]:
        """The reading folded in, and either a random opening act, a fresh waypoint, or the next act of the standing route"""
        k_draw, k_score, k_plan = jax.random.split(key, 3)
        planner = self.acquisition.planner
        z, belief = self._fold_reading(agent_state, obs)
        context = self._context(obs)  # the field is read, planned and scored at the context this reading carries
        claim = posterior_moments(belief, context) if int(agent_state.time_steps_taken) % self.claim_every == 0 else agent_state.claim
        if int(agent_state.time_steps_taken) < planner.replan_every * self.opening_legs:
            # the opening: a uniform act, no waypoint and no planning
            return WaypointAgentState(
                belief,
                agent_state.scores,
                agent_state.policy,
                agent_state.time_steps_since_waypoint,
                agent_state.time_steps_taken + 1,
                claim,
            ), Decision(jax.random.choice(k_draw, self._legal_actions(obs)), claim)
        budget_spent = agent_state.time_steps_since_waypoint >= planner.replan_every
        if bool(budget_spent | planner.arrived(self.positions, agent_state.scores.waypoint[None])[z, 0]):
            actions = self._legal_actions(obs)
            scores = self.acquisition.choose(k_score, belief, self.transition, z, actions, self.candidates, context)
            # the walk actually taken is planned under the posterior mean, whatever field the winner was scored under
            policy = planner.plan(k_plan, self._checked_kernel(belief.mean(context)), scores.waypoint[None], actions)
            since = jnp.zeros((), jnp.int32)
        else:
            scores, policy, since = agent_state.scores, agent_state.policy, agent_state.time_steps_since_waypoint + 1
        return WaypointAgentState(belief, scores, policy, since, agent_state.time_steps_taken + 1, claim), Decision(policy.act[z, 0], claim)
