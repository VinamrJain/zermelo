"""x* = argmax over the reachable candidates of alpha(x), where

w(x)     = E over fields and walks of [ U(D + obs(tau)) - U(D) ]
alpha(x) = w(x) - lambda * (max w - min w) * (S(z, x) + mu C(z, x)) / B        lambda = 0 scores w alone
reachable: 0 < S(z, x) < L, and C(z, x) <= the budget left where the problem keeps one
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from zermelo.interface import Analytic, Subset
from zermelo.methods.waypoint_bo.belief import Belief, Dataset
from zermelo.methods.waypoint_bo.planner import Planner, Policy
from zermelo.methods.waypoint_bo.utility import Utility


@register_dataclass
@dataclass(frozen=True)
class Scores:
    """One acquisition pass: what each candidate was worth, which won, and the route to it"""

    candidate_indices: Int[Array, " n_candidates"]
    """The states scored, as position domain indices"""

    acquisition_value: Float[Array, " n_candidates"]
    """`alpha` at each of them"""

    waypoint: Int[Array, ""]
    """`x_{n+1}`: the candidate that won"""

    imagined_walk_to_waypoint: Int[Array, " time_steps"]
    """The winner's walk under the first field and walk"""

    route: Policy
    """`pi[mu_n](., x_{n+1})`: the winner's column of the plan under the posterior mean, one column wide"""


@dataclass(frozen=True)
class Acquisition:
    """Scores the reachable candidate waypoints and takes the best"""

    utility: Utility
    planner: Planner

    n_fields: int
    """`S`: posterior fields drawn per decision, or the posterior mean at zero"""

    n_walks: int
    """`M`: walks imagined per field, or the destination alone at zero"""

    n_candidates: int | None
    """How many candidates to score: `None` every live one, an integer that many drawn uniformly each decision, 0 the target alone"""

    improvement: bool
    """Whether to score the collected readings and the imagined ones together, less the collected ones alone"""

    penalty_rate: float
    """`lambda`: a full-budget trip is charged this share of the spread of the candidate values. Zero charges nothing"""

    def choose(
        self,
        key: PRNGKeyArray,
        belief: Belief,
        transition: Analytic[Int[Array, ""], Any],
        position: Int[Array, ""],
        actions: Int[Array, " n_actions"],
        candidates: Subset[Any],
        context: Float[Array, " n_context"],
        target: Int[Array, ""] | None,
        budget_left: Float[Array, ""] | None,
    ) -> Scores:
        """Every candidate scored under every field and walk, and the argmax over the reachable ones, a tie broken at random; `target` is the one candidate at `n_candidates = 0`"""
        if self.n_candidates == 0 and target is None:
            raise ValueError("the target alone is scored, and this decision names none")
        k_indices, k_field, k_walk, k_utility, k_tie = jax.random.split(key, 5)
        candidate_indices = jnp.flatnonzero(candidates.live)
        if self.n_candidates == 0:
            candidate_indices = jnp.asarray(target)[None]
        elif self.n_candidates is not None:
            # a prefix of one permutation, so a shorter shortlist is a subset of every longer one
            candidate_indices = jax.random.permutation(k_indices, candidate_indices)[: self.n_candidates]
        # keyed by index, so the draws of a smaller n_walks are a subset of a larger one
        walk_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(k_walk, jnp.arange(max(self.n_walks, 1)))
        collected_readings, n_scored = belief.data, candidate_indices.shape[0]
        # pi[mu_n], solved once and reused
        mean_policy = self.planner.plan(transition.kernel(belief.mean(context)), candidate_indices, actions, position)
        utility_samples, first_draw_walks = [], None

        # f-hat^(1..S)
        fields = [belief.mean(context)] if self.n_fields == 0 else belief.draw(k_field, self.n_fields, context)
        score_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(k_utility, jnp.arange(len(fields)))  # keyed by index
        for k_score, field in zip(score_keys, fields, strict=True):
            kernel = transition.kernel(field)  # p(.|f-hat)
            values = field.table  # f-hat(z, context), one row per state of the position domain
            # D_n with each reading re-taken from f-hat at its own state and context: the reading itself where readings are exact
            collected_under_field = Dataset(
                collected_readings.state_index,
                field.at_collected_readings,
                collected_readings.row_is_written,
                collected_readings.context_value,
            )
            base_utility = self.utility(k_score, belief, collected_under_field, candidates) if self.improvement else jnp.zeros(())  # U(D_n)

            if self.n_walks == 0:  # (1, n_candidates, 1): tau = (x), the destination and no route
                walk, moved = candidate_indices[None, :, None], jnp.ones((1, n_scored, 1), bool)
            else:  # both (n_walks, n_candidates, time_steps): tau ~ p(. | z, x, f-hat, pi)
                walk, moved = self.planner.roll(walk_keys, kernel, mean_policy, candidate_indices, position)
            first_draw_walks = (
                walk[0] if first_draw_walks is None else first_draw_walks
            )  # (n_candidates, time_steps) under f-hat^(1), tau^(1)

            # obs(tau) = {(z_i, f-hat(z_i, context), context)}: the walk's states read off f-hat, every row at the decision's own context
            imagined = Dataset(walk, values[walk], moved, jnp.broadcast_to(context, (*walk.shape, context.shape[-1])))
            scored = collected_under_field.broadcast(walk.shape[:-1]).concat(imagined) if self.improvement else imagined  # D_n + obs(tau)
            utility_samples.append(self.utility(k_score, belief, scored, candidates) - base_utility)  # (n_walks, n_candidates)

        # (n_fields, n_walks, n_candidates), averaged over both sample axes
        worth = jnp.mean(jnp.stack(utility_samples), axis=(0, 1))  # w(x)
        steps, spent = mean_policy.steps_to_target, mean_policy.budget_spent_to_target  # S(z, x), C(z, x)
        # lambda * (max w - min w) / B charges each unit of the plan's own cost-to-go S + mu C. A flat w charges nothing
        charge = self.penalty_rate * (jnp.max(worth) - jnp.min(worth)) / self.planner.cost_budget
        value = worth - charge * (steps + self.planner.cost_weight * spent)  # alpha(x)
        # within the arrival ball nothing is walked, and at S = L nothing arrived before the next decision
        reachable = (steps > 0.0) & (steps < self.planner.replan_every)
        if budget_left is not None:
            reachable &= spent <= budget_left
        masked = jnp.where(reachable, value, -jnp.inf)
        tied = masked == jnp.max(masked)  # the maxima among the reachable candidates
        best = jnp.where(
            jnp.any(reachable),
            jax.random.choice(k_tie, tied.shape[0], p=tied / jnp.maximum(jnp.sum(tied), 1)),  # one of the tied maxima, uniformly
            jnp.argmin(jnp.where(steps > 0.0, steps, jnp.inf)),  # nothing reachable: the nearest candidate
        )
        return Scores(
            candidate_indices=candidate_indices,
            acquisition_value=value,
            waypoint=candidate_indices[best],
            imagined_walk_to_waypoint=jnp.asarray(first_draw_walks)[best],
            route=Policy(mean_policy.act[:, best, None], steps[best, None], spent[best, None]),
        )
