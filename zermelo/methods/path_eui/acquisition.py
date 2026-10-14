"""x* ~ Uniform( argmax over reachable x of alpha(x) ), the nearest x by S when none is reachable

z the state scored from, D the readings so far, f-hat = E[f | D], pi = pi[f-hat] with its S(z, x) and C(z, x):

    w(x)      = (1 / S M) sum over s, m of [ U(D + obs(tau^(s,m)(x))) - U(D) ],   tau^(s,m)(x) a walk under pi(., x) and f^(s) ~ p(f | D)
    alpha(x)  = w(x) - lambda (max w - min w) (S(z, x) + mu C(z, x)) / B,         B = L (1 + mu), lambda = 0 choice w alone
    reachable = 0 < S(z, x) < L, and C(z, x) <= the budget left where the problem keeps one
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from zermelo.interface import Analytic, Subset
from zermelo.methods.path_eui.belief import Belief, Dataset
from zermelo.methods.path_eui.planner import Planner, Policy
from zermelo.methods.path_eui.utility import Utility


@register_dataclass
@dataclass(frozen=True)
class TargetChoice:
    """One `choose_target`: the candidates `x`, `alpha(x)`, `x*`, `tau^(1,1)(x*)` and `pi(., x*)`"""

    candidate_indices: Int[Array, " n_candidates_scored"]
    """The `x` scored, as position domain indices"""

    acquisition_value: Float[Array, " n_candidates_scored"]
    """`alpha(x)`"""

    target: Int[Array, ""]
    """`x*`"""

    imagined_path_to_target: Int[Array, " time_steps"]
    """`tau^(1,1)(x*)`"""

    policy_to_target: Policy
    """`pi(., x*)`, one column wide"""


@dataclass(frozen=True)
class Acquisition:
    """`choose_target`: the module's rule, `(belief, z) -> TargetChoice`"""

    utility: Utility
    """`U`"""

    planner: Planner
    """`pi[f]`"""

    n_field_samples: int
    """`S`, 0 for f-hat alone"""

    n_paths_per_field: int
    """`M`, 0 for tau = (x) and no walk"""

    n_candidates_scored: int | None
    """`x` scored: `None` every live candidate, an integer that many drawn uniformly, 0 the goal alone"""

    subtract_dataset_utility: bool
    """`U(D + obs(tau)) - U(D)` when True, `U(obs(tau))` when False"""

    penalty_rate: float
    """`lambda`"""

    def choose_target(
        self,
        key: PRNGKeyArray,
        belief: Belief,
        transition: Analytic[Int[Array, ""], Any],
        position: Int[Array, ""],
        actions: Int[Array, " n_actions"],
        candidates: Subset[Any],
        context: Float[Array, " n_context"],
        goal: Int[Array, ""] | None,
        budget_left: Float[Array, ""] | None,
    ) -> TargetChoice:
        """`(belief, z = position, ...) -> TargetChoice` by the module's rule; `goal` is the one `x` at `n_candidates_scored = 0`"""
        if self.n_candidates_scored == 0 and goal is None:
            raise ValueError("the goal alone is scored, and this decision names none")
        k_indices, k_field, k_walk, k_utility, k_tie = jax.random.split(key, 5)
        candidate_indices = jnp.flatnonzero(candidates.live)
        if self.n_candidates_scored == 0:
            candidate_indices = jnp.asarray(goal)[None]
        elif self.n_candidates_scored is not None:
            # a prefix of one permutation, so a shorter shortlist is a subset of every longer one
            candidate_indices = jax.random.permutation(k_indices, candidate_indices)[: self.n_candidates_scored]
        # keyed by index, so the draws of a smaller n_paths_per_field are a subset of a larger one
        walk_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(k_walk, jnp.arange(max(self.n_paths_per_field, 1)))
        collected_readings, n_scored = belief.dataset, candidate_indices.shape[0]
        # pi[mu_n], solved once and reused
        mean_policy = self.planner.policy_to_targets(transition.kernel(belief.mean_field(context)), candidate_indices, actions, position)
        utility_samples, first_draw_walks = [], None

        # f-hat^(1..S)
        fields = [belief.mean_field(context)] if self.n_field_samples == 0 else belief.sample_fields(k_field, self.n_field_samples, context)
        score_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(k_utility, jnp.arange(len(fields)))  # keyed by index
        for k_score, field in zip(score_keys, fields, strict=True):
            kernel = transition.kernel(field)  # p(.|f-hat)
            values = field.at_every_state  # f-hat(z, context), one row per state of the position domain
            # D_n with each reading re-taken from f-hat at its own state and context: the reading itself where readings are exact
            collected_under_field = Dataset(
                collected_readings.state_index, field.at_observations, collected_readings.is_observed, collected_readings.context_value
            )
            base_utility = (
                self.utility(k_score, belief, collected_under_field, candidates) if self.subtract_dataset_utility else jnp.zeros(())
            )  # U(D_n)

            if self.n_paths_per_field == 0:  # (1, n_candidates_scored, 1): tau = (x), the destination and no policy_to_target
                walk, moved = candidate_indices[None, :, None], jnp.ones((1, n_scored, 1), bool)
            else:  # both (n_paths_per_field, n_candidates_scored, time_steps): tau ~ p(. | z, x, f-hat, pi)
                walk, moved = self.planner.roll_paths(walk_keys, kernel, mean_policy, candidate_indices, position)
            first_draw_walks = (
                walk[0] if first_draw_walks is None else first_draw_walks
            )  # (n_candidates_scored, time_steps) under f-hat^(1), tau^(1)

            # obs(tau) = {(z_i, f-hat(z_i, context), context)}: the walk's states read off f-hat, every row at the decision's own context
            imagined = Dataset(walk, values[walk], moved, jnp.broadcast_to(context, (*walk.shape, context.shape[-1])))
            scored = (
                collected_under_field.broadcast(walk.shape[:-1]).concat(imagined) if self.subtract_dataset_utility else imagined
            )  # D_n + obs(tau)
            utility_samples.append(
                self.utility(k_score, belief, scored, candidates) - base_utility
            )  # (n_paths_per_field, n_candidates_scored)

        # (n_field_samples, n_paths_per_field, n_candidates_scored), averaged over both sample axes
        worth = jnp.mean(jnp.stack(utility_samples), axis=(0, 1))  # w(x)
        steps, spent = mean_policy.expected_path_length, mean_policy.expected_cost  # S(z, x), C(z, x)
        # lambda * (max w - min w) / B charges each unit of the plan's own cost-to-go S + mu C. A flat w charges nothing
        charge = self.penalty_rate * (jnp.max(worth) - jnp.min(worth)) / self.planner.cost_budget
        value = worth - charge * (steps + self.planner.cost_weight * spent)  # alpha(x)
        # within the arrival ball nothing is walked, and at S = L nothing in_goal_set before the next decision
        reachable = (steps > 0.0) & (steps < self.planner.replan_length)
        if budget_left is not None:
            reachable &= spent <= budget_left
        masked = jnp.where(reachable, value, -jnp.inf)
        tied = masked == jnp.max(masked)  # the maxima among the reachable candidates
        best = jnp.where(
            jnp.any(reachable),
            jax.random.choice(k_tie, tied.shape[0], p=tied / jnp.maximum(jnp.sum(tied), 1)),  # one of the tied maxima, uniformly
            jnp.argmin(jnp.where(steps > 0.0, steps, jnp.inf)),  # nothing reachable: the nearest candidate
        )
        return TargetChoice(
            candidate_indices=candidate_indices,
            acquisition_value=value,
            target=candidate_indices[best],
            imagined_path_to_target=jnp.asarray(first_draw_walks)[best],
            policy_to_target=Policy(mean_policy.action[:, best, None], steps[best, None], spent[best, None]),
        )
