"""`alpha(x)` over candidate targets `x`, and `x*` the target chosen

z_h the state decided from, d_h the dataset so far, phi-hat = E[phi | d_h], pi = pi[phi-hat] the policy under it with
H-hat(z_h, x) its expected path length to x and C-hat(z_h, x) its expected cost:

    u(x)     = (1 / S M) sum over s, m of [ U(d_h + tau-hat^(s,m)(x)) - U(d_h) ],   tau-hat^(s,m)(x) a path under pi(., x) and phi^(s) ~ P(phi | d_h)
    lambda'  = lambda (max u - min u) / (L (1 + kappa))
    alpha(x) = u(x) - lambda' (H-hat(z_h, x) + kappa C-hat(z_h, x))
    x*       ~ Uniform( argmax over x in delta_L of alpha(x) ),   delta_L = { x : 0 < H-hat(z_h, x) < L, and C-hat(z_h, x) <= the budget left }
               the nearest x by H-hat when delta_L is empty

A candidate is a state the rule may score, a target one the planner steers to, and x* the target chosen.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from zermelo.interface import Analytic, Subset
from zermelo.methods.path_eui.belief import Belief, Dataset
from zermelo.methods.path_eui.planner import Planner, Policy
from zermelo.methods.path_eui.utility import Utility


@register_dataclass
@dataclass(frozen=True)
class TargetChoice:
    """One `choose_target`: the candidates `x` scored, `alpha(x)`, `x*`, `tau-hat^(1,1)(x*)` and `pi(., x*)`"""

    candidate_indices: Int[Array, " n_candidates_scored"]
    """The `x` scored, as position domain indices"""

    acquisition_value: Float[Array, " n_candidates_scored"]
    """`alpha(x)`"""

    target: Int[Array, ""]
    """`x*`"""

    imagined_path_to_target: Int[Array, " replan_length"]
    """`tau-hat^(1,1)(x*)`: the states of one imagined path to `x*` under the first field sample"""

    policy_to_target: Policy
    """`pi(., x*)`, one column wide"""


@dataclass(frozen=True)
class Acquisition:
    """`choose_target`: the module's rule, `(belief, z_h) -> TargetChoice`"""

    utility: Utility
    """`U`"""

    planner: Planner
    """`pi[phi]`"""

    n_field_samples: int
    """`S`; 0 scores under phi-hat alone"""

    n_paths_per_field: int
    """`M`; 0 scores tau-hat = (x), the target alone with no path"""

    n_candidates_scored: int | None
    """Which `x` are scored: `None` every candidate, an integer that many drawn uniformly, 0 the goal alone"""

    subtract_dataset_utility: bool
    """`U(d_h + tau-hat) - U(d_h)` when True, `U(tau-hat)` when False"""

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
        """`(belief, z_h = position, ...) -> TargetChoice` by the module's rule; `goal` is the one `x` at `n_candidates_scored = 0`"""
        k_candidates, k_fields, k_paths, k_utility, k_tie = jax.random.split(key, 5)
        candidates_scored = self._candidates_to_score(k_candidates, candidates, goal)  # x
        # pi[phi-hat] with H-hat and C-hat at z_h, solved once and reused by every imagined path
        policy = self.planner.policy_to_targets(transition.kernel(belief.mean_field(context)), candidates_scored, actions, position)
        utility_improvement, imagined_paths = self._expected_utility_improvement(
            k_fields, k_paths, k_utility, belief, transition, policy, candidates_scored, position, context, candidates
        )  # u(x), tau-hat^(1,1)(x)
        # lambda' = lambda (max u - min u) / (L (1 + kappa)): a flat u charges nothing
        penalty = self.penalty_rate * (jnp.max(utility_improvement) - jnp.min(utility_improvement)) / self.planner.cost_budget
        acquisition_value = utility_improvement - penalty * (policy.expected_path_length + self.planner.cost_weight * policy.expected_cost)
        chosen = self._best_reachable_candidate(k_tie, acquisition_value, policy, budget_left)
        return TargetChoice(
            candidate_indices=candidates_scored,
            acquisition_value=acquisition_value,
            target=candidates_scored[chosen],
            imagined_path_to_target=imagined_paths[chosen],
            policy_to_target=Policy(
                policy.action[:, chosen, None], policy.expected_path_length[chosen, None], policy.expected_cost[chosen, None]
            ),
        )

    def _candidates_to_score(
        self, key: PRNGKeyArray, candidates: Subset[Any], goal: Int[Array, ""] | None
    ) -> Int[Array, " n_candidates_scored"]:
        """The `x` scored: every candidate, `n_candidates_scored` of them drawn uniformly, or the goal alone"""
        if self.n_candidates_scored == 0:
            if goal is None:
                raise ValueError("the goal alone is scored, and this decision names none")
            return jnp.asarray(goal)[None]
        every = jnp.flatnonzero(candidates.live)
        if self.n_candidates_scored is None:
            return every
        # a prefix of one permutation, so a smaller n_candidates_scored scores a subset of a larger one
        return jax.random.permutation(key, every)[: self.n_candidates_scored]

    def _expected_utility_improvement(
        self,
        k_fields: PRNGKeyArray,
        k_paths: PRNGKeyArray,
        k_utility: PRNGKeyArray,
        belief: Belief,
        transition: Analytic[Int[Array, ""], Any],
        policy: Policy,
        candidates_scored: Int[Array, " n_candidates_scored"],
        position: Int[Array, ""],
        context: Float[Array, " n_context"],
        candidates: Subset[Any],
    ) -> tuple[Float[Array, " n_candidates_scored"], Int[Array, "n_candidates_scored replan_length"]]:
        """`u(x) = (1 / S M) sum over s, m of [ U(d_h + tau-hat^(s,m)(x)) - U(d_h) ]`, and `tau-hat^(1,1)(x)` for every `x`

        Every observation of d_h is re-measured off phi^(s) at its own state and context, so d_h under phi^(s) is what the
        imagined path extends; where measurements are exact that is d_h itself.
        """
        dataset = belief.dataset  # d_h
        # keyed by index, so the paths of a smaller n_paths_per_field are a subset of a larger one
        path_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(k_paths, jnp.arange(max(self.n_paths_per_field, 1)))
        fields = (
            [belief.mean_field(context)] if self.n_field_samples == 0 else belief.sample_fields(k_fields, self.n_field_samples, context)
        )
        utility_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(k_utility, jnp.arange(len(fields)))  # keyed by index
        improvements, paths_under_first_field = [], None
        for k_score, field in zip(utility_keys, fields, strict=True):  # phi^(s)
            dataset_under_field = Dataset(dataset.state_index, field.at_observations, dataset.is_observed, dataset.context_value)
            dataset_utility = (
                self.utility(k_score, belief, dataset_under_field, candidates) if self.subtract_dataset_utility else jnp.zeros(())
            )
            if self.n_paths_per_field == 0:  # tau-hat = (x): both (1, n_candidates_scored, 1)
                path, before_arrival = candidates_scored[None, :, None], jnp.ones((1, candidates_scored.shape[0], 1), bool)
            else:  # tau-hat^(., m)(x) ~ P(. | z_h, pi(., x), phi^(s)): both (n_paths_per_field, n_candidates_scored, replan_length)
                path, before_arrival = self.planner.roll_paths(path_keys, transition.kernel(field), policy, candidates_scored, position)
            paths_under_first_field = path[0] if paths_under_first_field is None else paths_under_first_field
            # tau-hat as observations: the path's states measured off phi^(s), every one at the decision's own context
            imagined_observations = Dataset(
                path, field.at_every_state[path], before_arrival, jnp.broadcast_to(context, (*path.shape, context.shape[-1]))
            )
            dataset_with_path = (
                dataset_under_field.broadcast(path.shape[:-1]).concat(imagined_observations)
                if self.subtract_dataset_utility
                else imagined_observations
            )  # d_h + tau-hat
            improvements.append(self.utility(k_score, belief, dataset_with_path, candidates) - dataset_utility)  # (M, n_candidates_scored)
        # (S, M, n_candidates_scored) averaged over both sample axes
        return jnp.mean(jnp.stack(improvements), axis=(0, 1)), jnp.asarray(paths_under_first_field)

    def _best_reachable_candidate(
        self,
        key: PRNGKeyArray,
        acquisition_value: Float[Array, " n_candidates_scored"],
        policy: Policy,
        budget_left: Float[Array, ""] | None,
    ) -> Int[Array, ""]:
        """Index of `x* ~ Uniform( argmax over x in delta_L of alpha(x) )`, or of the nearest `x` by `H-hat` when `delta_L` is empty

        delta_L = { x : 0 < H-hat(z_h, x) < L }: within the goal radius nothing is walked, and at H-hat = L nothing arrives before
        the next decision; where the problem keeps a budget, C-hat(z_h, x) <= the budget left as well.
        """
        is_reachable: Bool[Array, " n_candidates_scored"] = (policy.expected_path_length > 0.0) & (
            policy.expected_path_length < self.planner.replan_length
        )
        if budget_left is not None:
            is_reachable &= policy.expected_cost <= budget_left
        among_reachable = jnp.where(is_reachable, acquisition_value, -jnp.inf)
        is_best = among_reachable == jnp.max(among_reachable)  # the maxima among the reachable candidates, tied
        return jnp.where(
            jnp.any(is_reachable),
            jax.random.choice(key, is_best.shape[0], p=is_best / jnp.maximum(jnp.sum(is_best), 1)),  # one of the tied maxima, uniformly
            jnp.argmin(jnp.where(policy.expected_path_length > 0.0, policy.expected_path_length, jnp.inf)),  # the nearest candidate
        )
