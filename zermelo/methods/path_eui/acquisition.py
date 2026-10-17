"""`alpha(x)` over candidate targets `x`, and `x*` the target chosen

z_h the state decided from, d_h the dataset so far, phi-hat = E[phi | d_h], pi = pi[phi-hat] the policy under it,
g(z_h) the states within rho of z_h, c(z, a) >= 0 a step's cost:

    u(x)     = (1 / S M) sum over s, m of [ U(d_h + tau-hat^(s,m)(x)) - U(d_h) ],   tau-hat^(s,m)(x) a path under pi(., x) and phi^(s) ~ P(phi | d_h)
    C(x)     = (1 / S M) sum over s, m of sum over t of c(z_t, pi(z_t, x)),          z_t the states of tau-hat^(s,m)(x), z_0 = z_h
    alpha(x) = (1 - lambda) u~(x) - lambda C~(x),                                    u~, C~ = u, C rescaled to [0, 1]
    x*       = argmax over x not in g(z_h) of alpha(x),                              ties broken by a fixed rule

A candidate is a state the rule may score, a target one the planner steers to, and x* the target chosen.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from zermelo.interface import Analytic, Subset, TransitionKernel
from zermelo.methods.path_eui.belief import Belief, Dataset, FieldEstimate
from zermelo.methods.path_eui.planner import Planner, Policy, index_in_actions
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
    """Which `x` are scored: `None` every candidate, an integer that many drawn uniformly"""

    subtract_dataset_utility: bool
    """`U(d_h + tau-hat) - U(d_h)` when True, `U(tau-hat)` when False"""

    lambda_: float
    """`lambda` in [0, 1): the weight of `C` against the rescaled `u`"""

    def choose_target(
        self,
        key: PRNGKeyArray,
        belief: Belief,
        transition: Analytic[Int[Array, ""], Any],
        position: Int[Array, ""],
        actions: Int[Array, " n_actions"],
        candidates: Subset[Any],
        context: Float[Array, " n_context"],
    ) -> TargetChoice:
        """`(belief, z_h = position, ...) -> TargetChoice` by the module's rule"""
        k_candidates, k_fields, k_paths, k_utility = jax.random.split(key, 4)

        # plan to every candidate once, under the mean field
        candidates_scored = self._candidates_to_score(k_candidates, candidates)
        mean_kernel = transition.kernel(belief.mean_field(context))
        policy = self.planner.policy_to_targets(mean_kernel, candidates_scored, actions, position)

        # imagine M paths to every candidate under each of S fields, then score them
        fields = (
            [belief.mean_field(context)] if self.n_field_samples == 0 else belief.sample_fields(k_fields, self.n_field_samples, context)
        )
        paths, step_taken_mask = self._imagine_paths(k_paths, transition, fields, policy, candidates_scored, position)
        utility_improvement = self._expected_utility_improvement(k_utility, belief, fields, paths, step_taken_mask, context, candidates)
        path_cost_rescaled = self._expected_path_cost(mean_kernel, actions, policy, paths, step_taken_mask, position)

        # x not in g(z_h), i.e. H-hat(z_h, x) > 0; every x if none is
        eligible_mask = (policy.expected_path_length > 0.0) | jnp.all(policy.expected_path_length == 0.0)

        # u~ = (u - min u) / (max u - min u), min and max over eligible x; u~ = 0 if max u = min u
        u_min = jnp.min(jnp.where(eligible_mask, utility_improvement, jnp.inf))
        u_max = jnp.max(jnp.where(eligible_mask, utility_improvement, -jnp.inf))
        u_rescaled = (utility_improvement - u_min) / jnp.where(u_max > u_min, u_max - u_min, jnp.inf)
        # alpha = (1 - lambda) u~ - lambda C~
        acquisition_value = (1.0 - self.lambda_) * u_rescaled - self.lambda_ * path_cost_rescaled

        chosen_index = self._argmax_with_tie_break(acquisition_value, policy.expected_path_length, eligible_mask)
        return TargetChoice(
            candidate_indices=candidates_scored,
            acquisition_value=acquisition_value,
            target=candidates_scored[chosen_index],
            imagined_path_to_target=paths[0, 0, chosen_index],
            policy_to_target=Policy(policy.action[:, chosen_index, None], policy.expected_path_length[chosen_index, None]),
        )

    def _candidates_to_score(self, key: PRNGKeyArray, candidates: Subset[Any]) -> Int[Array, " n_candidates_scored"]:
        """The `x` scored: every candidate, or `n_candidates_scored` of them drawn uniformly"""
        every = jnp.flatnonzero(candidates.live)
        if self.n_candidates_scored is None:
            return every
        # a prefix of one permutation, so a smaller n_candidates_scored scores a subset of a larger one
        return jax.random.permutation(key, every)[: self.n_candidates_scored]

    def _imagine_paths(
        self,
        k_paths: PRNGKeyArray,
        transition: Analytic[Int[Array, ""], Any],
        fields: list[FieldEstimate],
        policy: Policy,
        candidates_scored: Int[Array, " n_candidates_scored"],
        position: Int[Array, ""],
    ) -> tuple[
        Int[Array, "n_fields n_paths n_candidates_scored replan_length"], Bool[Array, "n_fields n_paths n_candidates_scored replan_length"]
    ]:
        """`tau-hat^(s,m)(x)` for every field `phi^(s)` and candidate `x`, and which of its steps were taken"""
        n_fields, n_candidates = len(fields), candidates_scored.shape[0]
        if self.n_paths_per_field == 0:  # tau-hat = (x): the candidate alone, counted as observed
            return jnp.broadcast_to(candidates_scored[:, None], (n_fields, 1, n_candidates, 1)), jnp.ones(
                (n_fields, 1, n_candidates, 1), bool
            )
        # one key per path index, so fewer paths are a subset of more
        path_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(k_paths, jnp.arange(self.n_paths_per_field))
        rolled = [self.planner.roll_paths(path_keys, transition.kernel(field), policy, candidates_scored, position) for field in fields]
        return jnp.stack([path for path, _ in rolled]), jnp.stack([step_taken_mask for _, step_taken_mask in rolled])

    def _expected_utility_improvement(
        self,
        k_utility: PRNGKeyArray,
        belief: Belief,
        fields: list[FieldEstimate],
        paths: Int[Array, "n_fields n_paths n_candidates_scored replan_length"],
        step_taken_mask: Bool[Array, "n_fields n_paths n_candidates_scored replan_length"],
        context: Float[Array, " n_context"],
        candidates: Subset[Any],
    ) -> Float[Array, " n_candidates_scored"]:
        """`u(x)`: `U(d_h + tau-hat) - U(d_h)`, averaged over fields and paths"""
        utility_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(k_utility, jnp.arange(len(fields)))  # one key per field
        improvements = []
        for k_score, field, field_paths, field_step_taken_mask in zip(utility_keys, fields, paths, step_taken_mask, strict=True):
            # d_h re-measured under this field (unchanged if measurements are exact)
            dataset = Dataset(belief.dataset.state_index, field.at_observations, belief.dataset.is_observed, belief.dataset.context_value)
            # the paths as observations: each state measured under this field, at the current context
            path_data = Dataset(
                field_paths,
                field.at_every_state[field_paths],
                field_step_taken_mask,
                jnp.broadcast_to(context, (*field_paths.shape, context.shape[-1])),
            )
            if self.subtract_dataset_utility:  # U(d_h + tau-hat) - U(d_h)
                with_path = dataset.broadcast(field_paths.shape[:-1]).concat(path_data)
                improvements.append(
                    self.utility(k_score, belief, with_path, candidates) - self.utility(k_score, belief, dataset, candidates)
                )
            else:  # U(tau-hat)
                improvements.append(self.utility(k_score, belief, path_data, candidates))
        return jnp.mean(jnp.stack(improvements), axis=(0, 1))  # (S, M, n_candidates_scored) -> (n_candidates_scored,)

    def _expected_path_cost(
        self,
        kernel: TransitionKernel[Int[Array, ""]],
        actions: Int[Array, " n_actions"],
        policy: Policy,
        paths: Int[Array, "n_fields n_paths n_candidates_scored replan_length"],
        step_taken_mask: Bool[Array, "n_fields n_paths n_candidates_scored replan_length"],
        position: Int[Array, ""],
    ) -> Float[Array, " n_candidates_scored"]:
        """`C~(x)`: the cost of the paths to `x`, averaged over fields and paths, rescaled to [0, 1]"""
        cost_per_action = [c for a in actions if (c := kernel.step_cost(jnp.asarray(a))) is not None]
        if self.n_paths_per_field == 0 or len(cost_per_action) < actions.shape[0]:  # no path walked, or every step free
            return jnp.zeros(paths.shape[2])
        cost_table = jnp.stack(cost_per_action)  # (n_actions, n_states): c(z, a)
        # the state each step starts from, and the action pi takes there
        step_start = jnp.concatenate([jnp.broadcast_to(position, (*paths.shape[:-1], 1)), paths[..., :-1]], axis=-1)
        action_taken = index_in_actions(actions, policy.action[step_start, jnp.arange(paths.shape[2])[:, None]].astype(actions.dtype))
        step_cost = jnp.where(step_taken_mask, cost_table[action_taken, step_start], 0.0)  # (S, M, n_candidates_scored, L)
        # C~ = C / (L c_max), c_max = max over z, a of c(z, a): a path of L costliest steps scores 1
        return jnp.mean(jnp.sum(step_cost, axis=-1), axis=(0, 1)) / (self.planner.replan_length * jnp.maximum(jnp.max(cost_table), 1e-12))

    def _argmax_with_tie_break(
        self,
        acquisition_value: Float[Array, " n_candidates_scored"],
        expected_path_length: Float[Array, " n_candidates_scored"],
        eligible_mask: Bool[Array, " n_candidates_scored"],
    ) -> Int[Array, ""]:
        """Index of the eligible candidate of highest `alpha`, ties broken by a fixed rule"""
        # B = argmax over eligible x of alpha(x); x* = the first x in B of least H-hat(z_h, x)
        best_mask = eligible_mask & (acquisition_value == jnp.max(jnp.where(eligible_mask, acquisition_value, -jnp.inf)))
        return jnp.argmin(jnp.where(best_mask, expected_path_length, jnp.inf))
