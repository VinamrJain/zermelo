"""alpha(x) = w(x) - c H(z, x) or w(x) - H(z, x), where w(x) = E over fields and walks of [ U(D + obs(tau)) - U(D) ]

plus, where an information utility I is given, lambda * L / (max I - min I) * I(x), the range over the candidates within L steps, for L the planner's
replan_every, and only such a candidate may win then
"""

from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from zermelo.interface import Analytic, Subset
from zermelo.methods.waypoint_bo.belief import Belief, Dataset
from zermelo.methods.waypoint_bo.planner import Planner, Policy
from zermelo.methods.waypoint_bo.utility import TripUtility, Utility


@register_dataclass
@dataclass(frozen=True)
class Scores:
    """One acquisition pass: what each candidate was worth, which won, and how the pass went"""

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
    """Scores candidate waypoints and takes the best"""

    utility: Utility
    planner: Planner

    n_fields: int
    """`S`: posterior fields drawn per decision, or the posterior mean at zero"""

    n_walks: int
    """`M`: walks imagined per field, or the destination alone at zero"""

    n_candidates: int | None
    """How many candidates to score: `None` every live one, an integer that many drawn uniformly each decision, 0 the target alone"""

    information: Utility | None
    """`I`: what the readings along a walk tell, added to the worth at `information_rate` once put in the worth's units"""

    information_rate: float | None
    """`lambda`: the most informative candidate within L steps is credited this many L of travel"""

    improvement: bool
    """Whether to score the collected readings and the imagined ones together, less the collected ones alone"""

    step_rate: float | None
    """`lambda`: a full-budget trip costs this share of the spread of the candidate values"""

    combination: Literal["linear", "steps"]
    """Whether travel is subtracted from the worth at `step_rate`, or as it is, the worth being in steps"""

    def __post_init__(self) -> None:
        if (self.step_rate is None) != (self.combination == "steps"):
            raise ValueError("a linear combination is priced by step_rate and steps takes none; state exactly one of the two")
        if (self.information is None) != (self.information_rate is None):
            raise ValueError("an information utility is added at information_rate; state both or neither")

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
    ) -> Scores:
        """Every candidate scored under every field and walk, and the argmax over them; `target` is where a trip utility steers"""
        if (isinstance(self.utility, TripUtility) or self.n_candidates == 0) and target is None:
            raise ValueError("a trip utility scores the trip to a target, and this decision names none")
        k_indices, k_field, k_plan, k_walk, k_utility = jax.random.split(key, 5)
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
        mean_policy = self.planner.plan(k_plan, transition.kernel(belief.mean(context)), candidate_indices, actions)
        utility_samples, information_samples, first_draw_walks = [], [], None

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

            def worth_of(data: Dataset) -> Float[Array, " *batch"]:
                """U(data), the trip to `target` where the utility scores one"""
                if isinstance(self.utility, TripUtility):
                    return self.utility.remaining(k_score, belief, data, transition, field, jnp.asarray(target), actions)
                return self.utility(k_score, belief, data, candidates)

            base_utility = worth_of(collected_under_field) if self.improvement else jnp.zeros(())  # U(D_n)

            if self.n_walks == 0:  # (1, n_candidates, 1): tau = (x), the destination and no route
                walk, moved = candidate_indices[None, :, None], jnp.ones((1, n_scored, 1), bool)
            else:  # both (n_walks, n_candidates, time_steps): tau ~ p(. | z, x, f-hat, pi)
                walk, moved = self.planner.roll(walk_keys, kernel, mean_policy, position)
            first_draw_walks = (
                walk[0] if first_draw_walks is None else first_draw_walks
            )  # (n_candidates, time_steps) under f-hat^(1), tau^(1)

            # obs(tau) = {(z_i, f-hat(z_i, context), context)}: the walk's states read off f-hat, every row at the decision's own context
            imagined = Dataset(walk, values[walk], moved, jnp.broadcast_to(context, (*walk.shape, context.shape[-1])))
            scored = collected_under_field.broadcast(walk.shape[:-1]).concat(imagined) if self.improvement else imagined  # D_n + obs(tau)
            utility_samples.append(worth_of(scored) - base_utility)  # (n_walks, n_candidates)
            if self.information is not None:
                information_samples.append(
                    self.information(k_score, belief, scored, candidates)
                )  # I(D_n + obs(tau)), (n_walks, n_candidates)

        # every stack is (n_fields, n_walks, n_candidates); averaging runs over both sample axes
        worth = jnp.mean(jnp.stack(utility_samples), axis=(0, 1))  # w(x), one per candidate
        expected_spend = mean_policy.hitting_cost[position]  # H(z, x)
        # c = lambda * (max w - min w) / B, for B what a full-budget route may spend. A flat w charges
        # nothing, and the argmax then takes the lowest index.
        rate = 0.0 if self.step_rate is None else self.step_rate
        charge = jnp.zeros(()) if rate == 0.0 else rate * (jnp.max(worth) - jnp.min(worth)) / self.planner.cost_budget
        value = worth - charge * expected_spend if self.combination == "linear" else worth - expected_spend  # alpha(x)
        allowed = expected_spend > 0.0  # a candidate within the planner's radius spends nothing and would win every tie: never chosen
        if self.information is not None and self.information_rate is not None:
            information = jnp.mean(jnp.stack(information_samples), axis=(0, 1))  # I(x)
            reach = expected_spend <= self.planner.replan_every  # the candidates a walk gets to before the next decision
            # lambda * L / (max I - min I) over the reachable candidates: information in steps, none where I is flat there
            spread = jnp.max(jnp.where(reach, information, -jnp.inf)) - jnp.min(jnp.where(reach, information, jnp.inf))
            charge = jnp.where(spread > 0.0, self.information_rate * self.planner.replan_every / jnp.maximum(spread, 1e-12), 0.0)
            value, allowed = value + charge * information, allowed & reach
        best = jnp.argmax(jnp.where(allowed, value, -jnp.inf))
        return Scores(
            candidate_indices=candidate_indices,
            acquisition_value=value,
            waypoint=candidate_indices[best],
            imagined_walk_to_waypoint=jnp.asarray(first_draw_walks)[best],
            route=Policy(mean_policy.act[:, best, None], mean_policy.hitting_cost[:, best, None]),
        )
