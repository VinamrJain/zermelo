"""alpha(x) = w(x) - c |tau| or w(x) / |tau|, where w(x) = E over fields and walks of [ U(D + obs(tau)) - U(D) ]"""

from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from zermelo.interface import Analytic, Subset
from zermelo.methods.waypoint_bo.belief import Belief, Dataset, elements
from zermelo.methods.waypoint_bo.planner import Planner
from zermelo.methods.waypoint_bo.utility import Utility


@register_dataclass
@dataclass(frozen=True)
class Scores:
    """One acquisition pass: what each candidate was worth, which won, and how the pass went"""

    candidate_indices: Int[Array, " n_candidates"]
    """The cells scored, as position domain indices"""

    acquisition_value: Float[Array, " n_candidates"]
    """`alpha` at each of them"""

    waypoint: Int[Array, ""]
    """`x_{n+1}`: the candidate that won"""

    imagined_walk_to_waypoint: Int[Array, " steps"]
    """The winner's route under the first field and walk"""

    waypoint_value_spread: Float[Array, ""]
    """Standard deviation of the winner's value over fields and walks"""

    predicted_steps: Float[Array, ""]
    """The winner's expected step count off the hitting-time table"""

    rolled_steps: Float[Array, ""]
    """The winner's mean rolled walk length"""

    frac_zero_value_candidates: Float[Array, ""]
    """Share of the candidates scored that are worth exactly zero before travel is charged"""

    step_charge: Float[Array, ""]
    """`c`: what one imagined step was charged on this pass"""

    frac_reachable_candidates: Float[Array, ""]
    """Share of the candidates scored whose expected step count fell under the planner's truncation"""


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
    """How many candidates to score: `None` every live one, an integer that many drawn uniformly each decision"""

    improvement: bool
    """Whether to score the held readings and the imagined ones together, less the held ones alone"""

    step_rate: float | None
    """`lambda`: a full-budget trip costs this share of the spread of the candidate values"""

    steps_from: Literal["predicted", "rolled"]
    """Whether travel is counted off the hitting-time table or off the rolled walks"""

    combination: Literal["linear", "fraction"]
    """Whether travel is subtracted from the worth or divides it"""

    def __post_init__(self) -> None:
        if self.n_walks == 0 and self.steps_from == "rolled":
            raise ValueError("steps_from='rolled' needs walks to average; set n_walks above zero or price off the planner")
        if (self.step_rate is None) != (self.combination == "fraction"):
            raise ValueError("a linear combination is priced by step_rate and a fraction takes none; state exactly one of the two")

    def choose(
        self,
        key: PRNGKeyArray,
        belief: Belief,
        transition: Analytic[Int[Array, ""], Any],
        position: Int[Array, ""],
        actions: Int[Array, " n_actions"],
        candidates: Subset[Any],
    ) -> Scores:
        """Every candidate scored under every field and walk, and the argmax over them"""
        k_indices, k_field, k_plan, k_walk, k_utility = jax.random.split(key, 5)
        candidate_indices = jnp.flatnonzero(candidates.live)
        if self.n_candidates is not None:
            # a prefix of one permutation, so a shorter shortlist is a subset of every longer one
            candidate_indices = jax.random.permutation(k_indices, candidate_indices)[: self.n_candidates]
        # keyed by index, so the draws of a smaller n_walks are a subset of a larger one
        walk_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(k_walk, jnp.arange(max(self.n_walks, 1)))
        held, n_scored = belief.data, candidate_indices.shape[0]
        # pi[mu_n], solved once and reused
        mean_policy = self.planner.plan(k_plan, transition.kernel(belief.mean()), candidate_indices, actions)
        utility_samples, predicted_samples, rolled_samples, first_draw_walks = [], [], [], None

        # f-hat^(1..S)
        fields = [belief.mean()] if self.n_fields == 0 else belief.draw(k_field, self.n_fields)
        score_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(k_utility, jnp.arange(len(fields)))  # keyed by index
        for k_score, field in zip(score_keys, fields, strict=True):
            kernel = transition.kernel(field)  # p(.|f-hat)
            values = field(elements(kernel.domain))  # f-hat(z), one row per cell of the position domain
            held_under_field = Dataset(held.z, values[held.z], held.live)  # D_n, its readings re-taken from f-hat
            base_utility = self.utility(k_score, belief, held_under_field, candidates) if self.improvement else jnp.zeros(())  # U(D_n)

            if self.n_walks == 0:  # (1, n_candidates, 1): tau = (x), the destination and no route
                walk, stepped = candidate_indices[None, :, None], jnp.ones((1, n_scored, 1), bool)
            else:  # both (n_walks, n_candidates, steps): tau ~ p(. | z, x, f-hat, pi)
                walk, stepped = self.planner.roll(walk_keys, kernel, mean_policy, position)
            first_draw_walks = walk[0] if first_draw_walks is None else first_draw_walks  # (n_candidates, steps) under f-hat^(1), tau^(1)

            imagined = Dataset(walk, values[walk], stepped)  # obs(tau)
            scored = held_under_field.broadcast(walk.shape[:-1]).concat(imagined) if self.improvement else imagined  # D_n + obs(tau)
            utility_term = self.utility(k_score, belief, scored, candidates) - base_utility  # (n_walks, n_candidates)
            rolled = jnp.sum(stepped, axis=-1).astype(utility_term.dtype)  # |tau|, the steps the walk actually took
            utility_samples.append(utility_term)
            rolled_samples.append(rolled)
            predicted_samples.append(jnp.broadcast_to(mean_policy.hitting_time[position], rolled.shape))  # H(z, x)

        # every stack is (n_fields, n_walks, n_candidates); averaging and spread run over both sample axes
        utility, rolled, predicted = jnp.stack(utility_samples), jnp.stack(rolled_samples), jnp.stack(predicted_samples)
        worth = jnp.mean(utility, axis=(0, 1))  # w(x), one per candidate
        steps = predicted if self.steps_from == "predicted" else rolled
        # c = lambda * (max w - min w) / L. A flat w charges nothing, and the argmax then takes the lowest index.
        rate = 0.0 if self.step_rate is None else self.step_rate
        charge = jnp.zeros(()) if rate == 0.0 else rate * (jnp.max(worth) - jnp.min(worth)) / self.planner.max_steps
        sampled = utility - charge * steps if self.combination == "linear" else utility / jnp.maximum(steps, 1.0)
        value = jnp.mean(sampled, axis=(0, 1))  # alpha(x)
        expected_steps = jnp.mean(predicted, axis=(0, 1))  # H(z, x), averaged over the draws that solved it
        # a candidate already within the planner's radius costs zero steps and has no route to it, so it
        # would win every tie the cost term decides -- scored like the rest, but never chosen
        best = jnp.argmax(jnp.where(expected_steps == 0.0, -jnp.inf, value))
        return Scores(
            candidate_indices=candidate_indices,
            acquisition_value=value,
            waypoint=candidate_indices[best],
            imagined_walk_to_waypoint=jnp.asarray(first_draw_walks)[best],
            waypoint_value_spread=jnp.std(sampled, axis=(0, 1))[best],
            predicted_steps=expected_steps[best],
            rolled_steps=jnp.mean(rolled, axis=(0, 1))[best],
            frac_zero_value_candidates=jnp.mean((worth == 0.0).astype(value.dtype)),
            step_charge=charge,
            frac_reachable_candidates=jnp.mean((expected_steps < self.planner.max_steps).astype(value.dtype)),
        )
