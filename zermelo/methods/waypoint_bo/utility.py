"""`U(D_n)`: what a set of readings is worth"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.stats import norm
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from zermelo.interface import Subset
from zermelo.methods.waypoint_bo.belief import Belief, Dataset


class Utility(ABC):
    """Scores a dataset under a belief's model, considering only the written rows that sit on a candidate state"""

    @abstractmethod
    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        """What `data` is worth, one score per leading batch axis of `data`"""


def row_counts_toward_utility(data: Dataset, candidates: Subset[Any]) -> Bool[Array, "*batch n"]:
    """Which rows a utility counts: written, and read on a candidate state"""
    return data.row_is_written & candidates.live[data.state_index]


def query_context(data: Dataset) -> Float[Array, " n_context"]:
    """The context the belief is read at to score `data`: its last row's

    context_value[-1, ..., -1, :], one -1 per leading batch axis of a (*batch, n, n_context) array.
    """
    return data.context_value[(-1,) * (data.context_value.ndim - 1)]


@dataclass(frozen=True)
class MaxMagnitude(Utility):
    """U(D) = max over the counted rows of ||state_value||, the incumbent beta_n"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        # 0.0 is the infimum of a magnitude, so a set with no counted row scores the floor rather than -inf
        return jnp.max(jnp.where(row_counts_toward_utility(data, candidates), jnp.linalg.norm(data.state_value, axis=-1), 0.0), axis=-1)


@dataclass(frozen=True)
class SumMagnitude(Utility):
    """U(D) = sum over the counted rows of ||state_value||, the magnitude collected over the set"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        return jnp.sum(jnp.where(row_counts_toward_utility(data, candidates), jnp.linalg.norm(data.state_value, axis=-1), 0.0), axis=-1)


@dataclass(frozen=True)
class TotalVariance(Utility):
    """U(D) = - sum over the candidate states c of Var[f(c) | the states D was read at], f one component of the field

    U(D) = - sum over c of k_n(c, c)  +  trace( k_n(C, w) [k_n(w, w) + 1e-6 I]^-1 k_n(w, C) )
    w = the rows of D past the belief's own buffer, not yet conditioned on, taken as exact readings
    """

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        context = query_context(data)
        candidate_states = jnp.flatnonzero(candidates.live)  # (n_candidates,)
        variance_now = jnp.sum(belief.predict(candidate_states, context)[1][:, 0])  # sum over c of k_n(c, c)
        n_conditioned = belief.data.state_index.shape[0]
        imagined_states, imagined_is_live = data.state_index[..., n_conditioned:], data.row_is_written[..., n_conditioned:]
        if imagined_states.shape[-1] == 0:
            return jnp.broadcast_to(-variance_now, imagined_states.shape[:-1])
        if imagined_states.ndim == 1:
            return variance_removed(belief, candidate_states, imagined_states, imagined_is_live, context) - variance_now
        # one leading batch element at a time: the block to the candidates is (n_candidates, n_candidates, n_imagined) per walk
        return (
            jnp.stack(
                [
                    variance_removed(belief, candidate_states, states, is_live, context)
                    for states, is_live in zip(imagined_states, imagined_is_live, strict=True)
                ]
            )
            - variance_now
        )


def variance_removed(
    belief: Belief,
    candidate_states: Int[Array, " n_candidates"],
    imagined_states: Int[Array, "*batch n_imagined"],
    imagined_is_live: Bool[Array, "*batch n_imagined"],
    context: Float[Array, " n_context"],
) -> Float[Array, " *batch"]:
    """trace( k_n(C, w) [k_n(w, w) + 1e-6 I]^-1 k_n(w, C) ), a dead row of `w` made an identity row that removes nothing"""
    both_live = imagined_is_live[..., :, None] & imagined_is_live[..., None, :]  # (*batch, n_imagined, n_imagined)
    among_imagined = jnp.where(both_live, belief.posterior_covariance(imagined_states, imagined_states, context), 0.0)
    among_imagined = among_imagined + jnp.where(imagined_is_live, 1e-6, 1.0)[..., None, :] * jnp.eye(imagined_states.shape[-1])
    to_candidates = belief.posterior_covariance(candidate_states, imagined_states, context)  # (*batch, n_candidates, n_imagined)
    to_candidates = jnp.where(imagined_is_live[..., None, :], to_candidates, 0.0)
    whitened = solve_triangular(
        jnp.linalg.cholesky(among_imagined), jnp.swapaxes(to_candidates, -1, -2), lower=True
    )  # (*batch, n_imagined, n_candidates)
    return jnp.sum(whitened**2, axis=(-1, -2))


@dataclass(frozen=True)
class PosteriorSpread(Utility):
    """The posterior standard deviations at the counted rows' states, summed"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        spread = jnp.sqrt(
            jnp.sum(belief.predict(data.state_index, query_context(data))[1], axis=-1)
        )  # sqrt(sum_j var_j), components independent
        return jnp.sum(jnp.where(row_counts_toward_utility(data, candidates), spread, 0.0), axis=-1)


@dataclass(frozen=True)
class ExpectedImprovement(Utility):
    """The largest `E[(|f(z)| - beta)^+]` over the counted rows' states, `beta` the largest magnitude collected so far"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        if data.state_value.shape[-1] != 1:
            raise ValueError(
                f"a closed-form improvement of a magnitude needs a one-component field, and this one has {data.state_value.shape[-1]}"
            )
        collected = belief.data  # the incumbent comes from what has actually been read, not from the scored rows
        # zero where nothing has been read yet, which is the infimum of a magnitude and makes the tails below disjoint
        beta = jnp.max(jnp.where(row_counts_toward_utility(collected, candidates), jnp.linalg.norm(collected.state_value, axis=-1), 0.0))
        mean, var = belief.predict(data.state_index, query_context(data))
        mu, spread = mean[..., 0], jnp.sqrt(jnp.maximum(var[..., 0], 1e-12))  # (*batch, n): spread floored to 1e-6
        # |f| clears beta above or below, disjointly for beta >= 0, so the two tails add. Each is
        # `E[(x - c)^+] = gap * Phi(gap / spread) + spread * phi(gap / spread)` at its own gap
        over, under = (mu - beta) / spread, (-mu - beta) / spread
        improvement = (mu - beta) * norm.cdf(over) + (-mu - beta) * norm.cdf(under) + spread * (norm.pdf(over) + norm.pdf(under))
        return jnp.max(jnp.where(row_counts_toward_utility(data, candidates), improvement, 0.0), axis=-1)  # 0.0 floors an improvement


@dataclass(frozen=True)
class UpperConfidence(Utility):
    """The largest optimistic magnitude `||mu(z)|| + c * s(z)` at the counted rows' states"""

    c: float
    """How much of the spread to add"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        mean, var = belief.predict(data.state_index, query_context(data))
        bound = jnp.linalg.norm(mean, axis=-1) + self.c * jnp.sqrt(jnp.sum(var, axis=-1))
        return jnp.max(jnp.where(row_counts_toward_utility(data, candidates), bound, 0.0), axis=-1)  # 0.0 floors a magnitude


@dataclass(frozen=True)
class Uniform(Utility):
    """A uniform draw per scored dataset, ignoring the data"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        return jax.random.uniform(key, data.state_index.shape[:-1])
