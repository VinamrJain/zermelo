"""`U(D_n)`: what a set of readings is worth"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.scipy.stats import norm
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from zermelo.interface import Subset
from zermelo.methods.waypoint_bo.belief import Belief, Dataset, coordinates


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


def magnitude(values: Float[Array, "*batch n m"], components: tuple[int, ...]) -> Float[Array, "*batch n"]:
    """||values[components]||"""
    return jnp.linalg.norm(values[..., jnp.asarray(components)], axis=-1)


@dataclass(frozen=True)
class MaxMagnitude(Utility):
    """U(D) = max over the counted rows of ||state_value[components]||, the incumbent beta_n"""

    components: tuple[int, ...]
    """Which components of the field are measured"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        # 0.0 is the infimum of a magnitude, so a set with no counted row scores the floor rather than -inf
        return jnp.max(jnp.where(row_counts_toward_utility(data, candidates), magnitude(data.state_value, self.components), 0.0), axis=-1)


@dataclass(frozen=True)
class SumMagnitude(Utility):
    """U(D) = sum over the counted rows of ||state_value[components]||, the magnitude collected over the set"""

    components: tuple[int, ...]
    """Which components of the field are measured"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        return jnp.sum(jnp.where(row_counts_toward_utility(data, candidates), magnitude(data.state_value, self.components), 0.0), axis=-1)


@dataclass(frozen=True)
class TotalVariance(Utility):
    """U(D) = - sum over the candidate states c and j in components of Var[f_j(c) | the states D was read at]

    U(D) = sum over j in components of [ - sum over c of k_n,g(c, c)  +  trace( k_n,g(C, w) [k_n,g(w, w) + 1e-6 I]^-1 k_n,g(w, C) ) ],   g = group_of[j]
    w = the rows of D past the belief's own buffer, not yet conditioned on, taken as exact readings
    """

    components: tuple[int, ...]
    """Which components of the field count"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        context = query_context(data)
        candidate_states = jnp.flatnonzero(candidates.live)  # (n_candidates,)
        components = jnp.asarray(self.components)
        variance_now = jnp.sum(belief.predict(candidate_states, context)[1][:, components])  # sum over c and components of k_n(c, c)
        imagined_states, imagined_is_live = imagined_rows(belief, data)
        if imagined_states.shape[-1] == 0:
            return jnp.broadcast_to(-variance_now, imagined_states.shape[:-1])
        if imagined_states.ndim == 1:
            return (
                belief.over_components(
                    variance_removed(belief, candidate_states, imagined_states, imagined_is_live, context), self.components
                )
                - variance_now
            )
        # one leading batch element at a time: the covariance to the candidates is (n_candidates, n_candidates, n_imagined) per walk
        return (
            jnp.stack(
                [
                    belief.over_components(variance_removed(belief, candidate_states, states, is_live, context), self.components)
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
) -> Float[Array, "*batch groups"]:
    """trace( k_n(C, w) [k_n(w, w) + 1e-6 I]^-1 k_n(w, C) ) per group, a dead row of `w` made an identity row that removes nothing"""
    both_live = (imagined_is_live[..., :, None] & imagined_is_live[..., None, :])[..., None, :, :]  # (*batch, 1, n_imagined, n_imagined)
    among_imagined = jnp.where(both_live, belief.posterior_covariance(imagined_states, imagined_states, context), 0.0)
    among_imagined = among_imagined + jnp.where(imagined_is_live, 1e-6, 1.0)[..., None, None, :] * jnp.eye(imagined_states.shape[-1])
    to_candidates = belief.posterior_covariance(candidate_states, imagined_states, context)  # (*batch, groups, n_candidates, n_imagined)
    to_candidates = jnp.where(imagined_is_live[..., None, None, :], to_candidates, 0.0)
    whitened = solve_triangular(
        jnp.linalg.cholesky(among_imagined), jnp.swapaxes(to_candidates, -1, -2), lower=True
    )  # (*batch, groups, n_imagined, n_candidates)
    return jnp.sum(whitened**2, axis=(-1, -2))


@dataclass(frozen=True)
class PosteriorSpread(Utility):
    """The posterior standard deviations at the counted rows' states, summed:  sum over rows of sqrt( sum over j in components of var_j )"""

    components: tuple[int, ...]
    """Which components of the field count"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        var = belief.predict(data.state_index, query_context(data))[1][..., jnp.asarray(self.components)]
        spread = jnp.sqrt(jnp.sum(var, axis=-1))  # components independent
        return jnp.sum(jnp.where(row_counts_toward_utility(data, candidates), spread, 0.0), axis=-1)


@dataclass(frozen=True)
class ExpectedImprovement(Utility):
    """The largest `E[(|f(z)| - beta)^+]` over the counted rows' states, f the one component, `beta` the largest magnitude collected so far"""

    components: tuple[int, ...]
    """Which component of the field is measured, exactly one"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        if len(self.components) != 1:
            raise ValueError(
                f"a closed-form improvement of a magnitude is over one component, and {self.components} names {len(self.components)}"
            )
        component = self.components[0]
        collected = belief.data  # the incumbent comes from what has actually been read, not from the scored rows
        # zero where nothing has been read yet, which is the infimum of a magnitude and makes the tails below disjoint
        beta = jnp.max(jnp.where(row_counts_toward_utility(collected, candidates), magnitude(collected.state_value, self.components), 0.0))
        mean, var = belief.predict(data.state_index, query_context(data))
        mu, spread = mean[..., component], jnp.sqrt(jnp.maximum(var[..., component], 1e-12))  # (*batch, n): spread floored to 1e-6
        # |f| clears beta above or below, disjointly for beta >= 0, so the two tails add. Each is
        # `E[(x - c)^+] = gap * Phi(gap / spread) + spread * phi(gap / spread)` at its own gap
        over, under = (mu - beta) / spread, (-mu - beta) / spread
        improvement = (mu - beta) * norm.cdf(over) + (-mu - beta) * norm.cdf(under) + spread * (norm.pdf(over) + norm.pdf(under))
        return jnp.max(jnp.where(row_counts_toward_utility(data, candidates), improvement, 0.0), axis=-1)  # 0.0 floors an improvement


@dataclass(frozen=True)
class UpperConfidence(Utility):
    """The largest optimistic magnitude `||mu(z)[components]|| + c * s(z)` at the counted rows' states"""

    c: float
    """How much of the spread to add"""

    components: tuple[int, ...]
    """Which components of the field are measured"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        mean, var = belief.predict(data.state_index, query_context(data))
        components = jnp.asarray(self.components)
        bound = jnp.linalg.norm(mean[..., components], axis=-1) + self.c * jnp.sqrt(jnp.sum(var[..., components], axis=-1))
        return jnp.max(jnp.where(row_counts_toward_utility(data, candidates), bound, 0.0), axis=-1)  # 0.0 floors a magnitude


@dataclass(frozen=True)
class TargetsReached(Utility):
    """U(D) = number of candidates within `radius` of some counted row's state, on embedded coordinates"""

    radius: float
    """How near a candidate a row must sit to reach it"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        with jax.ensure_compile_time_eval():
            coords = coordinates(candidates.base)  # (n_states, k)
            targets = coords[jnp.flatnonzero(candidates.live)]  # (n_candidates, k)
        gap = coords[data.state_index][..., None, :] - targets  # (*batch, n, n_candidates, k)
        near = jnp.sum(gap**2, axis=-1) <= self.radius**2  # (*batch, n, n_candidates)
        return jnp.sum(jnp.any(near & data.row_is_written[..., None], axis=-2), axis=-1).astype(float)


@dataclass(frozen=True)
class Uniform(Utility):
    """A uniform draw per scored dataset, ignoring the data"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        return jax.random.uniform(key, data.state_index.shape[:-1])


def half_log_det(covariance: Float[Array, "*batch groups n n"], noise: Float[Array, " groups"]) -> Float[Array, "*batch groups"]:
    """(1/2) log det( I + covariance / noise^2 ) per group"""
    chol = jnp.linalg.cholesky(covariance / noise[:, None, None] ** 2 + jnp.eye(covariance.shape[-1]))
    return jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)


def imagined_rows(belief: Belief, data: Dataset) -> tuple[Int[Array, "*batch n_imagined"], Bool[Array, "*batch n_imagined"]]:
    """The rows of `data` past the belief's own buffer, and which of them are written"""
    n_conditioned = belief.data.state_index.shape[0]
    return data.state_index[..., n_conditioned:], data.row_is_written[..., n_conditioned:]


@dataclass(frozen=True)
class InformationGain(Utility):
    """U(D) = sum over j in components of (1/2) log det( I + K_n,g(w, w) / noise_g^2 ),   g = group_of[j]: the information the imagined
    rows carry about those components

    w = the rows of D past the belief's own buffer, a dead row made an identity row that carries nothing
    """

    components: tuple[int, ...]
    """Which components of the field count"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        context = query_context(data)
        imagined_states, imagined_is_live = imagined_rows(belief, data)
        if imagined_states.shape[-1] == 0:
            return jnp.zeros(imagined_states.shape[:-1])
        both_live = (imagined_is_live[..., :, None] & imagined_is_live[..., None, :])[..., None, :, :]
        among_imagined = jnp.where(both_live, belief.posterior_covariance(imagined_states, imagined_states, context), 0.0)
        return belief.over_components(half_log_det(among_imagined, belief.reading_noise), self.components)


@dataclass(frozen=True)
class RegionInformation(Utility):
    """U(D) = sum over j in components of (1/2) [ log det( I + K_n,g(w, w) / noise_g^2 ) - log det( I + S_g / noise_g^2 ) ],   g = group_of[j]:
    the information the imagined rows carry about those components at the candidate states C

    S_g = K_n,g(w, w) - K_n,g(w, C) [K_n,g(C, C) + 1e-6 I]^-1 K_n,g(C, w)
    w = the rows of D past the belief's own buffer, a dead row made an identity row that carries nothing
    """

    components: tuple[int, ...]
    """Which components of the field count"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        context = query_context(data)
        imagined_states, imagined_is_live = imagined_rows(belief, data)
        if imagined_states.shape[-1] == 0:
            return jnp.zeros(imagined_states.shape[:-1])
        candidate_states = jnp.flatnonzero(candidates.live)  # (n_candidates,)
        among_candidates = belief.posterior_covariance(candidate_states, candidate_states, context)  # (groups, n_candidates, n_candidates)
        chol_candidates = jnp.linalg.cholesky(among_candidates + 1e-6 * jnp.eye(candidate_states.shape[0]))
        inverse_candidates = jnp.vectorize(lambda chol: cho_solve((chol, True), jnp.eye(chol.shape[0])), signature="(n,n)->(n,n)")(
            chol_candidates
        )  # (groups, n_candidates, n_candidates): [K_n(C, C) + 1e-6 I]^-1

        def one(states: Int[Array, "*batch n_imagined"], is_live: Bool[Array, "*batch n_imagined"]) -> Float[Array, " *batch"]:
            """The bracket summed over the components, one per leading batch element"""
            both_live = (is_live[..., :, None] & is_live[..., None, :])[..., None, :, :]
            among_imagined = jnp.where(both_live, belief.posterior_covariance(states, states, context), 0.0)
            to_candidates = jnp.where(
                is_live[..., None, :, None], belief.posterior_covariance(states, candidate_states, context), 0.0
            )  # (*batch, groups, n_imagined, n_candidates)
            explained = jnp.einsum("...bic,bcd,...bjd->...bij", to_candidates, inverse_candidates, to_candidates)
            noise = belief.reading_noise
            return belief.over_components(
                half_log_det(among_imagined, noise) - half_log_det(among_imagined - explained, noise), self.components
            )

        if imagined_states.ndim <= 2:
            return one(imagined_states, imagined_is_live)
        # one leading batch element at a time: the covariance to the candidates is (n_candidates, n_imagined, n_candidates) per walk
        return jnp.stack([one(states, is_live) for states, is_live in zip(imagined_states, imagined_is_live, strict=True)])
