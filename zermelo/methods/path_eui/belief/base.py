"""The contract: `D_n` as a `Dataset`, one field of a belief's as a `FieldEstimate`, and what every model of the field answers"""

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from zermelo.interface import Domain, Function
from zermelo.methods.path_eui.belief.states import indexer


@register_dataclass
@dataclass(frozen=True)
class Dataset:
    """`d`: the observations collected, as a buffer of `n` slots filled in order"""

    state_index: Int[Array, "*batch observations"]
    """`z`: the position domain's index of the state each observation was made at"""

    measurement: Float[Array, "*batch observations components"]
    """`w`: what was measured there, or what a sampled field says is there"""

    is_observed: Bool[Array, "*batch observations"]
    """True where the slot holds an observation; False on an empty slot, and on a path's steps after it reached its target"""

    context_value: Float[Array, "*batch observations n_context"]
    """The coordinates each observation carries beyond its state, `n_context` wide and zero wide where there are none"""

    @classmethod
    def empty(cls, n_observations: int, n_components: int, n_context: int) -> "Dataset":
        """`d_0`: a buffer of `n_observations` slots, none holding an observation"""
        return cls(
            jnp.zeros(n_observations, jnp.int32),
            jnp.zeros((n_observations, n_components)),
            jnp.zeros(n_observations, bool),
            jnp.zeros((n_observations, n_context)),
        )

    def add_observation(
        self, slot: int, state_index: Int[Array, ""], measurement: Float[Array, " components"], context_value: Float[Array, " n_context"]
    ) -> "Dataset":
        """`d + o`: this dataset with the observation `o = (state, measurement, context)` written into `slot`, raising past the buffer's end"""
        if not 0 <= slot < self.state_index.shape[0]:
            raise IndexError(f"slot {slot} of a {self.state_index.shape[0]}-slot buffer: the agent's horizon is shorter than the episode's")
        return Dataset(
            self.state_index.at[slot].set(state_index),
            self.measurement.at[slot].set(measurement),
            self.is_observed.at[slot].set(True),
            self.context_value.at[slot].set(context_value),
        )

    def concat(self, other: "Dataset") -> "Dataset":
        """The two laid end to end along the observation axis"""
        return Dataset(
            jnp.concatenate([self.state_index, other.state_index], axis=-1),
            jnp.concatenate([self.measurement, other.measurement], axis=-2),
            jnp.concatenate([self.is_observed, other.is_observed], axis=-1),
            jnp.concatenate([self.context_value, other.context_value], axis=-2),
        )

    def broadcast(self, batch: tuple[int, ...]) -> "Dataset":
        """This dataset repeated over the leading axes `batch`"""
        return Dataset(
            jnp.broadcast_to(self.state_index, (*batch, *self.state_index.shape)),
            jnp.broadcast_to(self.measurement, (*batch, *self.measurement.shape)),
            jnp.broadcast_to(self.is_observed, (*batch, *self.is_observed.shape)),
            jnp.broadcast_to(self.context_value, (*batch, *self.context_value.shape)),
        )


@register_dataclass
@dataclass(frozen=True)
class FieldEstimate:
    """One field of a belief's, its mean or a sample: its values at every state at one context, and at the dataset's own observations"""

    positions: Domain = dataclasses.field(metadata=dict(static=True))

    at_every_state: Float[Array, "n_states components"]
    """`f(state, context)` at every state, at the one context the field was read at"""

    at_observations: Float[Array, "observations components"]
    """`f(state_i, context_i)` at each slot of the belief's dataset, observed or not"""

    def __call__(self, x: Any) -> Float[Array, "*batch components"]:
        """`table` at whatever state this is called on"""
        return self.at_every_state[indexer(self.positions)(x)]


class Belief(ABC):
    """A model of the field `f: state -> R^components` over `positions`, conditioned on `dataset`

    Components modelled together form a group; groups are independent of each other.
    """

    positions: Domain
    """The states the field is defined on: finite, and handing out coordinates"""

    dataset: Dataset

    @property
    @abstractmethod
    def group_of(self) -> Int[Array, " components"]:
        """Which group models each component, groups numbered from 0"""

    @property
    @abstractmethod
    def measurement_noise(self) -> Float[Array, " groups"]:
        """Standard deviation of a measurement's error, one per group"""

    def over_components(self, per_group: Float[Array, "*batch groups"], components: tuple[int, ...]) -> Float[Array, " *batch"]:
        """sum over j in `components` of per_group[..., group_of[j]]"""
        return jnp.sum(jnp.take(per_group, self.group_of[jnp.asarray(components)], axis=-1), axis=-1)

    @abstractmethod
    def with_prior_mean(self, prior_mean: Function | None) -> Self:
        """This belief about `prior_mean`, a function of a state together with its named context parts; `None` is zero"""

    @abstractmethod
    def fit(self, dataset: Dataset) -> Self:
        """This belief conditioned on `dataset`, one dataset and never a batch, its own parameters re-estimated from it where configured"""

    @abstractmethod
    def mean_and_variance(
        self, state_index: Int[Array, "*batch observations"], context: Float[Array, " n_context"]
    ) -> tuple[Float[Array, "*batch observations components"], Float[Array, "*batch observations components"]]:
        """Posterior mean and variance of each component at those states, at one context"""

    @abstractmethod
    def posterior_covariance(
        self, a: Int[Array, "*batch_a n_a"], b: Int[Array, "*batch_b n_b"], context: Float[Array, " n_context"]
    ) -> Float[Array, "*batch groups n_a n_b"]:
        """`k_n(a, b)` of each group for every state `a` of the first set and `b` of the second, at one context; the batch axes broadcast"""

    @abstractmethod
    def mean_field(self, context: Float[Array, " n_context"]) -> FieldEstimate:
        """`mu_n`, the posterior mean field at one context"""

    @abstractmethod
    def sample_fields(self, key: PRNGKeyArray, n_field_samples: int, context: Float[Array, " n_context"]) -> list[FieldEstimate]:
        """`n_field_samples` fields drawn from the posterior at one context"""
