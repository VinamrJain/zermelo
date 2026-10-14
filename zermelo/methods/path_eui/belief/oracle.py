"""`OracleBelief`: the true field in place of a model"""

import dataclasses
from dataclasses import dataclass

import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from zermelo.interface import Domain, Function
from zermelo.methods.path_eui.belief.base import Belief, Dataset, FieldEstimate
from zermelo.methods.path_eui.belief.states import context_parts, coordinates, states_at


@register_dataclass
@dataclass(frozen=True)
class OracleBelief(Belief):
    """The true field in place of a model: the mean is exactly it and the variance is zero"""

    positions: Domain = dataclasses.field(metadata=dict(static=True))
    dataset: Dataset

    field: Function = dataclasses.field(metadata=dict(static=True))
    """The field the world drew"""

    context: dict[str, int] = dataclasses.field(metadata=dict(static=True))
    """The coordinates an observation carries beyond its state, each named and how wide it is"""

    @classmethod
    def empty(cls, positions: Domain, n_observations: int, n_components: int, field: Function, context: dict[str, int]) -> "OracleBelief":
        """Seen nothing, over a buffer of `n_observations` slots, holding the field the world drew"""
        return cls(positions, Dataset.empty(n_observations, n_components, sum(context.values())), field, context)

    @property
    def group_of(self) -> Int[Array, " components"]:
        """Zeros: every component in the one group"""
        return jnp.zeros(self.dataset.measurement.shape[-1], jnp.int32)

    @property
    def measurement_noise(self) -> Float[Array, " groups"]:
        """1, unread: the truth has no spread to weigh it against"""
        return jnp.ones(1)

    def with_prior_mean(self, prior_mean: Function | None) -> "OracleBelief":
        """Unchanged: the truth needs no prior"""
        return self

    def fit(self, dataset: Dataset) -> "OracleBelief":
        return dataclasses.replace(self, dataset=dataset)

    def _at(self, state_index: Int[Array, "*batch n"], context_value: Float[Array, "*batch n n_context"]) -> Float[Array, "*batch n m"]:
        """The field read at each state at its own context"""
        return self.field({**states_at(self.positions, state_index), **context_parts(self.context, context_value)})

    def _estimate(self, context: Float[Array, " n_context"]) -> FieldEstimate:
        """The field at every state at one context, and at the dataset's own observations"""
        n_states = coordinates(self.positions).shape[0]
        table = self._at(jnp.arange(n_states), jnp.broadcast_to(context, (n_states, context.shape[-1])))
        return FieldEstimate(self.positions, table, self._at(self.dataset.state_index, self.dataset.context_value))

    def mean_and_variance(
        self, state_index: Int[Array, "*batch n"], context: Float[Array, " n_context"]
    ) -> tuple[Float[Array, "*batch n m"], Float[Array, "*batch n m"]]:
        """The field's own values, and a variance of zero"""
        mean = self._estimate(context).at_every_state[state_index]
        return mean, jnp.zeros_like(mean)

    def posterior_covariance(
        self, a: Int[Array, "*batch_a n_a"], b: Int[Array, "*batch_b n_b"], context: Float[Array, " n_context"]
    ) -> Float[Array, "*batch groups n_a n_b"]:
        """Zero everywhere: the truth has no spread"""
        return jnp.zeros((*jnp.broadcast_shapes(a.shape[:-1], b.shape[:-1]), 1, a.shape[-1], b.shape[-1]))

    def mean_field(self, context: Float[Array, " n_context"]) -> FieldEstimate:
        return self._estimate(context)

    def sample_fields(self, key: PRNGKeyArray, n_field_samples: int, context: Float[Array, " n_context"]) -> list[FieldEstimate]:
        """The same field every time, whatever the key"""
        return [self._estimate(context)] * n_field_samples
