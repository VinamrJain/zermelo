"""`OracleBelief`: the true field in place of a model"""

import dataclasses
from dataclasses import dataclass

import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from zermelo.interface import Domain, Function
from zermelo.methods.waypoint_bo.belief.base import Belief, BeliefField, Dataset
from zermelo.methods.waypoint_bo.belief.states import context_parts, coordinates, states_at


@register_dataclass
@dataclass(frozen=True)
class OracleBelief(Belief):
    """The true field in place of a model: the mean is exactly it and the variance is zero"""

    positions: Domain = dataclasses.field(metadata=dict(static=True))
    data: Dataset

    field: Function = dataclasses.field(metadata=dict(static=True))
    """The field the world drew"""

    context: dict[str, int] = dataclasses.field(metadata=dict(static=True))
    """The coordinates a reading carries beyond its state, each named and how wide it is"""

    @classmethod
    def empty(cls, positions: Domain, rows: int, m: int, field: Function, context: dict[str, int]) -> "OracleBelief":
        """Seen nothing, over a buffer of `rows` rows, holding the field the world drew"""
        return cls(positions, Dataset.empty(rows, m, sum(context.values())), field, context)

    @property
    def group_of(self) -> Int[Array, " components"]:
        """Zeros: every component in the one group"""
        return jnp.zeros(self.data.state_value.shape[-1], jnp.int32)

    @property
    def reading_noise(self) -> Float[Array, " groups"]:
        """1, unread: the truth has no spread to weigh it against"""
        return jnp.ones(1)

    def with_prior_mean(self, prior_mean: Function | None) -> "OracleBelief":
        """Unchanged: the truth needs no prior"""
        return self

    def fit(self, data: Dataset) -> "OracleBelief":
        return dataclasses.replace(self, data=data)

    def _at(self, state_index: Int[Array, "*batch n"], context_value: Float[Array, "*batch n n_context"]) -> Float[Array, "*batch n m"]:
        """The field read at each state at its own context"""
        return self.field({**states_at(self.positions, state_index), **context_parts(self.context, context_value)})

    def _field(self, context: Float[Array, " n_context"]) -> BeliefField:
        """The field at every state at one context, and at the collected readings' own rows"""
        n_states = coordinates(self.positions).shape[0]
        table = self._at(jnp.arange(n_states), jnp.broadcast_to(context, (n_states, context.shape[-1])))
        return BeliefField(self.positions, table, self._at(self.data.state_index, self.data.context_value))

    def predict(
        self, state_index: Int[Array, "*batch n"], context: Float[Array, " n_context"]
    ) -> tuple[Float[Array, "*batch n m"], Float[Array, "*batch n m"]]:
        """The field's own values, and a variance of zero"""
        mean = self._field(context).table[state_index]
        return mean, jnp.zeros_like(mean)

    def posterior_covariance(
        self, a: Int[Array, "*batch_a n_a"], b: Int[Array, "*batch_b n_b"], context: Float[Array, " n_context"]
    ) -> Float[Array, "*batch groups n_a n_b"]:
        """Zero everywhere: the truth has no spread"""
        return jnp.zeros((*jnp.broadcast_shapes(a.shape[:-1], b.shape[:-1]), 1, a.shape[-1], b.shape[-1]))

    def mean(self, context: Float[Array, " n_context"]) -> BeliefField:
        return self._field(context)

    def draw(self, key: PRNGKeyArray, n_fields: int, context: Float[Array, " n_context"]) -> list[BeliefField]:
        """The same field every time, whatever the key"""
        return [self._field(context)] * n_fields
