"""`GroupedBelief`: independent beliefs, one per group of the field's components"""

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from zermelo.interface import Domain, Function
from zermelo.methods.waypoint_bo.belief.base import Belief, BeliefField, Dataset


@register_dataclass
@dataclass(frozen=True)
class GroupedBelief(Belief):
    """Independent beliefs over groups of the field's components:

    f[components[g]] ~ beliefs[g],   the groups together covering every component once
    """

    positions: Domain = dataclasses.field(metadata=dict(static=True))
    data: Dataset

    beliefs: tuple[Belief, ...]
    """One belief per group, over its own components of `data`"""

    components: tuple[tuple[int, ...], ...] = dataclasses.field(metadata=dict(static=True))
    """Which components of the field each group models"""

    @property
    def group_of(self) -> Int[Array, " components"]:
        n_components = self.data.state_value.shape[-1]
        return jnp.zeros(n_components, jnp.int32).at[self._order()].set(jnp.repeat(jnp.arange(len(self.beliefs)), self._sizes()))

    @property
    def reading_noise(self) -> Float[Array, " groups"]:
        return jnp.concatenate([belief.reading_noise for belief in self.beliefs])

    def _sizes(self) -> Int[Array, " groups"]:
        return jnp.asarray([len(group) for group in self.components])

    def _order(self) -> Int[Array, " components"]:
        """The components in the order the groups lay them side by side"""
        return jnp.asarray([j for group in self.components for j in group])

    def _split(self, data: Dataset) -> list[Dataset]:
        """`data` with its components cut into the groups"""
        return [dataclasses.replace(data, state_value=data.state_value[..., jnp.asarray(group)]) for group in self.components]

    def _joined(self, fields: list[BeliefField]) -> BeliefField:
        """One field per group laid side by side, then put back into the field's own component order"""
        inverse = jnp.argsort(self._order())
        return BeliefField(
            self.positions,
            jnp.concatenate([f.table for f in fields], axis=-1)[..., inverse],
            jnp.concatenate([f.at_collected_readings for f in fields], axis=-1)[..., inverse],
        )

    def with_prior_mean(self, prior_mean: Function | None) -> "GroupedBelief":
        """Each group about its own components of `prior_mean`"""
        if prior_mean is None:
            return self
        return dataclasses.replace(
            self,
            beliefs=tuple(
                belief.with_prior_mean(lambda x, group=jnp.asarray(group): prior_mean(x)[..., group])
                for belief, group in zip(self.beliefs, self.components, strict=True)
            ),
        )

    def fit(self, data: Dataset) -> "GroupedBelief":
        fitted = tuple(belief.fit(part) for belief, part in zip(self.beliefs, self._split(data), strict=True))
        return dataclasses.replace(self, data=data, beliefs=fitted)

    def predict(
        self, state_index: Int[Array, "*batch n"], context: Float[Array, " n_context"]
    ) -> tuple[Float[Array, "*batch n m"], Float[Array, "*batch n m"]]:
        """Each group's mean and variance, in the field's own component order"""
        inverse = jnp.argsort(self._order())
        moments = [belief.predict(state_index, context) for belief in self.beliefs]
        return (
            jnp.concatenate([mean for mean, _ in moments], axis=-1)[..., inverse],
            jnp.concatenate([var for _, var in moments], axis=-1)[..., inverse],
        )

    def posterior_covariance(
        self, first_states: Int[Array, "*batch_a n_a"], second_states: Int[Array, "*batch_b n_b"], context: Float[Array, " n_context"]
    ) -> Float[Array, "*batch groups n_a n_b"]:
        """Each group's `k_n(a, b)`, stacked"""
        return jnp.concatenate([belief.posterior_covariance(first_states, second_states, context) for belief in self.beliefs], axis=-3)

    def mean(self, context: Float[Array, " n_context"]) -> BeliefField:
        return self._joined([belief.mean(context) for belief in self.beliefs])

    def draw(self, key: PRNGKeyArray, n_fields: int, context: Float[Array, " n_context"]) -> list[BeliefField]:
        """Draw `s` is each group's draw `s` laid side by side"""
        keys = jax.random.split(key, len(self.beliefs))
        drawn = [belief.draw(k, n_fields, context) for belief, k in zip(self.beliefs, keys, strict=True)]
        return [self._joined(list(per_group)) for per_group in zip(*drawn, strict=True)]
