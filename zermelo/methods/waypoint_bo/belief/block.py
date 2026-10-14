"""`BlockBelief`: independent beliefs over consecutive blocks of the field's channels"""

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
class BlockBelief(Belief):
    """Independent beliefs over consecutive blocks of the field's channels:

    f = (f_1, ..., f_B),   block b the next widths[b] channels, modelled by blocks[b] alone
    """

    positions: Domain = dataclasses.field(metadata=dict(static=True))
    data: Dataset

    blocks: tuple[Belief, ...]
    """One belief per block, each over its own channels of `data`"""

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(block.data.state_value.shape[-1] for block in self.blocks)

    @property
    def reading_noise(self) -> Float[Array, " blocks"]:
        return jnp.concatenate([block.reading_noise for block in self.blocks])

    def _split(self, data: Dataset) -> list[Dataset]:
        """`data` with its channels cut into the blocks"""
        return [dataclasses.replace(data, state_value=data.state_value[..., low:high]) for low, high in self.edges()]

    def with_prior_mean(self, prior_mean: Function | None) -> "BlockBelief":
        """Each block about its own channels of `prior_mean`"""
        if prior_mean is None:
            return self
        return dataclasses.replace(
            self,
            blocks=tuple(
                block.with_prior_mean(lambda x, low=low, high=high: prior_mean(x)[..., low:high])
                for block, (low, high) in zip(self.blocks, self.edges(), strict=True)
            ),
        )

    def fit(self, data: Dataset) -> "BlockBelief":
        return BlockBelief(self.positions, data, tuple(block.fit(part) for block, part in zip(self.blocks, self._split(data), strict=True)))

    def predict(
        self, state_index: Int[Array, "*batch n"], context: Float[Array, " n_context"]
    ) -> tuple[Float[Array, "*batch n m"], Float[Array, "*batch n m"]]:
        """The blocks' means and variances laid side by side"""
        moments = [block.predict(state_index, context) for block in self.blocks]
        return jnp.concatenate([mean for mean, _ in moments], axis=-1), jnp.concatenate([var for _, var in moments], axis=-1)

    def posterior_covariance(
        self, first_states: Int[Array, "*batch_a n_a"], second_states: Int[Array, "*batch_b n_b"], context: Float[Array, " n_context"]
    ) -> Float[Array, "*batch blocks n_a n_b"]:
        """Each block's `k_n(a, b)`, stacked"""
        return jnp.concatenate([block.posterior_covariance(first_states, second_states, context) for block in self.blocks], axis=-3)

    def _joined(self, fields: list[BeliefField]) -> BeliefField:
        """One field per block laid side by side into one field"""
        return BeliefField(
            self.positions,
            jnp.concatenate([f.table for f in fields], axis=-1),
            jnp.concatenate([f.at_collected_readings for f in fields], axis=-1),
        )

    def mean(self, context: Float[Array, " n_context"]) -> BeliefField:
        return self._joined([block.mean(context) for block in self.blocks])

    def draw(self, key: PRNGKeyArray, n_fields: int, context: Float[Array, " n_context"]) -> list[BeliefField]:
        """Draw `s` is each block's draw `s` laid side by side"""
        drawn = [block.draw(k, n_fields, context) for block, k in zip(self.blocks, jax.random.split(key, len(self.blocks)), strict=True)]
        return [self._joined(list(per_block)) for per_block in zip(*drawn, strict=True)]
