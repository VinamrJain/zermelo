"""The contract: `D_n` as a `Dataset`, one field of a belief's as a `BeliefField`, and what every model of the field answers"""

import dataclasses
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from zermelo.interface import Domain, Function
from zermelo.methods.waypoint_bo.belief.states import indexer


@register_dataclass
@dataclass(frozen=True)
class Dataset:
    """`D_n`: where readings were taken, what they said, and which rows count"""

    state_index: Int[Array, "*batch n"]
    """The position domain's index of the state each row was read at"""

    state_value: Float[Array, "*batch n m"]
    """What was read there, or what a hypothesis says is there"""

    row_is_written: Bool[Array, "*batch n"]
    """False on an unwritten buffer row, and on a walk's rows after it arrived"""

    context_value: Float[Array, "*batch n n_context"]
    """The coordinates each row carries beyond its state, `n_context` wide and zero wide where there are none"""

    @classmethod
    def empty(cls, rows: int, m: int, n_context: int) -> "Dataset":
        """A buffer of `rows` rows with none of them written"""
        return cls(jnp.zeros(rows, jnp.int32), jnp.zeros((rows, m)), jnp.zeros(rows, bool), jnp.zeros((rows, n_context)))

    def write(
        self, row: int, state_index: Int[Array, ""], state_value: Float[Array, " m"], context_value: Float[Array, " n_context"]
    ) -> "Dataset":
        """This dataset with `row` overwritten and marked written, raising past the end of the buffer"""
        if not 0 <= row < self.state_index.shape[0]:
            raise IndexError(f"row {row} of a {self.state_index.shape[0]}-row buffer: the agent's horizon is shorter than the episode's")
        return Dataset(
            self.state_index.at[row].set(state_index),
            self.state_value.at[row].set(state_value),
            self.row_is_written.at[row].set(True),
            self.context_value.at[row].set(context_value),
        )

    def concat(self, other: "Dataset") -> "Dataset":
        """The two laid end to end along the row axis"""
        return Dataset(
            jnp.concatenate([self.state_index, other.state_index], axis=-1),
            jnp.concatenate([self.state_value, other.state_value], axis=-2),
            jnp.concatenate([self.row_is_written, other.row_is_written], axis=-1),
            jnp.concatenate([self.context_value, other.context_value], axis=-2),
        )

    def broadcast(self, batch: tuple[int, ...]) -> "Dataset":
        """This dataset repeated over the leading axes `batch`"""
        return Dataset(
            jnp.broadcast_to(self.state_index, (*batch, *self.state_index.shape)),
            jnp.broadcast_to(self.state_value, (*batch, *self.state_value.shape)),
            jnp.broadcast_to(self.row_is_written, (*batch, *self.row_is_written.shape)),
            jnp.broadcast_to(self.context_value, (*batch, *self.context_value.shape)),
        )


@register_dataclass
@dataclass(frozen=True)
class BeliefField:
    """One field of a belief's, the mean or a draw: its values at every state at one context, and at the collected readings' own rows"""

    positions: Domain = dataclasses.field(metadata=dict(static=True))

    table: Float[Array, "n_states m"]
    """`f(state, context)` at every state, at the one context the field was read at"""

    at_collected_readings: Float[Array, "rows m"]
    """`f(state_i, context_i)` at each row of the belief's dataset, written or not"""

    def __call__(self, x: Any) -> Float[Array, "*batch m"]:
        """`table` at whatever state this is called on"""
        return self.table[indexer(self.positions)(x)]


class Belief(ABC):
    """A model of the field over `positions`, conditioned on `data`, as independent blocks of its channels:

    f = (f_1, ..., f_B),   block b the next widths[b] channels
    """

    positions: Domain
    """The states the field is defined on: finite, and handing out coordinates"""

    data: Dataset

    @property
    @abstractmethod
    def widths(self) -> tuple[int, ...]:
        """Channels in each block, summing to the field's `m`"""

    @property
    @abstractmethod
    def reading_noise(self) -> Float[Array, " blocks"]:
        """Standard deviation of a reading's error per channel, one per block"""

    def edges(self) -> list[tuple[int, int]]:
        """(first channel, one past the last) of each block"""
        ends = list(itertools.accumulate(self.widths))
        return list(zip([0, *ends[:-1]], ends, strict=True))

    def channels(self, blocks: tuple[int, ...]) -> Int[Array, " channels"]:
        """The field's channel indices the `blocks` cover, in order"""
        edges = self.edges()
        return jnp.concatenate([jnp.arange(*edges[b]) for b in blocks])

    def over_blocks(self, per_block: Float[Array, "*batch blocks"], blocks: tuple[int, ...]) -> Float[Array, " *batch"]:
        """sum over b in `blocks` of widths[b] * per_block[..., b]"""
        chosen = jnp.asarray(blocks)
        return jnp.sum(jnp.asarray(self.widths)[chosen] * jnp.take(per_block, chosen, axis=-1), axis=-1)

    @abstractmethod
    def with_prior_mean(self, prior_mean: Function | None) -> Self:
        """This belief about `prior_mean`, a function of a state together with its named context parts; `None` is zero"""

    @abstractmethod
    def fit(self, data: Dataset) -> Self:
        """This belief conditioned on `data`, one dataset and never a batch, its own parameters re-estimated from it where configured"""

    @abstractmethod
    def predict(
        self, state_index: Int[Array, "*batch n"], context: Float[Array, " n_context"]
    ) -> tuple[Float[Array, "*batch n m"], Float[Array, "*batch n m"]]:
        """Posterior mean and per-component variance at those states, at one context"""

    @abstractmethod
    def posterior_covariance(
        self, a: Int[Array, "*batch_a n_a"], b: Int[Array, "*batch_b n_b"], context: Float[Array, " n_context"]
    ) -> Float[Array, "*batch blocks n_a n_b"]:
        """`k_n(a, b)` of each block for every state `a` of the first set and `b` of the second, at one context; the batch axes broadcast"""

    @abstractmethod
    def mean(self, context: Float[Array, " n_context"]) -> BeliefField:
        """`mu_n`, the posterior mean field at one context"""

    @abstractmethod
    def draw(self, key: PRNGKeyArray, n_fields: int, context: Float[Array, " n_context"]) -> list[BeliefField]:
        """`n_fields` fields drawn from the posterior at one context"""
