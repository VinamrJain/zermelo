"""The finite position domain as a table: its elements, their coordinates, an index into it, and an observation's context cut into parts"""

import functools
from typing import Any

import jax
from jaxtyping import Array, Float, Int

from zermelo.interface import Domain, Embeddable, Enumerable, Function


@functools.lru_cache(maxsize=8)
def coordinates(positions: Domain) -> Float[Array, "n_states k"]:
    """Every element of `positions` embedded as real coordinates, cached and evaluated eagerly"""
    if not (isinstance(positions, Enumerable) and isinstance(positions, Embeddable)):
        raise TypeError(f"a belief regresses over a finite set that hands out coordinates, which a {type(positions).__name__} is not")
    with jax.ensure_compile_time_eval():
        return positions.embed(positions.elements())


@functools.lru_cache(maxsize=8)
def elements(positions: Domain) -> Any:
    """Every element of `positions`, batched along a leading axis, cached and evaluated eagerly"""
    if not isinstance(positions, Enumerable):
        raise TypeError(f"a {type(positions).__name__} cannot be enumerated")
    with jax.ensure_compile_time_eval():
        return positions.elements()


@functools.lru_cache(maxsize=8)
def indexer(positions: Domain) -> Any:
    """`index_of` over a batch of elements, jitted and cached per domain"""
    if not isinstance(positions, Enumerable):
        raise TypeError(f"a field over a table needs an index, which a {type(positions).__name__} has none of")
    return jax.jit(jax.vmap(positions.index_of))


def table_as_field(positions: Domain, table: Float[Array, "n_states m"]) -> Function:
    """A field that reads `table` at whatever element it is called on"""
    index = indexer(positions)
    return lambda x: table[index(x)]


def states_at(positions: Domain, state_index: Int[Array, "*batch n"]) -> Any:
    """The elements of `positions` at those indices, as the domain hands them out"""
    return jax.tree.map(lambda part: part[state_index], elements(positions))


def context_parts(context: dict[str, int], context_value: Float[Array, "*batch n_context"]) -> dict[str, Float[Array, "..."]]:
    """A context row split into its named parts, a part one column wide given as a scalar per row"""
    parts, start = {}, 0
    for name, width in context.items():
        parts[name] = context_value[..., start] if width == 1 else context_value[..., start : start + width]
        start += width
    return parts
