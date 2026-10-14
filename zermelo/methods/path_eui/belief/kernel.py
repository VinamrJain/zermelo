"""The kernel over coordinate rows: a product of stationary factors, each over the columns its named parts occupy

A coordinate row is a state's coordinates followed by its context:  z = [coords(state), context]
"""

import functools
import operator
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from gpjax.kernels.base import AbstractKernel, CombinationKernel
from gpjax.kernels.stationary.base import StationaryKernel
from jaxtyping import Array, Float, PRNGKeyArray

from zermelo.interface import Domain, Embeddable, ProductDomain


@dataclass(frozen=True)
class BeliefKernel:
    """One factor of the belief's kernel, reading the columns its parts occupy:

    k_i(x, x') = family(x[columns], x'[columns]),   columns = the columns `parts` occupy
    """

    family: type[StationaryKernel]
    """Which stationary kernel `lengthscale` parameterizes"""

    parts: tuple[str, ...]
    """Which parts of a coordinate row this reads, by name"""

    lengthscale: tuple[float, ...]
    """One per column those parts occupy, in the order `parts` names them"""

    @classmethod
    def of(cls, family: type[StationaryKernel], parts: list[str], lengthscale: list[float]) -> "BeliefKernel":
        """One factor from the lists a configuration holds it as"""
        return cls(family=family, parts=tuple(parts), lengthscale=tuple(float(x) for x in lengthscale))

    def build(self, layout: dict[str, tuple[int, int]], variance: Float[Array, ""]) -> StationaryKernel:
        """`k_i`, reading the columns `layout` gives this factor's parts"""
        missing = [name for name in self.parts if name not in layout]
        if missing:
            raise KeyError(f"a kernel factor names {missing}, and a coordinate row holds {sorted(layout)}")
        columns = [column for name in self.parts for column in range(*layout[name])]
        if len(columns) != len(self.lengthscale):
            raise ValueError(f"a factor over {self.parts} spans {len(columns)} columns and gives {len(self.lengthscale)} lengthscales")
        return self.family(lengthscale=jnp.asarray(self.lengthscale), variance=variance, n_dims=len(columns), active_dims=columns)


def column_layout(positions: Domain, context: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Which columns of a coordinate row each part occupies: the position domain's parts in order, then the context's

    Example: A balloon over a sphere grid at a specific time given as context, lays out as

        columns    0     1     2       3          4
        part       <-- position -->    altitude   hours (context)
        layout     position: (0, 3)    (3, 4)     (4, 5)
    """
    if not isinstance(positions, ProductDomain):
        raise TypeError(f"a {type(positions).__name__} holds no named parts, so no kernel factor can name one")
    widths = [(name, part.dim()) for name, part in positions.parts.items() if isinstance(part, Embeddable)]
    layout, start = {}, 0
    for name, width in [*widths, *context.items()]:
        layout[name], start = (start, start + width), start + width
    return layout


def product_kernel(factors: tuple[BeliefKernel, ...], layout: dict[str, tuple[int, int]], amplitude: Float[Array, ""]) -> AbstractKernel:
    """The factors multiplied:  k(x, x') = amplitude^2 * prod over i of k_i(x, x')"""
    ones = jnp.ones(())
    built = [f.build(layout, amplitude**2 if i == 0 else ones) for i, f in enumerate(factors)]
    return functools.reduce(operator.mul, built)


def factors_of(kernel: AbstractKernel) -> list[Any]:
    """The factors of a product kernel, a lone kernel being its own one factor"""
    return list(kernel.kernels) if isinstance(kernel, CombinationKernel) else [kernel]


def fourier_features(
    kernel: AbstractKernel, key: PRNGKeyArray, coordinate_rows: Float[Array, "q dim"], n_features: int
) -> Float[Array, "q features"]:
    """`Phi(z) = [cos(Omega z), sin(Omega z)]` for a product of stationary kernels, `features = 2 n_features`:

    Omega z = sum over factors i of  omega_i . z[columns_i] / lengthscale_i,   omega_i ~ the spectral density of factor i
    """
    factors = factors_of(kernel)
    phase = jnp.zeros((coordinate_rows.shape[0], n_features))  # (q, n_features): Omega z
    for key_of_factor, factor in zip(jax.random.split(key, len(factors)), factors, strict=True):
        frequencies = factor.spectral_density.sample(key=key_of_factor, sample_shape=(n_features, factor.n_dims))
        phase = phase + coordinate_rows[:, factor.active_dims] @ (frequencies / factor.lengthscale[...]).T
    return jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
