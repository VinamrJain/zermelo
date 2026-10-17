"""The latent field an episode draws, and the Gaussian process it is drawn from"""

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from gpjax.kernels.stationary.base import StationaryKernel
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, PRNGKeyArray

from zermelo.interface import Domain, Embeddable, FunctionDomain, Prior


@register_dataclass
@dataclass(frozen=True)
class GPField:
    """One draw from a Gaussian process as random Fourier features:
    f(x) = [cos(P), sin(P)] @ weights, P = source.embed(x) @ frequencies^T
    """

    frequencies: Float[Array, "n_features k"]
    """Drawn from the kernel's spectral density and divided by its lengthscale"""

    weights: Float[Array, "two_n_features ambient_axes"]
    """Standard normal, scaled by `sqrt(variance / n_features)`"""

    source: Embeddable[Any] = dataclasses.field(metadata=dict(static=True))  # `field` alone is reserved for the latent field
    """Which embedding to apply before evaluating"""

    def __call__(self, x: Any) -> Float[Array, "*batch ambient_axes"]:
        proj = self.source.embed(x) @ self.frequencies.T  # (*batch, n_features)
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1) @ self.weights


@dataclass(frozen=True)
class GP(Prior[GPField]):
    """The law a field is drawn by: a Gaussian process, approximated by `n_features` random Fourier features"""

    kernel: StationaryKernel
    """Any gpjax stationary kernel carrying a spectral density: RBF, Matern12, Matern32, Matern52"""

    n_features: int

    def sample(self, domain: Domain[Any], key: PRNGKeyArray) -> GPField:
        """Draw a field on `domain.source`, valued in `domain.target`"""
        if not isinstance(domain, FunctionDomain):
            raise TypeError(f"a Gaussian process draws a function, and a {type(domain).__name__} is not a set of those")
        source, target = domain.source, domain.target
        if not (isinstance(source, Embeddable) and isinstance(target, Embeddable)):
            raise TypeError("a Gaussian process needs coordinates: both the source and the target of the domain must embed")
        if self.kernel.n_dims not in (None, source.dim()):
            raise ValueError(f"the kernel is over {self.kernel.n_dims} dimensions and this domain embeds into {source.dim()}")
        k_frequencies, k_weights = jax.random.split(key)
        frequencies = self.kernel.spectral_density.sample(key=k_frequencies, sample_shape=(self.n_features, source.dim()))
        weights = jax.random.normal(k_weights, (2 * self.n_features, target.dim()))
        return GPField(
            frequencies / self.kernel.lengthscale[...],  # the spectral density samples at unit scale
            weights * jnp.sqrt(self.kernel.variance[...] / self.n_features),
            source,
        )
