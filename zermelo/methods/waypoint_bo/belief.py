"""The readings a method has collected, and a model of the field fitted to them"""

import dataclasses
import functools
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

import gpjax as gpx
import jax
import jax.numpy as jnp
from gpjax.kernels import RFF
from gpjax.kernels.stationary.base import StationaryKernel
from jax.scipy.linalg import solve_triangular
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from zermelo.interface import Domain, Embeddable, Enumerable, Function

# float32 is the problem side's choice, not this file's; gpjax warns per dataset
warnings.filterwarnings("ignore", message=".*not of type float64.*", category=UserWarning)


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
def _indexer(positions: Domain) -> Any:
    """`index_of` over a batch of elements, jitted and cached per domain"""
    if not isinstance(positions, Enumerable):
        raise TypeError(f"a field over a table needs an index, which a {type(positions).__name__} has none of")
    return jax.jit(jax.vmap(positions.index_of))


def lookup(positions: Domain, table: Float[Array, "n_states m"]) -> Function:
    """A field that reads `table` at whatever element it is called on"""
    index = _indexer(positions)
    return lambda x: table[index(x)]


@register_dataclass
@dataclass(frozen=True)
class Dataset:
    """`D_n`: where readings were taken, what they said, and which rows count"""

    z: Int[Array, "*batch n"]
    """The position domain's index of the cell each row sits at"""

    r: Float[Array, "*batch n m"]
    """What was read there, or what a hypothesis says is there"""

    live: Bool[Array, "*batch n"]
    """False on an unwritten buffer row, and on a walk's rows after it arrived"""

    @classmethod
    def empty(cls, rows: int, m: int) -> "Dataset":
        """A buffer of `rows` rows with none of them written"""
        return cls(jnp.zeros(rows, jnp.int32), jnp.zeros((rows, m)), jnp.zeros(rows, bool))

    def write(self, row: int, z: Int[Array, ""], r: Float[Array, " m"]) -> "Dataset":
        """This dataset with `row` overwritten and marked live, raising past the end of the buffer"""
        if not 0 <= row < self.z.shape[0]:
            raise IndexError(f"row {row} of a {self.z.shape[0]}-row buffer: the agent's horizon is shorter than the episode's")
        return Dataset(self.z.at[row].set(z), self.r.at[row].set(r), self.live.at[row].set(True))

    def concat(self, other: "Dataset") -> "Dataset":
        """The two laid end to end along the row axis"""
        return Dataset(
            jnp.concatenate([self.z, other.z], axis=-1),
            jnp.concatenate([self.r, other.r], axis=-2),
            jnp.concatenate([self.live, other.live], axis=-1),
        )

    def broadcast(self, batch: tuple[int, ...]) -> "Dataset":
        """This dataset repeated over the leading axes `batch`"""
        return Dataset(
            jnp.broadcast_to(self.z, (*batch, *self.z.shape)),
            jnp.broadcast_to(self.r, (*batch, *self.r.shape)),
            jnp.broadcast_to(self.live, (*batch, *self.live.shape)),
        )


class Belief(ABC):
    """A model of the field over `positions`, conditioned on `data`"""

    positions: Domain
    """The cells the field is defined on: finite, and handing out coordinates"""

    data: Dataset

    @abstractmethod
    def fit(self, data: Dataset) -> Self:
        """This belief conditioned on `data`, its own parameters re-estimated from it where configured"""

    @abstractmethod
    def condition(self, data: Dataset) -> Self:
        """This belief conditioned on `data`, its parameters unchanged, over one dataset and never a batch"""

    @abstractmethod
    def predict(self, z: Int[Array, "*batch n"]) -> tuple[Float[Array, "*batch n m"], Float[Array, "*batch n m"]]:
        """Posterior mean and per-component variance at those cells"""

    @abstractmethod
    def mean(self) -> Function:
        """`mu_n`, the posterior mean field"""

    @abstractmethod
    def draw(self, key: PRNGKeyArray, n_fields: int) -> list[Function]:
        """`n_fields` fields drawn from the posterior"""


@register_dataclass
@dataclass(frozen=True)
class OracleBelief(Belief):
    """The true field in place of a model: the mean is exactly it and the variance is zero"""

    positions: Domain = dataclasses.field(metadata=dict(static=True))
    data: Dataset

    field: Function = dataclasses.field(metadata=dict(static=True))
    """The field the world drew, frozen to a table"""

    @classmethod
    def empty(cls, positions: Domain, rows: int, m: int, field: Function) -> "OracleBelief":
        """Seen nothing, over a buffer of `rows` rows, holding the field the world drew"""
        return cls(positions, Dataset.empty(rows, m), lookup(positions, field(elements(positions))))

    def fit(self, data: Dataset) -> "OracleBelief":
        return dataclasses.replace(self, data=data)

    def condition(self, data: Dataset) -> "OracleBelief":
        return dataclasses.replace(self, data=data)

    def predict(self, z: Int[Array, "*batch n"]) -> tuple[Float[Array, "*batch n m"], Float[Array, "*batch n m"]]:
        """The field's own values, and a variance of zero"""
        mean = self.field(elements(self.positions))[z]  # (n_states, m): read at every cell, then gathered
        return mean, jnp.zeros_like(mean)

    def mean(self) -> Function:
        return self.field

    def draw(self, key: PRNGKeyArray, n_fields: int) -> list[Function]:
        """The same field every time, whatever the key"""
        return [self.field] * n_fields


@jax.jit
def _conjugate(
    gram: Float[Array, "rows rows"],
    cross: Float[Array, "rows q"],
    y: Float[Array, "rows m"],
    keep: Bool[Array, " rows"],
    noise_var: Float[Array, " rows"],
    variance: Float[Array, ""],
) -> tuple[Float[Array, "q m"], Float[Array, " q"]]:
    """Posterior mean and variance at `cross`'s columns, over the rows `keep` marks:

        K = gram + diag(noise_var) where kept, the identity elsewhere;   L L^T = K
        V = L^-1 cross,   mean = V^T L^-1 y,   var = variance - sum over rows of V^2

    Time O(rows^3 + q rows^2), memory O(q rows).
    """
    pair = keep[:, None] & keep[None, :]  # (rows, rows): both ends kept
    diagonal = jnp.where(keep, noise_var, 1.0) + 1e-6  # (rows,): dropped rows get variance 1; +1e-6 jitter for float32
    chol = jnp.linalg.cholesky(jnp.where(pair, gram, 0.0) + jnp.diag(diagonal))  # (rows, rows)
    solved = jnp.where(keep[:, None], solve_triangular(chol, cross, lower=True), 0.0)  # (rows, q), a dropped row zeroed
    mean = solved.T @ solve_triangular(chol, jnp.where(keep[:, None], y, 0.0), lower=True)  # (q, m)
    return mean, jnp.maximum(variance - jnp.sum(solved**2, axis=0), 0.0)  # (q,)


@register_dataclass
@dataclass(frozen=True)
class GPBelief(Belief):
    """`m` zero-mean Gaussian processes over the field's components, independent and sharing one kernel"""

    positions: Domain = dataclasses.field(metadata=dict(static=True))
    data: Dataset

    lengthscale: Float[Array, " dim"]
    """One lengthscale per coordinate the position domain embeds to"""

    amplitude: Float[Array, ""]
    noise: Float[Array, ""]
    """The observation noise standard deviation"""

    kernel_family: type[StationaryKernel] = dataclasses.field(metadata=dict(static=True))
    """Which stationary kernel the three numbers above parameterize"""

    n_features: int = dataclasses.field(metadata=dict(static=True))
    """How many random Fourier features a pathwise draw uses"""

    refit: bool = dataclasses.field(metadata=dict(static=True))
    """Whether `fit` re-estimates the three parameters by marginal likelihood"""

    refit_steps: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def empty(cls, positions: Domain, rows: int, m: int, **config: Any) -> "GPBelief":
        """Seen nothing, over a buffer of `rows` rows"""
        return cls(positions, Dataset.empty(rows, m), **config)

    @property
    def coords(self) -> Float[Array, "n_states k"]:
        """Every cell's coordinates, in the position domain's own index order"""
        return coordinates(self.positions)

    def _conditioning(self) -> tuple[Float[Array, "rows dim"], Float[Array, "rows m"], Bool[Array, " rows"], Float[Array, " rows"]]:
        """Per buffer row: coordinates, its cell's mean reading, whether it is the first live row there, and `noise^2 / c` at `c` reads"""
        rows, n_states = self.data.z.shape[0], self.coords.shape[0]
        z = jnp.where(self.data.live, self.data.z, 0)  # (rows,): dead rows get cell 0, and are masked out below
        c = jnp.zeros(n_states).at[z].add(self.data.live.astype(self.data.r.dtype))  # (n_states,): readings per cell
        total = jnp.zeros((n_states, self.data.r.shape[-1])).at[z].add(jnp.where(self.data.live[:, None], self.data.r, 0.0))
        # (n_states,): earliest live row at each cell, a dead row scattering `rows` so it never wins the min
        first = jnp.full(n_states, rows).at[z].min(jnp.where(self.data.live, jnp.arange(rows), rows))
        at = jnp.maximum(c[z], 1.0)  # (rows,): readings at this row's cell, floored to 1
        # rows kept: live and the first at its cell, so a cell appears once
        return self.coords[z], total[z] / at[:, None], self.data.live & (jnp.arange(rows) == first[z]), self.noise**2 / at

    def _posterior(self, width: int, n: int) -> Any:
        """The conjugate posterior over one output component of a `width`-wide field seen at `n` points"""
        prior = gpx.gps.Prior(
            kernel=self.kernel_family(lengthscale=self.lengthscale, variance=self.amplitude**2, n_dims=width),
            mean_function=gpx.mean_functions.Zero(),
        )
        return prior * gpx.likelihoods.Gaussian(num_datapoints=max(n, 1), obs_stddev=self.noise)

    def fit(self, data: Dataset) -> "GPBelief":
        held = dataclasses.replace(self, data=data)
        if not self.refit:
            return held
        x, y, keep, _ = held._conditioning()
        x, y = x[keep], y[keep]  # (k, dim), (k, m): the distinct cells read
        if x.shape[0] < 2:  # a marginal likelihood on one point prefers an infinite lengthscale
            return held

        def loss(model: Any, d: Any) -> Float[Array, ""]:
            # one kernel over m independent components, so the joint log marginal likelihood is their sum
            terms = [gpx.objectives.conjugate_mll(model, gpx.Dataset(d.X, d.y[:, j : j + 1])) for j in range(d.y.shape[-1])]
            return -jnp.sum(jnp.stack([jnp.asarray(t) for t in terms]))

        tuned, _ = gpx.fit_scipy(
            model=held._posterior(x.shape[-1], x.shape[0]),
            objective=loss,
            train_data=gpx.Dataset(x, y),
            max_iters=self.refit_steps,
            verbose=False,
        )
        return dataclasses.replace(
            held,
            lengthscale=jnp.asarray(tuned.prior.kernel.lengthscale.value).reshape(-1),
            amplitude=jnp.sqrt(jnp.asarray(tuned.prior.kernel.variance.value).reshape(())),
            noise=jnp.asarray(tuned.likelihood.obs_stddev.value).reshape(()),
        )

    def condition(self, data: Dataset) -> "GPBelief":
        return dataclasses.replace(self, data=data)

    def _moments(self, at: Float[Array, "q dim"]) -> tuple[Float[Array, "q m"], Float[Array, "q m"]]:
        """Posterior mean and variance per component of the field at `at`"""
        x, y, keep, noise_var = self._conditioning()
        kernel = self.kernel_family(lengthscale=self.lengthscale, variance=self.amplitude**2, n_dims=x.shape[-1])
        mean, var = _conjugate(kernel.gram(x).to_dense(), kernel.cross_covariance(x, at), y, keep, noise_var, self.amplitude**2)
        return mean, jnp.broadcast_to(var[:, None], mean.shape)  # (q, m) each, one kernel serving every component

    def predict(self, z: Int[Array, "*batch n"]) -> tuple[Float[Array, "*batch n m"], Float[Array, "*batch n m"]]:
        mean, var = self._moments(self.coords)  # (n_states, m) each: solved at every cell, then gathered
        return mean[z], var[z]

    def mean(self) -> Function:
        return lookup(self.positions, self._moments(self.coords)[0])

    def draw(self, key: PRNGKeyArray, n_fields: int) -> list[Function]:
        """`n_fields` fields drawn from the posterior, a shorter draw being the opening of a longer one

        Time O(n_read^3 + n_fields (n_states n_features + n_read n_states)).
        """
        x, y, keep, _ = self._conditioning()
        x, y = x[keep], y[keep]  # (n_read, dim), (n_read, m): the distinct cells read
        width = self.data.r.shape[-1]
        drawn = [self._paths(jax.random.split(key, width)[j], x, y[:, j], n_fields) for j in range(width)]
        table = jnp.stack(drawn, axis=-1)  # (n_fields, n_states, m)
        return [lookup(self.positions, table[s]) for s in range(n_fields)]

    def _paths(
        self, key: PRNGKeyArray, x: Float[Array, "n_read dim"], y: Float[Array, " n_read"], n_fields: int
    ) -> Float[Array, "n_fields n_states"]:
        """`n_fields` draws of one field component at every cell:

        f_s(z) = Phi(z) theta_s + k(z, x) v_s,   v_s = (K + noise^2 I)^-1 (y + eps_s - Phi(x) theta_s)
        """
        kernel = self.kernel_family(lengthscale=self.lengthscale, variance=self.amplitude**2, n_dims=x.shape[-1])
        basis = RFF(base_kernel=kernel, num_basis_fns=self.n_features, key=key)
        scale = jnp.sqrt(self.amplitude**2 / self.n_features)
        theta = jax.random.normal(key, (n_fields, 2 * self.n_features))  # (n_fields, 2 n_features): row s is path s
        features = basis.compute_features(self.coords) * scale  # (n_states, 2 n_features)
        prior_part = theta @ features.T  # (n_fields, n_states)
        if x.shape[0] == 0:  # nothing read: the draw is from the prior
            return prior_part
        # drawn (n_fields, n_read) then turned: filling the other way round would change path s with n_fields
        eps = self.noise * jax.random.normal(key, (n_fields, x.shape[0])).T  # (n_read, n_fields)
        gram = kernel.gram(x).to_dense() + (self.noise**2 + 1e-6) * jnp.eye(x.shape[0])  # (n_read, n_read)
        residual = y[:, None] + eps - (basis.compute_features(x) * scale) @ theta.T  # (n_read, n_fields)
        canonical = jnp.linalg.solve(gram, residual)  # (n_read, n_fields): one factorization, a column per field
        return prior_part + (kernel.cross_covariance(self.coords, x) @ canonical).T
