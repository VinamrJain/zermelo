"""`GPBelief`: exact Gaussian process regression over coordinate rows, one kernel shared by every component"""

import dataclasses
import warnings
from dataclasses import dataclass
from typing import Any

import gpjax as gpx
import jax
import jax.numpy as jnp
from gpjax.kernels.base import AbstractKernel
from jax.scipy.linalg import solve_triangular
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from zermelo.interface import Domain, Function
from zermelo.methods.waypoint_bo.belief.base import Belief, BeliefField, Dataset
from zermelo.methods.waypoint_bo.belief.kernel import BeliefKernel, column_layout, factors_of, fourier_features, product_kernel
from zermelo.methods.waypoint_bo.belief.states import context_parts, coordinates, states_at

# float32 is the problem side's choice, not this file's; gpjax warns per dataset
warnings.filterwarnings("ignore", message=".*not of type float64.*", category=UserWarning)

REFIT_FLOOR = 1e-3
"""Smallest lengthscale, amplitude and noise a refit may return"""


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
    chol = _cholesky(gram, keep, noise_var)
    solved = _whiten(chol, keep, cross)  # (rows, q)
    mean = solved.T @ solve_triangular(chol, jnp.where(keep[:, None], y, 0.0), lower=True)  # (q, m)
    return mean, jnp.maximum(variance - jnp.sum(solved**2, axis=0), 0.0)  # (q,)


def _cholesky(gram: Float[Array, "rows rows"], keep: Bool[Array, " rows"], noise_var: Float[Array, " rows"]) -> Float[Array, "rows rows"]:
    """`L` with `L L^T = K`, `K = gram + diag(noise_var)` over the rows `keep` marks and the identity elsewhere"""
    pair = keep[:, None] & keep[None, :]  # (rows, rows): both ends kept
    diagonal = jnp.where(keep, noise_var, 1.0) + 1e-6  # (rows,): dropped rows get variance 1; +1e-6 jitter for float32
    return jnp.linalg.cholesky(jnp.where(pair, gram, 0.0) + jnp.diag(diagonal))


def _whiten(chol: Float[Array, "rows rows"], keep: Bool[Array, " rows"], cross: Float[Array, "rows q"]) -> Float[Array, "rows q"]:
    """`V = L^-1 cross`, a dropped row zeroed"""
    return jnp.where(keep[:, None], solve_triangular(chol, cross, lower=True), 0.0)


@register_dataclass
@dataclass(frozen=True)
class GPBelief(Belief):
    """One Gaussian process per component of the field, independent, sharing one kernel, about a given prior mean:

    f(state, context) = prior_mean(state, context) + e(state, context),   e ~ GP(0, k)
    """

    positions: Domain = dataclasses.field(metadata=dict(static=True))
    data: Dataset

    kernel_factors: tuple[BeliefKernel, ...] = dataclasses.field(metadata=dict(static=True))
    """The factors the kernel is a product of, each over the parts it names"""

    context: dict[str, int] = dataclasses.field(metadata=dict(static=True))
    """The coordinates a reading carries beyond its state, each named and how wide it is"""

    amplitude: Float[Array, ""]
    noise: Float[Array, ""]
    """The observation noise standard deviation"""

    n_features: int = dataclasses.field(metadata=dict(static=True))
    """How many random Fourier features a pathwise draw uses"""

    refit: bool = dataclasses.field(metadata=dict(static=True))
    """Whether `fit` re-estimates the three parameters by marginal likelihood"""

    refit_steps: int = dataclasses.field(metadata=dict(static=True))

    prior_mean: Function | None = dataclasses.field(metadata=dict(static=True))
    """The field's mean before any reading, a function of a state with its named context parts; `None` is zero everywhere"""

    @classmethod
    def empty(cls, positions: Domain, rows: int, m: int, n_context: int, **config: Any) -> "GPBelief":
        """Seen nothing, over a buffer of `rows` rows"""
        return cls(positions, Dataset.empty(rows, m, n_context), **config)

    @property
    def group_of(self) -> Int[Array, " components"]:
        """Zeros: every component in the one group"""
        return jnp.zeros(self.data.state_value.shape[-1], jnp.int32)

    @property
    def reading_noise(self) -> Float[Array, " groups"]:
        return self.noise[None]

    def with_prior_mean(self, prior_mean: Function | None) -> "GPBelief":
        return dataclasses.replace(self, prior_mean=prior_mean)

    @property
    def coords(self) -> Float[Array, "n_states k"]:
        """Every state's coordinates, in the position domain's own index order"""
        return coordinates(self.positions)

    @property
    def kernel(self) -> AbstractKernel:
        """The product of the factors, over the columns the position domain and the context lay out"""
        return product_kernel(self.kernel_factors, column_layout(self.positions, self.context), self.amplitude)

    def query(self, context: Float[Array, " n_context"]) -> Float[Array, "n_states dim"]:
        """Every state's coordinates at one context, that context repeated down the query"""
        return self._coordinate_rows(jnp.arange(self.coords.shape[0]), context)

    def _coordinate_rows(self, state_index: Int[Array, "*batch n"], context: Float[Array, " n_context"]) -> Float[Array, "*batch n dim"]:
        """The kernel's row for each state at one context: its coordinates, then the context"""
        return jnp.concatenate([self.coords[state_index], jnp.broadcast_to(context, (*state_index.shape, context.shape[-1]))], axis=-1)

    def _prior_mean_at(
        self, state_index: Int[Array, "*batch n"], context_value: Float[Array, "*batch n n_context"]
    ) -> Float[Array, "*batch n m"]:
        """`prior_mean` at each state at its own context, zeros when none"""
        m = self.data.state_value.shape[-1]
        if self.prior_mean is None:
            return jnp.zeros((*state_index.shape, m))
        read = self.prior_mean({**states_at(self.positions, state_index), **context_parts(self.context, context_value)})
        return jnp.broadcast_to(read, (*state_index.shape, m))

    def _conditioning(self) -> tuple[Float[Array, "rows dim"], Float[Array, "rows m"], Bool[Array, " rows"], Float[Array, " rows"]]:
        """Per buffer row: coordinates, the residual read there, whether the row is conditioned on, and the noise variance it carries

        Rows at one state are repeat measurements only where no context separates them, and are then averaged
        into the earliest of them at `noise^2 / c` for `c` reads.
        """
        data = self.data
        rows, n_states = data.state_index.shape[0], self.coords.shape[0]
        z = jnp.where(data.row_is_written, data.state_index, 0)  # (rows,): unwritten rows get state 0, and are masked out below
        x = jnp.concatenate([self.coords[z], data.context_value], axis=-1)  # (rows, dim)
        offset = self._prior_mean_at(z, data.context_value)  # (rows, m)
        if self.context:  # no deduplication: a context tells two readings of one state apart
            return x, data.state_value - offset, data.row_is_written, jnp.full(rows, self.noise**2)
        # deduplicate: without a context a repeat visit is a repeat measurement, averaged for a better conditioned gram
        c = jnp.zeros(n_states).at[z].add(data.row_is_written.astype(data.state_value.dtype))  # (n_states,): readings per state
        total = jnp.zeros((n_states, data.state_value.shape[-1])).at[z].add(jnp.where(data.row_is_written[:, None], data.state_value, 0.0))
        # (n_states,): earliest written row at each state, an unwritten row scattering `rows` so it never wins the min
        first = jnp.full(n_states, rows).at[z].min(jnp.where(data.row_is_written, jnp.arange(rows), rows))
        n_reads = jnp.maximum(c[z], 1.0)  # (rows,): readings at this row's state, floored to 1
        # rows kept: written and the first at its state, so a state appears once
        return x, total[z] / n_reads[:, None] - offset, data.row_is_written & (jnp.arange(rows) == first[z]), self.noise**2 / n_reads

    def _posterior(self, n: int) -> Any:
        """The conjugate posterior over one output component of the field seen at `n` points"""
        prior = gpx.gps.Prior(kernel=self.kernel, mean_function=gpx.mean_functions.Zero())
        return prior * gpx.likelihoods.Gaussian(num_datapoints=max(n, 1), obs_stddev=self.noise)

    def fit(self, data: Dataset) -> "GPBelief":
        conditioned = dataclasses.replace(self, data=data)
        if not self.refit:
            return conditioned
        x, y, keep, _ = conditioned._conditioning()
        x, y = x[keep], y[keep]  # (k, dim), (k, m): the distinct rows read
        if x.shape[0] < 2:  # a marginal likelihood on one point prefers an infinite lengthscale
            return conditioned

        def loss(model: Any, d: Any) -> Float[Array, ""]:
            # one kernel over m independent components, so the joint log marginal likelihood is their sum
            per_component = [gpx.objectives.conjugate_mll(model, gpx.Dataset(d.X, d.y[:, j : j + 1])) for j in range(d.y.shape[-1])]
            return -jnp.sum(jnp.stack([jnp.asarray(t) for t in per_component]))

        tuned, _ = gpx.fit_scipy(
            model=conditioned._posterior(x.shape[0]),
            objective=loss,
            train_data=gpx.Dataset(x, y),
            max_iters=self.refit_steps,
            verbose=False,
        )
        fitted = factors_of(tuned.prior.kernel)
        return dataclasses.replace(
            conditioned,
            kernel_factors=tuple(
                dataclasses.replace(
                    factor, lengthscale=tuple(max(v, REFIT_FLOOR) for v in jnp.asarray(k.lengthscale.value).reshape(-1).tolist())
                )
                for factor, k in zip(self.kernel_factors, fitted, strict=True)
            ),
            amplitude=jnp.maximum(jnp.sqrt(jnp.asarray(fitted[0].variance.value).reshape(())), REFIT_FLOOR),
            noise=jnp.maximum(jnp.asarray(tuned.likelihood.obs_stddev.value).reshape(()), REFIT_FLOOR),
        )

    def _moments(self, query: Float[Array, "q dim"]) -> tuple[Float[Array, "q m"], Float[Array, "q m"]]:
        """Posterior mean and variance per component of `e` at `query`"""
        x, y, keep, noise_var = self._conditioning()
        kernel = self.kernel
        mean, var = _conjugate(kernel.gram(x).to_dense(), kernel.cross_covariance(x, query), y, keep, noise_var, self.amplitude**2)
        return mean, jnp.broadcast_to(var[:, None], mean.shape)  # (q, m) each, one kernel serving every component

    def _query_and_rows(self, context: Float[Array, " n_context"]) -> Float[Array, "query dim"]:
        """Every state at `context`, then the collected readings' own rows"""
        x, _, _, _ = self._conditioning()
        return jnp.concatenate([self.query(context), x], axis=0)

    def _field(self, context: Float[Array, " n_context"], e: Float[Array, "query m"]) -> BeliefField:
        """`prior_mean + e` split into the table over every state and the values at the collected readings' rows"""
        n_states = self.coords.shape[0]
        z = jnp.where(self.data.row_is_written, self.data.state_index, 0)
        table = e[:n_states] + self._prior_mean_at(jnp.arange(n_states), jnp.broadcast_to(context, (n_states, context.shape[-1])))
        return BeliefField(self.positions, table, e[n_states:] + self._prior_mean_at(z, self.data.context_value))

    def predict(
        self, state_index: Int[Array, "*batch n"], context: Float[Array, " n_context"]
    ) -> tuple[Float[Array, "*batch n m"], Float[Array, "*batch n m"]]:
        mean, var = self._moments(self.query(context))  # (n_states, m) each: solved at every state, then gathered
        at = jnp.broadcast_to(context, (*state_index.shape, context.shape[-1]))
        return mean[state_index] + self._prior_mean_at(state_index, at), var[state_index]

    def posterior_covariance(
        self, first_states: Int[Array, "*batch_a n_a"], second_states: Int[Array, "*batch_b n_b"], context: Float[Array, " n_context"]
    ) -> Float[Array, "*batch groups n_a n_b"]:
        """k_n(a, b) = k(a, b) - V_a^T V_b,   V = L^-1 k(X_n, .),   each side whitened once over its own batch, as the one group

        Memory O(rows (n_a + n_b) + n_a n_b) per batch element.
        """
        x, _, keep, noise_var = self._conditioning()
        kernel = self.kernel
        chol = _cholesky(kernel.gram(x).to_dense(), keep, noise_var)
        first, second = self._coordinate_rows(first_states, context), self._coordinate_rows(second_states, context)
        prior = jnp.vectorize(kernel.cross_covariance, signature="(a,d),(b,d)->(a,b)")(first, second)  # (*batch, n_a, n_b): k(a, b)
        whiten = jnp.vectorize(lambda rows: _whiten(chol, keep, kernel.cross_covariance(x, rows)), signature="(n,d)->(r,n)")
        return (prior - jnp.einsum("...ra,...rb->...ab", whiten(first), whiten(second)))[..., None, :, :]

    def mean(self, context: Float[Array, " n_context"]) -> BeliefField:
        return self._field(context, self._moments(self._query_and_rows(context))[0])

    def draw(self, key: PRNGKeyArray, n_fields: int, context: Float[Array, " n_context"]) -> list[BeliefField]:
        """`n_fields` fields drawn from the posterior at one context, a shorter draw being a subset of a longer one

        Time O(n_read^3 + n_fields ((n_states + rows) n_features + n_read (n_states + rows))).
        """
        x, y, keep, _ = self._conditioning()
        x, y = x[keep], y[keep]  # (n_read, dim), (n_read, m): the rows conditioned on
        width = self.data.state_value.shape[-1]
        query = self._query_and_rows(context)
        drawn = [self._paths(jax.random.split(key, width)[j], x, y[:, j], n_fields, query) for j in range(width)]
        e = jnp.stack(drawn, axis=-1)  # (n_fields, n_states + rows, m): a draw of e
        return [self._field(context, e[s]) for s in range(n_fields)]

    def _paths(
        self, key: PRNGKeyArray, x: Float[Array, "n_read dim"], y: Float[Array, " n_read"], n_fields: int, query: Float[Array, "q dim"]
    ) -> Float[Array, "n_fields q"]:
        """`n_fields` draws of one component of `e` at every query row:

        e_s(z) = Phi(z) theta_s + k(z, x) v_s,   v_s = (K + noise^2 I)^-1 (y + eps_s - Phi(x) theta_s)
        """
        kernel = self.kernel
        scale = jnp.sqrt(self.amplitude**2 / self.n_features)
        theta = jax.random.normal(key, (n_fields, 2 * self.n_features))  # (n_fields, 2 n_features): row s is path s
        features = fourier_features(kernel, key, query, self.n_features) * scale  # (q, 2 n_features)
        prior_part = theta @ features.T  # (n_fields, q)
        if x.shape[0] == 0:  # nothing read: the draw is from the prior
            return prior_part
        # drawn (n_fields, n_read) then turned: filling the other way round would change path s with n_fields
        eps = self.noise * jax.random.normal(key, (n_fields, x.shape[0])).T  # (n_read, n_fields)
        gram = kernel.gram(x).to_dense() + (self.noise**2 + 1e-6) * jnp.eye(x.shape[0])  # (n_read, n_read)
        residual = (
            y[:, None] + eps - (fourier_features(kernel, key, x, self.n_features) * scale) @ theta.T
        )  # (n_read, n_fields): the same key, so the same Omega
        canonical = jnp.linalg.solve(gram, residual)  # (n_read, n_fields): one factorization, a column per field
        return prior_part + (kernel.cross_covariance(query, x) @ canonical).T
