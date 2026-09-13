"""Validating our pathwise sampler against gpjax's own on a dummy problem."""

import argparse
from dataclasses import dataclass

import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np

N_CELLS, SPACING = 41, 0.25
"""A line of `N_CELLS` cells `SPACING` apart."""

LENGTHSCALE, AMPLITUDE, NOISE = 1.0, 1.0, 0.1
"""k(x, x') = AMPLITUDE^2 exp(-|x - x'|^2 / (2 LENGTHSCALE^2)), and a row's noise standard deviation is NOISE."""

N_READ, SEED, N_FEATURES = 8, 0, 256
"""`N_READ` cells read once each under `SEED`, and the features a pathwise draw uses."""

COUNTS = (1, 4, 8, 16)
"""The draw sizes compared, and the sizes nesting is checked across."""

MAX_DEVIATION = 2e-4
"""How far apart two versions of one path may sit, at float32 over `N_FEATURES` features."""

N_DISTRIBUTION = 4000
"""Paths drawn when the moments are compared rather than the bits."""

MAX_SIGMAS = 5.0
"""How many standard errors a sample mean may sit from the exact one."""

MAX_MOMENT = 0.15
"""How far a sample second moment may sit from another, as a share of the largest exact variance (the bias `N_FEATURES` leaves)."""


@dataclass(frozen=True)
class Agreement:
    """How well two versions of one quantity match."""

    deviation: float
    passed: bool

    @classmethod
    def between(cls, mine: np.ndarray, theirs: np.ndarray) -> "Agreement":
        """The largest absolute gap between two arrays of the same shape."""
        if mine.shape != theirs.shape:
            return cls(float("inf"), False)
        deviation = float(np.max(np.abs(mine - theirs))) if mine.size else 0.0
        return cls(deviation, deviation <= MAX_DEVIATION)

    @classmethod
    def within(cls, gap: np.ndarray, tolerance: np.ndarray | float) -> "Agreement":
        """The largest gap, as a multiple of the tolerance allowed at that entry."""
        ratio = np.abs(gap) / np.where(np.asarray(tolerance) > 0.0, tolerance, np.inf)
        return cls(float(np.max(ratio)), bool(np.all(ratio <= 1.0)))


def dummy_problem() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The problem both sides solve: the grid, the cells read, and what was read there."""
    coords = (SPACING * jnp.arange(N_CELLS, dtype=jnp.float32))[:, None]  # (n_cells, 1)
    rng = np.random.default_rng(SEED)
    read = np.sort(rng.choice(N_CELLS, N_READ, replace=False))
    return coords, coords[jnp.asarray(read)], jnp.asarray(rng.standard_normal(N_READ), jnp.float32)


def exact_moments(x: jnp.ndarray, y: jnp.ndarray, coords: jnp.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The closed-form posterior at every cell: the mean, and the covariance the paths are drawn from.

    mean = K(z, x) (K(x, x) + NOISE^2 I)^-1 y,   cov = K(z, z) - K(z, x) (K(x, x) + NOISE^2 I)^-1 K(x, z)
    """
    kernel = gpx.kernels.RBF(lengthscale=LENGTHSCALE, variance=AMPLITUDE**2, n_dims=1)
    prior = np.asarray(kernel.gram(coords).to_dense(), np.float64)  # (n_cells, n_cells)
    if x.shape[0] == 0:
        return np.zeros(coords.shape[0]), prior
    gram = np.asarray(kernel.gram(x).to_dense(), np.float64) + NOISE**2 * np.eye(x.shape[0])
    cross = np.asarray(kernel.cross_covariance(coords, x), np.float64)  # (n_cells, n_read)
    solved = np.linalg.solve(gram, cross.T)  # (n_read, n_cells)
    return cross @ np.linalg.solve(gram, np.asarray(y, np.float64)), prior - cross @ solved


def their_paths(x: jnp.ndarray, y: jnp.ndarray, coords: jnp.ndarray, key: jax.Array, n_fields: int) -> np.ndarray:
    """gpjax's own pathwise draw at every cell, as (n_fields, n_cells)."""
    prior = gpx.gps.Prior(
        kernel=gpx.kernels.RBF(lengthscale=LENGTHSCALE, variance=AMPLITUDE**2, n_dims=1), mean_function=gpx.mean_functions.Zero()
    )
    posterior = prior * gpx.likelihoods.Gaussian(num_datapoints=max(x.shape[0], 1), obs_stddev=NOISE)
    if x.shape[0] == 0:
        return np.asarray(posterior.prior.sample_approx(n_fields, key, N_FEATURES)(coords)).T
    return np.asarray(posterior.sample_approx(n_fields, gpx.Dataset(x, y[:, None]), key, N_FEATURES)(coords)).T


def our_paths(x: jnp.ndarray, y: jnp.ndarray, coords: jnp.ndarray, key: jax.Array, n_fields: int) -> np.ndarray:
    """Our pathwise draw at the same cells, as (n_fields, n_cells)."""
    from zermelo.methods.waypoint_bo.belief import GPBelief
    from zermelo.problems.ambient_dynamics.ambient_grid import PaddedGridDomain

    positions = PaddedGridDomain(n_axes=1, n_cells=N_CELLS, cell_spacing=SPACING, pad=0)
    belief = GPBelief.empty(
        positions,
        1,
        1,
        lengthscale=jnp.asarray(LENGTHSCALE),
        amplitude=jnp.asarray(AMPLITUDE),
        noise=jnp.asarray(NOISE),
        kernel_family=gpx.kernels.RBF,
        n_features=N_FEATURES,
        refit=False,
        refit_steps=0,
    )
    return np.asarray(belief._paths(key, x, y, n_fields))


def compare() -> int:
    """Our sampler against gpjax's, and our own draws against their prefixes."""
    coords, x, y = dummy_problem()
    key = jax.random.key(SEED)
    empty = jnp.zeros((0, 1), jnp.float32), jnp.zeros((0,), jnp.float32)
    cases = (("prior", empty), ("posterior", (x, y)))
    failed = 0

    # one path off a key is drawn alike either side, so the two agree bit for bit
    print(f"{'quantity':<34}{'deviation':>12}")
    for label, (xs, ys) in cases:
        agreed = Agreement.between(our_paths(xs, ys, coords, key, 1), their_paths(xs, ys, coords, key, 1))
        failed += not agreed.passed
        print(f"{f'{label}, 1 path vs gpjax':<34}{agreed.deviation:>12.3e}{'' if agreed.passed else '   FAIL'}")

    # past one path the two fill their noise draws in different orders, so only the moments are shared
    for label, (xs, ys) in cases:
        ours = our_paths(xs, ys, coords, key, N_DISTRIBUTION)  # (n_distribution, n_cells)
        theirs = their_paths(xs, ys, coords, key, N_DISTRIBUTION)
        mean, covariance = exact_moments(xs, ys, coords)
        spread = float(np.max(np.diag(covariance)))  # the largest exact variance, which tolerances scale by
        for name, ours_moment, their_moment, exact, room in (
            ("mean", np.mean(ours, axis=0), np.mean(theirs, axis=0), mean, MAX_SIGMAS * np.sqrt(np.diag(covariance) / N_DISTRIBUTION)),
            ("variance", np.var(ours, axis=0), np.var(theirs, axis=0), np.diag(covariance), MAX_MOMENT * spread),
            ("covariance", np.cov(ours.T), np.cov(theirs.T), covariance, MAX_MOMENT * spread),
        ):
            for against, agreed in (
                ("gpjax", Agreement.within(ours_moment - their_moment, MAX_MOMENT * spread)),
                ("exact", Agreement.within(ours_moment - exact, room)),
            ):
                failed += not agreed.passed
                print(f"{f'{label}, {name} vs {against}':<34}{agreed.deviation:>12.3e}{'' if agreed.passed else '   FAIL'}")

    # the property gpjax does not have: a shorter draw is the opening of a longer one
    for label, (xs, ys) in cases:
        largest = our_paths(xs, ys, coords, key, max(COUNTS))
        for n in COUNTS:
            agreed = Agreement.between(our_paths(xs, ys, coords, key, n), largest[:n])
            failed += not agreed.passed
            print(f"{f'{label}, {n} paths is a prefix':<34}{agreed.deviation:>12.3e}{'' if agreed.passed else '   FAIL'}")
    return failed


def main() -> None:
    """Comparing our pathwise sampler against gpjax's."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(compare())


if __name__ == "__main__":
    main()
