"""Validating our baseline implementations against botorch on a dummy problem."""

import argparse
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np

SAVED = pathlib.Path(__file__).with_suffix(".npz")
"""Botorch's values, committed alongside this file."""

N_CELLS, SPACING = 41, 0.25
"""A line of `N_CELLS` cells `SPACING` apart."""

LENGTHSCALE, AMPLITUDE, NOISE, JITTER = 1.0, 1.0, 0.1, 1e-6
"""k(x, x') = AMPLITUDE^2 exp(-|x - x'|^2 / (2 LENGTHSCALE^2)), and a row's noise variance is NOISE^2 + JITTER."""

N_READ, SEED = 8, 0
"""`N_READ` cells read once each, drawn under `SEED`."""

C_UPPER = 2.0
"""The multiple of the posterior deviation the optimistic bound adds."""

SETTING: dict[str, Any] = dict(
    n_cells=N_CELLS,
    spacing=SPACING,
    lengthscale=LENGTHSCALE,
    amplitude=AMPLITUDE,
    noise=NOISE,
    jitter=JITTER,
    n_read=N_READ,
    seed=SEED,
    c_upper=C_UPPER,
)
"""Every number both sides must agree on."""

QUANTITIES = ("gram", "mean", "variance", "improvement", "spread", "upper")
"""What is compared."""

MAX_DEVIATION, MIN_RANK = 1e-5, 1 - 1e-12
"""How far apart two versions of a quantity may sit, and how well their orderings must match."""


def dummy_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The problem both sides solve: the grid, the cells read, and what was read there."""
    coords = (SPACING * np.arange(N_CELLS, dtype=np.float64))[:, None]  # (n_cells, 1)
    rng = np.random.default_rng(SEED)
    read = np.sort(rng.choice(N_CELLS, N_READ, replace=False))  # (n_read,) cell indices, each read once
    return coords, read, rng.standard_normal(N_READ)


def botorch_values() -> dict[str, Any]:
    """What botorch computes on the dummy problem, and the versions that produced it."""
    import botorch
    import gpytorch
    import torch
    from botorch.acquisition.analytic import ExpectedImprovement
    from botorch.models import SingleTaskGP
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from gpytorch.means import ZeroMean

    torch.set_default_dtype(torch.float64)
    coords, read, y = dummy_problem()
    x = torch.from_numpy(coords)
    covar = ScaleKernel(RBFKernel())
    covar.base_kernel.lengthscale, covar.outputscale = LENGTHSCALE, AMPLITUDE**2
    model = SingleTaskGP(
        train_X=x[read],
        train_Y=torch.from_numpy(y)[:, None],  # (n_read, 1)
        train_Yvar=torch.full((N_READ, 1), NOISE**2 + JITTER),  # one variance per row, as a cell read once carries
        mean_module=ZeroMean(),
        covar_module=covar,
        outcome_transform=None,  # a SingleTaskGP standardizes its outcomes, which a zero-mean prior does not
        input_transform=None,
    )
    model.eval()
    beta = float(np.max(np.abs(y)))  # the largest magnitude read, which is what an improvement is measured over
    with torch.no_grad():
        posterior = model.posterior(x)
        mean, variance = posterior.mean.numpy()[:, 0], posterior.variance.numpy()[:, 0]  # (n_cells,) each
        # |f| clears beta above or below, disjointly, and each tail is one standard call
        batched = x.unsqueeze(1)  # (n_cells, 1, 1): an acquisition takes a batch of one-point candidate sets
        improvement = ExpectedImprovement(model, best_f=beta, maximize=True)(batched)
        improvement = improvement + ExpectedImprovement(model, best_f=-beta, maximize=False)(batched)
    return {
        "gram": covar(x).to_dense().detach().numpy(),  # (n_cells, n_cells)
        "mean": mean,
        "variance": variance,
        "improvement": improvement.numpy(),
        # the two below are arithmetic on the posterior above, botorch's own optimistic bound taking no
        # magnitude and so scoring a different quantity from ours
        "spread": np.sqrt(variance),
        "upper": np.abs(mean) + C_UPPER * np.sqrt(variance),
        "beta": beta,
        "stamp": np.array([f"torch=={torch.__version__}", f"botorch=={botorch.__version__}", f"gpytorch=={gpytorch.__version__}"]),
    }


def our_values() -> dict[str, Any]:
    """What we compute on the dummy problem."""
    import gpjax
    import jax.numpy as jnp

    from zermelo.methods.waypoint_bo.belief import Dataset, GPBelief
    from zermelo.methods.waypoint_bo.utility import ExpectedImprovement, PosteriorSpread, UpperConfidence
    from zermelo.problems.ambient_dynamics import PaddedGridDomain

    _, read, y = dummy_problem()
    grid = PaddedGridDomain(n_axes=1, n_cells=N_CELLS, cell_spacing=SPACING, pad=0)
    kernel = gpjax.kernels.RBF(lengthscale=jnp.asarray(LENGTHSCALE), variance=jnp.asarray(AMPLITUDE**2), n_dims=1)
    belief = GPBelief.empty(
        grid,
        N_READ,
        1,
        lengthscale=jnp.asarray(LENGTHSCALE),
        amplitude=jnp.asarray(AMPLITUDE),
        noise=jnp.asarray(NOISE),
        kernel_family=gpjax.kernels.RBF,
        n_features=64,
        refit=False,
        refit_steps=0,
    ).condition(Dataset(jnp.asarray(read), jnp.asarray(y)[:, None], jnp.ones(N_READ, bool)))
    mean, variance = belief.predict(jnp.arange(N_CELLS))  # (n_cells, 1) each
    # one cell per batch element, so a utility reduces over a single row and comes back as one value per cell
    cells = Dataset(jnp.arange(N_CELLS)[:, None], jnp.zeros((N_CELLS, 1, 1)), jnp.ones((N_CELLS, 1), bool))
    candidates = grid.narrow(jnp.ones(N_CELLS, bool))
    key = jnp.zeros(2, jnp.uint32)  # no utility below draws, so the stream is never split
    return {
        "gram": np.asarray(kernel.gram(grid.embed(grid.elements())).to_dense()),
        "mean": np.asarray(mean[:, 0]),
        "variance": np.asarray(variance[:, 0]),
        "improvement": np.asarray(ExpectedImprovement()(key, belief, cells, candidates)),
        "spread": np.asarray(PosteriorSpread()(key, belief, cells, candidates)),
        "upper": np.asarray(UpperConfidence(C_UPPER)(key, belief, cells, candidates)),
    }


@dataclass(frozen=True)
class Agreement:
    """How closely two versions of one quantity match."""

    deviation: float
    """max|ours - theirs|, over the largest value botorch reported."""

    rank: float
    """Rank correlation, over the values float32 can tell from zero."""

    same_best: bool
    """Whether both put their largest value in the same place."""

    resolved: int
    """How many values float32 can tell from zero."""

    total: int
    """How many values there are."""

    @classmethod
    def between(cls, ours: np.ndarray, theirs: np.ndarray) -> "Agreement":
        """Ours against botorch's, for one quantity."""
        peak = float(np.max(np.abs(theirs)))
        resolved = np.abs(theirs) > np.finfo(np.float32).eps * peak  # anything smaller is zero once we hold it
        our_rank = np.argsort(np.argsort(ours[resolved])).astype(np.float64)
        their_rank = np.argsort(np.argsort(theirs[resolved])).astype(np.float64)
        return cls(
            deviation=float(np.max(np.abs(ours - theirs))) / peak,
            rank=float(np.corrcoef(our_rank, their_rank)[0, 1]),
            same_best=bool(np.argmax(ours) == np.argmax(theirs)),
            resolved=int(np.sum(resolved)),
            total=int(ours.size),
        )

    @property
    def passed(self) -> bool:
        """Whether deviation, rank and best place are all within limits."""
        return self.deviation <= MAX_DEVIATION and self.rank >= MIN_RANK and self.same_best


def compare(path: pathlib.Path) -> int:
    """How many quantities ours disagrees with botorch on."""
    theirs = np.load(path)
    for name, value in SETTING.items():
        if not np.isclose(theirs[name], value):
            raise ValueError(f"{path.name} holds {name}={theirs[name]} and this file now says {value}")
    _, read, y = dummy_problem()
    if not (np.array_equal(theirs["read"], read) and np.allclose(theirs["y"], y)):
        raise ValueError(f"{path.name} was written on other readings, so nothing below is comparable")
    mine, failed = our_values(), 0
    print(f"{'quantity':<14}{'resolved':>10}{'deviation':>12}{'rank':>10}{'best':>6}")
    for name in QUANTITIES:
        agreed = Agreement.between(np.ravel(mine[name]), np.ravel(theirs[name]))
        failed += not agreed.passed
        print(
            f"{name:<14}{f'{agreed.resolved}/{agreed.total}':>10}{agreed.deviation:>12.3e}"
            f"{agreed.rank:>10.6f}{int(agreed.same_best):>6}{'' if agreed.passed else '   FAIL'}"
        )
    return failed


def main() -> None:
    """Saving botorch's values, or comparing ours against them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="compute botorch's values and save them; needs torch")
    parser.add_argument("--out", type=pathlib.Path, default=SAVED)
    args = parser.parse_args()
    if not args.generate:
        raise SystemExit(compare(args.out))
    _, read, y = dummy_problem()
    np.savez(args.out, read=read, y=y, **SETTING, **botorch_values())
    print(f"wrote {args.out}, stamped {', '.join(np.load(args.out)['stamp'])}")


if __name__ == "__main__":
    main()
