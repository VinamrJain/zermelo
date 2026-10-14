"""Where each rule goes over an episode: how much of the region it covers, how long it stays inside it, and how uncertain its belief remains

Rows: 2D grid, balloon (pooled over frames). Columns, each up to the time_step drawn:
distinct cells visited as a share of the region's cells, share of time_steps spent inside the region, and the posterior variance, mean over the region's cells of the variance summed over the field's axes.
2D grid: the region is the 50 x 50 scored cells. Balloon: the box's 1 degree cells at each altitude level.
Median over episodes; the quartile band is drawn for our rules only.
"""

import json

import numpy as np
import pandas as pd

from figures.common import FRAMES, LABELS, OURS, launch, look, plt, save, size

ARMS = ("emi", "evi", "mc_ei", "ucb", "ts", "max_var", "rand_target", "rand_act")
SKIPPED = 0.04  # opening share of the episode left out, where an average over a few steps is noise


def visited_share(cells: np.ndarray, inside: np.ndarray, total: int) -> np.ndarray:
    """Share of `total` cells stood in by each time_step, `cells` one integer row per step and `inside` whether it is in the region"""
    _, first = np.unique(cells, axis=0, return_index=True)
    new = np.zeros(len(cells), dtype=bool)
    new[first] = True
    return np.cumsum(new & inside) / total


def shares(x: np.ndarray, cells: np.ndarray, inside: np.ndarray, total: int, log_variance: np.ndarray) -> dict:
    """Share of the region covered, share of time_steps spent inside it, and the posterior variance, each up to every time_step

    `log_variance` is (time_steps + 1, cells, axes), one claim per time_step.
    """
    steps = np.arange(1, len(x) + 1)
    variance = np.exp(np.maximum(log_variance[1:], -12.0)).sum(axis=-1).mean(axis=-1)
    return {
        "x": x,
        "coverage": 100 * visited_share(cells, inside, total)[1:],
        "inside": 100 * np.cumsum(inside[1:]) / steps,
        "variance": variance,
    }


def grid(held, problem: dict) -> dict:
    """The 2D grid's region is its scored cells"""
    at = np.concatenate([held["state/ambient"], held["state/controllable"]], axis=1)  # (time_steps + 1, 2)
    cells = np.rint(at).astype(int)
    inside = ((cells >= 0) & (cells < [problem["ambient_cells"], problem["controllable_cells"]])).all(axis=1)
    log_variance = np.split(held["objective_state/claim"], 2, axis=-1)[1]  # the claim is the mean then the log variance
    return shares(np.arange(1, len(at)), cells, inside, problem["ambient_cells"] * problem["controllable_cells"], log_variance)


def balloon(held, problem: dict) -> dict:
    """The balloon's region is the box's (latitude, longitude, altitude) cells"""
    at = held["state/position"]  # (time_steps + 1, 2), latitude then longitude
    inside = (
        (at[:, 0] >= problem["lat_min"])
        & (at[:, 0] <= problem["lat_max"])
        & (at[:, 1] >= problem["lon_min"])
        & (at[:, 1] <= problem["lon_max"])
    )
    cells = np.column_stack([np.rint(at).astype(int), held["state/altitude"]])
    return shares(
        held["state/hours_elapsed"][1:], cells, inside, held["objective_state/truth"].shape[1], held["objective_state/claim"][..., 2:4]
    )


ROWS = (
    ("ambient_waypoint", "2D grid", "time step", ("paper_main",), grid),
    ("balloon_waypoint", "balloon", "hours elapsed", tuple(f"paper_main_{date}" for date in FRAMES), balloon),
)


def draw(axes) -> None:
    """The six panels on a 2 x 3 block of axes"""
    for row, (experiment, name, x_label, sweeps, measure) in enumerate(ROWS):
        curves = []
        for sweep in sweeps:
            for run in sorted(launch(experiment, sweep).glob("*/record.npz")):
                arm = run.parent.name.split(",")[0].removeprefix(f"arm={sweep}_")
                if arm not in ARMS:
                    continue
                config = json.loads((run.parent / "config.json").read_text())
                with np.load(run) as record:
                    held = {
                        key: record[key]
                        for key in record.files
                        if key.startswith("state/") or key in ("objective_state/truth", "objective_state/claim")
                    }
                measured = measure(held, config["problem"])
                if config.get("method") is None:  # no rule, so no belief of its own
                    measured["variance"] = np.full(len(measured["x"]), np.nan)
                curves.append(pd.DataFrame({"acquisition": arm, "episode": f"{sweep}/{run.parent.name}", **measured}))
        curves = pd.concat(curves, ignore_index=True)
        curves = curves[curves["x"] >= SKIPPED * curves["x"].max()]
        for ax, (metric, words) in zip(
            axes[row], (("coverage", "region covered (%)"), ("inside", "time in region (%)"), ("variance", "posterior variance"))
        ):
            for arm in reversed(ARMS):  # ours drawn last, over the baselines
                by_x = curves[curves["acquisition"] == arm].groupby("x")[metric].quantile([0.25, 0.5, 0.75]).unstack()
                if by_x.empty or by_x[0.5].isna().all():
                    continue
                style = look(arm)
                ax.plot(
                    by_x.index,
                    by_x[0.5],
                    label=LABELS[arm],
                    **{**style, "lw": 1.5 if arm in OURS else 0.9, "zorder": 3 if arm in OURS else 2},
                )
                if arm in OURS:
                    ax.fill_between(by_x.index, by_x[0.25], by_x[0.75], color=style["color"], alpha=0.15, lw=0, zorder=1)
            ax.set_ylabel(words if metric != "coverage" else f"{name}\n{words}")
            ax.set_xlabel(x_label)
            ax.margins(x=0)
            ax.grid(axis="y", lw=0.3, alpha=0.5)


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 3, figsize=size(2), constrained_layout=True)
    draw(axes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="outside lower center", ncol=8, fontsize=8, columnspacing=1.0, handlelength=2.2)
    save(fig, "fig_travel", "main")
