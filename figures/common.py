"""What every paper figure script shares: reading launches pooled over worlds, the arms' looks, and a median curve with its band

Scripts run from code/:  PYTHONPATH=.:../paper/junk/figures pixi run python ../paper/junk/figures/<script>.py
Figures land in results/figures/{main,appendix,extra}/.
"""

import importlib
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path(os.environ["RESULTS"])  # the results tree read; unset, the script refuses to run
LAUNCH_AT = os.environ.get("LAUNCH_AT")  # a launch's timestamp, e.g. 2026-10-14_23-59-59: read the newest launch at or before it
BUILD = Path("results") / "figures"
WIDTH = 6.75  # inches, the full text width
ROW_HEIGHT = 1.35  # inches one row of panels is drawn at; every figure's height follows from it
LEGEND_HEIGHT = 0.35  # inches the legend under the panels takes


def size(rows: int, width: float = 1.0) -> tuple[float, float]:
    """Figure size in inches for `rows` rows of panels at `width` text widths, with room for the legend"""
    return (WIDTH * width, rows * ROW_HEIGHT + LEGEND_HEIGHT)


FRAMES = ("sep06", "sep09", "sep19", "sep24")
OCEAN_FRAMES = ("jun04", "jun16", "jun22", "jun28")

LABELS = {
    "emi": "EMI", "evi": "EVI", "eii": "EII", "esi": "ESI", "ei": "EI", "mc_ei": "MC-EI", "ucb": "UCB", "ts": "TS", "max_var": "MaxVar",
    "rand_target": "RandTarget", "rand_act": "RandAct", "oracle_emi": "Oracle-EMI",
    "eti": "ETI", "info": "IG", "direct": "Direct", "rand_wp": "RandTarget", "oracle": "Oracle",
    "etr_l0": r"ETR $\lambda$=0", "etr_l05": r"ETR $\lambda$=0.5", "etr_l1": r"ETR $\lambda$=1", "nearest": "Nearest", "oracle_etr": "Oracle-ETR",
}  # fmt: skip
RATES = ("0.25", "0.5", "1", "2")  # lambda of each IG arm, light to dark
LABELS.update({f"info_{rate}": rf"IG $\lambda$={rate}" for rate in RATES})
LABELS.update({f"nearest_{suffix}": rf"Nearest $\lambda$={rate}" for suffix, rate in (("l0", "0"), ("l05", "0.5"), ("l1", "1"))})
LABELS.update(
    {f"oracle_{rule}_{suffix}": "Oracle-" + LABELS[f"{rule}_{suffix}"] for rule in ("etr", "nearest") for suffix in ("l0", "l05", "l1")}
)
FAMILY_COLOURS = {
    "EMI": "0173B2", "ESI": "D55E00", "EVI": "029E73", "EII": "0173B2", "EI": "DE8F05", "MC-EI": "56B4E9", "UCB": "7F3C8D", "TS": "CC78BC",
    "MaxVar": "CA9161", "RandTarget": "555555", "RandAct": "A0A0A0", "Oracle-EMI": "000000", "Forecast": "000000", "ETI": "0173B2",
    "IG": "D55E00", "Direct": "7F3C8D", "Oracle": "000000", "ETR": "0173B2", "Nearest": "7F3C8D", "Oracle-ETR": "000000",
}  # hex colour per acquisition family  # fmt: skip
IG_LOOK = {"0.25": ("#56B4E9", "s"), "0.5": ("#009E73", "D"), "1": ("#D55E00", "o"), "2": ("#B2182B", "P")}  # (colour, marker) per lambda
OURS = ("emi", "evi", "eii", "esi", "eti", "etr_l0", "etr_l05", "etr_l1")
MARKERS = {
    "etr_l0": "s",
    "etr_l05": "o",
    "etr_l1": "D",
    "nearest": "v",
    "eti": "o",
    "eii": "o",
    "info": "s",
    "direct": "^",
    "rand_wp": "*",
    "emi": "o",
    "evi": "s",
    "esi": "^",
    "ei": "D",
    "mc_ei": "D",
    "ucb": "v",
    "ts": "P",
    "max_var": "X",
    "rand_target": "*",
}

plt.rcParams.update(
    {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.linewidth": 0.6, "axes.spines.top": False,
     "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 600, "lines.linewidth": 1.3, "xtick.major.width": 0.5, "ytick.major.width": 0.5}
)  # fmt: skip


def look(arm: str) -> dict:
    """Colour, line and marker of one arm: ours solid, baselines dashed, the two references unmarked"""
    family = LABELS.get(arm, arm)
    colour = "#" + FAMILY_COLOURS.get(family, "333333")
    if arm in ("oracle_emi", "oracle", "oracle_etr"):
        return dict(color="black", ls=(0, (5, 2)), lw=1.0)
    if arm == "rand_act":
        return dict(color=colour, ls=":", lw=1.2)
    if arm.startswith(("oracle_etr_l", "oracle_nearest_l")):  # its rule's look, dotted
        return {**look(arm.removeprefix("oracle_")), "ls": ":"}
    if arm.startswith(("etr_l", "nearest_l")):  # one colour and marker per lambda, as the IG arms; ETR solid, Nearest dashed
        colour, marker = IG_LOOK[{"0": "0.25", "05": "0.5", "1": "1"}[arm.split("_l")[-1]]]
        return dict(color=colour, ls="-" if arm.startswith("etr") else "--", marker=marker, ms=3.5, markevery=0.2)
    if arm.startswith("info_"):  # one colour and marker per lambda
        colour, marker = IG_LOOK[arm.removeprefix("info_")]
        return dict(color=colour, ls="-", marker=marker, ms=3.5, markevery=0.2)
    return dict(color=colour, ls="-" if arm in OURS else "--", marker=MARKERS.get(arm), ms=3.5, markevery=0.2)


def launch(experiment: str, sweep: str) -> Path:
    """The newest launch of one sweep, at or before LAUNCH_AT where it is set, printed"""
    newest = max(
        found for found in (RESULTS / experiment / sweep).iterdir() if found.is_dir() and (LAUNCH_AT is None or found.name <= LAUNCH_AT)
    )
    print(f"reading {newest}")
    return newest


def read(experiment: str, sweeps: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Curves and scalars of the newest launch of each sweep, pooled, `world` naming the sweep's key and `episode` one (world, seed)

    Cached beside each launch as paper_curves.pkl and paper_scalars.pkl, redone when a record is newer.
    """
    data = importlib.import_module(f"zermelo.experiments.{experiment}.metrics.data")
    curves, scalars = [], []
    for world, sweep in sweeps.items():
        where = launch(experiment, sweep)
        cached = [where / "paper_curves.pkl", where / "paper_scalars.pkl"]
        newest = max(record.stat().st_mtime for record in where.glob("*/record.npz"))
        if all(path.exists() and path.stat().st_mtime > newest for path in cached):
            one_curves, one_scalars = (pd.read_pickle(path) for path in cached)
        else:
            one_curves, one_scalars = data.tables(data.finished_runs(where))
            one_curves.to_pickle(cached[0])
            one_scalars.to_pickle(cached[1])
        for table, held in ((one_curves, curves), (one_scalars, scalars)):
            held.append(table.assign(world=world, episode=world + "/" + table["seed"].astype(str)))
    pooled_curves, pooled_scalars = pd.concat(curves, ignore_index=True), pd.concat(scalars, ignore_index=True)
    if "average_regret" not in pooled_curves and "cumulative_regret" in pooled_curves:
        steps = pooled_curves.groupby(["acquisition", "episode"]).cumcount() + 1
        pooled_curves["average_regret"] = pooled_curves["cumulative_regret"] / steps
        pooled_scalars["average_regret"] = pooled_scalars["cumulative_regret"] / steps.max()
    return pooled_curves, pooled_scalars


def band(ax, curves: pd.DataFrame, arm: str, metric: str, quartiles: bool = True, **over) -> None:
    """Median over episodes against x, with the quartile band where `quartiles`"""
    held = curves[curves["acquisition"] == arm]
    if held.empty:
        return
    by_x = held.groupby("x")[metric].quantile([0.25, 0.5, 0.75]).unstack()
    style = {**look(arm), **over}
    ax.plot(by_x.index, by_x[0.5], label=LABELS.get(arm, arm), **style)
    if quartiles:
        ax.fill_between(by_x.index, by_x[0.25], by_x[0.75], color=style["color"], alpha=0.12, lw=0)


def quartiles(scalars: pd.DataFrame, arm: str, metric: str) -> tuple[float, float, float]:
    """Median, and the distance down to the lower and up to the upper quartile, of one arm's final values"""
    values = scalars.loc[scalars["acquisition"] == arm, metric].dropna()
    low, mid, high = np.quantile(values, [0.25, 0.5, 0.75]) if len(values) else (np.nan,) * 3
    return mid, mid - low, high - mid


def save(fig, name: str, where: str) -> None:
    """The figure as pdf and png under the build directory's `where` folder (main, appendix or extra)"""
    (BUILD / where).mkdir(parents=True, exist_ok=True)
    fig.savefig(BUILD / where / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(BUILD / where / f"{name}.png", dpi=300, bbox_inches="tight")
    print(BUILD / where / f"{name}.png")
