"""The ocean visiting study: ETR against its endpoint-only twin and RandGoal

(a) goals visited against hours elapsed at `SHOWN` goals; (b) share of goals visited by the horizon against the number of goals.
Median over frames x seeds; (a) the quartile band, (b) bars at the quartiles. No argument: 10 goals from `visit_<date>`, the
others from `visit_g<NN>_<date>`. `_tracks_side` adds one episode's tracks at `SHOWN` goals,
a map per rule under (a), (b) and the legend, each track fading from light at the start (open dot) to dark at the end (filled dot).

Run from code/:  pixi run fig ocean.fig_goals [prefix] [shown]     (prefix: every count from `<prefix>_g<NN>_<date>`, e.g. visit_L50)
"""

import sys

import cartopy.feature as cfeature
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon
from shapely.geometry import box as rectangle

from figures.common import LABELS, OCEAN_FRAMES, band, launch, look, plt, read, save, size

GOALS = (5, 10, 20, 30)
RULES = ("etr", "etr_endpoint", "rand_goal")
PREFIX = sys.argv[1] if len(sys.argv) > 1 else None
SHOWN = int(sys.argv[2]) if len(sys.argv) > 2 else 20
NAME = f"fig_ocean_goals{'_' + PREFIX if PREFIX else ''}{'' if SHOWN == 20 else f'_{SHOWN}'}"
TRACKED = ("jun16", 0)  # ETR 17, ETR-Endpoint 10, RandGoal 8 of 20: the gap nearest the median over the 12 episodes
MARGIN = 0.4  # degrees of sea shown around the tracks and goals
SHORE, COAST, GOAL = "#efe6d2", "#7a6f60", "#555555"


def sweep(n: int, date: str) -> str:
    """The sweep holding `n` goals on `date`"""
    return f"{PREFIX}_g{n:02d}_{date}" if PREFIX else f"visit_{date}" if n == 10 else f"visit_g{n:02d}_{date}"


tables = {n: read("ocean_waypoint", {date: sweep(n, date) for date in OCEAN_FRAMES}) for n in GOALS}
scalars = pd.concat(tables[n][1].assign(goals=n) for n in GOALS)
scalars["visited_share"] = 100 * scalars["targets_visited"] / scalars["goals"]


def summary(over_time, by_goals) -> None:
    """(a) goals visited against hours at `SHOWN` goals; (b) the share visited by the horizon against the goal count"""
    for rule in RULES:
        band(over_time, tables[SHOWN][0], rule, "targets_visited", markevery=0.15)
    over_time.set(ylim=(0, SHOWN), xlabel="hours elapsed", ylabel="goals visited")
    over_time.margins(x=0)
    over_time.set_title(f"(a) {SHOWN} goals", fontsize=9)
    for shift, rule in enumerate(RULES):
        by_n = (
            scalars[scalars["acquisition"] == rule].groupby("goals")["visited_share"].quantile([0.25, 0.5, 0.75]).unstack().reindex(GOALS)
        )
        mid = by_n[0.5].to_numpy()
        by_goals.errorbar(
            np.array(GOALS) * 1.05 ** (shift - 1), mid, yerr=[mid - by_n[0.25], by_n[0.75] - mid], capsize=1.5, elinewidth=0.7,
            **{k: v for k, v in look(rule).items() if k != "markevery"},
        )  # rules side by side at each count  # fmt: skip
    by_goals.set(xscale="log", ylim=(0, 105), xlabel="goals", ylabel="goals visited (%)")
    by_goals.set_xticks(GOALS, [str(n) for n in GOALS])
    by_goals.minorticks_off()
    by_goals.set_title("(b) by the horizon", fontsize=9)


def tracks(date: str, seed: int) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Each rule's track to its last visit or the horizon, (moves, 2) as (lat, lon), and the goals it visited, (goals,); the goals, (goals, 2)"""
    where = launch("ocean_waypoint", sweep(SHOWN, date))
    held = {}
    for rule in RULES:
        record = np.load(where / f"arm={sweep(SHOWN, date)}_{rule},seed={seed}" / "record.npz")
        visited = record["objective_state/visited"]  # (moves + 1, goals)
        end = int(np.argmax(visited.all(axis=-1))) + 1 if visited[-1].all() else len(visited)
        held[rule] = (record["state/position"][:end].astype(float), visited[-1])
    return held, record["objective_state/targets"][0]


def chart(ax, shown: np.ndarray) -> None:
    """Land around `shown`, (points, 2) as (lat, lon), with `MARGIN` degrees of sea, kilometres equal on both axes, no ticks"""
    lon_min, lat_min = shown[:, 1].min() - MARGIN, shown[:, 0].min() - MARGIN
    lon_max, lat_max = shown[:, 1].max() + MARGIN, shown[:, 0].max() + MARGIN
    frame = rectangle(lon_min, lat_min, lon_max, lat_max)
    for shape in cfeature.NaturalEarthFeature("physical", "land", "50m").geometries():
        for part in getattr(shape.intersection(frame), "geoms", [shape.intersection(frame)]):
            if not part.is_empty and part.geom_type == "Polygon":
                ax.add_patch(Polygon(np.asarray(part.exterior.coords), closed=True, fc=SHORE, ec=COAST, lw=0.5, zorder=1))
    ax.set(xlim=(lon_min, lon_max), ylim=(lat_min, lat_max), xticks=[], yticks=[])
    ax.set_aspect(1 / np.cos(np.radians(shown[:, 0].mean())))


def legend(fig, ax) -> None:
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=8)


fig, axes = plt.subplots(1, 2, figsize=size(1, 0.8), constrained_layout=True)
summary(*axes)
legend(fig, axes[0])
save(fig, NAME, "main")

held, goals = tracks(*TRACKED)
shown = np.r_[goals, *(track for track, _ in held.values())]
start = held[RULES[0]][0][0]

fig = plt.figure(figsize=size(2, 1.0), constrained_layout=True)
top, bottom = fig.subfigures(2, 1, height_ratios=(1, 1.1))
over_time, by_goals, key = top.subplots(1, 3, width_ratios=(1, 1, 0.45))
summary(over_time, by_goals)
key.axis("off")
key.legend(*over_time.get_legend_handles_labels(), loc="center", fontsize=8, frameon=False)
for ax, rule, letter in zip(bottom.subplots(1, 3), RULES, "cde", strict=True):
    track, visited = held[rule]
    colour = look(rule)["color"]
    chart(ax, shown)
    fade = np.linspace(0.05, 1, len(track) - 1)[:, None] ** 0.7  # (moves - 1, 1): light at the start, the rule's colour at the end
    path = track[:, ::-1].reshape(-1, 1, 2)  # (moves, 1, 2) as (lon, lat)
    ax.add_collection(
        LineCollection(np.concatenate([path[:-1], path[1:]], axis=1), colors=1 - fade * (1 - np.array(to_rgb(colour))), lw=1.0, zorder=3)
    )
    ax.scatter(
        goals[:, 1], goals[:, 0], marker="*", s=45, c=[colour if seen else "white" for seen in visited], edgecolors=GOAL, lw=0.6, zorder=4
    )
    ax.scatter(start[1], start[0], marker="o", s=22, color="white", edgecolors="black", lw=1.0, zorder=5)
    ax.scatter(track[-1, 1], track[-1, 0], marker="o", s=22, color="black", zorder=5)
    ax.set_title(f"({letter}) {LABELS[rule]}: {int(visited.sum())} of {SHOWN}", fontsize=9)
save(fig, NAME + "_tracks_side", "main")
