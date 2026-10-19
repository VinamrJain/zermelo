"""The ocean visiting study: ETR against its endpoint-only twin and RandGoal

(a) goals visited against hours elapsed at `SHOWN` goals; (b) share of goals visited by the horizon against the number of goals.
Median over frames x seeds; (a) the quartile band, (b) bars at the quartiles. No argument: 10 goals from `visit_<date>`, the
others from `visit_g<NN>_<date>`.

Run from code/:  pixi run fig ocean.fig_goals [prefix] [shown]     (prefix: every count from `<prefix>_g<NN>_<date>`, e.g. visit_L50)
"""

import sys

import numpy as np
import pandas as pd

from figures.common import LABELS, OCEAN_FRAMES, band, look, plt, read, save, size

GOALS = (5, 10, 20, 30)
RULES = ("etr", "etr_endpoint", "rand_goal")
PREFIX = sys.argv[1] if len(sys.argv) > 1 else None
SHOWN = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def sweep(n: int, date: str) -> str:
    """The sweep holding `n` goals on `date`"""
    return f"{PREFIX}_g{n:02d}_{date}" if PREFIX else f"visit_{date}" if n == 10 else f"visit_g{n:02d}_{date}"


tables = {n: read("ocean_waypoint", {date: sweep(n, date) for date in OCEAN_FRAMES}) for n in GOALS}
scalars = pd.concat(tables[n][1].assign(goals=n) for n in GOALS)
scalars["visited_share"] = 100 * scalars["targets_visited"] / scalars["goals"]

fig, (over_time, by_goals) = plt.subplots(1, 2, figsize=size(1, 0.8), constrained_layout=True)
for rule in RULES:
    band(over_time, tables[SHOWN][0], rule, "targets_visited", markevery=0.15)
over_time.set_ylim(0, SHOWN)
over_time.margins(x=0)
over_time.set_xlabel("hours elapsed")
over_time.set_ylabel("goals visited")
over_time.set_title(f"(a) {SHOWN} goals", fontsize=9)

for shift, rule in enumerate(RULES):
    by_n = scalars[scalars["acquisition"] == rule].groupby("goals")["visited_share"].quantile([0.25, 0.5, 0.75]).unstack().reindex(GOALS)
    at = np.array(GOALS) * 1.05 ** (shift - 1)  # rules side by side at each count
    mid = by_n[0.5].to_numpy()
    style = {k: v for k, v in look(rule).items() if k != "markevery"}
    by_goals.errorbar(at, mid, yerr=[mid - by_n[0.25], by_n[0.75] - mid], capsize=1.5, elinewidth=0.7, **style)
by_goals.set_xscale("log")
by_goals.set_xticks(GOALS, [str(n) for n in GOALS])
by_goals.minorticks_off()
by_goals.set_ylim(0, 105)
by_goals.set_xlabel("goals")
by_goals.set_ylabel("goals visited (%)")
by_goals.set_title("(b) by the horizon", fontsize=9)

fig.legend(*over_time.get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=8)
save(fig, f"fig_ocean_goals{'_' + PREFIX if PREFIX else ''}{'' if SHOWN == 20 else f'_{SHOWN}'}", "main")
