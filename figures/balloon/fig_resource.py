"""F5: the balloon on still frames as the altitude changes an episode may spend are cut

Columns: final simple regret, final RMSE, altitude changes spent (the budget dashed), share of targets reached, against the budget. Median and quartile band over frames and seeds.
`emi_w0`, `evi_w0` charge nothing; `emi`, `evi` charge travel at step rate 1 and price an altitude change at 1.
"""

import numpy as np
import pandas as pd

from figures.common import FRAMES, LABELS, look, plt, read, save, size

BUDGETS = (20, 40, 60, 80, 100)
PANELS = (
    ("simple_regret", "simple regret"),
    ("rmse", "RMSE"),
    ("altitude_changes", "altitude changes"),
    ("arrival_share", "targets reached (%)"),
)
ARMS = ("emi", "emi_w0", "evi", "evi_w0", "mc_ei", "ucb", "ts", "max_var", "rand_target", "rand_act")

held = []
for budget in BUDGETS:
    try:
        for sweep in ("paper_resource", "paper_resource_extra"):
            held.append(read("balloon_waypoint", {date: f"{sweep}_b{budget:03d}_{date}" for date in FRAMES})[1].assign(budget=budget))
    except (FileNotFoundError, ValueError) as missing:
        print("skipped budget", budget, missing)
scalars = pd.concat(held, ignore_index=True)

fig, axes = plt.subplots(1, 4, figsize=(size(1)[0], size(1)[1] + 0.3), constrained_layout=True)
for ax, (metric, words) in zip(axes, PANELS):
    for arm in ARMS:
        rule = arm.removesuffix("_w0")
        if metric in ("rmse", "arrival_share") and arm == "rand_act":
            continue
        by_budget = scalars[scalars["acquisition"] == arm].groupby("budget")[metric].quantile([0.25, 0.5, 0.75]).unstack()
        style = {**look(rule), "markevery": 1}
        if arm.endswith("_w0"):
            style.update(ls=":", mfc="white")
        ax.plot(by_budget.index, by_budget[0.5], label=LABELS[rule] + (" (no cost)" if arm.endswith("_w0") else ""), **style)
        ax.fill_between(by_budget.index, by_budget[0.25], by_budget[0.75], color=style["color"], alpha=0.10, lw=0)
    if metric == "altitude_changes":
        ax.plot(BUDGETS, BUDGETS, color="black", lw=0.6, ls="--", zorder=0)
    ax.set_xlabel("altitude budget")
    ax.set_title(words, fontsize=9)
    ax.set_xticks(BUDGETS)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8, columnspacing=1.0, handlelength=2.4)
save(fig, "fig_resource", "main")
