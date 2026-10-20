"""F5: the balloon on still frames as the altitude changes an episode may spend are cut

Columns: final simple regret against the budget, simple regret against hours with no altitude budget, altitude changes per target chosen (total over the episode / targets chosen),
share of targets reached, against the budget. Median and quartile band over frames and seeds.
`emi_w0` charges nothing; `emi` charges the altitude changes of its walks at lambda 0.5.
"""

import numpy as np
import pandas as pd

from figures.common import FRAMES, LABELS, launch, look, plt, read, save, size

BUDGETS = (20, 40, 60, 80, 100, 120, 240)
PANELS = (
    ("simple_regret", "simple regret"),
    ("simple_regret", "simple regret, no budget"),
    ("changes_per_target", "alt. changes per target"),
    ("arrival_share", "targets reached (%)"),
)
ARMS = ("emi", "emi_w0", "mc_ei", "ucb", "ts", "rand_target", "rand_act")


def changes_per_target(budget: int, date: str) -> pd.DataFrame:
    """Per run of one sweep: altitude changes over the episode / targets chosen, a reset of steps_since_target starting a new one"""
    rows = []
    for record in launch("balloon_waypoint", f"paper_resource_b{budget:03d}_{date}").glob("arm=*/record.npz"):
        arm, seed = record.parent.name.removeprefix(f"arm=paper_resource_b{budget:03d}_{date}_").split(",seed=")
        held = np.load(record)
        if "agent_state/steps_since_target" not in held:  # a rule choosing no target
            continue
        targets = (np.diff(held["agent_state/steps_since_target"].ravel()) < 0).sum() + 1
        rows.append((arm, int(seed), date, budget, np.abs(np.diff(held["state/altitude"].ravel())).sum() / targets))
    return pd.DataFrame(rows, columns=["acquisition", "seed", "world", "budget", "changes_per_target"])


held = []
for budget in BUDGETS:
    try:
        held.append(read("balloon_waypoint", {date: f"paper_resource_b{budget:03d}_{date}" for date in FRAMES})[1].assign(budget=budget))
    except (FileNotFoundError, ValueError) as missing:
        print("skipped budget", budget, missing)
scalars = pd.concat(held, ignore_index=True).merge(
    pd.concat([changes_per_target(budget, date) for budget in BUDGETS for date in FRAMES]),
    on=["acquisition", "seed", "world", "budget"],
    how="left",
)
curves = read("balloon_waypoint", {date: f"paper_main_{date}" for date in FRAMES})[
    0
]  # the altitude budget never binding, our rules uncharged

fig, axes = plt.subplots(1, 4, figsize=size(1), constrained_layout=True)
for col, (ax, (metric, words)) in enumerate(zip(axes, PANELS)):
    for arm in ARMS:
        rule = arm.removesuffix("_w0")
        if (metric in ("arrival_share", "changes_per_target") and arm == "rand_act") or (col == 1 and arm == "emi_w0"):
            continue
        by_budget = (
            (curves if col == 1 else scalars)[lambda table: table["acquisition"] == arm]
            .groupby("x" if col == 1 else "budget")[metric]
            .quantile([0.25, 0.5, 0.75])
            .unstack()
        )
        style = {**look(rule), "markevery": 0.2 if col == 1 else 1}
        if arm.endswith("_w0"):
            style.update(ls=":", mfc="white")
        at = by_budget.index if col == 1 else [BUDGETS.index(budget) for budget in by_budget.index]  # budgets evenly spaced
        ax.plot(at, by_budget[0.5], label=LABELS[rule] + (" (no cost)" if arm.endswith("_w0") else ""), **style)
        ax.fill_between(at, by_budget[0.25], by_budget[0.75], color=style["color"], alpha=0.10, lw=0)
    ax.set_title(words, fontsize=9)
    if col == 1:
        ax.set_xlabel("hours elapsed")
        ax.margins(x=0)
    else:
        ax.set_xlabel("altitude budget")
        ax.set_xticks(range(len(BUDGETS)), [*map(str, BUDGETS[:-1]), r"$\infty$"], fontsize=7)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), fontsize=8, columnspacing=0.9, handlelength=2.2)
save(fig, "fig_resource", "main")
