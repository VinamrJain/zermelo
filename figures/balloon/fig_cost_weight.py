"""The balloon's altitude-price tuning: each rule against w, one line per altitude budget, at step rates 0 and 1, on the held-out frame

One figure per metric (average regret, RMSE, targets reached). Rows: c = 0, c = 1. Columns: EMI, EVI.
Median over the 5 episodes, bars at the quartiles.
"""

from figures.common import LABELS, plt, quartiles, read, save, size

WEIGHTS = {"0": 0.0, "0p5": 0.5, "1": 1.0, "2": 2.0, "4": 4.0}
BUDGETS = {20: ("#fdae6b", "o"), 40: ("#e6550d", "s"), 80: ("#7f2704", "^")}
METRICS = {"average_regret": "average regret", "rmse": "RMSE", "arrival_share": "targets reached (%)"}
SCALARS = {budget: read("balloon_waypoint", {"tuning": f"paper_tuning_cost_weight_b{budget:03d}"})[1] for budget in BUDGETS}

for metric, words in METRICS.items():
    fig, axes = plt.subplots(2, 2, figsize=size(2, 0.75), constrained_layout=True, sharey="col")
    for row, rate in enumerate(("0", "1")):
        for col, rule in enumerate(("emi", "evi")):
            ax = axes[row, col]
            for shift, (budget, (colour, marker)) in enumerate(BUDGETS.items()):
                held = [quartiles(SCALARS[budget], f"{rule}_c{rate}_w{name}", metric) for name in WEIGHTS]
                ax.errorbar(
                    [at + 0.07 * (shift - 1) for at in range(len(WEIGHTS))], [mid for mid, _, _ in held],
                    yerr=[[down for _, down, _ in held], [up for _, _, up in held]],
                    color=colour, marker=marker, ms=3.5, capsize=1.5, lw=1.1, elinewidth=0.7, label=f"budget {budget}",
                )  # fmt: skip
            ax.set_xticks(range(len(WEIGHTS)), [f"{weight:g}" for weight in WEIGHTS.values()])
            ax.set_title(f"{LABELS[rule]}, $c={rate}$", fontsize=9)
            if col == 0:
                ax.set_ylabel(words)
            if row:
                ax.set_xlabel("price of an altitude change $w$")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=8)
    save(fig, f"fig_cost_weight_{metric}", "extra")
