"""B2a: the weight lambda on altitude changes, alpha = (1 - lambda) u~ - lambda C~, on the held-out frame, one line per altitude budget

Rows: EMI, EVI. Columns: the rule's own scores, then altitude changes spent and targets reached. x = lambda, 0 charges nothing.
Median over the episodes, bars at the quartiles.
"""

from figures.common import LABELS, plt, quartiles, read, save, size

WEIGHTS = {"0": 0.0, "0p25": 0.25, "0p5": 0.5, "0p75": 0.75}
BUDGETS = {20: ("#fdae6b", "o"), 40: ("#e6550d", "s"), 80: ("#7f2704", "^")}
COLUMNS = {
    "emi": (("simple_regret", "simple regret"), ("average_regret", "average regret")),
    "evi": (("rmse", "RMSE"), ("posterior_variance", "posterior variance")),
}  # rule: its own scores
SHARED = (("altitude_changes", "altitude changes"), ("arrival_share", "targets reached (%)"))
SCALARS = {budget: read("balloon_waypoint", {"tuning": f"paper_tuning_cost_weight_b{budget:03d}"})[1] for budget in BUDGETS}

fig, axes = plt.subplots(2, 4, figsize=size(2), constrained_layout=True)
for row, (rule, own) in enumerate(COLUMNS.items()):
    for col, (metric, words) in enumerate((*own, *SHARED)):
        ax = axes[row, col]
        for shift, (budget, (colour, marker)) in enumerate(BUDGETS.items()):
            held = [quartiles(SCALARS[budget], f"{rule}_c{name}", metric) for name in WEIGHTS]
            ax.errorbar(
                [at + 0.07 * (shift - 1) for at in range(len(WEIGHTS))], [mid for mid, _, _ in held],
                yerr=[[down for _, down, _ in held], [up for _, _, up in held]],
                color=colour, marker=marker, ms=3.5, capsize=1.5, lw=1.1, elinewidth=0.7, label=f"budget {budget}",
            )  # fmt: skip
        if metric == "altitude_changes":
            for budget, (colour, _) in BUDGETS.items():
                ax.axhline(budget, color=colour, lw=0.6, ls="--", zorder=0)
        ax.set_xticks(range(len(WEIGHTS)), [f"{weight:g}" for weight in WEIGHTS.values()])
        ax.set_title(f"{LABELS[rule]}: {words}", fontsize=8)
        ax.tick_params(labelsize=7)
        if row:
            ax.set_xlabel(r"$\lambda$", fontsize=8)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=8)
save(fig, "fig_cost_weight", "appendix")
