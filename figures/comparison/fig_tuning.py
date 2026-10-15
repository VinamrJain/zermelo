"""EMI and EVI against the step rate c, one line per planning budget L, on the held-out seeds and frame

appendix/fig_sensitivity          rows 2D grid, balloon; EMI simple regret, EMI average regret, EVI RMSE, EMI targets reached
extra/fig_tuning_<problem> rows EMI, EVI; columns simple regret, average regret, RMSE, targets reached
Median over the 5 episodes, bars at the quartiles.
"""

from figures.common import LABELS, plt, quartiles, read, save, size

RATES = {"0": 0.0, "0p5": 0.5, "1": 1.0, "2": 2.0}
BUDGETS = {25: ("#9ecae1", "o"), 50: ("#4292c6", "s"), 100: ("#08306b", "^")}
METRICS = {"simple_regret": "simple regret", "average_regret": "average regret", "rmse": "RMSE", "arrival_share": "targets reached (%)"}
PROBLEMS = {"2D grid": "ambient_waypoint", "balloon": "balloon_waypoint"}
SCALARS = {name: read(experiment, {"tuning": "paper_tuning"})[1] for name, experiment in PROBLEMS.items()}


def panel(ax, problem: str, rule: str, metric: str) -> None:
    """One rule's final `metric` against c, a line per L"""
    for shift, (budget, (colour, marker)) in enumerate(BUDGETS.items()):
        held = [quartiles(SCALARS[problem], f"{rule}_L{budget}_c{name}", metric) for name in RATES]
        ax.errorbar(
            [at + 0.07 * (shift - 1) for at in range(len(RATES))], [mid for mid, _, _ in held],
            yerr=[[down for _, down, _ in held], [up for _, _, up in held]],
            color=colour, marker=marker, ms=3.5, capsize=1.5, lw=1.1, elinewidth=0.7, label=f"$L={budget}$",
        )  # fmt: skip
    ax.set_xticks(range(len(RATES)), [f"{rate:g}" for rate in RATES.values()])
    ax.set_ylabel(METRICS[metric])


def legend(fig, ax) -> None:
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=8, handlelength=2.2)


fig, axes = plt.subplots(2, 4, figsize=size(2, 1.25), constrained_layout=True)
for row, problem in enumerate(PROBLEMS):
    for ax, (rule, metric) in zip(
        axes[row], (("emi", "simple_regret"), ("emi", "average_regret"), ("evi", "rmse"), ("emi", "arrival_share"))
    ):
        panel(ax, problem, rule, metric)
        ax.set_title(f"{LABELS[rule]}, {problem}", fontsize=9)
        if row:
            ax.set_xlabel("step rate $c$")
legend(fig, axes[0, 0])
save(fig, "fig_sensitivity", "appendix")

for problem in PROBLEMS:
    fig, axes = plt.subplots(2, 4, figsize=size(2, 1.25), constrained_layout=True)
    for row, rule in enumerate(("emi", "evi")):
        for col, metric in enumerate(METRICS):
            panel(axes[row, col], problem, rule, metric)
            axes[row, col].set_title(f"{LABELS[rule]}, {problem}", fontsize=9)
            if row:
                axes[row, col].set_xlabel("step rate $c$")
    legend(fig, axes[0, 0])
    save(fig, "fig_tuning_" + problem.lower().replace(" ", "_"), "extra")
