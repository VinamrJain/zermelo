"""Four rules on the 2D grid under each planner: value iteration, a greedy step toward the target, and a random walk

Rows: average regret, RMSE, against the time_step. Columns: the planner. Median and quartile band over seeds; a row shares its axis.
"""

from figures.common import band, plt, read, save, size

PLANNERS = (("value_iteration", "value iteration"), ("greedy", "greedy step"), ("random_walk", "random walk"))
RULES = ("emi", "evi", "mc_ei", "rand_target")
PANELS = (("average_regret", "average regret"), ("rmse", "RMSE"))

curves, _ = read("ambient_waypoint", {"planners": "paper_planners"})
curves = curves[curves["x"] >= 0.04 * curves["x"].max()]  # the first moves, where an average over a few steps is noise
fig, axes = plt.subplots(2, 3, figsize=size(2), sharex=True, sharey="row", constrained_layout=True)
for top, (row, (metric, words)) in zip((True, False), zip(axes, PANELS)):
    for ax, (planner, name) in zip(row, PLANNERS):
        for rule in RULES:
            band(ax, curves.assign(acquisition=curves["acquisition"].str.removesuffix(f"_{planner}")), rule, metric)
        ax.margins(x=0)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        if top:
            ax.set_title(name, fontsize=9)
        else:
            ax.set_xlabel("time step")
    row[0].set_ylabel(words)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8, columnspacing=1.0, handlelength=2.2)
save(fig, "fig_planners", "appendix")
