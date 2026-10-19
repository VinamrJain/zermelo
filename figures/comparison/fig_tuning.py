"""Our rules against the planning budget L on held-out instances: one row per problem, final scores and seconds per move

appendix/fig_sensitivity          rows 2D grid, balloon, ocean; x = L; median over episodes, bars at the quartiles
extra/fig_tuning_curves_<problem>  rows our rules; columns the scores against moves, one line per L, median and quartile band
"""

import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

from figures.common import LABELS, look, plt, read, save, size

PROBLEMS = {
    "2D grid": ("ambient_waypoint", "paper_tuning", ("emi", "evi", "esi"), (10, 25, 50, 100)),
    "balloon": ("balloon_waypoint", "paper_tuning", ("emi", "evi", "esi"), (25, 50, 100, 200)),
    "ocean": ("ocean_waypoint", "tuning_jun10", ("etr",), (25, 50, 100, 200)),
}  # name: (experiment, sweep, rules, budgets L)
METRICS = {
    "2D grid": {"average_regret": "average regret", "simple_regret": "simple regret", "rmse": "RMSE"},
    "balloon": {"average_regret": "average regret", "simple_regret": "simple regret", "rmse": "RMSE"},
    "ocean": {"steps_to_completion": "moves to visit all goals", "targets_visited": "goals visited"},
}  # name: {final scalar: axis label}, every one lower is better but goals visited
LABELS["etr"] = "ETR"
SHADES = ("#7fcdbb", "#41b6c4", "#2c7fb8", "#253494")  # one per L, light to dark
TABLES = {name: read(experiment, {"tuning": sweep}) for name, (experiment, sweep, _, _) in PROBLEMS.items()}
for curves, scalars in TABLES.values():  # seconds per move: the episode's wall clock over its moves
    moves = curves.groupby(["acquisition", "seed"]).size().rename("moves")
    scalars["seconds_per_move"] = scalars["seconds"] / scalars.join(moves, on=["acquisition", "seed"])["moves"]


def final(ax, name: str, metric: str) -> None:
    """Final `metric` against L, one line per rule: median, bars at the quartiles"""
    _, _, rules, budgets = PROBLEMS[name]
    scalars = TABLES[name][1]
    for shift, rule in enumerate(rules):
        held = [scalars.loc[scalars["acquisition"] == f"{rule}_L{budget}", metric].dropna() for budget in budgets]
        low, mid, high = (np.array([np.quantile(v, q) if len(v) else np.nan for v in held]) for q in (0.25, 0.5, 0.75))
        at = np.array(budgets) * 1.06 ** (shift - (len(rules) - 1) / 2)  # rules side by side at each L
        style = {k: v for k, v in look(rule).items() if k != "markevery"} if rule != "etr" else dict(color="#7F3C8D", marker="D", ms=3.5)
        ax.errorbar(at, mid, yerr=[mid - low, high - mid], capsize=1.5, elinewidth=0.7, label=LABELS[rule], **style)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets, [str(b) for b in budgets])
    ax.minorticks_off()


rows = len(PROBLEMS)
cols = 1 + max(len(m) for m in METRICS.values())
fig, axes = plt.subplots(rows, cols, figsize=size(rows, 1.25), constrained_layout=True)
for row, name in enumerate(PROBLEMS):
    panels = [*METRICS[name].items(), ("seconds_per_move", "seconds per move")]
    for col in range(cols):
        ax = axes[row, col]
        if col >= len(panels):
            ax.set_visible(False)
            continue
        metric, label = panels[col]
        final(ax, name, metric)
        ax.set_title(f"{name}: {label}", fontsize=8)
        ax.tick_params(labelsize=7)
        if metric == "seconds_per_move":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
for row in range(rows):
    for col in range(cols):
        if axes[row, col].get_visible() and (row == rows - 1 or not axes[row + 1, col].get_visible()):
            axes[row, col].set_xlabel("planning budget $L$", fontsize=8)
handles = {}
for ax in axes[:, 0]:
    handles.update(zip(*reversed(ax.get_legend_handles_labels()), strict=True))
fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=len(handles), fontsize=8, handlelength=2.2)
save(fig, "fig_sensitivity", "appendix")

for name, (_, _, rules, budgets) in PROBLEMS.items():
    curves = TABLES[name][0]
    metrics = {m: label for m, label in METRICS[name].items() if m in curves}
    fig, axes = plt.subplots(len(rules), len(metrics), figsize=size(len(rules), 1.0), constrained_layout=True, squeeze=False)
    for row, rule in enumerate(rules):
        for col, (metric, label) in enumerate(metrics.items()):
            ax = axes[row, col]
            for budget, shade in zip(budgets, SHADES, strict=True):
                held = curves[curves["acquisition"] == f"{rule}_L{budget}"]
                if held.empty:
                    continue
                by_x = held.groupby("x")[metric].quantile([0.25, 0.5, 0.75]).unstack()
                ax.plot(by_x.index, by_x[0.5], color=shade, lw=1.1, label=f"$L={budget}$")
                ax.fill_between(by_x.index, by_x[0.25], by_x[0.75], color=shade, alpha=0.15, lw=0)
            ax.set_title(f"{LABELS[rule]}: {label}", fontsize=8)
            ax.tick_params(labelsize=7)
            if row == len(rules) - 1:
                ax.set_xlabel("move" if name == "2D grid" else "hours elapsed", fontsize=8)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=len(budgets), fontsize=7)
    save(fig, "fig_tuning_curves_" + name.lower().replace(" ", "_").replace("2d_", ""), "extra")
