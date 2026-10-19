"""B1: our rules against the baselines on the still balloon frames, three rules also planning on the true wind

Columns: simple regret, average regret, RMSE against hours elapsed. Median over frames x seeds; quartile band for our rules.
Planning on the true wind: the rule's colour, dotted, hollow markers (regret panels only, the true wind has nothing to learn).
"""

from figures.common import FRAMES, LABELS, band, look, plt, read, save, size

PANELS = (("simple_regret", "simple regret"), ("average_regret", "average regret"), ("rmse", "RMSE"))
ARMS = ("emi", "evi", "mc_ei", "ucb", "ts", "max_var", "rand_target", "rand_act")
ORACLES = ("emi", "mc_ei", "rand_target")  # arm `oracle_<rule>`

curves, _ = read("balloon_waypoint", {date: f"paper_main_{date}" for date in FRAMES})
fig, axes = plt.subplots(1, 3, figsize=size(1), constrained_layout=True)
for ax, (metric, words) in zip(axes, PANELS, strict=True):
    for arm in ARMS:
        if not (metric == "rmse" and arm == "rand_act"):  # it claims the prior, off the panel's scale
            band(ax, curves, arm, metric, quartiles=arm in ("emi", "evi"))
    if metric != "rmse":
        for rule in ORACLES:
            held = curves[curves["acquisition"] == f"oracle_{rule}"]
            mid = held.groupby("x")[metric].median()
            ax.plot(mid.index, mid, **{**look(rule), "ls": ":", "mfc": "white"}, label=f"{LABELS[rule]}, true wind")
    ax.set_title(words, fontsize=9)
    ax.set_xlabel("hours elapsed")
    ax.margins(x=0)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=6, fontsize=7, columnspacing=1.0, handlelength=2.4)
save(fig, "fig_balloon_main", "main")
