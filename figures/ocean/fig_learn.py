"""The ocean learning study: the path-aware rules against the point rules on the Gulf Stream frame, scored on the ocean current learned over the box

Two rows of three panels against hours elapsed, the ocean current above and the wind below: posterior variance over the box, RMSE of the
claimed mean over the box, information gained by the belief over the box in nats; each the median over episodes with the quartile band.
Below, a table: mean +- standard error of the three at the horizon; the best mean of a competed column bold.

Run from code/:  pixi run fig ocean.fig_learn [sweep]
"""

import sys

from figures.common import BUILD, LABELS, OCEAN_FRAMES, band, plt, read, save, size

SWEEP = sys.argv[1] if len(sys.argv) > 1 else "learn"
ARMS = ("evi", "max_var", "rand_wp", "rand_act")
NO_CLAIM = ("rand_act",)  # no rule, so its claim is the prior's
PANELS = (
    ("posterior_variance", r"current variance (m$^2$/s$^2$)"), ("rmse", "current RMSE (m/s)"), ("information", "current information (nats)"),
    ("wind_posterior_variance", r"wind variance (m$^2$/s$^2$)"), ("wind_rmse", "wind RMSE (m/s)"), ("wind_information", "wind information (nats)"),
)  # fmt: skip
TABLE = (
    ("posterior_variance", "current variance", 3, "min"), ("rmse", "current RMSE", 3, "min"), ("information", "current nats", 0, "max"),
    ("wind_posterior_variance", "wind variance", 2, "min"), ("wind_rmse", "wind RMSE", 2, "min"), ("wind_information", "wind nats", 0, "max"),
)  # fmt: skip

curves, scalars = read("ocean_waypoint", {date: f"{SWEEP}_{date}" for date in OCEAN_FRAMES})
fig, axes = plt.subplots(2, 3, figsize=size(2), constrained_layout=True)
axes = axes.ravel()
for ax, (metric, title) in zip(axes, PANELS, strict=True):
    ax.tick_params(labelsize=7, pad=2)
    for arm in ARMS:
        if arm not in NO_CLAIM:
            band(ax, curves, arm, metric)
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_xlabel("hours elapsed", fontsize=8, labelpad=1)
    ax.margins(x=0, y=0.04)  # variance and RMSE sit on their own range; information starts at zero
    if metric.endswith("information"):
        ax.set_ylim(bottom=0)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=len(ARMS), fontsize=8, columnspacing=1.0, handlelength=2.2)
save(fig, "fig_ocean_learn", "main")

means = scalars.groupby("acquisition")[[m for m, *_ in TABLE]].mean().reindex(list(ARMS))
best = {m: getattr(means.loc[[a for a in ARMS if a not in NO_CLAIM], m], f"idx{end}")() for m, _, _, end in TABLE}
lines = [
    r"\begin{tabular}{l" + "c" * len(TABLE) + "}",
    r"\toprule",
    "rule & " + " & ".join(w for _, w, _, _ in TABLE) + r" \\",
    r"\midrule",
]
for arm in ARMS:
    if arm == "max_var":
        lines.append(r"\midrule")
    held = scalars[scalars["acquisition"] == arm]
    cells = [
        "--"
        if held[m].isna().all()
        else (r"$\mathbf{%s} \pm %s$" if best[m] == arm else r"$%s \pm %s$") % (f"{held[m].mean():.{d}f}", f"{held[m].sem():.{d}f}")
        for m, _, d, _ in TABLE
    ]
    lines.append(LABELS[arm] + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(BUILD / "main").mkdir(parents=True, exist_ok=True)
(BUILD / "main" / "tab_ocean_learn.tex").write_text("\n".join(lines) + "\n")
print(BUILD / "main" / "tab_ocean_learn.tex")
