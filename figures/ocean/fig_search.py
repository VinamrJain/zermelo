"""The ocean search study: the path-aware rule against the point rules on the Gulf Stream frame, scored on the fastest current stood on in the box

Two panels against hours elapsed: simple regret g - b_t and cumulative regret, g the fastest current in the box and b_t the fastest
stood on by then, in m/s; each the median over episodes with the quartile band.
Below, a table: mean +- standard error of both at the horizon; the best mean bold.

Run from code/:  pixi run fig ocean.fig_search [sweep]
"""

import sys

from figures.common import BUILD, LABELS, OCEAN_FRAMES, band, plt, read, save, size

SWEEP = sys.argv[1] if len(sys.argv) > 1 else "search"
ARMS = ("emi", "mc_ei", "ucb", "ts", "rand_wp", "rand_act")
PANELS = (("simple_regret", "simple regret (m/s)"), ("cumulative_regret", "cumulative regret (m/s)"))
TABLE = (("simple_regret", "simple regret (m/s)", 3, "min"), ("cumulative_regret", "cumulative regret (m/s)", 1, "min"))

curves, scalars = read("ocean_waypoint", {date: f"{SWEEP}_{date}" for date in OCEAN_FRAMES})
fig, axes = plt.subplots(1, 2, figsize=size(1), constrained_layout=True)
for ax, (metric, title) in zip(axes, PANELS, strict=True):
    ax.tick_params(labelsize=7, pad=2)
    for arm in ARMS:
        band(ax, curves, arm, metric)
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_xlabel("hours elapsed", fontsize=8, labelpad=1)
    ax.margins(x=0, y=0.04)
    ax.set_ylim(bottom=0)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=len(ARMS), fontsize=8, columnspacing=1.0, handlelength=2.2)
save(fig, "fig_ocean_search", "main")

means = scalars.groupby("acquisition")[[m for m, *_ in TABLE]].mean().reindex(list(ARMS))
best = {m: getattr(means[m], f"idx{end}")() for m, _, _, end in TABLE}
lines = [
    r"\begin{tabular}{l" + "c" * len(TABLE) + "}",
    r"\toprule",
    "rule & " + " & ".join(w for _, w, _, _ in TABLE) + r" \\",
    r"\midrule",
]
for arm in ARMS:
    if arm == "mc_ei":
        lines.append(r"\midrule")
    held = scalars[scalars["acquisition"] == arm]
    cells = [
        (r"$\mathbf{%s} \pm %s$" if best[m] == arm else r"$%s \pm %s$") % (f"{held[m].mean():.{d}f}", f"{held[m].sem():.{d}f}")
        for m, _, d, _ in TABLE
    ]
    lines.append(LABELS[arm] + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(BUILD / "main").mkdir(parents=True, exist_ok=True)
(BUILD / "main" / "tab_ocean_search.tex").write_text("\n".join(lines) + "\n")
print(BUILD / "main" / "tab_ocean_search.tex")
