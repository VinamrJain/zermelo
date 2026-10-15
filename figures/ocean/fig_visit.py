"""The ocean visiting study: the path-aware rule at three travel charges against the nearest-unvisited rule, scored on the moves until every target is visited

Left, targets visited against hours elapsed, the median over episodes with the quartile band. Right, steps to completion per arm,
mean +- standard error, the oracle as a dashed line. Below, a table of both at the horizon; the best mean of a competed column bold.

Run from code/:  pixi run fig ocean.fig_visit [sweep]
"""

import sys

from figures.common import BUILD, LABELS, OCEAN_FRAMES, band, look, plt, read, save, size

SWEEP = sys.argv[1] if len(sys.argv) > 1 else "visit"
ARMS = ("etr_l0", "etr_l05", "etr_l1", "nearest", "rand_wp", "oracle_etr")
COMPETED = ARMS[:-1]
TABLE = (("steps_to_completion", "steps to completion", 0, "min"), ("targets_visited", "targets visited", 2, "max"))

curves, scalars = read("ocean_waypoint", {date: f"{SWEEP}_{date}" for date in OCEAN_FRAMES})
fig, (left, right) = plt.subplots(1, 2, figsize=size(1), constrained_layout=True, gridspec_kw={"width_ratios": [1.6, 1.0]})
left.tick_params(labelsize=7, pad=2)
for arm in ARMS:
    band(left, curves, arm, "targets_visited")
left.set_title("targets visited", fontsize=8, pad=2)
left.set_xlabel("hours elapsed", fontsize=8, labelpad=1)
left.margins(x=0, y=0.04)
left.set_ylim(bottom=0)
means = scalars.groupby("acquisition")[[m for m, *_ in TABLE]].agg(["mean", "sem"])
right.tick_params(labelsize=7, pad=2)
for at, arm in enumerate(COMPETED):
    right.bar(
        at,
        means.loc[arm, ("steps_to_completion", "mean")],
        yerr=means.loc[arm, ("steps_to_completion", "sem")],
        color=look(arm)["color"],
        width=0.7,
        capsize=2,
    )
right.axhline(means.loc["oracle_etr", ("steps_to_completion", "mean")], **look("oracle_etr"))
right.set_xticks(range(len(COMPETED)), [LABELS[a] for a in COMPETED], fontsize=6, rotation=20)
right.set_title("steps to completion", fontsize=8, pad=2)
handles, labels = left.get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=len(ARMS), fontsize=7, columnspacing=1.0, handlelength=2.2)
save(fig, "fig_ocean_visit", "main")

best = {m: getattr(means.loc[list(COMPETED), (m, "mean")], f"idx{end}")() for m, _, _, end in TABLE}
lines = [
    r"\begin{tabular}{l" + "c" * len(TABLE) + "}",
    r"\toprule",
    "rule & " + " & ".join(w for _, w, _, _ in TABLE) + r" \\",
    r"\midrule",
]
for arm in ARMS:
    if arm in ("nearest", "oracle_etr"):
        lines.append(r"\midrule")
    cells = [
        (r"$\mathbf{%s} \pm %s$" if best[m] == arm else r"$%s \pm %s$")
        % (f"{means.loc[arm, (m, 'mean')]:.{d}f}", f"{means.loc[arm, (m, 'sem')]:.{d}f}")
        for m, _, d, _ in TABLE
    ]
    lines.append(LABELS[arm] + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(BUILD / "main").mkdir(parents=True, exist_ok=True)
(BUILD / "main" / "tab_ocean_visit.tex").write_text("\n".join(lines) + "\n")
print(BUILD / "main" / "tab_ocean_visit.tex")
