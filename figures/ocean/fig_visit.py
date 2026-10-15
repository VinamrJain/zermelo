"""The ocean visiting study: the path-aware rule against the nearest-unvisited rule at three travel charges, scored on the moves until every target is visited

Left, targets visited against hours elapsed, the median over episodes with the quartile band, the learned-field arms only. Right, steps to
completion per rule, mean +- standard error, each rule's oracle (the true field in place of the belief) as a hatched bar beside it.
Below, a table of both at the horizon over every arm the launch holds; the best mean of a learned-field arm bold.

Run from code/:  pixi run fig ocean.fig_visit [sweep]     (LAUNCH_AT=<timestamp> reads an older launch)
"""

import sys

from figures.common import BUILD, LABELS, OCEAN_FRAMES, band, look, plt, read, save, size

SWEEP = sys.argv[1] if len(sys.argv) > 1 else "visit"
RULES = ("etr_l0", "etr_l05", "etr_l1", "nearest_l0", "nearest_l05", "nearest_l1", "nearest", "rand_wp")
ORACLES = {rule: f"oracle_{rule}" for rule in RULES[:6]}
TABLE = (("steps_to_completion", "steps to completion", 0, "min"), ("targets_visited", "targets visited", 2, "max"))

curves, scalars = read("ocean_waypoint", {date: f"{SWEEP}_{date}" for date in OCEAN_FRAMES})
held = set(scalars["acquisition"])
rules = [rule for rule in RULES if rule in held]
oracles = {rule: ORACLES[rule] for rule in rules if ORACLES.get(rule) in held}
lone_oracle = "oracle_etr" if "oracle_etr" in held else None  # a launch from before every rule had its own oracle

fig, (left, right) = plt.subplots(1, 2, figsize=size(1), constrained_layout=True, gridspec_kw={"width_ratios": [1.4, 1.2]})
left.tick_params(labelsize=7, pad=2)
for rule in rules:
    band(left, curves, rule, "targets_visited")
left.set_title("targets visited", fontsize=8, pad=2)
left.set_xlabel("hours elapsed", fontsize=8, labelpad=1)
left.margins(x=0, y=0.04)
left.set_ylim(bottom=0)
means = scalars.groupby("acquisition")[[m for m, *_ in TABLE]].agg(["mean", "sem"])
right.tick_params(labelsize=7, pad=2)
for at, rule in enumerate(rules):
    for offset, arm, hatch in ((-0.2, rule, None), (0.2, oracles.get(rule), "////")):
        if arm is None:
            continue
        right.bar(
            at + (offset if rule in oracles else 0.0),
            means.loc[arm, ("steps_to_completion", "mean")],
            yerr=means.loc[arm, ("steps_to_completion", "sem")],
            color=look(rule)["color"],
            alpha=0.45 if hatch else 1.0,
            hatch=hatch,
            width=0.4 if rule in oracles else 0.7,
            capsize=2,
        )
if lone_oracle:
    right.axhline(means.loc[lone_oracle, ("steps_to_completion", "mean")], **look(lone_oracle))
right.set_xticks(range(len(rules)), [LABELS[rule] for rule in rules], fontsize=6, rotation=30, ha="right")
right.set_title("steps to completion (hatched: oracle)" if oracles else "steps to completion", fontsize=8, pad=2)
handles, labels = left.get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=min(len(rules), 4), fontsize=7, columnspacing=1.0, handlelength=2.2)
save(fig, "fig_ocean_visit", "main")

best = {m: getattr(means.loc[rules, (m, "mean")], f"idx{end}")() for m, _, _, end in TABLE}
lines = [
    r"\begin{tabular}{l" + "c" * len(TABLE) + "}",
    r"\toprule",
    "rule & " + " & ".join(w for _, w, _, _ in TABLE) + r" \\",
    r"\midrule",
]
for group in (rules, list(oracles.values()) + ([lone_oracle] if lone_oracle else [])):
    if group is not rules and group:
        lines.append(r"\midrule")
    for arm in group:
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
