"""The ocean visiting study: the path-aware rule against its endpoint-only twin and two random rules, scored on the goals visited

Goals visited against hours elapsed, the median over episodes with the quartile band; each rule's oracle (the true field in place of the
belief) dotted in its colour. Below, a table of steps to completion and goals visited at the horizon over every arm the launch holds.

Run from code/:  pixi run fig ocean.fig_visit [sweep]     (LAUNCH_AT=<timestamp> reads an older launch)
"""

import sys

from figures.common import BUILD, LABELS, OCEAN_FRAMES, band, look, plt, read, save, size

SWEEP = sys.argv[1] if len(sys.argv) > 1 else "visit"
RULES = (
    "etr_l0",
    "etr_l05",
    "etr_l1",
    "etr_endpoint",
    "nearest_l0",
    "nearest_l05",
    "nearest_l1",
    "nearest",
    "rand_goal",
    "rand_box",
    "rand_wp",
)
ORACLES = {rule: f"oracle_{rule}" for rule in RULES if rule.startswith(("etr_l", "nearest_l"))}
TABLE = (("steps_to_completion", "steps to completion", 0, "min"), ("targets_visited", "goals visited", 2, "max"))

curves, scalars = read("ocean_waypoint", {date: f"{SWEEP}_{date}" for date in OCEAN_FRAMES})
held = set(scalars["acquisition"])
rules = [rule for rule in RULES if rule in held]
oracles = {rule: ORACLES[rule] for rule in rules if ORACLES.get(rule) in held}
lone_oracle = "oracle_etr" if "oracle_etr" in held else None  # a launch from before every rule had its own oracle
one_rate = not held & {"etr_l05", "etr_l1", "nearest_l05", "nearest_l1"}


def name(arm: str) -> str:
    """The arm's label, its lambda dropped when the launch holds only one"""
    return LABELS[arm].replace(r" $\lambda$=0", "") if one_rate else LABELS[arm]


fig, ax = plt.subplots(figsize=size(1, 0.55), constrained_layout=True)
ax.tick_params(labelsize=7, pad=2)
for rule in rules:
    band(ax, curves, rule, "targets_visited")
for oracle in oracles.values():
    band(ax, curves, oracle, "targets_visited", quartiles=False)
if lone_oracle:
    band(ax, curves, lone_oracle, "targets_visited", quartiles=False)
ax.set_ylabel("goals visited", fontsize=8, labelpad=1)
ax.set_xlabel("hours elapsed", fontsize=8, labelpad=1)
ax.margins(x=0, y=0.04)
ax.set_ylim(bottom=0)
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles,
    [name(arm) for arm in [*rules, *oracles.values()]] if not lone_oracle else labels,
    loc="outside lower center",
    ncol=3,
    fontsize=6.5,
    columnspacing=1.0,
    handlelength=2.2,
)
means = scalars.groupby("acquisition")[[m for m, *_ in TABLE]].agg(["mean", "sem"])
save(fig, f"fig_ocean_{SWEEP}", "main")

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
        lines.append(name(arm) + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(BUILD / "main").mkdir(parents=True, exist_ok=True)
(BUILD / "main" / f"tab_ocean_{SWEEP}.tex").write_text("\n".join(lines) + "\n")
print(BUILD / "main" / f"tab_ocean_{SWEEP}.tex")
