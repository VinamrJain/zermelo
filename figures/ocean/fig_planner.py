"""The ocean planner study: how far the planner looks (K), how often it replans (L), and whether only reachable targets are chosen

One column per L in {25, 50, 100}, rows the task's metric over the ocean current and the wind's posterior variance, against hours
elapsed; one line per K in {L, 2L, H - h}, solid with the reachable filter on and dashed with it off, each the median over the frames
with the band between their quartiles. Below, a table: mean +- standard error over the frames at the horizon; the best mean bold.

Run from code/:  pixi run fig ocean.fig_planner -- <search|learn>
"""

import sys

from matplotlib.lines import Line2D

from figures.common import BUILD, band, plt, read, save, size

TASK = sys.argv[1]
FRAMES = ("jun16", "jun28")
RULE = {"search": "emi", "learn": "evi"}[TASK]
LENGTHS = (25, 50, 100)
ROUNDS = {"K1L": ("$K = L$", "#0173B2"), "K2L": ("$K = 2L$", "#DE8F05"), "Kend": ("$K = H - h$", "#029E73")}
FILTERS = {"on": ("reachable filter on", "-"), "off": ("reachable filter off", "--")}
PANELS = {
    "search": (("simple_regret", "simple regret\n(m/s)", 3, "min"), ("wind_posterior_variance", "wind variance\n(m$^2$/s$^2$)", 2, "min")),
    "learn": (
        ("posterior_variance", "current variance\n(m$^2$/s$^2$)", 4, "min"),
        ("wind_posterior_variance", "wind variance\n(m$^2$/s$^2$)", 2, "min"),
    ),
}[TASK]


def name(length: int, rounds: str, filtered: str) -> str:
    """The arm's name in the sweep"""
    return f"{RULE}_L{length}_{rounds}_filter_{filtered}"


curves, scalars = read("ocean_waypoint", {date: f"planner_{TASK}_{date}" for date in FRAMES})
fig, axes = plt.subplots(
    len(PANELS), len(LENGTHS), figsize=size(len(PANELS)), constrained_layout=True, sharex=True, sharey="row", squeeze=False
)
for row, (metric, title, *_) in enumerate(PANELS):
    for column, length in enumerate(LENGTHS):
        ax = axes[row, column]
        ax.tick_params(labelsize=7, pad=2)
        for rounds, (_, colour) in ROUNDS.items():
            for filtered, (_, ls) in FILTERS.items():
                band(ax, curves, name(length, rounds, filtered), metric, color=colour, ls=ls, marker=None)
        ax.set_title(f"$L = {length}$" if row == 0 else "", fontsize=8, pad=2)
        ax.set_ylabel(title if column == 0 else "", fontsize=8, labelpad=1)
        ax.set_xlabel("hours elapsed" if row == len(PANELS) - 1 else "", fontsize=8, labelpad=1)
        ax.margins(x=0, y=0.04)
        ax.set_ylim(bottom=0)
handles = [Line2D([], [], color=colour, lw=1.3, label=words) for words, colour in ROUNDS.values()]
handles += [Line2D([], [], color="black", lw=1.0, ls=ls, label=words) for words, ls in FILTERS.values()]
fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=8, columnspacing=1.0, handlelength=2.2)
save(fig, f"fig_ocean_planner_{TASK}", "extra")

arms = [(length, rounds, filtered) for length in LENGTHS for rounds in ROUNDS for filtered in FILTERS]
means = scalars.groupby("acquisition")[[m for m, *_ in PANELS]].mean()
best = {m: getattr(means[m], f"idx{end}")() if means[m].notna().any() else None for m, _, _, end in PANELS}
lines = [
    r"\begin{tabular}{rll" + "c" * len(PANELS) + "}",
    r"\toprule",
    "$L$ & $K$ & filter & " + " & ".join(words.replace("\n", " ") for _, words, _, _ in PANELS) + r" \\",
    r"\midrule",
]
for length, rounds, filtered in arms:
    held = scalars[scalars["acquisition"] == name(length, rounds, filtered)]
    cells = [
        (r"$\mathbf{%s} \pm %s$" if best[m] == name(length, rounds, filtered) else r"$%s \pm %s$")
        % (f"{held[m].mean():.{d}f}", f"{held[m].sem():.{d}f}")
        for m, _, d, _ in PANELS
    ]
    lines.append(f"{length} & {ROUNDS[rounds][0]} & {filtered} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(BUILD / "extra").mkdir(parents=True, exist_ok=True)
(BUILD / "extra" / f"tab_ocean_planner_{TASK}.tex").write_text("\n".join(lines) + "\n")
print(BUILD / "extra" / f"tab_ocean_planner_{TASK}.tex")
