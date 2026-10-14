"""The ocean study: the information rule at each lambda against the baselines on the Gulf Stream frame

Three panels: distance to the target against hours elapsed on a log axis floored at the arrival radius, held there once arrived;
steps to arrival per arm, mean +- s.e.m. over episodes; information gained by the
belief over the box, nats, against hours elapsed, the oracle left out. Distance is the median over episodes with the quartile band; information the same, the band only for the arms that differ.
Below, a table: mean +- standard error of steps to arrival (unarrived episodes at the horizon), arrived %, and information gained
by the horizon, and steps beyond the oracle's on the same episode; the best mean of a competed column bold.

Run from code/:  pixi run fig fig_ocean [sweep]
"""

import sys

import numpy as np

from figures.common import BUILD, LABELS, band, look, plt, read, save, size

SWEEP = sys.argv[1] if len(sys.argv) > 1 else "core"
ARMS = ("info_0.25", "info_0.5", "info_1", "info_2", "direct", "rand_wp", "rand_act", "oracle")
RULES = ("info_0.25", "info_0.5", "info_1", "info_2", "direct", "oracle")  # the arms that reach
NO_CLAIM = ("rand_act", "oracle")  # no rule, so its claim is the prior's; and the oracle, whose claim is exact
ARRIVAL_KM = 20.0
TABLE = (
    ("steps_to_arrival", "steps to arrival", 1, "min"),
    ("excess", "beyond the oracle", 1, "min"),
    ("arrived", r"arrived \%", 0, "max"),
    ("information", "information (nats)", 1, "max"),
)

curves, scalars = read("ocean_waypoint", {"gulf": SWEEP})
# the trip is over once arrived, so the distance is held at the arrival radius from there
curves["distance_to_target"] = np.maximum(curves["distance_to_target"].where(curves["arrived"] == 0, 0.0), ARRIVAL_KM)
fig, axes = plt.subplots(1, 3, figsize=size(1), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.2, 1.0, 1.0]))
for ax in axes:
    ax.tick_params(labelsize=7, pad=2)

# A: distance, the median over episodes, cut where the last rule arm's last episode arrives
distance, information = axes[0], axes[2]
for arm in ARMS:
    band(distance, curves, arm, "distance_to_target", lw=1.0 if arm in RULES else 0.8)
medians = curves[curves["acquisition"].isin(RULES)].groupby(["acquisition", "x"])["distance_to_target"].median()
settled = medians[medians <= ARRIVAL_KM].reset_index().groupby("acquisition")["x"].min().max()  # when the slowest median arrives
distance.set_yscale("log")
distance.set_xlim(0, 1.15 * settled)
distance.set_yticks([ARRIVAL_KM, 100, 1000])
distance.set_yticklabels(["20", "100", "1000"])
distance.yaxis.set_minor_formatter(plt.NullFormatter())
distance.axhline(ARRIVAL_KM, color="#999999", lw=0.5, ls=":")
distance.set_title("distance to target (km)", fontsize=8, pad=2)
distance.set_xlabel("hours elapsed", fontsize=8, labelpad=1)
distance.margins(x=0)

# B: steps to arrival, mean +- s.e.m. over episodes, the oracle a row like the rest
steps = axes[1]
oracle_steps = scalars.loc[scalars["acquisition"] == "oracle"].set_index("episode")["steps_to_arrival"]
drawn = list(RULES)
for row, arm in enumerate(drawn):
    held = scalars.loc[scalars["acquisition"] == arm, "steps_to_arrival"]
    style = look(arm)
    steps.errorbar(
        held.mean(),
        row,
        xerr=held.sem(),
        fmt=style.get("marker") or "o",
        ms=5.5,
        color=style["color"],
        mec="black",
        mew=0.4,
        ecolor=style["color"],
        elinewidth=1.2,
        capsize=2.5,
        zorder=4,
    )
steps.set_yticks(range(len(drawn)))
steps.set_yticklabels([LABELS[arm] for arm in drawn], fontsize=7)
steps.set_ylim(len(drawn) - 0.5, -0.5)
steps.set_title("steps to arrival", fontsize=8, pad=2)
steps.set_xlabel("time steps of 3 h", fontsize=8, labelpad=1)
steps.set_xlim(left=0)

# C: information gained
for arm in ARMS:
    if arm in NO_CLAIM:
        continue
    band(information, curves, arm, "information", quartiles=arm in ("info_2", "direct"))
information.set_title("information gained (nats)", fontsize=8, pad=2)
information.set_xlabel("hours elapsed", fontsize=8, labelpad=1)
information.margins(x=0)
information.set_ylim(bottom=0)
handles, labels = distance.get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=8, fontsize=8, columnspacing=1.0, handlelength=2.2)
save(fig, "fig_ocean", "main")

scalars = scalars.assign(arrived=100.0 * scalars["arrived"], excess=scalars["steps_to_arrival"] - scalars["episode"].map(oracle_steps))
means = scalars.groupby("acquisition")[[m for m, *_ in TABLE]].mean().reindex(list(ARMS))
best = {m: getattr(means.loc[[a for a in ARMS if a != "oracle"], m], f"idx{end}")() for m, _, _, end in TABLE}
lines = [
    r"\begin{tabular}{l" + "c" * len(TABLE) + "}",
    r"\toprule",
    "rule & " + " & ".join(w for _, w, _, _ in TABLE) + r" \\",
    r"\midrule",
]
for arm in ARMS:
    if arm in ("rand_wp", "oracle"):
        lines.append(r"\midrule")
    held = scalars[scalars["acquisition"] == arm]
    cells = [
        "--"
        if held[m].isna().all() or (m == "information" and arm in NO_CLAIM)
        else (r"$\mathbf{%s} \pm %s$" if best[m] == arm else r"$%s \pm %s$") % (f"{held[m].mean():.{d}f}", f"{held[m].sem():.{d}f}")
        for m, _, d, _ in TABLE
    ]
    lines.append(LABELS[arm] + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(BUILD / "main").mkdir(parents=True, exist_ok=True)
(BUILD / "main" / "tab_ocean.tex").write_text("\n".join(lines) + "\n")
print(BUILD / "main" / "tab_ocean.tex")
