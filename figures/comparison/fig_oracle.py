"""Three rules planning on their learned belief against the same rules planning on the true field

One row: simple and average regret on the 2D grid, then on the balloon (pooled over frames). Median over episodes.
A solid line with filled markers learns the field; a dotted line of the same colour with hollow markers is given it.
The quartile band is drawn for EMI on its learned belief only.
"""

from matplotlib.lines import Line2D

from figures.common import FRAMES, LABELS, plt, read, save, size

LOOKS = {"emi": ("#0072B2", "o"), "mc_ei": ("#E69F00", "D"), "rand_target": ("#3B3B3B", "^")}  # colour and marker, this figure's own
ROWS = (
    ("ambient_waypoint", "2D grid", "time step", {"main": "paper_main"}, {"oracle": "paper_oracle"}),
    ("balloon_waypoint", "balloon", "hours elapsed", {date: f"paper_main_{date}" for date in FRAMES}, {date: f"paper_oracle_{date}" for date in FRAMES}),
)  # fmt: skip
LEARNED = dict(ls="-", lw=1.6)
TRUE = dict(ls=(0, (1, 1.2)), lw=1.6, mfc="white", mew=1.1)

fig, axes = plt.subplots(1, 4, figsize=size(1), constrained_layout=True)
for row, (experiment, name, x_label, learned, true) in enumerate(ROWS):
    held = {"learned": read(experiment, learned)[0], "true": read(experiment, true)[0]}
    for col, (metric, words) in enumerate((("simple_regret", "simple regret"), ("average_regret", "average regret"))):
        ax = axes[2 * row + col]
        for arm, (colour, marker) in reversed(LOOKS.items()):
            for which, style in (("learned", LEARNED), ("true", TRUE)):
                curves = held[which]
                by_x = curves[curves["acquisition"] == arm].groupby("x")[metric].quantile([0.25, 0.5, 0.75]).unstack()
                offset = 0.0 if which == "learned" else 0.08  # hollow markers sit between the filled ones
                ax.plot(by_x.index, by_x[0.5], color=colour, marker=marker, ms=3.8, markevery=(0.08 + offset, 0.22), zorder=3, **style)
                if arm == "emi" and which == "learned":
                    ax.fill_between(by_x.index, by_x[0.25], by_x[0.75], color=colour, alpha=0.13, lw=0, zorder=1)
        ax.set_ylabel(words)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(x_label)
        ax.margins(x=0)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        low, high = ax.get_ylim()
        if low < 0.5 * high:  # a regret that nears zero is drawn from zero
            ax.set_ylim(bottom=0)
handles = [Line2D([], [], color=colour, marker=marker, ms=4.5, lw=1.6) for colour, marker in LOOKS.values()] + [
    Line2D([], [], color="0.35", marker="s", ms=4.5, **LEARNED),
    Line2D([], [], color="0.35", marker="s", ms=4.5, **TRUE),
]
fig.legend(
    handles,
    [LABELS[arm] for arm in LOOKS] + ["learned field", "true field"],
    loc="outside lower center",
    ncol=5,
    fontsize=8,
    columnspacing=1.6,
    handlelength=3.0,
)
save(fig, "fig_oracle", "main")
