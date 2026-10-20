"""Our rules against the baselines on the 2D grid, EMI also planning on the true field

One row: simple regret and average regret (EMI against MC-EI, UCB, TS, RandTarget; EMI given the true field dotted, hollow
markers), RMSE and posterior spread (EVI against MaxVar, RandTarget). Median and quartile band over episodes.
"""

from matplotlib.lines import Line2D

from figures.common import LABELS, band, look, plt, read, save, size

PANELS = (
    ("simple_regret", "simple regret", ("emi", "mc_ei", "ucb", "ts", "rand_target"), ("emi",)),
    ("average_regret", "average regret", ("emi", "mc_ei", "ucb", "ts", "rand_target"), ("emi",)),
    ("rmse", "RMSE", ("evi", "max_var", "rand_target"), ()),
    ("posterior_spread", "posterior spread", ("evi", "max_var", "rand_target"), ()),
)  # (metric, words, arms on the learned field, arms on the true field)
TRUE = dict(ls=(0, (1, 1.2)), mfc="white", mew=1.0, markevery=(0.1, 0.2))

learned = read("ambient_waypoint", {"main": "paper_main"})[0]
true = learned.assign(acquisition=learned["acquisition"].str.removeprefix("oracle_"))[learned["acquisition"].str.startswith("oracle_")]
fig, axes = plt.subplots(1, 4, figsize=size(1), constrained_layout=True)
for ax, (metric, words, arms, given), letter in zip(axes, PANELS, "abcd"):
    for arm in arms:
        band(ax, learned, arm, metric)
    for arm in given:
        band(ax, true, arm, metric, quartiles=False, **TRUE)
    ax.set_title(f"({letter}) {words}", fontsize=9)
    ax.set_xlabel("time step")
    ax.margins(x=0)
    if metric != "rmse":
        ax.set_ylim(bottom=0)
arms = dict.fromkeys(arm for _, _, held, _ in PANELS for arm in held)
handles = [Line2D([], [], **{**look(arm), "markevery": None}) for arm in arms] + [
    Line2D([], [], **{**look("emi"), **TRUE, "markevery": None})
]
fig.legend(
    handles,
    [LABELS[arm] for arm in arms] + ["EMI, true field"],
    loc="outside lower center",
    ncol=8,
    fontsize=8,
    columnspacing=1.2,
    handlelength=2.2,
)
save(fig, "fig_grid_comparison", "main")
