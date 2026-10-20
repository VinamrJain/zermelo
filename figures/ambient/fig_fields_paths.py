"""EMI and EVI on the 2D grid as the sampled fields S and the sampled walks per field M move

Panels: EMI's final average regret, EVI's final RMSE, seconds per time_step (EMI and EVI pooled). Two curves: S at M = 16 (S = 0 scores
the mean field), M at S = 16 (M = 0 scores the target alone). Median and quartile band over seeds.
"""

import numpy as np

from figures.common import plt, quartiles, read, save, size

COUNTS = (0, 1, 4, 8, 16, 32, 64)
SWEPT = (
    ("fields $S$ ($M$ = 16)", "#0173B2", "o", lambda count: f"fields{count}_paths16"),
    ("walks per field $M$ ($S$ = 16)", "#DE8F05", "s", lambda count: f"fields16_paths{count}"),
)
PANELS = (
    ("emi", "average_regret", "(a) EMI: average regret"),
    ("evi", "rmse", "(b) EVI: RMSE"),
    (None, "seconds_per_step", "(c) seconds per time step"),
)

curves, scalars = read("ambient_waypoint", {"fields_paths": "paper_fields_paths"})
scalars = scalars.assign(seconds_per_step=scalars["seconds"] / curves["x"].nunique())
at = np.arange(len(COUNTS))
fig, axes = plt.subplots(1, 3, figsize=size(1), constrained_layout=True)
for ax, (rule, metric, title) in zip(axes, PANELS):
    for label, colour, marker, arm_of in SWEPT:
        names = [rule] if rule else ["emi", "evi"]
        mids = np.array(
            [
                quartiles(scalars.replace({"acquisition": {f"{name}_{arm_of(count)}": "held" for name in names}}), "held", metric)
                for count in COUNTS
            ]
        )
        ax.plot(at, mids[:, 0], color=colour, marker=marker, ms=3.5, label=label)
        ax.fill_between(at, mids[:, 0] - mids[:, 1], mids[:, 0] + mids[:, 2], color=colour, alpha=0.12, lw=0)
    ax.set_xticks(at, [str(count) for count in COUNTS])
    ax.set_xlabel("samples drawn")
    ax.set_title(title, fontsize=9)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
save(fig, "fig_fields_paths", "appendix")
