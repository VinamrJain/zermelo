"""EMI and EVI on the 2D grid as the sampled fields and the sampled paths per field move

Rows: against the fields at 16 paths (0 fields scores the mean field), against the paths at 16 fields.
Columns: final average regret, final RMSE, seconds per time_step. Median and quartile band over seeds.
"""

import numpy as np

from figures.common import LABELS, look, plt, quartiles, read, save, size

SWEPT = (
    ("sampled fields", ("mean", 1, 4, 16, 32), lambda count: f"fields{0 if count == 'mean' else count}_paths16"),
    ("sampled paths per field", (1, 4, 16, 32), lambda count: f"fields16_paths{count}"),
)
PANELS = (("average_regret", "average regret"), ("rmse", "RMSE"), ("seconds_per_step", "seconds per time step"))

curves, scalars = read("ambient_waypoint", {"fields_paths": "paper_fields_paths"})
scalars = scalars.assign(seconds_per_step=scalars["seconds"] / curves["x"].nunique())
fig, axes = plt.subplots(2, 3, figsize=size(2), constrained_layout=True)
for row, (x_label, counts, arm_of) in zip(axes, SWEPT):
    at = np.arange(len(counts))
    for ax, (metric, words) in zip(row, PANELS):
        for rule in ("emi", "evi"):
            mids = np.array([quartiles(scalars, f"{rule}_{arm_of(count)}", metric) for count in counts])
            style = {**look(rule), "markevery": 1}
            ax.plot(at, mids[:, 0], label=LABELS[rule], **style)
            ax.fill_between(at, mids[:, 0] - mids[:, 1], mids[:, 0] + mids[:, 2], color=style["color"], alpha=0.12, lw=0)
        ax.set_xticks(at, [str(count) for count in counts])
        ax.set_xlabel(x_label)
        ax.set_title(words, fontsize=9)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
save(fig, "fig_fields_paths", "appendix")
