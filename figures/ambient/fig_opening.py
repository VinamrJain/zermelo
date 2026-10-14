"""EMI and EVI on the 2D grid as the random opening time_steps move: final average regret and RMSE, median and quartile band"""

import numpy as np

from figures.common import LABELS, look, plt, quartiles, read, save, size

STEPS = (0, 25, 50, 100)
_, scalars = read("ambient_waypoint", {"opening": "paper_opening_steps"})
fig, axes = plt.subplots(1, 2, figsize=size(1, 0.7), constrained_layout=True)
for ax, (metric, words) in zip(axes, (("average_regret", "average regret"), ("rmse", "RMSE"))):
    for rule in ("emi", "evi"):
        mids = np.array([quartiles(scalars, f"{rule}_o{steps}", metric) for steps in STEPS])
        style = {**look(rule), "markevery": 1}
        ax.plot(STEPS, mids[:, 0], label=LABELS[rule], **style)
        ax.fill_between(STEPS, mids[:, 0] - mids[:, 1], mids[:, 0] + mids[:, 2], color=style["color"], alpha=0.12, lw=0)
    ax.set_xticks(STEPS)
    ax.set_xlabel("random opening time steps")
    ax.set_ylabel(words)
axes[0].legend(fontsize=8)
save(fig, "fig_opening", "appendix")
