"""The comparison and the travel figures as one: our rules against the baselines on the 2D grid and on the still balloon frames

Rows: 2D grid then balloon (pooled over frames), two each: simple regret, average regret, RMSE over region covered, time in region, posterior variance.
Median over episodes with the quartile band (in the last three columns for our rules only).
"""

from figures import fig_comparison, fig_travel
from figures.common import LEGEND_HEIGHT, WIDTH, plt, save

fig, axes = plt.subplots(4, 3, figsize=(WIDTH, 4 * 1.28 + LEGEND_HEIGHT), constrained_layout=True)
fig_comparison.draw(axes[[0, 2]])
fig_travel.draw(axes[[1, 3]])
for ax in axes[[0, 2]].ravel():
    ax.set_xlabel("")
    ax.tick_params(labelbottom=False)
for ax in axes.ravel():
    ax.tick_params(labelsize=8, pad=2)
    ax.set_title(ax.get_ylabel().split("\n")[-1], fontsize=8, pad=2)
    ax.set_ylabel("")
    ax.set_xlabel(ax.get_xlabel(), fontsize=8, labelpad=1)
fig.get_layout_engine().set(rect=(0.03, 0, 0.97, 1))  # room on the left for the problems' names
handles = {label: handle for ax in axes[[0, 2]].ravel() for handle, label in zip(*ax.get_legend_handles_labels())}
fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=8, fontsize=8, columnspacing=1.0, handlelength=2.2)
fig.canvas.draw()
for row, name in ((0, "2D grid"), (2, "balloon")):  # one name per problem, centred on its two rows
    fig.text(0.012, (axes[row, 0].get_position().y1 + axes[row + 1, 0].get_position().y0) / 2, name, rotation=90, va="center", ha="center")
save(fig, "fig_main", "main")
