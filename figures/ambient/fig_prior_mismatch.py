"""Five rules on the 2D grid when the belief is wrong about the field's lengthscale, amplitude or kernel

Columns: the belief. Rows: average regret and RMSE against the time_step, first with the belief held fixed, then with the kernel's
numbers refit as it goes. Median and quartile band over seeds; a metric shares its axis.
The truth is a Matern-5/2 field of lengthscale 2.5 and amplitude 3.
"""

from figures.common import band, plt, read, save, size

BELIEFS = (
    ("matched", "matched"), ("lengthscale1p25", r"$\ell$ / 2"), ("lengthscale5", r"$2\ell$"),
    ("amplitude1p5", r"$\sigma$ / 2"), ("amplitude6", r"$2\sigma$"), ("matern32", "Matern-3/2"),
)  # fmt: skip
REFIT = ("lengthscale1p25", "lengthscale5", "amplitude1p5", "amplitude6")
RULES = ("emi", "evi", "mc_ei", "ts", "rand_target")
ROWS = (
    ("average_regret", "average regret", ""),
    ("rmse", "RMSE", ""),
    ("average_regret", "average regret,\nkernel refit", "_refit"),
    ("rmse", "RMSE,\nkernel refit", "_refit"),
)

curves, _ = read("ambient_waypoint", {"prior": "paper_prior_mismatch"})
curves = curves[curves["x"] >= 0.04 * curves["x"].max()]  # the first moves, where an average over a few steps is noise
fig, axes = plt.subplots(4, 6, figsize=size(4), sharex=True, constrained_layout=True)
for nth, (row, (metric, words, suffix)) in enumerate(zip(axes, ROWS)):
    for ax, (belief, name) in zip(row, BELIEFS):
        if suffix and belief not in REFIT:
            ax.axis("off")
            if belief == "matched":
                ax.text(0.5, 0.5, words, transform=ax.transAxes, ha="center", va="center", fontsize=8)
            continue
        for rule in RULES:
            band(
                ax,
                curves.assign(acquisition=curves["acquisition"].str.removesuffix(f"_{belief}{suffix}")),
                rule,
                metric,
                markevery=0.25,
                ms=3,
            )
        ax.sharey(axes[nth % 2, 0])
        ax.margins(x=0)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        ax.tick_params(labelsize=7, pad=2, labelleft=belief == ("lengthscale1p25" if suffix else "matched"), labelbottom=bool(nth % 2))
        if nth == 0:
            ax.set_title(name, fontsize=9)
        if nth % 2:
            ax.set_xlabel("time step", fontsize=8, labelpad=1)
    if not suffix:
        row[0].set_ylabel(words, fontsize=8)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8, columnspacing=1.0, handlelength=2.2)
save(fig, "fig_prior_mismatch", "appendix")
