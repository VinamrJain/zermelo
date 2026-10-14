"""F2 and T1: our rules against the baselines on the 2D grid and on the still balloon frames

Rows: 2D grid, balloon (pooled over frames). Columns: simple regret, average regret, RMSE. Median and quartile band over episodes.
T1: mean +- standard error of the final simple regret, cumulative regret and RMSE, and seconds per time_step; the lowest mean of a column is bold.
"""

from figures.common import BUILD, FRAMES, LABELS, band, plt, read, save, size

PANELS = (("simple_regret", "simple regret"), ("average_regret", "average regret"), ("rmse", "RMSE"))
ROWS = (
    ("ambient_waypoint", "2D grid", "time step", {"main": "paper_main"},
     ("emi", "evi", "mc_ei", "ucb", "ts", "max_var", "rand_target", "rand_act")),
    ("balloon_waypoint", "balloon", "hours elapsed", {date: f"paper_main_{date}" for date in FRAMES},
     ("emi", "evi", "mc_ei", "ucb", "ts", "max_var", "rand_target", "rand_act")),
)  # fmt: skip
NO_CLAIM = ("rand_act",)  # no rule, so its RMSE is the prior's
TABLE = (
    ("simple_regret", "simple regret", 2),
    ("cumulative_regret", "cum. regret", 0),
    ("rmse", "RMSE", 2),
    ("seconds_per_step", "s / step", 2),
)
UNBOLDED = ("seconds_per_step",)  # reported, not competed on


def draw(axes) -> dict[str, dict[str, list[str]]]:
    """The six panels on a 2 x 3 block of axes; gives the table's cells, by arm then problem"""
    cells: dict[str, dict[str, list[str]]] = {}
    for row, (experiment, name, x_label, sweeps, arms) in enumerate(ROWS):
        curves, scalars = read(experiment, sweeps)
        for col, (metric, words) in enumerate(PANELS):
            ax = axes[row, col]
            for arm in arms:
                if metric == "rmse" and arm in NO_CLAIM:
                    continue
                band(ax, curves, arm, metric)
            ax.set_ylabel(f"{words}" if col else f"{name}\n{words}")
            ax.set_xlabel(x_label)
            ax.margins(x=0)
            low, high = ax.get_ylim()
            if metric != "rmse" and low < 0.5 * high:  # a regret that nears zero is drawn from zero
                ax.set_ylim(bottom=0)
        scalars = scalars.assign(seconds_per_step=scalars["seconds"] / curves["x"].nunique())
        best = {
            metric: scalars[~((metric == "rmse") & scalars["acquisition"].isin(NO_CLAIM))]
            .groupby("acquisition")[metric]
            .mean()
            .reindex(list(arms))
            .idxmin()
            for metric, _, _ in TABLE
        }
        for arm in arms:
            held = scalars[scalars["acquisition"] == arm]
            cells.setdefault(arm, {})[name] = [
                "--"
                if held[metric].isna().all() or (metric == "rmse" and arm in NO_CLAIM)
                else (r"$\mathbf{%s} \pm %s$" if best[metric] == arm and metric not in UNBOLDED else r"$%s \pm %s$")
                % (f"{held[metric].mean():.{digits}f}", f"{held[metric].sem():.{digits}f}")
                for metric, _, digits in TABLE
            ]
    return cells


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 3, figsize=size(2), constrained_layout=True)
    cells = draw(axes)
    handles = {label: handle for ax in axes.ravel() for handle, label in zip(*ax.get_legend_handles_labels())}
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=8, fontsize=8, columnspacing=1.0, handlelength=2.2)
    save(fig, "fig_comparison", "main")

    names = [name for _, name, *_ in ROWS]
    width = len(TABLE)
    lines = [
        r"\begin{tabular}{l" + (r"@{\hskip 1.2em}" + "r" * width) * len(names) + "}", r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{{width}}}{{c}}{{\textbf{{{name}}}}}" for name in names) + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + width * at}-{1 + width * (at + 1)}}}" for at in range(len(names))),
        "acquisition & " + " & ".join(words + ("" if metric in UNBOLDED else r" $\downarrow$") for _ in names for metric, words, _ in TABLE) + r" \\", r"\midrule",
    ]  # fmt: skip
    for arm in dict.fromkeys(arm for *_, arms in ROWS for arm in arms):
        if arm == "rand_act":
            lines.append(r"\midrule")
        lines.append(LABELS[arm] + " & " + " & ".join(" & ".join(cells[arm].get(name, ["--"] * width)) for name in names) + r" \\")
        if arm == "evi":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (BUILD / "main" / "tab_comparison.tex").write_text("\n".join(lines) + "\n")
    print(BUILD / "main" / "tab_comparison.tex")
    (BUILD / "main" / "tab_comparison_standalone.tex").write_text(
        "\\documentclass[border=4pt]{standalone}\n\\usepackage{booktabs}\n\\begin{document}\n\\input{tab_comparison.tex}\n\\end{document}\n"
    )
