"""The comparison's remaining numbers, as two appendix tables

tab_metrics: per rule on each problem, mean +- standard error of the final posterior variance (mean over the region's cells of the
variance summed over the field's axes), the share of targets reached, seconds per time_step and per episode; the balloon also its share
of time in the region and its altitude changes.
tab_per_date: the balloon's final simple regret, cumulative regret and RMSE on each start date.
The best mean of a competed column is bold.
"""

import numpy as np
import pandas as pd

from figures.common import BUILD, FRAMES, LABELS, launch, read

ARMS = ("emi", "evi", "mc_ei", "ucb", "ts", "max_var", "rand_target", "rand_act")
STANDALONE = "\\documentclass[border=4pt]{standalone}\n\\usepackage{booktabs}\n\\renewcommand{\\arraystretch}{1.15}\n\\begin{document}\n\\input{%s.tex}\n\\end{document}\n"
DATES = {"sep06": "Sep 6", "sep09": "Sep 9", "sep19": "Sep 19", "sep24": "Sep 24"}


def cell(values, digits: int, bold: bool) -> str:
    """Mean +- standard error of one arm's values, the mean bold where asked, or a dash where it has none"""
    if values.isna().all():
        return "--"
    return (r"$\mathbf{%s} \pm %s$" if bold else r"$%s \pm %s$") % (f"{values.mean():.{digits}f}", f"{values.sem():.{digits}f}")


def write(name: str, lines: list[str]) -> None:
    """One table as a tex fragment and a standalone document including it"""
    (BUILD / "appendix").mkdir(parents=True, exist_ok=True)
    (BUILD / "appendix" / f"{name}.tex").write_text("\n".join(lines) + "\n")
    (BUILD / "appendix" / f"{name}_standalone.tex").write_text(STANDALONE % name)
    print(BUILD / "appendix" / f"{name}.tex")


def rows(scalars, columns, arms=ARMS) -> list[str]:
    """One line per arm, ours then the baselines then no rule; a column's fourth entry is the end that is best, or None"""
    means = scalars.groupby("acquisition")[[metric for metric, *_ in columns]].mean().reindex(list(arms))
    best = {metric: getattr(means[metric], f"idx{end}")() if end else None for metric, _, _, end in columns}
    lines = []
    for arm in arms:
        held = scalars[scalars["acquisition"] == arm]
        if arm == "rand_act" or arm == "mc_ei":
            lines.append(r"\midrule")
        lines.append(
            LABELS[arm] + " & " + " & ".join(cell(held[metric], digits, best[metric] == arm) for metric, _, digits, _ in columns) + r" \\"
        )
    return lines


def grid_variance() -> pd.DataFrame:
    """The 2D grid's final posterior variance per run, read off its last claim (mean then log variance per axis)"""
    found = []
    for run in sorted(launch("ambient_waypoint", "paper_main").glob("*/record.npz")):
        arm, seed = (pair.split("=", 1)[1] for pair in run.parent.name.split(","))
        with np.load(run) as record:
            log_variance = np.split(record["objective_state/claim"][-1], 2, axis=-1)[1]
        found.append(
            {
                "acquisition": arm.removeprefix("paper_main_"),
                "seed": int(seed),
                "posterior_variance": np.exp(np.maximum(log_variance, -12.0)).sum(axis=-1).mean(),
            }
        )
    return pd.DataFrame(found)


grid_curves, grid = read("ambient_waypoint", {"main": "paper_main"})
balloon_curves, balloon = read("balloon_waypoint", {date: f"paper_main_{date}" for date in FRAMES})
grid = grid.assign(seconds_per_step=grid["seconds"] / grid_curves["x"].nunique()).merge(grid_variance(), on=["acquisition", "seed"])
grid.loc[grid["acquisition"] == "rand_act", "posterior_variance"] = np.nan  # no rule, so no belief of its own
balloon = balloon.assign(seconds_per_step=balloon["seconds"] / balloon_curves["x"].nunique())

GRID = (
    ("posterior_variance", r"post. variance $\downarrow$", 2, "min"), ("arrival_share", r"reached \% $\uparrow$", 1, "max"),
    ("seconds_per_step", "s / step", 2, None), ("seconds", "s / episode", 0, None),
)  # fmt: skip
BALLOON = (
    ("posterior_variance", r"post. variance $\downarrow$", 2, "min"), ("arrival_share", r"reached \% $\uparrow$", 0, "max"),
    ("in_box_share", r"in region \% $\uparrow$", 0, "max"), ("altitude_changes", "alt. changes", 0, None),
    ("seconds_per_step", "s / step", 2, None), ("seconds", "s / episode", 0, None),
)  # fmt: skip
lines = [
    r"\begin{tabular}{l" + "r" * len(GRID) + r"@{\hskip 1.2em}" + "r" * len(BALLOON) + "}", r"\toprule",
    rf" & \multicolumn{{{len(GRID)}}}{{c}}{{\textbf{{2D grid}}}} & \multicolumn{{{len(BALLOON)}}}{{c}}{{\textbf{{balloon}}}} \\",
    rf"\cmidrule(lr){{2-{1 + len(GRID)}}} \cmidrule(lr){{{2 + len(GRID)}-{1 + len(GRID) + len(BALLOON)}}}",
    "acquisition & " + " & ".join(words for _, words, *_ in GRID + BALLOON) + r" \\", r"\midrule",
]  # fmt: skip
for left, right in zip(rows(grid, GRID), rows(balloon, BALLOON)):
    lines.append(left if left == r"\midrule" else left.removesuffix(r" \\") + " & " + right.split(" & ", 1)[1])
write("tab_metrics", lines + [r"\bottomrule", r"\end{tabular}"])

PER_DATE = (
    ("simple_regret", r"simple regret $\downarrow$", 1, "min"), ("cumulative_regret", r"cum. regret $\downarrow$", 0, "min"),
    ("rmse", r"RMSE $\downarrow$", 2, "min"),
)  # fmt: skip
lines = [
    r"\begin{tabular}{l" + (r"@{\hskip 1.2em}" + "r" * len(PER_DATE)) * len(FRAMES) + "}", r"\toprule",
    " & " + " & ".join(rf"\multicolumn{{{len(PER_DATE)}}}{{c}}{{\textbf{{{DATES[date]}}}}}" for date in FRAMES) + r" \\",
    " ".join(rf"\cmidrule(lr){{{2 + len(PER_DATE) * at}-{1 + len(PER_DATE) * (at + 1)}}}" for at in range(len(FRAMES))),
    "acquisition & " + " & ".join(words for _ in FRAMES for _, words, *_ in PER_DATE) + r" \\", r"\midrule",
]  # fmt: skip
per_date = [rows(balloon[balloon["world"] == date], PER_DATE) for date in FRAMES]
for across in zip(*per_date):
    lines.append(
        across[0]
        if across[0] == r"\midrule"
        else across[0].removesuffix(r" \\") + "".join(" & " + one.split(" & ", 1)[1].removesuffix(r" \\") for one in across[1:]) + r" \\"
    )
write("tab_per_date", lines + [r"\bottomrule", r"\end{tabular}"])
