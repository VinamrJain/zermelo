"""One launch written for the paper: `python -m zermelo.experiments.balloon_waypoint.metrics <sweep name or its directory> [--top N] [--without TOKEN ...]`"""

import re
import sys
from pathlib import Path

import numpy as np

from zermelo.experiments import report
from zermelo.experiments.balloon_waypoint.metrics.data import (
    CURVES,
    DRAWN_ARMS,
    TABLE_COLUMNS,
    acquisition_label,
    finished_runs,
    latest_launch,
    tables,
)

SWEEPS = Path("results") / __package__.split(".")[-2]
"""Where a sweep named rather than pointed at is looked for: this experiment's own results"""

REGRET_ONLY_SWEEPS = ("oracle",)
"""Sweeps whose figure draws the two regret panels and nothing else"""

asked = Path(sys.argv[1])
where = asked if asked.exists() else SWEEPS / asked
if not where.is_dir():
    sys.exit(f"no sweep or launch directory at {asked}, and none at {SWEEPS / asked}")
launch = latest_launch(where)
sweep = launch.parent.name
runs = finished_runs(launch)  # names and configurations only, the arrays read one run at a time
if "--without" in sys.argv:  # arms whose name carries one of the tokens are left out of the figure and the table
    dropped = set(sys.argv[sys.argv.index("--without") + 1 :])
    runs = [run for run in runs if not dropped & set(run[0].split("_"))]
curves, scalars = tables(runs)
arms = sorted(curves["acquisition"].unique())
labels = {arm: acquisition_label(arm) for arm in arms}
budgets = {found.group() for label in labels.values() if (found := re.search(r"-L\d+", label))}
if len(budgets) == 1:  # one planning budget across the launch is left out of every label
    labels = {arm: label.replace(next(iter(budgets)), "") for arm, label in labels.items()}
drawn = arms if DRAWN_ARMS is None else [arm for arm in DRAWN_ARMS if arm in arms]
if "--top" in sys.argv:  # the N arms of lowest median cumulative regret, the table keeping every arm
    ranked = scalars.groupby("acquisition")["cumulative_regret"].median().sort_values()
    drawn = [arm for arm in ranked.index[: int(sys.argv[sys.argv.index("--top") + 1])] if arm in drawn]
panels = [(name, words, scale) for name, (words, scale) in CURVES.items() if name.endswith("regret") or sweep not in REGRET_ONLY_SWEEPS]

# the x up to which every arm walked at random: the x of the last opening time_step, every ruled arm sharing one count
opening_steps = {int(config["method"]["opening_steps"]) for *_, config in runs if config.get("method") is not None}
opening_x = float(np.sort(curves["x"].unique())[max(opening_steps) - 1]) if opening_steps and max(opening_steps) > 0 else None

# an arm with no rule claims nothing it learned, so the panels read off a claim leave it out
ruleless = {arm for arm, _, _, config in runs if config.get("method") is None}
omitted = {(name, arm) for name, _, _ in panels if not name.endswith("regret") for arm in ruleless}

paper = launch / "paper"
stem = f"{SWEEPS.name}_{sweep}"
csvs = report.write_curves(paper / "results" / stem, curves, [name for name, _, _ in panels], labels)
report.write_figure(paper / f"fig_{stem}.tex", csvs, panels, drawn, labels, "hours elapsed", opening_x, omitted)
table = report.write_table(paper / f"tab_{stem}.tex", scalars, TABLE_COLUMNS, labels)
report.compile_document(paper / f"fig_{stem}.tex")
report.compile_document(table)
print(
    f"wrote {len(csvs)} CSVs, fig_{stem}.tex and tab_{stem}.tex under {paper}, over {len(arms)} acquisitions and {scalars['seed'].nunique()} seeds"
)
