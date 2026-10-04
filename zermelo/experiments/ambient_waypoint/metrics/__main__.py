"""One launch written for the paper: `python -m zermelo.experiments.ambient_waypoint.metrics <sweep name or its directory>`"""

import sys
from pathlib import Path

import numpy as np

from zermelo.experiments import report
from zermelo.experiments.ambient_waypoint.metrics.data import (
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
curves, scalars = tables(runs)
arms = sorted(curves["acquisition"].unique())
labels = {arm: acquisition_label(arm) for arm in arms}
drawn = arms if DRAWN_ARMS is None else [arm for arm in DRAWN_ARMS if arm in arms]
panels = [(name, words, scale) for name, (words, scale) in CURVES.items() if name.endswith("regret") or sweep not in REGRET_ONLY_SWEEPS]

# the x up to which every arm walked at random: the x of the last opening time_step, every ruled arm sharing one count
opening_steps = {int(config["method"]["opening_steps"]) for *_, config in runs if config.get("method") is not None}
opening_x = float(np.sort(curves["x"].unique())[max(opening_steps) - 1]) if opening_steps and max(opening_steps) > 0 else None

paper = launch / "paper"
stem = f"{SWEEPS.name}_{sweep}"
csvs = report.write_curves(paper / "results" / stem, curves, [name for name, _, _ in panels], labels)
report.write_figure(paper / f"fig_{stem}.tex", csvs, panels, drawn, labels, "time step", opening_x, set())
table = report.write_table(paper / f"tab_{stem}.tex", scalars, TABLE_COLUMNS, labels)
report.compile_document(paper / f"fig_{stem}.tex")
report.compile_document(table)
print(
    f"wrote {len(csvs)} CSVs, fig_{stem}.tex and tab_{stem}.tex under {paper}, over {len(arms)} acquisitions and {scalars['seed'].nunique()} seeds"
)
