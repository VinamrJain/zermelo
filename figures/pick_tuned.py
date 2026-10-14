"""The settings a tuning launch picks: per rule, the arm of lowest median metric over the held-out seeds

Run from code/:  pixi run fig pick_tuned <ambient_waypoint|balloon_waypoint> <launch dir>
EMI is picked on cumulative regret and EVI on RMSE. Prints every arm's median and the pick; a tie keeps the first arm listed.
"""

import importlib
import sys
from pathlib import Path

METRIC = {"emi": "cumulative_regret", "evi": "rmse"}

data = importlib.import_module(f"zermelo.experiments.{sys.argv[1]}.metrics.data")
runs = data.finished_runs(data.latest_launch(Path(sys.argv[2])))
_, scalars = data.tables(runs)
for rule, metric in METRIC.items():
    held = scalars[scalars["acquisition"].str.startswith(rule)]
    table = held.groupby("acquisition").agg(
        median=(metric, "median"),
        mean=(metric, "mean"),
        seeds=("seed", "nunique"),
        other=(METRIC["evi" if rule == "emi" else "emi"], "median"),
    )
    print(f"\n{rule.upper()} by {metric} (other = median of the other rule's metric)")
    print(table.sort_values("median").to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"PICK {rule}: {table['median'].idxmin()}")
