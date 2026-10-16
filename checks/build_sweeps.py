"""Builds the utility and the planner of every arm of every sweep, and prints each failure with the arms it hit; exits 1 on any failure"""

import importlib
import sys

from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

for experiment in ("ambient_waypoint", "balloon_waypoint", "ocean_waypoint"):
    importlib.import_module(f"zermelo.experiments.{experiment}.sweeps")
failures: dict[str, list[str]] = {}  # the error's last line -> the arms it hit
built = 0
for name, node in ConfigStore.instance().repo["arm"].items():
    method = OmegaConf.to_container(node.node, resolve=False).get("method")
    for part in ("utility", "planner") if method else ():
        built += 1
        try:
            instantiate(method[part])
        except Exception as error:  # noqa: BLE001
            failures.setdefault(f"{part}: {str(error).splitlines()[-1][:160]}", []).append(name)
print(f"{built} built, {sum(map(len, failures.values()))} failed")
for error, arms in failures.items():
    print(f"{error} | {len(arms)} arms, e.g. {arms[0]}")
sys.exit(1 if failures else 0)
