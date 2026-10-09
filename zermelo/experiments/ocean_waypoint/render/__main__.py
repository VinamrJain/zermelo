"""One still per arm of a launch: `python -m zermelo.experiments.ocean_waypoint.render <sweep name or launch directory> [--seed N]`

Three panels, over the candidate box in degrees:
    truth      ||W|| at every grid point, the boat's track, the target, the waypoints it aimed at
    claimed    ||mu_T|| at every candidate at the last move
    spread     sqrt(exp(v_T,u) + exp(v_T,v)) at every candidate at the last move
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display on a compute node
import matplotlib.pyplot as plt  # noqa: E402

from zermelo.experiments.ocean_waypoint.metrics.data import channels, claimed, latest_launch, settings  # noqa: E402

SWEEPS = Path("results") / __package__.split(".")[-2]
"""Where a sweep named rather than pointed at is looked for: this experiment's own results"""

CHANNELS = (
    "state/position",
    "state/target",
    "objective_state/truth",
    "objective_state/claim",
    "objective_state/arrived",
    "objective_state/steps_to_arrival",
    "agent_state/scores.waypoint",
    "agent_state/scores.candidate_indices",
)
"""What a still is drawn from, as the names a record holds them under"""


def grid_of(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(latitude, longitude, land) of the grid a run was built on, strided as the run read it"""
    raw = np.load(Path(config["problem"]["data_path"]))
    stride = int(config["problem"]["grid_stride"])
    return raw["latitude"][::stride], raw["longitude"][::stride], raw["land"][::stride, ::stride]


def still(run: Path, out: Path) -> None:
    """Draw one episode's sheet to `out`"""
    config = json.loads((run / "config.json").read_text())
    held = channels(run, CHANNELS)
    lat, lon, land = grid_of(config)
    n_lat, n_lon = len(lat), len(lon)
    problem = config["problem"]
    live = (
        (lat[:, None] >= problem["lat_min"])
        & (lat[:, None] <= problem["lat_max"])
        & (lon[None, :] >= problem["lon_min"])
        & (lon[None, :] <= problem["lon_max"])
    ) & ~land  # (n_lat, n_lon): the candidate box, as `ocean_objective` marks it
    candidates = np.flatnonzero(live.reshape(-1))

    def on_grid(per_candidate: np.ndarray) -> np.ndarray:
        """A value per candidate laid onto the grid, nan elsewhere"""
        sheet = np.full(n_lat * n_lon, np.nan)
        sheet[candidates] = per_candidate
        return sheet.reshape(n_lat, n_lon)

    mean, spread = claimed(held["objective_state/claim"][-1])  # (n_candidates, 2) and (n_candidates,)
    speed = np.linalg.norm(mean, axis=-1)
    track = held["state/position"]  # (time_steps + 1, 2) as (lat, lon)
    sheets = [
        ("true ocean current speed (m/s)", on_grid(np.linalg.norm(held["objective_state/truth"][-1], axis=-1))),
        ("claimed speed (m/s)", on_grid(speed)),
        ("claimed spread (m/s)", on_grid(spread)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
    top = np.nanmax(sheets[0][1])
    for ax, (words, sheet) in zip(axes, sheets, strict=True):
        drawn = ax.pcolormesh(lon, lat, sheet, shading="nearest", cmap="viridis", vmin=0, vmax=top if "spread" not in words else None)
        fig.colorbar(drawn, ax=ax, label=words)
        ax.plot(track[:, 1], track[:, 0], "w.-", lw=1, ms=3)
        ax.plot(track[0, 1], track[0, 0], "wo", ms=6, mfc="none")
        ax.plot(track[-1, 1], track[-1, 0], "r*", ms=9)
        ax.plot(held["state/target"][0, 1], held["state/target"][0, 0], "m*", ms=12, mec="w", label="target")
        ax.set_aspect(1 / np.cos(np.radians(np.mean(lat))))
        ax.set_xlabel("lon")
    if "agent_state/scores.waypoint" in held:  # an arm with a rule: every waypoint it aimed at
        aimed = np.unique(held["agent_state/scores.waypoint"])
        axes[0].plot(lon[aimed % n_lon], lat[aimed // n_lon], "x", color="orange", ms=6, label="waypoints")
        axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("lat")
    fig.suptitle(
        f"{run.parent.parent.name} / {run.name}: {track.shape[0] - 1} moves of {problem['step_hours']} h, arrived at step {int(held['objective_state/steps_to_arrival'][-1])}"
        if held["objective_state/arrived"][-1]
        else f"{run.parent.parent.name} / {run.name}: {track.shape[0] - 1} moves of {problem['step_hours']} h, never arrived",
        fontsize=10,
    )
    fig.savefig(out, dpi=120)
    plt.close(fig)


asked = Path(sys.argv[1])
where = asked if asked.exists() else SWEEPS / asked
if not where.is_dir():
    sys.exit(f"no sweep or launch directory at {asked}, and none at {SWEEPS / asked}")
launch = latest_launch(where)
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
drawn = 0
for marker in sorted(launch.glob("*/record.npz")):
    varied = settings(marker.parent.name)
    if int(varied.get("seed", 0)) == seed:
        (launch / "stills").mkdir(exist_ok=True)
        still(marker.parent, launch / "stills" / f"{varied.get('arm', marker.parent.name)}.png")
        drawn += 1
print(f"drew {drawn} stills under {launch / 'stills'}")
