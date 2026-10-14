"""One still per arm of a launch, or one film per arm sent to the scheduler

    python -m zermelo.experiments.ocean_waypoint.render <sweep name, launch directory or run directory> [--seed N] [--film [--local]]

Three panels, over the candidate box in degrees:
    truth      ||W|| at every grid point, the boat's track, the target, the waypoints it aimed at
    claimed    ||mu_T|| of the ocean current at every candidate at the last move
    spread     sqrt(exp(v_T,u) + exp(v_T,v)) of the ocean current, then of the wind, at every candidate at the last move
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display on a compute node
import matplotlib.pyplot as plt  # noqa: E402

from zermelo.experiments.ocean_waypoint.metrics.data import channels, claimed, latest_launch, settings  # noqa: E402
from zermelo.experiments.ocean_waypoint.render.film import candidate_box, film, send  # noqa: E402
from zermelo.problems.ocean import OCEAN_CURRENT, WIND  # noqa: E402

SWEEPS = Path("results") / __package__.split(".")[-2]
"""Where a sweep named rather than pointed at is looked for: this experiment's own results"""

CHANNELS = (
    "state/position",
    "state/target",
    "objective_state/truth",
    "objective_state/claim",
    "objective_state/arrived",
    "objective_state/information",
    "objective_state/steps_to_arrival",
    "objective_state/best_observed_magnitude",
    "objective_state/best_possible_magnitude",
    "objective_state/targets",
    "objective_state/visited",
    "objective_state/steps_to_completion",
    "agent_state/scores.waypoint",
    "agent_state/scores.candidate_indices",
    "agent_state/time_steps_since_waypoint",
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
    targets_drawn = held["objective_state/targets"][0] if "objective_state/targets" in held else None
    candidates = np.flatnonzero(candidate_box(problem, lat, lon, land, targets_drawn))
    reaching = "objective_state/arrived" in held  # else the target the world drew is unread
    searching, visiting = "objective_state/best_observed_magnitude" in held, "objective_state/targets" in held
    where = "" if reaching else "target_"  # the candidate box the waypoints are aimed in
    box_lat, box_lon = (problem[f"{where}lat_min"], problem[f"{where}lat_max"]), (problem[f"{where}lon_min"], problem[f"{where}lon_max"])

    def on_grid(per_candidate: np.ndarray) -> np.ndarray:
        """A value per candidate laid onto the grid, nan elsewhere"""
        sheet = np.full(n_lat * n_lon, np.nan)
        sheet[candidates] = per_candidate
        return sheet.reshape(n_lat, n_lon)

    mean, spread = claimed(held["objective_state/claim"][-1], OCEAN_CURRENT)  # (n_candidates, 2) and (n_candidates,)
    _, wind_spread = claimed(held["objective_state/claim"][-1], WIND)
    track = held["state/position"]  # (time_steps + 1, 2) as (lat, lon)
    sheets = [
        ("true ocean current speed (m/s)", on_grid(np.linalg.norm(held["objective_state/truth"][-1][:, list(OCEAN_CURRENT)], axis=-1))),
        ("claimed current speed (m/s)", on_grid(np.linalg.norm(mean, axis=-1))),
        ("claimed current spread (m/s)", on_grid(spread)),
        ("claimed wind spread (m/s)", on_grid(wind_spread)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.2), constrained_layout=True)
    top = np.nanmax(sheets[0][1])
    for ax, (words, sheet) in zip(axes, sheets, strict=True):
        drawn = ax.pcolormesh(lon, lat, sheet, shading="nearest", cmap="viridis", vmin=0, vmax=top if "spread" not in words else None)
        fig.colorbar(drawn, ax=ax, label=words)
        ax.plot(track[:, 1], track[:, 0], "k-", lw=2.5)  # an outline, so the track shows off the candidate box too
        ax.plot(track[:, 1], track[:, 0], "w.-", lw=1, ms=3)
        ax.add_patch(
            plt.Rectangle(
                (box_lon[0], box_lat[0]), box_lon[1] - box_lon[0], box_lat[1] - box_lat[0], fill=False, ec="orange", lw=0.8, ls="--"
            )
        )
        ax.plot(track[0, 1], track[0, 0], "o", ms=7, mfc="none", mec="k", mew=1.5, label="start")
        ax.plot(track[-1, 1], track[-1, 0], "r*", ms=9)
        if reaching:
            ax.plot(held["state/target"][0, 1], held["state/target"][0, 0], "m*", ms=12, mec="w", label="target")
        if searching:  # the fastest current in the box, the point the search is scored against
            fastest = candidates[np.argmax(np.linalg.norm(held["objective_state/truth"][-1], axis=-1))]
            ax.plot(lon[fastest % n_lon], lat[fastest // n_lon], "*", ms=14, color="gold", mec="k", label="fastest current")
        if visiting:  # every target, filled once visited
            for (t_lat, t_lon), seen in zip(held["objective_state/targets"][0], held["objective_state/visited"][-1], strict=True):
                ax.plot(t_lon, t_lat, "*", ms=14, color="m" if seen else "none", mec="m", mew=1.5)
        ax.set_aspect(1 / np.cos(np.radians(np.mean(lat))))
        ax.set_xlabel("lon")
    if "agent_state/scores.waypoint" in held:  # an arm with a rule: every waypoint it aimed at
        decided = np.flatnonzero(held["agent_state/time_steps_since_waypoint"][1:] == 0) + 1  # the moves a waypoint was chosen at
        aimed = np.unique(held["agent_state/scores.waypoint"][decided])
        axes[0].plot(lon[aimed % n_lon], lat[aimed // n_lon], "x", color="orange", ms=6, label="waypoints")
    if visiting:
        axes[0].plot([], [], "*", ms=10, color="m", mec="m", label="target, visited")
        axes[0].plot([], [], "*", ms=10, color="none", mec="m", mew=1.5, label="target, not yet")
    axes[0].legend(loc="lower right", fontsize=7)
    axes[0].set_ylabel("lat")
    fig.suptitle(
        f"{run.parent.parent.name} / {run.name}: {track.shape[0] - 1} moves of {problem['step_hours']} h, arrived at step {int(held['objective_state/steps_to_arrival'][-1])}"
        if reaching and held["objective_state/arrived"][-1]
        else f"{run.parent.parent.name} / {run.name}: {track.shape[0] - 1} moves of {problem['step_hours']} h, "
        + (
            "never arrived"
            if reaching
            else f"{held['objective_state/information'][-1]:.0f} nats learned"
            if "objective_state/information" in held
            else f"best {held['objective_state/best_observed_magnitude'][-1]:.2f} of {held['objective_state/best_possible_magnitude'][-1]:.2f} m/s"
            if searching
            else f"{int(held['objective_state/visited'][-1].sum())} of {held['objective_state/visited'].shape[1]} targets visited"
            + (f", all by step {int(held['objective_state/steps_to_completion'][-1])}" if held["objective_state/visited"][-1].all() else "")
        ),
        fontsize=10,
    )
    fig.savefig(out, dpi=120)
    plt.close(fig)


parse = argparse.ArgumentParser(prog="render", description="Draw recorded ocean episodes.")
parse.add_argument("target", type=Path, help="a sweep name, a launch directory, or one run directory")
parse.add_argument("--seed", type=int, default=0, help="which seed's arms to draw")
parse.add_argument("--film", action="store_true", help="write video instead of a still")
parse.add_argument("--local", action="store_true", help="draw the film here rather than sending it to the scheduler")
parse.add_argument("--fps", type=int, default=12, help="frames a second the film plays at")
parse.add_argument("--dpi", type=int, default=120, help="how finely a frame is drawn")
parse.add_argument("--still", type=int, help="film only: one frame at this move as a png, drawn here")
given = parse.parse_args()
where = given.target if given.target.exists() else SWEEPS / given.target
if not where.is_dir():
    raise SystemExit(f"no sweep, launch or run directory at {given.target}, and none at {SWEEPS / given.target}")
if (where / "record.npz").exists():
    launch, runs = where.parent, [where]
else:
    launch = latest_launch(where)
    runs = [
        marker.parent for marker in sorted(launch.glob("*/record.npz")) if int(settings(marker.parent.name).get("seed", 0)) == given.seed
    ]
if given.film and not (given.local or given.still is not None):
    print(send(runs, launch / "films", f"--fps {given.fps} --dpi {given.dpi}"))
elif given.film:
    for run in runs:
        (launch / "films").mkdir(exist_ok=True)
        film(run, launch / "films" / f"{settings(run.name).get('arm', run.name)}.mp4", fps=given.fps, dpi=given.dpi, still=given.still)
else:
    (launch / "stills").mkdir(exist_ok=True)
    for run in runs:
        still(run, launch / "stills" / f"{settings(run.name).get('arm', run.name)}.png")
    print(f"drew {len(runs)} stills under {launch / 'stills'}")
