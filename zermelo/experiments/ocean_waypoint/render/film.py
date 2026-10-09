"""One ocean episode as a film: the boat on the true ocean current, what it believes, and how far it still is from the target

Top: ||W|| shaded with its streamlines, the wind as a quiver, land masked, the track fading behind the boat, the boat as a
triangle pointing along its heading, at the boat the wind, the ocean current, the heading through the water and the net motion as
arrows, the target with its arrival ring, the waypoint aimed at; beside it the numbers of the move. Bottom: the forecast's error
||F - W|| over the box, the belief's error ||mu_t - W|| as it learns, and the polar, boat speed against heading for the wind at
the boat with the heading marked. Moves are interpolated over `SUBFRAMES` frames; the film ends a few moves after arrival.

The heading is not recorded: it is read back as the net motion of the move less the ocean current at the point it left.
"""

import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display on a compute node
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from matplotlib.projections.polar import PolarAxes  # noqa: E402

from zermelo.experiments.ocean_waypoint.metrics.data import channels, claimed  # noqa: E402
from zermelo.experiments.ocean_waypoint.schema import Resources  # noqa: E402
from zermelo.problems.ocean import KM_PER_DEGREE  # noqa: E402
from zermelo.problems.ocean.transition import Polar  # noqa: E402

SUBFRAMES = 6
"""Frames one move is spread over"""

AFTER_ARRIVAL = 4
"""Moves shown once arrived"""

TRAIL = 30
"""Moves the trail fades over"""

ARROW_KM_PER_MS = 70.0
"""km of arrow per m/s of velocity, all four arrows at the boat"""

WIND, CURRENT, HEADING, NET, TRACK, TARGET = "#ffd54f", "#4fc3f7", "#ffffff", "#ff7043", "#4fc3f7", "#ff5fa2"

RESOURCES_PER_FILM = Resources(
    cpus=2, mem_gb=8, timeout_min=90, gres=None, partition="default_partition", constraint="avx2", array_parallelism=8
)
"""What one film is given"""

SCRIPT = """#!/bin/bash
#SBATCH --job-name=zermelo-film
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_gb}G
#SBATCH --time={timeout_min}
#SBATCH --array=0-{last}%{parallelism}
#SBATCH --chdir={repository}
#SBATCH --output={into}/film-%a.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
RUN=(
{lines}
)
{python} -m zermelo.experiments.ocean_waypoint.render ${{RUN[$SLURM_ARRAY_TASK_ID]}} --film --local {flags}
"""
"""The array a submission writes, one task per run"""


def send(runs: Sequence[Path], into: Path, flags: str) -> str:
    """What the scheduler said, having written `into`/film.sbatch with one task per run and submitted it"""
    if shutil.which("sbatch") is None:
        raise SystemExit("no sbatch on this machine. Draw the film here instead, with --local")
    into.mkdir(parents=True, exist_ok=True)
    script = into / "film.sbatch"
    script.write_text(
        SCRIPT.format(
            partition=RESOURCES_PER_FILM.partition,
            cpus=RESOURCES_PER_FILM.cpus,
            mem_gb=RESOURCES_PER_FILM.mem_gb,
            timeout_min=RESOURCES_PER_FILM.timeout_min,
            last=len(runs) - 1,
            parallelism=RESOURCES_PER_FILM.array_parallelism,
            repository=Path.cwd().resolve(),
            into=into.resolve(),
            python=sys.executable,
            lines="\n".join(f'  "{run.resolve()}"' for run in runs),
            flags=flags,
        )
    )
    said = subprocess.run(["sbatch", str(script)], capture_output=True, text=True, check=True)  # noqa: S603
    return f"{said.stdout.strip()}, {len(runs)} film{'s' if len(runs) != 1 else ''} from {script}"


def film(run: Path, out: Path, *, fps: int, dpi: int, still: int | None) -> None:
    """The episode recorded at `run` written to `out` as video, or the frame at move `still` as a png beside it"""

    config = json.loads((run / "config.json").read_text())
    problem = config["problem"]
    held = channels(
        run,
        (
            "state/position",
            "state/target",
            "objective_state/truth",
            "objective_state/claim",
            "objective_state/arrived",
            "agent_state/scores.waypoint",
        ),
    )
    raw = np.load(Path(problem["data_path"]))
    stride = int(problem["grid_stride"])
    lat, lon, land = raw["latitude"][::stride].astype(float), raw["longitude"][::stride].astype(float), raw["land"][::stride, ::stride]
    frame = int(np.argmin(np.abs(raw["hours"] - problem["frame_hour"])))
    lagged = int(np.argmin(np.abs(raw["hours"] - (problem["frame_hour"] - problem["forecast_lag_hours"]))))
    forecast_u, forecast_v = raw["u"][lagged][::stride, ::stride].astype(float), raw["v"][lagged][::stride, ::stride].astype(float)  # F
    wind_u, wind_v = raw["wind_u"][frame][::stride, ::stride].astype(float), raw["wind_v"][frame][::stride, ::stride].astype(float)
    n_lat, n_lon = len(lat), len(lon)
    live = (
        (lat[:, None] >= problem["lat_min"])
        & (lat[:, None] <= problem["lat_max"])
        & (lon[None, :] >= problem["lon_min"])
        & (lon[None, :] <= problem["lon_max"])
    ) & ~land
    candidates = np.flatnonzero(live.reshape(-1))
    polar = Polar(
        problem["top_speed_ms"],
        problem["no_go_degrees"],
        problem["peak_degrees"],
        problem["rise_exponent"],
        problem["downwind_factor"],
        problem["wind_scale_ms"],
    )

    def on_grid(per_candidate: np.ndarray) -> np.ndarray:
        """A value per candidate laid onto the grid, nan elsewhere"""
        sheet = np.full(n_lat * n_lon, np.nan)
        sheet[candidates] = per_candidate
        return sheet.reshape(n_lat, n_lon)

    def at(sheet_u: np.ndarray, sheet_v: np.ndarray, point: np.ndarray) -> np.ndarray:
        """(u, v) of a gridded field at the grid point nearest degrees `point` (lat, lon), zero where unknown"""
        r, c = int(np.argmin(np.abs(lat - point[0]))), int(np.argmin(np.abs(lon - point[1])))
        return np.nan_to_num(np.array([sheet_u[r, c], sheet_v[r, c]]))

    truth = held["objective_state/truth"][0]  # (n_candidates, 2) as (u, v), still over the episode
    true_u, true_v = on_grid(truth[:, 0]), on_grid(truth[:, 1])
    true_speed = np.hypot(true_u, true_v)
    forecast_error = np.where(live, np.hypot(true_u - forecast_u, true_v - forecast_v), np.nan)  # ||W - F|| over the candidates
    track, target = held["state/position"].astype(float), held["state/target"][0].astype(float)  # (T + 1, 2) and (2,) as (lat, lon)
    arrived = held["objective_state/arrived"]
    step_hours = float(problem["step_hours"])
    cos_mid = np.cos(np.radians(np.mean(lat)))
    km_per_degree = np.array([KM_PER_DEGREE, KM_PER_DEGREE * cos_mid])  # per degree of (lat, lon)
    distance = np.linalg.norm((track - target) * km_per_degree, axis=-1)  # (T + 1,) km
    last = min(track.shape[0] - 1, int(np.argmax(arrived)) + AFTER_ARRIVAL) if arrived.any() else track.shape[0] - 1
    claims = held["objective_state/claim"]  # (T + 1, n_candidates, 4)
    waypoints = held.get("agent_state/scores.waypoint")

    # per move: the velocities at the point it left, (east, north) m/s
    net = np.diff(track, axis=0)[:, ::-1] * km_per_degree[::-1] / (3.6 * step_hours)  # (T, 2): the move as a velocity
    current = np.stack([at(true_u, true_v, p) for p in track[:-1]])  # (T, 2)
    wind = np.stack([at(wind_u, wind_v, p) for p in track[:-1]])  # (T, 2)
    through = net - current  # (T, 2): the heading's velocity through the water, read back
    heading = through / np.maximum(np.linalg.norm(through, axis=-1, keepdims=True), 1e-9)  # (T, 2) unit (east, north)

    plt.rcParams.update(
        {"font.family": "serif", "font.size": 13, "text.color": "#e6e6e6", "axes.labelcolor": "#e6e6e6", "xtick.color": "#bbbbbb",
         "ytick.color": "#bbbbbb", "axes.edgecolor": "#555555", "axes.titlesize": 14}
    )  # fmt: skip
    fig = plt.figure(figsize=(16, 9.5), facecolor="#0b0f14")
    grid = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[1, 1, 1, 0.9],
        height_ratios=[1.55, 1.0],
        hspace=0.3,
        wspace=0.24,
        left=0.045,
        right=0.985,
        top=0.93,
        bottom=0.06,
    )
    main = fig.add_subplot(grid[0, :3])
    side = fig.add_subplot(grid[0, 3])  # the numbers of the move and the legend, beside the map
    side.axis("off")
    error_ax, learned_ax = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    rose = fig.add_subplot(grid[1, 2], projection="polar", facecolor="#0b0f14")
    if not isinstance(rose, PolarAxes):
        raise TypeError("a polar subplot was asked for")
    for ax in (main, error_ax, learned_ax):
        ax.set_facecolor("#0b0f14")
        ax.set_xticks(np.arange(-75, -54, 5))
        ax.set_aspect(1 / cos_mid)
        ax.set_xlim(problem["lon_min"], problem["lon_max"])
        ax.set_ylim(problem["lat_min"], problem["lat_max"])
        ax.tick_params(labelsize=10)
    land_sheet = np.where(land, 1.0, np.nan)
    land_cmap = matplotlib.colors.ListedColormap(["#3a3a3a"])

    # main: the truth and the wind
    shade = main.pcolormesh(lon, lat, true_speed, shading="nearest", cmap="YlGnBu_r", vmin=0, vmax=np.nanmax(true_speed), alpha=0.95)
    even_lon, even_lat = np.linspace(lon[0], lon[-1], n_lon), np.linspace(lat[0], lat[-1], n_lat)  # streamplot wants exact even spacing
    main.streamplot(
        even_lon, even_lat, np.nan_to_num(true_u), np.nan_to_num(true_v), color="white", linewidth=0.5, density=1.5, arrowsize=0.7, zorder=2
    )
    main.pcolormesh(lon, lat, land_sheet, shading="nearest", cmap=land_cmap, vmin=0, vmax=1, zorder=3)
    every = 5
    main.quiver(
        lon[::every],
        lat[::every],
        wind_u[::every, ::every],
        wind_v[::every, ::every],
        color=WIND,
        alpha=0.5,
        scale=220,
        width=0.0025,
        headwidth=4,
        zorder=2.5,
    )
    cbar = fig.colorbar(shade, ax=main, fraction=0.03, pad=0.01)
    cbar.set_label("true ocean current speed (m/s)", fontsize=11)
    cbar.ax.tick_params(labelsize=10, colors="#bbbbbb")
    ring_km = float(problem["arrival_km"])
    main.add_patch(
        Ellipse(
            (target[1], target[0]),
            2 * ring_km / km_per_degree[1],
            2 * ring_km / km_per_degree[0],
            fill=False,
            ec=TARGET,
            lw=1.6,
            ls="--",
            zorder=5,
        )
    )
    main.plot(target[1], target[0], marker="*", ms=20, color=TARGET, mec="white", mew=0.8, zorder=6, ls="none")
    main.plot(track[0, 1], track[0, 0], marker="o", ms=9, mfc="none", mec="white", mew=1.5, zorder=6, ls="none")
    (aim,) = main.plot([], [], marker="x", ms=11, color="#ffa726", mew=2.2, zorder=6, ls="none")
    trail = [main.plot([], [], color=TRACK, lw=2.8, alpha=0.0, zorder=5, solid_capstyle="round")[0] for _ in range(TRAIL)]
    (done,) = main.plot([], [], color=TRACK, lw=1.4, alpha=0.45, zorder=4)  # the track older than the trail
    (glow,) = main.plot([], [], marker="o", ms=30, color=TRACK, alpha=0.25, zorder=6.5, ls="none")
    (boat,) = main.plot([], [], marker=(3, 0, 0), ms=18, color=HEADING, mec="#0b0f14", mew=0.8, zorder=8, ls="none")
    arrows = {name: main.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.4, mutation_scale=18, shrinkA=0, shrinkB=0), zorder=7.5)
              for name, colour in (("wind", WIND), ("current", CURRENT), ("heading", HEADING), ("net", NET))}  # fmt: skip
    side.legend(
        handles=[
            Line2D([], [], color=WIND, lw=2.5, label="wind"),
            Line2D([], [], color=CURRENT, lw=2.5, label="ocean current"),
            Line2D([], [], color=HEADING, lw=2.5, label="heading (through water)"),
            Line2D([], [], color=NET, lw=2.5, label="net motion"),
            Line2D([], [], marker="*", ms=14, color=TARGET, mec="white", ls="none", label="target, 20 km ring"),
            Line2D([], [], marker="x", ms=10, color="#ffa726", mew=2, ls="none", label="waypoint aimed at"),
        ],
        loc="lower left", fontsize=12, frameon=False, labelcolor="#e6e6e6", handlelength=2.5,
    )  # fmt: skip
    main.set_xlabel("longitude", fontsize=12)
    main.set_ylabel("latitude", fontsize=12)
    caption = side.text(
        0.0, 1.0, "", transform=side.transAxes, ha="left", va="top", fontsize=12, color="white", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#121820", ec="#555555"), zorder=9,
    )  # fmt: skip
    badge = main.text(
        0.5,
        0.55,
        "",
        transform=main.transAxes,
        ha="center",
        va="center",
        fontsize=34,
        color=TARGET,
        fontweight="bold",
        alpha=0.0,
        zorder=10,
    )

    # the polar inset: boat speed against heading for the wind at the boat
    rose.set_theta_zero_location("N")
    rose.set_theta_direction(-1)  # compass: clockwise from north
    rose.set_ylim(0, problem["top_speed_ms"] * 1.05)
    rose.set_yticks([problem["top_speed_ms"] / 2, problem["top_speed_ms"]])
    rose.tick_params(labelsize=10, colors="#bbbbbb")
    rose.grid(color="#444444", lw=0.5)
    rose.set_title("boat speed by heading at this wind (m/s)", fontsize=12, color="#e6e6e6", pad=12)
    compass = np.radians(np.arange(0.0, 360.0, 2.0))  # bearings clockwise from north
    unit = np.stack([np.sin(compass), np.cos(compass)], axis=-1)  # (east, north)
    (rose_curve,) = rose.plot([], [], color=HEADING, lw=1.6)
    rose_fill = [rose.fill_between(compass, 0, 0, color=HEADING, alpha=0.12)]
    (rose_heading,) = rose.plot([], [], marker="o", ms=9, color=NET, mec="white", ls="none", zorder=5)
    (rose_wind,) = rose.plot([], [], color=WIND, lw=3)

    # bottom: the forecast's error, and the belief's correction of the forecast so far
    error_top = float(np.nanmax(forecast_error))
    error_ax.pcolormesh(lon, lat, forecast_error, shading="nearest", cmap="magma", vmin=0, vmax=error_top)
    learned_mesh = learned_ax.pcolormesh(
        lon, lat, on_grid(np.zeros(candidates.size)), shading="nearest", cmap="magma", vmin=0, vmax=error_top
    )
    for ax, words in ((error_ax, "forecast error  ||F - W||  (m/s)"), (learned_ax, "belief's error  ||mu - W||  (m/s)")):
        ax.pcolormesh(lon, lat, land_sheet, shading="nearest", cmap=land_cmap, vmin=0, vmax=1, zorder=3)
        ax.set_title(words, fontsize=12, pad=6)
        ax.plot(target[1], target[0], marker="*", ms=12, color=TARGET, mec="white", mew=0.5, zorder=6, ls="none")
    bar = fig.colorbar(learned_mesh, ax=learned_ax, fraction=0.035, pad=0.01)
    bar.set_label("m/s", fontsize=10)
    bar.ax.tick_params(labelsize=9, colors="#bbbbbb")
    small_tracks = [ax.plot([], [], color=TRACK, lw=1.3, alpha=0.9, zorder=5)[0] for ax in (error_ax, learned_ax)]
    small_boats = [ax.plot([], [], marker="o", ms=5, color="white", zorder=6, ls="none")[0] for ax in (error_ax, learned_ax)]

    arm = run.name.split(",")[0].removeprefix("arm=")
    fig.suptitle(
        f"{arm}: a sailboat on the Gulf Stream, {problem['frame_hour'] / 24 + 1:.0f} June 2019 held still",
        fontsize=16,
        color="white",
        y=0.97,
    )

    def arrow_to(name: str, origin: np.ndarray, velocity: np.ndarray) -> None:
        """Point the named arrow from degrees `origin` (lat, lon) along `velocity` (east, north) m/s"""
        tip = origin + ARROW_KM_PER_MS * velocity[::-1] / km_per_degree
        arrows[name].xy = (tip[1], tip[0])
        arrows[name].set_position((origin[1], origin[0]))

    def draw(t: int, s: float) -> None:
        """Frame at move `t` plus fraction `s` of the next"""
        here = track[t] if t >= last else track[t] + s * (track[t + 1] - track[t])
        m = min(t, track.shape[0] - 2)  # the move under way
        bearing = np.degrees(np.arctan2(heading[m, 0], heading[m, 1]))  # clockwise from north
        boat.set_marker((3, 0, -bearing))  # a marker angle turns counter-clockwise, pointing up at 0
        boat.set_data([here[1]], [here[0]])
        glow.set_data([here[1]], [here[0]])
        for name, velocity in (("wind", wind[m]), ("current", current[m]), ("heading", through[m]), ("net", net[m])):
            arrow_to(name, here, velocity)
        for k, seg in enumerate(trail):
            a = t - TRAIL + k
            if a < 0:
                seg.set_data([], [])
                continue
            b = np.vstack([track[a : a + 2][:1], here[None]]) if a + 1 > t else track[a : a + 2]
            seg.set_data(b[:, 1], b[:, 0])
            seg.set_alpha(0.12 + 0.88 * (k + 1) / TRAIL)
        older = track[: max(t - TRAIL + 1, 0) + 1]
        done.set_data(older[:, 1], older[:, 0])
        if waypoints is not None:
            w = int(waypoints[t])
            aim.set_data([lon[w % n_lon]], [lat[w // n_lon]])
        mean_t, _ = claimed(claims[t])
        learned_mesh.set_array(on_grid(np.linalg.norm(mean_t - truth, axis=-1)).ravel())
        for line, dot in zip(small_tracks, small_boats):
            line.set_data(np.append(track[: t + 1, 1], here[1]), np.append(track[: t + 1, 0], here[0]))
            dot.set_data([here[1]], [here[0]])
        # the polar at this wind
        speeds = np.asarray(polar.speed(jnp.asarray(unit, jnp.float32), jnp.broadcast_to(jnp.asarray(wind[m], jnp.float32), unit.shape)))
        rose_curve.set_data(compass, speeds)
        rose_fill[0].remove()
        rose_fill[0] = rose.fill_between(compass, 0, speeds, color=HEADING, alpha=0.12)
        rose_heading.set_data([np.radians(bearing)], [np.linalg.norm(through[m])])
        wind_bearing = np.arctan2(wind[m, 0], wind[m, 1])
        rose_wind.set_data(
            [wind_bearing, wind_bearing],
            [0, problem["top_speed_ms"] * 1.05 * (1 - np.exp(-np.linalg.norm(wind[m]) / problem["wind_scale_ms"]))],
        )
        day, hour = divmod(t * step_hours, 24)
        caption.set_text(
            f"move {t:3d}   day {int(day)} {int(hour):02d}:00\n"
            f"{np.linalg.norm((here - target) * km_per_degree):6.0f} km to target\n"
            f"wind     {np.linalg.norm(wind[m]):4.1f} m/s\n"
            f"current  {np.linalg.norm(current[m]):4.1f} m/s\n"
            f"boat     {np.linalg.norm(through[m]):4.1f} m/s\n"
            f"net      {np.linalg.norm(net[m]):4.1f} m/s"
        )
        if arrived[t]:
            badge.set_text("arrived")
            badge.set_alpha(min(1.0, 0.3 * (t - int(np.argmax(arrived)) + 1)))

    if still is not None:
        draw(still, 0.0)
        fig.savefig(out.with_suffix(".png"), dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        return print(out.with_suffix(".png"))
    writer = FFMpegWriter(fps=fps, bitrate=6000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(out), dpi=dpi):
        for t in range(last + 1):
            for k in range(SUBFRAMES if t < last else 2 * SUBFRAMES):
                draw(t, k / SUBFRAMES)
                writer.grab_frame(facecolor=fig.get_facecolor())
    plt.close(fig)
    print(out, f"{last} moves, {SUBFRAMES} frames each at {fps} fps")
