"""One ocean episode as a film: the boat on the true ocean current, what it believes, and how far it still is from the target

Top: ||W|| shaded with its streamlines, the wind as a quiver, land masked, the track fading behind the boat, the boat as a
triangle pointing along its heading, at the boat the wind, the ocean current, the heading through the water and the net motion as
arrows, the target with its arrival ring, the waypoint aimed at and the walk imagined to it; beside it the numbers of the move.
Bottom: the belief's mean current speed ||mu_t|| on the truth's scale, its current spread (var_u + var_v)^(1/2) on the prior's, the polar, boat speed
against heading for the wind at the boat with the heading marked, and the belief's wind spread on its prior's. A move is spread over
`subframes` frames; the film ends a few moves after arrival.

The heading is not recorded: it is read back as the net motion of the move less the ocean current at the point it left.
"""

import json
import multiprocessing
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display on a compute node
from matplotlib.artist import Artist  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.axis import Axis  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse, Polygon  # noqa: E402
from matplotlib.projections.polar import PolarAxes  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from zermelo.experiments.ocean_waypoint.metrics.data import belief_groups, channels, claimed  # noqa: E402
from zermelo.experiments.ocean_waypoint.schema import Resources  # noqa: E402
from zermelo.problems.ocean import KM_PER_DEGREE, OCEAN_CURRENT, PlaneGrid, load_ocean  # noqa: E402
from zermelo.problems.ocean import WIND as WIND_COMPONENTS  # noqa: E402
from zermelo.problems.ocean.transition import Polar  # noqa: E402

AFTER_ARRIVAL = 4
"""Moves shown once arrived"""

TRAIL = 30
"""Moves the trail fades over"""

ARROW_KM_PER_MS = 70.0
"""km of arrow per m/s of velocity, all four arrows at the boat"""

WIND, CURRENT, HEADING, NET, TRACK, TARGET, FASTEST = "#ffd54f", "#4fc3f7", "#ffffff", "#ff7043", "#4fc3f7", "#ff5fa2", "#ffd700"

RESOURCES_PER_FILM = Resources(
    cpus=8, mem_gb=8, timeout_min=90, gres=None, partition="default_partition", constraint="avx2", array_parallelism=8
)
"""What one film is given"""

SCRIPT = """#!/bin/bash
#SBATCH --job-name=zermelo-film
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_gb}G
#SBATCH --time={timeout_min}
#SBATCH --constraint={constraint}
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
            constraint=RESOURCES_PER_FILM.constraint,
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


def candidate_box(problem: dict, lat: np.ndarray, lon: np.ndarray, land: np.ndarray, targets: np.ndarray | None) -> np.ndarray:
    """(n_lat * n_lon,): the grid points a claim is scored on: the `targets` (n_targets, 2) of a visiting episode as (lat, lon), else the
    target box when the episode learns or searches the region and the whole box otherwise"""
    grid = PlaneGrid(len(lat), len(lon), float(lat[0]), float(lon[0]), float(lat[1] - lat[0]), float(lon[1] - lon[0]))
    if targets is not None:
        rows, cols = np.abs(lat[:, None] - targets[:, 0]).argmin(axis=0), np.abs(lon[:, None] - targets[:, 1]).argmin(axis=0)
        return np.isin(np.arange(len(lat) * len(lon)), rows * len(lon) + cols)
    where = "target_" if problem.get("objective") in ("learn", "search") else ""  # a record from before the field is a reaching one
    inside = grid.inside(problem[f"{where}lat_min"], problem[f"{where}lat_max"], problem[f"{where}lon_min"], problem[f"{where}lon_max"])
    return np.asarray(inside) & ~land.reshape(-1)


def scene(run: Path, dpi: int, until: int | None) -> tuple[FigureCanvasAgg, Callable[[int, float], None], list[tuple[Axes, Artist]], int]:
    """The film's canvas, the call posing it at move t plus fraction s of the next, the artists a pose redraws in order, and the last move"""

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
            "objective_state/information",
            "objective_state/best_observed_magnitude",
            "objective_state/best_possible_magnitude",
            "objective_state/targets",
            "objective_state/visited",
            "agent_state/choice.target",
            "agent_state/choice.imagined_path_to_target",
        ),
    )
    raw = np.load(Path(problem["data_path"]))
    stride = int(problem["grid_stride"])
    lat, lon, land = raw["latitude"][::stride].astype(float), raw["longitude"][::stride].astype(float), raw["land"][::stride, ::stride]
    n_lat, n_lon = len(lat), len(lon)
    ocean = load_ocean(Path(problem["data_path"]), stride)
    field = np.asarray(ocean.field(ocean.frame(problem["frame_hour"])), float).reshape(n_lat, n_lon, -1)  # (n_lat, n_lon, components)
    true_u, true_v, wind_u, wind_v = (field[..., i] for i in (*OCEAN_CURRENT, *WIND_COMPONENTS))
    targets_drawn = held["objective_state/targets"][0] if "objective_state/targets" in held else None
    candidates = np.flatnonzero(candidate_box(problem, lat, lon, land, targets_drawn))
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

    true_speed = np.hypot(true_u, true_v)
    track, target = held["state/position"].astype(float), held["state/target"][0].astype(float)  # (T + 1, 2) and (2,) as (lat, lon)
    reaching = "objective_state/arrived" in held  # else the target the world drew is unread
    searching, visiting = "objective_state/best_observed_magnitude" in held, "objective_state/targets" in held
    visited = held["objective_state/visited"] if visiting else np.zeros((track.shape[0], 0))  # (T + 1, n_targets)
    arrived = held["objective_state/arrived"] if reaching else visited.all(axis=-1) if visiting else np.zeros(track.shape[0], bool)
    step_hours = float(problem["step_hours"])
    cos_mid = np.cos(np.radians(np.mean(lat)))
    km_per_degree = np.array([KM_PER_DEGREE, KM_PER_DEGREE * cos_mid])  # per degree of (lat, lon)
    distance = np.linalg.norm((track - target) * km_per_degree, axis=-1)  # (T + 1,) km
    last = min(track.shape[0] - 1, int(np.argmax(arrived)) + AFTER_ARRIVAL) if arrived.any() else track.shape[0] - 1
    last = last if until is None else min(last, until)
    claims = held["objective_state/claim"]  # (T + 1, n_candidates, 8)
    waypoints = held.get("agent_state/choice.target")
    walks = held.get("agent_state/choice.imagined_path_to_target")  # (T + 1, L) flat indices, the last repeated past the end

    # per move: the velocities at the point it left, (east, north) m/s
    net = np.diff(track, axis=0)[:, ::-1] * km_per_degree[::-1] / (3.6 * step_hours)  # (T, 2): the move as a velocity
    current = np.stack([at(true_u, true_v, p) for p in track[:-1]])  # (T, 2)
    wind = np.stack([at(wind_u, wind_v, p) for p in track[:-1]])  # (T, 2)
    through = net - current  # (T, 2): the heading's velocity through the water, read back
    heading = through / np.maximum(np.linalg.norm(through, axis=-1, keepdims=True), 1e-9)  # (T, 2) unit (east, north)

    matplotlib.rcParams.update(
        {"font.family": "serif", "font.size": 13, "text.color": "#e6e6e6", "axes.labelcolor": "#e6e6e6", "xtick.color": "#bbbbbb",
         "ytick.color": "#bbbbbb", "axes.edgecolor": "#555555", "axes.titlesize": 14}
    )  # fmt: skip
    fig = Figure(figsize=(16, 9.5), dpi=dpi, facecolor="#0b0f14")
    canvas = FigureCanvasAgg(fig)
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
    error_ax, learned_ax, wind_ax = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[1, 3])
    rose = fig.add_subplot(grid[1, 2], projection="polar", facecolor="#0b0f14")
    if not isinstance(rose, PolarAxes):
        raise TypeError("a polar subplot was asked for")
    for ax in (main, error_ax, learned_ax, wind_ax):
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
    if reaching:
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
    if searching:  # the fastest current in the box, the point the search is scored against
        fastest = candidates[np.argmax(np.linalg.norm(held["objective_state/truth"][0][:, list(OCEAN_CURRENT)], axis=-1))]
        main.plot(lon[fastest % n_lon], lat[fastest // n_lon], marker="*", ms=22, color=FASTEST, mec="white", mew=0.8, zorder=6, ls="none")
    target_stars = []
    if visiting:
        for t_lat, t_lon in held["objective_state/targets"][0]:
            main.add_patch(
                Ellipse(
                    (t_lon, t_lat),
                    2 * ring_km / km_per_degree[1],
                    2 * ring_km / km_per_degree[0],
                    fill=False,
                    ec=TARGET,
                    lw=1.4,
                    ls="--",
                    zorder=5,
                )
            )
            target_stars.append(main.plot(t_lon, t_lat, marker="*", ms=20, color="none", mec=TARGET, mew=1.6, zorder=6, ls="none")[0])
    main.plot(track[0, 1], track[0, 0], marker="o", ms=9, mfc="none", mec="white", mew=1.5, zorder=6, ls="none")
    (aim,) = main.plot([], [], marker="x", ms=11, color="#ffa726", mew=2.2, zorder=6, ls="none")
    (route,) = main.plot([], [], color="#ffa726", lw=1.8, ls="--", alpha=0.9, zorder=5.5)  # the walk imagined to the waypoint
    trail = LineCollection(
        [], color=TRACK, lw=2.8, zorder=5, capstyle="round", alpha=(0.12 + 0.88 * np.arange(1, TRAIL + 1) / TRAIL).tolist()
    )
    main.add_collection(trail, autolim=False)
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
            *([Line2D([], [], marker="*", ms=14, color=TARGET, mec="white", ls="none", label="target, 20 km ring")] if reaching else []),
            *([Line2D([], [], marker="*", ms=14, color=FASTEST, mec="white", ls="none", label="fastest current in the box")] if searching else []),
            *([Line2D([], [], marker="*", ms=14, color="none", mec=TARGET, mew=1.6, ls="none", label="target, filled once visited")] if visiting else []),
            Line2D([], [], marker="x", ms=10, color="#ffa726", mew=2, ls="none", label="waypoint aimed at"),
            Line2D([], [], color="#ffa726", lw=1.8, ls="--", label="walk imagined to it"),
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
    rose_fill = Polygon(np.zeros((1, 2)), color=HEADING, alpha=0.12)
    rose.add_patch(rose_fill)
    winds = jnp.asarray(wind, jnp.float32)[:, None].repeat(unit.shape[0], axis=1)  # (T, n_bearings, 2)
    rose_speeds = np.asarray(polar.speed(jnp.broadcast_to(jnp.asarray(unit, jnp.float32), winds.shape), winds))  # (T, n_bearings)
    (rose_heading,) = rose.plot([], [], marker="o", ms=9, color=NET, mec="white", ls="none", zorder=5)
    (rose_wind,) = rose.plot([], [], color=WIND, lw=3)

    # bottom: what the belief holds so far, its mean on the truth's scale and its spread on the prior's
    (_, current_amplitude), (_, wind_amplitude) = belief_groups(config)  # the ocean current's Gaussian process, then the wind's
    prior_spread = float(np.sqrt(2.0)) * current_amplitude  # (var_u + var_v)^(1/2) before any reading
    wind_prior_spread = float(np.sqrt(2.0)) * wind_amplitude
    blank, half_lon, half_lat = on_grid(np.zeros(candidates.size)), (lon[1] - lon[0]) / 2, (lat[1] - lat[0]) / 2
    cells = dict(origin="lower", extent=(lon[0] - half_lon, lon[-1] + half_lon, lat[0] - half_lat, lat[-1] + half_lat), aspect=1 / cos_mid,
                 interpolation="nearest", zorder=3.5)  # above land, which no candidate lies on  # fmt: skip
    error_mesh = error_ax.imshow(blank, cmap="YlGnBu_r", vmin=0, vmax=np.nanmax(true_speed), **cells)
    learned_mesh = learned_ax.imshow(blank, cmap="viridis", vmin=0, vmax=prior_spread, **cells)
    wind_mesh = wind_ax.imshow(blank, cmap="viridis", vmin=0, vmax=wind_prior_spread, **cells)
    for ax, mesh, words, top in (
        (error_ax, error_mesh, "belief's mean current speed  ||mu||  (m/s)", None),
        (learned_ax, learned_mesh, "belief's current spread  (var_u + var_v)^(1/2)  (m/s)", prior_spread),
        (wind_ax, wind_mesh, "belief's wind spread  (m/s)", wind_prior_spread),
    ):
        ax.pcolormesh(lon, lat, land_sheet, shading="nearest", cmap=land_cmap, vmin=0, vmax=1, zorder=3)
        ax.set_title(words, fontsize=12, pad=6)
        if reaching:
            ax.plot(target[1], target[0], marker="*", ms=12, color=TARGET, mec="white", mew=0.5, zorder=6, ls="none")
        bar = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.06, pad=0.22, shrink=0.85)
        bar.ax.tick_params(labelsize=9, colors="#bbbbbb")
        if top is not None:  # a spread is drawn on its prior's scale
            bar.set_ticks([0, top / 2, top])
            bar.set_ticklabels(["0", f"{top / 2:.2f}", f"{top:.2f} prior"])
    small_tracks = [ax.plot([], [], color=TRACK, lw=1.3, alpha=0.9, zorder=5)[0] for ax in (error_ax, learned_ax, wind_ax)]
    small_boats = [ax.plot([], [], marker="o", ms=5, color="white", zorder=6, ls="none")[0] for ax in (error_ax, learned_ax, wind_ax)]

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

    posed = -1

    def draw(t: int, s: float) -> None:
        """Frame at move `t` plus fraction `s` of the next"""
        nonlocal posed
        here = track[t] if t >= last else track[t] + s * (track[t + 1] - track[t])
        m = min(t, track.shape[0] - 2)  # the move under way
        bearing = np.degrees(np.arctan2(heading[m, 0], heading[m, 1]))  # clockwise from north
        boat.set_marker((3, 0, -bearing))  # a marker angle turns counter-clockwise, pointing up at 0
        boat.set_data([here[1]], [here[0]])
        glow.set_data([here[1]], [here[0]])
        for name, velocity in (("wind", wind[m]), ("current", current[m]), ("heading", through[m]), ("net", net[m])):
            arrow_to(name, here, velocity)
        ends = np.arange(t - TRAIL, t)[:, None] + [0, 1]  # (TRAIL, 2): each trail segment's two moves, nan before move 0
        trail.set_segments(list(np.where((ends >= 0)[..., None], track[ends.clip(0)], np.nan)[..., ::-1]))  # (TRAIL, 2, 2) as (lon, lat)
        older = track[: max(t - TRAIL + 1, 0) + 1]
        done.set_data(older[:, 1], older[:, 0])
        if waypoints is not None:
            w = int(waypoints[t])
            aim.set_data([lon[w % n_lon]], [lat[w // n_lon]])
        if t != posed:
            posed = t
            mean_t, spread_t = claimed(claims[t], OCEAN_CURRENT)
            error_mesh.set_data(on_grid(np.linalg.norm(mean_t, axis=-1)))
            learned_mesh.set_data(on_grid(spread_t))
            wind_mesh.set_data(on_grid(claimed(claims[t], WIND_COMPONENTS)[1]))
        if walks is not None:
            steps = walks[t]
            kept = np.r_[True, steps[1:] != steps[:-1]]  # the walk ends where its index stops changing
            path = steps[kept]
            route.set_data(np.r_[here[1], lon[path % n_lon]], np.r_[here[0], lat[path // n_lon]]) if path.size > 1 else route.set_data(
                [], []
            )
        for line, dot in zip(small_tracks, small_boats):
            line.set_data(np.append(track[: t + 1, 1], here[1]), np.append(track[: t + 1, 0], here[0]))
            dot.set_data([here[1]], [here[0]])
        # the polar at this wind
        rose_curve.set_data(compass, rose_speeds[m])
        rose_fill.set_xy(np.c_[np.r_[compass, compass[::-1]], np.r_[rose_speeds[m], np.zeros(compass.size)]])
        rose_heading.set_data([np.radians(bearing)], [np.linalg.norm(through[m])])
        wind_bearing = np.arctan2(wind[m, 0], wind[m, 1])
        rose_wind.set_data(
            [wind_bearing, wind_bearing],
            [0, problem["top_speed_ms"] * 1.05 * (1 - np.exp(-np.linalg.norm(wind[m]) / problem["wind_scale_ms"]))],
        )
        day, hour = divmod(t * step_hours, 24)
        caption.set_text(
            f"move {t:3d}   day {int(day)} {int(hour):02d}:00\n"
            + (
                f"{np.linalg.norm((here - target) * km_per_degree):6.0f} km to target\n"
                if reaching
                else f"{held['objective_state/information'][t]:6.0f} nats learned\n"
                if "objective_state/information" in held
                else f"best {held['objective_state/best_observed_magnitude'][t]:4.2f} of {held['objective_state/best_possible_magnitude'][t]:4.2f} m/s\n"
                if searching
                else f"{int(visited[t].sum())} of {visited.shape[1]} targets visited\n"
            )
            + f"wind     {np.linalg.norm(wind[m]):4.1f} m/s\n"
            f"current  {np.linalg.norm(current[m]):4.1f} m/s\n"
            f"boat     {np.linalg.norm(through[m]):4.1f} m/s\n"
            f"net      {np.linalg.norm(net[m]):4.1f} m/s"
        )
        for star, seen in zip(target_stars, visited[t], strict=True):
            star.set_color(TARGET if seen else "none")
        if arrived[t]:
            badge.set_text("all visited" if visiting else "arrived")
            badge.set_alpha(min(1.0, 0.3 * (t - int(np.argmax(arrived)) + 1)))

    moving: list[Artist] = [aim, route, trail, done, glow, boat, *arrows.values(), caption, badge, *target_stars, rose_curve, rose_fill,
                            rose_heading, rose_wind, error_mesh, learned_mesh, wind_mesh, *small_tracks, *small_boats]  # fmt: skip
    redrawn = []  # per axes: every artist from the lowest moving one up, except ticks and unchanging text
    for ax in fig.axes:
        mine = [a for a in moving if a.axes is ax]
        low = min((a.get_zorder() for a in mine), default=np.inf)
        above = [
            a
            for a in ax.get_children()
            if any(a is b for b in mine) or a.get_zorder() >= low and a is not ax.patch and not isinstance(a, Axis | Text)
        ]
        redrawn += [(ax, a) for a in sorted(above, key=Artist.get_zorder)]
    for _, a in redrawn:
        a.set_animated(True)
    return canvas, draw, redrawn, last


def reel(part: int, out: Path, *, run: Path, parts: int, fps: int, dpi: int, subframes: int, until: int | None) -> int:
    """Part `part` of `parts` of the film written to `out` as video, the background drawn once; the last move of the whole film"""
    canvas, pose, redrawn, last = scene(run, dpi, until)
    frames = [(t, k / subframes) for t in range(last + 1) for k in range(subframes if t < last else 2 * subframes)]
    width, height = canvas.get_width_height()
    canvas.draw()
    background = canvas.copy_from_bbox(canvas.figure.bbox)
    encode = subprocess.Popen(  # noqa: S603
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast", "-b:v", "6000k", "-pix_fmt", "yuv420p", str(out)],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    if encode.stdin is None:
        raise RuntimeError("ffmpeg opened without a pipe")
    for t, s in frames[part * len(frames) // parts : (part + 1) * len(frames) // parts]:
        pose(t, s)
        canvas.restore_region(background)
        for ax, artist in redrawn:
            ax.draw_artist(artist)
        encode.stdin.write(canvas.buffer_rgba())
    encode.stdin.close()
    if encode.wait():
        raise RuntimeError(f"ffmpeg failed writing {out}")
    return last


def film(run: Path, out: Path, *, fps: int, dpi: int, subframes: int, workers: int, until: int | None, still: int | None) -> None:
    """The episode recorded at `run` written to `out` as video over `workers` processes, or the frame at move `still` as a png beside it"""
    if still is not None:
        canvas, pose, _, _ = scene(run, dpi, until)
        pose(still, 0.0)
        canvas.figure.savefig(out.with_suffix(".png"), facecolor=canvas.figure.get_facecolor())
        return print(out.with_suffix(".png"))
    shoot = partial(reel, run=run, parts=workers, fps=fps, dpi=dpi, subframes=subframes, until=until)
    if workers == 1:
        last = shoot(0, out)
    else:
        with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
            pieces = [Path(scratch) / f"{k}.mp4" for k in range(workers)]
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:  # a fork after jax starts hangs
                last = max(pool.map(shoot, range(workers), pieces))
            listed = Path(scratch) / "pieces.txt"
            listed.write_text("".join(f"file '{piece}'\n" for piece in pieces))
            subprocess.run(  # noqa: S603
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listed), "-c", "copy", str(out)], check=True
            )
    print(out, f"{last} moves, {subframes} frames each at {fps} fps")
