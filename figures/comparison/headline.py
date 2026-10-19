"""Headline figure draft, 1x3: (a) one grid episode, (b) one static balloon episode on the globe, (c) one ocean visiting episode

Run from code/:  pixi run fig comparison.headline <grid run dir> <balloon run dir> <ocean run dir> [grid move] [balloon move] [ocean move]
Writes results/figures/main/headline.{pdf,png}. The paper's copy: ambient_waypoint `planning_budget_steps50,seed=7` at move 1590,
balloon_waypoint `static_esi_L50,seed=7` at move 160, ocean_waypoint `visit_L50_g10_jun28_etr,seed=2` at move 104
"""

import json
import sys
from pathlib import Path

import cartopy.feature as cfeature
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402
from matplotlib.legend_handler import HandlerPatch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, DrawingArea  # noqa: E402
from matplotlib.patches import ConnectionPatch, Ellipse, FancyArrow, Polygon, Rectangle  # noqa: E402
from shapely.geometry import box as rectangle  # noqa: E402

from figures.common import BUILD  # noqa: E402
from zermelo.experiments.ambient_waypoint.render import panels as grid_panels  # noqa: E402
from zermelo.experiments.ambient_waypoint.render import read as read_grid  # noqa: E402
from zermelo.experiments.ambient_waypoint.render.style import Style as GridStyle  # noqa: E402
from zermelo.experiments.balloon_waypoint.render import globe as sphere  # noqa: E402
from zermelo.experiments.balloon_waypoint.render import read as read_balloon  # noqa: E402
from zermelo.experiments.ocean_waypoint.metrics.data import channels  # noqa: E402
from zermelo.problems.ocean import KM_PER_DEGREE, OCEAN_CURRENT, load_ocean  # noqa: E402
from zermelo.problems.ocean import WIND as WIND_PARTS  # noqa: E402
from zermelo.problems.ocean.transition import Polar  # noqa: E402

WIDTH = 6.75  # inches, the full text width
SIZE = 10.0  # points, every label
TRACK, TARGET, ROUTE, INK, STAR = "#ff4fa6", "#e8112d", "#19d3f3", "#12161b", "#ffd23f"
TAIL = 40  # moves of the walk drawn behind the actor
STREAK_FRAMES = 45  # frames the particles are carried before the globe is drawn, so the streaks are full length
ZOOM = 0.80  # fraction of the globe's diameter panel (b) shows, centred on the box
ARROW_DEGREES = 7.0  # spacing of the wind arrows on the globe
SHORE, COAST, SEA = "#efe6d2", "#7a6f60", "YlGnBu_r"  # land fill, coastline, ocean current colour map
BREEZE = "#ff9f43"  # the wind arrows in panel (c)
MARGIN = 0.7  # degrees panel (c) shows around the episode's track and goals

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": SIZE, "axes.linewidth": 0.6})


def planned(replay, asked: int | None) -> int:
    """`asked`, or the move nearest 60% of the episode at which a target is held"""
    if asked is not None:
        return asked
    held = np.nonzero(np.isfinite(replay.plan.waypoint).all(axis=1))[0]
    held = held[held < replay.n_moves]
    return int(held[np.abs(held - 0.6 * replay.n_moves).argmin()])


def until_arrival(route, target, near: float):
    """The route cut where it first comes within `near` of the target, or where it comes nearest if it never does"""
    gap = np.linalg.norm(route[:, :2] - target[:2], axis=1)
    return route[: int(np.argmax(gap <= near) if (gap <= near).any() else gap.argmin()) + 1]


def halo(draw, *data, lw: float, **look) -> None:
    """One line drawn twice: white under, coloured over"""
    draw(
        *data,
        color="white",
        lw=lw + 1.2,
        zorder=look.pop("zorder", 5) - 0.1,
        **{k: v for k, v in look.items() if k != "color" and k != "ls"},
    )
    draw(*data, lw=lw, zorder=5, **look)


def grid(ax, replay, move: int):
    """|mu_t - f| shaded, f as arrows of length proportional to |f|, the last TAIL moves, the route to the target, the largest |f|"""
    image = grid_panels.raster(ax, replay, "error", move, GridStyle(), cropped=True)
    ax.tick_params(labelsize=SIZE - 2, length=2, width=0.5, pad=1)
    up, across = grid_panels._centres(replay)
    stride = max(1, min(replay.shape) // 12)
    field = replay.raster("truth", 0)[::stride, ::stride]
    right, high = np.meshgrid(across[::stride], up[::stride])
    ax.quiver(
        right, high, 0.95 * stride * replay.spacing * field / np.nanmax(np.abs(field)), np.zeros_like(field),
        angles="xy", scale_units="xy", scale=1.0, color="white", edgecolor=INK, linewidth=0.4, width=0.007, headwidth=3.6, headlength=4, pivot="mid", zorder=2,
    )  # fmt: skip
    walked = replay.path[max(0, move - TAIL) : move + 1]
    halo(ax.plot, *walked.T, lw=1.8, color=TRACK, solid_capstyle="round")
    halo(
        ax.plot,
        *until_arrival(replay.plan.walk[move], replay.plan.waypoint[move], 1.5 * replay.spacing).T,
        lw=1.4,
        color=ROUTE,
        ls=(0, (3, 2)),
    )
    row, col = replay.cell[replay.scored[int(np.argmax(np.abs(replay.truth)))]]
    ax.scatter(across[col], up[row], marker="*", s=260, color=STAR, edgecolors=INK, linewidths=0.8, zorder=6)
    ax.scatter(*replay.plan.waypoint[move], marker="X", s=80, color=TARGET, edgecolors="white", linewidths=0.8, zorder=7)
    ax.scatter(*walked[-1], s=55, color=INK, edgecolors="white", linewidths=1.0, zorder=7)
    ax.set_ylabel("controlled coordinate", labelpad=1, fontsize=SIZE - 2)
    return image, [0, 3, 6, 9]


def globe(ax, replay, move: int, pixels: int = 2400):
    """Wind speed at the altitude flown shaded on the sphere, the wind as arrows, and the episode's marks over it"""
    lat_min, lat_max, lon_min, lon_max = replay.box
    camera = sphere.Camera(pixels, (lon_min + lon_max) / 2, (lat_min + lat_max) / 2)
    scene = sphere.Scene.sown(camera)
    for _ in range(STREAK_FRAMES):
        picture = scene.frame(replay, move)
    y, x = np.mgrid[:pixels, :pixels]
    off_globe = np.hypot(x - pixels / 2, y - pixels / 2) > pixels / 2 - 1
    picture = np.where(off_globe[:, :, None], 255, picture).astype(np.uint8)
    ax.imshow(picture, extent=(0, pixels, pixels, 0), interpolation="bilinear", zorder=1)

    for coast in cfeature.COASTLINE.geometries():  # the coastlines again, over the shading
        for part in getattr(coast, "geoms", [coast]):
            lon_c, lat_c = np.asarray(part.coords).T[:2]
            sphere._line(ax, camera, lat_c, lon_c, color="#e6edf5", lw=0.6, alpha=0.9, zorder=2)

    frame, level = replay.frame_at(move), int(replay.flown[move])
    lat, lon = (axis.ravel() for axis in np.meshgrid(np.arange(-80.0, 81.0, ARROW_DEGREES), np.arange(-180.0, 180.0, ARROW_DEGREES)))
    lon = lon[np.abs(lat) < 75]
    lat = lat[np.abs(lat) < 75]
    u, v = sphere.wind_at(replay, frame, level, lat, lon)
    ahead = sphere.advance(lat, lon, u, v, 1.0)  # where one hour of this wind carries each point
    (x0, y0), (x1, y1) = camera.to_pixels(lat, lon), camera.to_pixels(*ahead)
    reach = 0.9 * ARROW_DEGREES / 360.0 * np.pi * pixels / np.nanmax(np.hypot(x1 - x0, y1 - y0))  # the fastest arrow spans one spacing
    ax.quiver(
        x0, y0, (x1 - x0) * reach, (y1 - y0) * reach,
        angles="xy", scale_units="xy", scale=1.0, color="white", alpha=0.8, width=0.0045, headwidth=3.6, headlength=4, zorder=3,
    )  # fmt: skip

    edge = np.linspace(0.0, 1.0, 64)
    sphere._line(
        ax, camera,
        np.concatenate([np.full(64, lat_min), lat_min + (lat_max - lat_min) * edge, np.full(64, lat_max), lat_max - (lat_max - lat_min) * edge]),
        np.concatenate([lon_min + (lon_max - lon_min) * edge, np.full(64, lon_max), lon_max - (lon_max - lon_min) * edge, np.full(64, lon_min)]),
        color="white", lw=1.0, ls=(0, (1, 1.5)), zorder=4,
    )  # fmt: skip
    inside = (
        (replay.latitude[:, None] >= lat_min) & (replay.latitude[:, None] <= lat_max)
        & (replay.longitude[None, :] >= lon_min) & (replay.longitude[None, :] <= lon_max)
    )  # fmt: skip
    row, col = np.unravel_index(int(np.argmax(np.where(inside, replay.speed[frame, level], -np.inf))), inside.shape)
    walked = replay.path[max(0, move - TAIL) : move + 1]
    aimed = replay.plan.waypoint[move]
    route = until_arrival(replay.plan.walk[move], aimed, 1.5)
    line = lambda *data, **look: sphere._line(ax, camera, *data, **look)  # noqa: E731
    halo(line, walked[:, 0], walked[:, 1], lw=1.8, color=TRACK, solid_capstyle="round")
    halo(line, route[:, 0], route[:, 1], lw=1.4, color=ROUTE, ls=(0, (3, 2)))
    for (at_lat, at_lon), look in (
        ((replay.latitude[row], replay.longitude[col]), dict(marker="*", s=260, color=STAR, edgecolors=INK, linewidths=0.8)),
        (aimed[:2], dict(marker="X", s=80, color=TARGET, edgecolors="white", linewidths=0.8)),
    ):
        px, py = camera.to_pixels(np.asarray([at_lat]), np.asarray([at_lon]))
        ax.scatter(px, py, zorder=8, **look)
    balloon(ax, *(float(at[0]) for at in camera.to_pixels(np.asarray([walked[-1, 0]]), np.asarray([walked[-1, 1]]))))
    half = ZOOM * pixels / 2
    ax.set_xlim(pixels / 2 - half, pixels / 2 + half)
    ax.set_ylim(pixels / 2 + half, pixels / 2 - half)
    ax.axis("off")
    return ScalarMappable(PowerNorm(sphere.SPEED_GAMMA, vmin=0.0, vmax=float(np.max(replay.speed))), sphere.WIND), [0, 10, 25, 50, 80]


def balloon(ax, x: float, y: float, size: float = 12.0) -> None:
    """A balloon icon `size` points tall, its envelope centred on (x, y)"""
    icon = DrawingArea(size, size, 0, 0)
    for a, b in ((0.30, 0.42), (0.70, 0.58)):  # the lines down to the basket
        icon.add_artist(Line2D([a * size, b * size], [0.50 * size, 0.16 * size], color="white", lw=0.7))
    icon.add_artist(Ellipse((0.5 * size, 0.66 * size), 0.68 * size, 0.62 * size, fc="#d62828", ec=INK, lw=0.7))
    icon.add_artist(Rectangle((0.40 * size, 0.02 * size), 0.20 * size, 0.14 * size, fc="#c8a165", ec="white", lw=0.5))
    ax.add_artist(AnnotationBbox(icon, (x, y), frameon=False, box_alignment=(0.5, 0.66), zorder=9))


def sailboat(ax, lon: float, lat: float, size: float = 17.0) -> None:
    """A sailboat icon `size` points tall, its waterline at (lon, lat)"""
    icon = DrawingArea(size, size, 0, 0)
    for corners, colour in (
        ([(0.06, 0.24), (0.94, 0.24), (0.76, 0.04), (0.24, 0.04)], INK),  # hull
        ([(0.48, 0.30), (0.48, 0.98), (0.10, 0.30)], "white"),  # mainsail
        ([(0.56, 0.30), (0.56, 0.84), (0.86, 0.30)], "white"),  # jib
    ):
        icon.add_artist(Polygon(np.array(corners) * size, closed=True, fc=colour, ec=INK, lw=0.7, joinstyle="round"))
    ax.add_artist(AnnotationBbox(icon, (lon, lat), frameon=False, box_alignment=(0.5, 0.22), zorder=9))


def sea(ax, run: Path, move: int):
    """||W|| shaded with its streamlines, land, the goals, the target, the boat and its polar at `move`"""
    problem = json.loads((run / "config.json").read_text())["problem"]
    held = channels(
        run,
        ("state/position", "objective_state/targets", "objective_state/visited", "agent_state/choice.target",
         "agent_state/choice.imagined_path_to_target"),
    )  # fmt: skip
    stride = int(problem["grid_stride"])
    raw = np.load(Path(problem["data_path"]))
    lat, lon, land = raw["latitude"][::stride].astype(float), raw["longitude"][::stride].astype(float), raw["land"][::stride, ::stride]
    ocean = load_ocean(Path(problem["data_path"]), stride)
    field = np.asarray(ocean.field(ocean.frame(problem["frame_hour"])), float).reshape(len(lat), len(lon), -1)  # (lat, lon, components)
    (u, v), wind_field = (field[..., i] for i in OCEAN_CURRENT), field[..., list(WIND_PARTS)]
    speed = np.where(land, np.nan, np.hypot(u, v))
    track = held["state/position"].astype(float)  # (moves + 1, 2) as (lat, lon)
    goals, visited = held["objective_state/targets"][0], held["objective_state/visited"]  # (goals, 2) as (lat, lon); (moves + 1, goals)
    finished = int(np.argmax(visited.all(axis=-1))) + 1 if visited[-1].all() else len(track)
    shown = np.r_[track[:finished], goals]
    view = (
        shown[:, 1].min() - MARGIN,
        shown[:, 1].max() + MARGIN,
        max(shown[:, 0].min() - MARGIN, lat[0]),
        min(shown[:, 0].max() + MARGIN, lat[-1]),
    )

    shaded = ax.pcolormesh(lon, lat, speed, shading="nearest", cmap=SEA, vmin=0.0, vmax=float(np.nanmax(speed)), zorder=1)
    ax.streamplot(
        np.linspace(lon[0], lon[-1], len(lon)), np.linspace(lat[0], lat[-1], len(lat)), np.nan_to_num(u), np.nan_to_num(v),
        color="white", linewidth=0.45, density=1.3, arrowsize=0.55, zorder=2,
    )  # fmt: skip
    frame = rectangle(view[0], view[2], view[1], view[3])
    for shape in cfeature.NaturalEarthFeature("physical", "land", "50m").geometries():
        for part in getattr(shape.intersection(frame), "geoms", [shape.intersection(frame)]):
            if not part.is_empty and part.geom_type == "Polygon":
                ax.add_patch(Polygon(np.asarray(part.exterior.coords), closed=True, fc=SHORE, ec=COAST, lw=0.5, zorder=3))
    ax.add_patch(
        plt.Rectangle(
            (problem["target_lon_min"], problem["target_lat_min"]), problem["target_lon_max"] - problem["target_lon_min"],
            problem["target_lat_max"] - problem["target_lat_min"], fill=False, ec="white", lw=1.0, ls=(0, (1, 1.5)), zorder=4,
        )
    )  # fmt: skip
    every = max(1, round(1.0 / (lon[1] - lon[0])))  # grid points between wind arrows, about a degree
    wind_u, wind_v = wind_field[::every, ::every, 0], wind_field[::every, ::every, 1]
    ax.quiver(
        lon[::every], lat[::every], wind_u, wind_v, color=BREEZE, scale_units="xy", angles="xy",
        scale=float(np.nanmax(np.hypot(wind_u, wind_v))) / (0.9 * every * (lon[1] - lon[0])), width=0.004, headwidth=3.6, headlength=4, zorder=2.5,
    )  # fmt: skip
    for (g_lat, g_lon), seen in zip(goals, visited[move], strict=True):
        ax.scatter(
            g_lon, g_lat, marker="*", s=95, color=STAR if seen else "none", edgecolors=STAR if not seen else INK, linewidths=0.9, zorder=6
        )
    aimed = int(held["agent_state/choice.target"][move])
    ax.scatter(lon[aimed % len(lon)], lat[aimed // len(lon)], marker="X", s=70, color=TARGET, edgecolors="white", linewidths=0.8, zorder=7)
    sailboat(ax, track[move, 1], track[move, 0])
    ax.set_xlim(*view[:2])
    ax.set_ylim(*view[2:])
    ax.set_aspect(1 / np.cos(np.radians(np.mean(lat))))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set(visible=True, linewidth=0.5)

    # the polar at the boat: speed through the water by heading under the wind there, the heading taken marked
    r, c = int(np.argmin(np.abs(lat - track[move, 0]))), int(np.argmin(np.abs(lon - track[move, 1])))
    km = np.array([KM_PER_DEGREE * np.cos(np.radians(np.mean(lat))), KM_PER_DEGREE])  # per degree of (lon, lat)
    net = (track[move + 1, ::-1] - track[move, ::-1]) * km / (3.6 * float(problem["step_hours"]))  # (east, north) m/s
    through, wind = net - np.nan_to_num([u[r, c], v[r, c]]), np.nan_to_num(wind_field[r, c])
    polar = Polar(problem["top_speed_ms"], problem["no_go_degrees"], problem["peak_degrees"], problem["rise_exponent"],
                  problem["downwind_factor"], problem["wind_scale_ms"])  # fmt: skip
    compass = np.radians(np.arange(0.0, 361.0, 3.0))  # bearings clockwise from north
    unit = np.stack([np.sin(compass), np.cos(compass)], axis=-1).astype(np.float32)
    reach = np.asarray(polar.speed(jnp.asarray(unit), jnp.broadcast_to(jnp.asarray(wind, jnp.float32), unit.shape)))
    fig = ax.figure
    ax.apply_aspect()
    boat_x = ax.transAxes.inverted().transform(ax.transData.transform((track[move, 1], track[move, 0])))[0]
    box = ax.get_position()
    side = 0.26 * box.height * fig.get_figheight() / WIDTH  # figure-width fraction of a square 26% of the panel tall
    rose = fig.add_axes(
        (box.x0 + (0.03 * box.width if boat_x > 0.5 else 0.97 * box.width - side), box.y0 + 0.04 * box.height, side, 0.26 * box.height),
        projection="polar",
    )
    rose.set_theta_zero_location("N")
    rose.set_theta_direction(-1)
    rose.patch.set_alpha(0.88)
    rose.fill(compass, reach, color=TRACK, alpha=0.25, lw=0)
    rose.plot(compass, reach, color=TRACK, lw=1.0)
    top = float(problem["top_speed_ms"])
    rose.annotate(
        "", xy=(np.arctan2(*wind), 0.95 * top), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, mutation_scale=7)
    )
    rose.scatter(np.arctan2(*through), np.linalg.norm(through), s=14, color=TARGET, edgecolors="white", linewidths=0.5, zorder=5)
    rose.set_ylim(0, top)
    rose.set_xticks([0.0], [""])
    rose.set_yticks([])
    rose.grid(lw=0.3, color="0.6")
    rose.spines["polar"].set_linewidth(0.5)
    fig.add_artist(
        ConnectionPatch(
            (0.5, 0.5),
            (track[move, 1], track[move, 0]),
            coordsA=rose.transAxes,
            coordsB=ax.transData,
            color=INK,
            lw=0.5,
            ls=(0, (2, 1.5)),
            zorder=1.5,
        )
    )
    return shaded, list(np.arange(0.0, float(np.nanmax(speed)), 0.5))


def main() -> None:
    grid_replay, balloon_replay, ocean_run = read_grid(Path(sys.argv[1])), read_balloon(Path(sys.argv[2])), Path(sys.argv[3])
    asked = [int(move) for move in sys.argv[4:7]] + [None, None, None]
    fig = plt.figure(figsize=(WIDTH, 2.6), dpi=300)
    bottom, height = 0.22, 0.68  # every panel's, figure fractions
    tall = height * 2.6 / WIDTH  # a square panel's width as a figure-width fraction
    left = fig.add_axes((0.065, bottom, tall, height))
    middle = fig.add_axes((0.065 + tall + 0.045, bottom, tall, height))
    right = fig.add_axes((0.065 + 2 * tall + 0.075, bottom, 0.995 - (0.065 + 2 * tall + 0.075), height))
    for ax, replay, draw, name, move, words in (
        (left, grid_replay, grid, "RMSE", None, "(a) 2D grid"),
        (middle, balloon_replay, globe, "wind (m/s)", None, "(b) Balloon in real winds, Sep 2017"),
        (right, ocean_run, sea, "current (m/s)", 104, "(c) Sailboat in the Gulf Stream"),
    ):
        moved = asked.pop(0)
        shaded, ticks = draw(ax, replay, moved if moved is not None else move if move is not None else planned(replay, None))
        ax.set_title(words, fontsize=SIZE - 1, pad=4)
        ax.apply_aspect()
        box = ax.get_position()
        bar = fig.colorbar(
            shaded, cax=fig.add_axes((box.x0 + 0.3 * box.width, 0.135, 0.65 * box.width, 0.022)), orientation="horizontal", ticks=ticks
        )
        bar.ax.tick_params(labelsize=SIZE - 3, length=2, width=0.5, pad=1)
        bar.ax.text(-0.03, 0.5, name, transform=bar.ax.transAxes, ha="right", va="center", fontsize=SIZE - 3)
        bar.outline.set_linewidth(0.5)
    fig.legend(
        handles=[
            Line2D([], [], color=TRACK, lw=2.0, label="trajectory"),
            Line2D([], [], color=ROUTE, lw=1.6, ls=(0, (3, 2)), label="planned route"),
            Line2D([], [], color=TARGET, marker="X", ls="", ms=7, mec=INK, mew=0.5, label="target"),
            Line2D([], [], color=STAR, marker="*", ls="", ms=10, mec=INK, mew=0.6, label="field max / goal visited"),
            Line2D([], [], color="none", marker="*", ls="", ms=10, mec=STAR, mew=0.9, label="goal ahead"),
            FancyArrow(0, 0, 1, 0, fc="white", ec=INK, lw=0.5, label="field"),
            FancyArrow(0, 0, 1, 0, fc=BREEZE, ec=BREEZE, lw=0.5, label="wind (c)"),
        ],
        handler_map={FancyArrow: HandlerPatch(patch_func=lambda legend, orig_handle, xdescent, ydescent, width, height, fontsize: FancyArrow(
            0, height / 2, width, 0, width=0.22 * height, head_width=0.8 * height, head_length=0.45 * width, length_includes_head=True))},
        loc="lower center", ncol=7, frameon=False, fontsize=SIZE - 3, handlelength=1.6, handletextpad=0.35, columnspacing=0.9, borderaxespad=0.1,
    )  # fmt: skip
    out = BUILD / "main"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "headline.pdf", dpi=600)
    fig.savefig(out / "headline.png")
    print(out / "headline.png")


if __name__ == "__main__":
    main()
