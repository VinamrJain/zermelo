"""Headline figure draft, 1x2: (a) one grid episode, (b) one static balloon episode on the globe

Run from code/:  pixi run fig comparison.headline <grid run dir> <balloon run dir> [grid move] [balloon move]
Writes results/figures/main/headline.{pdf,png}. The paper's copy: ambient_waypoint `planning_budget_steps50,seed=7` at move 1590, balloon_waypoint `static_esi_L50,seed=7` at move 160
"""

import sys
from pathlib import Path

import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from figures.common import BUILD  # noqa: E402
from zermelo.experiments.ambient_waypoint.render import panels as grid_panels  # noqa: E402
from zermelo.experiments.ambient_waypoint.render import read as read_grid  # noqa: E402
from zermelo.experiments.ambient_waypoint.render.style import Style as GridStyle  # noqa: E402
from zermelo.experiments.balloon_waypoint.render import globe as sphere  # noqa: E402
from zermelo.experiments.balloon_waypoint.render import read as read_balloon  # noqa: E402

WIDTH = 6.75  # inches, the full text width
SIZE = 10.0  # points, every label
TRACK, TARGET, ROUTE, INK, STAR = "#ff4fa6", "#e8112d", "#19d3f3", "#12161b", "#ffd23f"
TAIL = 40  # moves of the walk drawn behind the actor
STREAK_FRAMES = 45  # frames the particles are carried before the globe is drawn, so the streaks are full length
ZOOM = 0.80  # fraction of the globe's diameter panel (b) shows, centred on the box
ARROW_DEGREES = 7.0  # spacing of the wind arrows on the globe

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
    ax.tick_params(labelsize=SIZE - 1, length=2, width=0.5)
    up, across = grid_panels._centres(replay)
    stride = max(1, min(replay.shape) // 12)
    field = replay.raster("truth", 0)[::stride, ::stride]
    right, high = np.meshgrid(across[::stride], up[::stride])
    ax.quiver(
        right, high, 0.95 * stride * replay.spacing * field / np.nanmax(np.abs(field)), np.zeros_like(field),
        angles="xy", scale_units="xy", scale=1.0, color="#1d2838", width=0.006, headwidth=3.6, headlength=4, pivot="mid", zorder=2,
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
    ax.set_xlabel("field-driven coordinate", labelpad=2)
    ax.set_ylabel("controlled coordinate", labelpad=2)
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
        (walked[-1], dict(s=55, color=INK, edgecolors="white", linewidths=1.0)),
    ):
        px, py = camera.to_pixels(np.asarray([at_lat]), np.asarray([at_lon]))
        ax.scatter(px, py, zorder=8, **look)
    half = ZOOM * pixels / 2
    ax.set_xlim(pixels / 2 - half, pixels / 2 + half)
    ax.set_ylim(pixels / 2 + half, pixels / 2 - half)
    ax.axis("off")
    return ScalarMappable(PowerNorm(sphere.SPEED_GAMMA, vmin=0.0, vmax=float(np.max(replay.speed))), sphere.WIND), [0, 10, 25, 50, 80]


def main() -> None:
    grid_replay, balloon_replay = read_grid(Path(sys.argv[1])), read_balloon(Path(sys.argv[2]))
    asked = [int(move) for move in sys.argv[3:5]] + [None, None]
    fig = plt.figure(figsize=(WIDTH, 3.55), dpi=300)
    left = fig.add_axes((0.075, 0.235, 0.355, 0.675))
    right = fig.add_axes((0.550, 0.235, 0.355, 0.675))
    for ax, replay, draw, name, where in (
        (left, grid_replay, grid, "posterior RMSE", 0.440),
        (right, balloon_replay, globe, r"wind speed (m/s)", 0.915),
    ):
        shaded, ticks = draw(ax, replay, planned(replay, asked.pop(0)))
        box = ax.get_position()
        bar = fig.colorbar(shaded, cax=fig.add_axes((where, box.y0 + 0.15 * box.height, 0.013, 0.70 * box.height)), ticks=ticks)
        bar.ax.tick_params(labelsize=SIZE - 1, length=2, width=0.5)
        bar.set_label(name, labelpad=3)
        bar.outline.set_linewidth(0.5)
    left.set_title("(a) 2D grid", fontsize=SIZE + 1, pad=5)
    right.set_title("(b) Balloon in ERA5 winds", fontsize=SIZE + 1, pad=5)
    fig.legend(
        handles=[
            Line2D([], [], color=TRACK, lw=2.0, label="trajectory"),
            Line2D([], [], color=ROUTE, lw=1.6, ls=(0, (3, 2)), label="planned route"),
            Line2D([], [], color=TARGET, marker="X", ls="", ms=8, mec=INK, mew=0.5, label="target"),
            Line2D([], [], color=STAR, marker="*", ls="", ms=13, mec=INK, mew=0.6, label="field maximum"),
            Line2D([], [], color=INK, marker=r"$\rightarrow$", ls="", ms=13, label="field"),
        ],
        loc="lower center", ncol=5, frameon=False, fontsize=SIZE, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.1,
    )  # fmt: skip
    out = BUILD / "main"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "headline.pdf", dpi=600)
    fig.savefig(out / "headline.png")
    print(out / "headline.png")


if __name__ == "__main__":
    main()
