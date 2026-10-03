"""The globe a film's main panel is: wind speed shaded on an orthographic sphere, particles carried by the wind, and the episode's marks over it

A frame is array arithmetic over one basemap raster: the particle buffer fades, the particles advect, streak into it, and the buffer is
laid over the basemap as glowing wind.
"""

import dataclasses
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.path import Path

from zermelo.experiments.balloon_waypoint.render.replay import Replay

EARTH_RADIUS_KM = 6371.0

SPACE = "#05070d"
"""The background the globe sits on"""

OCEAN = "#0b1a2b"
LAND = "#16202c"
COASTLINE = "#8aa0b6"
LIMB = "#5ea8d8"
BOX = "#ffffff"
TRACK = "#ffd9a0"
ROUTE = "#19d3f3"
WAYPOINT = "#ff4f6d"
FASTEST = "#ffffff"
GLOW = (255, 252, 240)
"""The tint a particle streak lights the globe with"""

WIND = LinearSegmentedColormap.from_list(
    "wind",
    [
        (0.00, "#05070d"),
        (0.10, "#0a1430"),
        (0.24, "#152a63"),
        (0.40, "#1d5c96"),
        (0.55, "#27a0a8"),
        (0.68, "#5fd08a"),
        (0.79, "#d8de5c"),
        (0.89, "#f59a3c"),
        (0.96, "#f2582a"),
        (1.00, "#fff0cc"),
    ],
)
"""Calm reads as the colour of space, the jets as a glowing cream through orange"""

SPEED_GAMMA = 0.6
"""Exponent the speed scale is raised through: speed ** gamma lifts the slow winds out of the dark"""

TRAIL_DECAY = 0.94
"""What the particle buffer is multiplied by each frame: 0.94 ** 50 ~ 0.05, so a streak lives 50 frames"""

PARTICLE_COUNT = 13000
PARTICLE_LIFE = 50
"""Frames a particle is followed before it is sown somewhere else"""

STARS = 700
LIMB_RINGS = 26

ALTITUDE_COLOURS = ("#4aa3d8", "#5fd08a", "#d8de5c", "#f59a3c", "#ff5f5f")
"""One colour per altitude, low to high: the balloon's marker reads its altitude"""

TRACK_SHOWN = 56
"""Moves of the balloon's past drawn at full brightness behind it; the rest fades"""


@dataclasses.dataclass(frozen=True)
class Camera:
    """Where the globe is seen from, and how many pixels square it is drawn at"""

    size: int
    central_longitude: float
    central_latitude: float

    @property
    def projection(self) -> ccrs.Orthographic:
        return ccrs.Orthographic(central_longitude=self.central_longitude, central_latitude=self.central_latitude)

    def to_pixels(self, latitude: np.ndarray, longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Points on the sphere as pixel coordinates, the far side of the globe NaN"""
        projected = self.projection.transform_points(ccrs.PlateCarree(), np.asarray(longitude, float), np.asarray(latitude, float))
        limit = self.projection.x_limits[1]
        return (projected[..., 0] / limit + 1.0) * 0.5 * self.size, (1.0 - (projected[..., 1] / limit + 1.0) * 0.5) * self.size


def basemap(camera: Camera, replay: Replay, frame: int, level: int) -> np.ndarray:
    """The globe shaded by ||W|| at one frame and altitude, with stars, land, coastlines and a limb, as (size, size, 3) bytes"""
    figure = plt.figure(figsize=(camera.size / 100.0, camera.size / 100.0), dpi=100)
    figure.patch.set_facecolor(SPACE)
    axes = GeoAxes(figure, (0.0, 0.0, 1.0, 1.0), projection=camera.projection)
    figure.add_axes(axes)
    axes.set_global()
    axes.patch.set_facecolor(SPACE)
    rng = np.random.default_rng(7)
    axes.scatter(
        rng.uniform(0.0, 1.0, STARS),
        rng.uniform(0.0, 1.0, STARS),
        s=rng.uniform(0.05, 1.4, STARS),
        c=np.column_stack([np.ones((STARS, 3)), rng.uniform(0.15, 0.75, STARS)]),  # white at a random alpha each
        transform=axes.transAxes,
        zorder=0,
        linewidths=0,
    )
    axes.add_feature(cfeature.OCEAN, facecolor=OCEAN, zorder=1)
    axes.add_feature(cfeature.LAND, facecolor=LAND, zorder=1)
    speed = replay.speed[frame, level]  # (lat, lon)
    axes.pcolormesh(
        np.append(replay.longitude, replay.longitude[0] + 360.0),  # the first column repeated past the last closes the seam
        replay.latitude,
        np.concatenate([speed, speed[:, :1]], axis=1),
        transform=ccrs.PlateCarree(),
        cmap=WIND,
        norm=PowerNorm(SPEED_GAMMA, vmin=0.0, vmax=float(np.max(replay.speed))),
        shading="gouraud",
        alpha=0.88,
        zorder=2,
    )
    axes.add_feature(cfeature.COASTLINE, edgecolor=COASTLINE, linewidth=0.5, alpha=0.5, zorder=3)
    limit = camera.projection.x_limits[1]
    angle = np.linspace(0, 2 * np.pi, 256)
    for ring in range(LIMB_RINGS):  # thinning atmosphere just outside the edge
        out = 1.0 + 0.055 * (ring + 1) / LIMB_RINGS
        axes.plot(
            limit * out * np.cos(angle),
            limit * out * np.sin(angle),
            color=LIMB,
            alpha=0.16 * (1.0 - ring / LIMB_RINGS) ** 2,
            lw=2.2,
            zorder=5,
        )
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    picture = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(figure)
    return picture


def _cartesian(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """(n, 3) unit vectors"""
    lat, lon = np.radians(latitude), np.radians(longitude)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def advance(latitude: np.ndarray, longitude: np.ndarray, u: np.ndarray, v: np.ndarray, step_hours: float) -> tuple[np.ndarray, np.ndarray]:
    """One step along the great circle the wind points down:

    angle = ||(u, v)|| h / R,    p' = p cos(angle) + d sin(angle),    d the unit tangent along (u, v)
    """
    lat, lon = np.radians(latitude), np.radians(longitude)
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=-1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], axis=-1)
    velocity = u[:, None] * east + v[:, None] * north  # (n, 3) m/s
    pace = np.linalg.norm(velocity, axis=-1, keepdims=True)
    angle = pace * step_hours * 3.6 / EARTH_RADIUS_KM  # radians of great circle covered
    direction = np.divide(velocity, pace, out=np.zeros_like(velocity), where=pace > 0.0)
    moved = _cartesian(latitude, longitude) * np.cos(angle) + direction * np.sin(angle)
    return np.degrees(np.arcsin(np.clip(moved[:, 2], -1.0, 1.0))), np.degrees(np.arctan2(moved[:, 1], moved[:, 0]))


def wind_at(replay: Replay, frame: int, level: int, latitude: np.ndarray, longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) at each point, read from the nearest grid cell"""
    row = np.abs(replay.latitude[None, :] - latitude[:, None]).argmin(axis=1)
    gap = ((replay.longitude[None, :] - longitude[:, None] + 180.0) % 360.0) - 180.0
    col = np.abs(gap).argmin(axis=1)
    return replay.wind[frame, level, row, col, 0], replay.wind[frame, level, row, col, 1]


def _even_sphere(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Points spread evenly over the sphere"""
    return np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, count))), rng.uniform(-180.0, 180.0, count)


def splat(buffer: np.ndarray, x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> None:
    """Each weight added into the buffer, bilinearly over the four pixels it sits between"""
    size = buffer.shape[0]
    inside = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < size - 1) & (y >= 0) & (y < size - 1)
    if not inside.any():
        return
    x, y, weight = x[inside], y[inside], weight[inside]
    left, top = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - left, y - top
    flat = buffer.reshape(-1)
    for dx, dy, share in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        flat += np.bincount((top + dy) * size + (left + dx), weights=share * weight, minlength=flat.size)


@dataclasses.dataclass
class Scene:
    """The globe's moving parts across a film: the particles, the buffer their streaks live in, and the camera"""

    camera: Camera
    latitude: np.ndarray
    longitude: np.ndarray
    age: np.ndarray
    buffer: np.ndarray
    rng: np.random.Generator
    last_hours: float
    """hours_elapsed of the last frame drawn, NaN before the first"""

    @classmethod
    def sown(cls, camera: Camera, seed: int = 1) -> "Scene":
        """Particles spread over the sphere with staggered ages, over an empty buffer"""
        rng = np.random.default_rng(seed)
        latitude, longitude = _even_sphere(rng, PARTICLE_COUNT)
        return cls(
            camera,
            latitude,
            longitude,
            rng.integers(0, PARTICLE_LIFE, PARTICLE_COUNT),
            np.zeros((camera.size, camera.size), np.float32),
            rng,
            float("nan"),
        )

    def frame(self, replay: Replay, move: int) -> np.ndarray:
        """The globe at `move`: the particles carried the hours since the last frame at the altitude flown, streaked over the basemap"""
        frame, level = replay.frame_at(move), int(replay.flown[min(move, replay.flown.size - 1)])
        hours = float(replay.hours[min(move, replay.hours.size - 1)])
        step_hours = (
            float(replay.hours[1] - replay.hours[0])
            if np.isnan(self.last_hours)
            else max(hours - self.last_hours, float(replay.hours[1] - replay.hours[0]))
        )
        self.last_hours = hours
        was_lat, was_lon = self.latitude.copy(), self.longitude.copy()
        u, v = wind_at(replay, frame, level, self.latitude, self.longitude)
        self.latitude, self.longitude = advance(self.latitude, self.longitude, u, v, step_hours)
        self.age += 1
        spent = self.age >= PARTICLE_LIFE
        if spent.any():  # sown again somewhere fresh, with no streak from the old home
            lat, lon = _even_sphere(self.rng, int(spent.sum()))
            self.latitude[spent], self.longitude[spent], self.age[spent] = lat, lon, 0
            was_lat[spent], was_lon[spent] = lat, lon
        self.buffer *= TRAIL_DECAY
        pace = np.hypot(*wind_at(replay, frame, level, self.latitude, self.longitude))
        bright = (0.10 + 0.95 * np.clip(pace / float(np.max(replay.speed)), 0.0, 1.0) ** 0.8).astype(
            np.float32
        )  # jets glow, calms barely show
        (x0, y0), (x1, y1) = self.camera.to_pixels(was_lat, was_lon), self.camera.to_pixels(self.latitude, self.longitude)
        for at in np.linspace(0.0, 1.0, 3):  # three splats along the step: a fast particle draws a streak, not a dot
            splat(self.buffer, x0 + (x1 - x0) * at, y0 + (y1 - y0) * at, bright / 3)
        alpha = np.clip(self.buffer, 0.0, 1.0)[:, :, None]
        return (basemap(self.camera, replay, frame, level) * (1.0 - alpha) + np.asarray(GLOW, np.float32) * alpha).astype(np.uint8)


def balloon_path() -> Path:
    """A balloon as a marker: a round envelope over a small payload"""
    angle = np.linspace(0.0, 2.0 * np.pi, 48)
    envelope = np.stack([0.62 * np.sin(angle), 0.30 + 0.72 * np.cos(angle)], axis=1)
    neck = np.array([[-0.18, -0.42], [0.18, -0.42]])
    payload = np.array([[-0.26, -0.55], [0.26, -0.55], [0.26, -0.95], [-0.26, -0.55], [-0.26, -0.55]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(envelope) - 1) + [Path.MOVETO, Path.LINETO] + [Path.MOVETO] + [Path.LINETO] * 4
    return Path(np.concatenate([envelope, neck, payload]), codes)


def _line(ax: Axes, camera: Camera, latitude: np.ndarray, longitude: np.ndarray, **style: Any) -> None:
    """A path on the sphere drawn in pixels, broken where it passes behind the globe or jumps across it"""
    x, y = camera.to_pixels(latitude, longitude)
    jump = np.abs(np.diff(x, prepend=x[:1])) > camera.size * 0.5
    ax.plot(np.where(jump, np.nan, x), np.where(jump, np.nan, y), **style)


def marks(ax: Axes, camera: Camera, replay: Replay, move: int) -> None:
    """The episode over the globe at `move`: the candidate box, the track, the balloon, the waypoint and its route, and the fastest wind

    `ax` spans the globe image in pixel coordinates, y down.
    """
    lat_min, lat_max, lon_min, lon_max = replay.box
    edge = np.linspace(0.0, 1.0, 64)
    outline_lat = np.concatenate(
        [np.full(64, lat_min), lat_min + (lat_max - lat_min) * edge, np.full(64, lat_max), lat_max - (lat_max - lat_min) * edge]
    )
    outline_lon = np.concatenate(
        [lon_min + (lon_max - lon_min) * edge, np.full(64, lon_max), lon_max - (lon_max - lon_min) * edge, np.full(64, lon_min)]
    )
    _line(ax, camera, outline_lat, outline_lon, color=BOX, lw=1.4, alpha=0.8, zorder=4)
    frame, level = replay.frame_at(move), int(replay.flown[min(move, replay.flown.size - 1)])
    inside = (
        (replay.latitude[:, None] >= lat_min)
        & (replay.latitude[:, None] <= lat_max)
        & (replay.longitude[None, :] >= lon_min)
        & (replay.longitude[None, :] <= lon_max)
    )
    boxed = np.where(inside, replay.speed[frame, level], -np.inf)
    row, col = np.unravel_index(int(np.argmax(boxed)), boxed.shape)
    sx, sy = camera.to_pixels(np.asarray([replay.latitude[row]]), np.asarray([replay.longitude[col]]))
    if np.isfinite(sx[0]):
        ax.scatter(sx[0], sy[0], marker="*", s=260, facecolor="none", edgecolor=FASTEST, linewidths=1.3, alpha=0.9, zorder=7)
    walked = replay.path[: move + 1]
    if walked.shape[0] > 1:
        old = walked[: max(1, walked.shape[0] - TRACK_SHOWN) + 1]
        _line(ax, camera, old[:, 0], old[:, 1], color=TRACK, lw=1.4, alpha=0.35, solid_capstyle="round", zorder=5)
        recent = walked[-TRACK_SHOWN - 1 :]
        _line(ax, camera, recent[:, 0], recent[:, 1], color=TRACK, lw=3.0, alpha=0.9, solid_capstyle="round", zorder=5)
    if replay.plan is not None:
        aimed, route = replay.plan.waypoint[move], replay.plan.walk[move]
        if np.isfinite(route).all():
            _line(ax, camera, route[:, 0], route[:, 1], ls="--", lw=1.8, color=ROUTE, alpha=0.9, zorder=6)
        if np.isfinite(aimed).all():
            wx, wy = camera.to_pixels(aimed[:1], aimed[1:2])
            if np.isfinite(wx[0]):
                ax.scatter(wx[0], wy[0], marker="X", s=150, facecolor=WAYPOINT, edgecolor="#12181f", linewidths=0.8, zorder=8)
    bx, by = camera.to_pixels(walked[-1:, 0], walked[-1:, 1])
    if np.isfinite(bx[0]):
        tint = ALTITUDE_COLOURS[level]
        ax.scatter(bx[0], by[0], marker="o", s=210 * 5.0, color=tint, alpha=0.16, linewidths=0, zorder=8)
        ax.scatter(bx[0], by[0], marker=balloon_path(), s=210, facecolor=tint, edgecolor="#12181f", linewidths=0.8, zorder=9)
