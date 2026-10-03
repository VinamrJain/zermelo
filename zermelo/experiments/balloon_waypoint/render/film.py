"""One episode as a film: one sheet per move, its pixels piped straight into ffmpeg"""

import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from jaxtyping import Int
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from zermelo.experiments.balloon_waypoint.render.replay import Replay
from zermelo.experiments.balloon_waypoint.render.style import Style


def schedule(replays: Sequence[Replay], stride: int) -> Int[np.ndarray, " frames"]:
    """Every `stride`-th move up to the longest episode, with the last move and every episode's milestones kept"""
    horizon = max(replay.n_moves for replay in replays)
    paced = np.arange(0, horizon + 1, max(1, stride))
    marks = [replay.milestones() for replay in replays]
    return np.unique(np.concatenate([paced, np.asarray([horizon]), *marks]))


def _ffmpeg() -> str:
    """The ffmpeg beside the interpreter, or the one on the path"""
    beside = Path(sys.executable).parent / "ffmpeg"
    found = str(beside) if beside.exists() else shutil.which("ffmpeg")
    if found is None:
        raise RuntimeError("no ffmpeg beside the interpreter or on the path")
    return found


def _writer(into: Path, width: int, height: int, fps: int) -> subprocess.Popen[bytes]:
    """An ffmpeg encoding raw rgb24 frames from its standard input to h264"""
    command = [
        _ffmpeg(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
    ]
    command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(into)]
    return subprocess.Popen(command, stdin=subprocess.PIPE, bufsize=width * height * 6)  # noqa: S603


def film(
    fig: Figure, draw: Callable[[Figure, int], None], moves: Int[np.ndarray, " frames"], into: Path, style: Style, *, fps: int
) -> None:
    """`draw` at each of `moves`, every sheet the size of the first, written to `into` one frame at a time"""
    pipe = None
    canvas = fig.canvas
    if not isinstance(canvas, FigureCanvasAgg):
        raise TypeError(f"a film reads its pixels off an Agg canvas, and this figure draws on {type(canvas).__name__}")
    for frame, move in enumerate(moves):
        fig.clear()
        draw(fig, int(move))
        canvas.draw()
        picture = np.asarray(canvas.buffer_rgba())[:, :, :3]  # (height, width, 3)
        if pipe is None:
            height, width = picture.shape[:2]
            pipe = _writer(into, width, height, fps)
        elif picture.shape[:2] != (height, width):
            raise ValueError(f"frame {frame} is {picture.shape[1]}x{picture.shape[0]} and the film is {width}x{height}")
        assert pipe.stdin is not None
        pipe.stdin.write(np.ascontiguousarray(picture).tobytes())
        print(f"\rframe {frame + 1} of {moves.size}", end="", flush=True)
    print()
    if pipe is not None:
        assert pipe.stdin is not None
        pipe.stdin.close()
        pipe.wait()
