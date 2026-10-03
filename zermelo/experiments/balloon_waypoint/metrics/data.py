"""Every number one launch is reported by, read off its records

g_t          max over candidates x of ||W(x, t)||, the fastest wind on offer at time_step t (m/s)
b_t          the wind speed stood in at time_step t, 0 outside the candidate box (m/s)
mu_t, v_t    (mu_u, mu_v) and (log var_u, log var_v) claimed at every candidate after time_step t
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG_VARIANCE_FLOOR = -12.0
"""Smallest log variance a claim is read at"""

ACQUISITION_FAMILIES = {
    "rand_act": "RandAct",
    "rand_wp": "RandWP",
    "max_var": "MaxVar",
    "ucb": "UCB",
    "ei": "EI",
    "ts": "TS",
    "forecast": "Forecast",
    "emi": "EMI",
    "evi": "EVI",
    "esi": "ESI",
}
"""The paper's name for each acquisition family, keyed by the token an arm's name carries"""

DRAWN_ARMS: tuple[str, ...] | None = None
"""The arms the figure draws, in legend order; None draws every arm of the launch. Every arm appears in the table"""

CURVES = {
    "simple_regret": ("simple regret (m/s)", "linear"),
    "cumulative_regret": ("cumulative regret (m/s)", "linear"),
    "rmse": ("RMSE (m/s)", "linear"),
    "posterior_spread": ("posterior spread (m/s)", "linear"),
}
"""What an arm achieved, a value per time_step, as the words naming it and the scale it is drawn on"""

TABLE_COLUMNS = (
    ("simple_regret", "simple regret", 2, "min"),
    ("cumulative_regret", "cumulative regret", 0, "min"),
    ("rmse", "RMSE", 3, "min"),
    ("posterior_spread", "posterior spread", 3, "min"),
    ("arrival_share", r"arrival \%", 1, "max"),
    ("in_box_share", r"in-box \%", 1, "max"),
    ("altitude_changes", "altitude changes", 0, "none"),
    ("seconds", "seconds per episode", 0, "min"),
)
"""The table's columns: the scalar, its heading, its digits, and which end is better"""

CURVE_CHANNELS = ("objective_state/best_possible_speed", "objective_state/speed", "objective_state/truth", "objective_state/claim")
"""What the curves are computed from, as the names a record holds them under"""

TABLE_CHANNELS = CURVE_CHANNELS + (
    "state/altitude",
    "state/hours_elapsed",
    "agent_state/time_steps_since_waypoint",
    "reward",
    "time_per_episode",
)


def channels(run: Path, wanted: Sequence[str]) -> dict[str, Any]:
    """The arrays `wanted` names from one run's record (a name the record does not hold is absent)"""
    with np.load(run / "record.npz") as npz:
        return {name: npz[name] for name in wanted if name in npz}


def settings(name: str) -> dict[str, str]:
    """The pairs a run directory's name carries, as `key=value` joined by commas"""
    return dict(pair.split("=", 1) for pair in name.split(",") if "=" in pair)


def acquisition_label(arm: str) -> str:
    """The paper's label for `arm`: `core_emi_L25` -> `EMI-L25`, `tuning_esi_L8_wind_cw0p25_sr0p5` -> `ESI-L8 (wind) cw=0.25 sr=0.5`"""
    tokens = arm.split("_")
    for start in range(len(tokens)):
        for width in (2, 1):
            key = "_".join(tokens[start : start + width])
            if key in ACQUISITION_FAMILIES:
                label = ACQUISITION_FAMILIES[key]
                for token in tokens[start + width :]:
                    if token[0] in "Lc" and token[1:].isdigit():
                        label += f"-{token}"
                    elif token == "error":
                        pass
                    elif token == "wind":
                        label += " (wind)"
                    elif token[:2] in ("cw", "sr"):
                        label += f" {token[:2]}={token[2:].replace('p', '.')}"
                    else:
                        label += f" {token}"
                return label
    return arm.replace("_", " ")


def claimed_speed(claim: Any) -> tuple[Any, Any]:
    """The speed a claim predicts and the spread it holds: `||mu||` and `(sum_i exp(v_i))^(1/2)`"""
    mean, log_variance = claim[..., 0:2], np.maximum(claim[..., 2:4], LOG_VARIANCE_FLOOR)
    return np.linalg.norm(mean, axis=-1), np.sqrt(np.sum(np.exp(log_variance), axis=-1))


def at_each_snapshot(claim: Any, snapshots: int, claim_every: int) -> Any:
    """A claim stored every `claim_every` time_steps, spread back over `snapshots` by carrying each reading forward"""
    return claim[np.minimum(np.arange(snapshots) // claim_every, claim.shape[0] - 1)]


def episode_curves(held: dict[str, Any], claim_every: int) -> dict[str, Any]:
    """What one episode achieved, a value per time_step

    simple_regret     g_t - b_t
    cumulative_regret sum over u <= t of (g_u - b_u)
    rmse              sqrt(mean over candidates of (||W|| - ||mu_t||)^2)
    posterior_spread  mean over candidates of (exp(v_t,u) + exp(v_t,v))^(1/2)
    """
    best_possible_speed = held["objective_state/best_possible_speed"][1:]  # (time_steps,) g_t
    flown = held["objective_state/speed"][1:]  # (time_steps,) b_t
    truth = held["objective_state/truth"][1:]  # (time_steps, candidates) ||W|| at each candidate
    claim = at_each_snapshot(held["objective_state/claim"], flown.size + 1, claim_every)[1:]
    speed, spread = claimed_speed(claim)  # (time_steps, candidates) each
    regret = best_possible_speed - flown
    return {
        "simple_regret": regret,
        "cumulative_regret": np.cumsum(regret),
        "rmse": np.sqrt(np.mean((speed - truth) ** 2, axis=-1)),
        "posterior_spread": np.mean(spread, axis=-1),
    }


def opening_moves(config: dict[str, Any], moves: int) -> int:
    """Time_steps walked at random before any waypoint exists (the whole episode where there is no rule)"""
    method = config.get("method")
    if method is None:
        return moves
    return int(method["opening_steps"])


def episode_legs(held: dict[str, Any], config: dict[str, Any]) -> dict[str, np.ndarray]:
    """One row per finished waypoint leg: the time_step it ended at, the steps walked, and whether it arrived"""
    method = config.get("method")
    if method is None:
        return {name: np.zeros(0) for name in ("ended", "walked", "arrived")}
    budget = int(method["planner"]["replan_every"])
    since = held["agent_state/time_steps_since_waypoint"]  # (time_steps + 1,) zero on a replan and up one a time_step
    ends = np.flatnonzero(np.diff(since) < 0)  # the last snapshot of each waypoint leg, where the sawtooth drops
    ends = ends[ends > opening_moves(config, since.size - 1)]  # the drop closing the opening ends no waypoint leg
    walked = since[ends].astype(float)
    return {"ended": ends.astype(float), "walked": walked, "arrived": (walked < budget).astype(float)}


def latest_launch(sweep: Path) -> Path:
    """The directory holding runs: `sweep` itself, or the newest launch under it"""
    return sweep if any(sweep.glob("*/record.npz")) else max(launch for launch in sweep.iterdir() if launch.is_dir())


def finished_runs(launch: Path) -> list[tuple[str, int, Path, dict[str, Any]]]:
    """Every finished run under one launch: the arm with the sweep's prefix dropped, the seed, its directory, and its settings"""
    found: list[tuple[str, int, Path, dict[str, Any]]] = []
    for marker in sorted(launch.glob("*/record.npz")):  # a record is renamed into place whole, so its presence means finished
        varied = settings(marker.parent.name)
        arm = varied.get("arm", "one arm").removeprefix(launch.parent.name + "_")
        found.append((arm, int(varied.get("seed", 0)), marker.parent, json.loads((marker.parent / "config.json").read_text())))
    if not found:
        raise ValueError(f"no finished run under {launch}")
    return found


def tables(runs: list[tuple[str, int, Path, dict[str, Any]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two tables over one launch: the curves of every episode at every hour `x`, and one row of scalars per episode"""
    curves: list[pd.DataFrame] = []
    scalars: list[dict[str, Any]] = []
    for arm, seed, run, config in runs:
        held = channels(run, TABLE_CHANNELS)
        achieved = episode_curves(held, int(config["claim_every"]))
        legs = episode_legs(held, config)
        curves.append(pd.DataFrame({"acquisition": arm, "seed": seed, "x": held["state/hours_elapsed"][1:], **achieved}))
        scalars.append(
            {
                "acquisition": arm,
                "seed": seed,
                **{name: float(values[-1]) for name, values in achieved.items()},
                "arrival_share": 100.0 * float(np.mean(legs["arrived"])) if legs["arrived"].size else float("nan"),
                "in_box_share": 100.0 * float(np.mean(held["objective_state/speed"][1:] > 0)),
                "altitude_changes": float(np.sum(np.abs(np.diff(held["state/altitude"])))),
                "seconds": float(held["time_per_episode"]),
            }
        )
        del held  # one run's channels live at a time
    return pd.concat(curves, ignore_index=True), pd.DataFrame(scalars)
