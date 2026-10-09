"""Every number one launch is reported by, read off its records

arrived_t    1 once the boat is within arrival_km of the target, 0 before
W(c)         (u, v) at every candidate c, the truth
mu_t, v_t    (mu_u, mu_v) and (log var_u, log var_v) claimed at every candidate after time_step t
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zermelo.problems.ocean import KM_PER_DEGREE

LOG_VARIANCE_FLOOR = -12.0
"""Smallest log variance a claim is read at"""

ACQUISITION_FAMILIES = {
    "rand_act": "RandAct",
    "rand_wp": "RandTarget",
    "rand_target": "RandTarget",
    "max_var": "MaxVar",
    "ucb": "UCB",
    "ei": "EI",
    "mc_ei": "MC-EI",
    "ts": "TS",
    "forecast": "Forecast",
    "direct": "Direct",
    "info": "IG",
    "eti": "ETI",
    "oracle": "Oracle",
    "oracle_emi": "Oracle-EMI",
    "emi": "EMI",
    "evi": "EVI",
    "esi": "ESI",
}
"""The paper's name for each acquisition family, keyed by the token an arm's name carries"""

DRAWN_ARMS: tuple[str, ...] | None = None
"""The arms the figure draws, in legend order; None draws every arm of the launch. Every arm appears in the table"""

CURVES = {
    "distance_to_target": ("distance to target (km)", "linear"),
    "arrived": ("arrived", "linear"),
    "rmse": ("RMSE", "linear"),
    "posterior_variance": ("posterior variance", "linear"),
}
"""What an arm achieved, a value per time_step (distance in km, RMSE in m/s, the variance in m^2/s^2), as the words naming it and the scale it is drawn on"""

TABLE_COLUMNS = (
    ("steps_to_arrival", "steps to arrival", 0, "min"),
    ("arrived", r"arrived \%", 0, "max"),
    ("rmse", "RMSE", 2, "min"),
    ("posterior_variance", "posterior variance", 2, "min"),
    ("arrival_share", r"waypoint arrival \%", 0, "max"),
    ("seconds", "time (s)", 0, "none"),
)
"""The table's columns: the scalar, its heading, its digits, and which end is better"""

CURVE_CHANNELS = ("objective_state/arrived", "objective_state/steps_to_arrival", "objective_state/truth", "objective_state/claim")
"""What the curves are computed from, as the names a record holds them under"""

TABLE_CHANNELS = CURVE_CHANNELS + ("state/position", "state/target", "agent_state/time_steps_since_waypoint", "reward", "time_per_episode")


def channels(run: Path, wanted: Sequence[str]) -> dict[str, Any]:
    """The arrays `wanted` names from one run's record (a name the record does not hold is absent)"""
    with np.load(run / "record.npz") as npz:
        return {name: npz[name] for name in wanted if name in npz}


def settings(name: str) -> dict[str, str]:
    """The pairs a run directory's name carries, as `key=value` joined by commas"""
    return dict(pair.split("=", 1) for pair in name.split(",") if "=" in pair)


def acquisition_label(arm: str) -> str:
    """The paper's label for `arm`: `core_emi_L25` -> `EMI-L25`, `tuning_esi_L8_current_cw0p25_sr0p5` -> `ESI-L8 (ocean current) cw=0.25 sr=0.5`"""
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
                    elif token == "ocean_current":
                        label += " (ocean current)"
                    elif token[:2] in ("cw", "sr"):
                        label += f" {token[:2]}={token[2:].replace('p', '.')}"
                    else:
                        label += f" {token}"
                return label
    return arm.replace("_", " ")


def claimed(claim: Any) -> tuple[Any, Any]:
    """The (u, v) a claim predicts and the spread it holds: `mu` and `(sum_i exp(v_i))^(1/2)`"""
    mean, log_variance = claim[..., 0:2], np.maximum(claim[..., 2:4], LOG_VARIANCE_FLOOR)
    return mean, np.sqrt(np.sum(np.exp(log_variance), axis=-1))


def at_each_snapshot(claim: Any, snapshots: int, claim_every: int) -> Any:
    """A claim stored every `claim_every` time_steps, spread back over `snapshots` by carrying each reading forward"""
    return claim[np.minimum(np.arange(snapshots) // claim_every, claim.shape[0] - 1)]


def episode_curves(held: dict[str, Any], claim_every: int) -> dict[str, Any]:
    """What one episode achieved, a value per time_step

    distance_to_target  ||position - target|| on the plane tangent at the box's middle latitude, km
    arrived             1 once within arrival_km of the target
    rmse                sqrt(mean over candidates and components of (W - mu_t)^2)
    posterior_variance  mean over candidates of exp(v_t,u) + exp(v_t,v)
    information         (1/2) sum over candidates and components of (v_0 - v_t), nats the claim's marginals have shed
    """
    arrived = held["objective_state/arrived"][1:]  # (time_steps,)
    position, target = held["state/position"][1:], held["state/target"][1:]  # (time_steps, 2) as (lat, lon) degrees
    km_per_degree = KM_PER_DEGREE * np.array([1.0, np.cos(np.radians(np.mean(target[:, 0])))])
    distance = np.linalg.norm((position - target) * km_per_degree, axis=-1)
    truth = held["objective_state/truth"][1:]  # (time_steps, candidates, 2)
    claim = at_each_snapshot(held["objective_state/claim"], arrived.size + 1, claim_every)[1:]
    mean, spread = claimed(claim)  # (time_steps, candidates, 2) and (time_steps, candidates)
    log_variance = np.maximum(held["objective_state/claim"][..., 2:4], LOG_VARIANCE_FLOOR)  # (snapshots, candidates, 2)
    shed = 0.5 * np.sum(log_variance[0] - log_variance, axis=(-2, -1))  # (snapshots,)
    return {
        "distance_to_target": distance,
        "arrived": arrived,
        "rmse": np.sqrt(np.mean((mean - truth) ** 2, axis=(-2, -1))),
        "posterior_variance": np.mean(spread**2, axis=-1),
        "information": at_each_snapshot(shed, arrived.size + 1, claim_every)[1:],
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
    """Two tables over one launch: the curves of every episode at every hour `x`, and one row of scalars per episode

    A scalar is its curve's last value, nan for rmse and posterior_variance of an arm with no rule. `x` is hours elapsed.
    """
    curves: list[pd.DataFrame] = []
    scalars: list[dict[str, Any]] = []
    for arm, seed, run, config in runs:
        held = channels(run, TABLE_CHANNELS)
        achieved = episode_curves(held, int(config["claim_every"]))
        legs = episode_legs(held, config)
        hours = float(config["problem"]["step_hours"]) * np.arange(1, held["objective_state/arrived"].size)
        curves.append(pd.DataFrame({"acquisition": arm, "seed": seed, "x": hours, **achieved}))
        scalars.append(
            {
                "acquisition": arm,
                "seed": seed,
                **{name: float(values[-1]) for name, values in achieved.items()},
                "steps_to_arrival": float(held["objective_state/steps_to_arrival"][-1]),
                **{
                    name: float("nan") if config.get("method") is None else float(achieved[name][-1])
                    for name in ("rmse", "posterior_variance", "information")
                },
                "arrival_share": 100.0 * float(np.mean(legs["arrived"])) if legs["arrived"].size else float("nan"),
                "seconds": float(held["time_per_episode"]),
            }
        )
        del held  # one run's channels live at a time
    return pd.concat(curves, ignore_index=True), pd.DataFrame(scalars)
