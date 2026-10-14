"""Every number one launch is reported by, read off its records

arrived_t    1 once the boat is within arrival_km of the target, 0 before
g, b_t       max over candidates of ||W||, and the largest ||W|| stood on at a candidate by time_step t
field(c)     (W_u, W_v, U_u, U_v) at every candidate c, the truth: the ocean current W and the wind U
mu_t, v_t    the means and the log-variances claimed per channel at every candidate after time_step t
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zermelo.problems.ocean import KM_PER_DEGREE, OCEAN_CURRENT, WIND

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
    "eii": "EII",
    "esi": "ESI",
    "etr": "ETR",
    "nearest": "Nearest",
    "oracle_etr": "Oracle-ETR",
}
"""The paper's name for each acquisition family, keyed by the token an arm's name carries"""


CURVES = {
    "distance_to_target": ("distance to target (km)", "linear"),
    "arrived": ("arrived", "linear"),
    "rmse": ("RMSE", "linear"),
    "posterior_variance": ("posterior variance", "linear"),
    "wind_rmse": ("wind RMSE", "linear"),
    "wind_posterior_variance": ("wind posterior variance", "linear"),
    "information": ("information (nats)", "linear"),
    "wind_information": ("wind information (nats)", "linear"),
    "simple_regret": ("simple regret (m/s)", "linear"),
    "cumulative_regret": ("cumulative regret (m/s)", "linear"),
    "targets_visited": ("targets visited", "linear"),
}
"""What an arm achieved, a value per time_step (distance in km, RMSE in m/s, the variance in m^2/s^2), as the words naming it and the scale it is drawn on"""


CURVE_CHANNELS = (
    "objective_state/arrived",
    "objective_state/steps_to_arrival",
    "objective_state/truth",
    "objective_state/claim",
    "objective_state/best_observed_magnitude",
    "objective_state/best_possible_magnitude",
    "objective_state/visited",
    "objective_state/steps_to_completion",
)
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


def claimed(claim: Any, channels: slice) -> tuple[Any, Any]:
    """What a claim predicts over `channels` and the spread it holds there: `mu` and `(sum over the channels of exp(v))^(1/2)`"""
    m = claim.shape[-1] // 2
    mean, log_variance = claim[..., :m][..., channels], np.maximum(claim[..., m:][..., channels], LOG_VARIANCE_FLOOR)
    return mean, np.sqrt(np.sum(np.exp(log_variance), axis=-1))


def at_each_snapshot(claim: Any, snapshots: int, claim_every: int) -> Any:
    """A claim stored every `claim_every` time_steps, spread back over `snapshots` by carrying each reading forward"""
    return claim[np.minimum(np.arange(snapshots) // claim_every, claim.shape[0] - 1)]


def episode_curves(held: dict[str, Any], claim_every: int, prior_log_variance: np.ndarray) -> dict[str, Any]:
    """What one episode achieved, a value per time_step

    distance_to_target       ||position - target|| on the plane tangent at the box's middle latitude, km
    arrived                  1 once within arrival_km of the target
    rmse                     sqrt(mean over candidates and components of (W - mu_t)^2)
    posterior_variance       mean over candidates of exp(v_t,u) + exp(v_t,v) over W's channels
    wind_rmse, wind_posterior_variance    the same over U
    information              (1/2) sum over candidates and W's channels of (prior_log_variance - v_t), nats the claim's marginals have shed since the prior
    wind_information         the same over U's channels
    simple_regret       g - b_t, nan unless the episode searches for the fastest current
    cumulative_regret   sum over u <= t of (g - b_u)
    targets_visited     how many targets the boat has been within arrival_km of, nan unless the episode visits targets
    """
    truth = held["objective_state/truth"][1:]  # (time_steps, candidates, channels)
    arrived = held.get("objective_state/arrived", np.zeros(truth.shape[0] + 1))[1:]  # (time_steps,): never, when learning the region
    position, target = held["state/position"][1:], held["state/target"][1:]  # (time_steps, 2) as (lat, lon) degrees
    km_per_degree = KM_PER_DEGREE * np.array([1.0, np.cos(np.radians(np.mean(target[:, 0])))])
    distance = np.linalg.norm((position - target) * km_per_degree, axis=-1)
    claim = at_each_snapshot(held["objective_state/claim"], arrived.size + 1, claim_every)[1:]
    mean, spread = claimed(claim, OCEAN_CURRENT)  # (time_steps, candidates, 2) and (time_steps, candidates)
    wind_mean, wind_spread = claimed(claim, WIND)
    stored = held["objective_state/claim"]
    log_variance = np.maximum(stored[..., stored.shape[-1] // 2 :], LOG_VARIANCE_FLOOR)  # (snapshots, candidates, channels)
    shed = 0.5 * (prior_log_variance - log_variance)  # (snapshots, candidates, channels)
    regret = (
        (held["objective_state/best_possible_magnitude"] - held["objective_state/best_observed_magnitude"])[1:]
        if "objective_state/best_possible_magnitude" in held
        else np.full(arrived.size, np.nan)
    )  # (time_steps,)
    return {
        "distance_to_target": distance,
        "arrived": arrived,
        "rmse": np.sqrt(np.mean((mean - truth[..., OCEAN_CURRENT]) ** 2, axis=(-2, -1))),
        "posterior_variance": np.mean(spread**2, axis=-1),
        "wind_rmse": np.sqrt(np.mean((wind_mean - truth[..., WIND]) ** 2, axis=(-2, -1))),
        "wind_posterior_variance": np.mean(wind_spread**2, axis=-1),
        "information": at_each_snapshot(np.sum(shed[..., OCEAN_CURRENT], axis=(-2, -1)), arrived.size + 1, claim_every)[1:],
        "wind_information": at_each_snapshot(np.sum(shed[..., WIND], axis=(-2, -1)), arrived.size + 1, claim_every)[1:],
        "simple_regret": regret,
        "cumulative_regret": np.cumsum(regret),
        "targets_visited": np.sum(held["objective_state/visited"][1:], axis=-1)
        if "objective_state/visited" in held
        else np.full(arrived.size, np.nan),
    }


def prior_log_variance(config: dict[str, Any]) -> np.ndarray:
    """(channels,): 2 log amplitude of the block each channel sits in, the level a claim's log-variance is counted from"""
    return np.concatenate([np.full(block["width"], 2.0 * np.log(block["amplitude"])) for block in config["belief"]["blocks"]])


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
        achieved = episode_curves(held, int(config["claim_every"]), prior_log_variance(config))
        legs = episode_legs(held, config)
        hours = float(config["problem"]["step_hours"]) * np.arange(1, held["objective_state/truth"].shape[0])
        curves.append(pd.DataFrame({"acquisition": arm, "seed": seed, "x": hours, **achieved}))
        scalars.append(
            {
                "acquisition": arm,
                "seed": seed,
                **{name: float(values[-1]) for name, values in achieved.items()},
                "steps_to_arrival": float(held.get("objective_state/steps_to_arrival", [float("nan")])[-1]),
                "steps_to_completion": float(held.get("objective_state/steps_to_completion", [float("nan")])[-1]),
                **{
                    name: float("nan") if config.get("method") is None else float(achieved[name][-1])
                    for name in ("rmse", "posterior_variance", "wind_rmse", "wind_posterior_variance", "information", "wind_information")
                },
                "arrival_share": 100.0 * float(np.mean(legs["arrived"])) if legs["arrived"].size else float("nan"),
                "seconds": float(held["time_per_episode"]),
            }
        )
        del held  # one run's channels live at a time
    return pd.concat(curves, ignore_index=True), pd.DataFrame(scalars)
