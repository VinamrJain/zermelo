"""Every number one launch is reported by, read off its records

g, b_t       max over candidates of ||W||, and the largest ||W|| stood on at a candidate by time_step t
field(c)     (W_u, W_v, U_u, U_v) at every candidate c, the truth: the ocean current W and the wind U
mu_t, v_t    the means and the log-variances claimed per component at every candidate after time_step t
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zermelo.problems.ocean import OCEAN_CURRENT, WIND
from zermelo.run.record import RENAMED_CHANNELS

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
    "objective_state/truth",
    "objective_state/claim",
    "objective_state/best_observed_magnitude",
    "objective_state/best_possible_magnitude",
    "objective_state/visited",
    "objective_state/steps_to_completion",
)
"""What the curves are computed from, as the names a record holds them under"""

TABLE_CHANNELS = CURVE_CHANNELS + ("state/position", "agent_state/steps_since_target", "reward", "time_per_episode")


def channels(run: Path, wanted: Sequence[str]) -> dict[str, Any]:
    """The arrays `wanted` names from one run's record (a name the record does not hold is absent)"""
    old_name = {new: old for old, new in RENAMED_CHANNELS.items()}
    with np.load(run / "record.npz") as npz:
        return {name: npz[name] if name in npz else npz[old_name[name]] for name in wanted if name in npz or old_name.get(name) in npz}


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


def claimed(claim: Any, components: tuple[int, ...]) -> tuple[Any, Any]:
    """What a claim predicts over `components` and the spread it holds there: `mu` and `(sum over the components of exp(v))^(1/2)`"""
    m, chosen = claim.shape[-1] // 2, list(components)
    mean, log_variance = claim[..., :m][..., chosen], np.maximum(claim[..., m:][..., chosen], LOG_VARIANCE_FLOOR)
    return mean, np.sqrt(np.sum(np.exp(log_variance), axis=-1))


def at_each_snapshot(claim: Any, snapshots: int, claim_every: int) -> Any:
    """A claim stored every `claim_every` time_steps, spread back over `snapshots` by carrying each reading forward"""
    return claim[np.minimum(np.arange(snapshots) // claim_every, claim.shape[0] - 1)]


def episode_curves(held: dict[str, Any], claim_every: int, prior_log_variance: np.ndarray) -> dict[str, Any]:
    """What one episode achieved, a value per time_step

    rmse                     sqrt(mean over candidates and components of (W - mu_t)^2)
    posterior_variance       mean over candidates of exp(v_t,u) + exp(v_t,v) over W's components
    wind_rmse, wind_posterior_variance    the same over U
    information              (1/2) sum over candidates and W's components of (prior_log_variance - v_t), nats the claim's marginals have shed since the prior
    wind_information         the same over U's components
    simple_regret       g - b_t, nan unless the episode searches for the fastest current
    cumulative_regret   sum over u <= t of (g - b_u)
    targets_visited     how many targets the boat has been within arrival_km of, nan unless the episode visits targets
    """
    truth = held["objective_state/truth"][1:]  # (time_steps, candidates, components)
    steps = truth.shape[0]
    claim = at_each_snapshot(held["objective_state/claim"], steps + 1, claim_every)[1:]
    mean, spread = claimed(claim, OCEAN_CURRENT)  # (time_steps, candidates, 2) and (time_steps, candidates)
    wind_mean, wind_spread = claimed(claim, WIND)
    stored = held["objective_state/claim"]
    log_variance = np.maximum(stored[..., stored.shape[-1] // 2 :], LOG_VARIANCE_FLOOR)  # (snapshots, candidates, components)
    shed = 0.5 * (prior_log_variance - log_variance)  # (snapshots, candidates, components)
    regret = (
        (held["objective_state/best_possible_magnitude"] - held["objective_state/best_observed_magnitude"])[1:]
        if "objective_state/best_possible_magnitude" in held
        else np.full(steps, np.nan)
    )  # (time_steps,)
    return {
        "rmse": np.sqrt(np.mean((mean - truth[..., list(OCEAN_CURRENT)]) ** 2, axis=(-2, -1))),
        "posterior_variance": np.mean(spread**2, axis=-1),
        "wind_rmse": np.sqrt(np.mean((wind_mean - truth[..., list(WIND)]) ** 2, axis=(-2, -1))),
        "wind_posterior_variance": np.mean(wind_spread**2, axis=-1),
        "information": at_each_snapshot(np.sum(shed[..., list(OCEAN_CURRENT)], axis=(-2, -1)), steps + 1, claim_every)[1:],
        "wind_information": at_each_snapshot(np.sum(shed[..., list(WIND)], axis=(-2, -1)), steps + 1, claim_every)[1:],
        "simple_regret": regret,
        "cumulative_regret": np.cumsum(regret),
        "targets_visited": np.sum(held["objective_state/visited"][1:], axis=-1)
        if "objective_state/visited" in held
        else np.full(steps, np.nan),
    }


def belief_groups(config: dict[str, Any]) -> list[tuple[list[int], float]]:
    """(components, amplitude) of each group of the belief; a record from before 2026-10-15 stores `blocks` of a `width` in order"""
    if "groups" in config["belief"]:
        return [(list(group["components"]), float(group["amplitude"])) for group in config["belief"]["groups"]]
    groups, first = [], 0
    for block in config["belief"]["blocks"]:
        groups.append((list(range(first, first + block["width"])), float(block["amplitude"])))
        first += block["width"]
    return groups


def prior_log_variance(config: dict[str, Any]) -> np.ndarray:
    """(components,): 2 log amplitude of the group each component sits in, the level a claim's log-variance is counted from"""
    out = np.zeros(sum(len(components) for components, _ in belief_groups(config)))
    for components, amplitude in belief_groups(config):
        out[components] = 2.0 * np.log(amplitude)
    return out


def opening_moves(config: dict[str, Any], moves: int) -> int:
    """Time steps moved at random before any target exists (the whole episode where there is no rule)"""
    method = config.get("method")
    if method is None:
        return moves
    return int(method.get("random_opening_steps", method.get("opening_steps")))


def episode_legs(held: dict[str, Any], config: dict[str, Any]) -> dict[str, np.ndarray]:
    """One row per finished target leg: the time_step it ended at, the steps moved, and whether it arrived"""
    method = config.get("method")
    if method is None:
        return {name: np.zeros(0) for name in ("ended", "walked", "arrived")}
    budget = int(method["planner"].get("replan_length", method["planner"].get("replan_every")))
    since = held["agent_state/steps_since_target"]  # (time_steps + 1,) zero on a replan and up one a time_step
    ends = np.flatnonzero(np.diff(since) < 0)  # the last snapshot of each target leg, where the sawtooth drops
    ends = ends[ends > opening_moves(config, since.size - 1)]  # the drop closing the opening ends no target leg
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
