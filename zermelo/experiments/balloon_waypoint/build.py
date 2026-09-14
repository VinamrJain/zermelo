"""One resolved configuration into the objects an episode runs on"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from hydra.utils import get_class, instantiate

from zermelo.experiments.balloon_waypoint.schema import RunConfig
from zermelo.interface import Agent
from zermelo.methods.random_agent import RandomAgent
from zermelo.methods.waypoint_bo.acquisition import Acquisition
from zermelo.methods.waypoint_bo.agent import WaypointAgent, posterior_moments
from zermelo.methods.waypoint_bo.belief import GPBelief, OracleBelief
from zermelo.problems.balloon import (
    GaussianError,
    GridWind,
    balloon_objective,
    balloon_states,
    balloon_transition,
    balloon_world,
    load_wind,
)
from zermelo.run.episode import Episode

WHITELIST = ("zermelo.*", "gpjax.*")
"""Which modules a setting may name"""

WIND_COMPONENTS = 2
"""(u, v): what a belief over the wind is wide in"""


def assemble(cfg: RunConfig) -> Episode:
    """The episode one configuration describes, that configuration carried onto its record"""
    k_world, k_agent, k_steps = jax.random.split(jax.random.key(cfg.seed), 3)
    recording = load_wind(Path(__file__).resolve().parents[3] / cfg.problem.wind_path)
    grid, n_alt = recording.grid, recording.n_alt
    error = GaussianError(
        length_scale_km=cfg.problem.error_lengthscale_km, amplitude_ms=cfg.problem.error_scale, jitter=cfg.problem.error_jitter
    )
    states = balloon_states(grid, tuple(float(h) for h in recording.altitude_km))
    transition = balloon_transition(recording, states, cfg.problem.step_hours)
    world = balloon_world(
        recording,
        states,
        frame=cfg.problem.frame,
        error=error,
        transition=transition,
        readout=get_class(cfg.problem.readout),
        resource_units=cfg.problem.resource_units,
    )
    objective = balloon_objective(
        recording, states, instantiate(cfg.problem.target, _target_whitelist_=WHITELIST), cfg.problem.margin_lat, cfg.problem.margin_lon
    )
    candidates = objective.candidates
    forecast = GridWind(recording.at(cfg.problem.frame), grid)
    # three horizontal coordinates in km, then altitude in km
    lengthscale = jnp.asarray([cfg.belief.lengthscale_km] * 3 + [cfg.belief.lengthscale_altitude_km])
    belief = (
        # the wind of the world the episode runs in, drawn on its key
        OracleBelief.empty(states, cfg.horizon, WIND_COMPONENTS, world.reset(k_world)[0]["field"])
        if cfg.belief.oracle
        else GPBelief.empty(
            states,
            cfg.horizon,  # one row per move, the readings an episode folds in
            WIND_COMPONENTS,
            lengthscale=lengthscale,
            amplitude=jnp.asarray(cfg.belief.amplitude),
            noise=jnp.asarray(cfg.belief.noise),
            kernel_family=get_class(cfg.belief.kernel),
            n_features=cfg.belief.n_features,
            refit=cfg.belief.refit,
            refit_steps=cfg.belief.refit_steps,
            prior_mean=forecast if cfg.belief.forecast_prior else None,
        )
    )
    agent: Agent[Any]
    if cfg.method is None:
        # no rule and no update: it claims the prior every step
        agent = RandomAgent(claim=posterior_moments(belief))
    else:
        agent = WaypointAgent(
            candidates=candidates,
            transition=transition,
            belief=belief,
            acquisition=Acquisition(
                utility=instantiate(cfg.method.utility, _target_whitelist_=WHITELIST),
                planner=instantiate(cfg.method.planner, _target_whitelist_=WHITELIST),
                n_fields=cfg.method.n_fields,
                n_walks=cfg.method.n_walks,
                n_candidates=cfg.method.n_candidates,
                improvement=cfg.method.improvement,
                step_rate=cfg.method.step_rate,
                steps_from=cfg.method.steps_from,
                combination=cfg.method.combination,
            ),
            position_key="position",
            reading_key="wind",
            horizon=cfg.horizon,
            opening_legs=cfg.method.opening_legs,
            claim_every=cfg.claim_every,
        )
    # the settings as plain data, carried onto the record
    return Episode(world, objective, agent, cfg.horizon, asdict(cfg), k_world, k_agent, k_steps)
