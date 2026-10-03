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
from zermelo.methods.waypoint_bo.belief import BeliefKernel, GPBelief, OracleBelief
from zermelo.problems.balloon import (
    GaussianError,
    GEFSError,
    WindError,
    balloon_objective,
    balloon_states,
    balloon_transition,
    balloon_world,
    candidate_box,
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
    wind_data = load_wind(Path(__file__).resolve().parents[3] / cfg.problem.wind_path, cfg.problem.grid_stride, cfg.problem.hour_stride)
    grid = wind_data.grid
    # (frames, alt, pos, uv) twice, and the hours_elapsed each frame stands for; one frame where the wind is held still
    truth, predicted, frame_hours_elapsed = wind_data.frames_from(
        cfg.problem.start_hour, cfg.horizon * cfg.problem.step_hours, cfg.problem.time_varying
    )
    error: WindError
    if cfg.problem.forecast == "GEFS":
        error = GEFSError(truth - predicted)  # W - F at every frame the episode reads
    else:  # W is the forecast, so the drawn error is what makes them differ
        predicted = truth
        error = GaussianError(
            length_scale_km=cfg.problem.error_lengthscale_km, amplitude_ms=cfg.problem.error_scale, jitter=cfg.problem.error_jitter
        )
    states = balloon_states(grid, tuple(float(h) for h in wind_data.altitude_km))
    transition = balloon_transition(wind_data, states, cfg.problem.step_hours)
    box = candidate_box(grid, cfg.problem.lat_min, cfg.problem.lat_max, cfg.problem.lon_min, cfg.problem.lon_max)
    inset = cfg.problem.start_inset_deg
    start_box = candidate_box(
        grid, cfg.problem.lat_min + inset, cfg.problem.lat_max - inset, cfg.problem.lon_min + inset, cfg.problem.lon_max - inset
    )
    world = balloon_world(grid, states, predicted, frame_hours_elapsed, error, transition, cfg.problem.resource_units, start_box)
    objective = balloon_objective(states, grid, instantiate(cfg.problem.target, _target_whitelist_=WHITELIST), box)
    candidates = objective.candidates
    context = world.readout.context
    belief = (
        # the wind of the world the episode runs in, drawn on its key
        OracleBelief.empty(states, cfg.horizon, WIND_COMPONENTS, world.reset(k_world)[0]["field"], context)
        if cfg.belief.oracle
        else GPBelief.empty(
            states,
            cfg.horizon,  # one row per move, the readings an episode folds in
            WIND_COMPONENTS,
            n_context=sum(context.values()),
            kernel_factors=tuple(BeliefKernel.of(get_class(f.kernel), f.parts, f.lengthscale) for f in cfg.belief.kernel_factors),
            context=context,
            amplitude=jnp.asarray(cfg.belief.amplitude),
            noise=jnp.asarray(cfg.belief.noise),
            n_features=cfg.belief.n_features,
            refit=cfg.belief.refit,
            refit_steps=cfg.belief.refit_steps,
            prior_mean=None,  # the forecast the reading carries, set at the first reading
        )
    )
    agent: Agent[Any]
    if cfg.method is None:
        # no rule and no update: it claims the prior every step
        agent = RandomAgent(claim=posterior_moments(belief, jnp.zeros(sum(context.values()))))
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
            context_key="context",
            prior_mean_key="forecast" if cfg.belief.forecast_as_prior_mean else None,
            horizon=cfg.horizon,
            opening_legs=cfg.method.opening_legs,
            claim_every=cfg.claim_every,
        )
    # the settings as plain data, carried onto the record
    return Episode(world, objective, agent, cfg.horizon, dict(cfg.recorded), asdict(cfg), k_world, k_agent, k_steps)
