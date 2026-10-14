"""One resolved configuration into the objects an episode runs on"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from hydra.utils import get_class, instantiate
from jaxtyping import Array, Float

from zermelo.experiments.ocean_waypoint.registry import SCORED_CHANNELS
from zermelo.experiments.ocean_waypoint.schema import RunConfig
from zermelo.interface import Agent
from zermelo.methods.random_agent import RandomAgent
from zermelo.methods.waypoint_bo.acquisition import Acquisition
from zermelo.methods.waypoint_bo.agent import WaypointAgent, posterior_moments
from zermelo.methods.waypoint_bo.belief import BeliefKernel, BlockBelief, Dataset, GPBelief, OracleBelief
from zermelo.problems.ocean import (
    CHANNELS,
    FixedError,
    ForecastError,
    GaussianError,
    Polar,
    learn_objective,
    load_ocean,
    ocean_objective,
    ocean_states,
    ocean_transition,
    ocean_world,
    search_objective,
    visit_objective,
)
from zermelo.run.episode import Episode

WHITELIST = ("zermelo.*", "gpjax.*")
"""Which modules a setting may name"""


def prior_amplitude(cfg: RunConfig) -> Float[Array, " channels"]:
    """The standard deviation the belief assumes of each channel before any reading"""
    return jnp.concatenate([jnp.full(block.width, block.amplitude) for block in cfg.belief.blocks])


def assemble(cfg: RunConfig) -> Episode:
    """The episode one configuration describes, that configuration carried onto its record"""
    k_world, k_agent, k_steps = jax.random.split(jax.random.key(cfg.seed), 3)
    data = load_ocean(Path(__file__).resolve().parents[3] / cfg.problem.data_path, cfg.problem.grid_stride)
    grid = data.grid
    truth = data.field(data.frame(cfg.problem.frame_hour))  # (pos, channels)
    error: ForecastError
    if cfg.problem.forecast_lag_hours > 0:
        predicted = data.field(data.frame(cfg.problem.frame_hour - cfg.problem.forecast_lag_hours))
        error = FixedError(truth - predicted)
    else:  # the field is the forecast, so the drawn error is what makes them differ
        predicted = truth
        error = GaussianError(
            length_scale_km=cfg.problem.error_lengthscale_km, amplitude_ms=cfg.problem.error_scale, jitter=cfg.problem.error_jitter
        )
    states = ocean_states(grid)
    polar = Polar(
        cfg.problem.top_speed_ms,
        cfg.problem.no_go_degrees,
        cfg.problem.peak_degrees,
        cfg.problem.rise_exponent,
        cfg.problem.downwind_factor,
        cfg.problem.wind_scale_ms,
    )
    transition = ocean_transition(grid, polar, cfg.problem.n_headings, cfg.problem.step_hours, states)
    box = grid.inside(cfg.problem.lat_min, cfg.problem.lat_max, cfg.problem.lon_min, cfg.problem.lon_max) & ~data.land
    target_box = box & grid.inside(
        cfg.problem.target_lat_min, cfg.problem.target_lat_max, cfg.problem.target_lon_min, cfg.problem.target_lon_max
    )
    noise_ms = jnp.asarray([cfg.problem.noise_ms] * 2 + [cfg.problem.wind_noise_ms] * 2)  # (channels,)
    world = ocean_world(
        grid, states, predicted, error, transition, noise_ms, target_box, target_box
    )  # the boat starts in the target box too
    # the targets of a visiting episode, drawn uniformly from the target box on a key folded off the world's
    targets = (
        jnp.zeros(grid.size(), bool)
        .at[jax.random.choice(jax.random.fold_in(k_world, 1), jnp.flatnonzero(target_box), (cfg.problem.n_targets,), replace=False)]
        .set(True)
    )
    objective = (  # under learn and search the region of interest is the target box
        ocean_objective(states, grid, box, cfg.problem.arrival_km)
        if cfg.problem.objective == "reach"
        else learn_objective(states, target_box, 2.0 * jnp.log(prior_amplitude(cfg)), SCORED_CHANNELS[cfg.problem.learned])
        if cfg.problem.objective == "learn"
        else search_objective(states, target_box)
        if cfg.problem.objective == "search"
        else visit_objective(states, grid, targets, cfg.problem.arrival_km)
    )
    candidates = objective.candidates
    context = world.readout.context
    if sum(block.width for block in cfg.belief.blocks) != CHANNELS:
        raise ValueError(
            f"the belief's blocks span {sum(block.width for block in cfg.belief.blocks)} channels and the field has {CHANNELS}"
        )
    belief = (
        # the field of the world the episode runs in, drawn on its key
        OracleBelief.empty(states, cfg.horizon, CHANNELS, world.reset(k_world)[0]["field"], context)
        if cfg.belief.oracle
        else BlockBelief(
            states,
            Dataset.empty(cfg.horizon, CHANNELS, sum(context.values())),  # one row per move, the readings an episode folds in
            tuple(
                GPBelief.empty(
                    states,
                    cfg.horizon,
                    block.width,
                    n_context=sum(context.values()),
                    kernel_factors=tuple(BeliefKernel.of(get_class(f.kernel), f.parts, f.lengthscale) for f in block.kernel_factors),
                    context=context,
                    amplitude=jnp.asarray(block.amplitude),
                    noise=jnp.asarray(block.noise),
                    n_features=cfg.belief.n_features,
                    refit=cfg.belief.refit,
                    refit_steps=cfg.belief.refit_steps,
                    prior_mean=None,  # the forecast the reading carries, set at the first reading
                )
                for block in cfg.belief.blocks
            ),
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
                penalty_rate=cfg.method.penalty_rate,
            ),
            position_key="position",
            reading_key="field_reading",
            context_key="context",
            prior_mean_key="forecast" if cfg.belief.forecast_as_prior_mean else None,
            target_key="target",
            budget_column=None,
            horizon=cfg.horizon,
            opening_steps=cfg.method.opening_steps,
            claim_every=cfg.claim_every,
        )
    # the settings as plain data, carried onto the record
    return Episode(world, objective, agent, cfg.horizon, dict(cfg.recorded), asdict(cfg), k_world, k_agent, k_steps)
