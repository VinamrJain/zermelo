"""One resolved configuration into the objects an episode runs on"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from hydra.utils import get_class, instantiate

from zermelo.experiments.ocean_waypoint.schema import RunConfig
from zermelo.interface import Agent
from zermelo.methods.random_agent import RandomAgent
from zermelo.methods.waypoint_bo.acquisition import Acquisition
from zermelo.methods.waypoint_bo.agent import WaypointAgent, posterior_moments
from zermelo.methods.waypoint_bo.belief import BeliefKernel, GPBelief, OracleBelief
from zermelo.problems.ocean import (
    FixedError,
    GaussianError,
    OceanCurrentError,
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

OCEAN_CURRENT_COMPONENTS = 2
"""(u, v): what a belief over the ocean current is wide in"""


def assemble(cfg: RunConfig) -> Episode:
    """The episode one configuration describes, that configuration carried onto its record"""
    k_world, k_agent, k_steps = jax.random.split(jax.random.key(cfg.seed), 3)
    data = load_ocean(Path(__file__).resolve().parents[3] / cfg.problem.data_path, cfg.problem.grid_stride)
    grid = data.grid
    frame = data.frame(cfg.problem.frame_hour)
    truth, wind = data.ocean_current[frame], data.wind[frame]  # (pos, uv) each
    error: OceanCurrentError
    if cfg.problem.forecast_lag_hours > 0:
        predicted = data.ocean_current[data.frame(cfg.problem.frame_hour - cfg.problem.forecast_lag_hours)]
        error = FixedError(truth - predicted)
    else:  # W is the forecast, so the drawn error is what makes them differ
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
    transition = ocean_transition(grid, wind, polar, cfg.problem.n_headings, cfg.problem.step_hours, states)
    box = grid.inside(cfg.problem.lat_min, cfg.problem.lat_max, cfg.problem.lon_min, cfg.problem.lon_max) & ~data.land
    target_box = box & grid.inside(
        cfg.problem.target_lat_min, cfg.problem.target_lat_max, cfg.problem.target_lon_min, cfg.problem.target_lon_max
    )
    world = ocean_world(
        grid, states, predicted, error, transition, cfg.problem.noise_ms, target_box, target_box
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
        else learn_objective(states, target_box, 2.0 * float(jnp.log(cfg.belief.amplitude)))
        if cfg.problem.objective == "learn"
        else search_objective(states, target_box)
        if cfg.problem.objective == "search"
        else visit_objective(states, grid, targets, cfg.problem.arrival_km)
    )
    candidates = objective.candidates
    context = world.readout.context
    belief = (
        # the ocean current of the world the episode runs in, drawn on its key
        OracleBelief.empty(states, cfg.horizon, OCEAN_CURRENT_COMPONENTS, world.reset(k_world)[0]["field"], context)
        if cfg.belief.oracle
        else GPBelief.empty(
            states,
            cfg.horizon,  # one row per move, the readings an episode folds in
            OCEAN_CURRENT_COMPONENTS,
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
                penalty_rate=cfg.method.penalty_rate,
            ),
            position_key="position",
            reading_key="ocean_current",
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
