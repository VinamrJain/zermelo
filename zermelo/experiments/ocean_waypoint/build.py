"""One resolved configuration into the objects an episode runs on"""

import dataclasses
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from hydra.utils import get_class, instantiate
from jaxtyping import Array, Float

from zermelo.experiments.ocean_waypoint.registry import SCORED_COMPONENTS
from zermelo.experiments.ocean_waypoint.schema import RunConfig
from zermelo.interface import Agent
from zermelo.methods.path_eui.acquisition import Acquisition
from zermelo.methods.path_eui.agent import PathEUIAgent, posterior_mean_and_log_variance
from zermelo.methods.path_eui.belief import BeliefKernel, Dataset, GPBelief, GroupedBelief, OracleBelief
from zermelo.methods.path_eui.utility import GoalsReached
from zermelo.methods.random_agent import RandomAgent
from zermelo.problems.ocean import (
    COMPONENTS,
    FixedError,
    ForecastError,
    GaussianError,
    Polar,
    learn_objective,
    load_ocean,
    ocean_states,
    ocean_transition,
    ocean_world,
    search_objective,
    visit_objective,
)
from zermelo.run.episode import Episode

WHITELIST = ("zermelo.*", "gpjax.*")
"""Which modules a setting may name"""


def prior_amplitude(cfg: RunConfig) -> Float[Array, " components"]:
    """The standard deviation the belief assumes of each component before any reading"""
    amplitude = jnp.zeros(COMPONENTS)
    for group in cfg.belief.groups:
        amplitude = amplitude.at[jnp.asarray(group.components)].set(group.amplitude)
    return amplitude


def assemble(cfg: RunConfig) -> Episode:
    """The episode one configuration describes, that configuration carried onto its record"""
    k_world, k_agent, k_steps = jax.random.split(jax.random.key(cfg.seed), 3)
    data = load_ocean(Path(__file__).resolve().parents[3] / cfg.problem.data_path, cfg.problem.grid_stride)
    grid = data.grid
    truth = data.field(data.frame(cfg.problem.frame_hour))  # (pos, components)
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
    noise_ms = jnp.asarray([cfg.problem.noise_ms] * 2 + [cfg.problem.wind_noise_ms] * 2)  # (components,)
    world = ocean_world(grid, states, predicted, error, transition, noise_ms, target_box)
    # (n_goals,) state indices: the goals of a visiting episode, drawn uniformly from the target box on a key folded off the world's
    goals = jax.random.choice(jax.random.fold_in(k_world, 1), jnp.flatnonzero(target_box), (cfg.problem.n_goals,), replace=False)
    objective = (  # under learn and search the region of interest is the target box
        learn_objective(states, target_box, 2.0 * jnp.log(prior_amplitude(cfg)), SCORED_COMPONENTS[cfg.problem.learned])
        if cfg.problem.objective == "learn"
        else search_objective(states, target_box)
        if cfg.problem.objective == "search"
        else visit_objective(states, grid, target_box, grid.from_index(goals), cfg.problem.arrival_km)
    )
    context = world.readout.context
    if sorted(j for group in cfg.belief.groups for j in group.components) != list(range(COMPONENTS)):
        raise ValueError(
            f"the belief's groups cover {[group.components for group in cfg.belief.groups]} and the field has {COMPONENTS} components"
        )
    belief = (
        # the field of the world the episode runs in, drawn on its key
        OracleBelief.empty(states, cfg.horizon, COMPONENTS, world.reset(k_world)[0]["field"], context)
        if cfg.belief.oracle
        else GroupedBelief(
            states,
            Dataset.empty(cfg.horizon, COMPONENTS, sum(context.values())),  # one row per move, the readings an episode folds in
            tuple(
                GPBelief.empty(
                    states,
                    cfg.horizon,
                    len(group.components),
                    n_context=sum(context.values()),
                    kernel_factors=tuple(BeliefKernel.of(get_class(f.kernel), f.parts, f.lengthscale) for f in group.kernel_factors),
                    context=context,
                    amplitude=jnp.asarray(group.amplitude),
                    noise=jnp.asarray(group.noise),
                    n_features=cfg.belief.n_features,
                    refit=cfg.belief.refit,
                    refit_steps=cfg.belief.refit_steps,
                    prior_mean=None,  # the forecast the reading carries, set at the first reading
                )
                for group in cfg.belief.groups
            ),
            tuple(tuple(group.components) for group in cfg.belief.groups),
        )
    )
    agent: Agent[Any]
    if cfg.method is None:
        # no rule and no update: it claims the prior every step
        agent = RandomAgent(claim=posterior_mean_and_log_variance(belief, jnp.zeros(sum(context.values()))))
    else:
        if cfg.problem.objective != "visit" and cfg.method.candidates == "goals":
            raise ValueError(f"a {cfg.problem.objective} episode draws no goals to aim at")
        utility = instantiate(cfg.method.utility, _target_whitelist_=WHITELIST)
        if isinstance(utility, GoalsReached):  # it counts this episode's goals
            if cfg.problem.objective != "visit":
                raise ValueError(f"a {cfg.problem.objective} episode draws no goals to count")
            utility = dataclasses.replace(utility, goals=tuple(int(g) for g in goals))
        agent = PathEUIAgent(
            candidates=objective.candidates
            if cfg.method.candidates == "box"
            else states.narrow(jnp.zeros(grid.size(), bool).at[goals].set(True)),
            transition=transition,
            belief=belief,
            acquisition=Acquisition(
                utility=utility,
                planner=instantiate(cfg.method.planner, _target_whitelist_=WHITELIST),
                n_field_samples=cfg.method.n_field_samples,
                n_paths_per_field=cfg.method.n_paths_per_field,
                n_candidates_scored=cfg.method.n_candidates_scored,
                subtract_dataset_utility=cfg.method.subtract_dataset_utility,
                penalty_rate=cfg.method.penalty_rate,
            ),
            position_key="position",
            measurement_key="field_reading",
            context_key="context",
            prior_mean_key="forecast" if cfg.belief.forecast_as_prior_mean else None,
            budget_column=None,
            horizon=cfg.horizon,
            random_opening_steps=cfg.method.random_opening_steps,
            claim_every=cfg.claim_every,
        )
    # the settings as plain data, carried onto the record
    return Episode(world, objective, agent, cfg.horizon, dict(cfg.recorded), asdict(cfg), k_world, k_agent, k_steps)
