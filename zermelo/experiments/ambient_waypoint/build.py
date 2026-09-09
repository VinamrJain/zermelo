"""One resolved configuration into the objects an episode runs on"""

from dataclasses import asdict
from typing import Any

import jax
import jax.numpy as jnp
from hydra.utils import get_class, instantiate

from zermelo.experiments.ambient_waypoint.schema import RunConfig
from zermelo.interface import Agent
from zermelo.methods.random_agent import RandomAgent
from zermelo.methods.waypoint_bo.acquisition import Acquisition
from zermelo.methods.waypoint_bo.agent import WaypointAgent, posterior_moments
from zermelo.methods.waypoint_bo.belief import GPBelief, OracleBelief
from zermelo.problems.ambient_dynamics import (
    GP,
    AmbientObjective,
    LocalReading,
    PaddedGridDomain,
    ambient_candidates,
    ambient_positions,
    ambient_world,
)
from zermelo.problems.ambient_dynamics.ambient_transition import AmbientTransition
from zermelo.run.episode import Episode

WHITELIST = ("zermelo.*", "gpjax.*")
"""Which modules a setting may name"""


def assemble(cfg: RunConfig) -> Episode:
    """The episode one configuration describes, that configuration carried onto its record"""
    k_world, k_agent, k_steps = jax.random.split(jax.random.key(cfg.seed), 3)
    ambient = PaddedGridDomain(
        n_axes=cfg.problem.ambient_axes,
        n_cells=cfg.problem.ambient_cells,
        cell_spacing=cfg.problem.ambient_cell_spacing,
        pad=cfg.problem.ambient_pad,
    )
    controllable = PaddedGridDomain(
        n_axes=cfg.problem.controllable_axes,
        n_cells=cfg.problem.controllable_cells,
        cell_spacing=cfg.problem.controllable_cell_spacing,
        pad=cfg.problem.controllable_pad,
    )
    transition = AmbientTransition(
        ambient,
        controllable,
        sigma_flow=cfg.problem.sigma_flow,
        sigma_act=cfg.problem.sigma_act,
        choices_per_axis=cfg.problem.choices_per_axis,
        drift_bound_cells=cfg.problem.drift_bound_cells,
        control_step_cells=cfg.problem.control_step_cells,
        control_bound_cells=cfg.problem.control_bound_cells,
    )
    readout = LocalReading(ambient, controllable, sigma_obs=cfg.problem.sigma_obs)
    candidates = ambient_candidates(ambient, controllable)
    # the kernel takes a variance, and the amplitude is a standard deviation
    truth = get_class(cfg.problem.field_kernel)(lengthscale=cfg.problem.field_lengthscale, variance=cfg.problem.field_amplitude**2)
    field_prior = GP(truth, cfg.problem.field_features)
    world = ambient_world(ambient, controllable, transition, readout, field_prior, candidates)
    objective = AmbientObjective(candidates)
    positions = ambient_positions(ambient, controllable)
    belief = (
        # the field of the world the episode runs in, drawn on its key
        OracleBelief.empty(positions, cfg.horizon, ambient.n_axes, world.reset(k_world)[0]["field"])
        if cfg.belief.oracle
        else GPBelief.empty(
            positions,
            cfg.horizon,  # one row per move, the readings an episode folds in
            ambient.n_axes,
            lengthscale=jnp.full(positions.dim(), cfg.belief.lengthscale),  # one per embedded coordinate, isotropic here
            amplitude=jnp.asarray(cfg.belief.amplitude),
            noise=jnp.asarray(cfg.belief.noise),
            kernel_family=get_class(cfg.belief.kernel),
            n_features=cfg.belief.n_features,
            refit=cfg.belief.refit,
            refit_steps=cfg.belief.refit_steps,
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
            reading_key="reading",
            horizon=cfg.horizon,
            opening_legs=cfg.method.opening_legs,
            claim_every=cfg.claim_every,
        )
    # the settings as plain data, carried onto the record
    return Episode(world, objective, agent, cfg.horizon, asdict(cfg), k_world, k_agent, k_steps)
