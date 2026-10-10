import dataclasses

from zermelo.experiments.ocean_waypoint.registry import (
    ARRIVAL_RADIUS_KM,
    CLAIM_EVERY_STEPS,
    GULF_STREAM_2019JUN,
    HORIZON_LENGTH,
    N_SEEDS,
    PLANNING_BUDGET,
    RECORDED,
    RESOURCES,
    ocean_current_belief,
    oracle_belief,
)
from zermelo.experiments.ocean_waypoint.schema import MethodConfig
from zermelo.experiments.ocean_waypoint.setup import (
    Implementation,
    arm,
    information_gain,
    remaining_trip,
    replanned_trip,
    sweep,
    uniform,
    value_iteration,
)

N_DRAWS = 16
"""`S` fields and `M` walks per field the `eti` rule imagines; `info` rolls `M` walks under one field"""

INFORMATION_RATES = [0.25, 0.5, 1.0, 2.0]
"""`lambda` of each `info` arm; at 0 the rule picks a point on the mean route and is `direct`"""

CHUNK = 256
"""Imagined datasets the `eti` rule solves in one batch"""


def _method(
    utility: Implementation,
    *,
    n_fields: int,
    n_walks: int,
    n_candidates: int | None,
    information: Implementation | None,
    information_rate: float | None,
) -> MethodConfig:
    """One rule on a value-iteration planner, travel priced in steps, hitting costs truncated at the horizon"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(
            replan_every=PLANNING_BUDGET, backups=HORIZON_LENGTH, radius=ARRIVAL_RADIUS_KM, target_chunk=None, cost_weight=0.0
        ),
        improvement=True,  # the trip utilities condition on the collected readings and the imagined ones together
        n_fields=n_fields,
        n_walks=n_walks,
        combination="steps",
        step_rate=None,
        n_candidates=n_candidates,
        information=information,
        information_rate=information_rate,
        opening_steps=0,
    )


REMAINING = remaining_trip(backups=HORIZON_LENGTH, radius=ARRIVAL_RADIUS_KM, target_chunk=None, cost_weight=0.0)
"""-H_mu(end of walk, target), the trip left under the posterior mean"""


def _direct() -> MethodConfig:
    """Waypoint = the target, re-solved under the belief's mean every `L` moves"""
    return _method(REMAINING, n_fields=0, n_walks=0, n_candidates=0, information=None, information_rate=None)


sweep(
    "core",
    problem=GULF_STREAM_2019JUN,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=list(range(N_SEEDS)),
    resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high"),  # 48 GB cards and up
    arms=[
        arm("direct", _direct(), ocean_current_belief()),
        # -[H_mu(z, x) + H_mu(x, g)] + lambda * (spread of that) / (spread of I) * I(x), I the information along the walk to x
        *[
            arm(
                f"info_{rate:g}",
                _method(
                    REMAINING,
                    n_fields=0,
                    n_walks=N_DRAWS,
                    n_candidates=None,
                    information=information_gain(GULF_STREAM_2019JUN.noise_ms),
                    information_rate=rate,
                ),
                ocean_current_belief(),
            )
            for rate in INFORMATION_RATES
        ],
        # -E over draws f and walks tau of [ cost(tau) + H^{pi(D + obs(tau))}_f(end of tau, g) ]: the replanned trip, scored under the draw.
        # Over the whole grid one episode runs past 100 minutes; left out until the candidate set is cut.
        # arm(
        #     "eti",
        #     _method(
        #         replanned_trip(backups=HORIZON_LENGTH, radius=ARRIVAL_RADIUS_KM, target_chunk=None, cost_weight=0.0, chunk=CHUNK),
        #         n_fields=N_DRAWS,
        #         n_walks=N_DRAWS,
        #         n_candidates=None,
        #         information=None,
        #         information_rate=None,
        #     ),
        #     ocean_current_belief(),
        # ),
        arm(
            "rand_wp",
            _method(uniform(), n_fields=0, n_walks=0, n_candidates=None, information=None, information_rate=None),
            ocean_current_belief(),
        ),
        arm("rand_act", None, ocean_current_belief()),  # no rule at all
        arm("oracle", _direct(), oracle_belief()),
    ],
)
