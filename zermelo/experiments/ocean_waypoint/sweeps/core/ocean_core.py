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
from zermelo.experiments.ocean_waypoint.setup import Implementation, arm, sweep, uniform, value_iteration


def _method(utility: Implementation, *, n_candidates: int | None) -> MethodConfig:
    """One rule on a value-iteration planner, only the candidates within reach scored"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(
            backup_iterations=HORIZON_LENGTH, replan_every=PLANNING_BUDGET, radius=ARRIVAL_RADIUS_KM, target_chunk=None, cost_weight=0.0
        ),
        improvement=True,
        n_fields=0,
        n_walks=0,
        penalty_rate=0.0,
        n_candidates=n_candidates,
        opening_steps=0,
    )


def _direct() -> MethodConfig:
    """Waypoint = the target, re-solved under the belief's mean every `L` moves"""
    return _method(uniform(), n_candidates=0)


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
        arm("rand_wp", _method(uniform(), n_candidates=None), ocean_current_belief()),
        arm("rand_act", None, ocean_current_belief()),  # no rule at all
        arm("oracle", _direct(), oracle_belief()),
    ],
)
