import dataclasses

from zermelo.experiments.ocean_waypoint.registry import (
    ARRIVAL_RADIUS_KM,
    CLAIM_EVERY_STEPS,
    GULF_STREAM_2019JUN,
    HORIZON_LENGTH,
    N_SEEDS,
    ONLY_REACHABLE,
    PLANNING_BUDGET,
    RECORDED,
    RESOURCES,
    VALUE_ITERATIONS,
    field_belief,
    oracle_belief,
)
from zermelo.experiments.ocean_waypoint.schema import MethodConfig
from zermelo.experiments.ocean_waypoint.setup import Implementation, arm, sweep, uniform, value_iteration


def _method(utility: Implementation, *, n_candidates_scored: int | None) -> MethodConfig:
    """One rule on a value-iteration planner, only the candidates within reach scored"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(
            value_iterations=VALUE_ITERATIONS,
            replan_length=PLANNING_BUDGET,
            only_reachable=ONLY_REACHABLE,
            radius=ARRIVAL_RADIUS_KM,
            targets_per_solve=None,
            cost_weight=0.0,
        ),
        subtract_dataset_utility=True,
        n_field_samples=0,
        n_paths_per_field=0,
        penalty_rate=0.0,
        n_candidates_scored=n_candidates_scored,
        random_opening_steps=0,
    )


def _direct() -> MethodConfig:
    """Waypoint = the target, re-solved under the belief's mean every `L` moves"""
    return _method(uniform(), n_candidates_scored=0)


sweep(
    "core",
    problem=GULF_STREAM_2019JUN,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=list(range(N_SEEDS)),
    resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high"),  # 48 GB cards and up
    arms=[
        arm("direct", _direct(), field_belief()),
        arm("rand_wp", _method(uniform(), n_candidates_scored=None), field_belief()),
        arm("rand_act", None, field_belief()),  # no rule at all
        arm("oracle", _direct(), oracle_belief()),
    ],
)
