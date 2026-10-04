import dataclasses

from zermelo.experiments.balloon_waypoint.registry import IRMA_2017SEP06, RECORDED, static_error_belief
from zermelo.experiments.balloon_waypoint.schema import MethodConfig, Resources
from zermelo.experiments.balloon_waypoint.setup import Implementation, arm, posterior_spread, sum_magnitude, sweep, value_iteration

WORLD = dataclasses.replace(IRMA_2017SEP06, grid_stride=4, resource_units=3)
"""The still frame on a 4 degree grid: 45 rows by 90 columns"""

BELIEF = dataclasses.replace(static_error_belief(), n_features=32)


def _method(utility: Implementation) -> MethodConfig:
    """Expected improvement of `utility` over 2 imagined walks under 2 drawn fields, replanning every 3 time_steps"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(replan_every=3, radius=120.0, target_chunk=None, cost_weight=0.0),
        improvement=True,
        n_fields=2,
        n_walks=2,
        combination="linear",
        step_rate=1.0,
        steps_from="predicted",
        n_candidates=16,
        opening_steps=0,
    )


sweep(
    "static_smoke",
    problem=WORLD,
    horizon=10,
    claim_every=1,
    recorded=RECORDED,
    seeds=[0],
    resources=Resources(cpus=1, mem_gb=8, timeout_min=20, gres=None, partition="dean", constraint="avx2", array_parallelism=10),
    arms=[arm("esi", _method(sum_magnitude()), BELIEF), arm("evi", _method(posterior_spread()), BELIEF), arm("random", None, BELIEF)],
)
