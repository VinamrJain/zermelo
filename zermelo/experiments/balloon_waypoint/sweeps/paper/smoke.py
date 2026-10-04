"""The paper's rules on a 4 degree still frame, ten time_steps each"""

import dataclasses

from zermelo.experiments.balloon_waypoint.registry import RECORDED, TUNING_FRAME, static_error_belief, still
from zermelo.experiments.balloon_waypoint.schema import MethodConfig, Resources
from zermelo.experiments.balloon_waypoint.setup import arm, sweep, value_iteration
from zermelo.experiments.balloon_waypoint.sweeps.paper.method import mc_ei, ours

BELIEF = dataclasses.replace(static_error_belief(), n_features=32)


def _small(method: MethodConfig, cost_weight: float) -> MethodConfig:
    """`method` with two draws, sixteen candidates and a three-step planner pricing an altitude change at `cost_weight`"""
    return dataclasses.replace(
        method,
        planner=value_iteration(replan_every=3, radius=120.0, target_chunk=None, cost_weight=cost_weight),
        n_fields=2,
        n_walks=min(method.n_walks, 2),
        n_candidates=16,
    )


sweep(
    "paper_smoke",
    problem=dataclasses.replace(still(TUNING_FRAME, 3), grid_stride=4),
    horizon=10,
    claim_every=1,
    recorded=RECORDED,
    seeds=[0],
    resources=Resources(cpus=1, mem_gb=8, timeout_min=20, gres=None, partition="dean", constraint="avx2", array_parallelism=10),
    arms=[
        arm("mc_ei", _small(mc_ei(), 0.0), BELIEF),
        arm("emi", _small(ours("emi"), 1.0), BELIEF),
        arm("oracle_emi", _small(ours("emi"), 0.0), dataclasses.replace(BELIEF, oracle=True)),
    ],
)
