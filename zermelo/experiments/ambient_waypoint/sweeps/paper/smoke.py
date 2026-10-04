"""The paper's rules on a five by four world, six moves each"""

import dataclasses

from zermelo.experiments.ambient_waypoint.registry import RECORDED
from zermelo.experiments.ambient_waypoint.schema import MethodConfig, Resources
from zermelo.experiments.ambient_waypoint.setup import arm, sweep, value_iteration
from zermelo.experiments.ambient_waypoint.sweeps.paper.method import mc_ei, ours
from zermelo.experiments.ambient_waypoint.sweeps.smoke.smoke import WORLD, _belief


def _small(method: MethodConfig) -> MethodConfig:
    """`method` with two draws, a four-step planner and four opening moves"""
    return dataclasses.replace(
        method,
        planner=value_iteration(replan_every=4, radius=0.05, target_chunk=None),
        n_fields=2,
        n_walks=min(method.n_walks, 2),
        opening_steps=4,
    )


sweep(
    "paper_smoke",
    problem=WORLD,
    horizon=6,
    claim_every=1,
    recorded={**RECORDED, "objective_state/claim": 1},
    seeds=[0],
    resources=Resources(cpus=1, mem_gb=4, timeout_min=10, gres=None, partition="dean", constraint="avx2", array_parallelism=10),
    arms=[
        arm("mc_ei_charged", _small(mc_ei(0.5)), _belief(oracle=False)),
        arm("esi", _small(ours("esi")), _belief(oracle=False)),
        arm("oracle_emi", _small(ours("emi")), _belief(oracle=True)),
    ],
)
