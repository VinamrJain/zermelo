"""The paper's rules on a five by four world, six moves each"""

import dataclasses

from zermelo.experiments.ambient_waypoint.registry import RECORDED
from zermelo.experiments.ambient_waypoint.schema import MethodConfig, Resources
from zermelo.experiments.ambient_waypoint.setup import arm, sweep, value_iteration
from zermelo.experiments.ambient_waypoint.sweeps.paper.method import mc_ei, ours
from zermelo.experiments.ambient_waypoint.sweeps.smoke.smoke import WORLD, _belief


def _small(method: MethodConfig) -> MethodConfig:
    """`method` with two draws, and a four-step planner"""
    return dataclasses.replace(
        method,
        planner=value_iteration(replan_length=4, radius=0.05, targets_per_solve=None),
        n_field_samples=2,
        n_paths_per_field=min(method.n_paths_per_field, 2),
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
        arm("mc_ei", _small(mc_ei()), _belief(oracle=False)),
        arm("esi", _small(ours("esi")), _belief(oracle=False)),
        arm("oracle_emi", _small(ours("emi")), _belief(oracle=True)),
    ],
)
