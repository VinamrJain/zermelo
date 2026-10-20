"""The 2D grid ablations: the number of drawn fields and walks, and a belief that is wrong about the field"""

import dataclasses
from typing import Any

from zermelo.experiments.ambient_waypoint.registry import (
    AMBIENT1_CONTROL1,
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    N_SEEDS,
    RECORDED,
    RESOURCES,
    matched_belief,
)
from zermelo.experiments.ambient_waypoint.schema import BeliefConfig
from zermelo.experiments.ambient_waypoint.setup import arm, max_magnitude, sweep, uniform
from zermelo.experiments.ambient_waypoint.sweeps.method import baseline, mc_ei, ours

MATCHED = matched_belief(AMBIENT1_CONTROL1)
(KERNEL,) = MATCHED.kernel_factors


def with_kernel(**replaced: Any) -> BeliefConfig:
    """The matched belief with the named settings of its one kernel factor replaced"""
    return dataclasses.replace(MATCHED, kernel_factors=[dataclasses.replace(KERNEL, **replaced)])


WRONG = {
    "lengthscale1p25": with_kernel(lengthscale=[1.25, 1.25]),  # the truth's is 2.5
    "lengthscale5": with_kernel(lengthscale=[5.0, 5.0]),
    "amplitude1p5": dataclasses.replace(MATCHED, amplitude=1.5),  # the truth's is 3.0, as a standard deviation
    "amplitude6": dataclasses.replace(MATCHED, amplitude=6.0),
}
BELIEFS = {
    "matched": MATCHED,
    **WRONG,
    "matern32": with_kernel(kernel="gpjax.kernels.Matern32"),  # the truth's is Matern52
    **{f"{name}_refit": dataclasses.replace(belief, refit=True) for name, belief in WRONG.items()},
}
"""The belief every arm of the prior study holds, the last four refitting their kernel numbers as they go"""

COUNTS = (0, 1, 4, 8, 16, 32, 64)
"""Fields drawn at 16 walks each, then walks per field at 16 fields; 0 fields is the mean field, 0 walks the target alone"""

for swept, arms, seeds, timeout_min in (
    (
        "paper_fields_paths",
        [
            arm(
                f"{name}_fields{fields}_paths{paths}",
                dataclasses.replace(ours(name), n_field_samples=fields, n_paths_per_field=paths),
                MATCHED,
            )
            for name in ("emi", "evi")
            for fields, paths in dict.fromkeys([(count, 16) for count in COUNTS] + [(16, count) for count in COUNTS])
        ],
        range(5),
        120,
    ),
    (
        "paper_prior_mismatch",
        [
            arm(f"{name}_{held}", method, belief)
            for held, belief in BELIEFS.items()
            for name, method in (
                ("emi", ours("emi")),
                ("evi", ours("evi")),
                ("mc_ei", mc_ei()),
                ("ts", baseline(max_magnitude(), 1)),
                ("rand_target", baseline(uniform(), 0)),
            )
        ],
        range(N_SEEDS),
        60,
    ),
):
    sweep(
        swept,
        problem=AMBIENT1_CONTROL1,
        horizon=HORIZON_LENGTH,
        claim_every=CLAIM_EVERY_STEPS,
        recorded=RECORDED,
        seeds=seeds,
        resources=dataclasses.replace(RESOURCES, mem_gb=16, timeout_min=timeout_min),
        arms=arms,
    )
