import dataclasses

from zermelo.experiments.balloon_waypoint.registry import IRMA_JOSE, matched_belief
from zermelo.experiments.balloon_waypoint.schema import BeliefConfig, MethodConfig, Resources
from zermelo.experiments.balloon_waypoint.setup import (
    Implementation,
    arm,
    max_magnitude,
    speed_at,
    sweep,
    upper_confidence,
    value_iteration,
)

WORLD = dataclasses.replace(
    IRMA_JOSE,
    ballast_units=2,
    error_lengthscale_km=600.0,  # wide, so a handful of readings say something about the whole box
    target=speed_at(altitude=0),
    margin_lat=8,
    margin_lon=20,  # a narrow band of candidates, so a decision scores tens of states rather than thousands
)
"""The recorded world at a ballast budget of two, scored over a strip of it"""


def _belief(oracle: bool) -> BeliefConfig:
    """The model an arm holds: matched to this world, or the true wind itself"""
    return dataclasses.replace(matched_belief(WORLD), oracle=oracle, n_features=32, refit_steps=20)


def _method(utility: Implementation, *, improvement: bool, n_fields: int, n_walks: int) -> MethodConfig:
    """One rule on a value-iteration planner, at the settings the arms differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(max_steps=3, radius=120.0, target_chunk=None),
        improvement=improvement,
        n_fields=n_fields,
        n_walks=n_walks,
        combination="linear",
        step_rate=0.5,
        steps_from="predicted",
        n_candidates=16,
        opening_legs=1,  # three uniform moves, leaving three the rule decides
    )


# One entry per device the cells can run on; `balloon_smoke_gpu` asks for an accelerator and a gpu-high node
for name, resources in (
    ("balloon_smoke", Resources(cpus=1, mem_gb=8, timeout_min=20, gres=None, partition="dean", constraint="avx2", array_parallelism=10)),
    (
        "balloon_smoke_gpu",
        Resources(cpus=1, mem_gb=8, timeout_min=20, gres="gpu:1", partition="dean", constraint="avx2&gpu-high", array_parallelism=10),
    ),
):
    sweep(
        name,
        problem=WORLD,
        horizon=6,
        claim_every=1,
        seeds=[0],
        resources=resources,
        arms=[
            arm("mean", _method(max_magnitude(), improvement=True, n_fields=0, n_walks=0), _belief(oracle=False)),
            arm("oracle", _method(max_magnitude(), improvement=True, n_fields=0, n_walks=0), _belief(oracle=True)),
            arm("ucb", _method(upper_confidence(c=2.0), improvement=False, n_fields=0, n_walks=0), _belief(oracle=False)),
            arm("random", None, _belief(oracle=False)),  # no rule at all
        ],
    )
