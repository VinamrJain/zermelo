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
    STILL_FRAMES,
    VALUE_ITERATIONS,
    field_belief,
)
from zermelo.experiments.ocean_waypoint.schema import MethodConfig
from zermelo.experiments.ocean_waypoint.setup import Implementation, arm, max_magnitude, sweep, uniform, upper_confidence, value_iteration
from zermelo.problems.ocean import OCEAN_CURRENT

WORLD = dataclasses.replace(GULF_STREAM_2019JUN, objective="search")
"""The same frame, scored on the fastest ocean current stood on inside the target box, which is also where a target may be aimed"""

N_FIELDS = 16
"""`N` fields drawn from the posterior per decision"""

N_WALKS = 16
"""`M` walks the path-aware rule imagines per candidate under each drawn field"""


def _method(utility: Implementation, *, subtract_dataset_utility: bool, n_field_samples: int, n_paths_per_field: int) -> MethodConfig:
    """One rule on a value-iteration planner, travel not priced"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(
            value_iterations=VALUE_ITERATIONS,
            replan_length=PLANNING_BUDGET,
            radius=ARRIVAL_RADIUS_KM,
            targets_per_solve=None,
            cost_weight=0.0,
        ),
        subtract_dataset_utility=subtract_dataset_utility,
        n_field_samples=n_field_samples,
        n_paths_per_field=n_paths_per_field,
        penalty_rate=0.0,
        candidates="box",
        n_candidates_scored=None,
        random_opening_steps=0,
    )


for date, frame_hour in STILL_FRAMES.items():
    sweep(
        f"search_{date}",
        problem=dataclasses.replace(WORLD, frame_hour=frame_hour),
        horizon=HORIZON_LENGTH,
        claim_every=CLAIM_EVERY_STEPS,
        recorded={**RECORDED, "objective_state/best_observed_magnitude": 1, "objective_state/best_possible_magnitude": 1},
        seeds=list(range(N_SEEDS)),
        resources=dataclasses.replace(RESOURCES, constraint="avx2&gpu-high"),  # 48 GB cards and up
        arms=[
            # E over fields f and walks tau of max(0, max over z in tau of ||f(z)|| - beta_n)
            arm(
                "emi",
                _method(max_magnitude(OCEAN_CURRENT), subtract_dataset_utility=True, n_field_samples=N_FIELDS, n_paths_per_field=N_WALKS),
                field_belief(),
            ),
            # E over fields f of max(0, ||f(x)|| - beta_n) at the destination alone
            arm(
                "mc_ei",
                _method(max_magnitude(OCEAN_CURRENT), subtract_dataset_utility=True, n_field_samples=N_FIELDS, n_paths_per_field=0),
                field_belief(),
            ),
            arm(
                "ucb",
                _method(upper_confidence(2.0, OCEAN_CURRENT), subtract_dataset_utility=False, n_field_samples=0, n_paths_per_field=0),
                field_belief(),
            ),
            arm(
                "ts",
                _method(max_magnitude(OCEAN_CURRENT), subtract_dataset_utility=False, n_field_samples=1, n_paths_per_field=0),
                field_belief(),
            ),
            arm("rand_wp", _method(uniform(), subtract_dataset_utility=False, n_field_samples=0, n_paths_per_field=0), field_belief()),
            arm("rand_act", None, field_belief()),  # no rule at all
        ],
    )
