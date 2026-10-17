import dataclasses

from zermelo.experiments.ocean_waypoint.registry import (
    ARRIVAL_RADIUS_KM,
    CLAIM_EVERY_STEPS,
    GULF_STREAM_2019JUN,
    HORIZON_LENGTH,
    RECORDED,
    RESOURCES,
    STILL_FRAMES,
    field_belief,
)
from zermelo.experiments.ocean_waypoint.schema import MethodConfig
from zermelo.experiments.ocean_waypoint.setup import Implementation, arm, max_magnitude, sweep, total_variance, value_iteration
from zermelo.problems.ocean import OCEAN_CURRENT

FRAMES = ("jun16", "jun28")
"""The frames every arm runs on, one seed each"""

REPLAN_LENGTHS = (25, 50, 100)
"""`L`"""

ROUNDS = {"K1L": 1, "K2L": 2, "Kend": None}
"""`K` as a multiple of `L`, or `None` for the steps left in the episode"""

N_WALKS = 16
"""`M` walks imagined per candidate under each drawn field"""

SEARCH = dataclasses.replace(GULF_STREAM_2019JUN, objective="search")
LEARN = dataclasses.replace(GULF_STREAM_2019JUN, objective="learn", learned="ocean_current")
TASKS = {
    "search": (
        "emi",
        max_magnitude(OCEAN_CURRENT),
        16,
        SEARCH,
        {"objective_state/best_observed_magnitude": 1, "objective_state/best_possible_magnitude": 1},
    ),
    "learn": ("evi", total_variance(OCEAN_CURRENT), 0, LEARN, {"objective_state/information": 1}),
}
"""Per task: the rule's name, its utility over the ocean current, `S` fields drawn, the world, the channels it adds"""


def _method(utility: Implementation, n_field_samples: int, replan_length: int, rounds: int | None) -> MethodConfig:
    """One rule on a value-iteration planner at `L = replan_length`, `K = rounds`, travel not priced"""
    planner = value_iteration(value_iterations=rounds, replan_length=replan_length, radius=ARRIVAL_RADIUS_KM, targets_per_solve=None)
    return MethodConfig(
        utility=utility,
        planner=planner,
        subtract_dataset_utility=True,
        n_field_samples=n_field_samples,
        n_paths_per_field=N_WALKS,
        lambda_=0.0,
        candidates="box",
        n_candidates_scored=None,
    )


for task, (rule, utility, n_field_samples, world, recorded) in TASKS.items():
    for date in FRAMES:
        sweep(
            f"planner_{task}_{date}",
            problem=dataclasses.replace(world, frame_hour=STILL_FRAMES[date]),
            horizon=HORIZON_LENGTH,
            claim_every=CLAIM_EVERY_STEPS,
            recorded={**RECORDED, **recorded},
            seeds=[0],
            resources=dataclasses.replace(RESOURCES, timeout_min=480, constraint="avx2&gpu-high"),  # 48 GB cards and up
            arms=[
                arm(
                    f"{rule}_L{length}_{k}_filter_off",
                    _method(utility, n_field_samples, length, None if times is None else times * length),
                    field_belief(),
                )
                for length in REPLAN_LENGTHS
                for k, times in ROUNDS.items()
            ],
        )
