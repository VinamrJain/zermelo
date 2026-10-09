"""The settings one episode is built from, as dataclasses hydra composes and checks"""

from dataclasses import dataclass, field
from typing import Any, Literal

from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class ProblemConfig:
    """The world one episode runs in"""

    data_path: str = MISSING
    """Where the ocean current and the wind are read from, relative to the package"""

    grid_stride: int = MISSING
    """Every how many of the data's rows and columns the grid keeps, 1 keeping all"""

    frame_hour: float = MISSING
    """The data's hour whose frame the episode runs on, held still"""

    forecast_lag_hours: float = MISSING
    """F = the frame this many hours before `frame_hour`; 0 makes F = W under a drawn error"""

    step_hours: float = MISSING
    """Hours in one time_step"""

    n_headings: int = MISSING
    """Headings the boat may sail, equally spaced round the compass"""

    top_speed_ms: float = MISSING
    """Speed through the water at the best heading in enough wind (m/s)"""

    no_go_degrees: float = MISSING
    """Angle off the wind below which the boat makes no way"""

    peak_degrees: float = MISSING
    """Angle off the wind at which the boat is fastest"""

    rise_exponent: float = MISSING
    """Power the speed rises with from the no-go angle to the peak; below 1 it rises fast at first"""

    downwind_factor: float = MISSING
    """Speed straight downwind as a fraction of the peak"""

    wind_scale_ms: float = MISSING
    """Wind speed at which the boat reaches 1 - 1/e of `top_speed_ms` (0 ignores the wind's speed)"""

    noise_ms: float = MISSING
    """Standard deviation of a reading's error per component (m/s)"""

    error_scale: float = MISSING
    """Standard deviation of a drawn error (m/s)"""

    error_lengthscale_km: float = MISSING
    """How far a drawn error stays correlated (km)"""

    error_jitter: float = MISSING
    """Added to a drawn error's correlation diagonal so it factorises"""

    lat_min: float = MISSING
    lat_max: float = MISSING
    lon_min: float = MISSING
    lon_max: float = MISSING
    """The box of grid points, in degrees, a waypoint may be aimed at"""

    target_lat_min: float = MISSING
    target_lat_max: float = MISSING
    target_lon_min: float = MISSING
    target_lon_max: float = MISSING
    """The box of grid points, in degrees, a target is drawn from"""

    arrival_km: float = MISSING
    """How near the target counts as arrived"""

    start_lat: float = MISSING
    start_lon: float = MISSING
    """Where the boat starts, snapped to the grid"""


@dataclass
class BeliefKernelConfig:
    """One factor of the belief's kernel, over the parts it names"""

    kernel: str = MISSING
    """Dotted path to the stationary kernel family this factor is"""

    parts: list[str] = MISSING
    """Which parts of a coordinate row this reads, named as the problem names them"""

    lengthscale: list[float] = MISSING
    """One per column those parts occupy, in the order `parts` names them"""


@dataclass
class BeliefConfig:
    """A method's own model of the ocean current"""

    oracle: bool = MISSING
    """Whether the method is handed the true ocean current instead of fitting one"""

    kernel_factors: list[BeliefKernelConfig] = MISSING
    """The factors the kernel is a product of"""

    forecast_as_prior_mean: bool = MISSING
    """Whether the belief learns the ocean current about the forecast, or about zero"""

    amplitude: float = MISSING
    """Standard deviation the belief assumes of what it learns"""

    noise: float = MISSING
    """Standard deviation the method assumes a reading has"""

    n_features: int = MISSING
    refit: bool = MISSING

    refit_steps: int = MISSING
    """Gradient steps taken per refit"""


@dataclass
class MethodConfig:
    """One fully specified waypoint rule"""

    utility: Any = MISSING
    """Which quantity of a belief is maximized, as a dotted path with its own arguments"""

    planner: Any = MISSING
    """Which policy carries the balloon to a waypoint, as a dotted path with its own arguments"""

    n_fields: int = MISSING
    """Field draws averaged over (zero reads the posterior mean alone)"""

    n_walks: int = MISSING
    """Walks rolled per candidate (zero prices travel from the planner's own table)"""

    improvement: bool = MISSING
    """Whether a candidate is scored by the increment it would add rather than its level"""

    step_rate: float | None = MISSING
    """`lambda`: a full-budget trip costs this share of the spread of the candidate values"""

    steps_from: Literal["predicted", "rolled"] = MISSING
    """Whether travel is priced from the planner's table or from the walks that were rolled"""

    combination: Literal["linear", "fraction", "steps"] = MISSING
    """Whether travel subtracts from value at `step_rate`, divides it, or subtracts as it is, the value being in steps"""

    n_candidates: int | None = MISSING
    """Cap on candidates scored per move (None scores every one, 0 the target alone)"""

    information: Any = MISSING
    """What the readings along a walk tell, as a dotted path with its own arguments, or None to add nothing"""

    information_rate: float | None = MISSING
    """`lambda`: the most informative candidate's information is worth this share of the spread of the candidate values"""

    opening_steps: int = MISSING
    """time_steps of uniform random walking taken before the rule starts"""


@dataclass
class Resources:
    """What one cell of a sweep is given to run in"""

    cpus: int = MISSING

    mem_gb: int = MISSING

    timeout_min: int = MISSING
    """Wall clock in minutes a cell is allowed before the scheduler evicts it"""

    gres: str | None = MISSING
    """Generic resources a cell asks the scheduler for, such as `gpu:1` (None asks for none)"""

    partition: str = MISSING
    """Scheduler partition the runs are submitted to"""

    constraint: str | None = MISSING
    """Node features a cell demands of the scheduler, joined by `&` and `|` (None demands none)"""

    array_parallelism: int = MISSING
    """Cells of one array allowed to run at once, the rest queueing behind them"""


@dataclass
class RunConfig:
    """Everything one episode needs"""

    defaults: list[Any] = field(default_factory=lambda: ["_self_"])
    """Hydra's defaults list, naming no group: a run names its arm and its seed on the command line"""

    problem: ProblemConfig = MISSING

    belief: BeliefConfig = MISSING

    method: MethodConfig | None = MISSING
    """The waypoint rule's settings (None is an arm that acts uniformly and chooses no waypoint)"""

    horizon: int = MISSING
    """Moves an episode may make"""

    claim_every: int = MISSING
    """Moves between re-readings of the claim off a belief"""

    recorded: dict[str, int] = MISSING
    """What the record stores, and the moves between the snapshots it keeps of each"""

    resources: Resources = MISSING
    """What the run was given, carried onto the record"""

    seed: int = MISSING
    """The one number an episode replays from"""


def resolve(composed: DictConfig) -> RunConfig:
    """One composed configuration into the dataclasses it was checked against"""
    settings = OmegaConf.to_object(composed)
    if not isinstance(settings, RunConfig):
        raise TypeError(f"a composed configuration should be the schema it was registered against, and this is {type(settings).__name__}")
    return settings
