"""The settings one episode is built from, as dataclasses hydra composes and checks"""

from dataclasses import dataclass, field
from typing import Any, Literal

from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class ProblemConfig:
    """The world one episode runs in"""

    wind_path: str = MISSING
    """Where the wind is read from, relative to the package"""

    frame: int = MISSING
    """Which hour of the record the episode runs at, taking W and F there"""

    forecast: Literal["GEFS", "gaussian"] = MISSING
    """F = the forecast the record holds, or W itself under a drawn error"""

    resource_units: int = MISSING
    """`r_0`: altitude changes the balloon can afford over the whole episode"""

    step_hours: float = MISSING
    """`h`: hours the wind carries the balloon for in one step"""

    error_scale: float = MISSING
    """Standard deviation of a drawn error (m/s)"""

    error_lengthscale_km: float = MISSING
    """How far a drawn error stays correlated (km)"""

    error_jitter: float = MISSING
    """Added to a drawn error's correlation diagonal so it factorises"""

    readout: str = MISSING
    """Dotted path to what the balloon measures of the wind"""

    target: Any = MISSING
    """Which scalar of the wind the episode is scored on predicting, as a dotted path with its own arguments"""

    margin_lat: int = MISSING
    """Rows at each edge of the grid that no candidate sits in"""

    margin_lon: int = MISSING
    """Columns at each edge of the grid that no candidate sits in"""


@dataclass
class BeliefConfig:
    """A method's own model of the wind"""

    oracle: bool = MISSING
    """Whether the method is handed the true wind instead of fitting one"""

    kernel: str = MISSING
    """Dotted path to the kernel family the method assumes"""

    lengthscale_km: float = MISSING
    """How far, horizontally, the method assumes a forecast error stays correlated"""

    lengthscale_altitude_km: float = MISSING
    """The same, vertically"""

    amplitude: float = MISSING
    """How wrong the method assumes the forecast is, as a standard deviation"""

    noise: float = MISSING
    """Standard deviation the method assumes a reading has"""

    forecast_prior: bool = MISSING
    """Whether the forecast is the belief's mean before any reading, or zero is"""

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

    combination: Literal["linear", "fraction"] = MISSING
    """Whether travel subtracts from value or divides it"""

    n_candidates: int | None = MISSING
    """Cap on candidates scored per move (None scores every one)"""

    opening_legs: int = MISSING
    """Waypoint legs of uniform random walking taken before the rule starts"""


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
    """Scheduler partition the cells are submitted to"""

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

    resources: Resources = MISSING
    """What the cell was given, carried onto the record"""

    seed: int = MISSING
    """The one number an episode replays from"""


def resolve(composed: DictConfig) -> RunConfig:
    """One composed configuration into the dataclasses it was checked against"""
    settings = OmegaConf.to_object(composed)
    if not isinstance(settings, RunConfig):
        raise TypeError(f"a composed configuration should be the schema it was registered against, and this is {type(settings).__name__}")
    return settings
