"""What a boat is handed each time_step

reading = {position, ocean_current, target, forecast}
    position       (position,): the state the ocean current was measured at
    ocean_current  W there plus N(0, noise_ms^2 I), (u, v) in m/s
    target         the grid point to reach
    forecast       F, callable at any position
"""

from dataclasses import dataclass
from typing import Any

import jax
from jaxtyping import PRNGKeyArray

from zermelo.interface import BoxDomain, Domain, FunctionDomain, ProductDomain, Readout
from zermelo.problems.ocean.field import OceanCurrent
from zermelo.problems.ocean.grid import PlaneGrid


@dataclass(frozen=True)
class PointOceanCurrent(Readout):
    """W at the state itself, what a boat reads off its own drift, with the target and F handed over beside it"""

    forecast: OceanCurrent
    """F, the one object handed over at every time_step"""

    states: ProductDomain
    """(position,): the domain a belief over the ocean current is written on"""

    grid: PlaneGrid
    """What a target is a point of"""

    noise_ms: float
    """Standard deviation of the reading's error per component"""

    @property
    def context(self) -> dict[str, int]:
        """The ocean current never moves, so a reading is placed by its position alone"""
        return {}

    @property
    def readings(self) -> Domain:
        """The domain a reading lies in, part by part"""
        return ProductDomain(
            {
                "position": self.states,
                "ocean_current": BoxDomain((2,)),
                "target": self.grid,
                "forecast": FunctionDomain(self.states, BoxDomain((2,))),
            }
        )

    def reset(self, key: PRNGKeyArray, state: dict[str, Any]) -> dict[str, Any]:
        """The reading before acting"""
        return {
            "position": {part: state[part] for part in self.states.parts},
            "ocean_current": state["field"]({"position": state["position"]}) + self.noise_ms * jax.random.normal(key, (2,)),
            "target": state["target"],
            "forecast": self.forecast,
        }

    def step(self, key: PRNGKeyArray, state: dict[str, Any], action: Any, next_state: dict[str, Any]) -> dict[str, Any]:
        """The reading after acting, taken at the state arrived in"""
        return self.reset(key, next_state)
