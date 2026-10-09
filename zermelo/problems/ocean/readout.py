"""What a boat is handed each time_step

reading = {position, current, forecast}
    position    (position,): the state the current was measured at
    current     W there plus N(0, noise_ms^2 I), (u, v) in m/s
    forecast    F, callable at any position
"""

from dataclasses import dataclass
from typing import Any

import jax
from jaxtyping import PRNGKeyArray

from zermelo.interface import BoxDomain, Domain, FunctionDomain, ProductDomain, Readout
from zermelo.problems.ocean.field import CurrentField


@dataclass(frozen=True)
class PointCurrent(Readout):
    """W at the state itself, what a boat reads off its own drift, with F handed over beside it"""

    forecast: CurrentField
    """F, the one object handed over at every time_step"""

    states: ProductDomain
    """(position,): the domain a belief over the current is written on"""

    noise_ms: float
    """Standard deviation of the reading's error per component"""

    @property
    def context(self) -> dict[str, int]:
        """The current never moves, so a reading is placed by its position alone"""
        return {}

    @property
    def readings(self) -> Domain:
        """The domain a reading lies in, part by part"""
        return ProductDomain(
            {"position": self.states, "current": BoxDomain((2,)), "forecast": FunctionDomain(self.states, BoxDomain((2,)))}
        )

    def reset(self, key: PRNGKeyArray, state: dict[str, Any]) -> dict[str, Any]:
        """The reading before acting"""
        return {
            "position": {part: state[part] for part in self.states.parts},
            "current": state["field"](state) + self.noise_ms * jax.random.normal(key, (2,)),
            "forecast": self.forecast,
        }

    def step(self, key: PRNGKeyArray, state: dict[str, Any], action: Any, next_state: dict[str, Any]) -> dict[str, Any]:
        """The reading after acting, taken at the state arrived in"""
        return self.reset(key, next_state)
