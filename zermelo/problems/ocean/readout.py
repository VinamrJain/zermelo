"""What a boat is handed each time_step

reading = {position, field_reading, forecast}
    position       (position,): the state the field was measured at
    field_reading  field there plus N(0, diag(noise_ms^2)), (W_u, W_v, U_u, U_v) in m/s
    forecast       the forecast, callable at any position
"""

from dataclasses import dataclass
from typing import Any

import jax
from jaxtyping import Array, Float, PRNGKeyArray

from zermelo.interface import BoxDomain, Domain, FunctionDomain, ProductDomain, Readout
from zermelo.problems.ocean.field import COMPONENTS, GridField


@dataclass(frozen=True)
class PointField(Readout):
    """The field at the state itself, what a boat reads off its own drift and its anemometer, with the forecast handed over beside it"""

    forecast: GridField
    """The one object handed over at every time_step"""

    states: ProductDomain
    """(position,): the domain a belief over the field is written on"""

    noise_ms: Float[Array, " components"]
    """Standard deviation of the reading's error per component"""

    @property
    def context(self) -> dict[str, int]:
        """The field never moves, so a reading is placed by its position alone"""
        return {}

    @property
    def readings(self) -> Domain:
        """The domain a reading lies in, part by part"""
        return ProductDomain(
            {
                "position": self.states,
                "field_reading": BoxDomain((COMPONENTS,)),
                "forecast": FunctionDomain(self.states, BoxDomain((COMPONENTS,))),
            }
        )

    def reset(self, key: PRNGKeyArray, state: dict[str, Any]) -> dict[str, Any]:
        """The reading before acting"""
        return {
            "position": {part: state[part] for part in self.states.parts},
            "field_reading": state["field"]({"position": state["position"]}) + self.noise_ms * jax.random.normal(key, (COMPONENTS,)),
            "forecast": self.forecast,
        }

    def step(self, key: PRNGKeyArray, state: dict[str, Any], action: Any, next_state: dict[str, Any]) -> dict[str, Any]:
        """The reading after acting, taken at the state arrived in"""
        return self.reset(key, next_state)
