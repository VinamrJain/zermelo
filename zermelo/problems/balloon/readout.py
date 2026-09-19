"""What a balloon is handed each time_step

reading = {position, wind, context, forecast}
    position    (position, altitude): the state the wind was measured at
    wind        W there, (u, v) in m/s, measured without noise
    context     (hours_elapsed, balloon_resource): everything else a reading carries, as one row of numbers
    forecast    F, callable at any (position, altitude, hours_elapsed)
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from zermelo.interface import BoxDomain, Domain, FunctionDomain, ProductDomain, Readout
from zermelo.problems.balloon.field import WindField


@dataclass(frozen=True)
class PointWind(Readout):
    """W at the state itself, what a balloon carrying one anemometer reads, with F handed over beside it"""

    forecast: WindField
    """F, the one object handed over at every time_step"""

    states: ProductDomain
    """(position, altitude): the domain a belief over the wind is written on"""

    @property
    def context(self) -> dict[str, int]:
        """hours_elapsed and balloon_resource, one number each"""
        return {"hours_elapsed": 1, "balloon_resource": 1}

    @property
    def readings(self) -> Domain:
        """The domain a reading lies in, part by part"""
        return ProductDomain(
            {
                "position": self.states,
                "wind": BoxDomain((2,)),
                "context": BoxDomain((sum(self.context.values()),)),
                "forecast": FunctionDomain(ProductDomain({**self.states.parts, "hours_elapsed": BoxDomain(())}), BoxDomain((2,))),
            }
        )

    def reset(self, key: PRNGKeyArray, state: dict[str, Any]) -> dict[str, Any]:
        """The reading before acting"""
        return {
            "position": {part: state[part] for part in self.states.parts},
            "wind": state["field"](state),
            "context": jnp.stack([jnp.asarray(state[name], jnp.float32) for name in self.context]),
            "forecast": self.forecast,
        }

    def step(self, key: PRNGKeyArray, state: dict[str, Any], action: Any, next_state: dict[str, Any]) -> dict[str, Any]:
        """The reading after acting, taken at the state arrived in"""
        return self.reset(key, next_state)
