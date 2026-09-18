"""What the actor sees of a state: where it stands, and a noisy reading of the field there"""

from dataclasses import dataclass
from typing import Any

import jax
from jaxtyping import PRNGKeyArray

from zermelo.interface import BoxDomain, Domain, ProductDomain, Readout
from zermelo.problems.ambient_dynamics.ambient_grid import GridDomain, ambient_positions


@dataclass(frozen=True)
class LocalReading(Readout):
    """`r = f*(z) + sigma_obs * xi` wherever the actor stands, its noise independent of the drift's"""

    ambient: GridDomain
    controllable: GridDomain
    sigma_obs: float

    @property
    def context(self) -> dict[str, int]:
        """The field never moves, so a reading is placed by its position alone"""
        return {}

    @property
    def readings(self) -> Domain:
        """Where the actor stands and what it read there"""
        return ProductDomain({"position": ambient_positions(self.ambient, self.controllable), "reading": BoxDomain((self.ambient.n_axes,))})

    def _read(self, key: PRNGKeyArray, state: dict[str, Any]) -> dict[str, Any]:
        """Where the actor stands, and the field there under N(0, sigma_obs^2) noise on each axis"""
        z = {"ambient": state["ambient"], "controllable": state["controllable"]}
        return {"position": z, "reading": state["field"](z) + self.sigma_obs * jax.random.normal(key, (self.ambient.n_axes,))}

    def reset(self, key: PRNGKeyArray, state: dict[str, Any]) -> dict[str, Any]:
        """Read where the actor starts"""
        return self._read(key, state)

    def step(self, key: PRNGKeyArray, state: dict[str, Any], action: Any, next_state: dict[str, Any]) -> dict[str, Any]:
        """Read where the actor arrived"""
        return self._read(key, next_state)
