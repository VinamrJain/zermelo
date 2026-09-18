"""What an agent sees of the world, and the map that builds it"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from jax.tree_util import register_dataclass
from jaxtyping import PRNGKeyArray

from zermelo.interface.domain import Domain


class Readout(ABC):
    """A state-to-reading map"""

    @property
    @abstractmethod
    def readings(self) -> Domain:
        """The domain the readings live in"""

    @property
    @abstractmethod
    def context(self) -> dict[str, int]:
        """The coordinates a reading carries beyond the cell it was taken at, each named and how wide it is"""

    @abstractmethod
    def reset(self, key: PRNGKeyArray, state: dict[str, Any]) -> Any:
        """What an agent sees before it has acted"""

    @abstractmethod
    def step(self, key: PRNGKeyArray, state: dict[str, Any], action: Any, next_state: dict[str, Any]) -> Any:
        """What an agent sees after acting"""


@register_dataclass
@dataclass(frozen=True)
class Observation:
    """What an agent receives each step, never a state"""

    reading: Any
    """An element of `Readout.readings`"""

    legal_actions: Domain
    """The action domain, narrowed to what is legal at the state it acts from next"""
