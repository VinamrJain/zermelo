"""The problem, assembled: the state domain with its field part, and the law an episode starts from"""

from dataclasses import dataclass
from typing import Any

import jax
from jaxtyping import PRNGKeyArray

from zermelo.interface import BoxDomain, Domain, FunctionDomain, Prior, ProductDomain, Subset, Uniform, World
from zermelo.problems.ambient_dynamics.field import GP
from zermelo.problems.ambient_dynamics.grid import GridDomain, PaddedGridDomain, ambient_positions
from zermelo.problems.ambient_dynamics.readout import LocalReading
from zermelo.problems.ambient_dynamics.transition import AmbientTransition


@dataclass(frozen=True)
class AmbientPrior(Prior[dict[str, Any]]):
    """The law an episode starts from: a uniform position, and a field drawn from the prior"""

    candidates: Subset[dict[str, Any]]
    """Where a start is drawn from, uniformly"""

    field_prior: GP

    def sample(self, domain: Domain[dict[str, Any]], key: PRNGKeyArray) -> dict[str, Any]:
        """One state: a position drawn uniformly over `candidates`, and a field drawn from `field_prior`"""
        if not isinstance(domain, ProductDomain):
            raise ValueError(f"a state is named parts, and this prior was handed a {type(domain).__name__}")
        k_position, k_field = jax.random.split(key)
        position = Uniform().sample(self.candidates, k_position)  # a dict of the position parts, drawn as one element
        return position | {"field": self.field_prior.sample(domain.parts["field"], k_field)}


def ambient_world(
    ambient: PaddedGridDomain,
    controllable: GridDomain,
    transition: AmbientTransition,
    readout: LocalReading,
    field_prior: GP,
    candidates: Subset[dict[str, Any]],
) -> World:
    """The world these two grids make, raising unless every piece was built on the same two"""
    if any((built.ambient, built.controllable) != (ambient, controllable) for built in (transition, readout)):
        raise ValueError("the transition and the readout must be built on the same two grids this world is assembled from")
    positions = ambient_positions(ambient, controllable)
    if candidates.base != positions:
        raise ValueError("the candidate set must be over the same positions this world is assembled from")
    return World(
        state_domain=ProductDomain(positions.parts | {"field": FunctionDomain(positions, BoxDomain((ambient.n_axes,)))}),
        prior=AmbientPrior(candidates, field_prior),
        transition=transition,
        readout=readout,
    )
