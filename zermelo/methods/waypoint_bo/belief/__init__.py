"""`p(f | D_n)`: the readings collected, and a model of the field conditioned on them

- `base` -- the contract. `Dataset` is `D_n`, `BeliefField` one field of the belief's (its mean, or a draw), and
  `Belief` what every model answers: `predict`, `posterior_covariance`, `mean`, `draw`, and `fit` to a new `D_n`.
- `gp` -- `GPBelief`: exact Gaussian process regression, one kernel shared by every channel.
- `block` -- `BlockBelief`: independent beliefs over consecutive blocks of the channels, one `Belief` each.
- `oracle` -- `OracleBelief`: the true field, zero variance.
- `kernel` -- the kernel over coordinate rows, a product of factors over named parts, and its Fourier features.
- `states` -- the finite position domain as a table: elements, coordinates, an index, a field read off a table.
"""

from zermelo.methods.waypoint_bo.belief.base import Belief, BeliefField, Dataset
from zermelo.methods.waypoint_bo.belief.block import BlockBelief
from zermelo.methods.waypoint_bo.belief.gp import GPBelief
from zermelo.methods.waypoint_bo.belief.kernel import BeliefKernel
from zermelo.methods.waypoint_bo.belief.oracle import OracleBelief
from zermelo.methods.waypoint_bo.belief.states import coordinates, lookup

__all__ = ["Belief", "BeliefField", "BeliefKernel", "BlockBelief", "Dataset", "GPBelief", "OracleBelief", "coordinates", "lookup"]
