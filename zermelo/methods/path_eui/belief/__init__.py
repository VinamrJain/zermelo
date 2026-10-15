"""`p(f | D_n)`: the observations collected, and a model of the field conditioned on them

- `base` -- the contract. `Dataset` is `D_n`, `FieldEstimate` one field of the belief's (its mean, or a draw), and
  `Belief` what every model answers: `mean_and_variance`, `posterior_covariance`, `mean_field`, `sample_fields`, and `fit` to a new `d`.
- `gp` -- `GPBelief`: exact Gaussian process regression, one kernel shared by every component.
- `grouped` -- `GroupedBelief`: independent beliefs over groups of the components, one `Belief` each.
- `oracle` -- `OracleBelief`: the true field, zero variance.
- `kernel` -- the kernel over coordinate rows, a product of factors over named parts, and its Fourier features.
- `states` -- the finite position domain as a table: elements, coordinates, an index, a field read off a table.
"""

from zermelo.methods.path_eui.belief.base import Belief, Dataset, FieldEstimate
from zermelo.methods.path_eui.belief.gp import GPBelief
from zermelo.methods.path_eui.belief.grouped import GroupedBelief
from zermelo.methods.path_eui.belief.kernel import BeliefKernel
from zermelo.methods.path_eui.belief.oracle import OracleBelief
from zermelo.methods.path_eui.belief.states import coordinates, table_as_field

__all__ = [
    "Belief",
    "FieldEstimate",
    "BeliefKernel",
    "Dataset",
    "GPBelief",
    "GroupedBelief",
    "OracleBelief",
    "coordinates",
    "table_as_field",
]
