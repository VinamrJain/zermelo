"""Path-EUI: expected utility improvement over the path to a target, this project's own method.

- `belief/` -- `D_n` as a `Dataset`, and a model of the field conditioned on it: a GP, GPs over groups of its components, or the truth.
- `planner` -- the policy_to_target to a target under a guessed field, and the steps it takes.
- `utility` -- `U(D_n)`, what a dataset's observations are worth.
- `acquisition` -- the one score every named rule is a setting of, and which candidate wins.
- `agent` -- the four composed into an `Agent`, plus the posterior it asserts as its claim.
"""
