"""Bayesian optimization over waypoints: this project's own method.

- `belief/` -- `D_n` as a `Dataset`, and a model of the field conditioned on it: a GP, GPs over groups of its components, or the truth.
- `planner` -- the route to a waypoint under a guessed field, and the steps it takes.
- `utility` -- `U(D_n)`, what a set of readings is worth.
- `acquisition` -- the one score every named rule is a setting of, and which candidate wins.
- `agent` -- the four composed into an `Agent`, plus the posterior it asserts as its claim.
"""
