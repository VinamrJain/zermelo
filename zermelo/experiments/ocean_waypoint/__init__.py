"""Bayesian optimization over waypoints, against the problem where a boat sails a heading through an unknown ocean current.

- `schema` -- the settings one episode is built from.
- `build` -- one resolved configuration into the objects an episode runs on.
- `setup` -- what a sweep is written with.
- `registry` -- the world every study runs on.
- `sweeps` -- one module per sweep.
- `__main__` -- `python -m zermelo.experiments.ocean_waypoint +sweep=<name>`.
"""
