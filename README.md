# zermelo

An actor is carried by a flow it cannot see. It steers a little, the field does the rest, and it has to decide
where to go while it is still learning where the flow runs — a stratospheric balloon holding station, an ocean
drifter dropped to sample a current. The name is Zermelo's navigation problem, with the field unknown.

The state is a tree of named parts: some the actor steers, some the field displaces, and one part **is** the
field, which the actor never sees. Every move it emits a pair `(action, claim)` — where to go, and what it now
holds the field to be. The claim can be scored against the truth to evaluate the quality of the posterior estimate.

The package publishes the contracts, problems, methods, and the harness that runs them
across seeds on a cluster.

## Install and check

    pixi install                              # https://pixi.sh
    pixi run check                            # ruff, mypy, import-linter. Static: runs no episode
    pixi run smoke <experiment>               # a sweep end to end and its figures. The only thing that verifies a plot
    pixi run -e cuda smoke-gpu <experiment>   # the same cells as one Slurm array on a gpu node
    pixi run crosscheck <name>                # compare our implementation against standard libraries

Python 3.14 or newer (uses PEP 695 generics)

## Run

    pixi run sweep  <experiment> <name>       # every cell of one sweep, serially, in this process
    pixi run submit <experiment> <name>       # the same cells as one Slurm array
    pixi run draw   <experiment> <target>     # the figures and tables for one sweep
    pixi run render <experiment> <target>     # one recorded episode as a still
    pixi run film   <experiment> <target>     # the same sheets as video, submitted rather than drawn here


    pixi run draw   balloon_waypoint acquisitions_oracle
    pixi run render balloon_waypoint acquisitions_oracle -- --arm all

`pixi task list` names every task. Put `--` before any override or flag:

    pixi run submit balloon_waypoint acquisitions_oracle -- resources.partition=<partition> resources.cpus=2
    pixi run -e cuda submit ambient_waypoint acquisitions -- resources.gres=gpu:1
    pixi run film balloon_waypoint acquisitions_oracle -- --arm <arm> --stride 5 --fps 2

Sweep output is written under `results/<experiment>/<sweep>/<launch timestamp>/`

## Layout

    zermelo/interface/     the contracts. Imports stdlib, jax and jaxtyping
    zermelo/problems/      one package per problem, its modules mirroring the contracts they implement
    zermelo/methods/       one directory per method family, an ABC above its own concretes
    zermelo/run/           the driver, the record and the cluster settings. Generic in problem and in method
    zermelo/experiments/   one package per pairing of a problem with a method family, holding its settings,
                           its `sweeps/`, its `metrics/` and its `render/`
