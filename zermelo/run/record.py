"""What one episode leaves behind"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

RENAMED_CHANNELS = {
    "agent_state/time_steps_since_waypoint": "agent_state/steps_since_target",
    "agent_state/scores.waypoint": "agent_state/choice.target",
    "agent_state/scores.imagined_walk_to_waypoint": "agent_state/choice.imagined_path_to_target",
    "agent_state/scores.candidate_indices": "agent_state/choice.candidate_indices",
    "agent_state/scores.acquisition_value": "agent_state/choice.acquisition_value",
    "agent_state/scores.route.act": "agent_state/choice.policy_to_target.action",
    "agent_state/policy.act": "agent_state/choice.policy_to_target.action",
}
"""Channel names a record written before 2026-10-15 holds, and what they are called now; `policy.act` is from before 2026-10"""


@dataclass(frozen=True)
class Record:
    """Record of Everything one episode did."""

    state: dict[str, Any]
    """The true state at each snapshot, less its function-valued parts"""

    observation: dict[str, Any]
    """What the agent saw at each snapshot, one channel per leaf of the reading"""

    agent_state: dict[str, Any]
    """The agent's memory at each snapshot"""

    objective_state: dict[str, Any]
    """The objective's memory at each snapshot"""

    decision: dict[str, Any]
    """The action and the claim at each move"""

    reward: Float[Array, " T"]
    """What the objective scored each move"""

    time_per_decision: Float[Array, " T"]
    """Seconds inside `decide`, measured once the result landed"""

    memory_per_decision: Float[Array, " T"]
    """Bytes of every live jax array in the process once a decision completed"""

    config: dict[str, Any]
    """The fully resolved configuration"""

    terminated: bool
    """True where the objective ended the episode, False where the horizon ran out"""

    time_per_episode: float
    """Seconds from the draw of the instance to the last move"""

    peak_rss_per_process: int
    """Peak resident bytes of the whole process"""

    def save(self, path: Path) -> None:
        """Write `config.json` and `record.npz` under `path`. The npz appears only once fully written"""
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(self.config, indent=2, sort_keys=True))
        blocks = {
            "state": self.state,
            "observation": self.observation,
            "agent_state": self.agent_state,
            "objective_state": self.objective_state,
            "decision": self.decision,
        }
        staged = path / "record.partial.npz"  # must end in .npz, or savez appends it
        np.savez_compressed(
            staged,
            **{f"{block}/{channel}": value for block, columns in blocks.items() for channel, value in columns.items()},
            reward=self.reward,
            time_per_decision=self.time_per_decision,
            memory_per_decision=self.memory_per_decision,
            terminated=self.terminated,
            time_per_episode=self.time_per_episode,
            peak_rss_per_process=self.peak_rss_per_process,
        )
        staged.replace(path / "record.npz")

    @classmethod
    def load(cls, path: Path) -> "Record":
        """The record under `path`, its arrays read back as jax arrays"""
        blocks: dict[str, dict[str, Any]] = {"state": {}, "observation": {}, "agent_state": {}, "objective_state": {}, "decision": {}}
        plain: dict[str, Any] = {}
        with np.load(path / "record.npz") as npz:
            for name in npz.files:
                block, _, channel = RENAMED_CHANNELS.get(name, name).partition("/")
                if channel:
                    blocks[block][channel] = jnp.asarray(npz[name])
                else:
                    plain[name] = npz[name]
        return cls(
            state=blocks["state"],
            observation=blocks["observation"],
            agent_state=blocks["agent_state"],
            objective_state=blocks["objective_state"],
            decision=blocks["decision"],
            reward=jnp.asarray(plain["reward"]),
            time_per_decision=jnp.asarray(plain["time_per_decision"]),
            memory_per_decision=jnp.asarray(plain["memory_per_decision"], dtype=float),
            terminated=bool(plain["terminated"]),  # npz returns a scalar as a 0-d array; the annotation is bool
            time_per_episode=float(plain["time_per_episode"]),
            peak_rss_per_process=int(plain["peak_rss_per_process"]),
            config=json.loads((path / "config.json").read_text()),
        )
