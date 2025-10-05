# envs/franka_mujoco_env.py
"""
MuJoCo-based Franka environment (lightweight placeholder).

Notes
-----
- Requires the 'mujoco' Python package. Install with: pip install mujoco
- Expects the XML model at: assets\\franka\\franka.xml
- This is a minimal, training-friendly scaffold so that higher-level scripts run.
  Replace the step dynamics/reward with your project-specific implementation.
"""

from __future__ import annotations

import os
from typing import Tuple, Dict, Any

import numpy as np

# Try importing mujoco early to give a clean error if missing.
try:
    import mujoco  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "The 'mujoco' package is not installed. Install it with `pip install mujoco`."
    ) from e


ASSET_PATH = os.path.join("assets", "franka", "franka.xml")


def _load_xml() -> str:
    """Load MuJoCo XML from common search locations."""
    candidates = [
        ASSET_PATH,
        os.path.join(os.path.dirname(__file__), "..", ASSET_PATH),
        os.path.join(os.getcwd(), ASSET_PATH),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    msg = (
        "MuJoCo XML could not be found. Tried paths:\n- "
        + "\n- ".join(candidates)
        + "\nPlease ensure 'assets\\franka\\franka.xml' exists and has the .xml extension."
    )
    raise FileNotFoundError(msg)


def _model_from_xml(xml_text: str):
    try:
        return mujoco.MjModel.from_xml_string(xml_text)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MuJoCo XML (from_xml_string). Path hint: {ASSET_PATH}\nError: {e}"
        ) from e


class FrankaMujocoEnv:
    """
    Minimal MuJoCo Franka environment with a clean Gym-like API.

    Observation: 16-dim float32 vector
    Action:      7-dim float32 vector (interpreted as joint velocity command)
    Step limit:  episode_len
    Safety:      d_min, qdot_max are kept on the env for robustness computations
    """

    def __init__(
        self,
        d_min: float = 0.08,
        qdot_max: float = 0.8,
        episode_len: int = 256,
        seed: int = 0,
    ) -> None:
        self.d_min = float(d_min)
        self.qdot_max = float(qdot_max)
        self.episode_len = int(episode_len)
        self.seed = int(seed)

        # Load model (lightweight parse to validate XML); not used further in this stub.
        xml = _load_xml()
        _ = _model_from_xml(xml)

        # Basic state
        self._rng = np.random.default_rng(self.seed)
        self._obs_dim = 16
        self._act_dim = 7

        # Expose shapes like Gym
        self.observation_space = type("Box", (), {"shape": (self._obs_dim,)})
        self.action_space = type("Box", (), {"shape": (self._act_dim,)})

        # Goal used by placeholder reward
        self._goal = np.zeros(3, dtype=np.float32)

        # Internal buffers
        self._t = 0
        self._q = np.zeros(self._act_dim, dtype=np.float32)  # joint pos (placeholder)
        self._dq = np.zeros(self._act_dim, dtype=np.float32)  # joint vel (placeholder)

    # ----------------------------
    # Gym-like API
    # ----------------------------
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._t = 0
        self._q[:] = self._rng.normal(0.0, 0.02, size=self._act_dim)
        self._dq[:] = 0.0
        obs = self._make_obs()
        info = {"reset": True}
        return obs, info

    def step(self, a: np.ndarray):
        a = np.asarray(a, dtype=np.float32)
        a = np.clip(a, -self.qdot_max, self.qdot_max)  # enforce qdot_max

        # Placeholder joint integration
        self._dq = 0.9 * self._dq + 0.1 * a
        self._q = self._q + self._dq * 0.02  # dt=20ms

        # Placeholder end-effector position (fake kinematics)
        ee = self._fake_fk(self._q)

        # Reward: move towards goal, penalize speed; zero when at goal
        dist = float(np.linalg.norm(ee - self._goal))
        reward = -dist - 0.01 * float(np.linalg.norm(a))

        # Simple safety proxy (distance to a virtual obstacle at +x)
        d_proxy = abs(ee[0] - 0.2)  # pretend obstacle at x=0.2
        violation = d_proxy < self.d_min

        self._t += 1
        done = False
        trunc = self._t >= self.episode_len

        obs = self._make_obs()

        # info carries robustness-related hints expected by higher-level tracers
        info = {
            "G_dist": max(0.0, self.d_min - d_proxy),  # positive if within unsafe margin
            "G_qdot": max(0.0, float(np.max(np.abs(a))) - self.qdot_max),
            "F_goal": -dist,
            "violation": bool(violation),
        }
        return obs, float(reward), bool(done), bool(trunc), info

    # ----------------------------
    # Helpers
    # ----------------------------
    def _make_obs(self) -> np.ndarray:
        # [q(7), dq(7), ee(2)] → 16 dims (ee uses x,y placeholder)
        ee = self._fake_fk(self._q)
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        obs[:7] = self._q
        obs[7:14] = self._dq
        obs[14:16] = ee[:2]
        return obs

    @staticmethod
    def _fake_fk(q: np.ndarray) -> np.ndarray:
        # Very rough pseudo forward-kinematics for a 7-DOF arm (placeholder)
        x = float(np.sum(np.cos(q)))
        y = float(np.sum(np.sin(q)))
        z = 0.1 * float(np.sum(q))
        return np.array([x, y, z], dtype=np.float32)
