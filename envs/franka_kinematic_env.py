# envs/franka_kinematic_env.py
"""
Kinematic Franka environment (framework-compatible placeholder).

This is a lightweight, dependency-free version to support quick training/evaluation
loops in scripts. It keeps the same API as the MuJoCo variant:
- reset() -> (obs, info)
- step(a) -> (obs, reward, done, trunc, info)

Observation: 10-dim float32
Action:      2-dim float32 (planar velocity command)
"""

from __future__ import annotations

from typing import Tuple, Dict, Any

import numpy as np


class FrankaKinematicEnv:
    def __init__(
        self,
        episode_len: int = 256,
        seed: int = 0,
        d_min: float | None = None,
        qdot_max: float | None = None,
    ) -> None:
        self.episode_len = int(episode_len)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        # Safety parameters are optional here; if provided, keep them for tracers.
        # If not provided, higher-level scripts may inject defaults.
        self.d_min = float(d_min) if d_min is not None else None
        self.qdot_max = float(qdot_max) if qdot_max is not None else None

        # Simple planar point-mass with velocity control
        self._obs_dim = 10
        self._act_dim = 2

        # Export Gym-like spaces (shape only)
        self.observation_space = type("Box", (), {"shape": (self._obs_dim,)})
        self.action_space = type("Box", (), {"shape": (self._act_dim,)})

        # State: position/velocity (x, y) + goal (xg, yg)
        self._p = np.zeros(2, dtype=np.float32)
        self._v = np.zeros(2, dtype=np.float32)
        self._goal = np.array([0.5, 0.0], dtype=np.float32)

        self._t = 0

    # ----------------------------
    # Gym-like API
    # ----------------------------
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._t = 0
        self._p[:] = self._rng.normal(0.0, 0.05, size=2)
        self._v[:] = 0.0
        self._goal[:] = self._rng.uniform(-0.6, 0.6, size=2)
        obs = self._make_obs()
        info = {"reset": True}
        return obs, info

    def step(self, a: np.ndarray):
        a = np.asarray(a, dtype=np.float32)

        # If qdot_max is defined, clip the action magnitude.
        if self.qdot_max is not None:
            a = np.clip(a, -self.qdot_max, self.qdot_max)

        # Integrate planar dynamics
        dt = 0.05
        self._v = 0.9 * self._v + 0.1 * a
        self._p = self._p + self._v * dt

        # Reward: negative distance to goal minus velocity penalty
        dist = float(np.linalg.norm(self._p - self._goal))
        reward = -dist - 0.01 * float(np.linalg.norm(a))

        # A toy obstacle at (0.0, 0.0); consider proximity as a safety proxy
        d_proxy = float(np.linalg.norm(self._p - np.array([0.0, 0.0], dtype=np.float32)))
        violation = False
        if self.d_min is not None:
            violation = d_proxy < self.d_min

        self._t += 1
        done = False
        trunc = self._t >= self.episode_len

        obs = self._make_obs()

        # Forward robustness hints for tracers (tolerant to None)
        G_dist = 0.0
        if self.d_min is not None:
            G_dist = max(0.0, self.d_min - d_proxy)
        G_qdot = 0.0
        if self.qdot_max is not None:
            G_qdot = max(0.0, float(np.max(np.abs(a))) - self.qdot_max)

        info = {
            "G_dist": float(G_dist),
            "G_qdot": float(G_qdot),
            "F_goal": -dist,
            "violation": bool(violation),
        }
        return obs, float(reward), bool(done), bool(trunc), info

    # ----------------------------
    # Helpers
    # ----------------------------
    def _make_obs(self) -> np.ndarray:
        # [p(2), v(2), goal(2), delta(2), speed, bias] → 10 dims
        delta = self._goal - self._p
        speed = float(np.linalg.norm(self._v))
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        obs[0:2] = self._p
        obs[2:4] = self._v
        obs[4:6] = self._goal
        obs[6:8] = delta
        obs[8] = speed
        obs[9] = 1.0  # bias term
        return obs
