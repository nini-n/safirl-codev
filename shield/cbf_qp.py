# shield/cbf_qp.py
"""
Control Barrier Function (CBF)-style action projection (lightweight, dependency-free).

This module provides a minimal, training-friendly shield that adjusts the raw action
to reduce the likelihood of constraint violations. It is intentionally simple and
deterministic so that it can be used as a drop-in guard in research code without
pulling in QP solvers.

API
---
CBFShield(alpha, d_min, qdot_max)
    .project(obs, action) -> np.ndarray

Design notes
------------
- We "infer" a position proxy from the observation:
  * Mujoco stub (envs/franka_mujoco_env.py): obs shape=16, ee_xy ≈ obs[14:16]
    and we use a virtual obstacle at x=+0.2
  * Kinematic stub (envs/franka_kinematic_env.py): obs shape=10, pos ≈ obs[0:2]
    and a virtual obstacle at (0, 0)
- If distance to the obstacle is below d_min, we shrink the action magnitude and
  add a small repulsive component away from the obstacle.
- Regardless, the final action is clipped elementwise to [-qdot_max, qdot_max].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CBFShield:
    alpha: float = 2.0
    d_min: float = 0.08
    qdot_max: float = 0.8

    # ----------------------------
    # Public API
    # ----------------------------
    def project(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32)
        p, obst = _infer_position_and_obstacle(obs)

        # Distance proxy
        d = float(np.linalg.norm(p - obst))

        # Barrier coefficient: grows as we approach the obstacle
        # h = d - d_min; if h<0 we are within the unsafe margin.
        h = d - float(self.d_min)
        if h <= 0.0:
            # Repulsive direction (away from obstacle); fall back to small random if degenerate
            n = p - obst
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-8:
                n = np.array([1.0, 0.0], dtype=np.float32)
                n_norm = 1.0
            n = n / n_norm

            # Scale factor based on how deep we are inside the margin
            depth = max(1e-6, -h)
            repel = self.alpha * depth

            # Apply a mild projection: reduce magnitude and add a push-away component on the first 2 dims
            a_proj = a.copy()
            a_proj[:2] = 0.8 * a_proj[:2] + repel * n[:2]
        else:
            # Smooth attenuation near the boundary (within 2*d_min)
            s = float(np.clip((2.0 * self.d_min) / max(1e-6, d), 0.0, 1.0))
            scale = 1.0 - 0.3 * s
            a_proj = scale * a

        # Elementwise velocity bound
        a_proj = np.clip(a_proj, -self.qdot_max, self.qdot_max)
        return a_proj


# ----------------------------
# Helpers
# ----------------------------
def _infer_position_and_obstacle(obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infer a 2D position proxy and a static obstacle location from the observation.

    Returns
    -------
    (pos_xy, obst_xy) as float32 arrays of shape (2,)
    """
    o = np.asarray(obs, dtype=np.float32)

    # Mujoco stub: obs[14:16] ~ ee (x,y)
    if o.shape[0] >= 16:
        pos = o[14:16].copy()
        obst = np.array([0.2, 0.0], dtype=np.float32)  # virtual obstacle at +x
        return pos, obst

    # Kinematic stub: obs[0:2] ~ (x,y)
    if o.shape[0] >= 2:
        pos = o[0:2].copy()
        obst = np.array([0.0, 0.0], dtype=np.float32)  # obstacle at origin
        return pos, obst

    # Fallback: zero position and obstacle at origin
    return np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)
