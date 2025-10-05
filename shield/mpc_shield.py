# shield/mpc_shield.py
"""
Simple MPC-style action projection (dependency-free, deterministic).

This shield performs a short-horizon forward check using a very light dynamics
assumption to see whether the proposed action is likely to violate a distance
constraint. If so, it adjusts the action minimally to remain safe.

API
---
MPCShield(horizon, rho, d_min, qdot_max, dt=0.02)
    .project(obs, action) -> np.ndarray

Backward-compatibility
----------------------
Some older code may construct MPCShield with keyword names like:
    MPCShield(d_min=..., alpha=..., qdot_max=..., dt=0.02, horizon=..., rho=...)
We accept these and map them to the new signature. The 'alpha' argument is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Any, Dict

import numpy as np


@dataclass
class MPCShield:
    # Core parameters
    horizon: int = 8
    rho: float = 0.05
    d_min: float = 0.08
    qdot_max: float = 0.8
    dt: float = 0.02

    # Backward-compatible constructor
    def __init__(self, *args, **kwargs):
        # If called with the new signature, dataclass will set attributes below
        # via object.__setattr__ in __post_init__ if we simply populate from kwargs.
        if args:
            # Support positional: (horizon, rho, d_min, qdot_max)
            if len(args) >= 4 and not kwargs:
                self.horizon = int(args[0])
                self.rho = float(args[1])
                self.d_min = float(args[2])
                self.qdot_max = float(args[3])
                self.dt = 0.02
                return

        # Otherwise parse kwargs (both new and legacy)
        defaults: Dict[str, Any] = dict(
            horizon=8, rho=0.05, d_min=0.08, qdot_max=0.8, dt=0.02
        )
        # Legacy compatibility
        if "alpha" in kwargs:
            kwargs.pop("alpha", None)  # no effect in this simplified MPC

        for k, v in {**defaults, **kwargs}.items():
            setattr(self, k, v)

    # ----------------------------
    # Public API
    # ----------------------------
    def project(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, -self.qdot_max, self.qdot_max)

        pos, obst = _infer_position_and_obstacle(obs)

        # Predict future positions under constant-velocity surrogate on first 2 dims.
        # We only use the first two action components as planar velocity commands.
        p = pos.copy()
        v = a[:2].copy()

        danger = False
        worst_d = 1e9
        worst_p = p.copy()

        for _ in range(int(self.horizon)):
            # Simple velocity smoothing to discourage aggressive moves
            v = (1.0 - self.rho) * v + self.rho * a[:2]
            p = p + v * float(self.dt)

            d = float(np.linalg.norm(p - obst))
            if d < worst_d:
                worst_d = d
                worst_p = p.copy()
            if d < self.d_min:
                danger = True

        if not danger:
            return a  # original action is deemed safe enough

        # Compute a minimal adjustment pointing away from the obstacle
        n = worst_p - obst
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            n = np.array([1.0, 0.0], dtype=np.float32)
            n_norm = 1.0
        n = n / n_norm

        # Scale the repulsive component proportional to how much we violate
        depth = max(1e-6, self.d_min - worst_d)
        repel = 1.5 * depth  # mild constant; tuned for stability

        a_proj = a.copy()
        a_proj[:2] = 0.85 * a_proj[:2] + repel * n[:2]

        # Final elementwise clip
        a_proj = np.clip(a_proj, -self.qdot_max, self.qdot_max)
        return a_proj


# ----------------------------
# Helpers
# ----------------------------
def _infer_position_and_obstacle(obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infer a 2D position proxy and a static obstacle location from the observation.

    - Mujoco stub (obs len >= 16): use obs[14:16] as (x, y), obstacle at (0.2, 0.0)
    - Kinematic stub (obs len >= 2): use obs[0:2] as (x, y), obstacle at (0.0, 0.0)
    """
    o = np.asarray(obs, dtype=np.float32)
    if o.shape[0] >= 16:
        pos = o[14:16].copy()
        obst = np.array([0.2, 0.0], dtype=np.float32)
        return pos, obst
    if o.shape[0] >= 2:
        pos = o[0:2].copy()
        obst = np.array([0.0, 0.0], dtype=np.float32)
        return pos, obst
    return np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)
