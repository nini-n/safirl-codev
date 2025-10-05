# specs/specs.py
"""
Specification helpers for safety and robustness.

This module defines a small set of convenience functions and data structures
used by training/evaluation code to reason about:
- constraint margin on distance proxies (G_dist),
- joint-velocity limit (G_qdot),
- progress toward goal (F_goal),
- and whether a violation occurred (boolean).

It is deliberately dependency-free and keeps the field names consistent with:
  * envs/franka_mujoco_env.py
  * envs/franka_kinematic_env.py
  * shield/cbf_qp.py
  * shield/mpc_shield.py
  * verify/robustness.py (tracers that expect these keys)

All functions are pure and safe to call from hot loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class SafetyThresholds:
    """Thresholds used by the specs (units are environment-dependent)."""
    d_min: float = 0.08     # minimum allowed distance to the obstacle/proxy
    qdot_max: float = 0.8   # per-dimension max velocity magnitude


@dataclass
class RobustnessTerms:
    """
    Aggregated robustness terms.

    Attributes
    ----------
    G_dist : float
        Positive value indicates penetration into the unsafe margin for the distance proxy.
        Zero means outside (safe).
    G_qdot : float
        Positive value indicates the action exceeded qdot_max elementwise.
        Zero means within limits.
    F_goal : float
        Goal-progress term (higher is better). Many scripts use negative distance, so
        this is typically <= 0.
    violation : bool
        True if any safety violation was detected by the environment during the step/episode.
    """
    G_dist: float = 0.0
    G_qdot: float = 0.0
    F_goal: float = 0.0
    violation: bool = False


# ---------------------------
# Core spec computations
# ---------------------------

def compute_G_dist(d_proxy: float, d_min: float) -> float:
    """
    Compute distance-margin violation term.

    Parameters
    ----------
    d_proxy : float
        Distance proxy to the obstacle (environment-defined).
    d_min : float
        Minimum allowed distance.

    Returns
    -------
    float
        Positive value if d_proxy < d_min (inside unsafe margin), else 0.
    """
    return max(0.0, float(d_min) - float(d_proxy))


def compute_G_qdot(max_abs_action: float, qdot_max: float) -> float:
    """
    Compute velocity-limit violation term.

    Parameters
    ----------
    max_abs_action : float
        Maximum absolute action component (e.g., max(|a_i|)).
    qdot_max : float
        Per-dimension velocity bound.

    Returns
    -------
    float
        Positive value if max_abs_action > qdot_max, else 0.
    """
    return max(0.0, float(max_abs_action) - float(qdot_max))


def compute_F_goal(neg_distance_to_goal: float) -> float:
    """
    Pass through a goal-progress term.

    Many environments compute F_goal as negative Euclidean distance to the goal
    so that "higher is better". This function exists mainly for symmetry and
    type clarity.

    Parameters
    ----------
    neg_distance_to_goal : float
        Typically -||x - x_goal||.

    Returns
    -------
    float
        The provided value as float.
    """
    return float(neg_distance_to_goal)


def merge_episode_info(infos: Dict[str, Any], d_min: float, qdot_max: float) -> RobustnessTerms:
    """
    Merge a raw environment `info` dict into a typed RobustnessTerms structure.

    This is tolerant to missing keys and defaults to zeros/False when a key
    is not present. Use this when you want a stable object downstream.

    Parameters
    ----------
    infos : dict
        Environment-provided info (per-step or per-episode). Expected keys:
        - "G_dist": float (optional)
        - "G_qdot": float (optional)
        - "F_goal": float (optional)
        - "violation": bool (optional)
        If some are missing, they will be recomputed as zero-safe fallbacks.
    d_min : float
        Distance threshold (used only if we need a fallback computation).
    qdot_max : float
        Velocity threshold (used only if we need a fallback computation).

    Returns
    -------
    RobustnessTerms
        Stable robustness snapshot.
    """
    gd = float(infos.get("G_dist", 0.0))
    gq = float(infos.get("G_qdot", 0.0))
    fg = float(infos.get("F_goal", 0.0))
    vio = bool(infos.get("violation", False))

    # Clamp to be safe against NaNs/negatives for "G_*" terms
    gd = max(0.0, gd)
    gq = max(0.0, gq)

    return RobustnessTerms(G_dist=gd, G_qdot=gq, F_goal=fg, violation=vio)


def summarize_episode(terms: Tuple[RobustnessTerms, ...]) -> Dict[str, Any]:
    """
    Aggregate a sequence of RobustnessTerms across an episode.

    The default is to:
      - mark violation=True if *any* step had violation=True,
      - average G_dist and G_qdot over steps,
      - use the last F_goal (you can change this to max/avg if preferred).

    Parameters
    ----------
    terms : tuple[RobustnessTerms, ...]
        Per-step robustness terms across an episode.

    Returns
    -------
    dict
        {
          "violation": bool,
          "robustness": {
              "G_dist": float,
              "G_qdot": float,
              "F_goal": float,
          }
        }
    """
    if not terms:
        return {
            "violation": False,
            "robustness": {"G_dist": 0.0, "G_qdot": 0.0, "F_goal": 0.0},
        }

    n = float(len(terms))
    viol = any(t.violation for t in terms)
    gdist = sum(t.G_dist for t in terms) / n
    gqdot = sum(t.G_qdot for t in terms) / n
    fgoal = terms[-1].F_goal  # or max/avg depending on your analysis preferences

    return {
        "violation": bool(viol),
        "robustness": {"G_dist": float(gdist), "G_qdot": float(gqdot), "F_goal": float(fgoal)},
    }
