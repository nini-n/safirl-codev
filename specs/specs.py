import numpy as np

# Engel yarıçapı (env ile tutarlı)
_OBS_RADIUS = 0.05


def _safe_get(t, key, default=None):
    return t[key] if key in t else default


def stl_robustness(trace, d_min, qdot_max):
    """
    STL-benzeri sağlamlık skorları:
      G_dist  = min_t (dist_obs(t) - d_min)
      G_qdot  = min_t (qdot_max - ||qdot||_inf)
      F_goal  = max_t (-||ee - goal||)
    Dönüş: dict(G_dist, G_qdot, F_goal, violation)
    """
    if len(trace) == 0:
        return {
            "G_dist": -np.inf,
            "G_qdot": -np.inf,
            "F_goal": -np.inf,
            "violation": True,
        }

    g_dist_list, g_qdot_list, f_goal_list = [], [], []

    for t in trace:
        ee = np.asarray(_safe_get(t, "ee", np.zeros(2)), dtype=float)
        goal = np.asarray(_safe_get(t, "goal", np.zeros(2)), dtype=float)
        obs = np.asarray(_safe_get(t, "obs", np.zeros(2)), dtype=float)

        # dist_obs: varsa doğrudan kullan, yoksa ||ee-obs|| - r_obs
        if "dist_obs" in t:
            dist_obs = float(t["dist_obs"])
        else:
            dist_obs = float(np.linalg.norm(ee - obs) - _OBS_RADIUS)

        qdot = np.asarray(_safe_get(t, "qdot", np.zeros(3)), dtype=float)
        qinf = float(np.max(np.abs(qdot))) if qdot.size > 0 else 0.0

        g_dist_list.append(dist_obs - float(d_min))
        g_qdot_list.append(float(qdot_max) - qinf)
        f_goal_list.append(-float(np.linalg.norm(ee - goal)))

    G_dist = float(np.min(g_dist_list))
    G_qdot = float(np.min(g_qdot_list))
    F_goal = float(np.max(f_goal_list))

    violation = (G_dist < 0.0) or (G_qdot < 0.0)
    return {
        "G_dist": G_dist,
        "G_qdot": G_qdot,
        "F_goal": F_goal,
        "violation": violation,
    }


# --- EKSİK OLAN FONKSİYON (robustness'tan violation bayrağını üreten yardımcı) ---
def violation_from_robustness(robustness: dict) -> bool:
    """Uyumluluk için: robustness dict'inden ihlal bayrağı üret."""
    if robustness is None:
        return True
    g1 = float(robustness.get("G_dist", -np.inf))
    g2 = float(robustness.get("G_qdot", -np.inf))
    return (g1 < 0.0) or (g2 < 0.0)
