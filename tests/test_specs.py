# NOTE: floating-point equality -> use math.isclose for robustness.

import math
from specs.specs import (
    RobustnessTerms,
    compute_G_dist,
    compute_G_qdot,
    compute_F_goal,
    merge_episode_info,
    summarize_episode,
)

def test_stl_boundary_ok():
    # distance & velocity margins (use isclose for float safety)
    assert math.isclose(compute_G_dist(0.05, 0.08), 0.03, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(compute_G_dist(0.10, 0.08), 0.0,  rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(compute_G_qdot(0.9, 0.8),  0.1,  rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(compute_G_qdot(0.5, 0.8),  0.0,  rel_tol=1e-12, abs_tol=1e-12)
    assert compute_F_goal(-0.25) == -0.25

    info = {"G_dist": 0.02, "G_qdot": 0.1, "F_goal": -0.5, "violation": True}
    terms = merge_episode_info(info, d_min=0.08, qdot_max=0.8)
    assert terms.G_dist == 0.02 and terms.G_qdot == 0.1 and terms.F_goal == -0.5
    assert terms.violation is True

    ep = [
        RobustnessTerms(G_dist=0.01, G_qdot=0.0, F_goal=-0.7, violation=False),
        RobustnessTerms(G_dist=0.00, G_qdot=0.1, F_goal=-0.6, violation=True),
        RobustnessTerms(G_dist=0.02, G_qdot=0.0, F_goal=-0.4, violation=False),
    ]
    s = summarize_episode(tuple(ep))
    r = s["robustness"]
    assert s["violation"] is True
    assert math.isclose(r["G_dist"], (0.01 + 0.0 + 0.02) / 3, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(r["G_qdot"], (0.0 + 0.1 + 0.0) / 3, rel_tol=1e-12, abs_tol=1e-12)
    assert r["F_goal"] == -0.4
