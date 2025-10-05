import numpy as np
from shield.cbf_qp import CBFShield
from shield.mpc_shield import MPCShield

def test_cbf_project_basic():
    # Kinematic-like observation: position at obs[0:2], obstacle at (0,0)
    obs = np.array([0.01, 0.01] + [0.0] * 8, dtype=np.float32)
    a = np.array([0.5, 0.5], dtype=np.float32)
    sh = CBFShield(alpha=2.0, d_min=0.08, qdot_max=0.3)
    a_proj = sh.project(obs, a)
    assert np.all(np.abs(a_proj) <= 0.3 + 1e-6)
    # repulsion pushes away from (0,0)
    assert not np.allclose(a_proj[:2], a[:2])

def test_mpc_project_basic():
    # Mujoco-like observation: ee at obs[14:16], obstacle at (0.2, 0.0)
    obs = np.zeros(16, dtype=np.float32)
    obs[14:16] = np.array([0.19, 0.0], dtype=np.float32)
    a = np.array([0.6, 0.0, 0, 0, 0, 0, 0], dtype=np.float32)
    sh = MPCShield(horizon=6, rho=0.05, d_min=0.08, qdot_max=0.5, dt=0.02)
    a_proj = sh.project(obs, a)
    assert np.all(np.abs(a_proj) <= 0.5 + 1e-6)
    assert not np.allclose(a_proj[:2], a[:2])
