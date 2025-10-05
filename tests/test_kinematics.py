import numpy as np
from envs.franka_kinematic_env import FrankaKinematicEnv

def test_reset_shapes_and_info():
    env = FrankaKinematicEnv(episode_len=5, seed=123, d_min=0.1, qdot_max=0.5)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert info.get("reset") is True

def test_step_contract_and_truncation():
    env = FrankaKinematicEnv(episode_len=3, seed=0, d_min=0.1, qdot_max=0.5)
    env.reset()
    done = trunc = False
    steps = 0
    while not (done or trunc):
        a = np.array([10.0, -10.0], dtype=np.float32)  # will be clipped
        obs, r, done, trunc, info = env.step(a)
        assert isinstance(r, float)
        assert {"G_dist", "G_qdot", "F_goal", "violation"} <= set(info.keys())
        steps += 1
    assert trunc is True and steps == 3

def test_velocity_bound_effect():
    env = FrankaKinematicEnv(episode_len=2, seed=0, d_min=0.05, qdot_max=0.2)
    env.reset()
    a = np.array([1.0, 1.0], dtype=np.float32)
    _, _, _, _, info = env.step(a)
    # action clipped -> should not report velocity overflow
    assert info["G_qdot"] <= 1e-6
