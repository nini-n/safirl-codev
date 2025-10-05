import importlib
import numpy as np
import pytest

mj = importlib.util.find_spec("mujoco")
needs_mujoco = pytest.mark.skipif(mj is None, reason="mujoco package not installed")

if mj is not None:
    from envs.franka_mujoco_env import FrankaMujocoEnv

@needs_mujoco
def test_mujoco_reset_and_step_contract():
    env = FrankaMujocoEnv(d_min=0.08, qdot_max=0.8, episode_len=4, seed=0)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert info.get("reset") is True

    done = trunc = False
    steps = 0
    while not (done or trunc):
        a = np.zeros(env.action_space.shape[0], dtype=np.float32)
        obs, r, done, trunc, info = env.step(a)
        assert {"G_dist", "G_qdot", "F_goal", "violation"} <= set(info.keys())
        steps += 1
    assert trunc is True and steps == 4
