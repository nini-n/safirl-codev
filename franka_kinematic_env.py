import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FrankaKinematicEnv(gym.Env):
    """Basit 3-DoF planar kol (Franka benzeri) uç-efektör hedef takibi.
    Durum: [q1,q2,q3, qd1,qd2,qd3, target_x, target_y, obs_x, obs_y]
    Aksiyon: eklem hız komutu (rad/s).
    Güvenlik: engelden d_min kadar uzakta kal.
    """

    metadata = {"render_modes": []}

    def __init__(self, d_min=0.08, qdot_max=1.2, dt=0.02, episode_len=300, seed=0):
        super().__init__()
        self.dt = dt
        self.episode_len = episode_len
        self.step_count = 0
        self.d_min = d_min
        self.qdot_max = qdot_max
        self.link_lengths = np.array([0.3, 0.25, 0.2])
        high_q = np.array([np.pi] * 3, dtype=np.float32)
        high_dq = np.array([qdot_max] * 3, dtype=np.float32)
        high = np.concatenate([high_q, high_dq, [2.0, 2.0, 2.0, 2.0]]).astype(np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-qdot_max, high=qdot_max, shape=(3,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    def fk(self, q):
        l1, l2, l3 = self.link_lengths
        x1 = l1 * np.cos(q[0])
        y1 = l1 * np.sin(q[0])
        x2 = x1 + l2 * np.cos(q[0] + q[1])
        y2 = y1 + l2 * np.sin(q[0] + q[1])
        xe = x2 + l3 * np.cos(q.sum())
        ye = y2 + l3 * np.sin(q.sum())
        return np.array([xe, ye], dtype=np.float32)

    def distance_to_obstacle(self, p):
        obs = self.obs
        return np.linalg.norm(p - obs) - 0.05  # obstacle radius = 5 cm

    def _get_obs(self):
        return np.concatenate([self.q, self.qd, self.goal, self.obs]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.q = self.rng.uniform(low=-0.5, high=0.5, size=(3,)).astype(np.float32)
        self.qd = np.zeros(3, dtype=np.float32)
        self.goal = self.rng.uniform(low=[0.2, -0.3], high=[0.6, 0.6]).astype(np.float32)
        self.obs = self.rng.uniform(low=[0.1, -0.2], high=[0.5, 0.5]).astype(np.float32)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -self.qdot_max, self.qdot_max)
        self.qd = action
        self.q = self.q + self.qd * self.dt
        ee = self.fk(self.q)
        dist_goal = np.linalg.norm(ee - self.goal)
        dist_obs = self.distance_to_obstacle(ee)
        # reward: goal yakınlık - kontrol cezası; güvenlik ihlali cezası
        reward = -dist_goal - 0.01 * np.square(action).sum()
        safe = dist_obs - self.d_min
        if safe < 0:
            reward -= 5.0 + 10.0 * (-safe)

        self.step_count += 1
        terminated = dist_goal < 0.03
        truncated = self.step_count >= self.episode_len
        info = {
            "ee": ee.copy(),
            "goal": self.goal.copy(),  # <-- EKLENDİ
            "dist_goal": float(dist_goal),
            "dist_obs": float(dist_obs),
            "safe_margin": float(safe),
        }
        return self._get_obs(), reward, terminated, truncated, info
