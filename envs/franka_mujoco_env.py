import os
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --- MuJoCo kurulumu kontrolü ---
try:
    import mujoco
except Exception as e:
    raise RuntimeError(
        "mujoco paketi yüklü değil. `pip install mujoco` ile kur."
    ) from e


# --- XML yolu: dayanıklı çözümleyici ---
def resolve_asset_path() -> str:
    """
    MuJoCo XML dosyasını sağlam şekilde bul:
      1) envs/../assets/franka/franka.xml
      2) cwd/assets/franka/franka.xml
      3) FRANKA_XML ortam değişkeni
    """
    here = os.path.dirname(__file__)
    cand1 = os.path.normpath(os.path.join(here, "..", "assets", "franka", "franka.xml"))
    cand2 = os.path.normpath(
        os.path.join(os.getcwd(), "assets", "franka", "franka.xml")
    )
    cand3 = os.environ.get("FRANKA_XML", "")

    for p in [cand1, cand2, cand3]:
        if p and os.path.exists(p):
            return p

    raise FileNotFoundError(
        "MuJoCo XML bulunamadı. Denenen yollar:\n"
        f"  {cand1}\n  {cand2}\n  {cand3 or '(FRANKA_XML tanımlı değil)'}\n"
        "Lütfen 'assets\\franka\\franka.xml' dosyasının mevcut olduğundan ve uzantısının .xml olduğundan emin olun."
    )


ASSET_PATH = resolve_asset_path()


class FrankaMujocoEnv(gym.Env):
    """
    3-DoF planar Franka-benzeri kol (MuJoCo).
    Aksiyon: eklem hızları (rad/s).
    Gözlem: [q(3), qd(3), goal_x, goal_y, obs_x, obs_y]
    Güvenlik: ee ile obs arasındaki mesafe >= d_min (obs sadece ölçüm; sim nesnesi değil).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        d_min: float = 0.08,
        qdot_max: float = 1.2,
        episode_len: int = 256,
        seed: int = 0,
    ):
        super().__init__()
        self.d_min = float(d_min)
        self.qdot_max = float(qdot_max)
        self.episode_len = int(episode_len)
        self.step_count = 0

        # MuJoCo model/data
        # Dosya gerçekten var mı? (ek güvence)
        if not os.path.exists(ASSET_PATH):
            raise FileNotFoundError(f"MuJoCo XML bulunamadı: {ASSET_PATH}")

        # XML'i Python üzerinden oku ve MuJoCo'ya string olarak ver
        with open(ASSET_PATH, encoding="utf-8") as f:
            xml_text = f.read()

        try:
            self.model = mujoco.MjModel.from_xml_string(xml_text)
        except Exception as e:
            raise RuntimeError(
                f"MuJoCo XML yüklenemedi (from_xml_string). Yol: {ASSET_PATH}\nHata: {e}"
            ) from e

        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)  # XML'de 0.02 (50 Hz)

        # Eklem indexleri
        self.qpos_idx = [
            self.model.joint(name).qposadr[0] for name in ["j1", "j2", "j3"]
        ]
        self.qvel_idx = [
            self.model.joint(name).dofadr[0] for name in ["j1", "j2", "j3"]
        ]

        # EE site id (sürüm uyumluluğu için iki yöntem)
        try:
            self.ee_sid = self.model.site("ee_site").id
        except Exception:
            self.ee_sid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
            )

        # Uzaylar
        high_q = np.array([np.pi] * 3, dtype=np.float32)
        high_dq = np.array([self.qdot_max] * 3, dtype=np.float32)
        high = np.concatenate([high_q, high_dq, [2.0, 2.0, 2.0, 2.0]]).astype(
            np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.qdot_max, high=self.qdot_max, shape=(3,), dtype=np.float32
        )

        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    # ---------------- Yardımcılar ---------------- #
    def _get_q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].astype(np.float32)

    def _get_qd(self) -> np.ndarray:
        return self.data.qvel[self.qvel_idx].astype(np.float32)

    def get_ee(self) -> np.ndarray:
        # EE dünya koordinatı (x,y) — ileri kinematik
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.ee_sid][:2].astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [self._get_q(), self._get_qd(), self.goal, self.obs]
        ).astype(np.float32)

    def distance_to_obstacle(self, p: np.ndarray) -> float:
        # Engel merkezine öklidyen mesafe - yarıçap (r=0.05m)
        return float(np.linalg.norm(p - self.obs) - 0.05)

    # ---------------- Gym API ---------------- #
    def reset(
        self, seed: Optional[int] = None, options=None
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Başlangıç durumları
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        noise = self.rng.uniform(-0.5, 0.5, size=3)
        self.data.qpos[self.qpos_idx] = noise
        mujoco.mj_forward(self.model, self.data)

        # Hedef & Engel
        self.goal = self.rng.uniform(low=[0.2, -0.3], high=[0.6, 0.6]).astype(
            np.float32
        )
        self.obs = self.rng.uniform(low=[0.1, -0.2], high=[0.5, 0.5]).astype(np.float32)

        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # Aksiyon: eklem hızları (velocity actuators)
        action = np.clip(action, -self.qdot_max, self.qdot_max).astype(np.float64)
        self.data.ctrl[:] = 0.0
        self.data.ctrl[0:3] = action
        mujoco.mj_step(self.model, self.data)

        ee = self.get_ee()
        dist_goal = float(np.linalg.norm(ee - self.goal))
        dist_obs = self.distance_to_obstacle(ee)

        # Ödül: hedefe yakınlık + kontrol cezası + güvenlik cezası
        reward = -dist_goal - 0.01 * float(np.square(action).sum())
        safe_margin = dist_obs - self.d_min
        if safe_margin < 0:
            reward -= 5.0 + 10.0 * (-safe_margin)

        self.step_count += 1
        terminated = dist_goal < 0.03
        truncated = self.step_count >= self.episode_len

        info = {
            "ee": ee.copy(),
            "goal": self.goal.copy(),
            "dist_goal": dist_goal,
            "dist_obs": dist_obs,
            "safe_margin": float(safe_margin),
        }
        return self._get_obs(), reward, terminated, truncated, info
