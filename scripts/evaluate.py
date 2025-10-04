# scripts/evaluate.py
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch as th
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.franka_kinematic_env import FrankaKinematicEnv
from envs.franka_mujoco_env import FrankaMujocoEnv
from rl.ppo import PPOAgent
from shield.cbf_qp import CBFShield
from shield.mpc_shield import MPCShield
from verify.robustness import EpisodeTracer


def _append_eval_csv(row: dict, csv_path: str = "runs/eval_metrics.csv"):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header = [
        "episode",
        "return",
        "violation",
        "mean_int_rate",
        "G_dist",
        "G_qdot",
        "F_goal",
    ]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        # eksik anahtar varsa güvenli doldurma
        safe = {k: row.get(k, "") for k in header}
        w.writerow(safe)


row = {
    "episode": ep_idx,  # sen ep index'i nasıl tutuyorsan onu kullan
    "return": ret,
    "violation": bool(violation),
    "mean_int_rate": float(int_rate),  # sende int_rate/int_avg varsa kullan
    "G_dist": float(rob.get("G_dist", "")),
    "G_qdot": float(rob.get("G_qdot", "")),
    "F_goal": float(rob.get("F_goal", "")),
}
_append_eval_csv(row)


def make_env(cfg: dict):
    env_name = str(cfg["env"])
    if env_name == "franka_mujoco":
        return FrankaMujocoEnv(
            d_min=float(cfg["safety"]["d_min"]),
            qdot_max=float(cfg["safety"]["qdot_max"]),
            episode_len=int(cfg["horizon"]),
            seed=int(cfg.get("seed", 0)),
        )
    elif env_name == "franka_kinematic":
        return FrankaKinematicEnv(
            episode_len=int(cfg["horizon"]),
            seed=int(cfg.get("seed", 0)),
        )
    else:
        raise ValueError("env must be 'franka_mujoco' or 'franka_kinematic'")


def make_shield(cfg: dict, override: str | None):
    kind = (override or str(cfg["safety"].get("shield", "none"))).lower()
    if kind == "none":
        return None
    if kind == "cbf":
        return CBFShield(
            alpha=float(cfg["safety"].get("cbf_alpha", 2.0)),
            d_min=float(cfg["safety"]["d_min"]),
            qdot_max=float(cfg["safety"]["qdot_max"]),
        )
    if kind == "mpc":
        return MPCShield(
            horizon=int(cfg["safety"].get("mpc_horizon", 8)),
            rho=float(cfg["safety"].get("mpc_rho", 0.05)),
            d_min=float(cfg["safety"]["d_min"]),
            qdot_max=float(cfg["safety"]["qdot_max"]),
        )
    raise ValueError(f"unknown shield kind: {kind}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default="experiments/base.yaml")
    p.add_argument("--policy", type=str, default="runs/latest.pt")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--shield", type=str, default=None, help="none/cbf/mpc (override)")
    args = p.parse_args()

    with open(args.cfg, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = make_env(cfg)
    shield = make_shield(cfg, args.shield)

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    agent = PPOAgent(
        obs_dim, act_dim, tuple(cfg["hidden_sizes"]), float(cfg["learning_rate"])
    )
    agent.load_state_dict(th.load(args.policy, map_location="cpu"))

    tracer = EpisodeTracer()
    reset_out = env.reset()
    o = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    for ep in range(int(args.episodes)):
        ep_ret, ep_len, interv_n = 0.0, 0, 0
        tracer.buf.clear()

        done = False
        trunc = False
        while not (done or trunc):
            # politika
            a, logp, v = agent.select_action(o)
            a_before = a.copy()

            # shield
            if shield is not None:
                a = shield.project(o, a)
                if not np.allclose(a, a_before, atol=1e-6):
                    interv_n += 1
                    logp, _, _ = agent.evaluate(o, a)

            o2, r, done, trunc, info = env.step(a)
            ep_ret += float(r)
            ep_len += 1

            # robustness tracer
            tracer.add(info)

            o = o2

        # özet
        summ = tracer.summary(
            float(cfg["safety"]["d_min"]), float(cfg["safety"]["qdot_max"])
        )
        int_rate = interv_n / max(1, ep_len)
        int_avg = (
            0.0 if interv_n == 0 else 1.0
        )  # burada müdahale başına büyüklük tutmuyoruz

        print(
            f"[{ep + 1}] ret={ep_ret:.2f}  violation={summ['violation']}  rob={summ['robustness']}  "
            f"int_rate={int_rate:.3f}  int_avg={int_avg:.4f}"
        )

        # sonraki bölüm
        reset_out = env.reset()
        o = reset_out[0] if isinstance(reset_out, tuple) else reset_out


if __name__ == "__main__":
    main()
