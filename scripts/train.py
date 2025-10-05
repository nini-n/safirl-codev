# scripts/train.py
"""
PPO training loop (Option B):
- Action, logp, value -> agent.select_action()
- If a shield is used, recompute logp via agent.evaluate() after projection
- Write (obs, act, rew, val, logp) to the buffer in order
- For GAE(λ) at episode end: bootstrap last_val on timeout; otherwise 0
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import yaml

# ---- Add project root to sys.path (safe for imports) ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch as th  # noqa: E402

from envs.franka_kinematic_env import FrankaKinematicEnv  # noqa: E402
from envs.franka_mujoco_env import FrankaMujocoEnv  # noqa: E402
from rl.ppo import PPOAgent, PPOBuffer, ppo_update  # noqa: E402
from shield.cbf_qp import CBFShield  # noqa: E402
from shield.mpc_shield import MPCShield  # noqa: E402
from verify.robustness import EpisodeTracer  # noqa: E402


# ---------------------------
# Env & Shield factories
# ---------------------------
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


def make_shield(cfg: dict):
    kind = str(cfg["safety"].get("shield", "none")).lower()
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


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="experiments/base.yaml")
    parser.add_argument("--steps", type=int, default=20000)
    args = parser.parse_args()

    with open(args.cfg, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # hyperparameters
    horizon = int(cfg["horizon"])
    hidden_sizes = tuple(cfg["hidden_sizes"])
    lr = float(cfg["learning_rate"])
    clip_ratio = float(cfg["clip_ratio"])
    update_epochs = int(cfg["update_epochs"])
    minibatch_size = int(cfg["minibatch_size"])
    gamma = float(cfg.get("gamma", 0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))

    env = make_env(cfg)
    shield = make_shield(cfg)

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    agent = PPOAgent(obs_dim, act_dim, hidden_sizes, lr)
    buf = PPOBuffer(obs_dim, act_dim, horizon)
    tracer = EpisodeTracer()

    total_steps = int(args.steps)
    start_time = time.time()

    # reset obs (gymnasium/gym compatibility)
    reset_out = env.reset()
    o = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    ep_ret, ep_len = 0.0, 0
    interv_n = 0  # intervention count

    os.makedirs("runs", exist_ok=True)
    log_path = os.path.join("runs", "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "t",
                    "ep_len",
                    "ep_ret",
                    "violation",
                    "G_dist",
                    "G_qdot",
                    "F_goal",
                    "int_rate",
                    "int_avg",
                ]
            )

    for t in range(total_steps):
        # 1) Get action, logp, value from the policy
        a, logp, v = agent.select_action(o)
        a_before = a.copy()

        # 2) Shield projection
        if shield is not None:
            a = shield.project(o, a)
            # If the shield changed the action, recompute logp for the new action
            if not np.allclose(a, a_before, atol=1e-6):
                interv_n += 1
                logp, _, _ = agent.evaluate(o, a)

        # 3) Environment step
        o2, r, done, trunc, info = env.step(a)
        ep_ret += float(r)
        ep_len += 1

        # 4) Store to buffer (obs, act, rew, val, logp)
        buf.store(o, a, r, v, logp)

        # 5) Tracer (for robustness computation)
        tracer.add(info)

        # 6) Next observation
        o = o2

        # 7) Episode end?
        timeout = ep_len == horizon
        terminal = bool(done) or bool(trunc) or timeout
        if terminal:
            # Bootstrap on timeout; otherwise zero
            if timeout:
                # Value depends only on observation; action is irrelevant
                zero_act = np.zeros(act_dim, dtype=np.float32)
                last_val = agent.evaluate(o, zero_act)[2]
            else:
                last_val = 0.0

            buf.finish_path(last_val, gamma=gamma, lam=gae_lambda)

            # Robustness summary + lightweight log
            summ = tracer.summary(
                float(cfg["safety"]["d_min"]), float(cfg["safety"]["qdot_max"])
            )
            int_rate = interv_n / max(1, ep_len)
            int_avg = (interv_n / max(1, interv_n)) if interv_n > 0 else 0.0

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        t,
                        ep_len,
                        f"{ep_ret:.3f}",
                        int(summ["violation"]),
                        float(summ["robustness"]["G_dist"]),
                        float(summ["robustness"]["G_qdot"]),
                        float(summ["robustness"]["F_goal"]),
                        f"{int_rate:.3f}",
                        f"{int_avg:.4f}",
                    ]
                )

            print(
                f"ep_len={ep_len:4d}  ep_ret={ep_ret:7.3f}  "
                f"violation={summ['violation']}  "
                f"rob={{'G_dist': {summ['robustness']['G_dist']}, "
                f"'G_qdot': {summ['robustness']['G_qdot']}, "
                f"'F_goal': {summ['robustness']['F_goal']}}}  "
                f"int_rate={int_rate:.3f}  int_avg={int_avg:.4f}"
            )

            # Reset episode counters
            reset_out = env.reset()
            o = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            ep_ret, ep_len, interv_n = 0.0, 0, 0
            tracer.clear()

        # 8) PPO update at every horizon
        if (t + 1) % horizon == 0:
            ppo_update(
                agent,
                buf,
                clip_ratio=clip_ratio,
                update_epochs=update_epochs,
                minibatch_size=minibatch_size,
            )

    # Training finished: save policy
    th.save(agent.state_dict(), "runs/latest.pt")
    print("Saved policy to runs/latest.pt")
    print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
