# scripts/benchmark.py
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch as th
import yaml

# Add project root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.franka_kinematic_env import FrankaKinematicEnv
from envs.franka_mujoco_env import FrankaMujocoEnv
from rl.ppo import PPOAgent
from shield.cbf_qp import CBFShield
from shield.mpc_shield import MPCShield
from verify.robustness import EpisodeTracer


def make_env(cfg: dict):
    env_name = str(cfg["env"])
    if env_name == "franka_mujoco":
        return FrankaMujocoEnv(
            d_min=float(cfg["safety"]["d_min"]),
            qdot_max=float(cfg["safety"]["qdot_max"]),
            episode_len=int(cfg["horizon"]),
            seed=int(cfg.get("seed, 0").split(",")[0]) if isinstance(cfg.get("seed", 0), str) else int(cfg.get("seed", 0)),
        )
    elif env_name == "franka_kinematic":
        return FrankaKinematicEnv(
            episode_len=int(cfg["horizon"]),
            seed=int(cfg.get("seed", 0)),
        )
    else:
        raise ValueError("env must be 'franka_mujoco' or 'franka_kinematic'")


def make_shield(cfg: dict, kind: str | None):
    k = (kind or "none").lower()
    if k == "none":
        return None
    if k == "cbf":
        return CBFShield(
            alpha=float(cfg["safety"].get("cbf_alpha", 2.0)),
            d_min=float(cfg["safety"]["d_min"]),
            qdot_max=float(cfg["safety"]["qdot_max"]),
        )
    if k == "mpc":
        return MPCShield(
            horizon=int(cfg["safety"].get("mpc_horizon", 8)),
            rho=float(cfg["safety"].get("mpc_rho", 0.05)),
            d_min=float(cfg["safety"]["d_min"]),
            qdot_max=float(cfg["safety"]["qdot_max"]),
        )
    raise ValueError(f"unknown shield: {kind}")


def eval_once(env, agent, shield):
    tracer = EpisodeTracer()
    reset_out = env.reset()
    o = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    ep_ret, ep_len, interv_n = 0.0, 0, 0
    done = False
    trunc = False

    while not (done or trunc):
        # Policy action
        a, logp, v = agent.select_action(o)
        a_before = a.copy()

        # Shield projection
        if shield is not None:
            a = shield.project(o, a)
            if not np.allclose(a, a_before, atol=1e-6):
                interv_n += 1
                # Re-evaluate logp after shielding
                logp, _, _ = agent.evaluate(o, a)

        o2, r, done, trunc, info = env.step(a)
        ep_ret += float(r)
        ep_len += 1

        tracer.add(info)
        o = o2

    # Episode summary
    # If the env exposes d_min / qdot_max use them; otherwise fall back to cfg
    d_min = getattr(env, "d_min", None)
    qdot_max = getattr(env, "qdot_max", None)
    if d_min is None or qdot_max is None:
        # Safe defaults; tracer.summary doesn't accept None
        d_min = 0.08 if d_min is None else d_min
        qdot_max = 0.8 if qdot_max is None else qdot_max

    summ = tracer.summary(float(d_min), float(qdot_max))
    int_rate = interv_n / max(1, ep_len)
    int_avg = 0.0 if interv_n == 0 else 1.0  # we do not compute per-intervention magnitude here

    return {
        "ret": ep_ret,
        "violation": int(summ["violation"]),
        "int_rate": float(int_rate),
        "int_avg": float(int_avg),
        "G_dist": float(summ["robustness"]["G_dist"]),
        "G_qdot": float(summ["robustness"]["G_qdot"]),
    }


def run_benchmark(cfg, policy_path, episodes, variants):
    env = make_env(cfg)

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    agent = PPOAgent(
        obs_dim,
        act_dim,
        tuple(cfg["hidden_sizes"]),
        float(cfg["learning_rate"]),
    )
    agent.load_state_dict(th.load(policy_path, map_location="cpu"))

    rows = []
    for name in variants:
        shield = make_shield(cfg, name if name != "no_shield" else "none")

        rets, vios, int_rates, G_dist, G_qdot = [], [], [], [], []
        for _ in range(episodes):
            m = eval_once(env, agent, shield)
            rets.append(m["ret"])
            vios.append(m["violation"])
            int_rates.append(m["int_rate"])
            G_dist.append(m["G_dist"])
            G_qdot.append(m["G_qdot"])

        mean_ret = float(np.mean(rets))
        viol_sum = int(np.sum(vios))
        mean_ir = float(np.mean(int_rates))
        mean_d = float(np.mean(G_dist))
        mean_q = float(np.mean(G_qdot))

        print(
            f"{name:>9}  mean_ret={mean_ret:.2f}  viol={viol_sum}/{episodes}  int_rate={mean_ir:.3f}  G_dist={mean_d:.3f}  G_qdot={mean_q:.3f}"
        )
        rows.append([name, episodes, mean_ret, viol_sum, mean_ir, mean_d, mean_q])

    # Save
    os.makedirs("runs", exist_ok=True)

    # Aggregate summary file (per-episode summary is already printed)
    with open(os.path.join("runs", "benchmark_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "episodes",
                "mean_return",
                "violations",
                "int_rate",
                "G_dist",
                "G_qdot",
            ]
        )
        for r in rows:
            w.writerow(r)

    # If you need more detail later, keep per-episode logs separately; we only write aggregates here.
    # Use the same header names for compatibility.
    with open(os.path.join("runs", "benchmark.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "episodes",
                "mean_return",
                "violations",
                "int_rate",
                "G_dist",
                "G_qdot",
            ]
        )
        for r in rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default="experiments/base.yaml")
    p.add_argument("--policy", type=str, default="runs/latest.pt")
    p.add_argument("--episodes", type=int, default=8)
    args = p.parse_args()

    with open(args.cfg, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_benchmark(cfg, args.policy, int(args.episodes), ["no_shield", "cbf", "mpc"])


if __name__ == "__main__":
    main()
