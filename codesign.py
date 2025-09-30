# --- path fix ---
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ---------------

import argparse
import csv
import time

import numpy as np
import torch as th
import yaml

from envs.franka_kinematic_env import FrankaKinematicEnv
from rl.ppo import PPOAgent
from shield.cbf_qp import CBFShield
from shield.mpc_shield import MPCShield
from verify.robustness import EpisodeTracer


# ----------------- yardımcılar -----------------
def make_env(cfg):
    return FrankaKinematicEnv(
        d_min=float(cfg["safety"]["d_min"]), qdot_max=float(cfg["safety"]["qdot_max"])
    )


def load_agent(cfg, policy_path):
    env = make_env(cfg)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    lr = float(cfg["learning_rate"])
    hidden = tuple(int(x) for x in cfg["hidden_sizes"])
    agent = PPOAgent(obs_dim, act_dim, hidden, lr)
    agent.load_state_dict(th.load(policy_path, map_location="cpu"))
    agent.eval()
    return agent, env


def make_shield_from_candidate(safety_base, candidate):
    stype = candidate["shield"]
    if stype == "mpc":
        return MPCShield(
            d_min=float(safety_base["d_min"]),
            alpha=float(candidate["cbf_alpha"]),
            qdot_max=float(safety_base["qdot_max"]),
            dt=0.02,
            horizon=int(candidate["mpc_horizon"]),
            rho=float(candidate["mpc_rho"]),
        )
    elif stype == "cbf":
        return CBFShield(
            d_min=float(safety_base["d_min"]),
            alpha=float(candidate["cbf_alpha"]),
            qdot_max=float(safety_base["qdot_max"]),
        )
    else:
        return None


def eval_candidate(cfg, agent, env, candidate, episodes=5):
    """Aday parametre setini sabit politika ile değerlendir."""
    shield = make_shield_from_candidate(cfg["safety"], candidate)
    returns = []
    vio_count = 0
    int_rates = []
    avg_mags = []
    gdist = []
    gqdot = []
    for ep in range(episodes):
        o, _ = env.reset()
        done = False
        ep_ret = 0.0
        tracer = EpisodeTracer()
        interv_n = 0
        interv_mag = 0.0
        steps = 0
        while not done:
            a_before = agent.act(o)
            a = a_before if shield is None else shield.project(o, a_before)
            diff = float(np.linalg.norm(a - a_before))
            if diff > 1e-8:
                interv_n += 1
                interv_mag += diff
            o, r, term, trunc, info = env.step(a)
            ep_ret += r
            steps += 1
            tracer.add(info, a)
            done = term or trunc
        summ = tracer.summary(float(cfg["safety"]["d_min"]), float(cfg["safety"]["qdot_max"]))
        returns.append(ep_ret)
        vio_count += int(summ["violation"])
        int_rates.append(interv_n / max(1, steps))
        avg_mags.append((interv_mag / max(1, interv_n)) if interv_n > 0 else 0.0)
        gdist.append(summ["robustness"]["G_dist"])
        gqdot.append(summ["robustness"]["G_qdot"])
    out = {
        "mean_return": float(np.mean(returns)),
        "violations": int(vio_count),
        "int_rate": float(np.mean(int_rates)),
        "int_avg": float(np.mean(avg_mags)),
        "G_dist": float(np.mean(gdist)),
        "G_qdot": float(np.mean(gqdot)),
    }
    return out


def score(metrics, w):
    """
    Çok ölçütlü skor (büyüğü iyi).
    Öncelik: ihlal=0. İhlal varsa ağır ceza.
    Sonra return yüksek, müdahale oranı düşük, robustness yüksek.
    """
    if metrics["violations"] > 0:
        return -1e6 * metrics["violations"] - 1e3 * metrics["int_rate"]
    return (
        w["return"] * metrics["mean_return"]
        - w["int_rate"] * metrics["int_rate"]
        + w["gdist"] * metrics["G_dist"]
        + w["gqdot"] * metrics["G_qdot"]
        - w["int_avg"] * metrics["int_avg"]
    )


# --------------- ana arama döngüsü ---------------
def random_search(cfg, policy_path, trials=60, episodes=5, shield="mpc", seed=0, weights=None):
    rng = np.random.default_rng(seed)
    agent, env = load_agent(cfg, policy_path)
    os.makedirs("runs", exist_ok=True)
    log_path = "runs/codesign_log.csv"
    with open(log_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(
            [
                "trial",
                "shield",
                "cbf_alpha",
                "mpc_horizon",
                "mpc_rho",
                "mean_return",
                "violations",
                "int_rate",
                "int_avg",
                "G_dist",
                "G_qdot",
                "score",
            ]
        )

    if weights is None:
        weights = {"return": 1.0, "int_rate": 50.0, "gdist": 5.0, "gqdot": 2.0, "int_avg": 5.0}

    best = None
    best_row = None

    for t in range(1, trials + 1):
        cand = {"shield": shield}
        # Örnekleme aralıkları (mantıklı sınırlar)
        cand["cbf_alpha"] = float(rng.uniform(0.5, 4.0))
        if shield == "mpc":
            cand["mpc_horizon"] = int(rng.integers(6, 16))  # 6..15
            cand["mpc_rho"] = float(rng.uniform(0.0, 0.12))
        else:
            cand["mpc_horizon"] = 0
            cand["mpc_rho"] = 0.0

        metrics = eval_candidate(cfg, agent, env, cand, episodes=episodes)
        s = score(metrics, weights)

        row = [
            t,
            cand["shield"],
            cand["cbf_alpha"],
            cand["mpc_horizon"],
            cand["mpc_rho"],
            metrics["mean_return"],
            metrics["violations"],
            metrics["int_rate"],
            metrics["int_avg"],
            metrics["G_dist"],
            metrics["G_qdot"],
            s,
        ]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        if (best is None) or (s > best):
            best = s
            best_row = (cand, metrics, s)
        print(f"[{t:02d}/{trials}] cand={cand}  metrics={metrics}  score={s:.3f}")

    # En iyi sonucu kaydet
    best_cand, best_metrics, best_score = best_row
    print("\n=== BEST CANDIDATE ===")
    print(best_cand)
    print(best_metrics)
    print(f"score={best_score:.3f}")

    # YAML çıktı
    out_yaml = {
        "safety": {
            "d_min": float(cfg["safety"]["d_min"]),
            "qdot_max": float(cfg["safety"]["qdot_max"]),
            "cbf_alpha": float(best_cand["cbf_alpha"]),
            "shield": best_cand["shield"],
        }
    }
    if best_cand["shield"] == "mpc":
        out_yaml["safety"]["mpc_horizon"] = int(best_cand["mpc_horizon"])
        out_yaml["safety"]["mpc_rho"] = float(best_cand["mpc_rho"])

    with open("runs/best_safety.yaml", "w") as f:
        yaml.safe_dump(out_yaml, f)
    print("Saved best safety config to runs/best_safety.yaml")
    print(f"Full log at {log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="experiments/base.yaml")
    ap.add_argument("--policy", type=str, default="runs/latest.pt")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--shield", type=str, default="mpc", choices=["mpc", "cbf"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    start = time.time()
    random_search(
        cfg,
        args.policy,
        trials=int(args.trials),
        episodes=int(args.episodes),
        shield=args.shield,
        seed=int(args.seed),
    )
    print(f"Total time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
