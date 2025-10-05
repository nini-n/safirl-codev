# scripts/plot_metrics.py
"""
Generate training/evaluation figures from CSV logs under runs/.

Outputs:
  runs/fig_train_return.png
  runs/fig_train_intervention.png
  runs/fig_train_robustness.png
  (optionally) runs/fig_eval_return.png
  (optionally) runs/fig_eval_intervention.png
"""

from __future__ import annotations

import csv
import os
from typing import List, Dict

import matplotlib.pyplot as plt


RUNS = "runs"
TRAIN_LOG = os.path.join(RUNS, "train_log.csv")
EVAL_LOG = os.path.join(RUNS, "eval_log.csv")  # optional; create if you have one


def _read_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def plot_train(log_rows: List[Dict[str, str]]):
    if not log_rows:
        return

    # Expect headers from train.py writer: t, ep_len, ep_ret, violation, G_dist, G_qdot, F_goal, int_rate, int_avg
    t = list(range(len(log_rows)))
    ep_ret = [_safe_float(r.get("ep_ret", "0")) for r in log_rows]
    int_rate = [_safe_float(r.get("int_rate", "0")) for r in log_rows]
    G_dist = [_safe_float(r.get("G_dist", "0")) for r in log_rows]
    G_qdot = [_safe_float(r.get("G_qdot", "0")) for r in log_rows]

    os.makedirs(RUNS, exist_ok=True)

    # Return
    plt.figure()
    plt.plot(t, ep_ret)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training Return")
    plt.savefig(os.path.join(RUNS, "fig_train_return.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Intervention rate
    plt.figure()
    plt.plot(t, int_rate)
    plt.xlabel("Episode")
    plt.ylabel("Intervention Rate")
    plt.title("Training Intervention Rate")
    plt.savefig(os.path.join(RUNS, "fig_train_intervention.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Robustness (G_dist / G_qdot)
    plt.figure()
    plt.plot(t, G_dist, label="G_dist")
    plt.plot(t, G_qdot, label="G_qdot")
    plt.xlabel("Episode")
    plt.ylabel("Robustness")
    plt.title("Training Robustness")
    plt.legend()
    plt.savefig(os.path.join(RUNS, "fig_train_robustness.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_eval(log_rows: List[Dict[str, str]]):
    if not log_rows:
        return

    # A generic expectation for eval logs if present:
    # columns: ep, ret, int_rate, violation, G_dist, G_qdot
    ep = list(range(len(log_rows)))
    ret = [_safe_float(r.get("ret", r.get("ep_ret", "0"))) for r in log_rows]
    int_rate = [_safe_float(r.get("int_rate", "0")) for r in log_rows]

    os.makedirs(RUNS, exist_ok=True)

    plt.figure()
    plt.plot(ep, ret)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Evaluation Return")
    plt.savefig(os.path.join(RUNS, "fig_eval_return.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(ep, int_rate)
    plt.xlabel("Episode")
    plt.ylabel("Intervention Rate")
    plt.title("Evaluation Intervention Rate")
    plt.savefig(os.path.join(RUNS, "fig_eval_intervention.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    if not os.path.exists(TRAIN_LOG):
        print("train_log.csv not found. Please run train.py first.")
        return

    train_rows = _read_csv(TRAIN_LOG)
    plot_train(train_rows)

    if os.path.exists(EVAL_LOG):
        eval_rows = _read_csv(EVAL_LOG)
        plot_eval(eval_rows)

    print("Figures were saved to the runs/ folder.")


if __name__ == "__main__":
    main()
