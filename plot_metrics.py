# Grafikleri üretir: eğitim ve değerlendirme CSV'lerinden
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def to_float(arr, key):
    return np.array([float(x[key]) for x in arr], dtype=float)


def main():
    train_csv = "runs/train_log.csv"
    eval_csv = "runs/eval_log.csv"
    if not os.path.exists(train_csv):
        print("train_log.csv bulunamadı. Önce train.py çalıştırın.")
        return

    tr = read_csv(train_csv)
    ep = to_float(tr, "episode")
    ret = to_float(tr, "return")
    i_rate = to_float(tr, "intervention_rate")
    gdist = to_float(tr, "rob_G_dist")
    gqdot = to_float(tr, "rob_G_qdot")

    # 1) Return (eğitim)
    plt.figure()
    plt.plot(ep, ret)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training Return")
    plt.tight_layout()
    plt.savefig("runs/fig_train_return.png", dpi=150)

    # 2) Intervention rate (eğitim)
    plt.figure()
    plt.plot(ep, i_rate)
    plt.xlabel("Episode")
    plt.ylabel("Intervention Rate")
    plt.title("Shield Intervention Rate (Training)")
    plt.tight_layout()
    plt.savefig("runs/fig_train_intervention.png", dpi=150)

    # 3) STL robustness (G_qdot vs G_dist) scatter
    plt.figure()
    plt.scatter(gdist, gqdot, s=10)
    plt.xlabel("Robustness: G_dist")
    plt.ylabel("Robustness: G_qdot")
    plt.title("STL Robustness Scatter (Training Episodes)")
    plt.tight_layout()
    plt.savefig("runs/fig_train_robustness.png", dpi=150)

    # Eval grafikleri varsa
    if os.path.exists(eval_csv):
        ev = read_csv(eval_csv)
        eidx = to_float(ev, "episode")
        e_rate = to_float(ev, "intervention_rate")
        e_ret = to_float(ev, "return")
        plt.figure()
        plt.bar(eidx, e_rate)
        plt.xlabel("Episode")
        plt.ylabel("Intervention Rate")
        plt.title("Shield Intervention Rate (Eval)")
        plt.tight_layout()
        plt.savefig("runs/fig_eval_intervention.png", dpi=150)

        plt.figure()
        plt.bar(eidx, e_ret)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Eval Return per Episode")
        plt.tight_layout()
        plt.savefig("runs/fig_eval_return.png", dpi=150)

    print("Grafikler runs/ klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
