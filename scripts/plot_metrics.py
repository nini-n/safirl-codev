# scripts/plot_metrics.py
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

RUNS = "runs"
os.makedirs(RUNS, exist_ok=True)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_key(row_keys, *candidates):
    """row_keys: mevcut kolon adları set(), candidates: öncelik sırasıyla isimler"""
    for c in candidates:
        if c in row_keys:
            return c
    return None


def to_float(arr, key_candidates, default_seq=False):
    if not arr:
        return np.array([], dtype=float)
    row_keys = set(arr[0].keys())
    key = pick_key(row_keys, *key_candidates)
    if key is None:
        if default_seq:
            # 0..N-1 indeks dizisi döndür (ör. episode yoksa)
            return np.arange(len(arr), dtype=float)
        return np.array([], dtype=float)
    out = []
    for x in arr:
        try:
            out.append(float(x[key]))
        except Exception:
            # Bozuk satırları atla
            continue
    return np.array(out, dtype=float)


def main():
    tr_path = os.path.join(RUNS, "train_metrics.csv")
    ev_path = os.path.join(RUNS, "eval_metrics.csv")

    tr = load_csv(tr_path)
    ev = load_csv(ev_path)

    if not tr and not ev:
        print(
            "Uyarı: Görselleştirecek metrik bulunamadı (train_metrics.csv / eval_metrics.csv yok). Önce eğitim veya değerlendirme çalıştır."
        )
        return

    # ---- Eğitim grafikleri ----
    if tr:
        # X ekseni: episode / ep / iter / satır indeksi
        ep = to_float(tr, ["episode", "ep", "iter", "epoch"], default_seq=True)
        ret = to_float(tr, ["ep_ret", "return", "ret"])
        viol = to_float(tr, ["violation_rate", "viol_rate", "viol"])
        int_rate = to_float(tr, ["intervention_rate", "int_rate"])

        # 1) Return
        if ep.size and ret.size:
            plt.figure()
            plt.plot(ep, ret)
            plt.xlabel("Episode")
            plt.ylabel("Return")
            plt.title("Training: Episode Return")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RUNS, "train_return.png"), dpi=150)
            plt.close()

        # 2) Violation rate
        if ep.size and viol.size:
            plt.figure()
            plt.plot(ep, viol)
            plt.xlabel("Episode")
            plt.ylabel("Violation Rate")
            plt.title("Training: Safety Violations")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RUNS, "train_violations.png"), dpi=150)
            plt.close()

        # 3) Intervention rate
        if ep.size and int_rate.size:
            plt.figure()
            plt.plot(ep, int_rate)
            plt.xlabel("Episode")
            plt.ylabel("Intervention Rate")
            plt.title("Training: Shield Interventions")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RUNS, "train_interventions.png"), dpi=150)
            plt.close()

    # ---- Değerlendirme grafikleri ----
    if ev:
        idx = to_float(ev, ["episode", "ep", "iter"], default_seq=True)
        mean_ret = to_float(ev, ["mean_return", "mean_ret", "ret"])
        viol = to_float(ev, ["violations", "viol_rate", "violation_rate"])
        int_rate = to_float(ev, ["intervention_rate", "int_rate", "mean_int_rate"])
        g_dist = to_float(ev, ["G_dist"])
        g_qdot = to_float(ev, ["G_qdot"])

        if idx.size and mean_ret.size:
            plt.figure()
            plt.plot(idx, mean_ret)
            plt.xlabel("Eval Index")
            plt.ylabel("Mean Return")
            plt.title("Evaluation: Mean Return")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RUNS, "eval_mean_return.png"), dpi=150)
            plt.close()

        if idx.size and int_rate.size:
            plt.figure()
            plt.plot(idx, int_rate)
            plt.xlabel("Eval Index")
            plt.ylabel("Intervention Rate")
            plt.title("Evaluation: Shield Interventions")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RUNS, "eval_interventions.png"), dpi=150)
            plt.close()

        if idx.size and g_dist.size:
            plt.figure()
            plt.plot(idx, g_dist)
            plt.xlabel("Eval Index")
            plt.ylabel("G_dist (↑ iyi)")
            plt.title("Evaluation: STL G_dist")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RUNS, "eval_G_dist.png"), dpi=150)
            plt.close()

        if idx.size and g_qdot.size:
            plt.figure()
            plt.plot(idx, g_qdot)
            plt.xlabel("Eval Index")
            plt.ylabel("G_qdot (↑ iyi)")
            plt.title("Evaluation: STL G_qdot")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RUNS, "eval_G_qdot.png"), dpi=150)
            plt.close()

    print("Grafikler runs/ klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
