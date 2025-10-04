# scripts/plot_benchmark.py
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = "runs"
SUMM_PATH = os.path.join(RUNS_DIR, "benchmark_summary.csv")
PER_EP_PATH = os.path.join(RUNS_DIR, "benchmark.csv")


def _read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _get(d, *keys, default=0.0):
    for k in keys:
        if k in d and d[k] != "":
            try:
                return float(d[k])
            except Exception:
                pass
    return float(default)


def _ensure_summary_rows():
    """
    1) summary varsa onu kullan.
    2) yoksa per-episode dosyasından (runs/benchmark.csv) türet.
    """
    if os.path.exists(SUMM_PATH):
        return _read_csv(SUMM_PATH)

    if not os.path.exists(PER_EP_PATH):
        raise FileNotFoundError(
            "Ne runs/benchmark_summary.csv ne de runs/benchmark.csv bulunamadı."
        )

    per = _read_csv(PER_EP_PATH)
    by_var = {}
    for r in per:
        v = r["variant"]
        by_var.setdefault(
            v,
            {
                "int_rate": [],
                "G_dist": [],
                "G_qdot": [],
                "ret": [],
                "viol": [],
                "ep": [],
            },
        )
        by_var[v]["int_rate"].append(_get(r, "int_rate", "mean_int_rate"))
        by_var[v]["G_dist"].append(_get(r, "G_dist", "mean_G_dist"))
        by_var[v]["G_qdot"].append(_get(r, "G_qdot", "mean_G_qdot"))
        by_var[v]["ret"].append(_get(r, "mean_return"))
        by_var[v]["viol"].append(_get(r, "violations"))
        by_var[v]["ep"].append(_get(r, "episodes"))

    rows = []
    for v, agg in by_var.items():
        rows.append(
            {
                "variant": v,
                "episodes": int(np.max(agg["ep"])) if agg["ep"] else 0,
                "mean_return": float(np.mean(agg["ret"])) if agg["ret"] else 0.0,
                "violations": int(np.sum(agg["viol"])) if agg["viol"] else 0,
                "int_rate": float(np.mean(agg["int_rate"])) if agg["int_rate"] else 0.0,
                "G_dist": float(np.mean(agg["G_dist"])) if agg["G_dist"] else 0.0,
                "G_qdot": float(np.mean(agg["G_qdot"])) if agg["G_qdot"] else 0.0,
            }
        )
    return rows


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    rows = _ensure_summary_rows()

    # Eski/ yeni kolon isimlerine toleranslı oku
    S = {r["variant"]: r for r in rows}
    order = ["no_shield", "cbf", "mpc"]
    labels, mean_ret, viol, int_rate, gdist, gqdot = [], [], [], [], [], []

    for v in order:
        if v not in S:
            continue
        r = S[v]
        labels.append(v)
        mean_ret.append(_get(r, "mean_return"))
        viol.append(_get(r, "violations", default=0.0))
        int_rate.append(_get(r, "int_rate", "mean_int_rate"))
        gdist.append(_get(r, "G_dist", "mean_G_dist"))
        gqdot.append(_get(r, "G_qdot", "mean_G_qdot"))

    x = np.arange(len(labels))

    # 1) Mean return
    plt.figure()
    plt.bar(x, mean_ret)
    plt.xticks(x, labels)
    plt.ylabel("Mean Return")
    plt.title("Benchmark: Mean Return")
    plt.tight_layout()
    plt.savefig(os.path.join(RUNS_DIR, "benchmark_mean_return.png"))
    plt.close()

    # 2) Violations
    plt.figure()
    plt.bar(x, viol)
    plt.xticks(x, labels)
    plt.ylabel("Violations (sum)")
    plt.title("Benchmark: Violations")
    plt.tight_layout()
    plt.savefig(os.path.join(RUNS_DIR, "benchmark_violations.png"))
    plt.close()

    # 3) Intervention rate
    plt.figure()
    plt.bar(x, int_rate)
    plt.xticks(x, labels)
    plt.ylabel("Intervention Rate")
    plt.title("Benchmark: Intervention Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(RUNS_DIR, "benchmark_int_rate.png"))
    plt.close()

    # 4) STL robustness (G_dist & G_qdot)
    plt.figure()
    width = 0.35
    plt.bar(x - width / 2, gdist, width, label="G_dist")
    plt.bar(x + width / 2, gqdot, width, label="G_qdot")
    plt.xticks(x, labels)
    plt.ylabel("Robustness")
    plt.title("Benchmark: STL Robustness (avg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RUNS_DIR, "benchmark_robustness.png"))
    plt.close()

    print("Benchmark grafikleri runs/ klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
