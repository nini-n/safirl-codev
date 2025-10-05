# scripts/plot_benchmark.py
"""
Generate benchmark figures from runs/benchmark_summary.csv (preferred)
or runs/benchmark.csv (fallback).

Outputs:
  runs/bench_return.png
  runs/bench_violations.png
  runs/bench_int_rate.png
  runs/bench_robustness.png
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


RUNS = "runs"
PREF = os.path.join(RUNS, "benchmark_summary.csv")
ALT = os.path.join(RUNS, "benchmark.csv")


def _read_rows(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def plot(rows):
    if not rows:
        return

    # Tolerant headers
    name_key = "variant" if "variant" in rows[0] else (
        "name" if "name" in rows[0] else ("shield" if "shield" in rows[0] else None)
    )
    if name_key is None:
        name_key = "variant"

    # aggregate by variant just in case
    agg = defaultdict(lambda: {"ret": [], "viol": [], "ir": [], "gd": [], "gq": []})
    for r in rows:
        key = r.get(name_key, "variant")
        agg[key]["ret"].append(_safe_float(r.get("mean_return", r.get("return", 0.0))))
        agg[key]["viol"].append(_safe_float(r.get("violations", r.get("violation", 0.0))))
        agg[key]["ir"].append(_safe_float(r.get("int_rate", 0.0)))
        agg[key]["gd"].append(_safe_float(r.get("G_dist", 0.0)))
        agg[key]["gq"].append(_safe_float(r.get("G_qdot", 0.0)))

    variants = list(agg.keys())
    mean_ret = [sum(v["ret"]) / max(1, len(v["ret"])) for v in agg.values()]
    viol = [sum(v["viol"]) / max(1, len(v["viol"])) for v in agg.values()]
    int_rate = [sum(v["ir"]) / max(1, len(v["ir"])) for v in agg.values()]
    rob = [sum(v["gd"]) / max(1, len(v["gd"])) for v in agg.values()]
    rob2 = [sum(v["gq"]) / max(1, len(v["gq"])) for v in agg.values()]

    os.makedirs(RUNS, exist_ok=True)

    # Return
    plt.figure()
    plt.bar(variants, mean_ret)
    plt.ylabel("Mean Return")
    plt.title("Benchmark — Mean Return")
    plt.savefig(os.path.join(RUNS, "bench_return.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Violations
    plt.figure()
    plt.bar(variants, viol)
    plt.ylabel("Violations")
    plt.title("Benchmark — Violations")
    plt.savefig(os.path.join(RUNS, "bench_violations.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Intervention rate
    plt.figure()
    plt.bar(variants, int_rate)
    plt.ylabel("Intervention Rate")
    plt.title("Benchmark — Intervention Rate")
    plt.savefig(os.path.join(RUNS, "bench_int_rate.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Robustness (G_dist / G_qdot)
    plt.figure()
    x = range(len(variants))
    plt.plot(x, rob, marker="o", label="G_dist")
    plt.plot(x, rob2, marker="o", label="G_qdot")
    plt.xticks(list(x), variants)
    plt.ylabel("Robustness")
    plt.title("Benchmark — Robustness")
    plt.legend()
    plt.savefig(os.path.join(RUNS, "bench_robustness.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    path = PREF if os.path.exists(PREF) else ALT
    if not os.path.exists(path):
        print("Neither runs/benchmark_summary.csv nor runs/benchmark.csv was found.")
        return
    rows = _read_rows(path)
    plot(rows)
    print("Benchmark figures were saved to the runs/ folder.")


if __name__ == "__main__":
    main()
