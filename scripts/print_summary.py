# scripts/print_summary.py
"""
Quickly print a compact summary from the available CSVs in runs/.
Looks for: benchmark_summary.csv or benchmark.csv as a fallback.
"""

import csv
import os

RUNS = "runs"
SUM = os.path.join(RUNS, "benchmark_summary.csv")
FALLBACK = os.path.join(RUNS, "benchmark.csv")

if not os.path.exists(SUM) and not os.path.exists(FALLBACK):
    print("No CSV files found. Please run the benchmark first.")
    raise SystemExit(1)

path = SUM if os.path.exists(SUM) else FALLBACK

with open(path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    print(f"\nSummary from {os.path.basename(path)}")
    for r in reader:
        # tolerant to both headers
        name = r.get("variant") or r.get("name") or r.get("shield") or "variant"
        mean_ret = r.get("mean_return") or r.get("return") or r.get("ret") or "NA"
        viol = r.get("violations") or r.get("violation") or "NA"
        int_rate = r.get("int_rate") or "NA"
        gdist = r.get("G_dist") or "NA"
        gqdot = r.get("G_qdot") or "NA"
        print(f"- {name}: mean_return={mean_ret}  violations={viol}  int_rate={int_rate}  G_dist={gdist}  G_qdot={gqdot}")
