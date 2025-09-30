import csv
import os

per = "runs/benchmark.csv"
summ = "runs/benchmark_summary.csv"
if not (os.path.exists(per) and os.path.exists(summ)):
    print("CSV'ler yok. Önce benchmark çalıştırın.")
    raise SystemExit

print("\n== Benchmark Summary ==")
with open(summ, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        print(
            f"{row['variant']:>10s} | episodes={row['episodes']}  "
            f"viol={row['violations']}  mean_ret={float(row['mean_return']):.2f}  "
            f"int_rate={float(row['mean_int_rate']):.3f}  "
            f"G_dist={float(row['mean_G_dist']):.3f}  G_qdot={float(row['mean_G_qdot']):.3f}"
        )
print("\nPer-episode log: runs/benchmark.csv")
