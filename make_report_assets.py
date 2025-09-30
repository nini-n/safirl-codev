import os
import shutil

SRC = "runs"
DST = os.path.join(SRC, "report")
os.makedirs(DST, exist_ok=True)

FILES = [
    "fig_train_return.png",
    "fig_train_intervention.png",
    "fig_train_robustness.png",
    "fig_eval_intervention.png",
    "fig_eval_return.png",
    "bench_return.png",
    "bench_violations.png",
    "bench_int_rate.png",
    "bench_robustness.png",
]

for fn in FILES:
    src = os.path.join(SRC, fn)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(DST, fn))

print("Rapor görselleri runs/report klasöründe toplandı.")
