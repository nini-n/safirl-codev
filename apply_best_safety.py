import os
import sys

import yaml

base = "experiments/base.yaml"
best = "runs/best_safety.yaml"

if not os.path.exists(best):
    print("runs/best_safety.yaml bulunamadı.")
    sys.exit(1)

with open(base, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
with open(best, encoding="utf-8") as f:
    best_safety = yaml.safe_load(f)

cfg["safety"].update(best_safety)  # sadece safety bölümünü güncelle

with open(base, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("✔ base.yaml safety bölümü runs/best_safety.yaml ile güncellendi.")
