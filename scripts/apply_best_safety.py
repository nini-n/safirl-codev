# scripts/apply_best_safety.py
"""
Update the 'safety' section of a YAML config using runs/best_safety.yaml.

Usage:
    python scripts/apply_best_safety.py --cfg experiments/base.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="experiments/base.yaml",
                    help="Path to the target YAML config to update.")
    ap.add_argument("--best", type=str, default="runs/best_safety.yaml",
                    help="Path to the best-safety YAML produced by codesign.")
    ap.add_argument("--backup", action="store_true",
                    help="Make a .bak backup of the config before writing.")
    args = ap.parse_args()

    if not os.path.exists(args.best):
        print("runs/best_safety.yaml not found.")
        sys.exit(1)

    if not os.path.exists(args.cfg):
        print(f"Target config not found: {args.cfg}")
        sys.exit(1)

    with open(args.best, "r", encoding="utf-8") as f:
        best = yaml.safe_load(f) or {}
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    safety_new = (best or {}).get("safety", None)
    if not isinstance(safety_new, dict):
        print("No 'safety' section found in runs/best_safety.yaml.")
        sys.exit(1)

    if args.backup:
        shutil.copyfile(args.cfg, args.cfg + ".bak")

    if "safety" not in cfg or not isinstance(cfg["safety"], dict):
        cfg["safety"] = {}

    # update only the 'safety' section
    cfg["safety"].update(safety_new)

    with open(args.cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print("✔ Updated the 'safety' section of", args.cfg, "using runs/best_safety.yaml.")


if __name__ == "__main__":
    main()
