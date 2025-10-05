# scripts/reproduce_all.py
# Run everything with a single command: training → evaluation → benchmark → figures → report assets
# Windows/Linux/macOS compatible (shell=True)

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")


def run(cmd: str):
    print("\n>>", cmd)
    # shell=True: simplifies .bat/.cmd/.py invocation on Windows
    ret = subprocess.call(cmd, shell=True, cwd=ROOT)
    if ret != 0:
        print(f"\n!! Command exited with a non-zero code: {ret}\n{cmd}")
        sys.exit(ret)


def main():
    ap = argparse.ArgumentParser(description="SAFIRL reproducibility pipeline")
    ap.add_argument("--cfg", default="experiments/base.yaml", help="Path to the YAML config")
    ap.add_argument("--steps", type=int, default=20000, help="Total number of training steps")
    ap.add_argument("--episodes", type=int, default=8, help="Number of benchmark episodes")
    ap.add_argument("--eval_episodes", type=int, default=5, help="Number of evaluation episodes")
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; use the existing policy",
    )
    args = ap.parse_args()

    # 1) Training (optional)
    if not args.skip_train:
        run(f"python scripts/train.py --cfg {args.cfg} --steps {args.steps}")

    # 2) Evaluation (uses the shield from cfg unless overridden)
    run(
        f"python scripts/evaluate.py --cfg {args.cfg} --policy runs/latest.pt --episodes {args.eval_episodes}"
    )

    # 3) Benchmark (no-shield / CBF / MPC)
    run(
        f"python scripts/benchmark.py --cfg {args.cfg} --policy runs/latest.pt --episodes {args.episodes}"
    )

    # 4) Figures
    run("python scripts/plot_metrics.py")
    run("python scripts/plot_benchmark.py")

    # 5) Collate report assets in a single folder
    run("python scripts/make_report_assets.py")

    print("\n✔ Repro completed. Outputs are under runs/ and runs/report/.")


if __name__ == "__main__":
    main()
