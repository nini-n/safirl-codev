# scripts/reproduce_all.py
# Tek komutla: eğitim -> değerlendirme -> benchmark -> grafikler -> rapor varlıkları
# Windows/Linux/Mac uyumlu (shell=True)

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")


def run(cmd: str):
    print("\n>>", cmd)
    # shell=True: Windows'ta .bat/.cmd/py çağrılarını kolaylaştırır
    ret = subprocess.call(cmd, shell=True, cwd=ROOT)
    if ret != 0:
        print(f"\n!! Komut hata kodu ile bitti: {ret}\n{cmd}")
        sys.exit(ret)


def main():
    ap = argparse.ArgumentParser(description="SAFIRL reproducibility pipeline")
    ap.add_argument("--cfg", default="experiments/base.yaml", help="YAML konfig yolu")
    ap.add_argument(
        "--steps", type=int, default=20000, help="Eğitim adım sayısı (toplam)"
    )
    ap.add_argument("--episodes", type=int, default=8, help="Benchmark bölüm sayısı")
    ap.add_argument(
        "--eval_episodes", type=int, default=5, help="Değerlendirme bölüm sayısı"
    )
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Eğitimi atla; mevcut politikayı kullan",
    )
    args = ap.parse_args()

    # 1) Eğitim (isteğe bağlı)
    if not args.skip_train:
        run(f"python scripts/train.py --cfg {args.cfg} --steps {args.steps}")

    # 2) Değerlendirme (cfg’deki shield veya override yok)
    run(
        f"python scripts/evaluate.py --cfg {args.cfg} --policy runs/latest.pt --episodes {args.eval_episodes}"
    )

    # 3) Benchmark (no-shield / CBF / MPC)
    run(
        f"python scripts/benchmark.py --cfg {args.cfg} --policy runs/latest.pt --episodes {args.episodes}"
    )

    # 4) Grafikler
    run("python scripts/plot_metrics.py")
    run("python scripts/plot_benchmark.py")

    # 5) Rapor varlıkları tek klasörde
    run("python scripts/make_report_assets.py")

    print("\n✔ Repro tamamlandı. Çıktılar: runs/ ve runs/report/ içinde.")


if __name__ == "__main__":
    main()
