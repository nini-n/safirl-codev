# SAFIRL — Shielded RL for Franka (MuJoCo) with CBF/MPC

![CI](https://github.com/nini-n/safirl-codev-tr/actions/workflows/ci.yml/badge.svg)
![Tests](https://img.shields.io/badge/pytest-passing-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

**SAFIRL** is a compact research framework for **safe reinforcement learning** on a planar Franka-like 3‑DoF arm. It trains a PPO policy and evaluates it under **CBF** and **MPC** safety shields. The repo includes training, evaluation, benchmarking, plotting, and a small report asset maker. All experiments run in **simulation** (MuJoCo or kinematic).

> ✅ **Status:** v1.0.0 — training/evaluation/benchmark scripts, plots and report assets ready  
> ✅ **Tests:** 4/4 tests passing with `pytest`  
> ✅ **Docker:** headless image for reproducible runs

---

> English | [Türkçe](README_TR.md)

---

## Table of Contents
- [Setup](#setup)
- [Quickstart](#quickstart)
- [Training](#training)
- [Evaluation](#evaluation)
- [Benchmark](#benchmark)
- [Results](#results)
- [Plots & Report](#plots--report)
- [Tests](#tests)
- [One‑shot Reproduction](#one-shot-reproduction)
- [Docker](#docker)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)
- [Troubleshooting](#troubleshooting)

---

## Setup

> Examples below use **Windows CMD**. Python **3.9** is recommended.

1) Clone and enter the repo:
```bat
git clone https://github.com/nini-n/safirl-codev-tr.git
cd safirl-codev-tr
```

2) (Recommended) Virtual environment:
```bat
python -m venv .venv
.venv\Scripts\activate
```

3) Install deps:
```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest ruff
```
> Note: `torchaudio/torchvision` are **not required**. We test with **torch==2.2.2**.

---

## Quickstart

Skip training and reproduce figures with the provided policy (`runs/latest.pt`):
```bat
python scripts\reproduce_all.py --skip-train
```
Outputs and figures go to `runs/`, report assets to `runs/report/`.

---

## Training

Default settings live in `experiments/base.yaml`. Example (20k steps):
```bat
python scripts\train.py --cfg experiments\base.yaml --steps 20000
```
The policy is saved as `runs/latest.pt`. Metrics are written under `runs/`.

---

## Evaluation

Short evaluation with the trained (or provided) policy:
```bat
python scripts\evaluate.py --cfg experiments\base.yaml --policy runs\latest.pt --episodes 5
```
> Prints per‑episode return, violation flag, and STL‑robustness summaries.

---

## Benchmark

Compare **no_shield**, **CBF** and **MPC**:
```bat
python scripts\benchmark.py --cfg experiments\base.yaml --policy runs\latest.pt --episodes 8
```
CSV summaries:
- Per‑episode: `runs/benchmark.csv`
- Aggregate: `runs/benchmark_summary.csv`

Generate plots:
```bat
python scripts\plot_benchmark.py
```

---

## Results

**Policy:** `runs/latest.pt`  
**Episodes:** 8

| Variant    | Mean Return | Violations | Intervention Rate | G_dist | G_qdot |
|------------|-------------|------------|-------------------|--------|--------|
| no_shield  | -91.9       | 0/8        | 0.000             | 0.301  | 0.80   |
| cbf        | -90.9       | 0/8        | 1.000             | 0.285  | 0.80   |
| mpc        | -91.5       | 0/8        | 1.000             | 0.376  | 0.80   |

<p float="left">
  <img src="runs/bench_return.png" width="360">
  <img src="runs/bench_int_rate.png" width="360">
</p>
<p>
  <img src="runs/bench_violations.png" width="360">
</p>

> Summary: Both CBF/MPC shields prevent violations; under this config CBF achieves similar task performance with lower intervention cost.

---

## Plots & Report

Training/eval curves:
```bat
python scripts\plot_metrics.py
```
Pack report assets:
```bat
python scripts\make_report_assets.py
```
> Figures are stored in `runs/` and `runs/report/`.

---

## Tests

Quick check:
```bat
pytest -q
```
> Currently 4 tests **pass**.

---

## One‑shot Reproduction

With training:
```bat
python scripts\reproduce_all.py --steps 20000
```
Using the provided policy:
```bat
python scripts\reproduce_all.py --skip-train
```

---

## Docker

> Requires Docker Desktop on Windows.

Build:
```bat
docker build -t safirl .
```

Run and bind local `runs/` to collect artifacts:
```bat
docker run --rm -it -v %CD%\runs:/app/runs safirl
```
> Inside the container, helper scripts run `evaluate → benchmark → plot` and store outputs in the mounted `runs/` folder.

---

## Project Structure
```
.
├─ assets/                 # MuJoCo XML and assets
│  └─ franka/franka.xml
├─ envs/
│  ├─ franka_kinematic_env.py
│  └─ franka_mujoco_env.py
├─ rl/
│  └─ ppo.py               # Minimal PPO (mu, evaluate, update)
├─ shield/
│  ├─ cbf_qp.py
│  └─ mpc_shield.py
├─ specs/
│  └─ specs.py             # STL robustness & violation helpers
├─ verify/
│  └─ robustness.py        # EpisodeTracer (add/summary/clear)
├─ scripts/
│  ├─ train.py / evaluate.py / benchmark.py
│  ├─ plot_metrics.py / plot_benchmark.py / make_report_assets.py
│  ├─ reproduce_all.py / apply_best_safety.py / codesign.py
│  └─ print_summary.py
├─ tests/                  # 4 unit tests
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

---

## Citation

If you use this repository in academic work, please cite:

```
@software{SafirlCodevTR_2025,
  author  = {Nihan},
  title   = {SAFIRL — Shielded RL for Franka (MuJoCo) with CBF/MPC},
  year    = {2025},
  url     = {https://github.com/nini-n/safirl-codev-tr}
}
```

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## Troubleshooting

**Docker build: `libgl1-mesa-glx` not found**  
On newer Debian bases package names may differ. The Dockerfile installs the required minimal GL/X set. If you hit errors, try: `libgl1 libxrender1 libxext6 libxi6 libxxf86vm1 libxrandr2 libxcb1`.

**`PPOAgent` action sampling**  
We use a minimal PPO interface: `mu(obs)` yields the deterministic mean action; during training we sample via `torch.distributions.Normal(mu, std)`. Scripts are aligned with this.

**MuJoCo XML not found**  
Verify `assets/franka/franka.xml`. On Windows/OneDrive paths the code includes a simple fallback, but please ensure the file exists.

**Torchaudio/Torchvision warnings**  
They are not required. The repo is tested with `torch==2.2.2`.

