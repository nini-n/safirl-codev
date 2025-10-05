# SAFIRL: Shielded RL with CBF/MPC on Franka-MuJoCo

> Reproducible training/evaluation for a PPO agent with **Control-Barrier-Function (CBF)** and **MPC-style** shields. Includes STL-like robustness tracing, benchmarks, plotting utilities, and a one-shot reproduction script.

[English](README.md) · [Türkçe](README_TR.md)

---

## Quick Start

### 1) Clone
**Unix/macOS**
```bash
git clone https://github.com/nini-n/safirl-codev
cd safirl-codev
```

**Windows (PowerShell/CMD)**
```bat
git clone https://github.com/nini-n/safirl-codev
cd safirl-codev
```

### 2) Environment
This repo aims to run with a light footprint.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt || true  # if not present, install minimal set below
pip install numpy matplotlib pyyaml pytest
# If you want Mujoco env:
# pip install mujoco
```

> **Note:** The code runs without MuJoCo using the kinematic stub. MuJoCo-based tests are automatically **skipped** if the `mujoco` package is not installed.

---

## Reproducibility

Run everything in one command:

```bash
python scripts/reproduce_all.py --skip-train
```

Train and evaluate explicitly:

```bash
python scripts/train.py --cfg experiments/base.yaml --steps 20000
python scripts/evaluate.py --cfg experiments/base.yaml --policy runs/latest.pt --episodes 5
python scripts/benchmark.py --cfg experiments/base.yaml --policy runs/latest.pt --episodes 8
python scripts/plot_benchmark.py
python scripts/plot_metrics.py
```

Artifacts are written to `runs/`, and report figures are collated under `runs/report/`.

---

## Results (default config)

Abridged numbers from the included benchmark (8 episodes). These are illustrative and will vary with seeds and environment details.

| Variant     | Mean Return | Violations | Intervention Rate | G_dist | G_qdot |
|-------------|------------:|-----------:|------------------:|-------:|-------:|
| no_shield   |     −91.9   |       0/8  |             0.000 |  0.186 |  1.200 |
| cbf         |     −90.9   |       0/8  |             1.000 |  0.371 |  1.200 |
| mpc         |     −91.5   |       0/8  |             1.000 |  0.310 |  1.200 |

**Takeaway:** Both shields avoid violations. CBF achieves comparable performance with low intervention cost under this setup.

---

## Project Structure

```
.
├── envs/                # Mujoco-based and kinematic environments
├── rl/                  # PPO agent and buffer
├── shield/              # CBF and MPC shields (action projection)
├── specs/               # Robustness/spec utilities
├── verify/              # Episode tracer for robustness aggregation
├── scripts/             # Train, evaluate, benchmark, plotting, report collation
├── experiments/         # YAML configurations (hyperparameters, safety thresholds)
└── tests/               # Unit tests (MuJoCo tests are auto-skipped if unavailable)
```

---

## Troubleshooting

- **Matplotlib backend on CI/headless:** Set `MPLBACKEND=Agg` (already configured in CI).  
- **MuJoCo missing:** The Mujoco environment and tests are optional. Install `mujoco` to enable, otherwise they are skipped.  
- **Long training:** Use `--skip-train` with `reproduce_all.py` to reproduce plots using the provided or previously saved policy.  
- **Only run unit tests:**
  ```bash
  pytest -q tests
  ```

---

## Citation

If you use this repository in academic work, please cite:

```bibtex
@software{SafirlCodev_2025,
  author = {Nihan},
  title  = {SAFIRL — Shielded RL for Franka (MuJoCo) with CBF/MPC},
  year   = {2025},
  url    = {https://github.com/nini-n/safirl-codev}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
