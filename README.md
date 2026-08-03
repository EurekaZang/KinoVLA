# KiNO

Closed-loop embodied reflection for a Unitree Go2 quadruped in Isaac Lab: a 1 kHz
Kino-Monitor + Reflex layer keeps the robot alive while a VLA Recovery Planner
attributes physical failures (Kino-Fail v2 operator benchmark) and proposes recovery
primitives, gated by a CBF-QP Safety Shield. Design spec: `kino-vla-v2.md`.
Development protocol and milestone plan: `CLAUDE.md`.

## Quickstart (dev tier — any machine, no GPU required)

```bash
conda create -n kinovla python=3.11 -y
conda activate kinovla
pip install -e ".[dev]"
pre-commit install

python scripts/check_env.py     # report what this machine can run
pytest -m "not slow" -q         # unit tests (sim tests auto-skip without Isaac Lab)
ruff check . && ruff format --check .

python scripts/run_demo.py --backend surrogate   # walking skeleton (M1 deliverable)
```

The demo runs the full closed loop on a CPU surrogate backend: the Go2 walks
onto an O1 ice patch, the Kino-Monitor fires, the scripted FSM backsteps and
replans a detour, and the robot reaches the goal. Expected last line:
`PASS: walking skeleton demo`.

## GPU machine setup (sim tier — RTX 5090 / Blackwell)

The RTX 5090 (compute capability sm_120) is only supported by CUDA ≥ 12.8 builds —
use Isaac Sim **5.x** (Isaac Sim ≤ 4.5 ships pre-Blackwell PyTorch and will not run).

```bash
conda create -n kinovla python=3.11 -y
conda activate kinovla

# PyTorch with Blackwell kernels first, so Isaac Sim does not pin an older build
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Reproduce the benchmark-tested pair (do not mix the Isaac Lab 3.x beta with Sim 5.1)
pip install "isaacsim[all,extscache]==5.1.*" --extra-index-url https://pypi.nvidia.com
pip install isaaclab[isaacsim,all]==2.3.2 --extra-index-url https://pypi.nvidia.com

pip install -e ".[dev]"
python scripts/check_env.py             # must show the 5090 with sm_120 and cuda >= 12.8
python scripts/stand_go2.py --headless  # M0 bring-up: Go2 stands on flat terrain
python scripts/run_demo.py --backend isaac --headless  # M1 skeleton on the real Go2 asset
pytest -m sim                           # GPU-gated test suite
```

First Isaac Sim launch compiles shaders and downloads asset caches; expect several
minutes and ~10 GB of disk. If `isaaclab` pip pinning fails on your driver/OS combo,
fall back to the official Isaac Lab source install and `pip install -e .` this repo
into that same environment. For a source install, check out Isaac Lab `v2.3.2`; every
publication gate records the resolved Isaac Sim/Lab versions and input hashes.

## Repository layout

```
kino_vla/
  loop.py   closed-loop episode runner (walking skeleton; dual-rate split at M3)
  skeleton.py  demo assembly: configs -> wired pipeline
  sim/      backends (surrogate CPU / Isaac Go2), terrain, Kino-Fail operators (spec §8)
  monitor/  Kino-Monitor (rule-based v0; 1 kHz + Reflex at M2)
  shield/   Safety Shield (pass-through stub; CBF-QP at M3, spec §6)
  tokens/   Kino-Tokens extractor, privileged distillation (spec §4–5)
  map/      semantic traversability map (spec §7)
  vla/      recovery planner (scripted FSM stub; VLA + SFT/DPO at M7, spec §10–11)
  data/     Hindsight CoT pipeline + truth-consistency filter (spec §10)
  eval/     suites Cal/Sem/Comp/Bound/OOD, baselines B1–B5 (spec §12)
configs/    all thresholds & tolerances (no magic numbers in code)
scripts/    entry points (check_env, stand_go2, run_demo)
tests/      pytest; `-m sim` marks GPU-gated tests, auto-skipped without Isaac Lab
```

## Development rules (short form — full version in CLAUDE.md)

- Work proceeds milestone by milestone (M0–M8); `scripts/run_demo.py` (from M1 on)
  is the permanent walking-skeleton regression and must always pass on `main`.
- Every operator ships with θ-application, determinism, and composability tests.
- Safety code (`kino_vla/shield/`) has the strictest bar — never weaken a safety
  assertion to make a test pass.
- Conventional commits; no large binaries in git (`outputs/` is ignored).
