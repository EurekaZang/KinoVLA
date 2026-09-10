# KINO

Code for *Feel It, See It, Recover: Cross-Modal Failure Attribution for Safe
Quadrupedal Navigation Recovery*. KINO combines robot-front visual observations
and proprioception with a Gated Multimodal Unit (GMU) to infer recovery-relevant
physical causes. KINO-Fail is the benchmark and dataset of physical and sensing interventions
in Isaac Sim with a Unitree Go2.

## Names and paths

The method is **KINO**; the benchmark and dataset are **KINO-Fail**. Python
imports use the lowercase package `kino`. `kino.KINO` exposes the registered
GMU class, with the same model parameters and checkpoint format.

| Purpose | Canonical path |
| --- | --- |
| Local project entry | `/home/eureka/KINO/` |
| Method API | `kino/` |
| Benchmark specification | `benchmarks/KINO-Fail/evaluation.json` |
| Benchmark evaluation | `scripts/evaluate_kino_fail.py` |
| KINO checkpoints | `checkpoints/KINO/` |
| Baseline checkpoints | `checkpoints/baselines/<method>/` |
| Registered results | `results/KINO-Fail/<method>/` (KINO uses `KINO/`) |
| Dataset features | `/data/eureka/KINO-Fail/features/` |
| Benchmark and raw releases | `/data/eureka/KINO-Fail/releases/` |

Create the local directory aliases after placing the registered artifacts on
disk. The command checks every destination before creating links and leaves
the source directories in place:

```bash
python scripts/setup_kino_paths.py          # preview
python scripts/setup_kino_paths.py --apply  # create links
```

`--project-alias` and `--data-root` customize the two absolute paths. Directory
aliases share the existing files without copying data. `kino_vla`,
`scripts/evaluate_publication.py`, and `configs/eval/publication_evaluation.json`
remain compatibility entry points for existing imports and commands.

## Current paper evaluation

The canonical entry point is `scripts/evaluate_kino_fail.py`. Its registry,
[`benchmarks/KINO-Fail/evaluation.json`](benchmarks/KINO-Fail/evaluation.json),
specifies the checkpoint directories, feature inputs, training configurations,
aggregation rules, and reference scores for KINO and nine comparison methods.
The wrapper calls the existing evaluator with those paths explicitly.

The methods are vision-only, proprioception-only, early fusion, late fusion,
concatenation MLP, TFN, LMF, EmbraceNet, ModDrop, and GMU (KINO). Each uses its
registered three-seed, 11-class ensemble for Scale and Conflict evaluation.
The visual descriptor contains CLIP and DINOv2 features (3,072 dimensions); the
body descriptor has 251 dimensions. Recognition averages predictions over the
three appearance renders within each physical unit. Action evaluation uses
the primary robot-front render.

Install the package and test dependencies in the evaluation environment:

```bash
pip install -e ".[dev]"
```

The saved evaluators also require PyTorch and scikit-learn. The validated GPU
environment uses PyTorch 2.7.0 with CUDA 12.8. Feature extraction additionally
requires the CLIP and DINOv2 backbones used by the extraction scripts.
Checkpoints, cached features, and reference predictions are stored separately
from this source repository. Place them at the registered paths; use
`--feature-root` if the feature directory has moved.

```bash
# Validate the registered source dependencies, features, and checkpoint manifests.
python scripts/evaluate_kino_fail.py --check-only

# Select KINO (the registered GMU) by the public method name.
python scripts/evaluate_kino_fail.py --method kino --check-only

# Evaluate all ten methods without training or changing the original checkpoints.
python scripts/evaluate_kino_fail.py --method all --output results/KINO-Fail/new_run

# Compare every prediction and aggregate score with the registered references.
# Each invocation requires a new output directory.
python scripts/evaluate_kino_fail.py --method all --verify --include-action \
    --feature-root /data/eureka/KINO-Fail/features \
    --output results/KINO-Fail/regression
```

The registry records the underlying training, inference, analysis, and feature
extraction sources. These implementations retain their original filenames and
imports to preserve the registered evaluation paths.

## Tests and simulation

```bash
pytest tests/test_publication_evaluation.py tests/test_kino_naming.py -q
pytest -m "not sim and not slow" -q
```

Physical data collection uses Isaac Sim 5.1 and Isaac Lab 2.3.2.
Use the configured simulator environment for GPU tests and collection:

```bash
python scripts/check_env.py
python scripts/stand_go2.py --headless
pytest -m sim
```

## Repository layout

`kino/` provides the method API, and `benchmarks/KINO-Fail/` holds the benchmark
specification. The compatible implementation modules remain in `kino_vla/`.
`scripts/` contains collection, training, inference, and analysis entry points;
`configs/` contains simulator settings. `tests/` covers both unit behavior and
simulation. Local artifact directories (`outputs/`, `checkpoints/`, `results/`,
and `datasets/`) are excluded from Git.
