# KiNO

Code for *Feel It, See It, Recover: Cross-Modal Failure Attribution for Safe
Quadrupedal Navigation Recovery*. KiNO combines robot-front visual observations
and proprioception with a Gated Multimodal Unit (GMU) to infer recovery-relevant
physical causes. KINO-FAIL supplies controlled physical and sensing interventions
in Isaac Sim with a Unitree Go2.

## Current paper evaluation

The canonical entry point is `scripts/evaluate_publication.py`. Its registry,
[`configs/eval/publication_evaluation.json`](configs/eval/publication_evaluation.json),
specifies the checkpoint directories, feature inputs, training configurations,
aggregation rules, and reference scores for KiNO and nine comparison methods.
The wrapper calls the existing evaluator with those paths explicitly.

The methods are vision-only, proprioception-only, early fusion, late fusion,
concatenation MLP, TFN, LMF, EmbraceNet, ModDrop, and GMU (KiNO). Each uses its
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
python scripts/evaluate_publication.py --check-only

# Evaluate all ten methods without training or changing the original checkpoints.
python scripts/evaluate_publication.py --method all --output outputs/eval/paper_run

# Compare every prediction and aggregate score with the registered references.
# Each invocation requires a new output directory.
python scripts/evaluate_publication.py --method all --verify --include-action \
    --output outputs/eval/paper_regression
```

The underlying entry points remain available:

| Purpose | Source |
| --- | --- |
| Training and freezing | `scripts/develop_freeze_kinofail_single_gmu_dino_upgrade_v1.py` |
| Recognition and action inference | `scripts/evaluate_kinofail_single_gmu_v1.py` |
| Statistical analysis | `scripts/analyze_kinofail_single_gmu_v1.py` |
| Multimodal model definitions | `kino_vla/eval/known_multimodal_fusions.py` |
| Dataset feature preparation | `scripts/prepare_kino_v4_all191_features_v1.py` |
| DINOv2 extraction | `scripts/extract_kinofail_snapshot_dinov2_v1.py` |

Some shared loaders retain versioned filenames because the current evaluator
imports them directly. The registry lists these dependencies so that cleanup
does not remove them or redirect the evaluation to a different pipeline.

## Tests and simulation

```bash
pytest tests/test_publication_evaluation.py -q
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

`kino_vla/` contains simulation, sensing, monitoring, recovery, and evaluation
modules. `scripts/` contains collection, feature extraction, training, inference,
and analysis entry points. `configs/` contains simulator and evaluation settings;
`tests/` contains unit and simulator tests. Data and model artifacts belong in
external storage or the ignored `outputs/` directory.
