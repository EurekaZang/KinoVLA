#!/usr/bin/env bash
# M7 route-fidelity ablation driver (Gap-1, CLAUDE.md §6 #35). Trains the 4 arms under ONE identical
# harness (seed/epochs/split from vla/sft.yaml; only route + data.proprio_detail differ), evals each
# with the appearance-regime breakdown + θ-MAE + token cost, then combines. Run in the background.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kinovla
cd /home/eureka/KinoVLA
export KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

ARMS="vision_only text_scalar text_binned latent"
echo "[ablation] START $(date -Iseconds)"
for a in $ARMS; do
  echo "==================== TRAIN $a ===================="
  python scripts/m7_route_ablation.py --stage train --arm "$a"
done
for a in $ARMS; do
  echo "==================== EVAL $a (temp 0) ===================="
  python scripts/m7_route_ablation.py --stage eval --arm "$a" --temp 0.0
done
python scripts/m7_route_ablation.py --stage combine --temp 0.0
echo "[ablation] ABLATION_DONE $(date -Iseconds)"
