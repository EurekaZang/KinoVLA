#!/usr/bin/env bash
# Gap-2 step A — VALIDATE the root cause (CLAUDE.md §6 #36): re-train on-policy DPO on the EXISTING
# 32 pairs with loss_span=action (mask the confounded free-form Thought, score only the <Action>
# decision span). If pref_acc recovers from 0.19 toward ~1.0, the Thought-confound hypothesis is
# confirmed. Quick (~5 min).
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kinovla
cd /home/eureka/KinoVLA
export KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SFT=outputs/vla/sft_latent/adapter_best
echo "[dpo-fix A] action-masked DPO on the existing 32 on-policy pairs $(date -Iseconds)"
python scripts/train_vla_dpo.py \
  --pairs outputs/vla/dpo_onpolicy_pairs \
  --sft-adapter "$SFT" \
  --out outputs/vla/dpo_onpolicy_actionmask \
  --set train.loss_span=action train.lr=5.0e-6 \
  --epochs 3 --grad-accum 2
echo "VALIDATE_DONE"
