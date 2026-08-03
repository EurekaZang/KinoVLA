#!/usr/bin/env bash
# Gap-2 steps B–D (CLAUDE.md §6 #36) — the real experiment once step A confirms the fix.
#  B) collect MORE on-policy pairs across all ambiguity operators (the existing 32 were O1-skewed);
#  C) DPO-train the SFT planner on them with loss_span=action (the decision-span fix);
#  D) eval SFT-vs-DPO held-out attribution at temp 0 / 0.8 / 1.2 (the sampled regime has headroom —
#     the temp-0 ceiling is 0.974, but SFT is 0.912 @0.8 / 0.882 @1.2; a working DPO sharpens those).
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kinovla
cd /home/eureka/KinoVLA
export KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SFT=outputs/vla/sft_latent/adapter_best

echo "[dpo-fix B] collect scaled on-policy pairs $(date -Iseconds)"
python scripts/dpo_onpolicy.py --sft "$SFT" \
  --out outputs/vla/dpo_onpolicy_pairs_v2 --sample-temp 1.1 --n-samples 8 --max-items 150

echo "[dpo-fix C] DPO (action-mask, scaled pairs) $(date -Iseconds)"
python scripts/train_vla_dpo.py \
  --pairs outputs/vla/dpo_onpolicy_pairs_v2 --sft-adapter "$SFT" \
  --out outputs/vla/dpo_onpolicy_v2 \
  --set train.loss_span=action train.lr=5.0e-6 --epochs 3 --grad-accum 2

echo "[dpo-fix D] eval SFT vs DPO at temp 0 / 0.8 / 1.2 (apples-to-apples, same harness) $(date -Iseconds)"
for t in 0.0 0.8 1.2; do
  echo "--- SFT  temp $t ---"
  python scripts/eval_vla_suite_sem.py \
    --adapter "$SFT" --temperature "$t" --samples 5 \
    --out "outputs/vla/suite_sem_sftcmp_t${t}.json"
  echo "--- DPO  temp $t ---"
  python scripts/eval_vla_suite_sem.py \
    --adapter outputs/vla/dpo_onpolicy_v2/adapter_dpo --temperature "$t" --samples 5 \
    --out "outputs/vla/suite_sem_dpoopv2_t${t}.json"
done
echo "DPO_SCALE_DONE $(date -Iseconds)"
