#!/usr/bin/env bash
# Gap-2 (CLAUDE.md §6 #36) — the DPO>SFT margin in a HEADROOM regime. The full-data 4-epoch SFT is at
# the attribution ceiling (0.974), so DPO has no room on it (it ties — honest). Here we under-train
# the SFT (1 epoch) — the realistic low-data regime the 413-sample dataset lives in — which has real
# attribution errors, then apply the action-masked on-policy Embodied DPO (the §11 procedure, now
# stable) and measure the margin on the UNDILUTED appearance-ambiguous regime (n=24).
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kinovla
cd /home/eureka/KinoVLA
export KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WEAK=outputs/vla/sft_weak/adapter_best

# Also re-eval the STRONG (full-data) DPO vs SFT on the undiluted ambiguous regime (honest ceiling).
echo "[headroom 0] strong SFT vs strong on-policy DPO, AMBIGUOUS regime $(date -Iseconds)"
for t in 0.8 1.2; do
  python scripts/eval_vla_suite_sem.py --adapter outputs/vla/sft_latent/adapter_best \
    --temperature "$t" --samples 5 --regime ambiguous --out "outputs/vla/sem_strongsft_amb_t${t}.json"
  python scripts/eval_vla_suite_sem.py --adapter outputs/vla/dpo_onpolicy_v2/adapter_dpo \
    --temperature "$t" --samples 5 --regime ambiguous --out "outputs/vla/sem_strongdpo_amb_t${t}.json"
done

echo "[headroom 1] train under-resourced SFT (1 epoch) $(date -Iseconds)"
python scripts/train_vla_sft.py --route latent --epochs 1 --out outputs/vla/sft_weak

echo "[headroom 2] eval weak SFT (confirm headroom) $(date -Iseconds)"
python scripts/eval_vla_suite_sem.py --adapter "$WEAK" --temperature 0.0 --regime ambiguous \
  --out outputs/vla/sem_weaksft_amb_t0.json

echo "[headroom 3] collect on-policy pairs from the WEAK SFT $(date -Iseconds)"
python scripts/dpo_onpolicy.py --sft "$WEAK" --out outputs/vla/dpo_pairs_weak \
  --sample-temp 1.0 --n-samples 6 --max-items 120

echo "[headroom 4] action-masked DPO on the weak SFT $(date -Iseconds)"
python scripts/train_vla_dpo.py --pairs outputs/vla/dpo_pairs_weak --sft-adapter "$WEAK" \
  --out outputs/vla/dpo_weak --set train.loss_span=action train.lr=5.0e-6 --epochs 3 --grad-accum 2

echo "[headroom 5] eval weak SFT vs weak DPO, AMBIGUOUS + all $(date -Iseconds)"
for t in 0.0 0.8; do
  python scripts/eval_vla_suite_sem.py --adapter "$WEAK" --temperature "$t" --samples 5 \
    --regime ambiguous --out "outputs/vla/sem_weaksft_amb_s_t${t}.json"
  python scripts/eval_vla_suite_sem.py --adapter outputs/vla/dpo_weak/adapter_dpo \
    --temperature "$t" --samples 5 --regime ambiguous --out "outputs/vla/sem_weakdpo_amb_t${t}.json"
done
echo "HEADROOM_DONE $(date -Iseconds)"
