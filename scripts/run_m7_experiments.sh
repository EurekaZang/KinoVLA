#!/bin/bash
# Reproducible M7 experiment pipeline (spec §11/§12). Run on the GPU box from the repo root:
#   bash scripts/run_m7_experiments.sh
# Each step is a checkpoint; rerun from any step. KINOVLA_MODEL_ID points the loader at the
# locally-staged Qwen3-VL-4B (the canonical id stays in configs/vla/*.yaml for portability).
set -e
export KINOVLA_MODEL_ID=${KINOVLA_MODEL_ID:-/home/eureka/models/Qwen3-VL-4B-Instruct}
export HF_HUB_OFFLINE=1
export OMNI_KIT_ACCEPT_EULA=YES
PY="env -u PYTHONPATH KINOVLA_MODEL_ID=$KINOVLA_MODEL_ID HF_HUB_OFFLINE=1 OMNI_KIT_ACCEPT_EULA=YES \
    $HOME/miniconda3/envs/kinovla/bin/python"
DATA=outputs/hindsight_isaac

echo "=== [1] Kino-SFT (latent route, B5) ==="
$PY scripts/train_vla_sft.py --route latent --out outputs/vla/sft_latent

echo "=== [2] Kino-SFT (text route, B4 ablation) ==="
$PY scripts/train_vla_sft.py --route text --out outputs/vla/sft_text

echo "=== [3] Suite-Sem attribution eval (exit 1+2): latent + text ==="
$PY scripts/eval_vla_suite_sem.py --config vla/sft.yaml --adapter outputs/vla/sft_latent/adapter_best \
    --out outputs/vla/suite_sem_latent.json
$PY scripts/eval_vla_suite_sem.py --config vla/sft.yaml --adapter outputs/vla/sft_text/adapter_best \
    --route text --out outputs/vla/suite_sem_text.json

echo "=== [4] Surrogate closed-loop: SFT success + DPO pairs (controllable, reproducible) ==="
$PY scripts/eval_vla_closed_loop.py --policy model --adapter outputs/vla/sft_latent/adapter_best \
    --backend surrogate --out outputs/vla/closed_loop_sft.json
$PY scripts/build_dpo_pairs.py --policy model --adapter outputs/vla/sft_latent/adapter_best \
    --backend surrogate --out outputs/vla/dpo_pairs

echo "=== [5] Embodied DPO training ==="
$PY scripts/train_vla_dpo.py --pairs outputs/vla/dpo_pairs \
    --sft-adapter outputs/vla/sft_latent/adapter_best --out outputs/vla/dpo

echo "=== [6] Surrogate closed-loop: DPO success (exit 3) ==="
$PY scripts/eval_vla_closed_loop.py --policy model --adapter outputs/vla/dpo/adapter_dpo \
    --backend surrogate --out outputs/vla/closed_loop_dpo.json

echo "=== [7] Consolidated results table ==="
$PY scripts/m7_results.py --out outputs/vla/M7_RESULTS.md
echo "=== M7 experiment pipeline done. Isaac closed-loop (exit-3 on the real Go2) is a separate"
echo "    GPU+Isaac step: scripts/isaac_vla_rollout.py --headless --adapter ... --mode {pairs,eval}"
