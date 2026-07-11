# A8 External Validity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement A8a Guardian/FailCoT public transfer and A8b REFLECT multi-sensory conflict transfer, run real experiments, and produce claim-supporting metrics in `A实验/A8.md`.

**Architecture:** Dual-track offline pipeline: (1) official image/text VQA track on Guardian HF datasets with same-backbone Qwen3-VL-4B LoRA rows; (2) multi-sensory conflict track on REFLECT `real_data.zip` with real robot-state reattached. Shared metrics/leakage/strata helpers; no surrogate proprio.

**Tech Stack:** Python 3.11, PyTorch 2.7+cu128, Qwen3-VL-4B-Instruct + LoRA, Hugging Face datasets, existing `kino_vla.vla.sft` trainer patterns, Wilson/McNemar helpers.

## Global Constraints

- Real stack only: real Guardian HF data + real REFLECT multi-sensory; no synthetic state headlines.
- If REFLECT state unusable → A8b blocked with evidence; never substitute.
- Do not edit `kino-vla-v2.md`.
- Same backbone for method rows: Qwen3-VL-4B; Guardian-8B reference only.
- Leakage firewall: model sees only images + instruction + raw state; labels/captions/rewards eval-only.
- Every table: config hash + commit + seeds + source hashes; Wilson CIs; McNemar.
- Full A8a coverage: RLBench+BDV2 train; RLBench/BDV2/UR5/RoboFail test.
- A8c/A8d out of scope.
- Env recipe: `env -u PYTHONPATH HF_HUB_OFFLINE=0 OMNI_KIT_ACCEPT_EULA=YES ~/miniconda3/envs/kinovla/bin/python …`
- Model: `KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct`

---

### Task 1: Core metrics, leakage, strata helpers + tests

**Files:**
- Create: `kino_vla/eval/a8_metrics.py`
- Create: `kino_vla/eval/a8_leakage.py`
- Create: `kino_vla/eval/a8_strata.py`
- Create: `tests/test_a8_metrics.py`
- Create: `tests/test_a8_leakage.py`
- Create: `tests/test_a8_strata.py`
- Create: `configs/eval/a8.yaml`

**Interfaces:**
- Produces:
  - `wilson(k, n) -> (lo, hi)`
  - `macro_f1(y_true, y_pred) -> float`
  - `accuracy_ci(rows, field="correct") -> dict`
  - `cba(acc_e2, acc_e3) -> float`
  - `delta_fusion(cba_vp, cba_v, cba_p) -> float`
  - `delta_conflict(cba_conflict, cba_unshaped) -> float`
  - `mcnemar(a_ok, b_ok) -> dict` (reuse/import from a7 if possible)
  - `filter_allowed_fields(sample, mode) -> dict`
  - `ALLOWED_INPUT_FIELDS`, `EVAL_ONLY_FIELDS`
  - `label_stratum(episode_meta, state_summary) -> "E1"|"E2"|"E3"|"E4"|"unassigned"`

- [ ] **Step 1: Write failing tests for metrics/leakage/strata**

```python
# tests/test_a8_metrics.py
from kino_vla.eval.a8_metrics import cba, delta_fusion, delta_conflict, macro_f1, wilson

def test_cba_and_gains():
    assert cba(0.8, 0.6) == 0.7
    assert delta_fusion(0.8, 0.5, 0.6) == 0.2
    assert delta_conflict(0.85, 0.7) == 0.15

def test_macro_f1_balanced():
    y = ["a", "b", "a", "b"]
    p = ["a", "b", "a", "a"]
    assert 0.0 < macro_f1(y, p) < 1.0

def test_wilson_bounds():
    lo, hi = wilson(8, 10)
    assert 0.0 <= lo <= 0.8 <= hi <= 1.0
```

```python
# tests/test_a8_leakage.py
from kino_vla.eval.a8_leakage import filter_allowed_fields, EVAL_ONLY_FIELDS

def test_strips_failure_labels_from_model_inputs():
    sample = {
        "images": ["a.png"],
        "task_instruction": "pick cup",
        "robot_state": [0.1, 0.2],
        "failure_mode": "slip",
        "failure_reason": "object dropped",
        "reward": 0,
        "start_caption": "leak",
    }
    out = filter_allowed_fields(sample, mode="model_input")
    assert "images" in out and "task_instruction" in out and "robot_state" in out
    for k in EVAL_ONLY_FIELDS:
        assert k not in out
```

```python
# tests/test_a8_strata.py
from kino_vla.eval.a8_strata import label_stratum

def test_nominal_success_is_e4():
    assert label_stratum({"reward": 1, "failure_mode": "ground_truth"}, {}) == "E4"

def test_wrong_object_is_e2_when_state_nominal():
    meta = {"reward": 0, "failure_mode": "wrong_object"}
    state = {"gripper_delta": 0.0, "joint_motion_norm": 0.1}
    assert label_stratum(meta, state) == "E2"

def test_no_close_is_e3():
    meta = {"reward": 0, "failure_mode": "no_close"}
    state = {"gripper_cmd_closed": 1.0, "gripper_width": 0.08}
    assert label_stratum(meta, state) == "E3"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python -m pytest tests/test_a8_metrics.py tests/test_a8_leakage.py tests/test_a8_strata.py -q`
Expected: FAIL (modules missing)

- [ ] **Step 3: Implement helpers + minimal `configs/eval/a8.yaml`**

Implement the three modules and a config skeleton with `experiment`, `output_dir`, seeds, HF dataset IDs, REFLECT URLs, LoRA budget, arm list, and leakage lists.

- [ ] **Step 4: Re-run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kino_vla/eval/a8_*.py tests/test_a8_*.py configs/eval/a8.yaml
git commit -m "feat(a8): add metrics, leakage firewall, and strata helpers"
```

---

### Task 2: Download + REFLECT state admission gate

**Files:**
- Create: `scripts/a8_download.py`
- Create: `kino_vla/eval/a8_state_audit.py`
- Create: `tests/test_a8_state_audit.py`
- Create: `outputs/eval/a8/` (runtime)

**Interfaces:**
- Produces:
  - `download_guardian(cfg) -> dict[str, Path]`
  - `download_reflect(cfg, include_sim: bool) -> dict[str, Path]`
  - `audit_reflect_state(root: Path) -> dict` with keys `pass`, `n_episodes`, `state_fields`, `reason`

- [ ] **Step 1: Write failing audit test with a tiny fake tree**

```python
from pathlib import Path
from kino_vla.eval.a8_state_audit import audit_reflect_state

def test_audit_fails_without_state(tmp_path: Path):
    ep = tmp_path / "ep0"
    ep.mkdir()
    (ep / "rgb.png").write_bytes(b"x")
    out = audit_reflect_state(tmp_path)
    assert out["pass"] is False
    assert "state" in out["reason"].lower() or out["n_state_episodes"] == 0
```

- [ ] **Step 2: Implement download script and audit**

`scripts/a8_download.py` must:
1. Download HF datasets into `outputs/eval/a8/raw/guardian/…`
2. Download REFLECT `real_data.zip` (+ optional sim) into `outputs/eval/a8/raw/reflect/`
3. Extract, write source hashes into `outputs/eval/a8/download_manifest.json`
4. Run `audit_reflect_state` → `outputs/eval/a8/a8b_state_audit.json`

- [ ] **Step 3: Run real downloads (long)**

```bash
env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python scripts/a8_download.py \
  --config configs/eval/a8.yaml --reflect-real --guardian-all
```

Expected: manifest exists; audit JSON exists; if audit fail, stop A8b training path.

- [ ] **Step 4: Commit code only (not multi-GB raw data)**

```bash
git add scripts/a8_download.py kino_vla/eval/a8_state_audit.py tests/test_a8_state_audit.py
git commit -m "feat(a8): add Guardian/REFLECT download and state audit gate"
```

---

### Task 3: Dataset builders (A8a VQA cards + A8b multi-sensory cards)

**Files:**
- Create: `scripts/a8_build_datasets.py`
- Create: `kino_vla/eval/a8_datasets.py`
- Create: `tests/test_a8_datasets.py`

**Interfaces:**
- Produces:
  - `build_a8a_split(raw_dir, split, prompt_policy) -> Path`
  - `build_a8b_cards(reflect_root, strata_fn) -> Path`
  - dataset cards under `outputs/eval/a8/datasets/{a8a,a8b}/…`

- [ ] **Step 1: Failing tests for leakage-safe card construction**

Cards for model training must not embed `failure_reason` / captions into the prompt text.

- [ ] **Step 2: Implement builders**

A8a: convert InternVL JSONL + images into Kino-SFT-compatible samples (vision-only user content + answer label).  
A8b: pair RGB frames with state windows; attach E1–E4 labels in sidecars only; model views go through `filter_allowed_fields`.

- [ ] **Step 3: Build real datasets**

```bash
env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python scripts/a8_build_datasets.py --config configs/eval/a8.yaml
```

- [ ] **Step 4: Commit**

```bash
git add scripts/a8_build_datasets.py kino_vla/eval/a8_datasets.py tests/test_a8_datasets.py
git commit -m "feat(a8): build leakage-safe A8a/A8b dataset cards"
```

---

### Task 4: Training arms (same-backbone LoRA)

**Files:**
- Create: `scripts/a8_train.py`
- Modify if needed: `kino_vla/vla/sft.py` only for generic hooks (prefer no change)
- Create: `tests/test_a8_train_wiring.py` (argparse/arm routing unit test, no GPU)

**Interfaces:**
- Produces adapters under `outputs/eval/a8/adapters/{failcot_sft,kino_general,a8b_<arm>}/seed{k}/`

Arms:
1. `failcot_sft` (vision-only, A8a train mix)
2. `kino_general` (general + conflict replay; eval A8a masked)
3. A8b: `v_only`, `p_only`, `concat`, `text`, `latent`, `latent_conflict`

- [ ] **Step 1: Unit-test arm routing**
- [ ] **Step 2: Implement trainer wrapper around `train_sft`**
- [ ] **Step 3: Run real GPU training for all arms/seeds**

```bash
for seed in 0 1 2; do
  env -u PYTHONPATH KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct \
    ~/miniconda3/envs/kinovla/bin/python scripts/a8_train.py --arm failcot_sft --seed $seed
done
# similarly for kino_general and a8b arms if audit passed
```

- [ ] **Step 4: Commit scripts/tests (not weight blobs if gitignored)**

```bash
git add scripts/a8_train.py tests/test_a8_train_wiring.py
git commit -m "feat(a8): same-backbone LoRA training arms"
```

---

### Task 5: Evaluation (A8a official + A8b CBA)

**Files:**
- Create: `scripts/a8_eval.py`
- Create: `tests/test_a8_eval.py`

**Interfaces:**
- Produces:
  - `outputs/eval/a8/a8a/summary.json` (Acc_exec/plan, macro-F1, CIs, per dataset)
  - `outputs/eval/a8/a8b/summary.json` (per-stratum, CBA, Δ_fusion, Δ_conflict, McNemar)
  - or `outputs/eval/a8/a8b/blocked.json` if admission gate failed

- [ ] **Step 1: Unit tests on metric aggregation from synthetic prediction rows**
- [ ] **Step 2: Implement eval for zero-shot + trained adapters**
- [ ] **Step 3: Run real evals**

```bash
env -u PYTHONPATH KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct \
  ~/miniconda3/envs/kinovla/bin/python scripts/a8_eval.py --stage a8a
env -u PYTHONPATH KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct \
  ~/miniconda3/envs/kinovla/bin/python scripts/a8_eval.py --stage a8b
```

- [ ] **Step 4: Commit**

```bash
git add scripts/a8_eval.py tests/test_a8_eval.py
git commit -m "feat(a8): official and multi-sensory evaluation stages"
```

---

### Task 6: Report + CLAUDE records

**Files:**
- Create: `scripts/a8_report.py`
- Create: `A实验/A8.md` (generated)
- Modify: `CLAUDE.md` §2/§4/§5
- Create: `tests/test_a8_report.py`

- [ ] **Step 1: Test report renders tables from machine JSON and marks blocked A8b honestly**
- [ ] **Step 2: Implement report with required A-experiment sections**
- [ ] **Step 3: Generate report and update CLAUDE.md**

```bash
env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python scripts/a8_report.py --out A实验/A8.md
```

- [ ] **Step 4: Fast gate**

```bash
env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python -m pytest -m "not slow" -q
```

- [ ] **Step 5: Commit**

```bash
git add scripts/a8_report.py tests/test_a8_report.py A实验/A8.md CLAUDE.md
git commit -m "docs(a8): publication report and session progress"
```

---

### Task 7: Claim-strength verification pass

**Files:**
- Read: `outputs/eval/a8/**/*.json`, `A实验/A8.md`

- [ ] **Step 1: Verify exit criteria from design §10**
- [ ] **Step 2: If A8b gains weak/null, tighten report honesty; if blocked, document evidence**
- [ ] **Step 3: Optional push of final code/report commit**

```bash
git push origin HEAD
```

---

## Spec coverage checklist

| Spec section | Tasks |
|---|---|
| A8a official scores | 2,3,4,5,6 |
| A8b multi-sensory CBA/Δ | 2,3,4,5,6 |
| Leakage firewall | 1,3 |
| Same-backbone ablations | 4,5 |
| Guardian reference only | 5,6 |
| Blocked A8b path | 2,5,6 |
| `A实验/A8.md` + CLAUDE | 6 |
| No A8c/A8d | all |
