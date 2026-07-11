# A8 External Failure-Reasoning Benchmark Design

Date: 2026-07-11  
Project: KiNO / Paper-A  
Scope: A8a public Guardian/FailCoT transfer + A8b REFLECT multi-sensory conflict transfer  
Status: approved for implementation planning

## 1. Purpose

A8 proves **external validity** of cross-modal failure attribution. It does **not** re-prove internal C1–C3 on Kino-Fail / Go2 / Isaac.

Paper-facing conclusions to earn:

1. **A8a — Public score.** On official Guardian/FailCoT splits, same-backbone Qwen3-VL-4B methods remain competitive under official failure verification, so the visual-semantic failure module does not collapse off Kino-Fail.
2. **A8b — Method table.** On REFLECT multi-sensory RoboFail episodes with **real** robot-state/proprio reattached, conflict-balanced attribution shows:
   - fusion beats strong unimodal baselines,
   - conflict-trained fusion beats unshaped fusion,
   - gains concentrate on pre-registered conflict strata (E2/E3), not only Agree/Nominal.

Claim bridge:

| Claim | A8 role |
|---|---|
| C1 necessity | Structural external analogue only via E2/E3 strata (not Go2 byte-identity) |
| C2 learned conflict resolution | **Primary** — Δ_fusion and Δ_conflict on public multi-sensory conflict cells |
| C3 attribution ⇒ consequence | **Out of scope** — A4 already owns interventional consequence |
| C4 generalization / abstention | Secondary — official OOD splits + optional AURC/ECE diagnostic |
| C5 fair opponent | Keep strong unimodal rows; do not compare 4B method rows to Guardian-8B as method ablations |

## 2. Binding Constraints

- Real stack only: real Hugging Face Guardian/FailCoT data and real REFLECT multi-sensory archives. No synthetic proprio, no ScriptedOracle headlines, no fabricated state.
- If REFLECT robot-state/proprio is missing or unusable after inspection → **A8b blocked** with evidence in `CLAUDE.md` §5 and `A实验/A8.md`; do not substitute proxies as headline evidence.
- Do not edit `kino-vla-v2.md`.
- Every table carries config hash + commit + seeds + source hashes; Wilson CIs; McNemar on paired rows; bootstrap where used.
- Guardian-8B is **reference only**, never a same-method ablation against Qwen3-VL-4B.
- Leakage firewall is mandatory (Section 5).
- A8c (RLBench-Fail-CM regeneration + C2ST) and A8d (physical Go2 snapshots) are **out of scope** for this implementation.

## 3. Selected Approach

**Dual-track pipeline (Approach A).**

Rejected alternatives:

1. **Fork full Guardian InternVL/SLURM stack.** Strong official fidelity, but conflicts with the existing Qwen3-VL-4B + LoRA training stack and raises integration cost without improving claim support.
2. **Eval-first / train-later.** Faster tables, but breaks the same-backbone fair ablation required by `A8_design.md`.
3. **State-proxy A8b from Guardian metadata gripper flags.** Explicitly rejected by user choice: A8b depends on real REFLECT multi-sensory state.

Selected:

- Independent A8a official VQA track and A8b multi-sensory conflict track.
- Shared backbone, LoRA budget, leakage whitelist, metrics helpers, and report skeleton.
- Real-first REFLECT download: `real_data.zip` required; `sim_data.zip` only if real state is insufficient for the pre-registered strata.

## 4. Architecture

```text
configs/eval/a8.yaml
kino_vla/eval/a8_metrics.py      # Wilson, macro-F1, CBA, Δfusion/Δconflict, McNemar
kino_vla/eval/a8_leakage.py      # allowed-input whitelist / eval-only fields
kino_vla/eval/a8_strata.py       # E1–E4 deterministic labeling rules
scripts/a8_download.py           # HF Guardian + REFLECT multi-sensory
scripts/a8_build_datasets.py     # A8a VQA cards; A8b multi-sensory cards + strata
scripts/a8_train.py              # same-backbone Qwen3-VL-4B LoRA arms
scripts/a8_eval.py               # A8a official + A8b method metrics
scripts/a8_report.py             # → A实验/A8.md
outputs/eval/a8/                 # machine evidence root
A实验/A8.md                      # human report
tests/test_a8_*.py               # metrics, leakage, strata, report contracts
```

Reuse patterns from A7 (`configs/eval/a7.yaml`, `scripts/a7_*.py`, `kino_vla/eval/a7_ablation.py`) for config hashing, artifact metadata, Wilson/McNemar helpers, and report generation. Do not force A8 into A7 scripts.

## 5. Data

### 5.1 A8a — Guardian / FailCoT (full official coverage)

| Split | HF repo | Approx size | Role |
|---|---|---|---|
| RLBench-Fail train | `paulpacaud/rlbenchfail_train_dataset` | ~8.8 GB | FailCoT-SFT / general train mix |
| BDV2-Fail train | `paulpacaud/bdv2fail_train_dataset` | ~0.8 GB | FailCoT-SFT / general train mix |
| UR5-Fail train | `paulpacaud/ur5fail_train_dataset` | ~0.2 GB | optional extra real train |
| RLBench-Fail test | `paulpacaud/rlbenchfail_test_dataset` | ~45.8 GB | official test column |
| BDV2-Fail test | `paulpacaud/bdv2fail_test_dataset` | ~0.2 GB | official test column |
| UR5-Fail test | `paulpacaud/ur5fail_test_dataset` | ~0.06 GB | official test column |
| OOD bundle | `paulpacaud/Guardian-FailCoT-OOD-datasets` | RoboFail ~0.9 GB + RoboVQA + UR5 copy | RoboFail / RoboVQA / UR5 OOD |

Official protocol notes:

- Evaluation uses InternVL-style JSONL (`execution_*`, `planning_*`, vanilla/thinking variants).
- Primary A8a eval configs: execution and planning; report both Acc_exec and Acc_plan.
- Prefer vanilla or a fixed thinking policy consistently across same-backbone rows; document the chosen prompt policy in `configs/eval/a8.yaml`.

### 5.2 A8b — REFLECT multi-sensory

| Archive | URL | Size | Role |
|---|---|---|---|
| `real_data.zip` | `https://www.cs.columbia.edu/~liuzeyi/reflect_data/real_data.zip` | 28 GB | **Required** A8b source |
| `sim_data.zip` | `https://www.cs.columbia.edu/~liuzeyi/reflect_data/sim_data.zip` | 57 GB | Download only if real state is insufficient |
| `tasks_real_world.json` | same host | 21 KB | task metadata |

A8b admission gate (must pass before any A8b headline):

1. Archive extracts successfully.
2. Episodes expose RGB (and depth if present) plus **raw observable robot-state history** (joint and/or gripper and/or EE pose / contact / force — whatever REFLECT actually ships).
3. Enough episodes remain after filtering to support pre-registered E1–E4 strata and paired agent comparison.
4. If gate fails → write blocked artifact + report limitation; no state-proxy headlines.

### 5.3 Leakage firewall

Allowed at train/eval model inputs:

- images / multi-view frames,
- task or subtask instruction,
- raw observable robot-state history (A8b only),
- optional text summary derived only from allowed state (A8b text route).

Eval-only / oracle / stratification only:

- `failure_mode`, `failure_reason`,
- `reward` / planning_reward / execution_reward,
- start/end captions,
- ground-truth object identity,
- any field derived from failure labels.

Appendix of `A实验/A8.md` must include an **Allowed Inputs / Evaluation-only Metadata** table.

## 6. Models and Arms

All method rows use **Qwen3-VL-4B-Instruct** (`KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct`) with a fixed LoRA budget declared in `configs/eval/a8.yaml`.

### 6.1 A8a rows

| Row | Input | Training | Role |
|---|---|---|---|
| Guardian official | RGB multi-view | FailCoT (paper / optional their ckpt) | **Reference only** |
| Qwen3-VL zero-shot | RGB | none | floor |
| Qwen3-VL + FailCoT-SFT | RGB | RLBench-Fail + BDV2-Fail train | public transfer |
| Kino-VLA-General, Kino masked | RGB | general failure SFT + conflict replay; proprio masked at test | method without proprio on official track |

Do not market a T2/T3 conflict specialist as the universal A8a detector.

### 6.2 A8b rows

| Arm | Vision | Proprio / state | Conflict-trained | Purpose |
|---|---|---|---|---|
| V-only | ✓ | | optional general | vision baseline |
| P-only | | ✓ | optional general | body-state baseline |
| V+P concat | ✓ | ✓ | | ordinary fusion |
| V+P text | ✓ | text summary | ✓/unshaped | REFLECT-style injection |
| V+P latent | ✓ | latent state tokens | | latent without conflict shaping |
| **V+P latent + conflict** | ✓ | latent state tokens | ✓ | full method |
| Oracle modality | privileged | privileged | — | data solvability upper bound |

Latent state injection reuses the **same soft-token splice interface** as Kino-Tokens, but the encoder is trained on REFLECT/manipulator observable state windows. Do not pretend the Go2 1 kHz Kino extractor transfers zero-shot as a calibrated arm encoder; either retrain/adapt the projector on allowed A8b state or report the arm as blocked if state dimensionality is incompatible.

Curriculum for full method:

1. general / hindsight failure SFT,
2. conflict fine-tune with general replay to avoid catastrophic forgetting,
3. keep conflict-only specialist only for mechanism analysis, not overall leaderboard claims.

Seeds: train `[0, 1, 2]` unless an arm is explicitly zero-shot.

## 7. Conflict Strata (A8b)

Pre-registered evidence-structure groups (not quadruped operator names):

| Stratum | Definition | Expected pattern |
|---|---|---|
| E1 Agree | Vision and state support the same failure/success conclusion | V, P, fusion all non-regressing |
| E2 Vision-true | Motion/state similar; truth needs vision (wrong object / wrong place / semantic mismatch) | P fails; V/fusion succeed |
| E3 Proprio-true | Vision ambiguous/occluded/misleading; gripper/joint/contact/force separates (no-close, slip/drop, jam, no-progress) | V fails; P/fusion succeed |
| E4 Nominal / benign | Success or no intervention needed | calibration / false-intervene control |

Labeling rules must be deterministic, code-defined in `kino_vla/eval/a8_strata.py`, and depend only on eval-only metadata + observable signals — never on model predictions.

## 8. Metrics

### 8.1 A8a official

- Accuracy_exec, Accuracy_plan
- macro-F1 (protects class imbalance)
- per-failure-category recall where labels exist
- Wilson 95% CIs
- paired bootstrap vs zero-shot on shared items
- optional ECE / AURC diagnostic (not claimed as formal conformal)

### 8.2 A8b method

- Per-stratum accuracy with Wilson CIs
- Conflict-balanced attribution:
  - `CBA = 0.5 * (Acc_E2 + Acc_E3)`
- Fusion gain:
  - `Δ_fusion = CBA_{V+P} - max(CBA_V, CBA_P)`
- Conflict-training gain:
  - `Δ_conflict = CBA_{V+P+conflict} - CBA_{V+P unshaped}`
- McNemar on paired shared sample IDs
- Optional recovery-admissibility if benchmark-native correction labels exist; use attribution-gated admissible recovery classes, never force quadruped primitives (`Backstep`, `Switch_Gait`)

Headline claim numbers are CBA / Δ_fusion / Δ_conflict, not only overall accuracy.

## 9. Pipeline Stages

1. **download** — Guardian HF assets + REFLECT real (sim conditional)
2. **inspect-state** — A8b admission gate; write `outputs/eval/a8/a8b_state_audit.json`
3. **build-datasets** — A8a VQA cards; A8b multi-sensory cards + E1–E4 labels; leakage-safe views
4. **train** — FailCoT-SFT, Kino-VLA-General, A8b arms under fixed LoRA budget and seeds
5. **eval-a8a** — official splits → `outputs/eval/a8/a8a/`
6. **eval-a8b** — strata metrics → `outputs/eval/a8/a8b/` (or blocked artifact)
7. **report** — `A实验/A8.md` from machine JSON only
8. **record** — update `CLAUDE.md` §2/§4/§5

## 10. Exit Criteria

A8 is publication-grade only if all of the following hold:

1. A8a tables filled for RLBench-Fail, BDV2-Fail, RoboFail, and UR5-Fail (RoboVQA optional but preferred if present in OOD bundle).
2. Same-backbone rows completed: zero-shot, FailCoT-SFT, Kino-VLA-General masked; Guardian listed as reference.
3. A8b either:
   - **passes** state admission gate and reports CBA / Δ_fusion / Δ_conflict with CIs and McNemar on pre-registered strata, **or**
   - is explicitly **blocked** with audit evidence and no surrogate headline.
4. If A8b passes: gains concentrate on E2/E3 as predicted, or an honest null is reported without overselling overall accuracy.
5. `A实验/A8.md` exists with required A-experiment structure (Why, Claim role, Method, Results+CIs, claim bridge, Honest scope).
6. Config hash + seeds + commit + source hashes stamp every table.
7. Unit tests for metrics/leakage/strata pass; no surrogate numbers in headlines.
8. `CLAUDE.md` §2 marks A8 status; §4 has a dated log line; §5 records scope/blockers.

## 11. Honest Scope / Failure Modes

- Official Guardian track is image/text VQA; it alone cannot prove Kino-Tokens help.
- A8b is the load-bearing method experiment and depends on REFLECT multi-sensory state availability.
- Disk pressure: RLBench-Fail test (~46 GB) + REFLECT real (~28 GB) are large; prefer sequential download/extract and delete intermediate tars only after checksum verification and successful build.
- Do not interpret Guardian-8B vs Qwen3-VL-4B gaps as method superiority.
- Conflict specialist models must not be sold as universal detectors on overall official accuracy.
- No closed-loop manipulator recovery benchmark; A4 remains the consequence matrix.

## 12. Deliverables

- Code/config/tests listed in Section 4
- Machine evidence under `outputs/eval/a8/`
- Human report `A实验/A8.md`
- Session record updates in `CLAUDE.md`
- Design + later implementation plan under `docs/superpowers/`
