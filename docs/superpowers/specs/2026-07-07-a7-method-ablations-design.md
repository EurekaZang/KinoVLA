# A7 Method Ablations Design

Date: 2026-07-07
Project: KiNO / Paper-A
Scope: A7 method ablations for publication-grade Paper-A evidence

## 1. Purpose

A7 explains which method ingredients matter after A0-A6 have established the main scientific claims. It serves C2 and the method section by measuring whether learned conflict resolution depends on latent Kino-tokens, conflict-specific curriculum, truth-filtered rationales, privileged-distilled encoders, OOD-theta residuals, and consequence-aware ranking.

A7 must not introduce surrogate metrics or fabricated dependencies. If a real component is missing, the pipeline records a blocked artifact with evidence and the report narrows the claim rather than substituting synthetic data.

## 2. Inputs and Constraints

Binding sources:

- `CLAUDE.md` session protocol, real-stack rule, determinism rule, and A-experiment report requirements.
- `experiments_design.md` v1.1, especially A7 lines for latent-vs-text, conflict-dose, CoT filtering, encoder variants, and claim matrix.
- Existing frozen artifacts under `outputs/eval/a0` through `outputs/eval/a6`.
- Existing A7 code and partial artifacts under `configs/eval/a7.yaml`, `kino_vla/eval/a7_ablation.py`, and `scripts/a7_*.py`.

Hard constraints:

- Do not edit `kino-vla-v2.md`.
- Do not replace missing real `ApiOracle`, CLIP, VLA adapters, Isaac, or trained encoder artifacts with CPU/synthetic surrogates.
- Every table must carry config hash, commit, source hashes, seeds, and CIs where applicable.
- Closed-loop-derived consequence values must be inherited from already-gated A4/A6 artifacts, not newly simulated without the determinism gate.

## 3. Recommended Approach

Use a clean publication-grade consolidation with targeted gap-closing.

Rejected alternatives:

1. Minimal close-out from existing mixed-version artifacts: too weak for publication because it leaves config-hash drift and missing report coverage.
2. Full rerun/retrain of every ablation arm: rigorous in principle, but wastes time on arms already known blocked by absent real artifacts and risks violating the no-surrogate rule.

Selected approach:

- Normalize A7 outputs under the current `configs/eval/a7.yaml` hash.
- Rerun or extend available real arms: text-schema, conflict-dose across seeds, A3/A4/A5/A6 method slices, ERS/regret, OOD-theta and theta-star residual summaries.
- Preserve explicit `blocked` / `requires_run` artifacts for unavailable real dependencies such as unfiltered ApiOracle CoT or untrained encoder variants.
- Generate `A实验/A7.md` with honest scope: strong claims for available real evidence, narrowed claims where latent and text tie, and explicit blockers for incomplete arms.

## 4. Architecture

The A7 pipeline remains small and file-oriented:

- `configs/eval/a7.yaml`: single experiment config and seed grid.
- `scripts/a7_build_datasets.py`: builds dataset views, cards, and blocked-artifact metadata.
- `scripts/a7_train.py`: launches available training arms only.
- `scripts/a7_eval.py`: aggregates native and imported evidence into machine-readable summaries.
- `kino_vla/eval/a7_ablation.py`: shared hashing, Wilson intervals, McNemar, AUROC, dose reducers, rationale grounding, and blocked-artifact helpers.
- `scripts/a7_report.py`: converts summaries into `A实验/A7.md`.
- `outputs/eval/a7/`: machine evidence root.
- `A实验/A7.md`: final human-readable report following the required A-experiment structure.

## 5. Evidence Modules

### 5.1 Latent vs Text Summary

Compare latent Kino-token evaluation against REFLECT-style scalar text and rich per-joint statistics. The key output is not forced latent superiority. If text matches latent on the available corpus, the report states that the latent advantage narrows to token efficiency, integration, and privileged theta-head support rather than raw greedy attribution accuracy.

### 5.2 Conflict-Data Dose Curve

Run or aggregate dose values `{0, 5, 10, 20, 40}` across configured seeds. Report per-dose attribution/correct-recovery, Wilson CIs, seed robustness, and the dose knee. If the curve is non-monotonic, report that honestly and interpret it as evidence that the curriculum has a useful range rather than an always-more-is-better scaling law.

### 5.3 CoT Filtering and Rationale Grounding

Compare truth-filtered vs unfiltered scripted/Oracle CoT only when real unfiltered `ApiOracle` artifacts exist. Otherwise emit a blocked artifact that says the unfiltered real-Oracle arm is absent. Still report available rationale grounding on truth-filtered outputs if test-time rationales exist.

### 5.4 Encoder Ablations

Summarize privileged-distillation, contrastive-only, from-scratch, window length `{25,50,100}`, and anomaly-gate on/off. Only privileged-distillation is currently known available. Other arms remain `requires_run` unless real trained artifacts are found or generated. The report must separate available evidence from pending training, not imply completed encoder comparisons.

### 5.5 ERS / Regret Ranking

Use A4 consequence matrix artifacts to translate attribution choices into expected physical consequence units. This connects A7 method rows back to C3 without running new closed-loop agents.

### 5.6 OOD-theta Abstention and Theta-star Residual

Import A5/A6 residual and risk-coverage artifacts into A7 as method add-ons. The A7 role is to show that privileged-distilled theta residuals are useful ablation evidence for abstention and boundary calibration, not to restate the full C4 claim.

## 6. Data Flow

1. Build dataset views and cards from frozen A0/A2/A3 artifacts.
2. Train missing available arms only when their real data exists and the config requests them.
3. Evaluate all available arms against the same frozen snapshots.
4. Aggregate statistics with paired tests where shared predictions exist.
5. Import consequence and residual summaries from A4/A5/A6 with source hashes.
6. Generate `outputs/eval/a7/*summary*.json`, `outputs/eval/a7/a7_eval_manifest.json`, and `A实验/A7.md`.
7. Update `CLAUDE.md` §2 and §4, and §5 if blockers or scope deviations remain live.

## 7. Error Handling and Honesty Rules

- Missing required source artifact: fail the relevant stage or write a blocked artifact, depending on whether the stage is essential or optional.
- Mixed config hashes: regenerate summaries under the current config where possible and mention inherited old-hash artifacts only as historical sources.
- Missing unfiltered ApiOracle data: block CoT comparison; do not use scripted or synthetic replacement as if it were the real arm.
- Missing encoder variants: mark `requires_run`; do not infer their results from privileged-distillation.
- Failed tests or commands: leave A7 pending and report the exact failure.

## 8. Verification Plan

Run targeted tests first:

- `env -u PYTHONPATH pytest tests/test_a7_ablation.py tests/test_a7_datasets.py`

Run the A7 pipeline:

- `env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python scripts/a7_build_datasets.py --config configs/eval/a7.yaml --stage all`
- `env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python scripts/a7_eval.py --config configs/eval/a7.yaml --stage all`
- `env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python scripts/a7_report.py --config configs/eval/a7.yaml --out A实验/A7.md`

Then run the fast gate if feasible:

- `env -u PYTHONPATH pytest -m "not slow"`

Acceptance criteria:

- `A实验/A7.md` exists and follows the required A-experiment structure.
- A7 summaries contain current config hash, commit, source hashes, and seeds.
- Available arms have CIs/statistics; unavailable real arms are explicitly blocked.
- CLAUDE.md progress state and completed log are updated.

## 9. Expected Paper Claim Shape

A7 should close with a defensible method story:

- Conflict-specific data is efficient but has an empirical knee, not unlimited monotonic scaling.
- Text summaries can match latent on coarse matched O4/O2 attribution, so the latent claim is narrowed where appropriate.
- Privileged theta residuals remain method-critical for OOD abstention and boundary calibration.
- Consequence-aware ERS/regret ranking shows why method differences matter physically.
- Real missing ablations are reported as blocked/pending instead of replaced.
