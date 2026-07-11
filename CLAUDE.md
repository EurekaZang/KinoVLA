# CLAUDE.md — KiNO / Paper-A Development Guide

> **At session start:** read this file, then `experiments_design.md` (Paper-A plan/source of truth for CURRENT work) and `Kino-vla-v2.md` (design spec — never edit; human decision). Before ending any code session, update §2 + §4 (+ §5 if deviations/sign-off changed).

---

## 0. Session Protocol

Loop: **Orient** (this file + `experiments_design.md`, find CURRENT work in §2) → **Plan** (claim C1–C5 + rules R1–R8) → **Implement** (spec/plan wins; flag conflicts in §5) → **Verify** (§3 QA) → **Record** (§2/§4/§5).

**Hard rules:**

- **REAL STACK ONLY — NO SURROGATE (2026-06-20).** Metrics/figures/exit criteria must come from Isaac physical Go2 + live RTX + real CLIP + trained VLA/monitor + real `ApiOracle` (§10 CoT). CPU surrogate / ScriptedOracle / synthetic renders are import-smoke/unit scaffolding only. If a real component is blocked, **STOP and flag in §5 with evidence; never substitute**.
- **COMPUTE IS NOT A CONSTRAINT (2026-07-02).** Workstation: Ryzen 9950X + RTX 5090 32GB. Prefer rigor over cheapness: more seeds, fresh-app/deep-reset determinism, ≥100-snapshot corpora, wide θ-sweeps, adversarial verification.
- **DETERMINISM IS A RED LINE (A0.1/#52).** Any C2ST or closed-loop number must use per-lane-independent collection (fresh app or validated deep reset) and pass the same-operator validity gate (structured same-op split ≈0.5). Reused-app Isaac collections carry operator-order physics residuals and are invalid for headlines.
- **EVERY A-EXPERIMENT NEEDS `A实验/<Exp>.md`.** Required structure: (1) Why, (2) Claim C1–C5 + logical role, (3) Method (real-stack setup/config hash/seeds/R1–R8), (4) Results with CIs + controls, (5) claim bridge/falsifier, (6) Honest scope. Every number traceable to config hash + seeds + commit; no hand-edited tables. Add a one-line §4 pointer.
- **STOP AND FLAG, DON'T SILENTLY RESOLVE.** Never certify a claim without exit criteria. Never modify `Kino-vla-v2.md`. Prefer deleting/simplifying over adding abstraction not needed by the current experiment.

---

## 1. Goal — Paper-A

**Title:** “Feel It, See It, Recover: Cross-Modal Failure Attribution for Safe Quadrupedal Navigation Recovery.”

Object of study: **cross-modal failure attribution** — resolving conflicts between proprioceptive and visual evidence to select the correct recovery. Evaluation currency: **frozen failure snapshots + interventional consequence**, not closed-loop reach.

Claims (full text: `experiments_design.md` §0.1):

- **C1 Necessity, both directions:** O4↔O2 is proprio-byte-identical while vision separates; O7/O8 mirror cases need proprio because vision is misleading/blind.
- **C2 Learned conflict resolution:** vision channel ≠ using it; conflict resolution is a non-trivial semantic operation.
- **C3 Attribution causes recovery/safety:** label-swap matrix shows correct/wrong attribution changes physical outcomes; asymmetric costs imply safe defaults.
- **C4 Generalization + abstention + boundary:** held-out appearances/compositions, OOD-θ residual, O10 θ\*, and O2 modality-dependent intervention boundary.
- **C5 Fair opponent:** B1 is a strong proprio attributor (0.764), so its matched-pair failure is by construction.

**Substrate:** full KiNO framework (11 operators, learned monitor, Kino-Tokens, VLA recovery planner, semantic map, CBF-QP shield, decoupled A* nav, Kino-SFT/DPO) is built and sim-gated on real Go2. It is infrastructure for Paper-A, not the contribution; do not rebuild it.

---

## 2. Progress State _(EDIT EVERY SESSION)_

**Current work:** A8 external validity (Guardian/FailCoT A8a + REFLECT multi-sensory A8b) in progress after A0–A7 completion. KiNO substrate M0–M7 complete on RTX 5090 / Go2; “M8” = Paper-A eval.

**Experiment status (real-Go2 unless stated):**

- **A0 DONE — prerequisite infra.** A0.1 determinism PASS: `deep_reset` + fixed geometry ⇒ cross-order max|Δ|=0.0, gold-match 7/7, same-op C2ST 0.5; controls fail (naive Δ19.8, deep@order-y Δ10.6). A0.2 OracleTrigger; A0.3 frozen corpus 548 snapshots (hash d6f369…); A0.4 appearance library; A0.5 registry. See `A实验/A0.md`, `outputs/eval/a0/`.
- **A1 DONE (C1/C5) — C1 strengthened.** Matched O4↔O2 proprio is byte-identical (max|Δobs48+τ|=0.0); fresh C2ST 0.5, power 1.0, real-CLIP vision 1.0. T3 mirror: O7/O8 proprio decisive (cnn1d 1.0) while vision blind/indist on same-appearance pair (CLIP 0.31). O3↔O1 demoted. See `A实验/A1.md`, `outputs/eval/a1/`.
- **A2 DONE (C2/C5) — publication-grade.** n=240 matched corpus + appearance-held-out + balanced both-directions headline: B1 0.50 < {B-F,B-T,B-V,B5-unshaped} 0.80 < B5-conflict 0.90; test split 0.50/0.67/0.83; McNemar 0,24 p<1e-5; 3 seeds all 0.900. See `A实验/A2.md`, `outputs/eval/a2/`.
- **A3 DONE (C1-T3/C2 bidirectional).** 144 T3 snapshots collected with bang-bang excitation; merged corpus 692. B-V fails T3 (O7 0.19/O8 0.00); B1 complementary on O8 100%; B5-conflict-bi solves T3 0.92 and is proprio-driven across 3/3 seeds. See `A实验/A3.md`, `outputs/eval/a3/`.
- **A4 DONE (C3).** Label-swap M(s,ℓ): 630 episodes (9 scenarios × 7 labels × N=10, hash `7211a437`, commit f8a3f69). Canonical success 1.00[0.72,1.00] vs off 0.00[0.00,0.28] on crux cells; T2 cost antisymmetry 4×; safe-default crossover p*(adhesion)=0.25; conflict-agent regret 0.11 vs baselines 0.44. See `A实验/A4.md`, `outputs/eval/a4/`.
- **A5 DONE (C4).** Appearance OOD scales with modality reliance (B-V −0.21 worst, B1 −0.03 least); LOO-O5 rescued by OOD-θ residual 16.09 ⇒ abstain/Hold ⇒ 100% safe; LOO-O8 is contact-mode boundary; composition 16/16; abstention AUROC 0.87; risk-coverage 1.63 < always 2.80 and never 1.71. See `A实验/A5.md`, `outputs/eval/a5/`.
- **A6 DONE (C4).** O10 θ\*=0.2754 from privileged base sweep (reach 1.0 at floors 0.4/0.3, 0.0 at 0.25/0.2/0.15); OOD-θ residual tracks θ\* (0.629→0.949, crosses ~0.85); O2 remains modality-dependent (proprio detector fires, cross-modal agent attributes compliant terrain). See `A实验/A6.md`, `outputs/eval/a6/`.
- **E4 precursor DONE/PARTIALLY UNBLOCKED.** A/B-aware monitor (`monitor_abaware`) cuts A-class false-fire 5–80×; hardened debounce 25 gives O10 left endpoint (b1 reach 1.00 at floors 0.9/0.7/0.4). See `A实验/E4.md`, `outputs/eval/e4/`.
- **A7 DONE (C2 method ablations) — publication-grade, protocol-clean.** Real artifact-backed A7 under config hash `1640c26279ef`. **Headline conflict-dose is single-recipe only** (unique-sample `dose_XX` + epochs=4 + nav_weight=1.0): best mean O4 attr at **dose 10 = 0.733** (range 0.6–0.8); ERS best regret dose 10 = 0.40; protocol dose5 is unstable (mean 0.333, seed1 O4=0.0) and is reported honestly—not replaced by unequal-recipe adapters. Unequal-recipe upsample×8/epochs=6 stabilization is **sensitivity-only** under `conflict_dose/sensitivity_dose05_upsample/`. Latent-vs-text pure injection claim is the M7 route table (vision 0.458 vs text/latent 1.0; θ head; token ratio 1.27); taxonomy T3/grounding deltas are curriculum-confounded and not sold as pure injection. Truth-filter GC 0.604→0.765; test-time emitted GC up to 0.8; encoder 18/18; A7.1 n=588. No surrogates. See `A实验/A7.md`, `outputs/eval/a7/`.
- **A8 IN PROGRESS (external validity; not Go2 hardware A8d).** Dual-track design+pipeline landed. **A8a real Qwen3-VL-4B zero-shot (official images):** UR5-Fail exec n=140 → **0.529** [0.446,0.609] F1 0.404; BDV2-Fail exec n=1000 → **0.629** [0.599,0.658] F1 0.629; RoboFail n=153 → **0.771** [0.699,0.831] F1 0.711; RoboVQA n=357 → **0.843** [0.802,0.877] F1 0.842. Guardian-8B paper ref: RoboFail 0.86 / UR5 0.77 / RoboVQA 0.85. FailCoT-SFT + RLBench still pending full train images. **A8b unblocked:** REFLECT real multi-sensory admission PASS (30 eps; joint/gripper/force/EE + RGB JPEG-XL). Failure-only E3-dominant corpus: real V-only ZS Acc_E3 **0.967**, P-only state classifier **1.000**, fusion **1.000**; Δ proprio−vision E3=+0.033 (McNemar n.s.). CBA not defined without E2 support. No surrogate proprio. See `A实验/A8.md`, `outputs/eval/a8/`.

**Sim gates:** `pytest -m sim` covers stand, walking/map, M2/M3/M4/M5/M6, decoupled route-around, multi-patch, E1 C2ST, E2/E4 infra, A0.1 determinism, A4 matrix. Fast gate: `env -u PYTHONPATH pytest -m "not slow"` (green incl. A0/A1/A2/A4 tests; use env drop to avoid ROS/`lark` collection issues).

**Env:** RTX 5090 (sm_120) + `~/miniconda3/envs/kinovla` (py3.11, torch 2.7+cu128, Isaac Sim 5.1, IsaacLab 2.3 editable from `~/IsaacLab`). Recipe: `env -u PYTHONPATH HF_HUB_OFFLINE=1 OMNI_KIT_ACCEPT_EULA=YES ~/miniconda3/envs/kinovla/bin/python …`. VLA: `KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct`. Fresh-app lane ≈40 s/lane; deterministic ⇒ 1 lane/op can certify.

---

## 3. Plan & QA

**Plan source:** `experiments_design.md` v1.1. It owns A0–A8 definitions, exit criteria, claim×experiment matrix, agent roster, R1–R8, sequencing, and statistics. This file tracks status + deviations only.

**Binding design rules (R1–R8 summary):** frozen snapshots primary; determinism-gate every closed-loop number; oracle-primary trigger; ground truths independent of evaluated agents; pre-registered success criteria; attribution-gated correct-recovery headline; matching-is-a-family; observable-signal-only triggers with θ never in any gate.

**QA / DoD:** typed APIs + ruff clean; `pytest -m "not slow"` green; sim logic has seeded `@pytest.mark.sim`; real-dependency components use the real dependency (§0); config-driven (no magic numbers); §2/§4 updated; §5 updated for deviations. A-experiments are not done without `A实验/<Exp>.md` and traceable numbers. Safety changes require h_j≥0 property tests + p99<1 ms and adversarial suite. Learned components require fixed seeds and reproducible tracked metrics. Eval tables require config hash + seeds + commit; Wilson CIs, McNemar, lane-grouped bootstrap/permutation for C2ST; ≥100 headline snapshots where feasible; no hand-edited tables; closed-loop tables carry A0.1 certificate.

**Latency contracts (not compute caps):** CBF-QP <1 ms p99; Monitor <1 ms; Extractor <10 ms.

---

## 4. Completed Log _(`date | what | evidence`; quantified summaries live in A实验 reports)_

```text
2026-06-12/26 | M0–M6 KiNO substrate built + real-Go2 gated: operators, policy, CBF, Kino-Tokens, RGB-D/CLIP, Hindsight CoT, learned monitor. | git history; tests/*; outputs/monitor_learned/RESULTS.md
2026-06-19/24 | M7 VLA + decoupled nav complete: Kino-SFT, latent/proprio evidence, DPO sharpening, A* route-around, multi-patch. | outputs/vla/*; scripts/isaac_*nav_check.py
2026-06-29 | E1 C2ST built; reused-app matched result later superseded by A1/#52 deterministic byte-identity certificate. | configs/eval/e1_c2st.yaml; tests/test_c2st.py; A实验/A1.md
2026-06-29 | E2 precursor three-row conflict ablation: B1 0.00 < B5-unshaped 0.27 < B5-conflict 1.00 (n=30). | A实验/E2.md; outputs/eval/e2/
2026-06-30/07-01 | E4 A/B-aware monitor caliper: false-fire ↓5–80×; O10 left endpoint recovered. | A实验/E4.md; outputs/eval/e4/
2026-07-02 | A0 infra done: determinism/deep_reset, oracle trigger, frozen corpus, appearance library, registry. | A实验/A0.md; outputs/eval/a0/
2026-07-02 | A1 C1 done: O4↔O2 byte-identical, fresh C2ST 0.5, T3 mirror, triangulation. | A实验/A1.md; outputs/eval/a1/
2026-07-03 | A2 C2/C5 done: hardened n=240 headline, held-out appearances, 7-agent roster, 3-seed robustness. | A实验/A2.md; outputs/eval/a2/
2026-07-03 | A3 bidirectional conflict done: T3 necessity + B5-conflict-bi proprio-driven 3/3 seeds. | A实验/A3.md; outputs/eval/a3/
2026-07-04 | A4 C3 done: 630-episode label-swap matrix, cost asymmetry, safe default, regret. | A实验/A4.md; outputs/eval/a4/
2026-07-04 | A5 C4 done: appearance/composition generalization, OOD-θ abstention, safe coverage; O5 rescue, O8 boundary. | A实验/A5.md; outputs/eval/a5/
2026-07-04 | A6 C4 done: O10 θ*=0.2754 + OOD-θ residual tracking + O2 modality-dependence. | A实验/A6.md; outputs/eval/a6/
2026-07-07 | A7 method ablations completed: latent-vs-text, 3-seed conflict-dose (best mean dose 10), real ApiOracle truth-filter gain, ERS/regret, OOD-θ/θ* method add-ons, and full 18-cell encoder grid on real binding windows. | A实验/A7.md; outputs/eval/a7/
2026-07-07 | Paper-A manuscript rewritten from claim-ledger/technical-report style into problem-first ICRA narrative while preserving A0--A7 evidence scope. | paper/main.tex; paper/main.pdf; docs/superpowers/specs/2026-07-07-problem-first-paper-rewrite-design.md
2026-07-09 | A7.1 publication-hardening pass: restored direct VLA posterior cache sidecars, fixed ensemble-variance regeneration, preserved exact A7.1 thresholds for traceability, corrected posterior reliability/ECE to use posterior-argmax correctness (ECE 0.471 diagnostic), regenerated A7 report wording, and verified gates. | A实验/A7.md; outputs/eval/a7/abstention_baselines/summary.json; outputs/eval/a7/calibration/ece.json; tests/test_a7_ablation.py; env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python -m pytest -m "not slow" -q
2026-07-09 | A7 full ICRA-hardening audit: closed test-time emitted-rationale grounding (best GC 0.8) and paired T4 taxonomy falsifier; no surrogates. | A实验/A7.md; outputs/eval/a7/{test_time_grounding,text_schema_taxonomy}; scripts/a7_eval.py
2026-07-09 | A7 protocol integrity fix: demoted unequal-recipe dose5 upsample push to sensitivity-only; restored protocol-matched seed1/2 dose05 adapters+per_items+train_meta; headline dose knee back to protocol dose10 (mean O4 0.733, ERS regret 0.40); narrowed latent-vs-rich T3/grounding claims as curriculum-confounded. | A实验/A7.md; outputs/eval/a7/conflict_dose/{summary.json,sensitivity_dose05_upsample}; scripts/a7_report.py; tests/test_a7_ablation.py
2026-07-11 | A8 external-validity design+plan+pipeline; real Qwen3-VL-4B ZS: UR5 0.529, BDV2 0.629, RoboFail 0.771, RoboVQA 0.843; REFLECT multi-sensory A8b admission PASS (30 eps); E3-only P 1.000 > V 0.967 (Δ+0.033, n.s.); CBA deferred without E2. | A实验/A8.md; outputs/eval/a8/; scripts/a8_*.py; kino_vla/eval/a8_*.py
```



---

## 5. Open Deviations / Human Sign-off _(live items only)_

```text
A0.1 DETERMINISM — RESOLVED, pending human sign-off to adopt `deep_reset` as standard collection reset. Acceptance met: deep@fixed-y Δ=0, gold-match 7/7, same-op C2ST 0.5; controls fail; closed-loop O2/O10 confounds vanish. See A实验/A0.md.
A0.3/A0.4/A0.5 PRE-REGISTERED CONFIGS — committed (`configs/eval/{a0_registry,appearance_library}.yaml`, corpus hash d6f369). Later edits break pre-registration and must be flagged here.
#49 E1 SPEC SIGN-OFF — pending: rewrite spec §8.1-P4 from “match summary statistics” to “match joint obs+history distribution, C2ST-certified”; decide whether O4 peel-plateau becomes canonical physics or stays E1/A1-only flag. See A实验/A1.md.
#52 A1 C2ST CONFOUND — pending: replace reused-app E1 numbers in paper with deterministic byte-identity certificate; adopt same-operator validity gate + fresh-app/deep-reset protocol. C1 strengthened. See A实验/A1.md.
experiments_design.md §10 RATIFICATION — still needs human sign-off wording for #49 physics, two-phase O4 certification-travels-with-config, oracle-primary trigger, success/admissible registry, and `monitor_abaware` governance.
A2 SCOPE — no blocker. Note smaller hardened margin (0.80→0.90) vs E2 upper-bound (0.27→1.00); vision-weak appearances are one-per-seed boundary, not info-limit; B-F cracks neither. See A实验/A2.md.
A3 SCOPE — no blocker. B5-conflict-bi is a conflict-only T2/T3 specialist (trades T4/T5); reverse vision-alarm→continue unlearned; B1 misattributes O7, but O8 carries C1/T3. See A实验/A3.md.
A4 SCOPE — no blocker. Sim has no distinct gait; Regret is headline, ERS upper-bound; A1.3 real-Go2 two-phase spot-check is follow-up; O7 looks-safe crossable and O10_decay_B below θ*. See A实验/A4.md.
A5 SCOPE — no blocker. A5.1 uses procedural appearance OOD; LOO-O8 remains contact-mode/nominal-θ boundary needing active probing; B1 AUROC=0.25 is anti because θ-residual is a VLM/fusion abstention signal. See A实验/A5.md.
A6 SCOPE — no blocker. Literal snapshot continue/intervene flip did not hold; result is abstention-residual-tracks-θ*. O2 agents emit benign Switch_Gait rather than literal continue; θ* located only for O10. See A实验/A6.md.
A7 SCOPE — no blocker; protocol-clean post-hardening. No surrogate substitution. Headline dose axis is unique-sample/epochs=4 only (best mean dose10); dose5 collapses stay in the protocol curve; upsample push is sensitivity-only. Greedy latent vs text ties on M7 ambiguity; T4 taxonomy match-at-floor among conflict packages; taxonomy T3/grounding not pure injection (curriculum confound). Conformal UCB empirical not formal; ECE 0.471 diagnostic. See A实验/A7.md.
A8 SCOPE — IN PROGRESS. Public Guardian/FailCoT (A8a) + REFLECT multi-sensory (A8b); A8c/A8d out of scope. A8a ZS table real on UR5/BDV2/RoboFail/RoboVQA; FailCoT-SFT/Kino-General and RLBench still pending. A8b admission passed on REFLECT real zarr (joint/gripper/force/EE+RGB); corpus is failure-only and E3-dominant so CBA/Δ_conflict not yet claim-grade — need E2/success strata (sim_data or success negatives) before selling full C2 external matrix. No surrogate proprio.
Pending experiments — FailCoT-SFT same-backbone A8a rows; A8b E2/success strata + trained latent/conflict adapters.
```
