# CLAUDE.md — KiNO / Paper-A Development Guide

> **At session start:** read this file, then `experiments_design.md` (Paper-A plan/source of truth for CURRENT work) and `Kino-vla-v2.md` (design spec — never edit; human decision). Before ending any code session, update §2 + §4 (+ §5 if deviations/sign-off changed).

---

## 0. Session Protocol

Loop: **Orient** (this file + `experiments_design.md`, find CURRENT work in §2) → **Plan** (claim C1–C5 + rules R1–R8) → **Implement** (spec/plan wins; flag conflicts in §5) → **Verify** (§3 QA) → **Record** (§2/§4/§5).

**Hard rules:**

- **FULL ISAAC STACK ONLY — NO SURROGATE (2026-06-20).** Metrics/figures/exit criteria must come from the Isaac Go2 physics simulation + live RTX + real CLIP + trained VLA/monitor + real `ApiOracle` (§10 CoT). CPU point-robot surrogates / ScriptedOracle are import-smoke/unit scaffolding only. This is not authorization to call the results physical-robot experiments.
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
- **C4 Supported selective recovery + boundaries:** The realistic direct v3 test is complete: 75/75 paired cases over three test scenes/domains, `selective−always-safe=-5.934`, scene-clustered 95% CI `[-8.704,-3.557]`, success delta `+0.200 [0.120,0.293]`, coverage 0.20 and released precision 1.00. All prefix, frozen-gate, order-balance and fresh-process audits pass. The gate releases only supported O4 adhesion; O2/O5/O8/O9 are retained safe-fallback boundaries. Historical A4 `recovery−continue` remains negative side evidence for its old policy/action stack.
- **C5 Fair opponent:** B1/proprio-only is a strong realistic comparator (A2 held-out balanced accuracy 0.970–0.986; A3 macro 0.981–0.982). Multimodal gains require paired cluster-level superiority; scale-v8 must not call matched behavior “by construction” because realistic A1 did not pass its equivalence gate.

**Substrate:** full KiNO framework (11 operators, learned monitor, Kino-Tokens, VLA recovery planner, semantic map, CBF-QP shield, decoupled A* nav, Kino-SFT/DPO) is built and gated with the Isaac Go2 simulation asset. It is infrastructure for Paper-A, not the contribution; do not rebuild it.

---

## 2. Progress State _(EDIT EVERY SESSION)_

**Current work:** A0–A7 evidence audit and ICRA hardening, plus the realistic-scene vertical slice. Machine artifacts are the source of truth; consolidated A0–A7 findings are in `A实验/A0-A7_实验数据汇总.md`, and the realistic-scene plan/review are in `docs/superpowers/plans/2026-07-20-embodiedgen-kinofail-realistic-a0-a7.md` and `docs/demos/indoor_adhesion_icra_review.md`.

**Experiment status (Isaac simulation and frozen trained-model artifacts unless stated):**

- **A0 DONE — prerequisite infra.** A0.1 determinism PASS: `deep_reset` + fixed geometry ⇒ cross-order max|Δ|=0.0, gold-match 7/7, same-op C2ST 0.5; controls fail (naive Δ19.8, deep@order-y Δ10.6). A0.2 OracleTrigger; A0.3 frozen corpus 548 snapshots (hash d6f369…); A0.4 appearance library; A0.5 registry. See `A实验/A0.md`, `outputs/eval/a0/`.
- **A1 DONE (C1/C5) — C1 strengthened.** Matched O4↔O2 proprio is byte-identical (max|Δobs48+τ|=0.0); fresh C2ST 0.5, power 1.0, real-CLIP vision 1.0. T3 mirror: O7/O8 proprio decisive (cnn1d 1.0) while vision blind/indist on same-appearance pair (CLIP 0.31). O3↔O1 demoted. See `A实验/A1.md`, `outputs/eval/a1/`.
- **A2 DONE (C2/C5) — publication-grade.** n=240 matched corpus + appearance-held-out + balanced both-directions headline: B1 0.50 < {B-F,B-T,B-V,B5-unshaped} 0.80 < B5-conflict 0.90; test split 0.50/0.67/0.83; McNemar 0,24 p<1e-5; 3 seeds all 0.900. See `A实验/A2.md`, `outputs/eval/a2/`.
- **A3 DONE (C1-T3/C2 bidirectional; scope narrowed).** 144 T3 snapshots collected with bang-bang excitation; merged corpus 692. B-V T3=0.125 and O8=0; B1 O8=1.0; B5-conflict-bi T3=0.917 across a 3-seed conflict-only recipe. μ sweep is flat 1.0 (robustness, not dose response), and reverse remains 0/16. See `A实验/A3.md`, `outputs/eval/a3/`.
- **A4 MATRIX + COMPOSITION DONE (C3, scope-bounded).** M(s,ℓ): 630 episodes (9×7×10; matrix hash `7211a437`, result hash `4ad75986`). T2 cost asymmetry is 4 vs 1 and p*=0.25. Strict actual-action mapping plus two-stage bootstrap shows B5-conflict improves over its unshaped predecessor (ΔERS −0.5103, CI [−0.8125,−0.175]) but is not the overall winner. Old 0.11/0.44 claims are invalid. See `A实验/A4.md`.
- **A5 COMPLETE WITH A5.6 POSITIVE CLOSURE (C4).** A5.1–5.5 remain negative/boundary: scalar residual gate is worse than safe, LOO naming fails, `n_outside_train_range=0`. A5.6 adds an unchanged-base structured gate and passes final-v2 plus a six-appearance-cluster replication. See `A实验/A5.md`.
- **A6 BASE BOUNDARY DONE; LEARNED BOUNDARY NEGATIVE.** O10 base transition is bracketed at [0.25,0.30]; 0.2754 is descriptive only. All snapshot agents intervene at every floor. Residuals are archived in schema-v2 `a6_eval.json`; floor-group AUROC=1.0 but exact p=0.10. See `A实验/A6.md`.
- **E4 precursor DONE/PARTIALLY UNBLOCKED.** A/B-aware monitor (`monitor_abaware`) cuts A-class false-fire 5–80×; hardened debounce 25 gives O10 left endpoint (b1 reach 1.00 at floors 0.9/0.7/0.4). See `A实验/E4.md`, `outputs/eval/e4/`.
- **A7 DONE (C2 method ablations) — protocol-clean.** Config hash `1640c26279ef`. Dose10 gives mean O4 attr 0.733 and attribution-implied regret 0.40; cached A7 rows lack action params, so this is not actual-action ERS. Matched M7 route: vision 0.458 vs text/latent 1.0, θ head, token ratio 1.27. Truth-filter GC 0.604→0.765; encoder 18/18. A7.1 uses frozen appearance calibration/test + two-stage bootstrap: entropy/MSP test coverage 0.133, cost 1.42, Δsafe CI [−0.20,0]; ECE 0.522; no conformal score feasible. Taxonomy T5 was freshly rerun and is 0 for latent/reflect/rich. See `A实验/A7.md`.

**Sim gates:** `pytest -m sim` covers stand, walking/map, M2/M3/M4/M5/M6, decoupled route-around, multi-patch, E1 C2ST, E2/E4 infra, A0.1 determinism, A4 matrix. Fast gate: `env -u PYTHONPATH pytest -m "not slow"` (green incl. A0/A1/A2/A4 tests; use env drop to avoid ROS/`lark` collection issues).

**Env:** RTX 5090 (sm_120) + `~/miniconda3/envs/kinovla` (py3.11, torch 2.7+cu128, Isaac Sim 5.1, IsaacLab 2.3 editable from `~/IsaacLab`). Recipe: `env -u PYTHONPATH HF_HUB_OFFLINE=1 OMNI_KIT_ACCEPT_EULA=YES ~/miniconda3/envs/kinovla/bin/python …`. VLA: `KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct`. Fresh-app lane ≈40 s/lane; deterministic ⇒ 1 lane/op can certify.

---

## 3. Plan & QA

**Plan source:** `experiments_design.md` v1.1. It owns A0–A7 definitions, exit criteria, claim×experiment matrix, agent roster, R1–R8, sequencing, and statistics. This file tracks status + deviations only.

**Binding design rules (R1–R8 summary):** frozen snapshots primary; determinism-gate every closed-loop number; oracle-primary trigger; ground truths independent of evaluated agents; pre-registered success criteria; attribution-gated correct-recovery headline; matching-is-a-family; observable-signal-only triggers with θ never in any gate.

**QA / DoD:** typed APIs + ruff clean; `pytest -m "not slow"` green; sim logic has seeded `@pytest.mark.sim`; real-dependency components use the real dependency (§0); config-driven (no magic numbers); §2/§4 updated; §5 updated for deviations. A-experiments are not done without `A实验/<Exp>.md` and traceable numbers. Safety changes require h_j≥0 property tests + p99<1 ms and adversarial suite. Learned components require fixed seeds and reproducible tracked metrics. Eval tables require config hash + seeds + commit; Wilson CIs, McNemar, lane-grouped bootstrap/permutation for C2ST; ≥100 headline snapshots where feasible; no hand-edited tables; closed-loop tables carry A0.1 certificate.

**Latency contracts (not compute caps):** CBF-QP <1 ms p99; Monitor <1 ms; Extractor <10 ms.

---

## 4. Completed Log _(`date | what | evidence`; quantified summaries live in A实验 reports)_

```text
2026-06-12/26 | M0–M6 KiNO substrate built + Isaac-Go2 simulation gated: operators, policy, CBF, Kino-Tokens, RGB-D/CLIP, Hindsight CoT, learned monitor. | git history; tests/*; outputs/monitor_learned/RESULTS.md
2026-06-19/24 | M7 VLA + decoupled nav complete: Kino-SFT, latent/proprio evidence, DPO sharpening, A* route-around, multi-patch. | outputs/vla/*; scripts/isaac_*nav_check.py
2026-06-29 | E1 C2ST built; reused-app matched result later superseded by A1/#52 deterministic byte-identity certificate. | configs/eval/e1_c2st.yaml; tests/test_c2st.py; A实验/A1.md
2026-06-29 | E2 precursor three-row conflict ablation: B1 0.00 < B5-unshaped 0.27 < B5-conflict 1.00 (n=30). | A实验/E2.md; outputs/eval/e2/
2026-06-30/07-01 | E4 A/B-aware monitor caliper: false-fire ↓5–80×; O10 left endpoint recovered. | A实验/E4.md; outputs/eval/e4/
2026-07-02 | A0 infra done: determinism/deep_reset, oracle trigger, frozen corpus, appearance library, registry. | A实验/A0.md; outputs/eval/a0/
2026-07-02 | A1 C1 done: O4↔O2 byte-identical, fresh C2ST 0.5, T3 mirror, triangulation. | A实验/A1.md; outputs/eval/a1/
2026-07-03 | A2 C2/C5 done: hardened n=240 headline, held-out appearances, 7-agent roster, 3-seed robustness. | A实验/A2.md; outputs/eval/a2/
2026-07-03 | A3 bidirectional conflict done: T3 necessity + B5-conflict-bi proprio-driven 3/3 seeds. | A实验/A3.md; outputs/eval/a3/
2026-07-04 | A4 matrix done: 630-episode label-swap matrix, cost asymmetry and safe default; agent regret bridge later flagged inconsistent. | A实验/A4.md; outputs/eval/a4/
2026-07-04 | A5 data collected: appearance/composition, residual/LOO/risk-coverage; later audit narrows C4 due coverage=0 optimum and no out-of-range θ. | A实验/A5.md; outputs/eval/a5/
2026-07-04 | A6 base sweep done: O10 θ*=0.2754 interpolation; learned-agent boundary later flagged unclosed. | A实验/A6.md; outputs/eval/a6/
2026-07-07 | A7 method ablations completed: latent-vs-text, 3-seed conflict-dose (best mean dose 10), real ApiOracle truth-filter gain, ERS/regret, OOD-θ/θ* method add-ons, and full 18-cell encoder grid on real binding windows. | A实验/A7.md; outputs/eval/a7/
2026-07-07 | Paper-A manuscript rewritten from claim-ledger/technical-report style into problem-first ICRA narrative while preserving A0--A7 evidence scope. | paper/main.tex; paper/main.pdf; docs/superpowers/specs/2026-07-07-problem-first-paper-rewrite-design.md
2026-07-09 | A7.1 publication-hardening pass: restored direct VLA posterior cache sidecars, fixed ensemble-variance regeneration, preserved exact A7.1 thresholds for traceability, corrected posterior reliability/ECE to use posterior-argmax correctness (ECE 0.471 diagnostic), regenerated A7 report wording, and verified gates. | A实验/A7.md; outputs/eval/a7/abstention_baselines/summary.json; outputs/eval/a7/calibration/ece.json; tests/test_a7_ablation.py; env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python -m pytest -m "not slow" -q
2026-07-09 | A7 full ICRA-hardening audit: closed test-time emitted-rationale grounding (best GC 0.8) and paired T4 taxonomy falsifier; no surrogates. | A实验/A7.md; outputs/eval/a7/{test_time_grounding,text_schema_taxonomy}; scripts/a7_eval.py
2026-07-09 | A7 protocol integrity fix: demoted unequal-recipe dose5 upsample push to sensitivity-only; restored protocol-matched seed1/2 dose05 adapters+per_items+train_meta; headline dose knee back to protocol dose10 (mean O4 0.733, ERS regret 0.40); narrowed latent-vs-rich T3/grounding claims as curriculum-confounded. | A实验/A7.md; outputs/eval/a7/conflict_dose/{summary.json,sensitivity_dose05_upsample}; scripts/a7_report.py; tests/test_a7_ablation.py
2026-07-18 | A0–A7 hardening: corrected A3 T5 truth/action params; rebuilt A4 actual-action composition + two-stage bootstrap; rebuilt A5 frozen-split risk coverage; archived A6 schema-v2 boundary/residual; reran A7 taxonomy and leakage-resistant posterior baselines; synchronized claim ledger and hashes. | A实验/A0-A7_实验数据汇总.md; outputs/eval/a3–a7
2026-07-19 | C4 positive closure: froze supported See–Feel–Act gate; collected/evaluated final-v2 (288) and unchanged-gate six-appearance-cluster replication (300); both strictly beat always-safe with released precision 1.00. | A实验/A5.md; outputs/eval/a5/a5_6_c4_*.json
2026-07-20 | Realistic-scene vertical slice: layered indoor USD, body-fixed Go2 RTX RGB, foot-local adhesion with explicit attach/peel telemetry, same-seed nominal control, two replay views plus eight frozen-state RTX review angles, and reproducible demo bundle. | docs/demos/indoor_adhesion_icra_review.md; outputs/demos/indoor_adhesion_icra/{manifest.json,multiview_peak/}
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
A4 SCOPE — RESOLVED for frozen composition: strict scenario/action mapping and two-stage bootstrap are in `a4_results.json`. Conflict improves over unshaped but not B1/B-T; old 0.11/0.44 is permanently invalid. Closed-loop agent realization remains secondary. See A实验/A4.md.
REALISTIC A4/C4 CORRECTION — The scale-v8 A4 collector source-resolves to the legacy locomotion policy and did not runtime-attest the policy hash. A controlled realistic-route swap improves O4/O9 forward stability but worsens recovery−continue (+10.392→+42.300), because every historical recovery family exceeds the actor's command/posture/mass support. A frozen attribution-blind recovery actor shared by all arms was trained; its development gate improves mean recovery−continue to −5.887 and closes O4 backstep release (2/2 success), while O2/O5/O9 remain unsupported boundaries. The subsequent frozen direct C4 v3 test passes 75/75 paired cases with Δcost −5.934, scene-clustered 95% CI [−8.704,−3.557], coverage 0.20 and released precision 1.00. Preserve historical A4 as negative evidence for that exact old stack; use `a5_c4_direct.json` for the corrected estimand. See `outputs/eval/realistic_a0_a7_v6/a4_policy_*`, `a4_recovery_actor_development_analysis.json`, and `a5_c4_direct.json`.
A5 SCOPE — A5.1–5.5 remain negative/boundary; A5.6 supports only frozen in-material-support selective recovery. Do not rewrite it as out-of-range θ, open-world OOD, or LOO semantics. See A实验/A5.md.
A6 SCOPE — artifact closure resolved; learned boundary is a negative result. Keep [0.25,0.30] base bracket + residual p=0.10 in appendix/failure analysis. See A实验/A6.md.
A7 SCOPE — protocol-clean post-hardening. Dose ERS is attribution-implied because cached rows lack action params. Taxonomy T5=0 after fresh inference. Posterior appearance-heldout Δsafe CI touches zero; ECE 0.522; conformal screen all infeasible. See A实验/A7.md.
REALISTIC SCENE SLICE — demo-only. The 2026-07-20 indoor+adhesion artifact is an authored USD fallback with an EmbodiedGen-compatible visual-layer contract, not an EmbodiedGen-generated scene or A0–A7 paper statistic. It has RTX RGB only, uses separate same-seed front/review replays, and still needs multi-scene/multi-seed statistics, reliable depth and real-Go2 calibration. See docs/demos/indoor_adhesion_icra_review.md.
PAPER NEXT — lock C1/C2/C3 + A5.6 supported selective recovery + method ablations as positive contributions; retain A5/A6/A7 negative subclaims as explicit C4 boundaries. Add group-aware uncertainty to A2/A3 headline and enforce Isaac-simulation wording. Full checklist: A实验/A0-A7_实验数据汇总.md.
```
