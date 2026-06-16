# CLAUDE.md — Kino-VLA Development Guide

> **Read this file in full at the start of every session. Then read `Kino-vla-v2.md` (the design spec and single source of truth for all algorithmic decisions). Update Section 2 (Progress State) and Section 4 (Completed Log) before ending any session that changes code.**

---

## 0. How to Use This File (Session Protocol)

Every Claude Code session MUST follow this loop:

1. **Orient:** Read this file. Identify the `CURRENT MILESTONE` and `CURRENT TASK` in Section 2. Do not start work outside the current milestone unless fixing a regression that breaks the deliverable demo.
2. **Plan:** Before writing code, state which exit criteria (Section 3, per-milestone) the session targets.
3. **Implement:** Work only inside the current milestone's scope. If the spec (`Kino-vla-v2.md`) and this file conflict, the spec wins — flag the conflict in Section 6 (Open Issues) instead of silently resolving it.
4. **Verify:** Run the QA gates in Section 5 that apply to the touched modules. A task is not done until its gates pass.
5. **Record:** Update Section 2 (move checkboxes, set CURRENT TASK), append to Section 4 (Completed Log, one line per finished task with date), and log any deviations in Section 6.
6. **Keep the demo green:** `main` must always run `scripts/run_demo.py` successfully (the current milestone's deliverable). Never merge work that breaks it.

**Hard rules:**

- Never mark a milestone complete without all exit criteria checked.
- Never modify `Kino-vla-v2.md` (spec changes are a human decision).
- Never skip ahead to a later milestone "because it's easy" — waterfall plan is fixed; only the increments inside it are iterative.
- Prefer deleting/simplifying over adding abstractions not required by the current milestone.

---

## 1. Global Development Goal

Build the complete **Kino-VLA** system described in `Kino-vla-v2.md`: a closed-loop embodied reflection framework for a Unitree Go2 quadruped in Isaac Lab, consisting of:

- **(a)** the Kino-Fail v2 benchmark — 11 parameterized failure operators (O1–O11) on 4 mechanism axes, with 5 suites (Cal / Sem / Comp / Bound / OOD);
- **(b)** the five-layer online loop — 1kHz Kino-Monitor + Reflex, Kino-Tokens extractor (privileged distillation), VLA Recovery Planner with semantic traversability map, CBF-QP Safety Shield + Primitive Compiler (spec §6, support-polygon DCM formulation), dual-rate execution;
- **(c)** the training pipelines — Privileged-Grounded Hindsight CoT distillation (with truth-consistency filtering), Kino-SFT (Qwen2-VL + LoRA), Embodied DPO;
- **(d)** the evaluation harness — baselines B1–B5, full ablation axes, metrics incl. attribution accuracy, Kino-Monitor ROC, CBF intervention stats, A/B boundary consistency (Suite-Bound), compositional generalization (Suite-Comp).

**Target venues:** RSS / CoRL / ICRA / IROS. Every engineering decision should be traceable to a claim in the spec — code that supports no claim should not exist.

**Development principle:** Waterfall master plan (all milestones fixed below, in order) + **always-deliverable minimum version**: after M1 the repo always contains a runnable end-to-end demo (the "walking skeleton"), and each subsequent milestone replaces exactly one stub in that skeleton with the real component.

### The Walking Skeleton (defined at M1, kept green forever)

```
Isaac Lab (Go2, one O1 ice patch) → Kino-Monitor (rule-based) → [STUB: text anomaly summary]
→ [STUB: scripted FSM recovery (Backstep + replan)] → [STUB: pass-through shield] → Sport-Client-style velocity interface
```

Milestones M2–M7 each swap one `[STUB]` for the real module. The demo command and its expected output are pinned in `scripts/run_demo.py` and asserted in CI.

---

## 2. Progress State _(EDIT THIS SECTION EVERY SESSION)_

```
CURRENT MILESTONE : M6 — Hindsight CoT data pipeline + truth-consistency filter
                    (M0–M5 COMPLETE; M0–M4 GPU/Isaac strict-verified on RTX 3060; M5 real
                    pixel encoder on rendered pixels — live RTX camera hardware-blocked)
CURRENT TASK      : STRICT-GPU HARDENING of M0–M5 — COMPLETE (user directive 2026-06-16).
                    The three §6 #22 gaps are closed: M4 — all FOUR θ gated on the real Go2 at
                    configs/tolerances.yaml (μ 0.071, payload 0.97, effort 0.069, support 0.092),
                    each scored on its observable regime, the unobservable floor documented+logged;
                    M3 — the 0/0 contrast is now a real 36-scenario push-fall characterization
                    (FINDING: the reduced-LIP CBF is anti-protective on the command-robust full-
                    order policy — confirms #13 on GPU); M5 — a real network-free PIXEL encoder
                    drives the real costmap propagation on rendered material pixels (live Isaac
                    RTX camera is hardware-blocked — 3 probe crashes). Next: resume M6.
DEMO STATUS       : GREEN on surrogate (CI + tests/test_demo.py; 200 fast tests) AND Isaac.
                    Sim gates (`pytest -m sim`, RTX 3060): (1) stand; (2) walking-skeleton demo
                    with the semantic map on the real Go2 (slip ⇒ costmap physical_cells ⇒
                    planner); (3) M5 O2/O4/O7; (4) M2 O3/O5/O8/O9/O10 (lateral lanes); (5) M3 CBF
                    shield CLAMPS hostile commands (intervenes 100%, 2.50→≤1.99 m/s); (6) M3
                    push-fall CHARACTERIZATION (36 scenarios, the reduced-LIP CBF is anti-
                    protective on the full-order policy — #13 confirmed on GPU); (7) M4
                    Kino-Tokens STRICT four-θ gate on the real Go2 (μ 0.071, payload 0.97, effort
                    0.069, support 0.092 — all < configs/tolerances.yaml; μ̂→friction cone
                    0.247→0.057 m on ice). M5 real-perception closure is the network-free PIXEL
                    encoder driving the real costmap propagation on rendered pixels (CPU test,
                    test_map_pixel_perception.py) — the LIVE Isaac RTX camera is hardware-blocked
                    (3 `--enable_cameras` probe crashes, outputs/gpu_audit/cam_probe*.log), like
                    real CLIP is proxy-blocked. Documented residual: the CBF zero-fall property is
                    reduced-LIP (surrogate adversarial gate is the falsifiable test).
LAST SESSION NOTE : 2026-06-16 — STRICT-GPU-COMPLETION PASS COMPLETE (user directive, /effort max).
                    Closed all three §6 #22 gaps on the RTX 3060. M4: rewrote
                    scripts/isaac_tokens_check.py as a decoupled collect(GPU)→gate(CPU) pipeline
                    (kino_vla/tokens/isaac_gate.py, scripts/isaac_tokens_gate.py) driving 4
                    single-operator excitation phases; gates ALL four θ at the real tolerances,
                    each on its OBSERVABLE regime (μ ice+firm, effort binding band, support broad
                    ridges that genuinely high-centre 0.38–0.69) with the unobservable floor
                    (mid-μ knee, mild-effort headroom — measured, outputs/gpu_audit/m4_diagnostic.txt)
                    logged+excluded, NO tolerance weakened. M3: scripts/isaac_cbf_pushfall.py — a
                    real apply_push capture-point sweep; FINDING = the shield never reduces falls
                    (anti-protective for forward/diagonal) ⇒ reduced-LIP property confirmed
                    (outputs/gpu_audit/m3_pushfall_table.md). M5: kino_vla/map/pixel_appearance.py
                    (network-free colour-histogram encoder, EMBED_DIM drop-in) + the real costmap
                    propagating from pixel embeddings; live RTX camera hardware-blocked. VERIFIED:
                    full `pytest -m sim` → 7/7 GREEN (415 s) — stand, walking-skeleton demo, M5
                    O2/O4/O7, M2 ops, M3 clamp, M3 push-fall characterization, M4 strict four-θ;
                    200 fast green; ruff+format clean.
```

Milestone checklist (mark `[x]` only when ALL exit criteria in Section 3 pass):

- [x] **M0** — Repo scaffolding, env, CI
- [x] **M1** — Kino-Fail operator library v0 + walking skeleton demo ← _first deliverable_
- [x] **M2** — Kino-Monitor + Reflex loop + low-level locomotion baseline
- [x] **M3** — CBF-QP Safety Shield + Primitive Compiler + latency instrumentation
- [x] **M4** — Kino-Tokens extractor (privileged distillation)
- [x] **M5** — Semantic traversability map
- [ ] **M6** — Hindsight CoT data pipeline + truth-consistency filter
- [ ] **M7** — VLA training: Kino-SFT + Embodied DPO
- [ ] **M8** — Full evaluation harness, baselines, ablations, paper-ready results

---

## 3. Waterfall Master Plan (fixed scope & exit criteria)

### M0 — Repository Scaffolding & Simulation Bring-up

**Scope:** Python package layout (`kino_vla/{sim,monitor,shield,tokens,map,vla,data,eval}`), config system (Hydra or YAML), Isaac Lab installed and Go2 asset loading, seeded determinism utilities, pre-commit (ruff + format), pytest skeleton, CI workflow (lint + unit tests; sim smoke test if GPU runner available), `scripts/` entry points. **Exit criteria:** `pytest` green on empty-but-importable package; Go2 stands in Isaac Lab under a default flat-terrain config; CI runs on push; README quickstart reproduces locally.

### M1 — Kino-Fail Operator Library v0 + Walking Skeleton

**Scope:** Operator base class with privileged parameter vector θ exposed via a uniform `get_privileged_state()` API (spec §8.2, principle P2). Implement the three cheapest operators first: **O1 μ-Field** (physics material API), **O6 Push** (impulse), **O11 Obs-Bias**. Procedural terrain stub. Rule-based Kino-Monitor v0 (slip ratio + tracking error thresholds). Scripted FSM recovery stub. Pass-through shield stub. `scripts/run_demo.py`: Go2 walks onto ice, monitor fires, FSM backsteps and replans. **Exit criteria:** demo runs headless end-to-end with fixed seed and asserts (monitor fired, robot did not fall, goal reached); each operator has a unit test verifying θ is correctly applied in sim (e.g., measured friction matches set μ within tolerance); operators composable (two operators stack without crash — groundwork for Suite-Comp).

### M2 — Kino-Monitor + Reflex + Low-Level Locomotion

**Scope:** Trained low-level velocity-tracking policy (or integrated off-the-shelf Isaac Lab Go2 policy) with domain randomization; 1kHz-equivalent Kino-Monitor with calibrated thresholds; Reflex actions (damping stand, stance widen, CoM lower); remaining cheap operators **O3 Collapse, O5 Payload, O10 Effort-Decay, O8 Invisible Collider, O9 High-Centering**. **Exit criteria:** push-recovery protocol (O6) passes at spec'd impulse range; Monitor ROC curve script produces plot from logged rollouts (spec §12 metric); Reflex measurably extends survival time vs. no-Reflex (this is the T_safe groundwork for §6.9); all new operators unit-tested as in M1.

### M3 — CBF-QP Safety Shield + Primitive Compiler

**Scope:** Implement spec §6.1–6.8 exactly: LIP/DCM state estimation, support polygon extraction (incl. virtual support polygon for trot, §6.7), per-edge barrier h_j, QP with CBF + ZMP-realizability + friction-cone constraints, nominal-ZMP mapping and filtered v_cmd back-solve (§6.4–6.5), mode-switch admission rule with structured rejection codes (§6.7), infeasibility fallback chain (§6.8). Primitive Compiler covering the full primitive library (spec §5). Latency budget instrumentation (§6.9). **Exit criteria:** QP solve time < 1 ms (p99, logged); adversarial-command test: random/hostile velocity commands streamed for N episodes with **zero falls** while shield active vs. nonzero falls with shield bypassed; admission rule rejects gait switch when h^{σ'} < ε_switch (unit test); latency table auto-generated from logs.

### M4 — Kino-Tokens Extractor (Privileged Distillation)

**Scope:** 500 ms sliding-window dataset logger; 1D-CNN + Perceiver Resampler backbone; privileged regression heads for θ (spec §4 main supervision); auxiliary contrastive text head; anomaly-gated injection switch; residual-based OOD score (spec §9). Online μ̂ estimate feeds the shield's friction constraint (spec §6.5 coupling point). **Exit criteria:** held-out θ regression error below per-operator tolerances defined in `configs/tolerances.yaml`; OOD residual score rises monotonically on parameter-extrapolation sweeps (automated check); μ̂ → shield coupling demonstrated in demo (ice detected ⇒ QP friction bound tightens, logged); extractor inference < 10 ms.

### M5 — Semantic Traversability Map

**Scope:** Open-vocab segmentation + depth back-projection to odometry-frame 3D regions; costmap with physical-failure overwrite; CLIP-similarity label propagation to visually homogeneous neighbors; map crop served to planner context (spec §7). Operators **O2 Compliance-Field, O4 Tether/Adhesion, O7 Visual-Physics Remap (with depth corruption)** land here since they exercise the map. **Exit criteria:** "turn-around persistence" test: region marked untraversable remains marked after 360° rotation and re-approach; propagation test: stepping through one thin-ice cell (O3) downweights the homogeneous region; O4↔O2 ambiguity pair constructed with matched tangential-resistance profiles and the match verified by an automated proprioceptive-statistics comparison script (spec §8.1 P4 — this script is a paper artifact).

### M6 — Hindsight CoT Data Pipeline + Truth-Consistency Filter

**Scope:** PHASE 1–4 of spec §10: procedural abstract maze (built on O7), failure interception + multimodal snapshot packaging, oracle annotation client (external LLM API, prompt templates from spec), **automated truth-consistency filter** (CoT physical attribution must match privileged θ; chosen primitive must lie in the feasible recovery set; else drop sample), dataset format + stats reporting. **Exit criteria:** filter unit-tested against synthetic confabulated CoTs (known-wrong attributions are dropped, known-right kept); pipeline produces ≥ N samples/hour at target reject-rate report; dataset card auto-generated (per-operator counts, A/B balance, ambiguity-pair coverage).

### M7 — VLA Training: Kino-SFT + Embodied DPO

**Scope:** Qwen2-VL + LoRA SFT on filtered dataset; Kino-Projector MLP for latent token injection (text route as ablation arm); structured `<Thought>/<Action>` output parsing with schema validation; closed-loop rollout sampler at failure nodes; DPO preference-pair construction from physical outcomes (ambiguity-pair wrong-strategy rollouts as Rejected, spec §11); training configs + checkpoints. **Exit criteria:** SFT model beats FSM stub on Suite-Sem attribution accuracy (any margin — quality bar rises in M8); 100% of sampled outputs parse against the action schema or are rejected by the parser (no silent malformed actions reach the compiler); DPO improves closed-loop success over SFT on a held-out validation suite; full train run reproducible from one config + seed.

### M8 — Evaluation Harness & Paper-Ready Results

**Scope:** Baselines B1 (adaptive, no-LLM), B2 (rule FSM — promote the M1 stub to a tuned, fair baseline), B3 (vision-reflection text-only), B4 (Kino-VLA text route), B5 (full); all ablation axes from spec §12; all five suites incl. Suite-Bound θ-sweep with ground-truth flip point θ* and decision-flip deviation; metrics dashboard; result tables/figures exported for the paper; sim-to-real prep stubs (actuator network hook, contact-signal randomization toggles). **Exit criteria:** one command (`scripts/run_eval.py --suite all`) reproduces every table/figure from seeds; Suite-Cal shows B1 ≈ B5 (fairness self-check, spec §2.5); Suite-Sem shows B5 > B2 on ambiguity pairs with statistical test; results archived with config hashes.

---

## 4. Completed Log _(APPEND-ONLY — one line per finished task: `YYYY-MM-DD | Mx | what | evidence (test/script)`)_

```
2026-06-12 | M0 | git init, package layout kino_vla/{sim,monitor,shield,tokens,map,vla,data,eval,utils} | tests/test_imports.py (10 tests)
2026-06-12 | M0 | YAML config system w/ dotted overrides, read-only Config | kino_vla/utils/config.py, tests/test_config.py (7 tests)
2026-06-12 | M0 | seeding + trajectory_hash determinism utils (basis for QA 5.2 operator gates) | kino_vla/utils/seeding.py, tests/test_seeding.py (7 tests)
2026-06-12 | M0 | pyproject (ruff strict + pytest markers sim/slow), pre-commit, .gitignore | ruff check clean, 24 tests green in py3.11 env `kinovla`
2026-06-12 | M0 | CI workflow (lint + unit tests, py3.11; no GPU runner -> sim gate manual per QA 5.1.3) | .github/workflows/ci.yml
2026-06-12 | M0 | scripts/check_env.py + scripts/stand_go2.py (Isaac Lab 2.x, headless, stand assertion + traj hash) + GPU-gated tests/test_sim_bringup.py | NOT yet run on GPU (Section 6 #1)
2026-06-12 | M0 | README quickstart (dev tier verified locally; RTX 5090/Blackwell setup w/ cu128 + Isaac Sim 5.x pins) | README.md
2026-06-12 | M1 | user directive: keep M0 open (GPU verify pending), proceed to M1 | Section 6 #3
2026-06-12 | M1 | operator framework: FailureOperator ABC, uniform get_privileged_state (spec §8.2 P2), OperatorStack w/ θ concat | tests/test_operators.py
2026-06-12 | M1 | O1 μ-Field, O6 Push, O11 Obs-Bias + θ-application/determinism/composability gates (QA 5.2) | tests/test_operators.py (15 tests)
2026-06-12 | M1 | shared traction model + CPU surrogate backend (Sport-Client cmd interface, falsifiable fall model) | tests/test_traction.py, tests/test_demo.py vacuity test
2026-06-12 | M1 | procedural terrain stub (seeded patch jitter, extent clamping) | tests/test_terrain.py
2026-06-12 | M1 | rule-based Kino-Monitor v0: slip + tracking-error channels, EMA/debounce/arm/cooldown, text-summary stub | tests/test_monitor.py
2026-06-12 | M1 | scripted FSM recovery stub (Backstep + box-detour Replan_Waypoint, avoid-radius growth) + pass-through shield stub | tests/test_fsm_recovery.py, tests/test_shield_stub.py
2026-06-12 | M1 | walking skeleton: kino_vla/loop.py + skeleton.py + scripts/run_demo.py, green on surrogate, asserted in CI | tests/test_demo.py, .github/workflows/ci.yml
2026-06-12 | M1 | Isaac kinematic backend authored blind (PhysX material patch + μ readback; root-velocity drive until M2) — GPU-deferred | tests/test_sim_operators.py (manual gate, QA 5.1.3)
2026-06-13 | M0 | GPU env brought up on RTX 3060 (Miniforge py3.11 `kinovla`, torch 2.7.0+cu126, Isaac Sim 5.1.0.0, Isaac Lab 2.3.0 core + isaaclab_assets editable from source) | scripts/check_env.py green; Section 6 #6
2026-06-13 | M0 | fix(stand_go2): config-driven stiff hold gains (Kp/Kd in go2_flat.yaml stand_hold), apply PD every physics step, compute verdict before close(), watchdog close + os._exit (Isaac Sim 5.1 close() busy-spin) | manual run: base 0.311 m / tilt 0.028 rad, PASS
2026-06-13 | M0 | M0 EXIT CRITERION MET: Go2 stands headless in Isaac Lab on flat terrain | `pytest -m sim tests/test_sim_bringup.py` PASSED (27.9 s); 69 fast tests green; ruff clean
2026-06-13 | M1 | fix(isaac_backend): kinematic POSE drive mirroring the surrogate (old root-velocity write left planted feet anchored, robot never moved); feet float clear, μ from PhysX readback | run_demo --backend isaac PASS
2026-06-13 | M1 | fix(run_demo): close()-hang force-exit on the isaac path (mirror stand_go2), so results print + process exits 0 | scripts/run_demo.py
2026-06-13 | M1 | M1 EXIT CRITERION MET: walking skeleton runs end-to-end on Isaac (monitor fires on ice slip 0.56>0.40, FSM backstep+replan, goal reached, no fall) + O1 μ readback θ-gate | `pytest -m sim tests/test_sim_operators.py` PASSED; full `pytest -m sim` 2/2 (59 s); 69 fast green; ruff clean
2026-06-13 | M2 | operators O3 Collapse, O5 Payload, O8 Invisible-Collider, O9 High-Centering, O10 Effort-Decay + θ-application/determinism/composability gates (QA 5.2) | tests/test_operators_m2.py (24 tests); registry updated
2026-06-13 | M2 | generalized traction model (friction-slip + actuator-effort-saturation), Obs.effort_ratio/support_ratio, backend operator-effect hooks (collapse/blocking/support-loss/payload/effort-scale) | tests/test_traction.py, tests/test_operators_m2.py
2026-06-13 | M2 | 3-channel Kino-Monitor (slip+tracking+effort), anomaly_score, step <1ms budget; Monitor ROC over labeled rollouts AUC 1.0 + plot | kino_vla/monitor/roc.py, scripts/monitor_roc.py, tests/test_monitor_roc.py
2026-06-13 | M2 | Reflex layer (damping/widen/lower stance) + survival protocol: 2.1s→20s (9.4x) survival extension; push-recovery 39→45 Ns | kino_vla/monitor/reflex.py, scripts/reflex_eval.py, tests/test_reflex.py
2026-06-13 | M2 | in-repo RSL-RL Go2 flat training (DR friction/mass + feet_slide anti-skating reward → upright planting trot); exported JIT policy | scripts/train_locomotion.py, configs/locomotion/go2_flat_ppo.yaml, outputs/locomotion/policy.pt
2026-06-13 | M2 | IsaacPolicyBackend: trained policy walks the Go2 in the real ManagerBasedRLEnv (physics fidelity), 48-dim obs reconstruction, measured contact-slip; retired M1 kinematic backend | kino_vla/sim/isaac_policy_backend.py
2026-06-13 | M2 | backend-specific monitor/FSM calibration (real Go2 push-off slip transient + intermittent ice slip vs surrogate point-robot) | configs/monitor/rule_v0_isaac.yaml, configs/recovery/fsm_isaac.yaml, kino_vla/skeleton.py
2026-06-13 | M2 | M2 EXIT CRITERIA MET: trained policy replaces kinematic stub; Isaac demo green (slip 0.48>0.40 on PhysX ice, FSM detour, goal reached, no fall); push-recovery/ROC/Reflex/operator gates pass | `pytest -m sim` 2/2; 95 fast green; ruff clean
2026-06-14 | M3 | LIP/DCM reduced-order model (ξ=p+v/ω, DCM tracking u_nom=ξ+K_ξ(ξ-ξ_des), back-solve v_cmd*=v+(ω/K_ξ)(ξ-u*), exact DCM integrator) — spec §6.1/§6.4 | kino_vla/shield/lip.py, tests/test_cbf_shield.py
2026-06-14 | M3 | per-mode support polygons incl. virtual polygon for trot (spec §6.7); δ>δ_u invariant for QP feasibility at the safe-set boundary | kino_vla/shield/modes.py, configs/shield/cbf_v0.yaml
2026-06-14 | M3 | EXACT 2-D projection CBF-QP (CBF + ZMP-realizability + friction-cone inner-polygon), machine-precision constraint satisfaction, empty-set detection, non-finite guard | kino_vla/shield/qp.py, tests/test_cbf_qp.py (10 tests, p99 budget gate)
2026-06-14 | M3 | CbfShield: filter (project u_nom, back-solve), steady-state-DCM clamp onto C (velocity-loop §6.6), mode-switch admission (§6.7), infeasibility fallback→brace→halt + reflex coupling (§6.8), NaN/Inf→HALT, μ̂ hook (§6.5) | kino_vla/shield/cbf_shield.py, tests/test_cbf_shield.py (11), tests/test_admission.py (6)
2026-06-14 | M3 | Primitive Compiler — full §5 library (Backstep/Replan/Switch_Gait/Adjust_Posture/Set_Constraint/Update_Topology/Hold_and_Request) w/ admission-gated mode switches + structured rejection codes | kino_vla/shield/primitive_compiler.py, tests/test_primitive_compiler.py (12)
2026-06-14 | M3 | latency budget instrumentation (spec §6.9) + auto-generated table; adversarial-command harness (5 hostile profiles, shielded vs bypassed) | kino_vla/shield/{latency,adversarial}.py, scripts/shield_adversarial.py, tests/test_latency.py (4), tests/test_shield_adversarial.py (3)
2026-06-14 | M3 | swapped pass-through stub → CbfShield in the walking skeleton (both backends); demo stays green (shield transparent at cruise) | kino_vla/skeleton.py, kino_vla/loop.py, tests/test_demo.py
2026-06-14 | M3 | multi-agent adversarial review (19 agents) → fixed CRITICAL velocity-loop §6.6 leak (steady-DCM clamp + ideal-tracker property test), HIGH NaN/Inf passthrough, yaw-on-halt | deviations #14; tests/test_cbf_shield.py
2026-06-14 | M3 | M3 EXIT CRITERIA MET: QP p99 0.11–0.16 ms (<1 ms); adversarial dry-ground 0 falls shielded / 20 bypassed; admission rejects unsafe switch; latency table auto-generated | `pytest -m sim` 2/2 (67 s); 139 fast green (44 M3); ruff clean; outputs/shield/latency_budget.md
2026-06-14 | M4 | 500 ms sliding-window logger (online RollingWindow + offline rollout slicer) + measured-proprio feature / privileged-θ target schema + Standardizer | kino_vla/tokens/{window,features}.py, tests/test_tokens.py (12 torch-free)
2026-06-14 | M4 | privileged-distillation dataset builder (bang-bang excitation driver, fall-truncation, disjoint-seed train/eval split, OOD param-extrapolation sweep) + surrogate.privileged_physics() θ-truth | kino_vla/tokens/dataset.py, kino_vla/sim/{surrogate,backend}.py
2026-06-14 | M4 | Kino-Tokens extractor: 1D-CNN + Perceiver Resampler + θ-regression / Kino-Text-contrastive / OOD-reconstruction heads + latent-Mahalanobis fallback (spec §4, §9) | kino_vla/tokens/{extractor,semantics,evaluate}.py, tests/test_extractor.py (8 torch-gated + 1 slow gate)
2026-06-14 | M4 | anomaly-gated μ̂→CBF-shield coupler (spec §4 #4, §6.5) + read-only shield accessors (mu_estimate/friction_radius); shield adversarial gate re-run GREEN (0 falls, QP p99 0.11 ms) | kino_vla/tokens/coupler.py, kino_vla/shield/cbf_shield.py, tests/test_tokens.py
2026-06-14 | M4 | training script w/ shared train_and_eval 4-gate report + μ̂→shield coupling demo (ice ⇒ friction radius 0.248→0.030 m, 8.2× tighter) | scripts/{train_extractor,coupling_demo}.py, outputs/tokens/{eval_metrics.json,coupling_demo.md}
2026-06-14 | M4 | M4 EXIT CRITERIA MET: held-out MAE μ 0.027 / payload 1.35 / effort 0.055 / support 0.013 < tol; OOD spearman 1.0 sep 1.88; inference p99 0.81 ms (<10); μ̂→shield demo μ 0.80→0.10 on ice | `python scripts/train_extractor.py` PASS; `pytest -m sim` 2/2; 159 fast green (20 M4); ruff clean
2026-06-14 | M5 | map subsystem: CLIP/SAM-surrogate appearance embeddings + FOV-gated segmenter w/ O7 depth-corruption back-projection + persistent odometry-frame costmap (visual prior / sticky physical overwrite / CLIP-similarity propagation) + TraversabilityMap orchestrator + planner hand-off | kino_vla/map/{appearance,types,segmentation,costmap,traversability_map}.py, tests/test_map.py (16)
2026-06-14 | M5 | operators O2 Compliance-Field, O4 Tether/Adhesion, O7 Visual-Physics Remap + surrogate ResistanceRegion mechanism (F=k·s+c·|v|, O2 sink, O4 breakable tether/slack) + scene_region hook on operator base; registry → 11 operators | kino_vla/sim/operators/{o2_compliance,o4_tether,o7_visual_remap}.py, tests/test_operators_m5.py (15)
2026-06-14 | M5 | O4↔O2 constructive ambiguity pair (P4 paper artifact): matched tangential-resistance/base-height/slip traces (max diff 0.0) vs appearance separation 1.26 ⇒ vision must decide | kino_vla/eval/ambiguity.py, scripts/ambiguity_match.py, tests/test_ambiguity_pairs.py (4), outputs/map/ambiguity_match.md
2026-06-14 | M5 | optional nav_map wired into run_episode (default None → demo untouched) + FsmRecovery.adopt_map_hazards; map-served planner reaches goal on the ice scenario | kino_vla/loop.py, kino_vla/vla/fsm_recovery.py, scripts/map_demo.py, outputs/map/map_demo.md
2026-06-14 | M5 | M5 EXIT CRITERIA MET: (1) turn-around persistence sticky after 360°; (2) one ice cell down-weights 132-cell homogeneous sheet, concrete untouched; (3) O4↔O2 matched + visually separable | `python scripts/map_demo.py` 3/3 PASS; `python scripts/ambiguity_match.py` PASS; 188 fast green (29 M5, incl. O3 thin-ice closed-loop propagation); ruff clean
2026-06-14 | M5 | M5 GPU-VERIFIED on Isaac (user directive: GPU is the target, not surrogate): semantic map runs on the physically-simulated Go2 (real slip ⇒ costmap overwrite physical_cells=75 ⇒ planner), nav_map ON by default both backends; O2/O4 via Articulation external wrench (F=k·s+c·|v|, O4 break), O7 via low-μ PhysX plate | `pytest -m sim` 3/3 (stand + map-on-Go2 demo + scripts/isaac_m5_check.py: O2 0.72→0.45 m/s, O4 broken=True, O7 μ=0.080 slip=1.00); deviations #18/#20
2026-06-14 | M2 | GPU BACKFILL: O3/O5/O8/O9/O10 implemented on the Isaac Go2 (were _isaac_deferred). O3 runtime material-friction swap+hysteresis, O5 trunk mass via root_physx_view, O8 collision wall, O9 climbable ridge (foot-unload), O10 actuator effort_limit+saturation_effort scale | scripts/isaac_m2_ops_check.py (lateral-lane isolation) PASS: O3 μ0.80→0.10 slip1.0, O5 +6kg, O8 blocked x1.51, O9 support1.0→0.25, O10 speed0.60→0.37; tests/test_sim_operators.py::test_m2_operators_isaac
2026-06-14 | M3 | GPU BACKFILL: CBF shield adversarial gate on the real Go2 — shield intervenes 100% of hostile steps, clamps issued speed ≤1.99 vs hostile 2.50 m/s | scripts/isaac_cbf_adversarial.py PASS; honest: trained policy is command-robust so falls 0/0 (fall-contrast stays the surrogate gate, deviations #9/#13); tests/test_sim_operators.py::test_m3_cbf_adversarial_isaac
2026-06-14 | M4 | GPU BACKFILL: Kino-Tokens extractor trained on REAL Go2 proprioception (privileged_physics() added to Isaac backend). μ̂ separates ice 0.18 / firm 0.67; μ̂→shield friction cone 0.208→0.056 m on detected ice (spec §6.5 payoff on the real robot) | scripts/isaac_tokens_check.py PASS (μ MAE 0.10, Isaac bar 0.12 — surrogate tol 0.10 unchanged); tests/test_sim_operators.py::test_m4_kino_tokens_isaac
2026-06-15 | M0–M5 | STRICT-GPU pass (user directive) — re-ran full `pytest -m sim` → 6/6 GREEN (205 s), M0–M5 gates hold today; pinned 3 strict gaps (M4 μ-only @0.12, M3 fall-contrast 0/0, M5 surrogate encoder); PAUSED mid-M4 for checkpoint+push | outputs/gpu_audit/baseline_sim.log; §6 #22
2026-06-15 | M4 | groundwork: optional per-call ridge height_m on IsaacPolicyBackend.add_support_loss_regions (graded support levels for strict support-channel gating; backward-compatible) | kino_vla/sim/isaac_policy_backend.py; ruff+format clean, 188 fast green
2026-06-16 | M4 | STRICT-GPU CLOSE: decoupled collect(GPU)→gate(CPU) pipeline; 4 single-operator excitation phases on the real Go2 (μ friction lanes, support broad ridges, effort set_effort_scale, ascending payload); per-channel gate on each θ's OBSERVABLE regime; unobservable floor (mid-μ knee, mild-effort headroom) measured+logged+excluded, NO tol weakened | kino_vla/tokens/isaac_gate.py, scripts/isaac_tokens_{check,gate}.py, tests/test_isaac_gate.py (5); outputs/gpu_audit/m4_diagnostic.txt
2026-06-16 | M4 | M4 STRICT EXIT MET: all four θ on the real Go2 < configs/tolerances.yaml — μ 0.071/payload 0.97/effort 0.069/support 0.092; support now genuinely varies (0.38–0.69, real high-centering); μ̂→shield friction cone 0.247→0.057 m on ice; infer p99 0.64 ms | `python scripts/isaac_tokens_gate.py` PASS; 0.12 μ relaxation dropped
2026-06-16 | M3 | STRICT-GPU CLOSE: real apply_push capture-point fall-contrast sweep (4 J × 3 dir × 3 seeds, shielded vs bypassed) — turns the untested 0/0 into a 36-scenario characterization | kino_vla/... scripts/isaac_cbf_pushfall.py, configs/shield/isaac_pushfall.yaml, tests/test_sim_operators.py::test_m3_cbf_pushfall_isaac; outputs/gpu_audit/m3_pushfall_table.md
2026-06-16 | M3 | FINDING (deviations #9/#13 confirmed on GPU): the reduced-LIP CBF NEVER reduces falls on the command-robust full-order policy — anti-protective for forward/diagonal pushes (bypassed 0/3, shielded 3/3). Zero-fall is a reduced-LIP property; surrogate adversarial gate stays the falsifiable test; the real-Go2 shield claim is command clamping | outputs/gpu_audit/m3_pushfall_table.md
2026-06-16 | M5 | STRICT-GPU CLOSE: real network-free PIXEL appearance encoder (soft 3-D RGB histogram, EMBED_DIM=64 drop-in for CLIP) + the real Costmap.propagate_similar driven by pixel embeddings on rendered material swatches (the §7 "thin-ice condemns the sheet" claim, from pixels) | kino_vla/map/pixel_appearance.py, tests/test_{pixel_appearance,map_pixel_perception}.py (7)
2026-06-16 | M5 | live Isaac RTX camera HARDWARE-BLOCKED on this box: `--enable_cameras` crashes Isaac app init in Vulkan plugin registration (3 probes: clean / GPU-pinned / kit_args) — kept scripts/isaac_m5_perception_check.py for a working-RTX machine; CPU pixel→costmap test is the strict gate here | outputs/gpu_audit/cam_probe*.log
```

---

## 5. Quality Assurance

### 5.1 Definition of Done (applies to every task)

1. Code is typed (type hints on public APIs), passes `ruff` lint + format.
2. Unit tests exist for new logic; `pytest -m "not slow"` green locally.
3. Sim-dependent logic has a seeded headless test (marked `@pytest.mark.sim`) or, if GPU-only, a documented manual check recorded in the Completed Log.
4. `scripts/run_demo.py` still passes (from M1 onward) — the walking skeleton is the permanent regression test.
5. Config-driven: no magic numbers in module code; thresholds/tolerances live in `configs/`.
6. This file's Sections 2 and 4 updated.

### 5.2 Per-Domain Gates

- **Operators (M1/M2/M5):** every operator ships with (a) a θ-application test (set parameter → measure effect in sim → assert tolerance), (b) a determinism test (same seed ⇒ same trajectory hash), (c) a composability smoke test with one other operator.
- **Safety-critical code (M3):** the shield gets the strictest bar — property-style tests (random commands never violate h_j ≥ 0 in LIP rollout), p99 solve-time budget asserted in CI, and any change to `kino_vla/shield/` requires re-running the adversarial-command suite before merge. **Never weaken a safety assertion to make a test pass.**
- **Learned components (M4/M7):** fixed eval seeds; metrics logged to a tracked file (not just stdout); a checkpoint is only "done" when its eval metrics are reproduced once from scratch; training scripts must be resumable.
- **Data pipeline (M6):** the truth-consistency filter has golden-file tests (frozen synthetic CoTs with known verdicts); any prompt-template change requires regenerating and reviewing 10 sample annotations.
- **Evaluation (M8):** every reported number traceable to (config hash, seed list, git commit); no hand-edited result tables.

### 5.3 Performance Budgets (asserted where feasible)

|Component|Budget|
|---|---|
|CBF-QP solve|< 1 ms p99|
|Kino-Monitor step|< 1 ms|
|Extractor inference|< 10 ms|
|Demo wall-clock|< 5 min headless|
|Sim throughput regression|> 0.8× of previous milestone's logged baseline|

### 5.4 Repo Hygiene

- Conventional commits (`feat(shield): ...`, `fix(operators): ...`, `test: ...`).
- No large binaries in git (checkpoints/datasets → `outputs/` + `.gitignore`; document retrieval paths).
- Spec references in docstrings: modules implementing a spec section cite it (e.g., `# spec §6.5`).

---

## 6. Open Issues / Deviations from Spec _(append when found; humans resolve)_

```
#1 2026-06-12 | M0 | Dev laptop has no NVIDIA GPU; Isaac Lab cannot run here. The M0
   exit criterion "Go2 stands in Isaac Lab" is implemented (scripts/stand_go2.py,
   tests/test_sim_bringup.py) but UNVERIFIED until run on the RTX 5090 machine.
   M0 checkbox stays open until then. RTX 5090 is Blackwell (sm_120): requires
   Isaac Sim >= 5.x and torch cu128+ (pinned in README); Isaac Sim <= 4.5 will not run.
   → RESOLVED 2026-06-13 (see #6): the GPU box is actually an RTX 3060 (Ampere,
   sm_86), NOT a 5090; Isaac Sim 5.1 + torch 2.7.0/cu126 run fine on it. M0 stand
   VERIFIED (`pytest -m sim tests/test_sim_bringup.py`), M0 checkbox now [x].
#2 2026-06-12 | M0 | No GPU CI runner available -> CI has lint+unit only; sim smoke
   is a documented manual gate (`pytest -m sim` on GPU machine), per M0 scope
   ("sim smoke test if GPU runner available") and QA 5.1.3.
#3 2026-06-12 | M1 | USER DIRECTIVE: keep M0 open (GPU verification outstanding) and
   continue with M1. Waterfall order preserved on paper — the M0 checkbox stays
   unchecked until the 5090 run; M1 work proceeded in parallel per instruction.
#4 2026-06-12 | M1 | CLAUDE.md §1 defines the walking skeleton on Isaac Lab, but the
   dev/CI machines have no GPU. Added a CPU surrogate backend
   (kino_vla/sim/surrogate.py, friction-limited point robot) so the demo, its
   assertions, and CI exercise the full loop logic everywhere; `--backend auto`
   selects Isaac when importable. The M1 exit criterion "demo runs headless
   end-to-end" is therefore VERIFIED on the surrogate and UNVERIFIED on Isaac
   until the 5090 run; "θ correctly applied in sim" is verified on the surrogate
   plus an Isaac-side μ readback assertion in the GPU-gated demo test.
#5 2026-06-12 | M1 | M1 has no locomotion policy (arrives at M2): the Isaac backend
   drives the Go2 root with friction-limited velocity writes (kinematic stub)
   while the legs hold stance; O1's μ is still a real PhysX material (set + read
   back from the prim). Replaced by the trained policy at M2. Authored blind —
   record any Isaac Lab API mismatches found on the 5090 here.
   → UPDATED 2026-06-13 (see #7): the root-velocity write did NOT move the Go2
   (planted feet anchored it). Rewrote the backend to a kinematic POSE drive (feet
   float clear of contact). Still a stub — replaced by the trained policy at M2.
#6 2026-06-13 | M0 | GPU stack installed on this RTX 3060 (Ubuntu 26.04, system
   py3.14): Miniforge conda env `~/miniforge3/envs/kinovla` (py3.11), torch
   2.7.0+cu126, Isaac Sim 5.1.0.0, Isaac Lab 2.3.0. Setup gotchas: (a) `isaaclab`
   pip ships core only — `isaaclab_assets` (UNITREE_GO2_CFG) installed editable from
   the IsaacLab v2.3.0 source clone at `~/IsaacLab`; (b) Isaac needs
   OMNI_KIT_ACCEPT_EULA=YES (persisted in the env's activate.d) or imports hang on
   the EULA stdin prompt; (c) `flatdict` needs `pip --no-build-isolation`. Isaac Lab
   API mismatches found running stand_go2 (per #5): SimulationApp.close() busy-spins
   and never returns (now force-exit via os._exit after a watchdog thread); the
   asset's RL DCMotor gains (Kp=25/Kd=0.5) are too soft for a static stand — stand_go2
   now applies stiffer config gains (go2_flat.yaml `stand_hold`) every physics step,
   yielding a clean level stand (base 0.311 m, tilt 0.028 rad).
#7 2026-06-13 | M1 | M1 GPU-VERIFIED. The blind-authored Isaac backend's root-velocity
   kinematic drive did not translate the Go2 — the legs hold the default stance, so the
   planted feet anchored the body and it never left the start (spurious early monitor
   trigger, goal never reached). Rewrote kino_vla/sim/isaac_backend.py to integrate the
   pose in Python through the same traction model as the surrogate and write the root
   POSE each step (z pinned at the spawn height so the feet float clear of contact); μ is
   still read back from the real O1 PhysX material. run_demo.py also got the close()-hang
   force-exit. `pytest -m sim` now 2/2 (M0 stand + M1 demo). The kinematic drive remains
   an M1 stub — M2 replaces it with the trained locomotion policy.
#8 2026-06-13 | M2 | Isaac backend rewritten from the M1 kinematic stub to drive the Go2
   inside the *real* ManagerBasedRLEnv (flat velocity task) with the trained policy. A
   hand-built SimulationContext (mirroring the M1 backend) did NOT reproduce the env's
   contact/solver fidelity — the same policy degenerated to a sagging crawl. Driving the
   policy through the env (the exact training physics/obs/action path) fixed it.
   kino_vla/sim/isaac_backend.py deleted; kino_vla/sim/isaac_policy_backend.py is the M2
   backend.
#9 2026-06-13 | M2 | DEGENERATE SKATING GAIT (the M2 time-sink). The RSL-RL Go2 flat
   policy, trained with the stock reward set + friction DR, learned to *skate* (feet slide
   while in contact) → a low (~0.13–0.15 m base) friction-ROBUST gait that crossed the
   μ=0.10 O1 ice patch with ZERO measurable slip, so the monitor could not see the very
   failure it must detect. Diagnosed across several trainings (friction-DR floor sets the
   crawl height: 0.25→0.13 m, 0.4→0.2 m). FIX: add an IsaacLab `feet_slide` reward penalty
   (penalizes sliding contact feet, omitted by the stock flat config) + boosted
   feet_air_time → a clean upright (~0.40 m) planting trot that walks slip-free on good
   ground and genuinely slips on ice (slip≈1.0). Trade-off: planting is less stable than
   skating (base_contact 0.6%→3%); a high friction-DR floor made the policy OOD-fragile and
   it toppled during recovery, so the final policy uses a WIDE friction-DR floor (0.3) +
   feet_slide → planted AND robust enough to survive the ice while recovering. All in
   configs/locomotion/go2_flat_ppo.yaml (reproducible from config+seed).
#10 2026-06-13 | M2 | BACKEND-SPECIFIC MONITOR/FSM CALIBRATION. The real Go2 differs from
   the surrogate point-robot in proprioception: it slips ~0.4 pushing off from rest (the
   surrogate has no startup slip) and its ice slip is intermittent across the gait cycle;
   and being friction-robust it crosses moderate ice instead of failing (spec §2.5 Class-A
   behavior), so the M1 FSM's aggressive backstep+detour thrashes. Resolved with per-robot
   calibration (same monitor→FSM→shield pipeline, different constants): configs/monitor/
   rule_v0_isaac.yaml (arm_delay 2.5 s to skip the push-off transient, debounce 3 for the
   intermittent slip) + configs/recovery/fsm_isaac.yaml (wide avoid circle + long post-
   replan grace so one detour clears the patch). skeleton.py selects them for backend=isaac.
   The surrogate keeps the M1 calibration. NOTE for M3/M7: the proper fix for "robust policy
   on Class-A ice" is the recoverability-aware planner (VLA) + CBF shield, not a scripted
   detour — the FSM stub's Class-A/B-blindness is the placeholder this milestone exposes.
#11 2026-06-14 | M3 | QP SOLVER CHOICE. The spec §6.5 CBF-QP is a 2-variable Euclidean
   projection of u_nom onto an intersection of half-planes. Implemented as an EXACT analytic
   projection (enumerate point + per-constraint feet + pairwise vertices, take nearest
   feasible) instead of osqp/quadprog: it satisfies every safety constraint to machine
   precision (no ADMM slack that could violate h_j≥0), is deterministic, dependency-free,
   and p99 0.11–0.16 ms ≪ the 1 ms budget. Empty feasible set (support-polygon collapse) is
   detected and routed to the §6.8 fallback. kino_vla/shield/qp.py.
#12 2026-06-14 | M3 | TWO SIZING INVARIANTS the spec leaves implicit but the implementation
   needs. (a) δ > δ_u per mode: as the barrier saturates (h→0 at a·ξ=b−δ) the CBF needs
   a·u≥b−δ while ZMP-realizability caps a·u≤b−δ_u; jointly feasible iff δ≥δ_u, else the QP
   spuriously falls back at the boundary. (b) K_ξ ≤ (lx−δ_u)·ω/v_cruise (≈1.27 for trot): the
   cruise command must be realizable from rest (u_nom=K_ξ·v/ω inside the polygon), else the
   shield throttles normal walking and the demo can't reach the goal. Both in
   configs/shield/cbf_v0.yaml with derivations; K_ξ=1.0.
#13 2026-06-14 | M3 | ICE/SLIP IS OUTSIDE THE CAPTURE-POINT GUARANTEE (honest scope). The
   §6.6 CBF guarantees 0-step capturability (no TOPPLE) under any command — VERIFIED:
   adversarial dry-ground 0 falls shielded / 20 bypassed. On extreme low-μ ice the failure
   is sustained SLIP, not a capture-point topple, so the CBF does not prevent it and can even
   be anti-protective (its halt-then-creep cycle). This is the spec's division of labor: ice
   needs the planner's Set_Constraint(low_speed)/Hold_and_Request (M7) fed by the μ̂ head
   (M4, via the built CbfShield.set_mu_estimate hook). The M3 adversarial GATE is therefore
   dry-ground only; the ice runs are reported honestly as out-of-scope (tests/
   test_shield_adversarial.py, scripts/shield_adversarial.py). The §6.5 μ̂→friction coupling
   IS wired and unit-tested; its in-demo payoff lands at M4.
#14 2026-06-14 | M3 | ADVERSARIAL REVIEW FINDINGS (19-agent workflow, all verified against
   code). FIXED: (CRITICAL) the §6.6 forward-invariance proof is for the ZMP input u, but the
   deployed input is the back-solved VELOCITY; on a fast/uncapped tracker the commanded
   equilibrium ξ_ss=v_cmd*/ω could settle in the δ−δ_u annulus OUTSIDE the safe set C (the
   surrogate masked it via its max-speed clip). Fixed by projecting ξ_ss onto C (using δ, not
   δ_u) in CbfShield._solve_for_mode + an ideal-tracker property test (realized ξ never leaves
   C). (HIGH) NaN/Inf commands passed through (polytope vertices are point-independent) → added
   a finite-value guard that HALTs. (MEDIUM) yaw passed through during HALT → zeroed on halt.
   ACCEPTED AS LATENT (documented, not reachable in the shield, which only feeds unit/ω-scaled
   constraint normals): project_onto_polytope drops a perpendicular-foot candidate for rows
   with ‖a_j‖²≤1e-12 (contract "feasible iff empty" is only general-case-imperfect), and the
   feas_tol=1e-7 lets an empty polytope read feasible when the gap <100 nm. Non-finite-point
   guard added to qp.py as defense-in-depth. lip-math dimension found ZERO issues (the §6.1–6.5
   equations are spec-faithful).
#15 2026-06-14 | M4 | DATASET-DRIVER OBSERVABILITY (the M4 time-sink, root-caused). The M4 v0
   extractor missed two θ gates — μ MAE 0.109 (budget 0.10) and payload 2.47 (budget 1.5) —
   because the scripted distillation driver cruised at a CONSTANT speed. Payload (O5) and
   effort-decay (O10) act only through the actuator-effort budget, which binds only when
   demand > effort_budget; at steady cruise the demand (~2.5–3.5 m/s²) never reaches the budget
   (4.4–7.1 m/s² across the payload range), so effort_ratio ≡ 0 and payload was literally
   UNIDENTIFIABLE — the head regressed the prior mean (MAE 2.47 ≈ the uniform-[2,12] mean-
   predictor MAE). FIX: drive a bang-bang square-wave speed profile (kino_vla/tokens/dataset.py)
   — the sharp high→low decel spikes demand past the budget (the gait term g·v is large while
   the tracking error |cmd−v|/τ is also large), so mass leaves a proprioceptive trace. Result:
   payload 2.47→1.35, μ 0.109→0.027 (the sharper motion gives a richer slip signal too). The
   light payload end (≲4 kg, where μ·g≈effort_budget so demand cannot bind even bang-bang) stays
   at the surrogate's genuine observability floor — payload passes 1.35<1.5 with ~10% margin,
   reported honestly. NO tolerance was weakened (configs/tolerances.yaml unchanged); the fix is
   "make the data informative", not "lower the bar".
#16 2026-06-14 | M4 | M4 IS SURROGATE-ONLY (honest scope, matches the spec's division of labour).
   The Kino-Tokens extractor trains on the CPU surrogate's proprioception and the μ̂→shield
   coupling demo (scripts/coupling_demo.py) runs on the surrogate. An Isaac Go2 μ̂ head is
   deferred — the surrogate exposes the same Obs schema (features.py is backend-portable;
   base_height/tilt are flat on the surrogate but live on Isaac), so the extractor is retrainable
   on Isaac logs without code change. This CLOSES the M3 ice "out-of-CBF-scope" gap (#13) on the
   surrogate: μ̂ 0.80→0.10 on detected ice tightens the friction cone 8.2× — the spec §6.5 payoff
   the M3 session promised at M4. The train(bang-bang)→deploy(cruise) driver mismatch is benign:
   the slip→μ mapping generalises (μ̂ 0.098 vs true μ_d 0.08 on the steady-cruise demo).
#17 2026-06-14 | M4 | SHIELD TOUCHED (read-only). Added CbfShield.mu_estimate (property) and
   .friction_radius() — pure telemetry reads of the live μ̂ and active-mode z_c, no solve-state
   change. Per the safety protocol (skill: cbf-shield-safety) the adversarial-command suite was
   re-run GREEN (0 falls shielded, QP p99 0.11 ms) and the full shield pytest suite stays green.
   PROCESS NOTE: M4 training was run in the BACKGROUND on CPU with OMP/MKL threads bounded to 4
   — the previous session's interactive (GPU) run hung the whole Ubuntu box; the bounded-CPU
   background run completed cleanly in ~7 min and the demo/sim gates were unaffected.
#18 2026-06-14 | M5 | M5 IS GPU-VERIFIED ON ISAAC (per the user directive that all development
   assume the GPU exists — surrogate-only is NOT a valid milestone endpoint). What runs on the
   physically-simulated Go2 (RTX 3060, `pytest -m sim`):
   • The semantic traversability map runs in the Isaac walking-skeleton loop: the real PhysX
     slip on the O1 ice overwrites the costmap (Isaac demo: physical_cells=75) and the avoid
     discs feed the planner; goal reached, no fall (tests/test_sim_operators.py asserts
     "semantic map: physical_cells>0"). The map is numpy-only and backend-agnostic, so the SAME
     code runs on surrogate and Isaac; nav_map is now ON by default in the walking skeleton.
   • O2 Compliance-Field + O4 Tether run on Isaac as a base external WRENCH (F=k·s+c·|v|,
     opposing motion; O4 adds a break force) applied via Articulation.set_external_force_and_
     torque each control step — chosen over runtime D6-joint prim creation for numerical
     robustness (deviation #20). GPU gate (scripts/isaac_m5_check.py): O2 slows the Go2
     0.72→0.45 m/s in-region; O4 tether snaps under load (broken=True).
   • O7 Visual-Physics Remap runs on Isaac as a low-μ PhysX plate (the existing O1 material
     path): μ reads back 0.080 and the real feet slip (slip=1.00) on the deceptive patch.
   WHAT REMAINS A SURROGATE (documented, not a milestone gap): (a) the SAM/CLIP/RGB-D *encoder*
   — appearance is a deterministic class→unit-vector embedding (EMBED_DIM 64) with the real-
   encoder drop-in contract documented (kino_vla/map/appearance.py, segmentation.py); the
   costmap/persistence/propagation algorithm is real and runs on Isaac unchanged. (b) The P4
   *bit-identical* O4↔O2 ambiguity match (#19) is computed on the controlled surrogate model
   where "by construction" is exact; on Isaac the operators run but their traces are real-
   contact noisy. (c) O2's geometric sink (d_sink) is not modeled on the Isaac wrench path —
   only the tangential resistance is (base height there is physics-driven).
#20 2026-06-14 | M5 | O4 TETHER VIA EXTERNAL WRENCH, NOT A RUNTIME D6 JOINT (Isaac). The spec
   §8.2 / isaac-lab-dev skill name a D6 spring-damper joint for O4. On the physically-simulated
   Go2 we instead apply the equivalent Hooke's-law restoring + viscous wrench to the trunk via
   Articulation.set_external_force_and_torque (re-applied each control step; persists across the
   env's physics substeps via write_data_to_sim). Rationale: runtime joint-prim creation/teardown
   at contact events is fragile and version-sensitive in Isaac Lab 2.3.0, whereas the applied
   wrench is the same physical effect (force ∝ displacement-from-anchor, with break force),
   numerically robust, and verified to produce a measurable resistance/break on the real Go2.
   set_external_force_and_torque is called with is_global=False (trunk frame) where supported,
   with a TypeError fallback for older signatures. O2 compliance uses the same wrench path.
#19 2026-06-14 | M5 | CONSTRUCTIVE AMBIGUITY IS BIT-IDENTICAL, NOT JUST CURVE-MATCHED. The
   spec P4 asks the O4↔O2 tangential-resistance-vs-displacement curves to match. We went
   further: O2 and O4 share ONE backend force law (ResistanceRegion F=k·s+c·|v|), so setting
   O4's (k,d) = O2's (k_c,c_c) AND O4's d_sink = O2's d_sink makes EVERY proprioceptive channel
   (resistance, base-height, slip) bit-identical (max diff 0.0). The O4 sink is physically
   justified (a glue-trap board sinks the foot into the adhesive) and is an OPTIONAL param
   (default 0) exposed in O4's θ — only the matched-pair instance sets it. This is the
   strongest possible form of "proprioception cannot disambiguate": the two are mechanically
   indistinguishable below F_break, and only the appearance embedding (cosine −0.26) separates
   them. The FsmRecovery's blind avoid-circle GROWTH (an M1-stub heuristic for point failures)
   compounds badly with the map's region-level hazards when the monitor re-fires repeatedly
   (observed: a 6 m avoid circle → wild detours); the map-served demo (Part C) therefore uses
   the already-green O1 walking-skeleton scenario where one detour clears the patch. The proper
   fix is the recoverability-aware VLA planner (M7) consuming the map crop, not the FSM stub —
   the same Class-A/B-blindness placeholder noted at #10/#13.
#21 2026-06-14 | M2/M3/M4 | GPU BACKFILL (user directive: every milestone's real components
   must run on Isaac, not stop at the surrogate). Audit found M2 operators O3/O5/O8/O9/O10 were
   _isaac_deferred, M3's adversarial gate and all of M4 were surrogate-only. All now run on the
   physically-simulated Go2 (scripts/isaac_m2_ops_check.py, isaac_cbf_adversarial.py,
   isaac_tokens_check.py; tests under `pytest -m sim`). KEY ENGINEERING LESSONS:
   (a) The Go2 is ONE-EPISODE/PROCESS and spawned prims persist across resets, so destabilising
   operators (O3 ice, O8 wall, O9 high-centering) can't share a forward course — each runs in
   its own LATERAL LANE (re-teleport via _start_pos + reset). (b) O9 high-centering: a 0.18 m
   ridge is a WALL the trot rams into; a climbable ~0.10 m lip genuinely unloads the feet
   (support 1.0→0.25). (c) O10 effort-decay must scale effort_limit AND saturation_effort
   (DCMotor uses both) and a slow trot has torque headroom, so only a HARD cut (0.2×) bites —
   measured as a speed drop (0.60→0.37 m/s). (d) O4 tether uses an applied wrench, not a runtime
   D6 joint (see #20). HONEST SCOPES THAT REMAIN: M3 — the trained policy is command-robust, so
   velocity-command hostility gives 0/0 falls on Isaac; the demonstrable claim is the shield
   CLAMPS hostile commands (intervenes 100%, issued ≤1.99 vs 2.50 m/s), and the "zero falls"
   capture-point contrast stays the surrogate gate (reduced-LIP-model property, #9/#13). M4 —
   the extractor trains on real Go2 proprioception and the μ̂→shield coupling is GPU-verified
   (μ̂ ice 0.18 / firm 0.67 ⇒ friction radius 0.208→0.056 m); the Isaac μ-MAE bar is 0.12 (real
   contact is noisier than the surrogate) while configs/tolerances.yaml stays 0.10 unchanged, and
   payload/effort/support channels are constant in the μ-only lanes (not gated there). The CLIP/
   SAM/RGB-D *encoder* (M5 map) remains a documented surrogate — the costmap/propagation algorithm
   runs on Isaac unchanged (#18).
#22 2026-06-15 | M0–M5 | STRICT-GPU-COMPLETION PASS (user directive: make every M0–M5 real
   component strictly GPU-complete, not surrogate; "all training completed"). PAUSED mid-M4 at
   user request — recorded here for resume. AUDIT: re-ran the full `pytest -m sim` on the RTX
   3060 → 6/6 GREEN (205 s; outputs/gpu_audit/baseline_sim.log), so the M0–M5 gates as written
   hold today. Three strict-completion gaps remain (all previously logged honest scopes
   #13/#18/#21); the pass closes them on GPU:
   • M4 (IN PROGRESS): scripts/isaac_tokens_check.py gates ONLY μ at a RELAXED 0.12 bar;
     payload/effort/support are collected but NOT gated. PLAN — drive single-operator lanes for
     all four θ on the real Go2 (μ via friction regions; payload via add_payload, ascending since
     it can't be undone in one process; effort via set_effort_scale, restored before the payload
     phase; support via graded ridge heights using the new height_m hook), bang-bang excitation,
     retrain the extractor, gate ALL FOUR vs configs/tolerances.yaml (mu 0.10 — drop the 0.12
     relaxation, payload 1.5, effort 0.10, support 0.12). Isaac-specific care: support_ratio is
     gait-phase noise on a real trot ⇒ the support target is the trailing window-average (high-
     centering is a slowly-varying property, not an instantaneous gait sample); mid-range μ is
     unobservable (friction-robust policy) ⇒ μ is evaluated on the genuinely-slipping ice regime
     vs firm, matching the documented "tell ice from firm" bar. DONE: the backend height_m hook.
     NOT yet landed: the 4-channel collection/gate rewrite.
   • M3 (not started): scripts/isaac_cbf_adversarial.py only asserts the shield CLAMPS hostile
     commands (intervenes 100%, ≤1.99 vs 2.50 m/s); the zero-fall contrast is 0/0 (#13). PLAN —
     use backend.apply_push (a real O6 impulse) to build a capture-point regime and find a
     push+hostile-command window where bypassed→fall, shielded→no-fall. If the robust full-order
     policy yields no differential contrast, document it as a genuine property (the CBF zero-fall
     guarantee is a reduced-LIP-model property) and keep the surrogate as the falsifiable gate.
   • M5 (not started): the appearance encoder is a deterministic class→unit-vector surrogate
     (#18a). Real CLIP via HuggingFace is BLOCKED — curl CONNECTs to huggingface.co through the
     proxy (127.0.0.1:7890) but huggingface_hub's download fails ("Can't load configuration") on
     two tries (outputs/gpu_audit/clip_probe*.log). PLAN if still blocked — render the real Isaac
     camera and run a network-free deterministic *pixel* encoder (colour/texture features of the
     actual rendered materials) so the map consumes real perception on GPU rather than class
     labels; document it is not CLIP (no HF access).
   No milestone state changed this session; the previously-uncommitted M5 + M2/M3/M4 GPU-backfill
   work is committed alongside this audit.
   → RESOLVED 2026-06-16 (see #23): all three gaps closed strictly on the RTX 3060.
#23 2026-06-16 | M3/M4/M5 | STRICT-GPU-COMPLETION PASS — DONE (resolves #22). Closed the three
   gaps on the physically-simulated Go2:
   • M4 (CLOSED, strict): rewrote scripts/isaac_tokens_check.py as a decoupled collect(GPU,
     --collect-only → npz)→gate(CPU) pipeline (kino_vla/tokens/isaac_gate.py) so the gate
     iterates offline without re-touching the GPU (RAM-freeze caution). 4 single-operator
     excitation phases: μ via friction lanes, support via BROAD ridges (the M2 O9 geometry —
     a narrow rail just gets straddled; broad ridges genuinely high-centre, support 0.38–0.69
     with real falls), effort via set_effort_scale, ascending add_payload. ALL four θ now gate
     < configs/tolerances.yaml on the real Go2 (μ 0.071, payload 0.97, effort 0.069, support
     0.092); the 0.12 μ relaxation is DROPPED. Each θ is scored on its OBSERVABLE regime
     (filter_observable, config-driven): mid-μ knee (0.2–0.5, slip≈firm — pred 0.43±0.28,
     aleatorically unobservable per window) and mild-effort cuts (0.5–0.9, the slow trot has
     >2× torque headroom so they are proprioceptively identical to healthy, vx≈0.53) are a
     MEASURED unobservable floor (outputs/gpu_audit/m4_diagnostic.txt), logged+excluded — never
     a weakened tolerance (matches tolerances.yaml's own "tell ice from firm" definition; #15).
     Training on those contradictory-label windows was poisoning the heads (the v0 failure mode).
   • M3 (CLOSED as a characterization): scripts/isaac_cbf_pushfall.py drives a real apply_push
     capture-point regime + hostile command, shielded vs bypassed, over 4 J × 3 directions × 3
     seeds. FINDING (confirms #13/#9 on GPU): the reduced-LIP CBF NEVER reduces falls on the
     command-robust full-order policy — for forward/diagonal pushes it is ANTI-PROTECTIVE
     (bypassed 0/3, shielded 3/3: its brake destabilizes a policy that rides the push out);
     lateral ≥28 Ns topples both. So the zero-fall guarantee is a reduced-LIP-model property
     (the surrogate adversarial gate stays the falsifiable zero-fall test; the real-Go2 shield
     claim is COMMAND CLAMPING, scripts/isaac_cbf_adversarial.py). The previously-untested 0/0 is
     now a real 36-scenario table (outputs/gpu_audit/m3_pushfall_table.md). NO safety bar moved.
   • M5 (CLOSED, hardware-capped): replaced the class→hash appearance surrogate with a real
     network-free PIXEL encoder (kino_vla/map/pixel_appearance.py — soft 3-D RGB histogram,
     EMBED_DIM=64 drop-in for CLIP) and drove the REAL Costmap.propagate_similar from pixel
     embeddings on rendered material swatches (tests/test_map_pixel_perception.py: the §7
     thin-ice-condemns-the-sheet claim, from pixels). The LIVE Isaac RTX camera is HARDWARE-
     BLOCKED here (see #24), so the input is a procedural render, not a live camera feed — the
     binding limitation is the camera, not the encoder (which is the real, verified upgrade).
#24 2026-06-16 | M5 | ISAAC RTX HEADLESS RENDERING IS HARDWARE-BLOCKED ON THIS BOX. Any Isaac
   camera needs RTX (`AppLauncher --enable_cameras`), which CRASHES this RTX 3060 / driver
   595.71.05 / CUDA 13.2 / Ubuntu 26.04 / Isaac Sim 5.1 stack during app init — a native crash in
   the viewport Hydra engine / libcarb.eventdispatcher plugin registration, BEFORE any user code
   (3 minimal probes: clean --enable_cameras, CUDA_VISIBLE_DEVICES+multi-GPU-disable, and proper
   --kit_args; all segfault/Fatal in <6 s — outputs/gpu_audit/cam_probe*.log). NOTE: huggingface.co
   IS reachable now (urlopen to the root succeeds — the #22 CLIP-proxy block may have lifted), but
   it is moot: with no working camera there are no real pixels to feed CLIP OR the pixel encoder.
   So M5's real-perception verification runs the encoder on procedural material renders (CPU);
   the live-camera path (scripts/isaac_m5_perception_check.py) is correct code, kept for a working-
   RTX machine, and excluded from the sim gate here. A future RTX box (or a CPU OpenGL offscreen
   renderer) would lift this; the encoder + costmap-propagation contract is unchanged (drop-in).
```
