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
CURRENT MILESTONE : M1 — Kino-Fail operator library v0 + walking skeleton
                    (CPU tier COMPLETE; Isaac-side checks pending GPU)
CURRENT TASK      : GPU verification on the RTX 5090 machine — follow README "GPU
                    machine setup", then: (a) M0: `python scripts/stand_go2.py
                    --headless`; (b) M1: `python scripts/run_demo.py --backend isaac
                    --headless`; (c) `pytest -m sim`. Record results in Section 4,
                    then mark the M0 and M1 checkboxes and start M2.
DEMO STATUS       : GREEN on surrogate backend (asserted in CI + tests/test_demo.py);
                    Isaac backend authored, unverified (Section 6 #4/#5)
LAST SESSION NOTE : 2026-06-12 — M1 CPU tier done per user directive to proceed with
                    M0 left open (Section 6 #3): operator framework + O1/O6/O11 with
                    QA 5.2 gates, surrogate backend, monitor v0, FSM + shield stubs,
                    loop + run_demo. 69 unit tests green, ruff clean.
```

Milestone checklist (mark `[x]` only when ALL exit criteria in Section 3 pass):

- [ ] **M0** — Repo scaffolding, env, CI
- [ ] **M1** — Kino-Fail operator library v0 + walking skeleton demo ← _first deliverable_
- [ ] **M2** — Kino-Monitor + Reflex loop + low-level locomotion baseline
- [ ] **M3** — CBF-QP Safety Shield + Primitive Compiler + latency instrumentation
- [ ] **M4** — Kino-Tokens extractor (privileged distillation)
- [ ] **M5** — Semantic traversability map
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
```
