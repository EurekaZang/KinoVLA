# Experiment Design — Paper A (v1.1)
## "Feel It, See It, Recover: Cross-Modal Failure Attribution for Safe Quadrupedal Navigation Recovery"

> **One sentence:** This paper is not a navigation system paper. Its object of study is **cross-modal failure attribution** — the cognitive operation of resolving conflicts between proprioceptive evidence and visual evidence to select the correct recovery — and its currency of evaluation is **frozen failure snapshots + interventional consequence measurement**, not closed-loop reach. E1 and E2 already carry the two load-bearing claims; this document designs the experiments that complete the paper around them.
>
> Status of inherited evidence: **E1 (C2ST certification) DONE & reproducible. E2 three-row (open-loop, frozen snapshots) DONE & reproducible. E2 Suite-Cal closed-loop parity DOWNGRADED (E4). E4 (2026-07-01 update) PARTIALLY UNBLOCKED: detector-layer caliper fixed (`monitor_abaware`, A-class false-fire ↓5–80×, B-class TPR flat) + hardened operating point (debounce 25) ⇒ O10 closed-loop left endpoint EXISTS (b1 reach 1.00 at floors 0.9/0.7/0.4; monitor fully silent at 0.9). Remaining E4 items: determinism (item 3 → A0.1 here) and θ\*/agent protocol (item 2 → A6 here *is* that protocol).**

### Changelog v1.0 → v1.1 (incorporating E4 2026-07-01 update)

1. **A6 promoted** from optional/appendix to **conditional main-text** on the O10 axis — its blocking gate (closed-loop left endpoint) is now passed; grid must extend *below* floor=0.4, since θ\*∈(0.15, 0.4).
2. **O2 reframed:** no longer a "broken axis to fix" but the **modality-dependence-of-the-boundary demonstration** — E1+E4 jointly show a proprio-only trigger cannot be made silent on mild O2 without violating the red line; only the cross-modal layer can express `continue` there.
3. **New A1.5 — ambiguity-propagation triangulation** (C2ST bound × B1 attributor × hardened detector, all failing at the same O4↔O2 locus). Zero collection cost; artifacts already on disk; ships first.
4. **New A0.2-secondary + A4.5:** the hardened realistic trigger (`monitor_abaware` @ thr 0.6 / deb 25 / arm 2.5) becomes a validated *robustness condition*; oracle triggering remains primary everywhere.
5. **New R8 red-line rule** (observable-signals-only triggers; θ never enters any gate; A-class negatives labeled by observable outcome) — required to keep θ\* non-circular, now proven workable by E4.
6. **A0.1 (determinism) is now the single remaining closed-loop blocker** (E4 item 3); risk register and sequencing updated; #48 governance decision added to §10.

---

## 0. Scope pivot and claim structure

### 0.1 What the paper claims (C1–C5)

- **C1 — Necessity (constructive):** There exists a class of locomotion failures for which **no pure-proprioception function can attribute the cause**, by construction: the matched pair O4↔O2 is certified indistinguishable (C2ST AUC≈0.50 on the binding obs48+torque representation, all T, all discriminators — E1), while vision separates it perfectly (CLIP AUC=1.0). Symmetrically, there exists a class for which **vision alone is misleading and proprioception is decisive** (visual–physics deception). Neither modality subsumes the other. **v1.1 strengthening:** the ambiguity now shows up at *three independent proprio-only layers* — discriminator family, trained attributor, and deployed-grade detector (A1.5) — establishing it as a property of the information channel, not of any one model class. *(Carried by A1 + A1.5 + A3.)*
- **C2 — Conflict resolution is a learned, non-trivial semantic operation:** having a vision channel ≠ using it. B1 (strongest pure-proprio) 0.00 < B5-unshaped 0.27 < B5-conflict 1.00 on matched-O4 conflict cases, CI-separated (E2). The new bidirectional battery (A3) tests whether what is learned is genuine *evidence weighing* or a degenerate "always trust the camera" shortcut. *(Carried by A2 + A3 + A7.)*
- **C3 — Attribution correctness causally determines recovery success and safety:** an interventional label-swap study shows the same physical scene under different attributed labels yields opposite outcomes — including catastrophic ones (fall / catapult / immobilization) under misattribution, and measurable cost when intervening on benign anomalies. Misattribution costs are **asymmetric**, which yields a principled safe-default rule under uncertainty. E4 has now *fully quantified* the false-intervention chain in the wild (saturated detector → misattribution → `Set_Constraint` → stranded at final_dist≈3), giving A4's nominal rows a measured real-world referent. *(Carried by A4.)*
- **C4 — Generalization, calibrated abstention, and the intervention boundary:** attribution generalizes across held-out appearances, held-out operator families, and operator compositions; the privileged-distillation residual is a usable OOD/abstention signal; risk–coverage analysis connects abstention to the asymmetric costs of C3; and on the unlocked O10 axis the agent's intervene/continue flip point can now be measured against an agent-independent θ\* — while on O2, "when to intervene" is itself shown to be a cross-modal question. *(Carried by A5 + A6.)*
- **C5 — Fair-opponent guarantee:** B1 is a strong attributor (0.764 attribution fidelity on unshaped operators, TPR 0.977 / FPR 0.05), so its 0.00 on the matched pair is by construction (E1 upper bound), not under-training. *(Carried by existing E1+E2 artifacts; no closed-loop parity table needed.)*

### 0.2 What the paper explicitly does NOT claim

No navigation-system claim, no SLAM/map contribution, no monitor *contribution* (see cut list — the monitor work is scoped infrastructure + one corroboration datum), no formal safety-shield contribution, no sim-to-real locomotion contribution. Navigation outcomes appear **only** as the measured consequence of attributed labels (A4), never as a system benchmark.

### 0.3 v2.0 → Paper A: cut list

| v2.0 component | Paper A disposition |
|---|---|
| §6 CBF Safety Shield (full derivation) | **Cut.** The A4 scripted executor uses a fixed safety envelope (velocity/torque caps + fall-abort). CBF cited as complementary future work, one paragraph. |
| §7 Semantic traversability map | **Cut.** `Update_Topology` survives only as a *recovery label* in the admissible-set registry; no persistent map is built or evaluated. |
| Kino-Monitor learning + ROC (#48) | **Still not a contribution; now validated, scoped infrastructure.** Primary protocol stays **oracle triggering**. The E4 fix chain (saturation probe → A/B-aware retrain `monitor_abaware` → hardened operating point deb=25) is exploited for exactly two purposes: (i) **A1.5** detection-layer corroboration of C1; (ii) **A4.5 / A6-secondary** realistic-trigger robustness. Deployed `monitor.pt` stays byte-identical; new checkpoint default-off pending sign-off (§10.5). The first-ever A-class FPR measurement (effective ≈100% pre-fix) goes to an appendix infrastructure note. |
| §11 Embodied DPO | **Cut** (future work). Paper A's training story is SFT with scripted/truth-filtered CoT only — the "minimal-agent principle" from E2 stands. |
| §6.9 latency budget, five-layer runtime | **Cut.** One architecture figure, no timing claims. |
| Kino-Fail v2 full suite structure | **Re-cut** around the conflict taxonomy (§3). Suite-Sem core survives; Suite-Cal becomes the *nominal rows* of A4; Suite-Bound becomes A6 (now viable on O10); Suite-Comp/OOD survive at the attribution level inside A5. |
| Sim-to-real actuator network, real-robot closed loop | Only if optional A8 (snapshot-level transfer) is run; no closed-loop hardware. |
| Locomotion policy (re)training | **Frozen everywhere.** RSL-RL base policy is fixed infrastructure. |

---

## 1. Design rules extracted from E1/E2/E4 (binding for every experiment below)

- **R1 — Frozen snapshots are the primary evaluation substrate.** E2 proved single-decision attribution on disk snapshots is deterministic and re-runnable (n=30 rerun reproduced exactly); E4 proved closed-loop reach currently is not. Every headline number lives on frozen snapshots.
- **R2 — Any closed-loop number must first pass the determinism gate (A0.1)** and must be shown θ-driven, not residue-driven: report order-permutation variance alongside every closed-loop table. *(E4 evidence hardened this rule: O10 base flipped 0.00→1.00 at identical θ across runs; O2 k_c=6 base ∈ {0.40, 0.67}; floor=0.5 b1 shows a non-monotonic dip from one spurious fire. Even privileged θ\* curves are gated by A0.1.)*
- **R3 — Oracle triggering is the primary protocol.** Attribution is evaluated *given* an event; detector operating point is a separate problem. E4's 07-01 update *fully quantified and partially repaired* the confound this rule isolates: in-patch hazard EMA saturated at 1.000 with **no A/B gradient** (negatives never contained A-class perturbations) → misattribution → `Set_Constraint` strand. The repair (A/B-aware retrain + debounce 25) does not change the isolation argument for A2–A5; it supplies (a) a validated realistic trigger for robustness checks (A0.2-secondary) and (b) a third independent locus of the C1 ambiguity (A1.5).
- **R4 — Ground truths independent of evaluated agents.** Wherever a boundary or optimum is claimed (A4 canonical recoveries, A6 θ\*), it is defined by privileged/scripted controllers, never by B1/B5 (E4 §4.4 — and E4 item (2) has now independently converged on exactly this protocol).
- **R5 — Per-scenario pre-registered success criteria.** Reach-in-40s conflicts with legitimately slow recoveries (E4 §6.4). Each A4 scenario registers its criterion *before* any agent is evaluated.
- **R6 — Headline metric = attribution-gated correct-recovery rate** (attribution correct ∧ primitive ∈ admissible set of the *true* cause), with the ungated feasible-rate kept only as a logged diagnostic (E2's presentation fix — B1's lucky 0.567 must never resurface as a headline).
- **R7 — Matching is a family, not a point.** #49 certifies O4≡O2 for any shared (k_c, c_c) force law; the specific magnitude is a free parameter. Exploit this freedom (A4 uses it to place the pair where the consequence structure exists; A1 re-certifies whatever point is chosen — re-certification is cheap).
- **R8 — Red line for any learned gate (NEW, proven workable by E4 §3.3–3.6):** trigger logic consumes only observable signals + fixed thresholds; **θ never enters any monitor/gate**; A/B-graded negative labels come from *observable outcome* ("base policy crosses without falling"), never from θ. Longer debounce = "require the hazard signature to persist," a pure observable-signal criterion. This is what keeps the θ\* measurement in A6 non-circular, and the paper states it explicitly.

---

## 2. Prerequisites — A0 (shared infrastructure, gates before dependent experiments)

**A0.1 Closed-loop determinism gate** *(blocks A4, A6 closed-loop parts; snapshot experiments don't need it; NOW THE SINGLE REMAINING E4 BLOCKER — item 3).*
Fix the reset-residue problem. Two candidate mechanisms, decided by an acceptance test, cheapest first: (i) deep reset — explicitly re-randomize/zero joint states, contact buffers, and operator state on `backend.reset(seed)`; (ii) fresh backend per episode (slow but sound fallback; measure app-restart wall cost before committing).
*Acceptance:* same (scenario, θ, seed) run k=5 times in 2 different orderings → identical outcome label and trajectory divergence < ε; the E4 reconcile case (O2_A isolated vs after O1_A) must agree; and the two known noise signatures must vanish: the O10 floor=0.5 non-monotonic dip and the O10 base 0.00↔1.00 flip at identical θ. Report the permutation variance in the appendix — a methodological contribution in itself.

**A0.2 Trigger protocols.**
*Primary — oracle:* privileged hazard-onset event from sim state (entry into tagged region / operator activation time + fixed Δ), emitted identically for every agent. One implementation, unit-tested, reused by A2–A6.
*Secondary — hardened realistic trigger (NEW; used only by A4.5 and A6's closed-loop confirmation):* `monitor_abaware` (A-class per-window false-fire ↓5–80× vs deployed — O1 0.238→0.019, O5 0.657→0.008, O11 0.935→0.000; O2 residual 0.102 misattributed→O4; clean/maneuver FPR 0.020; B-class TPR 0.952) at **threshold 0.6 / debounce 25 / arm 2.5 s** (A-class lane-fire 0.048 — the single firing lane is O2, the certified-ambiguous operator; B-class TPR 0.977; clean/maneuver FPR 0.00). *Latency caveat:* debounce 25 ≈ 0.5 s persistence — acceptable for robustness checks, not for the primary consequence matrix. Governance: default-off checkpoint, deployed `monitor.pt` byte-identical (#48, §10.5).

**A0.3 Snapshot corpus scale-up + freeze.** Extend `matched_eval` from n=30 to the Paper-A corpus: schema `[5×RGB, 5×depth, proprio window (binding obs48+τ12, T∈{25,50,100}), θ (privileged), true label, admissible-recovery set, taxonomy cell, appearance-id, pair-id, seed]`. Target ≥100 test snapshots per taxonomy cell (§3), seeds ≥500 (disjoint from all training seeds, preserving E2's discipline), multiple appearances per semantic class (A0.4). Frozen with content hashes; all agents evaluate the same snapshots → paired statistics (McNemar) come for free.

**A0.4 Appearance library.** Per semantic class, ≥4 procedural appearances (adhesive: yellow board / gray tape / translucent sheet / checker decal; mud: brown / dark gray / reddish / wet-sheen; ice: 2–3 variants; plus "hazard-colored benign floor" decals for the reverse probes in A3). Train/test appearance splits pre-registered. *This is the direct counter to the strongest predictable review attack on E2: "B5-conflict learned yellow→Backstep, not conflict resolution."*

**A0.5 Success-criterion + admissible-set registry.** One YAML: per scenario → success criterion (R5), admissible recovery set, canonical recovery + parameters. Canonical recoveries are **scripted** controllers on top of the frozen base policy (Backstep(d), high-step traverse, slow+low-posture traverse, detour around tagged region, hold) — validated once by hand, then frozen (R4).

---

## 3. The conflict taxonomy — the benchmark spine (replaces Suite structure)

Every scenario in the corpus is assigned one cell. The taxonomy is the paper's Figure 1, because the predicted **block-structured failure pattern** of single-modality agents across it *is* the thesis:

| Cell | Evidence structure | Scenarios (operators) | Who can solve it (prediction) |
|---|---|---|---|
| **T1 — Agree** | vision & proprio consistent | matched-O2 (mud, looks like mud), O1 ice (looks like ice) | everyone with either modality (B5-unshaped already 0.87 on O2 control) |
| **T2 — Conflict, vision-true** | proprio misleading (matched), vision decisive | **matched-O4 vs matched-O2** (E1-certified), O3↔O1 if the temporal leak closes (A1.2) | fusion only; B1 at chance *by construction*; B-V (vision-only) passes |
| **T3 — Conflict, proprio-true** | vision misleading/deceptive, proprio decisive | O7 visual–physics remap both directions (looks-safe/is-slippery with depth corruption; looks-hazardous/is-nominal decal), O8 invisible collider | fusion only; B-V fails *by construction*; B1 passes |
| **T4 — Fine-structure, proprio-sufficient** | no terrain-visual signal; proprio separable but only in temporal/joint-wise fine structure (E1: AUC 0.96–1.0) | O5 payload vs O10 actuator decay | B1 and latent-fusion pass; **coarse text-summary injection predicted to fail** → the latent-vs-text lever |
| **T5 — Nominal / mild** | benign anomaly; correct answer is `continue` | **O10_A anchored at floor=0.9 (validated: hardened monitor fully silent, closed-loop reach 1.00 — E4 §3.7)**, O1_A, O6 push; **O2_A (k_c=6) carries the certified ambiguity even at nominal θ** — the proprio-only detector cannot be silenced on it (E4 §3.6); extend the grid to k_c∈{2,4} if a fully-clean O2 nominal row is needed | tests abstention/non-intervention; feeds A4's false-intervention rows |

**Detection-layer corroboration (new, under the table):** at the hardened operating point the *only* A-class operator that still fires is O2 — the same pair E1 certified indistinguishable, with per-window fires misattributed to O4. The ambiguity thus surfaces at three independent proprio-only layers: the C2ST discriminator family (AUC≈0.50), the trained attributor B1 (0.00 on matched cases), and a deployed-grade detector after an honest A/B-aware fix. A1.5 packages this triangulation.

Honest labels carried from E1: O5↔O10 is *not* claimed as a matched pair (it is a within-symptom-class cause-attribution problem); O3↔O1 is claimed only if A1.2 closes the cnn1d temporal leak, otherwise reported as "marginal" and excluded from certified claims.

---

## 4. Experiments

### A1 — Certification, extended *(status: core DONE in E1; extensions)*
**Serves C1.** Pure characterization; no agents trained.

- **A1.1 Tighten statistics.** Re-run the calibrated O4↔O2 grid with more seeds (10/8 → 16/12) to shrink the wide T=100 cnn1d CI ((0.35,0.67) → target half-width ≤0.10). Point estimates are already 0.49–0.53; this is polish, not risk.
- **A1.2 Close the O3↔O1 temporal leak (regime tightening).** Restrict windows to post-collapse-transient (window start ≥ t_collapse + Δ, Δ swept over {0.2, 0.5, 1.0}s), re-run C2ST. Gate: cnn1d AUC CI covers 0.5 at all T ⇒ O3↔O1 promoted into T2 with the scoped claim "indistinguishable *after* the transient, i.e., at the moment a re-plan decision is made." If it will not close after 2 iterations → demote permanently; the taxonomy stands on O4↔O2 alone.
- **A1.3 Certify every point the paper uses.** Whenever A4 re-places the matched pair at a different shared (k_c, c_c) (R7) or introduces the two-phase operator (A4.1), re-run C2ST restricted to the decision-window regime (pen < p₀). Certification travels with the config hash; the paper states "every matched configuration used anywhere in this paper carries its own C2ST certificate."
- **A1.4 Vision-side certificates for T3.** For O7/O8 scenarios: image discriminator (CLIP linear probe) at chance across the deceptive boundary *(trivial by construction — still report)*, and proprio discriminator (the E1 toolkit) AUC ≥ 0.95 across it — establishing the mirrored necessity claim with the same machinery.
- **A1.5 Ambiguity propagation across proprio-only layers *(NEW — zero collection cost; artifacts on disk; ships first)*.** One table/figure assembling three independent proprio-only systems failing at the same locus: (i) the C2ST bound (E1: AUC≈0.50 across all discriminators and T); (ii) the strongest trained attributor (E2: B1 0.00 [0.00, 0.20] on matched-O4); (iii) a deployed-grade detector after honest A/B-aware retraining + operating-point hardening (E4: O2 is the sole residual A-class firing lane at debounce 25; its per-window fires are misattributed to O4). Sources: `outputs/eval/e1/calibrated/`, `outputs/eval/e2/three_row.json`, `outputs/eval/e4/{monitor_probe_abaware,optpoint}.json` + the `monitor_abaware` report. Claim served: **C1 — the indistinguishability is a property of the information channel, not of any one model class.** Cheapest new experiment in the plan.

*Cost:* offline CPU analysis + a few 108-lane collections. Days, not weeks. **Do A1.5 immediately, then A1.2/A1.3 — they gate claims, and E1 showed iteration-1 usually fails.**

### A2 — Headline three-row, hardened *(status: DONE at n=30; scale + fortify)*
**Serves C2, C5.** Frozen snapshots, open-loop single decision, no Isaac at eval time.

- Scale matched_eval to ≥50 O4-conflict + ≥50 O2-control (A0.3), with the **appearance-held-out split** (A0.4): B5-conflict trains on ≤3 appearances per class, tests on the held-out ones. This is the shortcut-killer.
- **Extend the agent column** (full roster in §5): add B-V (vision-only: same VLM, Kino tokens masked), B-T (text-summary injection, REFLECT-style fixed schema), B-F (closed-set fusion classifier: CLIP features ⊕ proprio-encoder features → MLP over the label set — the honest "you don't need a VLM" baseline), and one off-the-shelf VLM zero-shot row for context.
- Metrics per R6: attribution accuracy, attribution-gated correct-recovery rate, Wilson 95% CIs, paired McNemar between adjacent rows on shared snapshots; ungated feasible-rate logged only.
- Predicted headline: monotone three rows survive at n≈100 with tight CIs; B-F is *expected to be competitive in-distribution* — that is fine and is said out loud, because B-F is then shown to collapse on A5 (appearance OOD, LOO-operator, composition) and cannot emit rationales, recovery parameters, or open-vocabulary labels. The paper's primary claims (C1–C3) are about the problem, not about VLM superiority; the method claims live on the generalization axes.

*Cost:* one snapshot-collection pass (real Isaac, ~150 lanes), one B5-conflict LoRA retrain per appearance split (E2 measured 712 s wall — trivially cheap), CPU/GPU eval minutes.

### A3 — Bidirectional conflict battery *(NEW — the paper's central figure)*
**Serves C1, C2.** The decisive test that B5-conflict learned *evidence weighing*, not *vision dominance*.

- **A3.1 Build T3 scenarios** (O7 both directions + O8) with depth-channel corruption per spec §8.2-O7 (otherwise the D channel trivially sees through the deception); snapshot them into the corpus.
- **A3.2 Reverse-conflict probe.** Appearance says hazard, proprio says nominal (hazard-colored decal on normal floor; fake-ice texture on μ=0.8) → correct answer: `continue`. An agent that learned "conflict ⇒ trust camera" backsteps around paint. Also the mirrored T2/T3 grid: an agent that learned "conflict ⇒ trust body" pushes into matched-O4.
- **A3.3 Training variants.** B5-conflict (vision-true conflicts only — the E2 artifact) vs **B5-conflict-bi** (both conflict directions in the 20–40 scripted-CoT samples, same recipe). Prediction worth publishing either way: if B5-conflict fails A3.2, the shortcut is real and B5-conflict-bi repairs it → "conflict resolution must be taught *per evidence-direction*"; if B5-conflict passes, the operation generalizes from one direction → stronger claim. Both outcomes are findings.
- **A3.4 Override dose–response curves.** Snapshot-level, θ-graded: in looks-safe/is-hazard scenarios sweep μ ∈ {0.6 … 0.09}; plot P(agent overrides the benign visual prior) vs proprio evidence strength; mirrored curve for the reverse direction (visual alarm strength vs P(override toward continue)). Two curves per agent — the quantitative signature of evidence weighing vs modality dominance. No closed loop needed.
- **Deliverable figure:** taxonomy (T1–T5) × agents (B1, B-V, B-T, B-F, B5-unshaped, B5-conflict, B5-conflict-bi) accuracy heatmap. The predicted **complementary block failures** of B1 (dies on T2) and B-V (dies on T3, T4) with only fusion covering all cells *is* "Feel It, See It" in one image.

*Cost:* scenario/decal implementation ~days; snapshots one Isaac pass; per-variant LoRA retrains ~12 min each; eval cheap.

### A4 — Interventional consequence: the label-swap matrix *(NEW — replaces all closed-loop navigation evaluation; carries "Recover" and "Safe")*
**Serves C3.** Question: does the *label* causally determine the physical outcome? Everything except the label is held fixed: oracle trigger (A0.2-primary), scripted canonical recoveries (A0.5), fixed base policy, deterministic backend (A0.1).

- **A4.1 The two-phase (delayed-divergence) O4 operator — resolves the E2 #50 matched-vs-trap tension.** E2's G1/G2 sweep failed because it demanded one force law be simultaneously matched (constant plateau ≡ O2) *and* a trap, everywhere. Dissolve the tension by separating regimes in penetration depth: `grip(pen) = k_c` (matched plateau, ≡ O2's law) for pen ≤ p₀; `grip(pen) = k_c + k₂·(pen − p₀)` beyond, with either finite f_break (→ **catapult**: sudden unloading under forward lean) or k₂ high + f_break=∞ (→ **immobilization**). This is defensible adhesion physics: the peel front saturates locally (plateau) until bulk stretch re-engages (ramp). One new knob (`p0_m`, `k2`) in the existing #49 mechanism.
  - *Placement constraint:* p₀ > max penetration reached at decision time across all lanes (from manifests: cruise 0.6 m/s, snapshot at monitor-intercept ≈ early; verify pen(t_snapshot) distribution and set p₀ with margin; if tight, use T=50 windows or drop cruise to 0.4 m/s). Then **A1.3 re-certifies** C2ST on pen<p₀ windows — by construction the force laws are identical there, so this is a verification, not a gamble.
  - *Pair placement (R7):* choose the shared plateau magnitude k_c such that the canonical O2 recovery (high-step traverse) actually succeeds — E4 showed the *nominal* gait fails k_c≥10, but high-step is untested; run a quick pre-sweep of high-step over k_c∈{6,8,10,14} and place the pair at the largest k_c the canonical recovery clears. Re-certify (cheap).
- **A4.2 The matrix.** Scenarios S (~10): matched-O4-two-phase, matched-O2, O7 looks-safe/is-hazard, O7 reverse, O8, O5-B (saturation), O10-B (heavy decay), + nominal rows O1_A / O2_A(k_c=6) / **O10_A(floor=0.9, now validated as a clean nominal row)**. Forced labels L (~6): {continue, backstep+detour, high-step traverse, slow+low-posture traverse, hold&request, detour-replan}. N=10 seeds per (s,ℓ) → ~600 episodes. For each: pre-registered success (R5 — e.g., O2: "escaped hazard region ∧ no fall ∧ net forward progress", not reach-in-40s), fall, immobilization (v<ε for >5 s in-hazard), catapult/energy-release event, time, peak |ω|.
- **A4.3 Readouts.**
  - **M(s,ℓ) success matrix** — the causal object. Claim lands if diagonal ≫ off-diagonal with the *specific antisymmetries*: `high-step` on matched-O4 → catapult/immobilization (catastrophic) while `backstep+detour` on matched-O2 → success-but-slow (benign); `continue` on nominal rows wins while any intervention loses time or strands — **E4's quantified pathology (misfire → `Set_Constraint` → stranded at final_dist≈3, reach 0.00 where base reached 1.00) becomes the measured cost of false semantic intervention.**
  - **Cost asymmetry → safe default.** From M, compute the expected-cost-minimizing action under label uncertainty; show it is the conservative recovery when P(adhesion) is non-negligible — the decision-theoretic content of "Safe" that replaces the CBF chapter.
  - **Composition metrics linking A2/A3 to consequence without closed-looping the agents:** ERS(agent) = E_s[Σ_ℓ P̂_agent(ℓ|snapshot_s)·M(s,ℓ)] and Regret(agent) = E_s[M(s,ℓ\*) − M(s,ℓ̂_agent)]. This is how B1's 0.00 attribution becomes a *predicted physical failure rate* — attribution error measured in units of falls.
  - **A4.4 Composition validation (agent-in-the-loop, small):** B5-conflict-bi truly closed-loop on 3 scenarios × 10 seeds; check realized success ≈ composed ERS within CI. Requires A0.1. If A0.1's fix caps throughput, this subsection shrinks before anything else does.
- **A4.5 Trigger-robustness condition *(NEW)*.** Re-run the A4.4 validation subset (and 2–3 matrix rows) with the **hardened realistic trigger** (A0.2-secondary) replacing the oracle. Expected: identical conclusions on O10/T3 scenarios (monitor silent or a single clean fire — E4 §3.7 shows b1 reach 1.00 at floors 0.9/0.7/0.4); on O2-family rows the residual ambiguity fires are reported as-is (they are the A1.5 signature, not noise). Purpose: pre-empt "your oracle trigger assumed away detection" with one cheap appendix table. Caveat the ~0.5 s persistence latency for fast-divergence scenarios (two-phase O4 catapult) — which is exactly why oracle stays primary; exclude that row from the realistic-trigger subset if latency dominates.

*Cost:* the dominant Isaac budget: ~600 + 30 + 60 episodes × ~40 s sim + reset overhead. Feasible in days on the 5090 if deep-reset passes A0.1; if fresh-backend-per-episode is required, batch by scenario and expect ~2–3× wall.

### A5 — Generalization and calibrated abstention
**Serves C4.** All frozen-snapshot; splits pre-registered in A0.3/A0.4.

- **A5.1 Appearance OOD** (held-out textures per class; plus 2–3 real-photo textures projected onto sim geometry). Report per-cell accuracy deltas for every agent; B-F predicted to degrade most.
- **A5.2 Leave-one-operator-out.** Train without an operator family (e.g., O3 entirely); zero-shot test scored by (i) open-vocabulary label judged against cause description, (ii) **admissible-set membership of the chosen primitive** (the metric that matters even when the class name is novel), (iii) abstention rate.
- **A5.3 Compositional stacking.** O1+O5, O4+slope, O2+O10 snapshot sets (parameter-vector concatenation per spec §8.3 — no new mechanisms). Multi-factor attribution: exact-match and per-factor recall; primitive scored against the composed admissible set.
- **A5.4 OOD-θ residual.** Privileged-distillation head's θ-prediction residual vs distance outside the training θ range: monotonicity + AUROC as an OOD score.
- **A5.5 Selective prediction (the "Safe" closer).** Risk–coverage curves where abstain → the A4-derived conservative default; y-axis is **expected physical cost via M**, not just error rate. One figure ties C3 and C4 together: a calibrated attributor with a safe default strictly dominates both always-intervene and never-intervene.

*Cost:* mostly retrains-on-splits (cheap LoRA) + snapshot evals; A5.3 needs one composed-snapshot collection pass.

### A6 — Severity boundary on the O10 axis + the modality-dependent boundary on O2 *(PROMOTED: conditional main-text; was optional)*
**Serves C4.** E4's 07-01 update removed the blocker this experiment was gated on: with `monitor_abaware` + debounce 25, the **closed-loop left endpoint on O10 exists** (b1 reach 1.00 at floors 0.9/0.7/0.4; monitor fully silent at 0.9), where before the fix every θ was 0.00. Independently, E4 item (2) converged on exactly this design's protocol: θ\* from a privileged controller, semantic agent (not B1) on the decision side — the two documents now agree by construction.

- **θ\* (ground truth, R4):** privileged base condition (monitor suppressed, frozen policy), deterministic backend (A0.1), N≥10/θ, logistic fit crossing 0.5. **The grid must extend downward:** E4 §3.7 shows base=1.00 even at floor=0.4 while §3.3 shows startup paralysis at 0.15 ⇒ **θ\*∈(0.15, 0.4)**; sweep floor∈{0.4, 0.3, 0.25, 0.2, 0.15}. The early §3.1 base zeros at floors 0.4–0.7 were residue-contaminated — direct evidence that **A0.1 gates the θ\* curve itself**, not just agent rows.
- **Agent side (snapshot-level, primary):** at each θ, B5-conflict-bi (and the roster) outputs `continue` vs an intervention label from frozen snapshots; report decision-flip point vs θ\* with overlaid curves. This satisfies E4 item (2)'s "real semantic agent, not B1" requirement without putting the agent in the loop.
- **Closed-loop confirmation (secondary, appendix):** the E4 leftpoint harness with B5 swapped in for B1 under the hardened trigger, at 3 θ points spanning θ\*. Requires A0.1 — the floor=0.5 non-monotonic dip (0.4 passes, 0.5 fails on one spurious fire) is exactly the noise A0.1 must remove.
- **O2 is deliberately NOT a boundary axis — it is the modality-dependence demonstration.** E1+E4 jointly show a proprio-only trigger *cannot* be made silent on mild O2 without violating R8: the certified ambiguity lives below the detector (at debounce 25, O2 is the only A-class operator still firing, misattributed to O4). So on O2 the intervene/continue boundary is only expressible by the cross-modal layer: show snapshot-level that B5-conflict-bi outputs `continue` on mild mud (vision confirms mud; benign) at θ where the proprio-only detector necessarily fires. One panel, and it turns E4's "hardest axis" into a headline point: **on ambiguous terrain, even *when* to intervene is a cross-modal question.** (If a fully-clean O2 nominal row is also wanted for A4, probe k_c∈{2,4}.)
- **Gates:** A0.1 for the θ\* curve and the closed-loop rows (the last remaining E4 item); the O10 downward pre-sweep. Kill criterion softened accordingly: if θ\* cannot be located deterministically, ship snapshot-level decision curves against the coarse bracket (0.15, 0.4) and say so.

### A7 — Method ablations
**Serves C2 and the method sections.**

- Latent Kino-tokens vs text-summary injection, with two honesty levels of the text schema: (i) fixed REFLECT-cited schema, (ii) a *rich* per-joint statistics variant — if rich text matches latent on T4, the latent claim narrows to bandwidth/integration, and the paper says so.
- **Conflict-data dose curve:** matched samples ∈ {0, 5, 10, 20, 40} → O4-conflict accuracy. Locates the knee (E2 used 20); "a 20-sample curriculum installs the operation" is a strong efficiency claim if it holds.
- Truth-filtered vs unfiltered scripted/Oracle CoT (spec §10 PHASE-3 filter), plus **rationale grounding rate** at test time: run the same automatic checker on emitted rationales (does the stated physical cause match privileged θ?) — measures whether accuracy comes with grounded explanations or confabulation.
- Privileged-distillation vs contrastive-only vs from-scratch encoder (attribution + θ-regression error + A5.4 residual quality). Window length T ∈ {25,50,100}. Anomaly-gated injection on/off.

### A8 — Real-world snapshot transfer *(OPTIONAL; only if hardware time is free)*
Staged hazards on physical Go2 (adhesive mat, elastic cord, low-μ board, foam/mud pit, weighted backpack), synchronized 1 kHz proprio + RGB-D, teleoperated approach, snapshot at anomaly; offline attribution zero-shot and after few-shot encoder calibration. ~20 trials × 5 hazards; confusion matrix; 2–3 qualitative recovery clips for the video. **Snapshot-level only — no closed-loop hardware navigation.** If the encoder gap is large, report as limitation; sim remains primary.

---

## 5. Agent / baseline roster (single table, reused across A2–A6)

| ID | Description | Exists? | Role |
|---|---|---|---|
| B1 | Strongest pure-proprio attributor (MonitorNet 1D-CNN+Perceiver, window 50, retrained; 0.764 fidelity unshaped) | ✔ E2 | C1 negative arm; T3/T4 positive arm |
| B-V | Same VLM, Kino tokens masked (vision-only) | new (trivial) | mirrored negative arm (dies on T3/T4) |
| B-T | Text-summary injection into same VLM (REFLECT-schema; + rich variant in A7) | new | latent-vs-text lever, T4 |
| B-F | Closed-set fusion classifier (CLIP feats ⊕ proprio-encoder feats → MLP) | new (small) | honest "no-VLM" baseline; expected to force the win onto A5 axes |
| B5-unshaped | Qwen3-VL-4B + LoRA + Kino-Projector, trained on unshaped ops only | ✔ E2 | "has vision ≠ uses vision" middle row |
| B5-conflict | + vision-true matched samples (20, scripted CoT) | ✔ E2 | E2 headline |
| B5-conflict-bi | + both conflict directions | new (12-min retrain) | A3 shortcut test; likely the paper's final model |
| Zero-shot VLM | off-the-shelf, vision(+text-summary) | new (API/eval only) | context row |

*(Not agents: `monitor_abaware` + hardened operating point is scoped infrastructure — A0.2-secondary, A1.5 datum, A4.5/A6 robustness. Deployed `monitor.pt` untouched.)*

---

## 6. Claim × experiment matrix

| | A1 (+A1.5) | A2 | A3 | A4 (+A4.5) | A5 | A6 | A7 | A8 |
|---|---|---|---|---|---|---|---|---|
| C1 necessity (both directions) | ●core + ●triangulation | ○ | ●T3 side | | | ○(O2 panel) | | |
| C2 learned conflict resolution | | ●rows | ●bi + dose–response | | | | ●dose curve | |
| C3 attribution ⇒ consequence/safety | ○(A1.3 gates) | | | ●core + ○robustness | ○(A5.5 uses M) | | | |
| C4 generalization + abstention + boundary | | | | | ●core | ●O10 boundary + O2 modality-dependence (gated A0.1) | ○ | ○transfer |
| C5 fair opponent | ●(bounds) | ●(0.764 row) | | | | | | |

Every experiment serves ≥1 claim; no claim rests on a single fragile leg; nothing closed-loop is load-bearing except A4 — which is interventional and scripted, not an agent system.

## 7. Statistics protocol (uniform)

Wilson 95% CIs on all proportions; McNemar for paired agent comparisons on shared frozen snapshots; lane-grouped bootstrap + two-sided permutation for all C2ST numbers (E1 machinery, unchanged); ≥100 snapshots per headline cell (CI half-width ≲0.09 at extremes), N=10 seeds per (s,ℓ) cell in A4 and per θ in A6; all train/eval seed sets and appearance splits disjoint and pre-registered in configs; every table stamped with config hash + commit + seed list (E1/E2 convention); closed-loop tables additionally stamped with the A0.1 permutation-variance certificate.

## 8. Sequencing (cheap → load-bearing → early, per the E-series criterion)

1. **A1.5 immediately** — zero collection cost, artifacts already on disk; a finished figure this week.
2. **A1.3 / A1.2** — certification of every configuration the paper will stand on (cheapest gates; expect iteration-1 failures, as E1 did).
3. **A0.3/A0.4 corpus + A2 hardening** — locks the headline at scale with the appearance-held-out split (kills the biggest known review attack early).
4. **A3** — central figure; only LoRA retrains + snapshots.
5. **A0.1 determinism gate** — in parallel from day 1; **it is the single remaining E4 blocker (item 3)** and the long pole for A4.4/A4.5/A6.
6. **Pre-sweeps:** O10 downward grid (floors {0.4→0.15}, base condition) and A4.1 placements (p₀; high-step over k_c; optional O2 k_c∈{2,4} nominal probe) → **A4 matrix (+A4.5)**.
7. **A6 (promoted)** once A0.1 passes; snapshot-level decision curves can start earlier against the bracketed θ\*.
8. **A5, A7** interleaved as GPU allows; **A8** only if hardware time appears.

## 9. Risk register

| Risk | Detection | Fallback |
|---|---|---|
| Two-phase O4 leaks into decision window (A1.3 fails) | C2ST > chance on pen<p₀ windows | increase p₀ margin / shorten window T / slow cruise; last resort: consequence phase uses native O4 with an explicit bridging argument, and the paper reports the gap honestly |
| No k_c where high-step clears matched-O2 | A4.1 pre-sweep all-fail | switch O2 success criterion to escape-based (R5) and/or re-place pair at lower k_c (R7 + re-certify) |
| B-F matches VLM everywhere incl. A5 | A5 deltas ≈ 0 | narrow method claim to open-vocab labels + recovery parameters + grounded rationales + abstention; C1–C3 are method-agnostic and unaffected — say so prominently |
| B5-conflict fails reverse probe (vision-dominance shortcut) | A3.2 | expected-possible; B5-conflict-bi is the repair and the finding ("evidence weighing must be taught bidirectionally") |
| Appearance-held-out collapse (color shortcut) | A2 held-out split | diversify training appearances, retrain (cheap); publish the shortcut→concept transition as an ablation |
| θ\* base curve unstable without A0.1 | evidence already in hand: O10 base 0.00↔1.00 flips at identical θ; O2 k_c=6 base ∈ {0.40, 0.67}; floor=0.5 b1 non-monotonic dip | deep reset / fresh backend + N≥10; if unresolved, ship snapshot-level decision curves vs the bracketed θ\*∈(0.15, 0.4) and say so |
| θ\*(O10) below tested grid | base=1.00 at floor 0.4 (E4 §3.7) | extend grid to 0.15 (known startup-paralysis point) — pre-sweep already scheduled in §8.6 |
| Hardened trigger latency (~0.5 s) distorts fast-divergence rows | A4.5 divergence on two-phase-O4 catapult | oracle stays primary; exclude that row from the realistic-trigger subset; report the latency explicitly |
| A0.1 unfixable at reasonable cost | acceptance test fails on both mechanisms | shrink A4.4/A4.5 to zero, keep the A4 matrix with fresh-backend batching (scripted recoveries need fewer episodes than agent loops); A6 degrades to snapshot-level curves vs bracketed θ\* |
| O3↔O1 leak won't close | A1.2 after 2 iterations | demote pair; taxonomy stands on O4↔O2 + T3 |
| Rich-text summary matches latent on T4 | A7 | latent claim narrows to integration/bandwidth; report both schemas |

## 10. Open spec-level decisions to ratify (human sign-off, per E1 §8.2 / E2 #50 / E4 §6.1)

1. Adopt **#49 peel-plateau as O4's official physics** and rewrite spec §8.1-P4 from "match summary statistics" to "match the joint distribution of the consumed obs+history vector, verified by C2ST" — this is what E1 actually established and what the paper must state.
2. Adopt the **two-phase O4** (A4.1) as the consequence-grade operator, with p₀ and the certification-travels-with-config rule.
3. Ratify **oracle-primary trigger** wording (detection out of scope as a contribution) so the paper's scope statement and the codebase stay consistent.
4. Ratify per-scenario success criteria (A0.5 registry) before any A4 run — pre-registration only counts if it is actually prior.
5. **NEW — #48 governance:** decide whether `monitor_abaware` + (thr 0.6 / debounce 25 / arm 2.5) is promoted to the deployed default (currently default-off checkpoint; deployed `monitor.pt` byte-identical). Independent of promotion, ratify (a) its use as the A0.2-secondary robustness trigger, (b) the appendix infrastructure note reporting the first A-class FPR measurement (effective ≈100% pre-fix → lane-fire 0.048 post-fix) and the R8 observable-outcome negative-labeling protocol.

## 11. Paper skeleton (for orientation)

Intro (visual illusion vs dynamic truth → attribution, not adaptation, is the missing operation) → Conflict taxonomy + certified benchmark (A1, A0), including the **three-layer ambiguity triangulation (A1.5)** → Method (Kino-tokens, scripted conflict CoT, truth filter) → Results: three rows at scale (A2) → bidirectional battery + override curves (A3) → consequence matrix, cost asymmetry, ERS/Regret (A4, robustness A4.5) → generalization + selective prediction (A5) → **intervention boundary: O10 decision curve vs θ\* + the O2 "when-to-intervene is cross-modal" panel (A6, now plausible main text)** → ablations (A7) → optional real-world snapshots (A8) → Limitations (sim physics approximations: spring-damper adhesion, pendulum payloads; detection as a contribution out of scope — only the corroboration datum and robustness condition are used; no closed-loop system claim). Related-work axis rotates accordingly: REFLECT/AHA become the primary comparison (low-frequency, post-hoc, manipulation-focused, text-injected — vs 1 kHz certified-necessary proprio channel, pre-failure, legged, latent), RMA-line becomes the *characterized opponent* (B1 + E1 bound) rather than a rival system, VLMaps/CBF/monitor literature each get one contrast sentence.