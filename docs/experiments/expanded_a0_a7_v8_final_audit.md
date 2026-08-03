# Expanded realistic Kino-Fail A0--A7 v8 final audit

## Material Passport

- Artifact type: experiment validation report
- Verification status: ANALYZED
- Evaluation date: 2026-08-03
- Scope: realistic expanded Kino-Fail A0--A7 only; A8 excluded
- Primary ledger: `outputs/eval/realistic_a0_a7_v8_expanded_f42/readiness_audit.json`
- Frozen score: `outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f42/confirmatory_report.json`
- Physical action report: `outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f42/a4_actual_action_policy_report_v8.json`
- Operator-boundary report: `outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f42/a6_operator_boundary_report.json`
- Evidence level: one-shot expanded simulation plus matched physical-simulation action interventions; no real-Go2 validation
- Integrity boundary: every unsuccessful operational attempt and every unfavorable scientific outcome is retained; the F38 task-alignment amendment occurred after acquisition but before prediction and is explicitly not counted as fully preregistered confirmation

## Executive verdict

Data collection, frozen prediction, A4 physical-action analysis, A6 boundary
analysis, and the A0--A7 ledger are complete. The expanded experiment is not a
positive confirmation of the current architecture. Only A0 and A1 pass; A2
through A7 fail their preregistered substantive gates. The current package is
therefore not ready to support a strong ICRA claim that the proposed system
solves cross-modal failure attribution on new realistic environments.

The negative result is diagnostically sharp. The simulator and recovery-action
layer support the causal statement that correct attribution matters for O4
adhesion and O5 overload. The failure lies in the frozen attribution and routing
architecture: T2 local visual evidence is poorly captured, while the absolute
proprioception representation shifts strongly in the new physical regime.

## Dataset and evaluation integrity

- 30 new scenes across life, production, and wild domains.
- All 11 operators, two severity strata, three appearance views, and 16 direct
  O9 high-centering parameter points are represented.
- Scale: 10,243/10,560 valid pairs; attrition 3.002%.
- T2: 1,500/1,500 valid cases.
- T3: 1,430/1,500 valid cases; complete-case attrition 4.667%.
- Combined Conflict attrition: 2.333%; overall attrition: 2.854%.
- Five frozen checkpoints and seven methods produced 2,766,330 prediction rows.
- A4 adds 50 frozen cases, 25 unexposed scenes, and 350 matched physical action
  episodes. All cases were accepted once; no unfavorable outcome was removed.
- Every ledger hash and dependency binding passes.

These results establish scale, coverage, traceability, low attrition, and
reproducibility of the evaluation chain. They do not establish model efficacy.

## A0--A7 ledger

| Experiment | Verdict | Evidence boundary |
|---|---:|---|
| A0 | PASS | All hashes match; one-shot score valid; attrition below 5%; 30 scenes and 11 operators; raw failures disclosed. |
| A1 | PASS | Direct O9 contact evidence, all Scale operators, and both conflict directions are present. |
| A2 | FAIL | Learned routing improves Scale by +0.04068 balanced accuracy versus late averaging, but loses Conflict by -0.06179; all ordered cross-battery gates fail. |
| A3 | FAIL | T3 proprio route fidelity passes at 0.9460, but T2 vision route fidelity is only 0.08160. |
| A4 | FAIL | Oracle-correct recovery is physically effective, but attribution accuracy on the action cases is 0.56 rather than at least 0.90, so the end-to-end action gate fails. |
| A5 | FAIL | Selective risk versus late averaging is -0.00152 with 95% CI [-0.01325, 0.01278]; the interval includes zero. The action policy also fails superiority and fall-noninferiority versus Continue. |
| A6 | FAIL | Appearance invariance passes, but the worst operator is O9 at 0.0375 accuracy and the minimum O9 point accuracy is 0.0. |
| A7 | FAIL | Learned routing does not beat every frozen ablation and loses the primary late-averaging comparison on the equal-battery endpoint. |

## What the four paper arguments can currently support

### A systematically neglected navigation-failure family can be constructed and evaluated

Support level: substantial but not yet benchmark-final. The operator coverage,
fresh scenes, direct O9 contact semantics, paired counterfactuals, realistic RTX
views, material randomization, low attrition, and complete provenance are strong.
However, F38 required a post-acquisition/pre-prediction task-alignment amendment,
some T2 cues are too small for the frozen global visual representation, and no
real-Go2 anchor has yet been collected. The paper may present the expanded
dataset as a rigorous simulation benchmark release candidate, not as a fully
validated sim-to-real benchmark.

### Neither vision nor proprioception alone is sufficient for both conflict directions

Support level: supported as a problem construction, not as a solved method claim.
Both causal directions are present, and no fixed unimodal baseline solves the
combined battery. This supports the information-structure argument that the task
requires access to both modalities. It does not imply that the current fusion
architecture uses them correctly.

### Conflict-aware multimodal learning solves the constructed attribution problem

Support level: refuted by the present independent expansion. Learned routing has
Conflict balanced accuracy 0.0367 versus 0.0985 for late averaging and 0.2981 for
fixed vision. It fails T2 route fidelity, worst-battery superiority, equal-battery
superiority, and the ablation family. This statement must not appear as an
established contribution until a new architecture is frozen and succeeds on a
new confirmation set.

### Correct failure attribution enables safer, more effective recovery

Support level: physically supported for the oracle action choice on O4 and O5,
but not supported end to end for the current attribution model. Correct recovery
succeeds at 0.98 with lower bound 0.94 and has substantially lower cost than
measured wrong recoveries. Because the classifier is correct on only 0.56 of
action cases, the deployed selective policy does not conclusively beat Continue.
The defensible result is that correct attribution is causally valuable; the
current system does not yet deliver it reliably.

## Statistical interpretation

- Cross-battery effects use scene/group-clustered intervals and ordered
  intersection-union gates. The negative Conflict effect is both statistically
  and practically adverse.
- Operator action benefits use two one-sided Bonferroni-corrected tests. Both O4
  and O5 benefits pass, with no operator-level harm.
- A6 appearance invariance uses a simultaneous worst-operator bootstrap, not a
  pooled-only average. Its upper bound of 0.01067 is below the 0.05 limit.
- The A4 oracle success interval is narrow enough to support a strong local
  simulation result. The selective-versus-Continue intervals include the null,
  so they cannot be described as superiority or noninferiority.
- Non-significant and adverse findings are retained. No p-value threshold is
  used to rescue a failed ordered gate.

## Statistical fallacy scan

Coverage: 11/11 checked.

| Fallacy | Finding |
|---|---|
| Simpson's paradox | No positive aggregate efficacy claim is made over adverse battery-level results. Scale and Conflict are reported separately; no reversal is hidden. |
| Ecological fallacy | Inference is made at physical pair/case level with scene clustering, not from frame-level counts to scene-level performance. No issue detected. |
| Berkson's paradox | Model-blind eligibility and less than 5% attrition reduce selection risk. Residual caution remains because only physically valid cases enter each battery. |
| Collider bias | No post-outcome variable is used to select A4 cases, and all unfavorable outcomes are retained. No issue detected. |
| Base-rate neglect | Balanced accuracy and equal-battery macro endpoints are primary; A4 has 25 cases per supported operator. No issue detected. |
| Regression to the mean | Not a pre/post study selected on extreme model scores. Not applicable. |
| Survivorship bias | Attrition is below 5%, explicitly reported, and no failed A4 outcome is removed. No issue detected. |
| Look-elsewhere effect | Primary gates are frozen and ordered; operator tests use Bonferroni and ablations use Holm correction. No issue detected. |
| Garden of forking paths | CAUTION: F38 task alignment was amended after acquisition. It was pre-prediction, result-blind, and disclosed, but prevents strict independent-confirmation status. |
| Correlation implies causation | A4 uses matched interventions and measured action arms, so its local simulator action-effect claim is causal. Attribution benchmark accuracy itself is not used as a causal claim. |
| Reverse causality | Operator interventions precede observations and recovery actions. No issue detected. |

## Publication readiness and next mandatory gate

Current ICRA readiness: major experimental revision required. The dataset
engineering and physical recovery evidence are strong components, but the core
method claim fails on the independent expansion. Paper figures and tables must
not mix the earlier favorable development result with F42 as if both were
confirmatory.

The next defensible route is the frozen F43 plan already documented in
`docs/experiments/f42_failure_recovery_plan.md`: develop a local visual branch,
relative temporal proprioception, and conflict-aware routing on development/F42
only; then freeze architecture, features, thresholds, and statistics before
generating entirely new scenes, materials, seeds, and operator ranges. F43 must
be scored once. Real-Go2 deployment remains a separate required external-validity
anchor.
