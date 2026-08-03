# Independent Cross-Battery Reconfirmation (v2)

## Purpose

This extension addresses the dependence between architecture development and
the previously reported scale and conflict batteries. The learned router,
experts, routing features, default proprioceptive policy, threshold
\(\eta=0.8\), tie rules, evaluated methods, endpoints, attrition rule, and
statistical tests were frozen before the extension was acquired. No
checkpoint is run on the extension until all 30 model-blind scene shards are
complete.

The extension is reported regardless of whether its preregistered gates pass.

## Prospective separation

- **Architecture freeze (F0):**
  `outputs/freeze/unified_moe_v3_reconfirmation_f0/freeze_manifest.json`,
  SHA-256
  `128c9a5c3feac0bf7696310734782162c8c3080318b03c71076a485c2f0ed4ac`.
- **Transitive source binding:** `outputs/freeze/`
  `unified_moe_v3_reconfirmation_f0_transitive_amendment1/`
  `freeze_manifest.json`, SHA-256
  `a2b11e281fd7a75dac3c2b443a47dc80609d5080a2a73139df6e5b6a5a9c68a3`.
- **Realized registry and schedule seal (F1):**
  `outputs/freeze/unified_moe_v3_reconfirmation_f1/seal_manifest.json`,
  SHA-256
  `f59870031e9f22944e80451c99d5ccb55cb46b3951993ba4929d3587056f6a0f`.
- **Scene registry:**
  `outputs/kinofail_reconfirmation_v2/scene_registry.json`, SHA-256
  `ba42af444425a5998aad05ef828b97f2ac9ac2df63e993d38a8790b530dd9952`.
- **Material lock:**
  `outputs/assets/terrain_pbr_confirmatory_v2/terrain_assets.lock.json`,
  SHA-256
  `b39405e58fa08b0bef7236f59c9568980ad16971a9d83b3a3c1ceb95629a32a3`.
- **Schedule manifest:**
  `outputs/kinofail_reconfirmation_v2/schedules/manifest.json`, SHA-256
  `75228359196273f50771be35d7de4c443f7a8a40e579da1050bea2b7e881d14a`.
- **Freshness audit:**
  `outputs/kinofail_reconfirmation_v2/schedules/freshness_audit.json`,
  SHA-256
  `92a081ed3b27f07cdf9635c66ed464da94c1372ded0ddacab05111c175e93c0d`.

The 30 scenes comprise 9 scenes generated after F0 and 21 previously
generated candidates that had never entered model evaluation or
architecture-selection results. These groups are described as *newly
generated* and *previously unexposed*, respectively. Nine sources touched by
the invalid acquisition pilot were excluded before the v2 registry was
sealed. Scene admission used geometry and simulator QA only and never model
predictions.

## Frozen extension

The design contains 10 life, 10 production, and 10 wild scenes, together with
30 held-out PBR material assets. Scene, material, appearance, physical, and
operator seeds are disjoint from the development and retrospective test
corpora.

- Scale battery: 10,560 planned counterfactual groups, 21,120 episodes, and
  63,360 render-view records.
- Conflict battery: 1,500 T2 cases and 1,500 T3 cases, totaling 3,000 matched
  cases and 18,000 render-view records.
- New scale seed range begins at 2,060,000,000; the conflict range begins at
  2,070,000,000.
- Moderate operator interpolation points:
  0.2025, 0.2475, 0.2925, 0.3375, 0.3825, 0.4275, 0.4725, and 0.5175.
- Hard operator interpolation points:
  0.73625, 0.76875, 0.80125, 0.83375, 0.86625, 0.89875, 0.93125, and
  0.96375.
- Sixteen physical nuisance profiles were fixed in F0.

Every counterfactual pair runs in a fresh Isaac application. Complete
physical groups are retained; partial groups are invalid. The maximum
preregistered attrition is 5% per battery and overall. A failed simulator QA
or infrastructure attempt is not rerun according to its observed physical or
model outcome.

## Frozen inference

All simulator collection, snapshot construction, visual encoding, unified
feature construction, T2 certificate extraction, and conflict-capsule
construction are model blind. Per-scene raw render data are hash-inventoried
and pruned only after the model-blind evidence needed for the exact frozen
features has been copied and verified.

The finalizer waits for 30 completed scene receipts and 30 validated conflict
capsules. It then:

1. merges the model-blind scene shards;
2. constructs the T2/T3 certificate features;
3. assembles a blind feature bundle with a separate truth key;
4. runs the five F0 checkpoints exactly once;
5. seals the prediction hash before joining ground truth;
6. executes the frozen scorer once.

The prediction contract rejects any prediction file containing a truth field
and records that neither fitting nor refitting was called.

## Frozen statistical decision

The five checkpoints are averaged within each physical group and are not
treated as independent experimental units. Uncertainty uses 20,000 paired
scene-by-material crossed-cluster multiplier-bootstrap replicates. The
sequential primary gates compare the learned router with posterior-averaged
late fusion:

1. noninferiority on both batteries: one-sided 97.5% lower bound
   \(> -0.01\);
2. worst-battery superiority: one-sided 97.5% lower bound \(> 0.005\);
3. equal-battery macro superiority: one-sided 97.5% lower bound \(>0\).

The T2 vision-route and T3 proprioception-route fidelity gates each require a
one-sided 97.5% lower bound above 0.85 on preregistered decision-critical
samples. Secondary baseline comparisons use a Holm family-wise correction.
Risk at 75% matched-group coverage and AURC are reported for every method.

The relevant scorer, blind predictor, freshness, schedule, pruning, and F0
contract tests pass 34/34 tests with external pytest plugin autoload disabled.

## Operational incidents and amendments

These amendments change no model, feature, simulator logic, scheduled value,
threshold, or analysis:

- **F2 collector provenance:** corrects the dynamically loaded wrapper's
  `__file__` metadata. The affected attempts occurred before episode creation.
- **F3 nuisance validation:** replaces a narrower legacy range guard with
  exact equality to the already frozen 16-profile F0 table.
- **F4 runtime watchdog:** terminates only a collector whose own PhysX log
  explicitly reports a fatal CUDA-700 simulation stop and that has no pair
  summary. The attempt remains infrastructure attrition and is not retried.
- **F5 snapshot transitive binding:** adds exact hashes for the snapshot entry
  point and its event-alignment implementation. Both files predate F0 and were
  unchanged; no reconfirmation feature, prediction, or score existed when F5
  was sealed. This binding includes the pre-existing narrow O4 physical
  certificate for a known backend/wrapper bookkeeping false negative.
- **F6 liveness watchdog:** adds a fixed 600 s pair deadline and fatal-log
  inactivity handling for native Isaac/PhysX processes that remain alive after
  simulation progress has stopped. A terminated attempt is retained as
  infrastructure attrition and is not retried.
- **F7 stable concurrency:** retains the four deterministic schedule
  partitions while limiting execution to three simultaneous Isaac processes.
  The change followed a model-blind observation that four processes occupied
  approximately 31.2 GiB of the 32.6 GiB GPU and repeatedly stopped making
  progress.
- **F8 ext4 recovery:** copies the first 14,230,237,410-byte scene shard from
  the dirty, read-only NTFS scratch volume to ext4. The 46,926-file
  relative-path, size, and content-hash inventories are identical. Sixteen
  terminal no-summary attempts and the pair deliberately interrupted by F7
  remain attrition. Of four launcher records left in `started` state by the
  host reboot, only the three with no RGB, proprioception, telemetry, or
  episode manifest are authorized for one exact recovery execution. The
  interrupted F7 pair is not retried. Fourteen complete pairs that failed
  frozen operator QA are likewise retained without reexecution.
- **F9 global attrition accounting:** corrects two operational accounting
  errors before any reconfirmation feature existed. Pairs missing both
  manifests are now counted as missing, and the already frozen 5% ceiling is
  enforced per battery and overall at finalization rather than incorrectly
  within every scene. Snapshot selection, event alignment, features, models,
  thresholds, and statistical analysis are unchanged.
- **F17 work-conserving scheduling and scratch relocation:** after scene 03
  was sealed and before scene 04 existed, replaces the three long-lived worker
  queue with a FIFO broker admitting at most the same three fresh Isaac
  processes. The four frozen partitions, within-partition order, fresh process
  per pair, scale-to-T3 barrier, schedules, sensors, physics, features, and
  analysis are unchanged. The empty scene-boundary scratch entry was moved to
  `/data`; all future writes resolve through the original bounded path.
- **F18 pruned F14 audit cache:** restores a 16,279-byte operational audit
  required only by finalizer preflight. Its canonical bytes exactly match the
  SHA-256 retained in the scene-01 prune inventory. F18 redirects only this
  preflight read; the frozen merge, blind prediction, truth join, scorer, and
  statistical decisions remain unchanged.
- **F19/F20 storage-aware postprocess and finalizer reference:** preserve the
  frozen feature and finalization implementations while resolving the
  physical `/data` corpus path after raw scene shards are pruned.
- **F21 atomic T2 collection:** replaces the recurrently hanging long-lived
  50-case T2 Isaac process with one fresh process per frozen case. Each attempt
  is written to a bounded same-filesystem staging directory, validated for its
  observables and 30 image hashes, and admitted with an atomic directory
  rename. A case without an admitted manifest has a fixed 300 s deadline and
  at most three operational attempts. An admitted pass or failure is never
  replayed. Schedules, seeds, simulator, sensors, features, models, thresholds,
  and analysis are unchanged.
- **F22 orphan-wrapper recovery:** records and removes the old slotted wrapper
  that remained asleep after its hanging Isaac child was terminated during
  the F21 handoff. No accepted observation was changed or reexecuted. This
  wrapper is absent from the F21 successor path.
- **F23 successor preflight order:** after scene 05 completed, the first v5
  successor validated F17 after replacing its T2 entrypoint, so the unchanged
  F17 hash guard correctly rejected the new path before scene 06 acquisition.
  F23 validates the exact F17/F19 predecessors first and only then installs the
  already sealed F21 atomic T2 entrypoint. Scene 06 had no pipeline state,
  corpus directory, receipt, or episode at the failure boundary.

The F4 amendment is
`outputs/freeze/unified_moe_v3_reconfirmation_f4_runtime_watchdog_amendment1/`
`amendment_manifest.json`, SHA-256
`c0dec3007b92fc6829c5052c7defdf6235a56ec9cfa940fd6843c2a9d0da9ecd`.

The F5 amendment is
`outputs/freeze/`
`unified_moe_v3_reconfirmation_f5_snapshot_transitive_binding_amendment1/`
`amendment_manifest.json`, SHA-256
`f51d70f3421c841818d992576e330b364df9ef516361d189dc7a313418ee7403`.

The F6 amendment is
`outputs/freeze/`
`unified_moe_v3_reconfirmation_f6_liveness_watchdog_amendment1/`
`amendment_manifest.json`, SHA-256
`976d9c83ba7cf31399a573d51871440f50bd6bdb5499a4ecf461a16a07e42d63`.

The F7 amendment is
`outputs/freeze/`
`unified_moe_v3_reconfirmation_f7_stable_concurrency_amendment1/`
`amendment_manifest.json`, SHA-256
`7f67b73d7ea83192351804065f6ac2799616faad88a7344159b1b60238f9a433`.

The F8 amendment is
`outputs/freeze/`
`unified_moe_v3_reconfirmation_f8_ext4_recovery_amendment1/`
`amendment_manifest.json`, SHA-256
`a73a9ed94969b861b713521b8334f9f39464270ec70eddffb9149974c25301e4`.
Its exact pre-recovery disposition is independently recorded in
`recovery_state.json`, SHA-256
`9f58b6a26e2c97128ee5e7534ad9e71102496de959f356ce09c6a9e5bb1c8b14`.

The F9 amendment is
`outputs/freeze/`
`unified_moe_v3_reconfirmation_f9_global_attrition_amendment1/`
`amendment_manifest.json`, SHA-256
`28f6eba67502f70c7d2f43377bc84b22360e3c532a72f8980623cdc02d573ad3`.

The F17 amendment is
`outputs/freeze/`
`unified_moe_v3_reconfirmation_f17_work_conserving_storage_amendment1/`
`amendment_manifest.json`, SHA-256
`4323597aebcb2b494fa7d1d0d343c897ba75966a008b729287af4eea94eb6a68`.

The F18 amendment is
`outputs/freeze/`
`unified_moe_v3_reconfirmation_f18_pruned_audit_cache_amendment1/`
`amendment_manifest.json`, SHA-256
`8e5f78b891ae93e00699a5be124fea0e9d64439b500912fdb1a6347298094eaa`.

The F21 amendment is
`outputs/freeze/`
`unified_moe_v3_reconfirmation_f21_atomic_t2_amendment1/`
`amendment_manifest.json`, SHA-256
`b2fabf4d8d40c3c691c85c04751b19df3783a54766dbe8c803d58b8b8b0cf624`.

The F22 amendment is
`outputs/freeze/`
`unified_moe_v3_reconfirmation_f22_orphan_wrapper_amendment1/`
`amendment_manifest.json`, SHA-256
`4ba1c106a393650e556385920d8c8828076b11d27bf2eb387f3e48dd806e0a4e`.

The F23 amendment is
`outputs/freeze/`
`unified_moe_v3_reconfirmation_f23_preflight_order_amendment1/`
`amendment_manifest.json`, SHA-256
`2ec8ec82707f5220befce8afe0819d418c5465605bff784d7ee33bc425af7560`.

## Status

The original 30-scene acquisition is complete. It contains 10,560 planned
Scale pairs, 1,500 planned T3 cases (3,000 source pairs), and the frozen T2
battery. No blind prediction has yet been run on the replenished corpus.

F25 completed the one-shot, model-blind Scale replenishment: 1,005 of 1,417
frozen replacement pairs passed, leaving 412/10,560 pair-level attrition
(3.902%). The accepted, no-copy feature overlay is sealed at
`outputs/eval/unified_moe_v3_replenishment_f25/accepted_overlay/`.

The original T3 extraction admitted 1,409/1,500 complete conflict cases.
F27 froze one new O7 and O8 pair for each of the 91 invalid cases. F28 was
terminated after 13 pair IDs were touched because six concurrent RTX
processes caused CUDA oversubscription; none of those artifacts is eligible.
F29 excluded every F28-touched case and completed all 160 newly frozen source
pairs with exactly three Isaac processes and no result-dependent retry. It
admitted 23 complete cases, yielding 1,432/1,500 valid T3 cases and 4.533%
case-level attrition. F30 then built and validated the model-blind union at
`outputs/eval/unified_moe_v3_t3_replenishment_f30/`; its combined-design audit
passes and no prediction or score has been read.

An additional semantic audit found that legacy expanded O9 snapshots were
aligned at first region exposure rather than direct chassis contact, and the
raw contact telemetry had already been pruned. The 24 accepted F25 O9 pairs
were therefore re-audited with the frozen strict definition; only 4/24 pass
all sustained base-force, duty-cycle, and foot-unloading gates. Neither these
24 pairs nor the unproven legacy O9 slots may carry high-centering evidence in
the final analysis.

Three permanently excluded, model-blind mechanism pilots localized the O9
admission problem. F32 and F32b each triggered strict sustained belly/base
contact in 12/12 pairs, while their legacy whole-episode nominal gate accepted
9/12 because the no-obstacle arm continued past the diagnostic instant and
could later encounter unrelated scene geometry. F32c used fresh IDs, seeds,
and appearances and confirmed that all 12 pairs pass when the nominal arm is
audited through the exact anomaly-matched decision horizon (0.42--0.62 s).
Later whole-episode outcomes remain reported metadata but do not determine
eligibility for an earlier attribution snapshot. The aligned audit is sealed
at `outputs/kinofail_confirmatory_o9_pilot_f32c/aligned_final_audit.json`.

F33 was sealed before collection at 2026-08-01 12:23 UTC and is now collecting
a fresh 960-pair, 30-scene O9 confirmation with 16 continuous width/support
points, three Isaac processes, one attempt per pair, and no result-dependent
retry. O9 is realized as a route-transverse pallet crossbar under the Go2
chassis; admission requires the strict direct-contact semantic event and a
stable matched nominal prefix. At least 750 strict pairs are required for the
final Scale corpus to remain below 5% attrition, with additional scene-,
domain-, and parameter-point coverage gates. At the 2026-08-01 13:47 UTC
checkpoint, 171/960 pairs were terminal and all 171 were accepted with no
scientific rejection; three additional pairs were active. The observed
throughput was about 122 pairs/hour. This is a liveness check, not a final
result.

F33 deliberately does not reject a physically valid direct-contact pair for
the two predeclared appearance-only warnings (`appearance_effect_too_small`
and `rgb_spatial_contrast_too_low`). These warnings often reflect a localized
bottom contact-band intervention whose whole-frame RGB L1 is below 0.015. F35
will retain every such pair, verify its terminal-attempt hash, and report the
warning counts and fraction explicitly; no warning-based post hoc selection is
allowed. Appearance performance is instead tested on all retained anomaly
samples by the simultaneous per-operator A6 gate.

Architecture, router features, threshold eta, checkpoints, and statistical
tests remain unchanged. Final A0--A7 numbers have not yet been synchronized:
the old v6 ledger certifies artifact presence on the earlier 990-pair corpus,
but A2, A3, A4, and A5 have failed substantive acceptance fields and must not
be described as strong expanded-corpus evidence. A0/A2/A3/A5/A6/A7 will be
rebuilt only after the F35 Scale overlay and F36 one-shot prediction are
sealed to one corpus/prediction hash. A4 additionally requires a new 30-scene
actual-action consequence study; the earlier five-scene artifact is not a
substitute.

The final fail-closed ledger now directly binds the pre-generation F0 freeze,
the pre-collection F1 seal, and the schedule freshness audit. The latter
certifies zero prior-exposure collisions for scene IDs, source scenes,
geometry hashes, material IDs and content hashes, seeds, and 176 Scale operator
parameter vectors; it covers 10,560 Scale pairs and 3,000 bidirectional
conflict cases. F36 must reproduce all three hashes in its finalization audit.

The expanded A0--A7 successor is now implemented as a fail-closed ledger in
`scripts/assemble_kinofail_reconfirmation_a0_a7_v7.py`. It requires substantive
positive gates and exact F30/F33/F34/F35/F36/A4/A6 hashes; file presence alone
cannot mark an experiment ready. `scripts/analyze_kinofail_reconfirmation_a6_v7.py`
freezes the 11-operator, two-severity, 30-scene, three-appearance-view, and
16-point direct-O9 stratified analysis before F36 prediction. All operator,
severity, appearance-gap, and O9 boundary gates use anomaly samples only;
nominal counterfactual accuracy cannot inflate a failure-attribution cell.
Appearance invariance is not accepted from a pooled average alone: a common
scene bootstrap computes the three-view accuracy range separately for every
operator and takes the worst of all 11 operators inside each replicate; its
one-sided 97.5% upper bound must remain at or below 0.05.

A new A4-v7 design is also prepared for execution after F35 and before the F36
seal. It selects one valid, model-blind case per
30 scenes x 5 actionable operators x 2 severities, then physically executes
seven matched arms per case: Continue, always-safe halt, and every one of the
five recovery labels (2,100 episodes total). The full label-swap matrix is
necessary because a wrong attribution must receive its measured wrong-action
consequence rather than an oracle-correct recovery. The frozen learned-router
majority vote will be compared directly with always-safe under scene-clustered
20,000-draw inference. Collection, policy scoring, and the A0--A7 ledger remain
pending until F33--F36 complete.

Pre-pilot control-law review found that the first draft gave O4 tether release,
O8 invisible-obstacle avoidance, and O9 high-centering nearly identical
backstep-detour parameters. No pilot or confirmatory A4 outcome had been
collected. They are now distinct physical programs: a long gentle tether
release and close skirt, a wide faster rigid-obstacle detour, and a slow
high-step chassis-unloading detour. The excluded pilot must show five distinct
phase programs and five distinct postdecision numeric trajectories in every
case; the full A4 audit and A0--A7 ledger enforce the same condition. The
pre-pilot horizon is 900 control steps (18 s at 50 Hz), long enough for the
posture controller's 0.5 speed cap and detour return; elapsed time remains in
the terminal cost, so the longer censoring horizon does not make slow recovery
free.

Before the excluded five-operator pilot, the A4 replay path received an
additional fail-closed source-observation certificate. Static inspection found
that an early draft had instantiated the legacy anomaly region around route
progress 1.25 m, whereas the actual Scale/F25/F33 collectors use the frozen v4
reachable region around 0.35 m. That draft had not collected any pilot or
confirmatory outcome. The replay collector now reproduces the exact v4 region,
uses the original O8 footprint-exposure event instead of a hard-coded progress
threshold, and reconstructs the same 21-sample x 19-channel raw proprioception
window used by the F35 anomaly-primary sample. Its 80-D invariant summary must
match the exact F35 policy input with predeclared absolute and relative
tolerances of 0.005 in every one of the seven action arms. A mismatch fails the
case and blocks both the A4 seal and the F36 prediction; it cannot trigger an
outcome-dependent recollection. The permanently excluded five-operator pilot
will validate this replay contract before the 2,100-episode matrix is sealed.

The A4 physical endpoint is local and causal rather than whole-route: an arm
succeeds after advancing 0.75 m beyond the frozen decision state and returning
within 0.25 m of the route centerline. It terminates at first success so later,
unrelated scene geometry cannot be misattributed to the indexed failure. The
pre-outcome terminal cost combines fall, failure to clear the hazard, remaining
local progress, postdecision time, and a safety-exposure AUC for sustained
tilt, slip, effort saturation, and foot unloading. The confirmatory policy must
beat both Continue and always-safe halt in scene-clustered cost while remaining
noninferior in fall rate; the oracle-correct recovery is additionally required
to have a one-sided 97.5% success lower bound above 0.75. Correct recovery must
also beat the mean measured cost of the other four wrong-label recovery arms;
the five operator-level versions use Bonferroni alpha 0.01 and at least three
must show benefit with none showing harm. The seven global gates form an
intersection-union decision and all must pass.

The continuation is now protected by a process-level unattended watchdog. It
may restart only the two idempotent orchestration waiters after an interruption;
it is explicitly not authorized to rerun scientific attempts, retry rejected
episodes, or overwrite a partial F36 one-shot prediction. Relative and absolute
script paths are normalized before process matching so a live waiter cannot be
duplicated by a path-spelling difference.
Each downstream subprocess also receives a persistent command, PID-start-time,
log, and expected-artifact record. After an orchestrator restart, it reattaches
to that exact live child or validates the predeclared write-once artifact after
the child exits; it never launches a second F36 prediction or A4 collection.

## 2026-08-03 F42 one-shot result and current interpretation

The expanded one-shot evaluation is complete at
`outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f42/`. It contains
10,243/10,560 valid Scale pairs, 1,500/1,500 valid T2 cases, and 1,430/1,500
valid T3 cases. Scale attrition is 3.002%, T3 complete-case attrition is 4.667%,
combined Conflict attrition is 2.333%, and overall analysis-unit attrition is
2.854%; all are strictly below the frozen 5% ceiling. The five frozen
checkpoints produced 2,766,330 prediction rows. The scoring protocol was valid
and was executed once.

The result is negative: `passed_all_preregistered_gates=false`. Learned routing
beats late averaging on Scale by +0.04068 balanced accuracy, but loses on the
Conflict battery by -0.06179 (97.5% lower bound -0.09129). T3 proprio-route
fidelity passes at 0.9460, while T2 vision-route fidelity fails at 0.08160.
Consequently battery-wise noninferiority, worst-battery superiority, and
equal-battery macro superiority all fail. This result must not be presented as
strong independent confirmation of the current architecture.

The preregistered A6 analysis also fails. All 11 operators, both severity
strata, 30 scenes, three appearance views, and 16 direct O9 parameter points
are present. Appearance randomization is stable: the pooled three-view gap has
a one-sided 97.5% upper bound of 0.001265 and the simultaneous worst-operator
view-gap upper bound is 0.01067. The substantive attribution boundary fails:
the worst operator is O9 high-centering at 0.0375 accuracy, and the minimum
direct-O9 parameter-point accuracy is 0.0.

F38's raw generic QA failure remains preserved. The F38-S2 task-alignment
amendment was made after acquisition but before any prediction and did not use
truth or model outputs. It reduced task-aligned T3 attrition below 5%, but it
does not count as a fully preregistered independent confirmation. F39, F40, and
F41 operational failures are also preserved. F39 stopped before bundle
assembly on a Scale schedule alias; F40 and F41 stopped before checkpoint or
feature loading on legacy dependency-binding checks. F42 accepted only the
already sealed F12 acquisition-backend amendment and the exact F1 binding of
the legacy freshness file. No model, feature, route threshold, or statistic was
changed by these operational recoveries.

The post-hoc diagnosis is recorded at
`outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f42/`
`posthoc_failure_diagnosis.json` and is not confirmatory evidence. Compared
with the six-scene development corpus, F42 T2 O2/O4 paired global-visual
distance falls from 0.538 to 0.369 and ground-ROI HOG distance falls from 3.727
to 2.602. Representative paired-image changed-pixel fraction falls from 0.299
to 0.126 and moves lower in the image. Visual inspection confirms that the O4
tether is frequently a small bottom-edge cue. The frozen HOG-augmented
structured-v5 specialist recovers T2 accuracy to 0.7221, but gives 0.0 on T3.
F42 proprio feature norms are 0.487 times the development norm, whereas the old
formal test is 1.041 times the development norm. This supports two simultaneous
failure mechanisms: undersalient/local T2 visual evidence and non-invariant raw
proprio features under the new physical regime.

A4-v8 physical action collection remained independent of the F42 score and
finished at 2026-08-03 08:00 UTC. Its terminal audit passes: all 50 frozen
cases from 25 unexposed scenes and all 350 physical action episodes are
present, every unfavorable outcome is retained, and no result-dependent retry
or selection occurred. The collection is stored under
`/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_a4_v8/corpus/` and is
bound by
`outputs/kinofail_reconfirmation_a4_v8/run/final_audit.json`.

The one-shot A4-v8 analysis is negative at the end-to-end gate. Oracle-correct
recovery succeeds in 0.98 of the 50 matched cases, with a one-sided 97.5%
scene-clustered lower bound of 0.94. Its terminal cost is 42.71 units lower
than the mean measured wrong-recovery cost (95% CI [-48.66, -36.68]); both O4
adhesion and O5 overload show Bonferroni-corrected benefit and neither shows
harm. This is strong physical evidence that choosing the correct recovery
matters for these two supported operators. The frozen attribution policy,
however, is correct on only 0.56 of the action cases, below the frozen 0.90
floor. Its selective policy beats always-safe halt in cost by 15.99 units
(95% CI [-21.65, -11.27]) but does not beat Continue conclusively: the cost
difference is -3.16 with 95% CI [-14.98, 8.44], and fall noninferiority against
Continue also fails. The report is sealed at
`outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f42/`
`a4_actual_action_policy_report_v8.json` with status
`confirmatory_gate_failed`.

The final expanded A0--A7 v8 ledger is now complete at
`outputs/eval/realistic_a0_a7_v8_expanded_f42/readiness_audit.json`. All
artifact-binding checks pass, but only A0 and A1 pass their substantive gates;
A2--A7 fail. The ledger status is `not_ready`, strict independent confirmation
is false, and strong expanded-simulation confirmation is false. A8 is not in
scope. The authoritative evidence interpretation and statistical fallacy scan
are recorded in `docs/experiments/expanded_a0_a7_v8_final_audit.md`.
