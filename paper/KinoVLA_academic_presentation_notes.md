# KinoVLA academic presentation — speaker notes

**Audience:** two advisors and lab colleagues outside the immediate area  
**Main talk:** Slides 1–17, approximately 18–20 minutes  
**Appendix:** Slides 18–22, for questions  

The speaking plan follows Amaral's receiver-oriented advice: open with a concrete conundrum, explain each visual before interpreting its data, use short signposts, and return to the opening visual at the end.

## Slide 1 — Feel It, See It, Recover (0:45)

Open with the conundrum, not an outline:

> Imagine that a quadruped suddenly slows down. Its joints report more effort and its feet stop advancing. Should it push its feet higher, or should it step backward? The same symptom can require opposite recoveries. Today I will show how we use vision and body sensing together to decide which recovery is safer.

State the scope once: all physical evidence is from Isaac Sim using the Unitree Go2 asset.

## Slide 2 — A symptom is not a diagnosis (1:00)

Explain the common trace first. Then reveal the two causes.

- In soft mud, extra clearance can free the foot, so High-step is appropriate.
- Under adhesion, pushing harder can immobilize the robot, so Backstep is safer.
- In the measured A4 pair, the wrong choice can cost four physical-cost units.

Transition: “The question is therefore not only whether the robot detects a failure, but whether it identifies the cause.”

## Slide 3 — One sensor cannot settle every conflict (1:10)

Walk left to right.

- Left: mud and adhesion were constructed to share the binding body trace. Body sensing alone must guess; vision separates the materials.
- Right: an invisible obstacle has the same safe-looking RGB as a clear path. Vision alone misses it; the body registers the impact.

Define *proprioception* in plain language as joint, torque, contact, and tracking signals.

## Slide 4 — Research question and thesis (0:55)

The four verbs are the paper's logic:

1. resolve which modality is decisive;
2. attribute the physical cause;
3. select a recovery using measured consequences;
4. abstain unless observable cues agree.

Emphasize that attribution is evaluated as a control decision, not as free-form description.

## Slide 5 — Evaluation contract (1:15)

Explain the protocol before the sample counts.

1. Freeze the observation at failure onset.
2. Score every method on that identical snapshot.
3. Hold the scene fixed and force one recovery action.
4. Measure physical outcome on a 0-to-6 scale.

The purpose is to remove navigation-path luck, reset history, and agent-specific closed-loop behavior from the causal comparison.

## Slide 6 — Approach (1:05)

Introduce only the boxes needed for the story.

- RGB is the “See” input.
- A compressed body window, represented by Kino-Tokens, is the “Feel” input.
- The cross-modal VLA proposes both a cause and an action.
- A frozen See–Feel–Act gate releases the proposal only when its observable cues agree; otherwise it uses the A4 Backstep fallback for this decision pair.

Do not discuss implementation details yet; thresholds are available in Slide 22 and the frozen artifacts.

## Slide 7 — C1: complementary blind spots (1:25)

Explain the left experiment before giving its statistics.

- A1 makes the binding proprioceptive input byte-identical for matched mud and adhesion.
- A T=25 logistic probe yields AUC 0.500 and permutation p=0.110: no evidence that the body channel separates the pair.

Then explain the mirror experiment.

- On 48 invisible-obstacle cases, vision-only gets 0 correct while body-only gets 48 correct.

Safe claim: complementary blind spots exist in these controlled ambiguity constructions.

## Slide 8 — C2: conflict resolution is learned (1:25)

First name the y-axis: attribution accuracy on the same 240 held-out matched-conflict snapshots.

- Body-only is at 0.50 by construction.
- Merely adding a vision-capable VLA reaches 0.80.
- Adding matched conflict supervision reaches 0.90.

The improvement is paired: 24 corrected items, zero regressions, exact McNemar p = 1.19 × 10⁻⁷, and 3/3 training seeds reach 0.90.

Interpretation: the gain is evidence weighting learned from conflict examples, not the mere presence of a camera.

## Slide 9 — Structured v2 frozen confirmation (1:35)

Start with the five green cells: T1–T5 are each 1.000 for every one of five training seeds. Then point to the three T3 subcells—looks-safe, invisible obstacle, and reverse/continue—which are also each 1.000.

Next explain the paired comparison block. On the identical confirmation battery, the cluster-balanced improvements over body-only, closed-set fusion, and the VLA conflict baseline are +0.545, +0.864, and +0.818. Their paired bootstrap intervals exclude zero and their exact sign-test p-values are all below 1.5×10⁻⁵.

Finish with the robustness certificate. The 237/237 snapshot rate is descriptive; inference is based on 33 stratified case–appearance clusters, with strict success 33/33 and exact 95% CI [0.894,1.000]. Coarsening all reused physical cases to 15 appearance IDs still gives 15/15 and exact CI [0.782,1.000]. Cause plus canonical admissible action is 237/237 for every seed.

The safe claim is a frozen, closed-procedural-domain confirmation of the structured evidence router—not open-world or physical-robot generalization.

## Slide 10 — C3: physical consequences (1:20)

Read the matrix as a controlled intervention:

- rows are the true physical cause;
- columns are the forced recovery action;
- each cell is measured physical cost, where lower is better.

High-step is clean in mud but costs 4 under adhesion. Backstep costs 1 in both cases. The complete A4 matrix contains 630 Isaac physics episodes: 9 scenes × 7 actions × 10 seeds.

## Slide 11 — Safe default from asymmetric harm (1:10)

Explain axes before lines.

- The horizontal axis is the belief that the surface is adhesive.
- Backstep has constant expected cost 1.
- High-step has expected cost 4p.

The crossover is p* = 0.25. Above that point, Backstep minimizes expected cost for this mud/adhesion decision pair. Do not generalize Backstep to the entire taxonomy.

## Slide 12 — Structured selective recovery (1:20)

The frozen gate checks four observable agreements:

1. RGB lies near calibrated soft-terrain material support;
2. body tracking error confirms a physical anomaly;
3. the attribution is compliant terrain;
4. the proposed action is High-step.

Only all-true cases are released. Deployment cannot read truth, scenario ID, appearance ID, theta, split, or cost.

## Slide 13 — C4: two final evaluations (1:15)

Name the measure first: expected physical cost; lower is better.

- Final-v2: 1.5417 versus always-safe 1.7500, paired delta −0.2083.
- Replication: 1.52 versus always-safe 1.72, paired delta −0.20.
- Both paired 95% confidence intervals lie strictly below zero.
- Released attribution precision is 1.00.

The claim is selective improvement over the safe fallback, not universal action dominance.

## Slide 14 — Post-freeze confirmation (0:55)

Use the timeline to establish independence.

- Observable conditions were selected during development.
- The gate was frozen and identified by its hash.
- Final-v2 used 288 snapshots.
- Confirmation used 300 snapshots and 24 new appearance IDs.

All six released mud appearance clusters improve by exactly one cost unit. The one-sided exact sign test is p = 0.015625.

## Slide 15 — A7: load-bearing choices (1:20)

Treat the three panels separately.

- Route: under the controlled M7 harness, vision-only is 0.458 and latent body input is 1.000. Binned text uses 1.27× as many prompt tokens as latent, while latent also provides a theta head.
- Dose: across three training seeds, dose 10 has the best mean O4 conflict accuracy, 0.733. The curve is non-monotone, so do not claim that more conflict data always helps.
- Grounding: truth filtering raises grounded-correct rationales from 0.604 to 0.765. The best encoder theta MAE is 0.0279.

## Slide 16 — Honest scope (1:05)

Pause on the left, then the right.

Supported: controlled complementary blind spots, learned bidirectional conflict resolution, frozen all-cell structured confirmation, causal consequence asymmetry, and in-support structured selective recovery.

Not claimed: physical-robot deployment, open-world material recognition, unseen-operator semantic naming, a universal scalar abstention score, a learned intervention boundary, or end-to-end VLA all-cell dominance.

This slide prevents the positive C4 result from being mistaken for broad OOD generalization.

## Slide 17 — Takeaways and discussion (0:50)

Return to the opening visual vocabulary.

1. Ambiguity is real: the same symptom can hide opposite mechanics.
2. Conflict resolution is robust: structured v2 passes every battery cell across five training seeds.
3. Recovery must be selective: the frozen observable gate reduces cost on two final datasets.

End with the discussion choice on the slide: nominal abstention, broader calibrated support, or hardware validation.

## Appendix use

- **Slide 18:** answer “Are all A0–A7 experiments represented?”
- **Slide 19:** answer questions about independence and statistical units, including why A3 uses 33 stratified clusters plus a 15-appearance sensitivity analysis rather than 237 independent trials.
- **Slide 20:** give the full paired-comparator audit for structured v2: cluster-balanced deltas, confidence intervals, exact sign tests, and the seed/cluster/appearance robustness certificates.
- **Slide 21:** answer exact acceptance-threshold questions: all five cells plus worst-cell and macro pass for every one of five seeds, with the frozen method hash shown on the slide.
- **Slide 22:** give the exact frozen gate thresholds, fallback action, hash, and forbidden inputs.

## Rehearsal cues

- Rehearse the main talk four times for an internal group meeting, consistent with Amaral's rule of thumb.
- On data slides, say the axis or cell semantics before the headline number.
- Pause after Slides 2, 8, 13, and 16; these are the four conceptual turns.
- Avoid reading slide text verbatim. The slides carry visual anchors; the notes carry the connective reasoning.
