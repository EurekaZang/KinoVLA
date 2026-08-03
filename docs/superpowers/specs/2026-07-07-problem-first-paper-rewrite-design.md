# Problem-first full-manuscript rewrite design

Date: 2026-07-07
Target file: `paper/main.tex`
Venue style: ICRA conference manuscript

## Goal

Rewrite the current Paper-A manuscript so it reads like a robotics conference paper rather than a technical evidence report. The new draft should lead with the legged-robot recovery problem, then introduce cross-modal failure attribution as the missing decision, and only then present the protocol and experiments as evidence.

The rewrite preserves the scientific record: no experiment numbers, artifact hashes, scope limits, or supported claims may be strengthened beyond `CLAUDE.md`, `experiments_design.md`, and the `A实验/` reports.

## Diagnosis of the current draft

The current draft is accurate but too report-like. It frequently announces how experiments support claims, uses C1--C5 as visible scaffolding, and lists the experiment battery in a way that resembles a technical report. The most affected regions are:

- Abstract: compresses A0--A7 into one evidence inventory.
- Introduction: motivates the problem well, but paragraph-level flow turns into a claim/experiment roadmap.
- Experiments and Results: organized around A0/A1/A2 and C1--C5 rather than reader-facing robotics questions.
- Tables/Figures: captions use phrases such as claim map and evidence block, making the manuscript feel like a proof ledger.
- Discussion: repeats C1--C5 rather than developing robotics implications and limitations.

## Rewrite strategy

Use a problem-first narrative:

1. A legged robot may know that it is failing before it knows why.
2. The recovery primitive depends on the cause, not just the anomaly signal.
3. Vision and proprioception can agree, conflict, or each be misleading in different regimes.
4. Cross-modal failure attribution is the decision that resolves this disagreement before recovery.
5. Matched operators, frozen snapshots, and forced-label consequence isolate this decision.
6. Experiments answer reader-facing questions: Can one modality suffice? Does multimodal access imply conflict resolution? Do attribution errors physically matter? When should the robot abstain? Which design choices matter?

## Section-level design

### Abstract

Replace the experiment-battery summary with a robotics-paper abstract:

- Opening problem: failure recovery requires knowing why the robot failed, not only detecting that it failed.
- Gap: proprioceptive adaptation and VLA-style semantic planning do not by themselves decide which evidence source to trust after a failure.
- Approach: cross-modal failure attribution, evaluated with matched operators, deterministic frozen snapshots, and forced-label consequence.
- Results: single-modality and unshaped multimodal baselines fail complementary conflict cases; conflict-trained attribution improves the relevant cells; wrong labels change physical outcomes; residual-based abstention reduces cost.
- Scope: recovery is attribution-gated intervention, not a new locomotion controller or full navigation benchmark.

### Introduction

Rewrite around the reader's problem-solving path:

1. Concrete ambiguity: mud versus adhesion can produce the same body signal but require opposite recoveries; visually safe terrain can hide physical hazards.
2. Existing work gap: locomotion adaptation handles continuous compensation, while VLA navigation handles semantic planning, but neither isolates post-failure evidence arbitration.
3. Definition: cross-modal failure attribution maps a visual/proprioceptive failure snapshot to a physical cause and an admissible recovery.
4. Core insight: safe recovery is explanation before intervention; wrong explanations select wrong primitives.
5. Contributions: formulate attribution-gated recovery, introduce matched-operator/frozen-snapshot/consequence evaluation, demonstrate bidirectional modality necessity, show physical consequence and abstention benefits.

Avoid listing A0--A7 or writing that each experiment supports a named claim.

### Related Work

Keep the current topical organization, but smooth transitions so each subsection positions Paper-A by problem gap rather than by claim ownership. The section should emphasize complementarity:

- adaptation is powerful when the response is continuous;
- VLA/navigation systems reason semantically but rarely isolate body-vision conflict at failure time;
- traversability predicts cost but not necessarily cause-specific recovery;
- safety filters are complementary to deciding when and why to intervene.

No new unverified citations should be added.

### Method

Keep the mathematical definitions and registry machinery, but revise the prose to make the method feel like an experimental design for a robotics question:

- Problem formulation: failure snapshot, cause label, recovery primitive.
- Benchmark design: operators create agreement, visual-truth conflict, proprioceptive-truth conflict, fine-structure, and nominal/mild cases.
- Snapshot protocol: identical observations for paired comparisons and deterministic consequence measurement.
- Attributors and baselines: experimental probes of evidence routes, not claim props.
- Forced-label consequence: causal test of what happens when the robot recovers under the wrong explanation.
- Abstention: conservative behavior when evidence is outside the known family or near a boundary.

### Experiments and Results

Reorganize the results around reader-facing questions rather than C1--C5:

1. `Can one modality explain every failure?`
   - O4/O2 byte identity plus vision separation.
   - O7/O8 mirror direction.
   - B1 fairness appears here naturally.
2. `Does multimodal input imply conflict resolution?`
   - B1, B-V, B-F, B-T, B5-unshaped, B5-conflict, B5-conflict-bi.
   - Keep McNemar and seed robustness, but describe them as robustness of the answer, not as claim proof.
3. `Do wrong explanations change physical recovery?`
   - A4 forced-label matrix, crux-cell success/off-label failure, safe-default crossover, regret.
4. `When should the robot abstain?`
   - A5 generalization, OOD residual, risk--coverage, A6 O10 boundary, scoped O2/O8 boundaries.
5. `Which design choices matter?`
   - A7 latent-vs-text, conflict-dose, truth filter, encoder grid.

Revise tables and figure captions to avoid proof-ledger language. For example:

- Rename claim-map table to something like `Experimental questions and measurements`, or remove it if it remains redundant.
- Rename main-results rows from `Evidence block` to `Question` or `Phenomenon`.
- Caption the result-story figure as a paper narrative figure rather than a claim chain.

### Discussion and Limitations

Replace C1--C5 prose with robotics implications:

- Failure recovery should be treated as attribution before intervention.
- Proprioceptive adaptation, VLA reasoning, and safety filters are complementary layers.
- Matched operators are useful because they expose when a modality is non-identifying.
- Consequence matrices are useful because they convert semantic errors into physical cost.
- Abstention is a recovery choice, not just a classifier option.

Limitations must remain honest:

- Isaac physical-Go2 stack, not outdoor hardware deployment.
- Operators are constructed to isolate phenomena and simplify terrain physics.
- Forced-label matrix uses scripted primitives, not optimized end-to-end autonomy.
- Detector is infrastructure; oracle triggering is primary.
- Paper evidence and future hardening remain confined to A0–A7.
- O8 is an active-probing/contact-mode boundary, not fully solved.

### Conclusion

Make the conclusion short and ICRA-like: restate the robotics principle and the empirical support without reciting the battery. End with the title idea only if it reads naturally.

## Non-negotiable constraints

- Do not edit `Kino-vla-v2.md`.
- Do not introduce surrogate or synthetic evidence as claim support.
- Do not add unverified references.
- Do not change reported numbers, hashes, seeds, or artifact scope.
- Do not claim latent Kino-Tokens dominate all text routes on greedy accuracy.
- Do not claim A6 shows a literal policy flip at exactly `theta*`.
- Do not claim O8 is solved.
- Do not present A7 as a new closed-loop benchmark.
- Do not present Paper-A as a general navigation-system benchmark.

## Implementation scope

Primary edit: `paper/main.tex`.

Likely follow-up edits after the manuscript rewrite:

- `paper/README.md` if the paper status or description changes.
- `paper/main.pdf` and `paper/main.bbl` only after a successful LaTeX rebuild.
- `CLAUDE.md` §2/§4/§5 before ending the code session if the manuscript rewrite is part of the session record.

## Verification plan

After editing:

1. Build the manuscript with `pdflatex main`, `bibtex main`, `pdflatex main`, `pdflatex main` from `paper/`.
2. Check for LaTeX errors, unresolved references, and bibliography issues.
3. Review the rewritten sections for overclaiming against the scope guardrails.
4. Confirm that tables and captions no longer present the paper as a claim-support ledger.
5. Update required session records if implementation proceeds.

## Self-review

- Placeholder scan: no TBD/TODO placeholders remain.
- Consistency check: all sections follow the problem-first strategy and retain the same evidence constraints.
- Scope check: the work is a single coherent manuscript rewrite centered on `paper/main.tex`.
- Ambiguity check: the implementation target is explicit; style changes must not alter scientific claims or artifact traceability.
