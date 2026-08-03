# F25 Scale attrition replenishment

Updated: 2026-08-01

## Purpose and reporting boundary

F25 is a post-hoc, supplementary replenishment cohort for the 1,417 Scale
counterfactual pairs declared attrited by the model-blind F13 launcher ledger.
It must not be described as part of the original independent confirmation and
must never overwrite that cohort.  The paper should report both the original
attrition and the replenished analysis population.

The fixed target is at least 890 successful one-to-one replacements.  This
leaves at most 527 of 10,560 Scale slots unresolved (4.9905%) and at most 641
of 13,560 Scale+T3 slots unresolved (4.7271%).  All 1,417 replacements were
fixed before F25 acquisition; collection does not stop when the target is
crossed.

## Root cause and correction

The original Scale attrition was concentrated in O4: 928 of the 1,417 failed
pairs, and 928 of 960 planned O4 pairs.  The physical backend had recorded
attachment cycles, but the generic reachable-exposure QA required a Python
operator object.  O4 is backend-native and intentionally returns `None`, so
valid attachment telemetry was rejected as `operator_local_qa_failed`.

F25 changes only the O4 QA wiring: exposure is present when
`total_attachment_cycles > 0`.  Simulation geometry, forces, peel mechanics,
severity, sensing, features, model, thresholds, and statistical analysis are
unchanged.  Interactive and non-interactive smoke cohorts both passed with
complete nominal/anomaly pairs.

## Frozen formal cohort

- Cohort: `F25-v1`
- Planned replacement pairs: 1,417 (2,834 physical episodes)
- Scenes: 30
- Fresh seed interval: 2,180,000,000--2,180,001,416
- Freeze manifest: `outputs/kinofail_replenishment_f25/freeze_manifest.json`
- Freeze-manifest SHA-256:
  `9881ff11281ec9339c6118897b17a2105c8488eb0e3bb7655ddc0e5b88216423`
- Physical corpus:
  `/data/eureka/KinoVLA/outputs/kinofail_replenishment_f25/corpus`
- Live supervisor state:
  `/data/eureka/KinoVLA/outputs/kinofail_replenishment_f25/supervisor_state.json`
- Completion-audit state:
  `/data/eureka/KinoVLA/outputs/kinofail_replenishment_f25/completion_audit_state.json`
- Strict final audit (created only after all 1,417 pairs become terminal):
  `outputs/kinofail_replenishment_f25/final_audit.json`

Each original attrited pair maps to exactly one new pair ID.  Scene, operator,
severity, physical parameters, and nuisance-profile stratum are preserved.
The replacement uses a fresh physics/operator seed and fresh visual-only
surface-state, UV, brightness, normal, and roughness realization.  No model
prediction or score was available at freeze time.

## Operational incidents

An earlier F24 launch under `nohup` inherited `TERM=dumb`.  IsaacLab's shell
exited at its `tabs` formatting command before Python or Isaac started, but the
worker recorded all IDs as return code 1.  F24 contains zero episode summaries
and is retained as an operational incident.  No F24 ID is reused.  F25 uses
new IDs, a disjoint seed range, and a worker that explicitly sets
`TERM=xterm-256color`; a non-interactive smoke cohort passed before F25 freeze.

During F25, seven O4 launches between 17:58 and 18:09 UTC returned code 1
because the NVIDIA S3 endpoint temporarily did not provide the Go2 USD.  The
same remote asset subsequently recovered without a protocol change.  The seven
slots remain rejected in the append-only ledger and were not retried.

## Final state

All 1,417 frozen replacements became terminal on 2026-07-31.  The three
persistent workers exited with return code 0 and zero worker restarts.  The
strict audit accepted 1,005 replacements and rejected 412; no pair was retried,
no frozen slot was left unattempted, and the audit error list is empty.

The accepted replacements reduce unresolved Scale attrition from 1,417 to 412
of 10,560 (3.9015%).  Together with the unchanged T3 attrition of 114 of 3,000
(3.8000%), unresolved Overall attrition is 526 of 13,560 (3.8791%).  All three
rates are strictly below 5%.  The strict audit also confirms that no model
prediction, feature, label, or score was read during replenishment selection or
auditing.

Authoritative artifacts:

- Final audit: `outputs/kinofail_replenishment_f25/final_audit.json`
- Independent audit rerun: `/tmp/f25_final_audit_recheck.json`
- Supervisor state:
  `/data/eureka/KinoVLA/outputs/kinofail_replenishment_f25/supervisor_state.json`
- Completion watcher state:
  `/data/eureka/KinoVLA/outputs/kinofail_replenishment_f25/completion_audit_state.json`
