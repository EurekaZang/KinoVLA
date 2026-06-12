---
name: kino-fail-operators
description: Use when implementing, modifying, or testing any Kino-Fail failure operator (O1-O11) or benchmark suite (Cal/Sem/Comp/Bound/OOD). Enforces spec section 8 design principles P1-P4 and the mandatory per-operator test triplet.
---
# Operator Development Rules
- Every operator: continuous parameter vector theta exposed via get_privileged_state(); theta IS the distillation target.
- Mandatory test triplet before marking done: (1) theta-application test (set param -> measure in sim -> assert tolerance), (2) determinism test (same seed -> same trajectory hash), (3) composability smoke test with one other operator.
- Ambiguity pairs (O4<->O2, O3<->O1, O5<->O10) require the proprioceptive-statistics matching script to pass (spec P4) - this script is a paper artifact, never delete it.
- A/B class labels and boundary-sweep ranges live in configs/, not in code.
