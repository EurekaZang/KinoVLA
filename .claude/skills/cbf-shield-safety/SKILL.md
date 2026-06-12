---
name: cbf-shield-safety
description: Use whenever touching kino_vla/shield/ or the Primitive Compiler - CBF-QP math (spec section 6, LIP/DCM, support polygon, mode-switch admission), safety test requirements, and non-negotiable rules.
---
# Safety Shield Rules (spec section 6)
- Barrier: h_j = (b_j - delta) - a_j^T xi, xi = p + v/omega; CBF constraint is LINEAR in ZMP u. QP: 2 vars, O(M) constraints, p99 < 1ms asserted in CI.
- Friction bound ||p - u|| <= mu*z_c uses online mu-hat from the extractor head (coupling point, spec 6.5).
- Trot uses virtual support polygon (gait-cycle foothold hull, spec 6.7); mode switch admitted only if h^sigma' >= eps_switch, else structured rejection code.
- NEVER weaken a safety assertion to make a test pass. Any change here requires re-running the adversarial-command suite (zero falls with shield active) before merge.
- Infeasible QP -> Reflex fallback chain, never soft slack on safety constraints.
