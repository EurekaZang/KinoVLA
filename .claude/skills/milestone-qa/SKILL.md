---
name: milestone-qa
description: Use at the start and end of every session, and before marking any task or milestone complete - session protocol, definition of done, performance budgets, and CLAUDE.md bookkeeping.
---
# Session & QA Protocol
- Start: read CLAUDE.md sections 2/3; work ONLY inside CURRENT MILESTONE.
- Done means: ruff clean, typed public APIs, pytest green, scripts/run_demo.py still passes (walking skeleton = permanent regression), thresholds in configs/ not code.
- End: update CLAUDE.md section 2 (progress state) and append section 4 (completed log: date | Mx | task | evidence). Spec conflicts go to section 6, never silently resolved.
- Budgets: QP <1ms p99, monitor <1ms, extractor <10ms, demo <5min headless.
