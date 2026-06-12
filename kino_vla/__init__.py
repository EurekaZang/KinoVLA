"""Kino-VLA: closed-loop embodied reflection for a Unitree Go2 quadruped in Isaac Lab.

Package layout (one subpackage per architecture layer, spec section 1):

- ``sim``     — Isaac Lab environments, Go2 bring-up, Kino-Fail operators (spec section 8)
- ``monitor`` — 1kHz Kino-Monitor + Reflex loop (spec sections 2.5, 6.9)
- ``shield``  — CBF-QP Safety Shield + Primitive Compiler (spec section 6)
- ``tokens``  — Kino-Tokens extractor, privileged distillation (spec sections 4, 5)
- ``map``     — semantic traversability map (spec section 7)
- ``vla``     — VLA Recovery Planner, Kino-SFT / Embodied DPO (spec sections 10, 11)
- ``data``    — Hindsight CoT pipeline + truth-consistency filter (spec section 10)
- ``eval``    — benchmark suites, baselines B1-B5, metrics (spec section 12)
"""

__version__ = "0.0.1"
