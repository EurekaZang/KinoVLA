"""Canonical closed-loop failure nodes (CI surrogate + the Isaac deliverable).

These are the Suite-Sem-relevant scenarios whose *outcome depends on the attribution*:

- ``o8_invisible`` — an invisible collider (B): only Update_Topology / Replan (detour) reaches the
  goal; pushing through (Set_Constraint / Switch_Gait) is blocked → dead-loop. A hard wall, so the
  signal holds on the real Go2 regardless of the policy's friction robustness.
- ``o3_collapse`` — thin ice (B): escape + mark + detour reaches; treating it as uniform ice
  (Set_Constraint, continue) dwells on the give-way and slides out.
- ``o1_ice`` — uniform ice (A): the sibling of O3 in the ambiguity pair, so a planner must read the
  proprioception (a slip *step* ⇒ O3) to tell them apart.

Each returns a :class:`~kino_vla.vla.rollout.Scenario` on a straight start→goal path at lane ``y``
(distinct ``y`` per lane lets the Isaac driver run them in one app, spec §8.1 / CLAUDE.md #21a).
"""

from __future__ import annotations

from kino_vla.map.types import SemanticRegion
from kino_vla.sim.operators.o1_mu_field import MuField
from kino_vla.sim.operators.o3_collapse import Collapse
from kino_vla.sim.operators.o8_invisible_collider import InvisibleCollider
from kino_vla.utils.geometry import Rect
from kino_vla.vla.rollout import Scenario

# The demo's known-good ice geometry (configs/demo/walking_skeleton.yaml): the monitor reliably
# fires and an escape+detour reaches the goal, so the outcome cleanly separates strategies.
_HX, _HY = 1.0, 1.0
_HAZARD_CX = 3.0
_GOAL_X = 6.0
_MAX_T = 40.0


def _rect(y: float) -> Rect:
    return Rect(cx=_HAZARD_CX, cy=y, hx=_HX, hy=_HY)


def o8_invisible(y: float = 0.0) -> Scenario:
    rect = _rect(y)
    return Scenario(
        name="O8_invisible_collider",
        operator=InvisibleCollider(region=rect),
        scene_region=SemanticRegion(rect=rect, appearance_class="solid_ground"),
        operator_name="O8_invisible_collider",
        appearance_class="solid_ground",
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
    )


def o3_collapse(y: float = 0.0) -> Scenario:
    rect = _rect(y)
    return Scenario(
        name="O3_collapse",
        # mu_collapsed=0.09 is the band (swept) where escape+detour (less ice exposure) SURVIVES
        # but continuing across (full-width dwell) accumulates a fatal slide on the surrogate; the
        # give-way is the intact→collapsed slip STEP (the proprio discriminator vs O1's flat slip).
        operator=Collapse(region=rect, mu_collapsed=0.09, trigger_dwell_s=0.3, mu_intact=0.8),
        scene_region=SemanticRegion(rect=rect, appearance_class="ice_sheet"),
        operator_name="O3_collapse",
        appearance_class="ice_sheet",
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
    )


def o1_ice(y: float = 0.0) -> Scenario:
    rect = _rect(y)
    return Scenario(
        name="O1_mu_field",
        operator=MuField(region=rect, mu_s=0.10, mu_d=0.08),
        scene_region=SemanticRegion(rect=rect, appearance_class="ice_sheet"),
        operator_name="O1_mu_field",
        appearance_class="ice_sheet",
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
    )


def all_scenarios(y: float = 0.0) -> list[Scenario]:
    return [o8_invisible(y), o3_collapse(y), o1_ice(y)]


def surrogate_scenarios(y: float = 0.0) -> list[Scenario]:
    """The B-class subset the surrogate models faithfully (detour→reach, continue→fail).

    O1 (uniform ice, A-class) is excluded on the surrogate: its correct recovery is *continue
    slowly*, but the surrogate fall model makes any ice crossing fatal regardless of speed, so it
    would wrongly punish the correct attribution. The Isaac closed loop covers O1 on real physics.
    """
    return [o8_invisible(y), o3_collapse(y)]
