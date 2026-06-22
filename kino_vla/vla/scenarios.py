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
from kino_vla.sim.operators import (
    Collapse,
    ComplianceField,
    EffortDecay,
    InvisibleCollider,
    MuField,
    Payload,
    Tether,
)
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


def o2_compliance(y: float = 0.0) -> Scenario:
    """Mud (B): the correct recovery is Switch_Gait (high-step through). A blind detour wastes the
    run; reading it as adhesion (Backstep) abandons a passable patch — the O2/O4 matched-proprio
    pair, so the planner must read the appearance (brown mud) to pick power-through over escape."""
    rect = _rect(y)
    op = ComplianceField(rect, k_c=15.0, c_c=8.0, d_sink=0.05)
    region = op.scene_region()
    return Scenario(
        name="O2_compliance",
        operator=op,
        scene_region=region,
        operator_name="O2_compliance",
        appearance_class=region.appearance_class,
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
    )


def o4_tether(y: float = 0.0) -> Scenario:
    """Adhesion / glue board (B): the *counter-intuitive* case — the correct recovery is Backstep
    (back off the sticky surface), not push-through. The O2↔O4 sibling: matched tangential
    resistance, separated only by the yellow-adhesive appearance (spec §8.1 P4)."""
    rect = _rect(y)
    op = Tether(rect, k=150.0, d=6.0, l0=0.0, f_break=22.0)
    region = op.scene_region()
    return Scenario(
        name="O4_tether",
        operator=op,
        scene_region=region,
        operator_name="O4_tether",
        appearance_class=region.appearance_class,
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
    )


def o5_payload(y: float = 0.0) -> Scenario:
    """Heavy external overload ~16 kg (B, global): the robot physically cannot proceed, so the
    correct recovery is Hold_and_Request — stop safely. A cause-blind detour topples under the load
    (the decisive dichotomy cell). ``success_mode="safe_halt"`` credits a deliberate non-falling
    stop as success. Global operator (no spatial patch); the surface is plain solid_ground."""
    rect = _rect(y)
    return Scenario(
        name="O5_payload",
        operator=Payload(mass_kg=16.0, com_offset_m=(0.0, 0.0)),
        scene_region=SemanticRegion(rect=rect, appearance_class="solid_ground"),
        operator_name="O5_payload",
        appearance_class="solid_ground",
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
        success_mode="safe_halt",
    )


def o10_effort_decay(y: float = 0.0) -> Scenario:
    """Actuator effort-decay to a severe floor (B, global): the robot's OWN motors weaken, so the
    correct recovery is Switch_Gait (a limp/crawl gait). The O5↔O10 sibling: both crouch, split by
    effort+slip (O10) vs grip+no-effort (O5). Global operator; plain solid_ground."""
    rect = _rect(y)
    return Scenario(
        name="O10_effort_decay",
        operator=EffortDecay(decay_rate_per_s=0.6, floor=0.15, t_start_s=2.0),
        scene_region=SemanticRegion(rect=rect, appearance_class="solid_ground"),
        operator_name="O10_effort_decay",
        appearance_class="solid_ground",
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
    )


def all_scenarios(y: float = 0.0) -> list[Scenario]:
    """The full B-class dichotomy set (Isaac deliverable): the 3 region pairs + the 2 global
    embodiment ops, each on its own lane. O1/O3 (friction), O2/O4 (resistance), O5/O10 (embodiment),
    O8 (unseen-generalization control)."""
    return [
        o1_ice(y),
        o3_collapse(y),
        o8_invisible(y),
        o2_compliance(y),
        o4_tether(y),
        o5_payload(y),
        o10_effort_decay(y),
    ]


def surrogate_scenarios(y: float = 0.0) -> list[Scenario]:
    """The B-class subset the surrogate models faithfully (detour→reach, continue→fail).

    O1 (uniform ice, A-class) is excluded on the surrogate: its correct recovery is *continue
    slowly*, but the surrogate fall model makes any ice crossing fatal regardless of speed, so it
    would wrongly punish the correct attribution. The Isaac closed loop covers O1 on real physics.
    """
    return [o8_invisible(y), o3_collapse(y)]
