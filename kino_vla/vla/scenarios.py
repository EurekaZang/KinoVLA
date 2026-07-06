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

import numpy as np

from kino_vla.map.types import SemanticRegion
from kino_vla.sim.operators import (
    Collapse,
    ComplianceField,
    EffortDecay,
    InvisibleCollider,
    MuField,
    ObsBias,
    Payload,
    Push,
    Tether,
    VisualPhysicsRemap,
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


# ----------------------------------------------------------------------------------------------
# E2 Suite-Cal (A-class): the calibration suite where LOW-LEVEL ADAPTATION SUFFICES and no semantic
# attribution is needed. The DR-trained base policy (friction[0.08,1]/mass±1-2kg/push) crosses these
# without help, so B1 (proprio-only) ≈ B5 (agent) — proving B1 is a FAIR opponent, not a strawman.
# A-class θ sit BELOW the A/B thresholds in configs/data/hindsight.yaml (O2 d_sink≤0.12, O5 mass≤5,
# O10 floor≥0.3) and configs/operators/m5_instances.yaml (o2_class_a); all success_mode="reach".
# ----------------------------------------------------------------------------------------------
def o1_ice_A(y: float = 0.0) -> Scenario:
    """Mild ice (A): higher μ than the B-class O1 — slow cruise / Set_Constraint crosses it. (Isaac
    only; the surrogate fall model makes any ice fatal, scenarios docstring above.)"""
    rect = _rect(y)
    return Scenario(
        name="O1_mu_field_A",
        operator=MuField(region=rect, mu_s=0.35, mu_d=0.30),
        scene_region=SemanticRegion(rect=rect, appearance_class="ice_sheet"),
        operator_name="O1_mu_field",
        appearance_class="ice_sheet",
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
    )


def o2_compliance_A(y: float = 0.0) -> Scenario:
    """Shallow firm mud (A): m5_instances.yaml o2_class_a — the base policy crosses without a
    semantic detour."""
    rect = _rect(y)
    op = ComplianceField(rect, k_c=6.0, c_c=3.0, d_sink=0.03)
    return Scenario(
        name="O2_compliance_A",
        operator=op,
        scene_region=op.scene_region(),
        operator_name="O2_compliance",
        appearance_class=op.scene_region().appearance_class,
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
    )


def o5_payload_A(y: float = 0.0) -> Scenario:
    """Light payload 3 kg (A, mass below the 5 kg B-threshold): the policy carries it and proceeds —
    NOT safe_halt (A-class can continue, unlike the 16 kg B-class O5)."""
    rect = _rect(y)
    return Scenario(
        name="O5_payload_A",
        operator=Payload(mass_kg=3.0, com_offset_m=(0.0, 0.0)),
        scene_region=SemanticRegion(rect=rect, appearance_class="solid_ground"),
        operator_name="O5_payload",
        appearance_class="solid_ground",
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
    )


def o10_effort_decay_A(y: float = 0.0) -> Scenario:
    """Mild effort decay to floor 0.4 (A, floor≥0.3 B-threshold): the policy absorbs the derate."""
    rect = _rect(y)
    return Scenario(
        name="O10_effort_decay_A",
        operator=EffortDecay(decay_rate_per_s=0.6, floor=0.4, t_start_s=2.0),
        scene_region=SemanticRegion(rect=rect, appearance_class="solid_ground"),
        operator_name="O10_effort_decay",
        appearance_class="solid_ground",
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
    )


def o6_push_A(y: float = 0.0) -> Scenario:
    """Sub-threshold lateral push (A, impulse below the O6 14–22 N·s B-band): the Reflex/low-level
    policy recovers without a semantic decision (intrinsic-A, hindsight.yaml)."""
    rect = _rect(y)
    return Scenario(
        name="O6_push_A",
        operator=Push(np.array([0.0, 8.0]), t_push_s=2.0),
        scene_region=SemanticRegion(rect=rect, appearance_class="solid_ground"),
        operator_name="O6_push",
        appearance_class="solid_ground",
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
    )


def o11_bias_A(y: float = 0.0) -> Scenario:
    """Mild IMU tilt bias (A, below the O11 0.25–0.60 B-band): low-level tolerant (intrinsic-A)."""
    rect = _rect(y)
    return Scenario(
        name="O11_obs_bias_A",
        operator=ObsBias({"tilt": 0.15}),
        scene_region=SemanticRegion(rect=rect, appearance_class="solid_ground"),
        operator_name="O11_obs_bias",
        appearance_class="solid_ground",
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
    )


def suite_cal_scenarios(y: float = 0.0) -> list[Scenario]:
    """E2 Suite-Cal: six A-class operators where low-level adaptation suffices (parity panel)."""
    return [
        o1_ice_A(y),
        o2_compliance_A(y),
        o5_payload_A(y),
        o10_effort_decay_A(y),
        o6_push_A(y),
        o11_bias_A(y),
    ]


# ----------------------------------------------------------------------------------------------
# E2 Suite-Sem (matched construction): the #49-shaped O4 (proprio-matched to O2) + the matched O2,
# for the closed-loop recovery half. The shaping params come from configs/eval/e2.yaml (calibrated
# by the G1/G2 sweep). Defaults = the E1 attribution-matched preset (force_cap=0, offset=k_c=14 ⇒
# O4 forward grip = O2's constant drag). peel_factor<1 keeps Backstep (reverse) able to escape while
# push-through stalls — the opposite-recovery asymmetry the metric needs.
# ----------------------------------------------------------------------------------------------
def o2_compliance_matched(y: float = 0.0) -> Scenario:
    """The matched O2 (k_c=14, c_c=6, d_sink=0.08 from configs/eval/e1_c2st.yaml)."""
    rect = _rect(y)
    op = ComplianceField(rect, k_c=14.0, c_c=6.0, d_sink=0.08)
    return Scenario(
        name="O2_compliance_matched",
        operator=op,
        scene_region=op.scene_region(),
        operator_name="O2_compliance",
        appearance_class=op.scene_region().appearance_class,
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
    )


def o4_tether_matched(
    y: float = 0.0,
    *,
    k: float = 14.0,
    d: float = 6.0,
    f_break: float = 1.0e9,
    force_cap_n: float = 0.0,
    force_offset_n: float = 14.0,
    peel_factor: float = 0.3,
) -> Scenario:
    """The #49-matched O4 (proprio-matched to O2 at the detection window). The closed-loop trapping
    preset (a slow ramp that compounds over the crossing while staying within-noise over T=25) is
    passed by the E2 G1/G2 sweep; defaults = the E1 attribution-matched constant-drag preset."""
    rect = _rect(y)
    op = Tether(
        rect, k=k, d=d, l0=0.0, f_break=f_break,
        force_cap_n=force_cap_n, force_offset_n=force_offset_n, peel_factor=peel_factor,
    )
    return Scenario(
        name="O4_tether_matched",
        operator=op,
        scene_region=op.scene_region(),
        operator_name="O4_tether",
        appearance_class=op.scene_region().appearance_class,
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
    )


def suite_sem_matched_scenarios(y: float = 0.0, **o4_preset: float) -> list[Scenario]:
    """E2 Suite-Sem closed-loop: the matched O4 (with the calibrated trap preset) + matched O2."""
    return [o4_tether_matched(y, **o4_preset), o2_compliance_matched(y)]


def o4_tether_twophase(
    y: float = 0.0,
    *,
    k_c: float = 14.0,
    c_c: float = 6.0,
    p0_m: float = 0.30,
    k2: float = 30.0,
    f_break: float = 1.0e9,
) -> Scenario:
    """A4.1 — the consequence-grade two-phase (delayed-divergence) O4 adhesion.

    A PLATEAU grip ``k_c`` (the constant drag byte-identical to ``o2_compliance_matched``'s k_c, so
    A1.3 re-certifies C2ST-indistinguishability for pen ≤ p0_m) followed by a linear RAMP
    ``k_c + k2·(pen − p0)`` beyond it. The plateau is where attribution happens (proprioceptively
    identical to mud); the ramp is the consequence region — finite ``f_break`` tears under forward
    lean (a "catapult"), ``f_break=inf`` + high ``k2`` grows without bound ("immobilization"). The
    damping ``c_c`` equals O2's, so the PLATEAU (pen≤p0) is proprioceptively airtight-identical to
    mud; R7 lets A4 place the plateau magnitude where the consequence structure exists.

    The viscous coefficient ``c_c`` equals O2's c_c so the plateau is byte-identical to mud at all
    speeds; ``k``/``l0`` are inert under two-phase (plateau offset + ramp k2 drive grip)."""
    rect = _rect(y)
    op = Tether(
        rect, k=k_c, d=c_c, l0=0.0, f_break=f_break,
        force_cap_n=float("inf"), force_offset_n=k_c, peel_factor=1.0,
        p0_m=p0_m, k2_n_per_m=k2,
    )
    return Scenario(
        name="O4_tether_twophase",
        operator=op,
        scene_region=op.scene_region(),
        operator_name="O4_tether",
        appearance_class=op.scene_region().appearance_class,
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
        success_mode="escape",
    )


# ----------------------------------------------------------------------------------------------
# A3 T3 — visual-physics remap (spec §8.2 Axis III). A benign/deceptive appearance is DECOUPLED from
# the physics material, + a depth_bias corrupts the D channel so RGB-D cannot see through. This is
# the proprio-decisive mirror of the T2 matched pair: vision lies, proprio decides (A1.4: proprio
# C2ST 1.0 / CLIP 0.31 on the same-appearance pair). Two directions for the bidirectional battery:
# ----------------------------------------------------------------------------------------------
def o7_visual_remap(
    y: float = 0.0,
    *,
    mu_s: float = 0.09,
    mu_d: float = 0.09,
    depth_bias_m: float = 0.6,
    appearance_class: str = "solid_ground",
) -> Scenario:
    """O7 looks-safe/is-slippery (T3, A3.1): the patch LOOKS like safe solid ground but is
    physically a low-friction hazard (μ ≈ 0.09 ice), and depth_bias corrupts the depth channel so
    RGB-D cannot see through. Proprio detects the slip (low_friction); vision is fooled. The
    proprio-decisive mirror of T2 — a vision-only agent fails here by construction (A1.4).
    ``mu_s/mu_d`` are exposed for the A3.4 dose–response sweep."""
    rect = _rect(y)
    op = VisualPhysicsRemap(
        rect, mu_s=mu_s, mu_d=mu_d, depth_bias_m=depth_bias_m, appearance_class=appearance_class
    )
    region = op.scene_region()
    return Scenario(
        name="O7_visual_remap",
        operator=op,
        scene_region=region,
        operator_name="O7_visual_remap",
        appearance_class=appearance_class,
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
    )


def o7_visual_remap_reverse(
    y: float = 0.0,
    *,
    appearance_class: str = "hazard_decal_yellow",
) -> Scenario:
    """O7 reverse probe (T3, A3.2): a hazard-COLOURED decal on NOMINAL floor (μ = 0.8). Vision says
    hazard; proprio says nominal ⇒ the correct answer is `continue` (do NOT intervene). An agent
    that learned "conflict ⇒ trust camera" backsteps around paint — the vision-dominance shortcut
    this probe exists to detect. Operator is O7_visual_remap with nominal friction."""
    rect = _rect(y)
    op = VisualPhysicsRemap(
        rect, mu_s=0.8, mu_d=0.8, depth_bias_m=0.0, appearance_class=appearance_class
    )
    region = op.scene_region()
    return Scenario(
        name="O7_visual_remap_reverse",
        operator=op,
        scene_region=region,
        operator_name="O7_visual_remap",
        appearance_class=appearance_class,
        goal_xy=(_GOAL_X, y),
        start_xy=(0.0, y),
        max_time_s=_MAX_T,
    )

