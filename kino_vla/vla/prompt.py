"""VLA Recovery Planner prompt + target formatting (spec §10 PHASE 4 / §11 Kino-SFT).

The deployed planner (spec §1 layer 03) reasons from its OWN sensors — the last RGB frames, a
500 ms proprioceptive window, the prior recovery outputs, and the semantic-map crop — and emits
``<Thought>…</Thought><Action>{json}</Action>`` with one atomic §5 primitive. This differs from
the M6 Oracle (a god's-eye *labeler*): the planner is the model under training, framed as the
robot. Two input routes (the spec §3 A/B comparison, the M7 ablation arm):

- **text route (B4)** — the proprioception is given as a per-channel time-series in the prompt
  text (identical conditioning to what the M6 Oracle saw, :func:`oracle._proprio_summary`);
- **latent route (B5)** — the proprioception is injected as continuous Kino-Tokens at the
  ``<kino_tokens>`` placeholder (spec §3 route B / §4), and the numeric summary is omitted.

All prompt text is English (project directive; QA 5.2 paper artifact). Pure-Python: no torch and
no processor here — :mod:`kino_vla.vla.model` runs ``apply_chat_template`` and splices the images
and the projected Kino-Tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from kino_vla.data.oracle import _proprio_summary
from kino_vla.data.schema import CoTAnnotation, Snapshot
from kino_vla.utils.config import Config

# The single placeholder where the Kino-Projector splices its soft tokens (latent route). It is
# a literal text marker in the user turn; the model replaces its tokens' embeddings in-place.
KINO_TAG = "<kino_tokens>"

ROUTES = ("text", "latent")


@dataclass(frozen=True)
class PlannerContext:
    """Everything the planner prompt needs about one failure instant (route-agnostic).

    Buildable from a dataset :class:`Snapshot` (training) or from the live obs stream (closed
    loop); :func:`build_messages` renders it for either route.
    """

    monitor_channel: str
    pose_xy: tuple[float, float]
    prior_outputs: list[str]
    appearance_class: str | None = None  # set iff the surface name is revealed (text-only ablation)
    proprio_summary: dict | None = None  # text route: the time-series dict; latent route: None
    map_note: str = ""  # optional semantic-map crop summary (spec §7), "" when no map context
    extras: dict = field(default_factory=dict)


def context_from_snapshot(
    snapshot: Snapshot,
    *,
    route: str = "latent",
    reveal_appearance: bool = False,
) -> PlannerContext:
    """Build a :class:`PlannerContext` from a dataset snapshot (training / offline eval)."""
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
    return PlannerContext(
        monitor_channel=snapshot.monitor_channel,
        pose_xy=(float(snapshot.pose_xy[0]), float(snapshot.pose_xy[1])),
        prior_outputs=list(snapshot.prior_outputs),
        appearance_class=snapshot.appearance_class if reveal_appearance else None,
        proprio_summary=_proprio_summary(snapshot) if route == "text" else None,
    )


def system_prompt(cfg: Config) -> str:
    """The planner system prompt (English; §5 primitive library + category vocab + safety rule).

    Encodes the spec §10 PHASE-4 lessons: integrate vision with proprioception, trust sustained
    physics over deceptive appearance, escape a sudden trap before detouring, and review the
    prior outputs to break a dead loop. Vocabulary is pulled from config (no magic lists in code).
    """
    p = cfg.oracle.prompt
    primitives = ", ".join(p.primitive_library)
    categories = ", ".join(p.category_vocabulary)
    return (
        "You are the recovery planner on board a Unitree Go2 quadruped. A high-frequency monitor "
        "has just flagged a kinodynamic anomaly. You are given the last few RGB frames from the "
        "body camera, a 500 ms window of your proprioception (motor-current/joint/IMU/contact "
        "features), your previous recovery outputs this episode, and any semantic-map context.\n"
        "Reflect on the PHYSICAL cause of the anomaly, then choose exactly ONE recovery "
        "primitive. Output a single object in this exact format and nothing else:\n"
        "<Thought>one or two sentences of physical reasoning</Thought>\n"
        '<Action>{"attribution": <category>, "primitive": <name>, "params": <object>}</Action>\n\n'
        f"'attribution' is exactly one of: {categories}.\n"
        "Category meanings:\n"
        "- low_friction: a slippery surface (ice/oil); the feet slide, sustained high slip.\n"
        "- compliant_terrain: soft deformable ground (mud) the feet sink into; resists motion.\n"
        "- region_collapse: the ground gives way (thin ice breaking); a sudden support loss "
        "(slip steps up partway through the window); trunk stays at normal height, motors idle.\n"
        "- adhesion: a sticky/elastic surface (glue board, tether) that grips or pulls the body "
        "back; a brief resistance then it eases.\n"
        "- overload: heavy EXTERNAL weight; the trunk crouches (a height sag) and the motion bogs "
        "but the feet keep grip (low slip) and effort stays ~0.\n"
        "- effort_decay: the robot's OWN actuators weaken (overheating); the trunk sags, the feet "
        "slip, and the effort trace spikes toward its reduced cap.\n"
        "- external_push: a sudden external impulse/shove.\n"
        "- invisible_obstacle: an unseen rigid barrier; speed deficit with no slip, no effort, "
        "trunk at normal height.\n"
        "- high_centering: the body is beached on a ridge with the feet partly unloaded.\n"
        "- obs_bias: a sensor/observation bias, not a real terrain interaction.\n"
        "- nominal: no physical failure.\n"
        f"'primitive' is exactly one of: {primitives} (ONE atomic action per round).\n"
        "Primitive parameters:\n"
        "- Backstep: {distance_m: number > 0}\n"
        "- Replan_Waypoint: {point_px: [u, v]}  (a 2D PIXEL coordinate in the RGB frame)\n"
        "- Switch_Gait: {mode: high_step | trot | crawl}\n"
        "- Adjust_Posture: {body_height_m: number > 0, pitch_deg: number}\n"
        "- Set_Constraint: {max_speed: number > 0, stiffness: number}\n"
        "- Update_Topology: {region_xy: [x, y], radius_m: number > 0, status: string}\n"
        "- Hold_and_Request: {reason: string}\n\n"
        "Rules: integrate BOTH the visible material and the proprioception; when sustained physics "
        "contradicts appearance, trust the physics, but when the physics is ambiguous (a velocity "
        "deficit with low slip could be mud or an adhesive board) use the visible material to "
        "decide. Safety iron-rule: on a sudden trap (adhesion/entanglement or a collapsing "
        "surface) escape first (Backstep) and mark the region (Update_Topology); do NOT replan a "
        "detour (Replan_Waypoint) in the same round. Review your previous outputs to avoid "
        "repeating a failed action (break the dead loop)."
    )


def user_text(ctx: PlannerContext, *, route: str = "latent") -> str:
    """The user turn's text block (route-dependent proprioception representation)."""
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
    if route == "latent":
        proprio_line = f"- proprioceptive Kino-Tokens (500 ms): {KINO_TAG}\n"
    else:
        summary = ctx.proprio_summary if ctx.proprio_summary is not None else {}
        proprio_line = (
            f"- proprioceptive window time-series (bins oldest->newest): {json.dumps(summary)}\n"
        )
    appearance_line = (
        f"- visible surface appearance ahead: {ctx.appearance_class}\n"
        if ctx.appearance_class is not None
        else "- the RGB frames of the surface ahead are attached (most recent last); read it\n"
    )
    map_line = f"- semantic-map context: {ctx.map_note}\n" if ctx.map_note else ""
    return (
        "Failure snapshot:\n"
        f"- monitor channel that fired: {ctx.monitor_channel}\n"
        f"{appearance_line}"
        f"{proprio_line}"
        f"{map_line}"
        f"- robot pose (odometry x, y): [{ctx.pose_xy[0]:.2f}, {ctx.pose_xy[1]:.2f}]\n"
        f"- previous recovery outputs this episode: {ctx.prior_outputs or 'none'}\n"
        "Attribute the physical cause and choose one recovery primitive."
    )


def build_messages(
    ctx: PlannerContext,
    cfg: Config,
    *,
    route: str = "latent",
    n_images: int = 1,
) -> list[dict]:
    """Build the chat messages (content-list form for a multimodal processor).

    The user turn carries ``n_images`` image placeholders (the model attaches the real frames)
    followed by the route-dependent text. ``n_images=0`` yields a text-only prompt (the
    no-vision ablation / the torch-free stub path).
    """
    content: list[dict] = [{"type": "image"} for _ in range(max(0, int(n_images)))]
    content.append({"type": "text", "text": user_text(ctx, route=route)})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt(cfg)}]},
        {"role": "user", "content": content},
    ]


def format_target(annotation: CoTAnnotation) -> str:
    """Render the SFT target completion: ``<Thought>…</Thought><Action>{json}</Action>``.

    The ``<Action>`` JSON is the flat ``{attribution, primitive, params}`` form the M7 parser
    (:func:`kino_vla.vla.output.parse_vla_decision`) and the M6 schema both accept.
    """
    action = {
        "attribution": annotation.attribution_raw or annotation.attribution,
        "primitive": annotation.primitive.name,
        "params": dict(annotation.primitive.params),
    }
    return f"<Thought>{annotation.thought}</Thought>\n<Action>{json.dumps(action)}</Action>"
