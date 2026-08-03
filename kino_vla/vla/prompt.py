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
import math
from dataclasses import dataclass, field

from kino_vla.data.oracle import _proprio_summary
from kino_vla.data.schema import CoTAnnotation, Snapshot
from kino_vla.utils.config import Config
from kino_vla.utils.geometry import wrap_angle
from kino_vla.vla.output import TURN_CLAMP_DEG

# The single placeholder where the Kino-Projector splices its soft tokens (latent route). It is
# a literal text marker in the user turn; the model replaces its tokens' embeddings in-place.
KINO_TAG = "<kino_tokens>"

ROUTES = ("text", "latent")
# Proprioception fidelity levels for the text route (the §3 information-fidelity ablation):
#   "binned" — the full per-channel time-series (the oracle-granularity serialization, B4);
#   "scalar" — only the sustained means + tilt peak (the realistic REFLECT-style text summary);
#   "rich"   — derived per-channel statistics (mean/std/min/max/slope) plus traces for A7;
#   "none"   — no proprioception at all (the vision-only floor: how appearance-solvable is it?).
# The latent route (B5) carries the full window as Kino-Tokens regardless of this knob.
PROPRIO_DETAILS = ("binned", "scalar", "rich", "none")
_SCALAR_KEYS = ("slip_mean", "effort_mean", "tracking_mean", "base_height_mean", "tilt_peak")
_TRACE_PREFIXES = ("slip", "effort", "tracking", "base_height")


def _rich_stats_from_summary(summary: dict) -> dict:
    """A7 rich text schema: traces plus derived stats, without privileged θ leakage.

    The frozen prompt path already exposes observable proprio traces via ``_proprio_summary``. A7's
    rich-text arm keeps those traces and adds deterministic reductions that a text-only model could
    reasonably compute from the same window: std/min/max/range/slope over the down-sampled trace.
    It intentionally does NOT include the privileged ``target_theta`` vector.
    """
    out = dict(summary)
    for prefix in _TRACE_PREFIXES:
        key = f"{prefix}_trace"
        vals = summary.get(key)
        if not isinstance(vals, list) or not vals:
            continue
        xs = [float(v) for v in vals]
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        out[f"{prefix}_std"] = round(math.sqrt(var), 3)
        out[f"{prefix}_min"] = round(min(xs), 3)
        out[f"{prefix}_max"] = round(max(xs), 3)
        out[f"{prefix}_range"] = round(max(xs) - min(xs), 3)
        out[f"{prefix}_slope"] = round(xs[-1] - xs[0], 3)
    return out


def _reduce_proprio(summary: dict, detail: str) -> dict | None:
    """Drop or enrich proprio text for the §3/A7 information-fidelity arms.

    ``binned`` keeps the full waveform; ``scalar`` keeps only the sustained means (discarding the
    temporal SHAPE that separates the matched pairs — a slip STEP vs flat-high, a crouch onset);
    ``rich`` adds per-channel statistics to the waveform; ``none`` returns ``None``. This is the
    information-fidelity axis the latent route is tested against — not a hobbled text route but a
    realistic one at each budget.
    """
    if detail == "none":
        return None
    if detail == "binned":
        return summary
    if detail == "scalar":
        return {k: summary[k] for k in _SCALAR_KEYS if k in summary}
    if detail == "rich":
        return _rich_stats_from_summary(summary)
    raise ValueError(f"proprio_detail must be one of {PROPRIO_DETAILS}, got {detail!r}")


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
    proprio_detail: str = "binned"  # how proprio_summary was rendered (label + the §3 fidelity arm)
    map_note: str = ""  # optional semantic-map crop summary (spec §7), "" when no map context
    extras: dict = field(default_factory=dict)


def context_from_snapshot(
    snapshot: Snapshot,
    *,
    route: str = "latent",
    reveal_appearance: bool = False,
    proprio_detail: str = "binned",
    map_note: str = "",
) -> PlannerContext:
    """Build a :class:`PlannerContext` from a dataset snapshot (training / offline eval).

    ``proprio_detail`` (text route only; spec §3 fidelity ablation) selects how much of the 500 ms
    proprioception reaches the prompt: ``binned`` (full waveform, B4), ``scalar`` (means only, the
    realistic text summary), or ``none`` (vision-only floor). The latent route always carries the
    full window as Kino-Tokens.
    """
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
    summary = None
    if route == "text":
        summary = _reduce_proprio(_proprio_summary(snapshot), proprio_detail)
    return PlannerContext(
        monitor_channel=snapshot.monitor_channel,
        pose_xy=(float(snapshot.pose_xy[0]), float(snapshot.pose_xy[1])),
        prior_outputs=list(snapshot.prior_outputs),
        appearance_class=snapshot.appearance_class if reveal_appearance else None,
        proprio_summary=summary,
        proprio_detail=proprio_detail,
        map_note=map_note,
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
        "has raised a candidate kinodynamic alert; the event may be a real failure or a benign "
        "transient. You are given the last few RGB frames from the "
        "body camera, a 500 ms window of your proprioception (motor-current/joint/IMU/contact "
        "features), your previous recovery outputs this episode, and any semantic-map context.\n"
        "Reflect on the PHYSICAL evidence, then choose continue for a benign/nominal event or "
        "exactly ONE recovery primitive for a real failure. Output a single object in this exact "
        "format and nothing else:\n"
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
        "- continue: {} (no semantic intervention; keep the nominal policy in control)\n"
        "- Backstep: {distance_m: number > 0}\n"
        "- Replan_Waypoint: {point_px: [u, v]}  (a 2D PIXEL coordinate in the RGB frame)\n"
        "- Switch_Gait: {mode: high_step | trot | crawl}\n"
        "- Adjust_Posture: {body_height_m: number > 0, pitch_deg: number}\n"
        "- Set_Constraint: {max_speed: number > 0, stiffness: number}\n"
        "- Update_Topology: {region_xy: [x, y], radius_m: number > 0, status: string}\n"
        "- Hold_and_Request: {reason: string}\n\n"
        "Rules: integrate BOTH the visible material and the proprioception. If the body evidence "
        "is nominal or the alert is a benign transient, attribute nominal and choose continue. "
        "When sustained physics "
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
    elif ctx.proprio_summary is None:  # vision-only arm (proprio_detail="none")
        proprio_line = "- proprioception: not provided this round (reason from the RGB frames)\n"
    elif ctx.proprio_detail == "scalar":
        proprio_line = (
            "- proprioceptive summary (500 ms sustained means): "
            f"{json.dumps(ctx.proprio_summary)}\n"
        )
    elif ctx.proprio_detail == "rich":
        proprio_line = (
            "- proprioceptive rich summary (500 ms traces + derived stats): "
            f"{json.dumps(ctx.proprio_summary)}\n"
        )
    else:
        proprio_line = (
            "- proprioceptive window time-series (bins oldest->newest): "
            f"{json.dumps(ctx.proprio_summary)}\n"
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


def nav_system_prompt(cfg: Config, *, has_hazard: bool = False) -> str:  # noqa: ARG001
    """NOMINAL-mode prompt: the VLA is the 1 Hz nav planner. It owns TWO nav actions — drive to a
    clear ground pixel (``Replan_Waypoint``) OR rotate in place to bring clear ground into view
    (``Turn``, an active-perception / information-gathering action). The Turn grammar is now
    UNCONDITIONAL (``has_hazard`` retained only for call-site compatibility): the model is trained
    on this exact prompt with both clean-cruise waypoints AND on-policy route-around Turns, so the
    earlier zero-shot cruise destabilization (which motivated gating Turn behind a hazard) is fixed
    at the data level — training and deployment share one prompt so the VLA transfers (#43)."""
    return (
        "You are the navigation planner on board a Unitree Go2 quadruped driving to a goal. From "
        "the body camera RGB (a forward-down view of the ground ahead), choose the NEXT navigation "
        "action toward the goal. You have TWO actions; output a single object in EXACTLY one of "
        "these two formats and nothing else:\n"
        "<Thought>one sentence of navigation reasoning</Thought>\n"
        '<Action>{"attribution": "nominal", "primitive": "Replan_Waypoint", '
        '"params": {"point_px": [u, v]}}</Action>\n'
        "  (A) drive toward a point on safe, traversable ground. point_px is [u, v] in 0..1000 "
        "normalised image coordinates (u left->right, v top->bottom) of a ground pixel in the goal "
        "direction.\n"
        '<Action>{"attribution": "nominal", "primitive": "Turn", "params": {"yaw_deg": d}}'
        "</Action>\n"
        f"  (B) rotate IN PLACE by d degrees in [-{int(TURN_CLAMP_DEG)}, {int(TURN_CLAMP_DEG)}] "
        "(+left, -right) to bring clear ground into view; you do not move, you only re-aim the "
        "camera, then pick a waypoint next step.\n"
        "DECISION RULE: PREFER a Replan_Waypoint when clear ground that continues toward the goal "
        "is VISIBLE in the forward view. Choose a Turn when the traversable ground toward the goal "
        "is OUTSIDE your current view (e.g. an untraversable hazard patch fills the view straight "
        "ahead, or the goal bearing is beyond the camera's horizontal field) so no safe forward "
        "waypoint exists; turn toward the clear side to reveal the route, then continue. Do NOT "
        "turn when a clear forward waypoint already exists.\n"
        "CRITICAL: if the map context lists an untraversable hazard region (a coloured patch you "
        "already got stuck in), you MUST route AROUND it: NEVER a pixel on the hazard surface nor "
        "one straight through it toward the goal, even though the goal is beyond it; pick a ground "
        "pixel on the CLEAR side, or Turn to bring that clear side into view."
    )


def format_map_note(pos: object, heading: float, regions: list) -> str:
    """The §7 map-crop note: each known hazard region's bearing + world box + the route-around rule.

    SHARED by the deployed planner (``VlaPlanner._map_note``) and the nav-SFT data generator so the
    training and deployment ``map_note`` are byte-identical (the VLA transfers). ``regions`` is a
    list of rects (``.cx/.cy/.hx/.hy``); ``pos`` is (x, y). Empty string when no regions."""
    if not regions:
        return ""
    px, py = float(pos[0]), float(pos[1])
    parts = []
    for r in regions:
        brg = math.degrees(wrap_angle(math.atan2(r.cy - py, r.cx - px) - float(heading)))
        parts.append(
            f"an untraversable hazard region (the coloured patch you got stuck in) at bearing "
            f"{brg:+.0f} deg, world x [{r.cx - r.hx:.1f}, {r.cx + r.hx:.1f}], y "
            f"[{r.cy - r.hy:.1f}, {r.cy + r.hy:.1f}]"
        )
    return (
        "; ".join(parts)
        + ". Head toward the goal but go AROUND that region on clear ground — any waypoint on "
        "it is REJECTED. Do not flee from the goal; keep just enough distance to skirt it."
    )


def nav_user_text(ctx: PlannerContext, goal_bearing_deg: float, *, route: str = "latent") -> str:
    """The NOMINAL-mode user turn: the goal bearing + prior outputs + (latent) Kino-Tokens."""
    side = (
        "straight ahead"
        if abs(goal_bearing_deg) < 8
        else ("ahead-left" if goal_bearing_deg > 0 else "ahead-right")
    )
    proprio_line = (
        f"- proprioceptive Kino-Tokens (500 ms): {KINO_TAG}\n" if route == "latent" else ""
    )
    map_line = f"- MAP — {ctx.map_note}\n" if ctx.map_note else ""
    tail = (
        "Head toward the goal on clear ground, going AROUND the hazard patch you can SEE — a pick "
        "ON the patch (or whose straight path crosses it) is REJECTED and you will be asked again. "
        "Skirt the patch just closely enough to keep progressing toward the goal."
        if ctx.map_note
        else "Pick the next ground waypoint pixel toward the goal."
    )
    return (
        "Navigation step:\n"
        f"- goal direction: {side} (bearing {goal_bearing_deg:.0f} deg)\n"
        "- the RGB frame of the ground ahead is attached\n"
        f"{proprio_line}"
        f"{map_line}"
        f"- previous outputs this episode: {ctx.prior_outputs or 'none'}\n"
        f"{tail}"
    )


def build_nav_messages(
    ctx: PlannerContext,
    cfg: Config,
    goal_bearing_deg: float,
    *,
    route: str = "latent",
    n_images: int = 1,
) -> list[dict]:
    """Chat messages for NOMINAL-mode nav-pick (point 2): the nav system prompt + the goal-bearing
    user turn + image placeholders (the model attaches the real frames)."""
    content: list[dict] = [{"type": "image"} for _ in range(max(0, int(n_images)))]
    content.append({"type": "text", "text": nav_user_text(ctx, goal_bearing_deg, route=route)})
    sys_text = nav_system_prompt(cfg, has_hazard=bool(ctx.map_note))  # Turn only after a hazard
    return [
        {"role": "system", "content": [{"type": "text", "text": sys_text}]},
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
