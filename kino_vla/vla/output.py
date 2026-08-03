"""VLA structured-output parsing: text → validated atomic §5 primitive (M7 exit criterion 2).

The VLA Recovery Planner emits a ``<Thought>…</Thought><Action>{json}</Action>`` reflection
(spec §10 "严格原子动作 / 2D 像素坐标输出格式"). This module is the single, hard boundary
between the language model and the Primitive Compiler: **every** sampled output is either parsed
into one schema-valid atomic recovery primitive or rejected with a structured code — *no silent
malformed action ever reaches the compiler* (M7 exit criterion 2). The reject codes feed the
next reflection round, exactly like the CBF admission rejections (spec §6.7).

Parsing + validation reuse :class:`kino_vla.data.schema.CoTAnnotation` (the same atomic-action /
2D-pixel schema the M6 Oracle output is held to), so the planner consumes data that already
passed the dataset's validity bar. :func:`to_compiler_primitive` then maps the validated
annotation onto the typed :mod:`kino_vla.shield.primitive_compiler` dataclasses; context-needing
primitives (``Replan_Waypoint`` pixel→odometry, ``Update_Topology`` region) take the planner's
unprojected coordinates when the model does not supply odometry coordinates itself.

Pure-Python/numpy (no torch): the parser is exercised on the CI machine and inside the closed
loop alike.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from kino_vla.data.schema import CoTAnnotation, CoTParseError, RecoveryPrimitive
from kino_vla.shield.primitive_compiler import (
    AdjustPosture,
    Backstep,
    Continue,
    HoldAndRequest,
    Primitive,
    ReplanWaypoint,
    SetConstraint,
    SwitchGait,
    UpdateTopology,
)

# Reject-code prefixes (a fixed vocabulary, mirroring the §6.7 structured rejection codes). A
# SCHEMA reject is "the model said something malformed/non-atomic"; it is the thing M7 exit
# criterion 2 counts (100% of outputs parse OR carry one of these).
REJECT_SCHEMA = "REJECT_SCHEMA"

# Max single in-place Turn magnitude (deg). Raised 90→120 (#44 round-2 fix): on a large patch the
# clear route can lie >90° to the side, so a 90° cap forced the model to spin in 90° steps without
# ever facing clear ground to commit a forward skirt-waypoint. A turn is in-place (no translation),
# so a larger rotation is safe; the low-level policy executes the commanded yaw over time.
TURN_CLAMP_DEG = 120.0


@dataclass(frozen=True)
class ParsedDecision:
    """One parsed VLA reflection — a kept atomic decision, or a structured rejection.

    ``ok`` is True iff the raw text parsed into a schema-valid atomic annotation (then
    ``annotation`` is set); otherwise ``reject_code`` carries the reason for the next round.
    Never raises: malformed model output is *data*, turned into a reject, not an exception.
    """

    ok: bool
    raw_text: str
    annotation: CoTAnnotation | None = None
    reject_code: str = ""
    nav_turn_deg: float | None = None  # set iff a NOMINAL-nav Turn (yaw_deg in [-120,120]); §5-free

    @property
    def attribution(self) -> str | None:
        return self.annotation.attribution if self.annotation is not None else None

    @property
    def primitive_name(self) -> str | None:
        return self.annotation.primitive.name if self.annotation is not None else None

    @property
    def thought(self) -> str:
        return self.annotation.thought if self.annotation is not None else ""


def parse_vla_decision(
    text: str,
    *,
    synonyms: dict[str, str],
    valid_categories: frozenset[str],
) -> ParsedDecision:
    """Parse one raw VLA output into a :class:`ParsedDecision` (never raises).

    Accepts the ``<Thought>…</Thought><Action>{json}</Action>`` form (the M7 target format) or a
    bare/fenced JSON object — both routed through the M6 atomic-action schema. Any malformed,
    non-atomic, or off-vocabulary output becomes ``ok=False`` with a ``REJECT_SCHEMA`` code.
    """
    try:
        annotation = CoTAnnotation.from_oracle_text(
            text, synonyms=synonyms, valid_categories=valid_categories
        )
    except CoTParseError as err:
        return ParsedDecision(ok=False, raw_text=text, reject_code=f"{REJECT_SCHEMA}: {err}")
    return ParsedDecision(ok=True, raw_text=text, annotation=annotation)


_NAV_ACTION_TAG = re.compile(r"<Action>\s*(.*?)\s*</Action>", re.DOTALL | re.IGNORECASE)


def _nav_action_json(text: str) -> dict | None:
    """Extract the <Action>{json} object (or a bare JSON object) from a nav output; None if none."""
    m = _NAV_ACTION_TAG.search(text)
    for blob in ([m.group(1)] if m else []) + [text.strip()]:
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_nav_decision(
    text: str, *, synonyms: dict[str, str], valid_categories: frozenset[str]
) -> ParsedDecision:
    """Parse a NOMINAL-mode nav output: a ``Replan_Waypoint`` (2D pixel) OR a ``Turn`` (``yaw_deg``
    in [-120, 120], relative to the current heading). ``Turn`` is a nav-only primitive — NOT a §5
    recovery primitive — so it is recognised here directly (the VLA may choose to rotate at ANY nav
    tick, the user directive); everything else goes to the §5 :func:`parse_vla_decision`."""
    obj = _nav_action_json(text)
    if obj is not None and str(obj.get("primitive", "")).strip().lower() == "turn":
        params = obj.get("params") or {}
        raw = params.get("yaw_deg", obj.get("yaw_deg"))
        try:
            yaw = max(-TURN_CLAMP_DEG, min(TURN_CLAMP_DEG, float(raw)))
        except (TypeError, ValueError):
            yaw = None
        if yaw is not None:  # Turn is §5-free ⇒ carried on nav_turn_deg, not a RecoveryPrimitive
            return ParsedDecision(ok=True, raw_text=text, nav_turn_deg=yaw)
    return parse_vla_decision(text, synonyms=synonyms, valid_categories=valid_categories)


def to_compiler_primitive(
    prim: RecoveryPrimitive,
    *,
    point_xy: tuple[float, float] | None = None,
    region_xy: tuple[float, float] | None = None,
    region_radius_m: float | None = None,
) -> Primitive:
    """Map a schema-valid :class:`RecoveryPrimitive` onto a typed compiler primitive (spec §5).

    The two spatial primitives carry a 2D *pixel* coordinate in the model's frame; the planner
    depth-unprojects it to an odometry-frame point and passes it as ``point_xy`` / ``region_xy``
    (spec §7). When the model already emits odometry coordinates (``region_xy`` in params, as the
    canonical recovery does) and the planner supplies none, those are used; otherwise a missing
    coordinate raises ``CoTParseError`` (a context the caller must provide — never a silent zero).
    """
    p = prim.params
    name = prim.name
    if name == "continue":
        return Continue()
    if name == "Backstep":
        return Backstep(distance_m=float(p["distance_m"]))
    if name == "Switch_Gait":
        return SwitchGait(mode=str(p["mode"]))
    if name == "Adjust_Posture":
        return AdjustPosture(
            body_height_m=float(p["body_height_m"]), pitch_deg=float(p.get("pitch_deg", 0.0))
        )
    if name == "Set_Constraint":
        return SetConstraint(
            max_speed=float(p["max_speed"]), stiffness=float(p.get("stiffness", 0.5))
        )
    if name == "Hold_and_Request":
        return HoldAndRequest(reason=str(p["reason"]))
    if name == "Replan_Waypoint":
        xy = point_xy if point_xy is not None else read_params_xy(p, "point_xy")
        if xy is None:
            raise CoTParseError("Replan_Waypoint needs an unprojected odometry point_xy")
        return ReplanWaypoint(point_xy=(float(xy[0]), float(xy[1])))
    if name == "Update_Topology":
        xy = region_xy if region_xy is not None else read_params_xy(p, "region_xy")
        if xy is None:
            raise CoTParseError("Update_Topology needs an odometry region_xy")
        radius = region_radius_m if region_radius_m is not None else p.get("radius_m")
        if radius is None:
            raise CoTParseError("Update_Topology needs a radius_m")
        return UpdateTopology(
            region_xy=(float(xy[0]), float(xy[1])),
            radius_m=float(radius),
            status=str(p.get("status", "untraversable")),
        )
    raise CoTParseError(f"no compiler mapping for primitive {name!r}")


def read_params_xy(params: dict, key: str) -> tuple[float, float] | None:
    """Read a 2D ``[x, y]`` coordinate from primitive params, or None if absent/malformed."""
    xy = params.get(key)
    if isinstance(xy, (list, tuple)) and len(xy) == 2:
        try:
            return float(xy[0]), float(xy[1])
        except (TypeError, ValueError):
            return None
    return None
