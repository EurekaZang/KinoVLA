"""Nominal-nav CoT: geometric truth-consistency filter + canonical target (the §10 filter, for nav).

The on-policy DAgger collector (#43, scripts/collect_nav_dagger.py) visits the exact states where
the deployed VLA freezes (post-Backstep, a hazard filling the forward view, repeated wp_forbidden)
and queries the privileged GEOMETRIC teacher (:func:`kino_vla.vla.nav_teacher.next_nav_label`) for
the correct next nav action — a route-around ``Replan_Waypoint`` pixel or an in-place ``Turn`` to
bring clear ground into view. A strong Oracle (the real :class:`~kino_vla.data.oracle.ApiOracle`,
same path as M6) then answers the SAME deployed nav prompt over the captured RGB and writes its own
``<Thought>`` + ``<Action>``.

This module is the §10-PHASE-3 truth-consistency filter, specialized to navigation: keep a sample
ONLY if the Oracle's action AGREES with the privileged teacher — same kind (turn vs waypoint), and
for a turn the same rotation DIRECTION, for a waypoint the same route-around SIDE. Confabulations
(the Oracle turns the wrong way, or drives a pixel straight through the patch) are dropped. The kept
sample's TARGET is the Oracle's reasoning thought + the teacher's geometrically-exact action, so the
VLA learns *why* it turns (active perception) with a perfectly grounded label. Pure logic (no torch,
no GPU) ⇒ golden-testable on the CI machine.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from kino_vla.vla.output import parse_nav_decision

_THOUGHT_TAG = re.compile(r"<Thought>\s*(.*?)\s*</Thought>", re.DOTALL | re.IGNORECASE)


def extract_thought(text: str, *, default: str = "Route toward the goal on clear ground.") -> str:
    """Pull the Oracle's ``<Thought>…</Thought>`` reasoning out of a raw nav CoT (the Action is
    canonicalized to the teacher's, but the THOUGHT is the Oracle's — and a Turn ParsedDecision
    carries no annotation, so we read the thought from the raw text, not the parse)."""
    m = _THOUGHT_TAG.search(text or "")
    thought = (m.group(1).strip() if m else "").replace("\n", " ").strip()
    return thought or default


# Verdict reason codes (single, golden-testable strings; mirror data/schema.py's recovery codes).
KEEP = "keep"
DROP_SCHEMA = "drop_schema"  # unparseable / non-nav primitive
DROP_KIND = "drop_kind"  # turn vs waypoint disagrees with the teacher
DROP_DIRECTION = "drop_direction"  # turn rotates the WRONG way (the dangerous confabulation)
DROP_PIXEL_SIDE = "drop_pixel_side"  # waypoint routes to the wrong side (toward/through the patch)


@dataclass(frozen=True)
class NavVerdict:
    """One filter judgment (mirrors :class:`kino_vla.data.schema.Verdict`)."""

    keep: bool
    reason: str
    detail: str


def _emitted_kind(decision: object) -> str | None:
    """The nav kind a parsed decision encodes: ``"turn"`` / ``"waypoint"`` / ``None`` (not nav)."""
    if getattr(decision, "nav_turn_deg", None) is not None:
        return "turn"
    if getattr(decision, "primitive_name", None) == "Replan_Waypoint":
        return "waypoint"
    return None


def filter_nav_cot(
    oracle_text: str,
    teacher_label: dict,
    *,
    synonyms: dict[str, str],
    valid_categories: frozenset[str],
    yaw_tol_deg: float = 20.0,
    side_deadband_px: float = 120.0,
) -> tuple[object | None, NavVerdict]:
    """Judge one Oracle nav CoT against the privileged geometric teacher; ``(parsed, verdict)``.

    ``teacher_label`` is a :func:`kino_vla.vla.nav_teacher.next_nav_label` dict — a turn
    (``{"kind": "turn", "yaw_deg": d}``) or a waypoint (``{"kind": "waypoint", "point_px": …}``).
    Checks run in order, first failure wins (so the reason is a single code):

    1. parses into a nav action (turn / Replan_Waypoint) — else ``DROP_SCHEMA``;
    2. same KIND as the teacher — else ``DROP_KIND``;
    3. turn: same rotation DIRECTION when the teacher turn is non-trivial — ``DROP_DIRECTION``;
       waypoint: same route-around SIDE when the teacher hop is off-centre — ``DROP_PIXEL_SIDE``.
    """
    decision = parse_nav_decision(oracle_text, synonyms=synonyms, valid_categories=valid_categories)
    if not getattr(decision, "ok", False):
        return None, NavVerdict(False, DROP_SCHEMA, "unparseable nav decision")
    emit_kind = _emitted_kind(decision)
    if emit_kind is None:
        prim = getattr(decision, "primitive_name", None)
        return decision, NavVerdict(False, DROP_SCHEMA, f"non-nav primitive {prim!r}")
    t_kind = str(teacher_label["kind"])
    if emit_kind != t_kind:
        return decision, NavVerdict(
            False, DROP_KIND, f"emitted {emit_kind!r} but teacher is {t_kind!r}"
        )
    if t_kind == "turn":
        t_yaw = float(teacher_label["yaw_deg"])
        e_yaw = float(decision.nav_turn_deg)
        # Turning the WRONG way is the failure; a near-zero teacher yaw ⇒ direction not critical.
        if abs(t_yaw) > yaw_tol_deg and (e_yaw == 0.0 or (e_yaw > 0.0) != (t_yaw > 0.0)):
            return decision, NavVerdict(
                False, DROP_DIRECTION, f"yaw {e_yaw:+.0f} deg vs teacher {t_yaw:+.0f} deg"
            )
    else:  # waypoint: must route to the SAME horizontal side as the teacher (not through the patch)
        t_px = teacher_label["point_px"]
        e_px = decision.annotation.primitive.params.get("point_px")
        if not (isinstance(e_px, (list, tuple)) and len(e_px) == 2):
            return decision, NavVerdict(False, DROP_SCHEMA, "waypoint missing point_px")
        t_side = float(t_px[0]) - 500.0  # u in 0..1000; <500 left, >500 right of the optical centre
        e_side = float(e_px[0]) - 500.0
        if abs(t_side) > side_deadband_px and (e_side == 0.0 or (e_side > 0.0) != (t_side > 0.0)):
            return decision, NavVerdict(
                False, DROP_PIXEL_SIDE, f"u {e_px[0]} vs teacher u {t_px[0]} (wrong side)"
            )
    return decision, NavVerdict(True, KEEP, "action agrees with the geometric teacher")


def teacher_action_brief(teacher_label: dict) -> str:
    """One-line English description of the teacher's action (for the Oracle to justify)."""
    if str(teacher_label["kind"]) == "turn":
        d = float(teacher_label["yaw_deg"])
        side = "left" if d > 0 else "right"
        return (
            f"TURN in place by yaw_deg={d:.0f} (rotate {side} to bring the clear route into view; "
            "you do NOT translate, only re-aim the camera)"
        )
    px = list(teacher_label["point_px"])
    return (
        f"Replan_Waypoint toward point_px={px} (drive to that clear ground pixel, skirting the "
        "patch toward the goal)"
    )


_TURN_WORDS = (
    "turn",
    "rotate",
    "around",
    "reorient",
    "re-aim",
    "face",
    "pivot",
    "blocked",
    "no clear",
    "sideways",
    "to the side",
    "to the left",
    "to the right",
)
_FWD_CONTRADICTS = ("straight ahead", "continue forward", "directly toward", "go forward")


def thought_coherent(thought: str, kind: str) -> bool:
    """Light truth-consistency check on the Oracle's REASONING vs the privileged action (the action
    itself is ground truth and never filtered): a ``turn`` thought must not claim clear-forward
    without any turn language; reject empty/degenerate. Keeps yield high while dropping reasoning
    that contradicts the action (the nav analogue of the §10 filter, on the rationale)."""
    t = (thought or "").lower().strip()
    if len(t) < 8:
        return False
    if kind == "turn":
        if any(w in t for w in _TURN_WORDS):
            return True
        return not any(c in t for c in _FWD_CONTRADICTS)
    return True  # waypoint rationales are permissive (any skirt/forward reasoning is coherent)


def format_nav_target(thought: str, teacher_label: dict) -> str:
    """The SFT target: the Oracle's reasoning ``<Thought>`` + the teacher's geometrically-exact
    ``<Action>`` (so the action label back-projects to the intended hop, the thought teaches *why*).
    """
    if str(teacher_label["kind"]) == "waypoint":
        action = {
            "attribution": "nominal",
            "primitive": "Replan_Waypoint",
            "params": {"point_px": list(teacher_label["point_px"])},
        }
    else:
        action = {
            "attribution": "nominal",
            "primitive": "Turn",
            "params": {"yaw_deg": float(teacher_label["yaw_deg"])},
        }
    return f"<Thought>{thought.strip()}</Thought>\n<Action>{json.dumps(action)}</Action>"
