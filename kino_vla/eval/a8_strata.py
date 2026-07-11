"""A8b pre-registered conflict strata (evidence structure, not quadruped ops)."""

from __future__ import annotations

from typing import Any, Mapping

Stratum = str  # E1 | E2 | E3 | E4 | unassigned

# Failure modes that are typically vision-semantic (wrong target / place / object).
VISION_TRUE_MODES = frozenset(
    {
        "wrong_object",
        "wrong object manipulated",
        "wrong_place",
        "wrong_target",
        "wrong_region",
        "translation",  # often pose looks fine but wrong contact target in VQA taxonomies
        "rotation",
    }
)

# Failure modes typically revealed by gripper/joint/contact/force.
PROPRIO_TRUE_MODES = frozenset(
    {
        "no_close",
        "no-close",
        "slip",
        "drop",
        "dropped",
        "jam",
        "misalignment",
        "no_progress",
        "no-progress",
        "blocked",
        "stuck",
        "contact_fail",
    }
)

NOMINAL_MODES = frozenset({"ground_truth", "success", "nominal", "none", ""})


def _norm_mode(meta: Mapping[str, Any]) -> str:
    mode = meta.get("failure_mode", meta.get("failure_reason", ""))
    return str(mode or "").strip().lower()


def _is_success(meta: Mapping[str, Any]) -> bool:
    for key in ("reward", "execution_reward", "planning_reward"):
        if key in meta:
            try:
                return float(meta[key]) >= 0.5
            except (TypeError, ValueError):
                continue
    mode = _norm_mode(meta)
    return mode in NOMINAL_MODES


def _state_nominal(state_summary: Mapping[str, Any]) -> bool:
    """Heuristic: small motion / no strong gripper mismatch ⇒ state looks nominal."""
    if not state_summary:
        return True
    if "gripper_mismatch" in state_summary:
        return not bool(state_summary["gripper_mismatch"])
    width = state_summary.get("gripper_width")
    cmd = state_summary.get("gripper_cmd_closed")
    if width is not None and cmd is not None:
        try:
            # Commanded closed but still wide ⇒ proprio evidence of failure.
            if float(cmd) >= 0.5 and float(width) > 0.04:
                return False
        except (TypeError, ValueError):
            pass
    motion = state_summary.get("joint_motion_norm", state_summary.get("gripper_delta", 0.0))
    try:
        return float(motion) < 0.5
    except (TypeError, ValueError):
        return True


def label_stratum(
    episode_meta: Mapping[str, Any],
    state_summary: Mapping[str, Any] | None = None,
) -> Stratum:
    """Deterministic E1–E4 label from eval-only metadata + observable state summary.

    Rules (pre-registered):
    - E4 Nominal: success / no failure.
    - E3 Proprio-true: modes needing gripper/contact/force, or explicit gripper mismatch.
    - E2 Vision-true: wrong object/place/target with non-contradictory/nominal state.
    - E1 Agree: remaining failures where both channels are expected informative.
    - unassigned: insufficient metadata.
    """
    state_summary = state_summary or {}
    if _is_success(episode_meta):
        return "E4"

    mode = _norm_mode(episode_meta)
    if not mode and not state_summary:
        return "unassigned"

    if mode in PROPRIO_TRUE_MODES:
        return "E3"
    if state_summary.get("gripper_mismatch") or (
        state_summary.get("gripper_cmd_closed") is not None
        and state_summary.get("gripper_width") is not None
        and float(state_summary.get("gripper_cmd_closed", 0)) >= 0.5
        and float(state_summary.get("gripper_width", 0)) > 0.04
    ):
        return "E3"

    if mode in VISION_TRUE_MODES and _state_nominal(state_summary):
        return "E2"

    if mode:
        return "E1"
    return "unassigned"
