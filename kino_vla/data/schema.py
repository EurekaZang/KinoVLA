"""Hindsight-CoT data types + structured-output parsing (spec §10).

The Privileged-Grounded Hindsight CoT dataset (spec §10) is built from four object types:

- ``Snapshot`` — the PHASE 2 multimodal package the Kino-Monitor intercepts at a failure:
  ``[5×RGB] + [5×depth] + [proprio window (Kino-Tokens precursor)] + [prior VLA outputs] +
  [privileged physics truth]``.
- ``CoTAnnotation`` — the Oracle's PHASE 3 reflection: a free-text thought, a physical
  *attribution* (normalized onto a canonical category), and exactly one atomic *recovery
  primitive* from the §5 library. Parsed + schema-validated here, so a malformed or
  non-atomic Oracle output is rejected before it can reach the filter (the spec's "严格原子
  动作 / 2D 像素坐标输出格式" prompt constraints become hard schema checks).
- ``GroundTruth`` — the privileged attribution derived from the active operator + θ
  (``kino_vla.data.taxonomy``): the canonical category, A/B class, and feasible recovery set
  the filter checks the annotation against.
- ``DataSample`` — one kept (or dropped, with a ``Verdict``) training example: scenario
  metadata + snapshot refs + ground truth + annotation + the privileged-θ distillation target.

Pure-Python/numpy on purpose (no torch): the whole M6 pipeline runs on the CI machine.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Canonical recovery primitives — the §5 library names (must match
# ``kino_vla.shield.primitive_compiler``; that compiler is what M7's planner emits into).
PRIMITIVE_NAMES: frozenset[str] = frozenset(
    {
        "continue",
        "Backstep",
        "Replan_Waypoint",
        "Switch_Gait",
        "Adjust_Posture",
        "Set_Constraint",
        "Update_Topology",
        "Hold_and_Request",
    }
)
GAITS: frozenset[str] = frozenset({"high_step", "trot", "crawl"})

# Verdict reason codes (a fixed, golden-testable vocabulary; spec §10 PHASE 3 filter).
KEEP = "keep"
DROP_SCHEMA = "schema_invalid"
DROP_ATTRIBUTION = "attribution_mismatch"
DROP_PRIMITIVE = "primitive_not_feasible"
DROP_SAFETY = "safety_rule_violation"
# NOT a truth-consistency verdict: the Oracle CALL itself failed (API HTTP error / timeout /
# dropped connection). Tracked as its own status so infrastructure flakiness is never counted as a
# filter rejection — the reported reject-rate must reflect the FILTER's judgments only (the metric a
# paper cites). Excluded from reject_rate; reported separately on the card. spec §10 PHASE 3.
ORACLE_ERROR = "oracle_error"


class CoTParseError(ValueError):
    """Raised when an Oracle CoT cannot be parsed into a valid atomic annotation.

    Carries a short ``reason`` so the pipeline can record *why* the sample was dropped
    (all parse failures map to the ``DROP_SCHEMA`` verdict).
    """


@dataclass(frozen=True)
class RecoveryPrimitive:
    """One atomic §5 recovery primitive: a library name + validated parameters."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in PRIMITIVE_NAMES:
            raise CoTParseError(f"unknown primitive {self.name!r} (not in the §5 library)")
        self._validate()

    def _validate(self) -> None:
        """Per-primitive schema (the spec's atomic-action / 2D-pixel output constraints)."""
        p = self.params
        if self.name == "continue":
            if p:
                raise CoTParseError("continue takes no parameters")
        elif self.name == "Backstep":
            if not _is_pos_number(p.get("distance_m")):
                raise CoTParseError("Backstep requires distance_m > 0")
        elif self.name == "Replan_Waypoint":
            pt = p.get("point_px")
            # spec §10: Replan outputs a 2D *pixel* coordinate (depth-unprojected downstream).
            if not (
                isinstance(pt, (list, tuple)) and len(pt) == 2 and all(_is_number(c) for c in pt)
            ):
                raise CoTParseError("Replan_Waypoint requires point_px = [u, v] (2D pixel coord)")
        elif self.name == "Switch_Gait":
            if p.get("mode") not in GAITS:
                raise CoTParseError(f"Switch_Gait.mode must be one of {sorted(GAITS)}")
        elif self.name == "Adjust_Posture":
            if not _is_pos_number(p.get("body_height_m")):
                raise CoTParseError("Adjust_Posture requires body_height_m > 0")
        elif self.name == "Set_Constraint":
            if not _is_pos_number(p.get("max_speed")):
                raise CoTParseError("Set_Constraint requires max_speed > 0")
        elif self.name == "Update_Topology":
            if not isinstance(p.get("status"), str):
                raise CoTParseError("Update_Topology requires a status string")
        elif self.name == "Hold_and_Request":
            if not isinstance(p.get("reason"), str):
                raise CoTParseError("Hold_and_Request requires a reason string")

    def to_dict(self) -> dict[str, Any]:
        return {"primitive": self.name, "params": dict(self.params)}


@dataclass(frozen=True)
class CoTAnnotation:
    """A parsed, schema-valid Oracle annotation (spec §10 PHASE 3).

    ``attribution`` is the *normalized* canonical category (e.g. ``"low_friction"``);
    ``attribution_raw`` keeps the Oracle's original word for the dataset card / audit.
    """

    thought: str
    attribution: str
    primitive: RecoveryPrimitive
    attribution_raw: str
    raw_text: str

    @classmethod
    def from_oracle_text(
        cls,
        text: str,
        *,
        synonyms: dict[str, str],
        valid_categories: frozenset[str],
    ) -> CoTAnnotation:
        """Parse + schema-validate raw Oracle output into an atomic annotation.

        Accepts either a bare JSON object or the ``<Thought>…</Thought><Action>{json}</Action>``
        format (the structured form M7's parser also consumes). Raises ``CoTParseError`` on
        any malformed / non-atomic / off-vocabulary output (→ ``DROP_SCHEMA``).
        """
        obj = _extract_json_object(text)
        thought = str(obj.get("thought", "")).strip()
        attribution_raw = obj.get("attribution")
        if not isinstance(attribution_raw, str) or not attribution_raw.strip():
            raise CoTParseError("missing 'attribution'")
        attribution_raw = attribution_raw.strip()
        category = _normalize_category(attribution_raw, synonyms)
        if category not in valid_categories:
            raise CoTParseError(f"attribution {attribution_raw!r} not in the category vocabulary")

        action = obj.get("action")
        if not isinstance(action, dict):
            raise CoTParseError("missing 'action' object (one atomic primitive required)")
        name = action.get("primitive")
        if not isinstance(name, str):
            raise CoTParseError("action.primitive must be a string")
        params = action.get("params", {})
        if not isinstance(params, dict):
            raise CoTParseError("action.params must be an object")
        primitive = RecoveryPrimitive(name=name, params=params)  # raises CoTParseError if invalid
        return cls(
            thought=thought,
            attribution=category,
            primitive=primitive,
            attribution_raw=attribution_raw,
            raw_text=text,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "thought": self.thought,
            "attribution": self.attribution,
            "attribution_raw": self.attribution_raw,
            "action": self.primitive.to_dict(),
        }


@dataclass(frozen=True)
class GroundTruth:
    """Privileged attribution for one failure (spec §10 PHASE 3 anchor)."""

    category: str
    ab_class: str  # "A" (low-level recoverable) | "B" (semantic intervention required)
    theta: dict[str, float]
    feasible: frozenset[str]
    canonical_primitive: str
    is_sudden_trap: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "ab_class": self.ab_class,
            "theta": dict(self.theta),
            "feasible": sorted(self.feasible),
            "canonical_primitive": self.canonical_primitive,
            "is_sudden_trap": self.is_sudden_trap,
        }


@dataclass(frozen=True)
class Verdict:
    """Truth-consistency filter result (spec §10 PHASE 3)."""

    keep: bool
    reason: str  # one of KEEP / DROP_* above
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"keep": self.keep, "reason": self.reason, "detail": self.detail}


@dataclass(frozen=True)
class Snapshot:
    """PHASE 2 multimodal failure snapshot (spec §10).

    Heavy modalities (``rgb``/``depth``/``proprio_window``) are numpy arrays serialized to a
    sidecar ``.npz``; ``to_meta`` is the light JSON record. ``privileged_theta`` is the
    god's-eye physics at the interception (the distillation anchor + filter ground truth).
    """

    operator_name: str
    appearance_class: str
    t: float
    pose_xy: np.ndarray  # (2,)
    heading: float
    rgb: np.ndarray  # (n_frames, H, W, 3)
    depth: np.ndarray  # (n_frames, H, W)
    proprio_window: np.ndarray  # (W, F) the 500 ms Kino-Tokens precursor window
    prior_outputs: list[str]  # prior-round recovery decisions (texts), spec §10 PHASE 2
    privileged_theta: dict[str, float]
    monitor_channel: str

    def to_meta(self) -> dict[str, Any]:
        return {
            "operator_name": self.operator_name,
            "appearance_class": self.appearance_class,
            "t": self.t,
            "pose_xy": [float(self.pose_xy[0]), float(self.pose_xy[1])],
            "heading": float(self.heading),
            "prior_outputs": list(self.prior_outputs),
            "privileged_theta": {k: float(v) for k, v in self.privileged_theta.items()},
            "monitor_channel": self.monitor_channel,
            "rgb_shape": list(self.rgb.shape),
            "depth_shape": list(self.depth.shape),
            "proprio_shape": list(self.proprio_window.shape),
        }


@dataclass(frozen=True)
class DataSample:
    """One Hindsight-CoT example (kept or dropped, with its verdict)."""

    sample_id: str
    snapshot: Snapshot
    ground_truth: GroundTruth
    annotation: CoTAnnotation | None  # None when the Oracle output failed to parse
    verdict: Verdict
    target_theta: list[float]  # privileged-θ distillation target (TARGET_SCHEMA order)
    ambiguity_pair: str | None  # "O4_tether|O2_compliance" if the operator is in a pair

    def to_record(self) -> dict[str, Any]:
        """The JSONL manifest record (arrays live in the sidecar ``.npz``)."""
        return {
            "sample_id": self.sample_id,
            "snapshot": self.snapshot.to_meta(),
            "ground_truth": self.ground_truth.to_dict(),
            "annotation": self.annotation.to_dict() if self.annotation is not None else None,
            "verdict": self.verdict.to_dict(),
            "target_theta": [float(x) for x in self.target_theta],
            "ambiguity_pair": self.ambiguity_pair,
        }


# ------------------------------------------------------------------- parse helpers
def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_pos_number(x: Any) -> bool:
    return _is_number(x) and float(x) > 0.0


def _normalize_category(word: str, synonyms: dict[str, str]) -> str:
    return synonyms.get(word.strip().lower().replace(" ", "_"), word.strip().lower())


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_ACTION_TAG = re.compile(r"<Action>\s*(.*?)\s*</Action>", re.DOTALL | re.IGNORECASE)
_THOUGHT_TAG = re.compile(r"<Thought>\s*(.*?)\s*</Thought>", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull a single JSON object out of raw Oracle text (bare, fenced, or tagged)."""
    if not isinstance(text, str) or not text.strip():
        raise CoTParseError("empty Oracle output")

    # <Thought>/<Action>{json}</Action> form: merge into one object.
    action_match = _ACTION_TAG.search(text)
    if action_match is not None:
        try:
            action = json.loads(action_match.group(1))
        except json.JSONDecodeError as err:
            raise CoTParseError(f"<Action> is not valid JSON: {err}") from err
        thought_match = _THOUGHT_TAG.search(text)
        thought = thought_match.group(1).strip() if thought_match else ""
        if not isinstance(action, dict):
            raise CoTParseError("<Action> must contain a JSON object")
        # The <Action> block may be flat ({attribution, primitive, params}) or nested
        # ({attribution, action: {primitive, params}}); normalize to the canonical object.
        merged: dict[str, Any] = {"thought": thought, "attribution": action.get("attribution")}
        if isinstance(action.get("action"), dict):
            merged["action"] = action["action"]
        else:
            merged["action"] = {
                "primitive": action.get("primitive"),
                "params": action.get("params", {}),
            }
        return merged

    candidate = text.strip()
    fence = _FENCE.search(candidate)
    if fence is not None:
        candidate = fence.group(1).strip()
    obj = _loads_json_object(candidate)
    if obj is None:
        raise CoTParseError("Oracle output is not a valid JSON object")
    return obj


def _balanced_object_spans(s: str) -> list[tuple[int, int]]:
    """Spans of top-level balanced ``{...}`` regions, brace-counting outside string literals."""
    spans: list[tuple[int, int]] = []
    depth = start = 0
    in_str = esc = False
    start = -1
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append((start, i + 1))
    return spans


def _loads_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from raw text, tolerating prose around (and before) the JSON.

    Tries the whole string first, then the balanced ``{...}`` spans (preferring the LAST one that
    decodes to a dict and carries an ``attribution`` — so a reply like ``"Reasoning {note}: {...}"``
    still parses instead of being mis-counted as a schema reject)."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    dicts: list[dict[str, Any]] = []
    for a, b in _balanced_object_spans(text):
        try:
            cand = json.loads(text[a:b])
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict):
            dicts.append(cand)
    if not dicts:
        return None
    for cand in reversed(dicts):  # prefer the last well-formed annotation object
        if "attribution" in cand or "action" in cand:
            return cand
    return dicts[-1]
