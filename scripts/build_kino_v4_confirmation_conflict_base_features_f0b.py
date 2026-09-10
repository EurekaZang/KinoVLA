#!/usr/bin/env python3
"""Build F0b Conflict features from F5-audited decision observations only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.data.runtime_manifest import schedule_record_sha256  # noqa: E402
from kino_vla.eval.c2_temporal_v5 import (  # noqa: E402
    MAX_END_SKEW_S,
    geometry_aligned_invariant_summary,
)
from scripts import build_kino_v4_confirmation_conflict_base_features_v1 as base  # noqa: E402


TASK_AUDIT = (
    ROOT
    / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h"
    / "f5_task_aligned_audit"
)
FORBIDDEN_SCORE_ARTIFACTS = (
    ROOT / "outputs/eval/kino_v4_confirmation_v1_score_once/report.json",
    ROOT / "outputs/eval/kino_v4_confirmation_v1_f4m_score_once/report.json",
    ROOT / "outputs/eval/kino_v4_confirmation_v1_f5e_score_once/report.json",
)
VIEWS = ("primary", "swap_01", "swap_02")
VISUAL_FRAMES = 5
MAX_RGB_DECISION_SKEW_S = 0.051


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _task_pair_lookup() -> dict[str, dict[str, Any]]:
    audit_path = TASK_AUDIT / "audit.json"
    pair_path = TASK_AUDIT / "pair_audits.jsonl"
    audit = _json(audit_path)
    if (
        audit.get("passed") is not True
        or audit.get("model_prediction_truth_key_or_score_read") is not False
        or audit.get("artifact_sha256", {}).get("pair_audits") != _sha256(pair_path)
    ):
        raise RuntimeError("F5 task-aligned audit is not valid and model blind")
    rows = _jsonl(pair_path)
    lookup = {str(row["pair_id"]): row for row in rows if row.get("passed") is True}
    if len(lookup) != int(audit["counts"]["accepted_pairs"]):
        raise RuntimeError("F5 accepted-pair registry drift")
    return lookup


def main() -> int:
    if any(path.exists() for path in FORBIDDEN_SCORE_ARTIFACTS):
        raise RuntimeError("confirmation score artifact exists before F0b features")
    pair_lookup = _task_pair_lookup()
    module = base._load()
    original_json = module._json
    decision_cache: dict[str, float] = {}

    def validated_anomaly(
        schedule_by_group: dict[str, list[dict[str, Any]]],
        group_id: str,
        corpus: Path,
    ) -> tuple[dict[str, Any], dict[str, Any], Path, np.ndarray]:
        candidates = [
            row
            for row in schedule_by_group[group_id]
            if row["condition"] == "anomaly"
        ]
        if len(candidates) != 1 or group_id not in pair_lookup:
            raise RuntimeError(f"missing F5-accepted anomaly: {group_id}")
        schedule = candidates[0]
        episode_dir = (
            corpus
            / str(schedule["scene_id"])
            / Path(schedule["required_outputs"]["episode_manifest"]).parent
        )
        manifest_path = episode_dir / "manifest.json"
        manifest = original_json(manifest_path)
        episode_audit = pair_lookup[group_id]["episodes"]["anomaly"]
        if (
            episode_audit.get("passed") is not True
            or episode_audit.get("task_input_integrity_passed") is not True
            or episode_audit.get("manifest_sha256") != _sha256(manifest_path)
            or manifest.get("episode_id") != schedule["episode_id"]
            or manifest.get("counterfactual_group_id") != group_id
            or manifest.get("schedule_record_sha256")
            != schedule_record_sha256(schedule)
        ):
            raise RuntimeError(f"F5 anomaly binding failed: {group_id}")
        proprio_path = episode_dir / "proprio.npz"
        if _sha256(proprio_path) != manifest["artifacts"]["proprio"]["sha256"]:
            raise RuntimeError(f"F5 proprio hash mismatch: {group_id}")
        with np.load(proprio_path, allow_pickle=False) as archive:
            proprio = np.asarray(archive["features"], dtype=np.float32)
        return schedule, manifest, episode_dir, proprio

    def rgb_sequence(
        episode_dir: Path,
        manifest: dict[str, Any],
        view: str,
    ) -> tuple[np.ndarray, list[str]]:
        if view not in VIEWS:
            raise RuntimeError(f"unexpected RGB view: {view}")
        key = str(episode_dir.resolve())
        if key not in decision_cache:
            _, alignment = geometry_aligned_invariant_summary(episode_dir)
            decision_cache[key] = float(alignment["decision_time_s"])
        decision_time = decision_cache[key]
        entries = manifest["artifacts"]["rgb_views"][view]
        times = np.asarray([float(row["timestamp_s"]) for row in entries])
        end = int(np.argmin(np.abs(times - decision_time)))
        skew = abs(float(times[end]) - decision_time)
        if skew > MAX_RGB_DECISION_SKEW_S or end + 1 < VISUAL_FRAMES:
            raise RuntimeError(f"decision RGB unavailable: {episode_dir} {view}")
        selected = entries[end - VISUAL_FRAMES + 1 : end + 1]
        frames: list[np.ndarray] = []
        paths: list[str] = []
        for entry in selected:
            path = episode_dir / str(entry["path"])
            if _sha256(path) != str(entry["sha256"]):
                raise RuntimeError(f"decision RGB hash mismatch: {path}")
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            paths.append(str(entry["path"]))
        return np.stack(frames).astype(np.uint8), paths

    module._validated_anomaly = validated_anomaly
    module._rgb_sequence = rgb_sequence
    result = int(module.main())
    return result


if __name__ == "__main__":
    raise SystemExit(main())
