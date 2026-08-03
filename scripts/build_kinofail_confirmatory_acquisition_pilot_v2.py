#!/usr/bin/env python3
"""Freeze an excluded two-pair acquisition pilot for collector-v2 QA."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = (
    ROOT / "outputs/kinofail_confirmatory_invalid_acquisition_pilot_v1_20260725"
)
SOURCE_SCHEDULE = (
    ARCHIVE
    / "schedules_f2_scene_source_amendment/scenes/"
    "confirm_v1_life_scene_00/c2_t3/schedule.jsonl"
)
SOURCE_PROTOCOL = (
    ARCHIVE
    / "schedules_f2_scene_source_amendment/scenes/"
    "confirm_v1_life_scene_00/c2_t3/collection_protocol.json"
)
SOURCE_REGISTRY = ARCHIVE / "scene_registry.json"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v2.py"
OUTPUT = ROOT / "outputs/kinofail_acquisition_pilot_v2/design"
GROUPS = (
    "cf_190ea2ab6e27fa8cf564",
    "cf_0b0345f0f292fc2034ca",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _replace_root(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(
            "outputs/kinofail_confirmatory_v1/",
            "outputs/kinofail_confirmatory_invalid_acquisition_pilot_v1_20260725/",
        )
    if isinstance(value, list):
        return [_replace_root(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace_root(item) for key, item in value.items()}
    return value


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    rows = [
        row
        for row in _jsonl(SOURCE_SCHEDULE)
        if str(row["counterfactual_group_id"]) in GROUPS
    ]
    if (
        len(rows) != 4
        or {str(row["counterfactual_group_id"]) for row in rows} != set(GROUPS)
    ):
        raise RuntimeError("excluded acquisition pilot pairs are incomplete")
    schedule_path = OUTPUT / "schedule.jsonl"
    schedule_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    source_registry = _json(SOURCE_REGISTRY)
    scene_rows = [
        row
        for row in source_registry["scenes"]
        if row["scene_id"] == "confirm_v1_life_scene_00"
    ]
    if len(scene_rows) != 1:
        raise RuntimeError("pilot scene registry binding is not unique")
    registry = {
        **source_registry,
        "schema_version": "kinofail.acquisition-pilot-registry.v2",
        "scenes": [_replace_root(scene_rows[0])],
        "status": "excluded_model_blind_acquisition_pilot",
    }
    registry_path = OUTPUT / "scene_registry.json"
    _write_json(registry_path, registry)

    protocol = _json(SOURCE_PROTOCOL)
    protocol["protocol_id"] = "kinofail-acquisition-pilot-v2-collector-tail"
    protocol["schedule_sha256"] = _sha256(schedule_path)
    protocol["scene_registry_sha256"] = _sha256(registry_path)
    protocol["collector_sha256"] = _sha256(COLLECTOR)
    protocol["allowed"] = {
        "conditions": sorted({str(row["condition"]) for row in rows}),
        "counterfactual_group_ids": list(GROUPS),
        "geometry_profiles": sorted(
            {str(row["geometry_profile"]) for row in rows}
        ),
        "physical_realizations": sorted(
            {str(row["physical_realization"]) for row in rows}
        ),
        "scene_families": sorted(
            {str(row["scene_family"]) for row in rows}
        ),
        "severity_ids": sorted({str(row["severity_id"]) for row in rows}),
        "target_operators": sorted(
            {str(row["target_operator"]) for row in rows}
        ),
    }
    protocol_path = OUTPUT / "collection_protocol.json"
    _write_json(protocol_path, protocol)

    audit = {
        "schema_version": "kinofail.acquisition-pilot-freeze.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_before_excluded_pilot_collection",
        "confirmatory_claim_eligible": False,
        "model_predictions_allowed": False,
        "purpose": [
            "verify the fixed post-failure diagnostic horizon on an excluded O8 failure",
            "retain one known O7 appearance-threshold failure as a negative control",
        ],
        "group_ids": list(GROUPS),
        "source_sha256": {
            "invalidated_v1_receipt": _sha256(
                ARCHIVE / "INVALIDATED_ACQUISITION_PILOT.json"
            ),
            "source_schedule": _sha256(SOURCE_SCHEDULE),
            "source_protocol": _sha256(SOURCE_PROTOCOL),
            "source_registry": _sha256(SOURCE_REGISTRY),
            "collector_v2": _sha256(COLLECTOR),
        },
        "artifact_sha256": {
            "schedule": _sha256(schedule_path),
            "scene_registry": _sha256(registry_path),
            "collection_protocol": _sha256(protocol_path),
        },
    }
    _write_json(OUTPUT / "freeze_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
