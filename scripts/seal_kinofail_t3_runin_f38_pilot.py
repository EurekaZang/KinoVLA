#!/usr/bin/env python3
"""Freeze a three-domain development-only pilot for the F38 T3 run-in."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_ROOT = ROOT / "outputs/kinofail_reconfirmation_v2/schedules/scenes"
REGISTRY = ROOT / "outputs/kinofail_reconfirmation_v2/scene_registry.json"
ASSET_LOCK = ROOT / "outputs/assets/terrain_pbr_confirmatory_v2/terrain_assets.lock.json"
F37_AUDIT = ROOT / "outputs/kinofail_reconfirmation_f37_failure_audit_v1/audit.json"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_confirmatory_t3_runin_f38.py"
OUTPUT = ROOT / "outputs/freeze/kinofail_t3_runin_f38_pilot"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_runin_f38_pilot/corpus")
SCENES = (
    "confirm_v2_life_scene_00",
    "confirm_v2_production_scene_00",
    "confirm_v2_wild_scene_00",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> int:
    if OUTPUT.exists() or CORPUS.exists():
        raise FileExistsError("refusing to overwrite F38 pilot seal or corpus")
    f37 = read_json(F37_AUDIT)
    if (
        f37.get("status") != "result_blind_systematic_temporal_contract_failure_preserved"
        or f37.get("scientific_attempt_started") is not False
        or f37.get("model_prediction_truth_key_or_score_read") is not False
        or f37.get("counts", {}).get("temporally_extractable") != 0
    ):
        raise RuntimeError("F37 failure audit is invalid")
    selected: list[dict[str, Any]] = []
    schedule_artifacts = []
    protocol_artifacts = []
    for scene in SCENES:
        root = SCHEDULE_ROOT / scene / "c2_t3"
        schedule = root / "schedule.jsonl"
        case_schedule = root / "case_schedule.jsonl"
        protocol = root / "collection_protocol.json"
        cases = read_jsonl(case_schedule)
        candidates = sorted(
            (
                row
                for row in cases
                if int(row["local_case_index"]) == 0
                and int(row["material_slot"]) == 0
            ),
            key=lambda row: str(row["case_id"]),
        )
        if len(candidates) != 1:
            raise RuntimeError(f"pilot case selection is not unique: {scene}")
        case = candidates[0]
        pair_ids = [
            str(case["o7_source_physics_group_id"]),
            str(case["o8_source_physics_group_id"]),
        ]
        selected.append(
            {
                "scene_id": scene,
                "case_id": str(case["case_id"]),
                "pair_ids": pair_ids,
                "schedule": str(schedule.relative_to(ROOT)),
                "protocol": str(protocol.relative_to(ROOT)),
                "development_only": True,
            }
        )
        schedule_artifacts.append({"path": str(schedule.relative_to(ROOT)), "sha256": sha256(schedule)})
        protocol_artifacts.append({"path": str(protocol.relative_to(ROOT)), "sha256": sha256(protocol)})
    if len({pair for row in selected for pair in row["pair_ids"]}) != 6:
        raise RuntimeError("pilot pairs are not unique")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    selected_path = OUTPUT / "selected_cases.jsonl"
    selected_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    scripts = [
        COLLECTOR,
        ROOT / "scripts/run_kinofail_t3_runin_f38_pilot.py",
        ROOT / "scripts/seal_kinofail_t3_runin_f38_pilot.py",
        ROOT / "kino_vla/eval/c2_temporal_v5.py",
    ]
    manifest = {
        "schema_version": "kinofail.t3-runin-f38-pilot-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_development_only_physical_pilot",
        "passed": True,
        "development_only": True,
        "counts_as_confirmatory_evidence": False,
        "model_prediction_truth_key_or_score_read": False,
        "selection_uses_runtime_outcomes": False,
        "selection_rule": "material_slot==0 and local_case_index==0 in the fixed scene-00 of each domain",
        "runin_contract": {
            "center_progress_m": 1.25,
            "route_half_length_m": 0.55,
            "frozen_v5_feature_function_unchanged": True,
            "operators_sensors_appearance_nuisance_thresholds_unchanged": True,
        },
        "selected_cases": str(selected_path.relative_to(ROOT)),
        "selected_cases_sha256": sha256(selected_path),
        "corpus_root": str(CORPUS),
        "scene_registry": str(REGISTRY.relative_to(ROOT)),
        "scene_registry_sha256": sha256(REGISTRY),
        "material_lock": str(ASSET_LOCK.relative_to(ROOT)),
        "material_lock_sha256": sha256(ASSET_LOCK),
        "source_sha256": {
            "f37_failure_audit": sha256(F37_AUDIT),
            "schedules": schedule_artifacts,
            "protocols": protocol_artifacts,
            "scripts": {str(path.relative_to(ROOT)): sha256(path) for path in scripts},
        },
    }
    path = OUTPUT / "seal_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUTPUT / "seal_manifest.sha256").write_text(f"{sha256(path)}  {path.name}\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
