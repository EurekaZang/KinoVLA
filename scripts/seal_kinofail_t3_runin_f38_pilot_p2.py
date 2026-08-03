#!/usr/bin/env python3
"""Seal the second, unexposed F38 run-in development pilot."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.seal_kinofail_t3_runin_f38_pilot import (
    ASSET_LOCK,
    REGISTRY,
    SCHEDULE_ROOT,
    SCENES,
    read_json,
    read_jsonl,
    sha256,
)


P1 = ROOT / "outputs/kinofail_t3_runin_f38_pilot/postrun_audit.json"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_confirmatory_t3_runin_f38_v2.py"
RUNNER = ROOT / "scripts/run_kinofail_t3_runin_f38_collection.py"
OUTPUT = ROOT / "outputs/freeze/kinofail_t3_runin_f38_pilot_p2"
RUN = ROOT / "outputs/kinofail_t3_runin_f38_pilot_p2"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_runin_f38_pilot_p2/corpus")


def main() -> int:
    if OUTPUT.exists() or RUN.exists() or CORPUS.exists():
        raise FileExistsError("refusing to overwrite F38 P2")
    p1 = read_json(P1)
    if (
        p1.get("passed") is not False
        or p1.get("development_only") is not True
        or p1.get("counts_as_confirmatory_evidence") is not False
        or p1.get("checks", {}).get("six_anomaly_windows_extractable") is not True
    ):
        raise RuntimeError("F38 P1 does not establish the expected development boundary")
    selected = []
    schedules = []
    protocols = []
    for scene in SCENES:
        root = SCHEDULE_ROOT / scene / "c2_t3"
        cases = read_jsonl(root / "case_schedule.jsonl")
        candidates = [
            row
            for row in cases
            if int(row["material_slot"]) == 0 and int(row["local_case_index"]) == 1
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"P2 selection not unique: {scene}")
        case = candidates[0]
        selected.append(
            {
                "scene_id": scene,
                "case_id": str(case["case_id"]),
                "pair_ids": [str(case["o7_source_physics_group_id"]), str(case["o8_source_physics_group_id"])],
                "schedule": str((root / "schedule.jsonl").relative_to(ROOT)),
                "protocol": str((root / "collection_protocol.json").relative_to(ROOT)),
                "development_only": True,
            }
        )
        schedules.append(root / "schedule.jsonl")
        protocols.append(root / "collection_protocol.json")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    selected_path = OUTPUT / "selected_cases.jsonl"
    selected_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    seal = {
        "schema_version": "kinofail.t3-runin-f38-pilot-p2-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_second_development_only_physical_pilot",
        "passed": True,
        "development_only": True,
        "counts_as_confirmatory_evidence": False,
        "strict_all_cases_pass": True,
        "model_prediction_truth_key_or_score_read": False,
        "result_dependent_retry": False,
        "selection_uses_runtime_outcomes": False,
        "runin_contract": {
            "center_progress_m": 0.82,
            "route_half_length_m": 0.30,
            "near_edge_progress_m": 0.52,
            "earliest_footprint_encounter_progress_m": 0.17,
            "frozen_v5_feature_function_unchanged": True,
        },
        "counts": {"planned_cases": 3, "pairs_to_collect": 6},
        "maximum_case_attrition_rate": 0.0,
        "maximum_concurrent_isaac_processes": 3,
        "audit_schema_version": "kinofail.t3-runin-f38-pilot-p2-audit.v1",
        "selected_cases": str(selected_path.relative_to(ROOT)),
        "selected_cases_sha256": sha256(selected_path),
        "collector": str(COLLECTOR.relative_to(ROOT)),
        "collector_sha256": sha256(COLLECTOR),
        "runner_sha256": sha256(RUNNER),
        "v5_feature_function_sha256": sha256(ROOT / "kino_vla/eval/c2_temporal_v5.py"),
        "scene_registry": str(REGISTRY.relative_to(ROOT)),
        "scene_registry_sha256": sha256(REGISTRY),
        "material_lock": str(ASSET_LOCK.relative_to(ROOT)),
        "material_lock_sha256": sha256(ASSET_LOCK),
        "output_root": str(RUN.relative_to(ROOT)),
        "corpus_root": str(CORPUS),
        "source_sha256": {
            "p1_failure_audit": sha256(P1),
            "schedules": {str(path.relative_to(ROOT)): sha256(path) for path in schedules},
            "protocols": {str(path.relative_to(ROOT)): sha256(path) for path in protocols},
        },
    }
    path = OUTPUT / "seal_manifest.json"
    path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    (OUTPUT / "seal_manifest.sha256").write_text(f"{sha256(path)}  {path.name}\n")
    print(json.dumps(seal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
