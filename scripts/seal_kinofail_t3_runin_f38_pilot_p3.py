#!/usr/bin/env python3
"""Seal the third unexposed F38 pilot after the O8 QA diagnosis."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.seal_kinofail_t3_runin_f38_pilot import (  # noqa: E402
    ASSET_LOCK,
    REGISTRY,
    SCHEDULE_ROOT,
    SCENES,
    read_json,
    read_jsonl,
    sha256,
)


P2 = ROOT / "outputs/kinofail_t3_runin_f38_pilot_p2/final_audit.json"
P2_CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_runin_f38_pilot_p2/corpus")
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_confirmatory_t3_runin_f38_v3.py"
RUNNER = ROOT / "scripts/run_kinofail_t3_runin_f38_collection.py"
OUTPUT = ROOT / "outputs/freeze/kinofail_t3_runin_f38_pilot_p3"
RUN = ROOT / "outputs/kinofail_t3_runin_f38_pilot_p3"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_runin_f38_pilot_p3/corpus")


def main() -> int:
    if OUTPUT.exists() or RUN.exists() or CORPUS.exists():
        raise FileExistsError("refusing to overwrite F38 P3")
    p2 = read_json(P2)
    if (
        p2.get("passed") is not False
        or p2.get("development_only") is not True
        or p2.get("counts", {}).get("passed_pairs") != 3
        or p2.get("counts", {}).get("accepted_cases") != 0
        or p2.get("model_prediction_truth_key_or_score_read") is not False
    ):
        raise RuntimeError("F38 P2 boundary is invalid")
    p2_targets = {"passed": [], "failed": []}
    for summary_path in sorted(P2_CORPUS.glob("*/pair_summaries/*.json")):
        summary = read_json(summary_path)
        anomaly = next(row for row in summary["results"] if row["condition"] == "anomaly")
        manifest = read_json(Path(anomaly["manifest"]))
        operator = str(manifest["operator_readback"]["operator_id"])
        key = "passed" if summary.get("passed") is True else "failed"
        p2_targets[key].append(operator)
        if key == "failed":
            telemetry = manifest["operator_readback"]["telemetry"]
            if (
                operator != "O8_invisible_collider"
                or telemetry.get("enabled") is not True
                or int(telemetry.get("scale_region_exposure_steps", -1)) != 0
                or not all(
                    obstacle.get("collision_requested") is True
                    for obstacle in telemetry.get("obstacles", [])
                )
            ):
                raise RuntimeError("P2 failure is not the diagnosed O8 QA boundary")
    if sorted(p2_targets["passed"]) != ["O7_visual_remap"] * 3 or sorted(p2_targets["failed"]) != ["O8_invisible_collider"] * 3:
        raise RuntimeError(f"unexpected P2 operator boundary: {p2_targets}")
    selected = []
    schedules = []
    protocols = []
    for scene in SCENES:
        root = SCHEDULE_ROOT / scene / "c2_t3"
        cases = read_jsonl(root / "case_schedule.jsonl")
        candidates = [
            row
            for row in cases
            if int(row["material_slot"]) == 0 and int(row["local_case_index"]) == 2
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"P3 selection not unique: {scene}")
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
        "schema_version": "kinofail.t3-runin-f38-pilot-p3-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_third_development_only_physical_pilot",
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
            "o8_exposure_qa_footprint_margin_m": 0.35,
            "frozen_v5_feature_function_unchanged": True,
            "physical_collision_geometry_unchanged_from_p2": True,
        },
        "counts": {"planned_cases": 3, "pairs_to_collect": 6},
        "maximum_case_attrition_rate": 0.0,
        "maximum_concurrent_isaac_processes": 3,
        "audit_schema_version": "kinofail.t3-runin-f38-pilot-p3-audit.v1",
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
            "p2_failure_audit": sha256(P2),
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
