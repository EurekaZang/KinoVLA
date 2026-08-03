#!/usr/bin/env python3
"""Seal the formal, result-blind F38 T3 run-in recollection."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.c2_temporal_v5 import geometry_aligned_invariant_summary  # noqa: E402
from scripts.seal_kinofail_t3_runin_f38_pilot import (  # noqa: E402
    ASSET_LOCK,
    REGISTRY,
    SCHEDULE_ROOT,
    read_json,
    read_jsonl,
    sha256,
)


F37 = ROOT / "outputs/kinofail_reconfirmation_f37_failure_audit_v1/audit.json"
PILOTS = (
    (
        ROOT / "outputs/freeze/kinofail_t3_runin_f38_pilot/selected_cases.jsonl",
        ROOT / "outputs/kinofail_t3_runin_f38_pilot/postrun_audit.json",
    ),
    (
        ROOT / "outputs/freeze/kinofail_t3_runin_f38_pilot_p2/selected_cases.jsonl",
        ROOT / "outputs/kinofail_t3_runin_f38_pilot_p2/final_audit.json",
    ),
    (
        ROOT / "outputs/freeze/kinofail_t3_runin_f38_pilot_p3/selected_cases.jsonl",
        ROOT / "outputs/kinofail_t3_runin_f38_pilot_p3/final_audit.json",
    ),
)
P3_CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_runin_f38_pilot_p3/corpus")
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_confirmatory_t3_runin_f38_v3.py"
RUNNER = ROOT / "scripts/run_kinofail_t3_runin_f38_collection_v2.py"
RUNNER_PREDECESSOR = ROOT / "scripts/run_kinofail_t3_runin_f38_collection.py"
OUTPUT = ROOT / "outputs/freeze/kinofail_t3_runin_f38_formal"
RUN = ROOT / "outputs/kinofail_t3_runin_f38_formal"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_runin_f38_formal/corpus")
GLOBAL_CASES = ROOT / "outputs/kinofail_reconfirmation_v2/schedules/c2_t3/case_schedule.jsonl"


def main() -> int:
    if OUTPUT.exists() or RUN.exists() or CORPUS.exists():
        raise FileExistsError("refusing to overwrite formal F38")
    f37 = read_json(F37)
    if (
        f37.get("status") != "result_blind_systematic_temporal_contract_failure_preserved"
        or f37.get("model_prediction_truth_key_or_score_read") is not False
    ):
        raise RuntimeError("F37 boundary invalid")
    touched_cases: set[str] = set()
    touched_pairs: set[str] = set()
    pilot_hashes = {}
    for selected_path, audit_path in PILOTS:
        audit = read_json(audit_path)
        if (
            audit.get("development_only") is not True
            or audit.get("counts_as_confirmatory_evidence") is not False
            or audit.get("model_prediction_truth_key_or_score_read") is not False
        ):
            raise RuntimeError(f"invalid development pilot boundary: {audit_path}")
        for row in read_jsonl(selected_path):
            touched_cases.add(str(row["case_id"]))
            touched_pairs.update(str(value) for value in row["pair_ids"])
        pilot_hashes[str(audit_path.relative_to(ROOT))] = sha256(audit_path)
        pilot_hashes[str(selected_path.relative_to(ROOT))] = sha256(selected_path)
    if len(touched_cases) != 9 or len(touched_pairs) != 18:
        raise RuntimeError("formal exclusion does not contain exactly nine pilot cases")

    # P3 confirms physical engagement and temporal feasibility in every domain.
    p3_temporal = 0
    p3_o7_engaged = 0
    p3_o8_engaged = 0
    p3_only_allowed_attrition = 0
    for summary_path in sorted(P3_CORPUS.glob("*/pair_summaries/*.json")):
        summary = read_json(summary_path)
        anomaly_result = next(row for row in summary["results"] if row["condition"] == "anomaly")
        manifest = read_json(Path(anomaly_result["manifest"]))
        operator = str(manifest["operator_readback"]["operator_id"])
        telemetry = manifest["operator_readback"]["telemetry"]
        _, alignment = geometry_aligned_invariant_summary(Path(anomaly_result["manifest"]).parent)
        if alignment["window_start_s"] < alignment["encounter_time_s"] and alignment["end_skew_s"] <= 0.021:
            p3_temporal += 1
        exposure = int(telemetry.get("scale_region_exposure_steps", 0))
        if operator == "O7_visual_remap" and exposure > 0:
            p3_o7_engaged += 1
        if (
            operator == "O8_invisible_collider"
            and exposure > 0
            and telemetry.get("scale_region_exposure_measurement") == "go2_footprint_margin_0.35m"
        ):
            p3_o8_engaged += 1
        if summary.get("passed") is not True:
            issues = set(anomaly_result.get("issues", []))
            if issues and issues <= {
                "rgb_view_swap_01_appearance_effect_too_small",
                "rgb_view_swap_02_appearance_effect_too_small",
            }:
                p3_only_allowed_attrition += 1
    if (p3_temporal, p3_o7_engaged, p3_o8_engaged, p3_only_allowed_attrition) != (6, 3, 3, 1):
        raise RuntimeError("P3 did not establish the formal run-in contract")

    global_cases = read_jsonl(GLOBAL_CASES)
    if len(global_cases) != 1500 or len({str(row["case_id"]) for row in global_cases}) != 1500:
        raise RuntimeError("global T3 design is not the frozen 1,500-case design")
    selected = []
    schedules: dict[str, str] = {}
    protocols: dict[str, str] = {}
    for case in global_cases:
        case_id = str(case["case_id"])
        if case_id in touched_cases:
            continue
        scene = str(case["scene_id"])
        root = SCHEDULE_ROOT / scene / "c2_t3"
        schedule = root / "schedule.jsonl"
        protocol = root / "collection_protocol.json"
        selected.append(
            {
                "scene_id": scene,
                "case_id": case_id,
                "pair_ids": [str(case["o7_source_physics_group_id"]), str(case["o8_source_physics_group_id"])],
                "schedule": str(schedule.relative_to(ROOT)),
                "protocol": str(protocol.relative_to(ROOT)),
                "development_only": False,
            }
        )
        schedules[str(schedule.relative_to(ROOT))] = sha256(schedule)
        protocols[str(protocol.relative_to(ROOT))] = sha256(protocol)
    if len(selected) != 1491 or len({pair for row in selected for pair in row["pair_ids"]}) != 2982:
        raise RuntimeError("formal F38 selection count mismatch")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    selected_path = OUTPUT / "selected_cases.jsonl"
    selected_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    seal = {
        "schema_version": "kinofail.t3-runin-f38-formal-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_formal_result_blind_t3_recollection",
        "passed": True,
        "development_only": False,
        "counts_as_confirmatory_evidence": True,
        "strict_all_cases_pass": False,
        "model_prediction_truth_key_or_score_read": False,
        "result_dependent_retry": False,
        "selection_uses_runtime_outcomes": False,
        "all_development_touched_cases_excluded": True,
        "runin_contract": {
            "center_progress_m": 0.82,
            "route_half_length_m": 0.30,
            "near_edge_progress_m": 0.52,
            "o8_exposure_qa_footprint_margin_m": 0.35,
            "frozen_v5_feature_function_unchanged": True,
            "model_architecture_router_threshold_and_statistics_unchanged": True,
        },
        "counts": {
            "planned_cases": 1500,
            "development_excluded_cases": 9,
            "selected_formal_cases": 1491,
            "pairs_to_collect": 2982,
            "physical_episodes": 5964,
        },
        "maximum_case_attrition_rate": 0.05,
        "minimum_accepted_cases": 1426,
        "maximum_concurrent_isaac_processes": 6,
        "audit_schema_version": "kinofail.t3-runin-f38-formal-audit.v1",
        "selected_cases": str(selected_path.relative_to(ROOT)),
        "selected_cases_sha256": sha256(selected_path),
        "collector": str(COLLECTOR.relative_to(ROOT)),
        "collector_sha256": sha256(COLLECTOR),
        "runner_sha256": sha256(RUNNER),
        "runner_predecessor_sha256": sha256(RUNNER_PREDECESSOR),
        "v5_feature_function_sha256": sha256(ROOT / "kino_vla/eval/c2_temporal_v5.py"),
        "scene_registry": str(REGISTRY.relative_to(ROOT)),
        "scene_registry_sha256": sha256(REGISTRY),
        "material_lock": str(ASSET_LOCK.relative_to(ROOT)),
        "material_lock_sha256": sha256(ASSET_LOCK),
        "output_root": str(RUN.relative_to(ROOT)),
        "corpus_root": str(CORPUS),
        "source_sha256": {
            "f37_failure_audit": sha256(F37),
            "global_case_schedule": sha256(GLOBAL_CASES),
            "development_pilots": pilot_hashes,
            "schedules": dict(sorted(schedules.items())),
            "protocols": dict(sorted(protocols.items())),
        },
    }
    path = OUTPUT / "seal_manifest.json"
    path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    (OUTPUT / "seal_manifest.sha256").write_text(f"{sha256(path)}  {path.name}\n")
    print(json.dumps(seal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
