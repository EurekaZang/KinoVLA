#!/usr/bin/env python3
"""Freeze a direct-contact O9 mild/current/hard action confirmation."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/kinofail_action_multiscene_v1_formal_v4"
OUTPUT = ROOT / "outputs/kinofail_o9_severity_v4"
COLLECTOR = "scripts/isaac_collect_kinofail_o9_severity_v4.py"
PROTOCOL_ID = "kinofail-o9-severity-v4-direct-crossbar-20260814"

SEVERITY = {
    "mild": {"ridge_height_m": 0.209, "ridge_width_m": 0.23, "residual_support": 0.55},
    "current": {"ridge_height_m": 0.220, "ridge_width_m": 0.26, "residual_support": 0.36},
    "hard": {"ridge_height_m": 0.235, "ridge_width_m": 0.30, "residual_support": 0.18},
}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_protocol = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    source_rows = [
        row
        for row in jsonl(SOURCE / "schedule.jsonl")
        if row.get("source_rank_contract") == "replicate_b"
        and row.get("operator") == "O9_high_centering"
    ]
    if len(source_rows) != 12:
        raise RuntimeError(f"expected 12 unused replicate-B O9 rows, found {len(source_rows)}")

    cases: list[dict] = []
    for source in sorted(source_rows, key=lambda row: str(row["scene_id"])):
        for severity, parameters in SEVERITY.items():
            case = copy.deepcopy(source)
            scene = str(case["scene_id"])
            case.update(
                {
                    "schema_version": "kinofail.o9-severity-v4-schedule.v1",
                    "case_id": (
                        f"o9severityv4__{severity}__{scene}__"
                        f"{case['source_counterfactual_group_id']}"
                    ),
                    "severity_id": severity,
                    "source_rank_contract": "replicate_b_unused_by_severity_v3",
                    "actions": [
                        "continue",
                        "always_safe_halt",
                        "recover_as_O9_high_centering",
                    ],
                    "decision_mode": "first_recoverable_operator_event",
                    "operator_engagement_dwell_steps": 1,
                    "capability_band_parameters": copy.deepcopy(parameters),
                    "pairing": "one direct-contact checkpoint restored across three arms",
                    "outcome_selection_used": False,
                    "selection_used_model_predictions_or_action_outcomes": False,
                }
            )
            nuisance = {
                "profile_index": int(case["source_physical_nuisance"]["profile_index"]),
                "start_progress_m": 0.0,
                "start_lateral_offset_m": 0.0,
                "start_heading_offset_rad": 0.0,
                "forward_speed_mps": 0.08,
                "controller_target_lateral_offset_m": 0.0,
                "physics_seed": int(case["reset_seed"]),
                "pair_shared": True,
                "spawn_base_height_m": 0.43,
            }
            case["source_physical_nuisance"] = copy.deepcopy(nuisance)
            case["source_record"]["physical_nuisance"] = copy.deepcopy(nuisance)
            case["source_record"]["physical_realization"] = "pallet_edge"
            case["source_record"]["physics_parameters"] = copy.deepcopy(parameters)
            case["source_f35_decision"] = {
                "certificate_mode": "same_run_action_boundary",
                "decision_time_s": 0.0,
                "event_time_s": 0.0,
                "external_snapshot_replay_claimed": False,
                "sample_id": f"{case['case_id']}__action_boundary",
            }
            case["source_observation_replay_required"] = False
            case["source_is_f33_direct_o9"] = False
            cases.append(case)

    OUTPUT.mkdir(parents=True)
    schedule = OUTPUT / "schedule.jsonl"
    schedule.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in cases),
        encoding="utf-8",
    )
    registry = json.loads((SOURCE / "scene_registry.json").read_text(encoding="utf-8"))
    (OUTPUT / "scene_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    protocol = copy.deepcopy(source_protocol)
    protocol.update(
        {
            "schema_version": "kinofail.o9-severity-v4-protocol.v1",
            "protocol_id": PROTOCOL_ID,
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "formal_direct-contact_confirmation",
            "development_only": False,
            "integrity_mode": "path_and_count",
            "schedule": str(schedule),
            "scene_registry": str(OUTPUT / "scene_registry.json"),
            "collector": COLLECTOR,
            "horizon_steps": 1200,
            "counts": {
                "scenes": 12,
                "operators": 1,
                "severity_levels": 3,
                "physical_cases": 36,
                "action_arms": 3,
                "physical_episodes": 108,
                "unique_recovery_control_programs": 1,
            },
            "severity_levels": SEVERITY,
            "selection_contract": {
                "source_rank": "replicate_b unused by severity-v3",
                "reads_model_predictions": False,
                "reads_action_outcomes": False,
                "result_dependent_parameter_change": False,
            },
            "physical_validity_contract": {
                "geometry": "route-transverse pallet crossbar",
                "approach_speed_mps": 0.08,
                "spawn_base_height_m": 0.43,
                "required_semantics": "measured belly contact plus partial foot unloading",
                "semantic_attrition_retained": True,
            },
        }
    )
    for key in list(protocol):
        if key.endswith("_sha256"):
            protocol.pop(key)
    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"protocol": PROTOCOL_ID, "cases": len(cases)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
