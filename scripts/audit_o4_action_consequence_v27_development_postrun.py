#!/usr/bin/env python3
"""Seal the frozen v27 phase-conditioned O4 development batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return rows if all(isinstance(row, dict) for row in rows) else []


def _canonical_rows_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _locked(record: dict[str, Any]) -> bool:
    path = _resolve(record["path"])
    if not path.is_file() or _sha256(path) != record["sha256"]:
        return False
    value = _json(path) if "json_passed" in record or "json_sealed" in record else {}
    return (
        ("json_passed" not in record or value.get("passed") is record["json_passed"])
        and ("json_sealed" not in record or value.get("sealed") is record["json_sealed"])
    )


def _audit_case(case: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    output = _resolve(case["output"])
    manifest_path = output / "pair_manifest.json"
    manifest = _json(manifest_path)
    summaries: dict[str, dict[str, Any]] = {}
    rows_by_take: dict[str, list[dict[str, Any]]] = {}
    artifacts: dict[str, bool] = {}
    for take in ("continue", "backstep"):
        take_dir = output / take
        summary_path = take_dir / "summary.json"
        summary = _json(summary_path)
        telemetry_path = Path(summary.get("telemetry", ""))
        if telemetry_path and not telemetry_path.is_absolute():
            telemetry_path = ROOT / telemetry_path
        rows = _jsonl(telemetry_path)
        summaries[take] = summary
        rows_by_take[take] = rows
        artifacts[f"{take}_summary_matches_manifest"] = bool(summary) and summary == manifest.get("takes", {}).get(take)
        artifacts[f"{take}_telemetry_hash_matches"] = telemetry_path.is_file() and _sha256(telemetry_path) == summary.get("telemetry_sha256")
        artifacts[f"{take}_step_count_matches"] = bool(rows) and len(rows) == summary.get("steps")
        for capture in summary.get("captures", []):
            image = take_dir / capture["image"]
            artifacts[f"{take}_{capture['label']}_image_hash_matches"] = image.is_file() and _sha256(image) == capture.get("image_sha256")

    complete = bool(manifest) and all(summaries.values()) and all(rows_by_take.values())
    outcome: dict[str, bool] = {}
    if complete:
        decision_step = int(summaries["continue"]["predecision_step"])
        branch_step = decision_step + 1
        continue_rows = rows_by_take["continue"]
        backstep_rows = rows_by_take["backstep"]
        continue_pre = continue_rows[: decision_step + 1]
        backstep_pre = backstep_rows[: decision_step + 1]
        pre_hash = _canonical_rows_hash(continue_pre)
        outcome = {
            "collector_reported_all_checks_passed": manifest.get("passed") is True and bool(manifest.get("checks")) and all(manifest["checks"].values()),
            "predecision_rows_exactly_equal": continue_pre == backstep_pre,
            "predecision_hash_recomputed": pre_hash == _canonical_rows_hash(backstep_pre) == summaries["continue"].get("predecision_rows_sha256") == summaries["backstep"].get("predecision_rows_sha256"),
            "branch_step_is_exact": summaries["continue"].get("action_branch_step") == summaries["backstep"].get("action_branch_step") == branch_step,
            "actions_diverge_only_after_decision": len(continue_rows) > branch_step and len(backstep_rows) > branch_step and continue_rows[branch_step]["action_phase"] == "matched_forward" and backstep_rows[branch_step]["action_phase"] == "backstep_clearance",
            "continue_falls_after_branch": summaries["continue"].get("fell") is True and int(summaries["continue"].get("first_fall_step", -1)) > branch_step,
            "backstep_completes_fixed_horizon": summaries["backstep"].get("fell") is False and summaries["backstep"].get("steps") == runtime["horizon_steps"],
            "backstep_has_physical_peel": summaries["backstep"].get("postdecision_peel_events", 0) >= 1,
            "backstep_closed_loop_exit_complete": summaries["backstep"].get("recovery_complete") is True,
            "backstep_retreat_distance_passes": summaries["backstep"].get("backward_recovery_distance_m", -1.0) >= runtime["minimum_recovery_distance_m"],
            "both_routes_contained": all(summaries[take].get("maximum_absolute_route_lateral_offset_m", float("inf")) <= runtime["maximum_route_deviation_m"] for take in ("continue", "backstep")),
            "command_agnostic_physics_readback": all(row.get("adhesion", {}).get("mode") == "command_agnostic_contact_surface_adhesion_v3" and row.get("adhesion", {}).get("command_conditioned_release") is False for take in ("continue", "backstep") for row in rows_by_take[take]),
        }
    provenance = manifest.get("provenance", {})
    decision = manifest.get("decision_contract", {})
    integrity = {
        "manifest_identity_matches": manifest.get("schema_version") == "kinofail.o4-action-consequence.v27" and manifest.get("scene_id") == case["scene_id"] and manifest.get("room_family") == case["room_family"] and manifest.get("material_id") == case["material_id"] and manifest.get("seed") == case["runtime_seed"],
        "operator_identity_matches": manifest.get("operator", {}).get("id") == "O4" and manifest.get("operator", {}).get("subtype") == "adhesive_contact" and manifest.get("operator", {}).get("mode") == "command_agnostic_contact_surface_adhesion_v3",
        "collector_quarantine_preserved": manifest.get("development_only") is True and manifest.get("counts_as_a0_a7_evidence") is False and manifest.get("realistic_a0_a7_readiness") == "0/8",
        "phase_conditioned_contract_matches": decision.get("minimum_decision_dwell_steps") == runtime["minimum_decision_dwell_steps"] and decision.get("maximum_decision_dwell_steps") == runtime["maximum_decision_dwell_steps"] and decision.get("decision_tilt_urgency_rad") == runtime["decision_tilt_urgency_rad"] and decision.get("decision_height_drop_urgency_m") == runtime["decision_height_drop_urgency_m"] and decision.get("decision_urgency_dwell_steps") == runtime["decision_urgency_dwell_steps"],
        "collector_hash_matches_v27": provenance.get("collector_sha256") == runtime["collector_sha256"],
        "physics_hashes_match_v21": provenance.get("surface_interface_sha256") == runtime["surface_interface_sha256"] and provenance.get("surface_model_sha256") == runtime["surface_model_sha256"] and provenance.get("backend_adapter_sha256") == runtime["backend_adapter_sha256"] and provenance.get("parent_backend_adapter_sha256") == runtime["parent_backend_adapter_sha256"],
        "all_artifacts_hash_verified": bool(artifacts) and all(artifacts.values()),
    }
    integrity_passed = all(integrity.values())
    outcome_passed = bool(outcome) and all(outcome.values())
    return {
        "scene_id": case["scene_id"],
        "room_family": case["room_family"],
        "material_id": case["material_id"],
        "runtime_seed": case["runtime_seed"],
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
        "artifact_checks": artifacts,
        "integrity_checks": integrity,
        "outcome_checks": outcome,
        "integrity_passed": integrity_passed,
        "outcome_passed": outcome_passed,
        "sealed": integrity_passed,
        "passed": integrity_passed and outcome_passed,
        "metrics": {
            "continue_first_fall_step": summaries.get("continue", {}).get("first_fall_step"),
            "backstep_recovery_distance_m": summaries.get("backstep", {}).get("backward_recovery_distance_m"),
            "backstep_postdecision_peel_events": summaries.get("backstep", {}).get("postdecision_peel_events"),
            "continue_max_route_deviation_m": summaries.get("continue", {}).get("maximum_absolute_route_lateral_offset_m"),
            "backstep_max_route_deviation_m": summaries.get("backstep", {}).get("maximum_absolute_route_lateral_offset_m"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    preflight_path = args.preflight.resolve()
    config = _json(config_path)
    preflight = _json(preflight_path)
    cases = [_audit_case(case, config["runtime"]) for case in config["cases"]]
    integrity_checks = {
        "preflight_passed_and_bound": preflight.get("passed") is True and preflight.get("config_sha256") == _sha256(config_path),
        "locked_files_unchanged": all(_locked(item) for item in config["locked_files"]),
        "every_frozen_case_has_manifest": all(_resolve(case["output"]).joinpath("pair_manifest.json").is_file() for case in config["cases"]),
        "all_case_integrity_passed": all(case["integrity_passed"] for case in cases),
    }
    integrity_passed = all(integrity_checks.values())
    outcome_passed = all(case["outcome_passed"] for case in cases)
    passed_count = sum(case["passed"] for case in cases)
    result = {
        "schema_version": "kinofail.o4-action-consequence-v27-development-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "preflight": str(preflight_path),
        "preflight_sha256": _sha256(preflight_path),
        "integrity_checks": integrity_checks,
        "integrity_passed": integrity_passed,
        "outcome_passed": outcome_passed,
        "passed": integrity_passed and outcome_passed,
        "sealed": integrity_passed,
        "total_scenes": len(cases),
        "passed_scenes": passed_count,
        "room_families": sorted({case["room_family"] for case in cases}),
        "cases": cases,
        "development_only": True,
        "counts_as_realistic_o4_operator_confirmation": False,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "interpretation": "This seals exposed-scene development only. A passing result authorizes a newly generated held-out cohort; it is not A0-A7 evidence.",
        "next_gate": "Freeze a new never-calibrated realistic scene cohort and execute the v27 contract once per admitted scene without parameter changes.",
    }
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite postrun: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "passed": result["passed"], "sealed": result["sealed"], "passed_scenes": passed_count, "total_scenes": len(cases)}, indent=2))
    return 0 if result["sealed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
