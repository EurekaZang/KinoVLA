#!/usr/bin/env python3
"""Fail-closed preflight for frozen v37 emergency route-control development."""

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


def _locked(record: dict[str, Any]) -> bool:
    path = _resolve(record["path"])
    if not path.is_file() or _sha256(path) != record["sha256"]:
        return False
    value = _json(path) if "json_passed" in record or "json_sealed" in record else {}
    return (
        ("json_passed" not in record or value.get("passed") is record["json_passed"])
        and ("json_sealed" not in record or value.get("sealed") is record["json_sealed"])
    )


def _v37_runtime_delta_is_exact(
    source: dict[str, Any], candidate: dict[str, Any], collector_sha256: str
) -> bool:
    additions = {
        "emergency_cross_track_gain_per_s": 2.0,
        "emergency_lateral_velocity_damping": 1.0,
        "emergency_lateral_limit_mps": 0.4,
    }
    if set(candidate) != set(source) | set(additions):
        return False
    for key, value in source.items():
        expected = collector_sha256 if key == "collector_sha256" else value
        if candidate.get(key) != expected:
            return False
    return all(candidate.get(key) == value for key, value in additions.items())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _json(config_path)
    cohort_path = _resolve(config["cohort_postrun"]["path"])
    cohort = _json(cohort_path)
    runtime_source_path = _resolve(config["runtime_source"]["path"])
    runtime_source = _json(runtime_source_path)
    cases = config.get("cases", [])
    case_ids = [case.get("scene_id") for case in cases]
    families = sorted({case.get("room_family") for case in cases})
    excluded = set(config.get("scene_exclusions", []))
    evidence = config.get("evidence_policy", {})
    execution = config.get("execution_policy", {})
    collector_sha256 = config.get("runtime", {}).get("collector_sha256", "")
    checks: dict[str, bool] = {
        "supported_schema": config.get("schema_version")
        == "kinofail.o4-action-consequence-v37-development-freeze.v1",
        "frozen_before_first_execution": config.get("freeze_status")
        == "frozen_before_first_v37_development_execution",
        "strict_development_quarantine": evidence.get("development_only") is True
        and evidence.get("counts_as_realistic_o4_operator_confirmation") is False
        and evidence.get("counts_as_a0_a7_evidence") is False
        and evidence.get("a8_in_scope") is False
        and evidence.get("old_a0_a7_metrics_may_be_mixed") is False
        and evidence.get("realistic_a0_a7_readiness") == "0/8",
        "cohort_passed_sealed_and_frozen": cohort.get("passed") is True
        and cohort.get("sealed") is True
        and cohort.get("cohort_frozen") is True,
        "cohort_hash_matches": cohort_path.is_file()
        and _sha256(cohort_path) == config["cohort_postrun"]["sha256"],
        "whole_admitted_cohort_bound": case_ids == cohort.get("admitted_scene_ids"),
        "cohort_size_and_family_coverage": len(cases) == cohort.get("admitted") == 6
        and families == cohort.get("admitted_families")
        and len(families) >= 4,
        "all_prior_o4_scenes_excluded": bool(excluded)
        and not excluded.intersection(case_ids),
        "one_attempt_each_no_selection": bool(cases)
        and execution.get("one_attempt_per_scene") is True
        and execution.get("execute_every_case_regardless_of_prior_outcome") is True
        and execution.get("scene_removal_after_outcome_forbidden") is True
        and execution.get("parameter_changes_after_preflight_forbidden") is True
        and execution.get("all_attempts_retained_and_sealed") is True,
        "v37_runtime_delta_exact": runtime_source_path.is_file()
        and _sha256(runtime_source_path) == config["runtime_source"]["sha256"]
        and _v37_runtime_delta_is_exact(
            runtime_source.get("runtime", {}), config.get("runtime", {}), collector_sha256
        ),
        "architecture_delta_exact": config.get("architecture_delta") == {
            "emergency_cross_track_gain_per_s": {"v35": None, "v37": 2.0},
            "emergency_lateral_velocity_damping": {"v35": None, "v37": 1.0},
            "emergency_lateral_limit_mps": {"v35": None, "v37": 0.4},
            "activation": "only_after_adaptive_backstep_guard_latches",
            "inputs": "route_state_only_no_scene_identity",
            "nominal_controller_unchanged": True,
        },
        "route_and_visual_contracts_inherited": config.get("phase_aware_route_contract")
        == runtime_source.get("phase_aware_route_contract")
        and config.get("visual_qa_contract") == runtime_source.get("visual_qa_contract"),
        "operator_output_root_unused": not _resolve(config["operator_output_root"]).exists(),
        "locked_files_match": bool(config.get("locked_files"))
        and all(_locked(item) for item in config["locked_files"]),
    }
    case_rows = []
    for case in cases:
        episode = _resolve(case["episode_usd"])
        output = _resolve(case["output"])
        prerequisites = case.get("prerequisites", {})
        prerequisite_checks = {name: _locked(record) for name, record in prerequisites.items()}
        compiled_record = prerequisites.get("terrain_compiled_audit", {})
        compiled = _json(_resolve(compiled_record.get("path", "__missing__")))
        episode_hash_bound = episode.is_file() and compiled.get("files", {}).get(episode.name) == _sha256(episode)
        ready = episode_hash_bound and not output.exists() and bool(prerequisite_checks) and all(prerequisite_checks.values())
        checks[f"{case['scene_id']}_ready"] = ready
        case_rows.append({
            "scene_id": case["scene_id"],
            "episode_usd": str(episode),
            "episode_hash_bound_to_compiled_audit": episode_hash_bound,
            "output": str(output),
            "output_absent": not output.exists(),
            "prerequisites": prerequisite_checks,
            "passed": ready,
        })
    passed = all(checks.values())
    result = {
        "schema_version": "kinofail.o4-action-consequence-v37-development-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": passed,
        "execution_authorized": passed,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "cohort_postrun": str(cohort_path),
        "cohort_postrun_sha256": _sha256(cohort_path) if cohort_path.is_file() else None,
        "checks": checks,
        "cases": case_rows,
        "development_only": True,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
    }
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "passed": passed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
