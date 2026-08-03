#!/usr/bin/env python3
"""Fail-closed preflight for the v31 phase-conditioned O4 confirmation."""

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
    output_root = _resolve(config["operator_output_root"])

    evidence = config.get("evidence_policy", {})
    execution = config.get("execution_policy", {})
    checks: dict[str, bool] = {
        "supported_schema": config.get("schema_version")
        == "kinofail.o4-action-consequence-v31-heldout-confirmation-freeze.v1",
        "frozen_before_first_execution": config.get("freeze_status")
        == "frozen_before_first_heldout_o4_execution",
        "strict_evidence_quarantine": evidence.get("counts_as_a0_a7_evidence") is False
        and evidence.get("a8_in_scope") is False
        and evidence.get("old_a0_a7_metrics_may_be_mixed") is False
        and evidence.get("realistic_a0_a7_readiness") == "0/8",
        "v30_cohort_passed_and_sealed": cohort.get("passed") is True
        and cohort.get("sealed") is True
        and cohort.get("cohort_frozen") is True
        and cohort.get("v27_heldout_confirmation_authorized") is True,
        "v30_cohort_hash_matches": cohort_path.is_file()
        and _sha256(cohort_path) == config["cohort_postrun"]["sha256"],
        "whole_admitted_cohort_bound": case_ids == cohort.get("admitted_scene_ids"),
        "cohort_size_and_family_coverage": len(cases) == cohort.get("admitted") == 6
        and families == cohort.get("admitted_families")
        and len(families) == 4,
        "development_and_prior_heldout_scenes_excluded": bool(excluded)
        and not excluded.intersection(case_ids),
        "one_attempt_each_no_selection": bool(cases)
        and execution.get("one_attempt_per_scene") is True
        and execution.get("execute_every_case_regardless_of_prior_outcome") is True
        and execution.get("scene_removal_after_outcome_forbidden") is True
        and execution.get("parameter_changes_after_preflight_forbidden") is True,
        "v27_runtime_exactly_inherited": runtime_source_path.is_file()
        and _sha256(runtime_source_path) == config["runtime_source"]["sha256"]
        and config.get("runtime") == runtime_source.get("runtime"),
        "operator_output_root_unused": not output_root.exists(),
        "locked_files_match": bool(config.get("locked_files"))
        and all(_locked(item) for item in config["locked_files"]),
    }
    case_rows = []
    for case in cases:
        episode = _resolve(case["episode_usd"])
        output = _resolve(case["output"])
        prerequisites = case.get("prerequisites", {})
        prerequisite_checks = {
            name: _locked(record) for name, record in prerequisites.items()
        }
        compiled_record = prerequisites.get("terrain_compiled_audit", {})
        compiled_path = _resolve(compiled_record.get("path", "__missing__"))
        compiled = _json(compiled_path)
        episode_hash_bound = (
            episode.is_file()
            and compiled.get("files", {}).get(episode.name) == _sha256(episode)
        )
        case_passed = (
            episode_hash_bound
            and not output.exists()
            and bool(prerequisite_checks)
            and all(prerequisite_checks.values())
        )
        checks[f"{case['scene_id']}_ready"] = case_passed
        case_rows.append(
            {
                "scene_id": case["scene_id"],
                "episode_usd": str(episode),
                "episode_hash_bound_to_compiled_audit": episode_hash_bound,
                "output": str(output),
                "output_absent": not output.exists(),
                "prerequisites": prerequisite_checks,
                "passed": case_passed,
            }
        )

    passed = all(checks.values())
    result = {
        "schema_version": "kinofail.o4-action-consequence-v31-heldout-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": passed,
        "execution_authorized": passed,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "cohort_postrun": str(cohort_path),
        "cohort_postrun_sha256": _sha256(cohort_path) if cohort_path.is_file() else None,
        "checks": checks,
        "cases": case_rows,
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
