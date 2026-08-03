#!/usr/bin/env python3
"""Freeze the realized valid-case subset without consulting model outcomes.

The preregistered design contains 1,500 cases in each conflict direction.
Acquisition failures are handled at the physical-case level: a case is valid
only when all six model records can be constructed.  This script performs
artifact/provenance validation, records every excluded planned case, enforces
the frozen five-percent attrition ceiling, and writes the T3 subset consumed
by the feature builder.  It never loads a model or any prediction artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.build_kinofail_confirmatory_c2_base_features_v1 import (  # noqa: E402
    _load as _load_feature_implementation,
)


PLANNED_CASES_PER_CELL = 1_500
MINIMUM_VALID_CASES_PER_CELL = 1_425


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(not isinstance(value, dict) for value in values):
        raise TypeError(path)
    return values


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _validate_t2(
    *,
    schedule_path: Path,
    features_dir: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    schedule = _jsonl(schedule_path)
    planned = {str(row["case_id"]): row for row in schedule}
    if len(schedule) != PLANNED_CASES_PER_CELL or len(planned) != len(schedule):
        raise RuntimeError("T2 schedule is not the frozen 1,500-case design")
    manifest_path = features_dir / "feature_manifest.json"
    records_path = features_dir / "records.jsonl"
    features_path = features_dir / "features.npz"
    manifest = _json(manifest_path)
    if (
        manifest.get("passed") is not True
        or manifest.get("output_sha256", {}).get("records")
        != _sha256(records_path)
        or manifest.get("output_sha256", {}).get("features")
        != _sha256(features_path)
        or manifest.get("t2_schedule", {}).get("sha256")
        != _sha256(schedule_path)
    ):
        raise RuntimeError("invalid merged T2 feature artifact")
    records = _jsonl(records_path)
    records_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        records_by_case.setdefault(str(row["case_id"]), []).append(row)
    valid = []
    for case_id, rows in sorted(records_by_case.items()):
        expected = planned.get(case_id)
        if expected is None:
            raise RuntimeError(f"unplanned T2 case: {case_id}")
        expected_ids = {
            f"{case_id}__{operator}__{view}"
            for operator in ("O2_compliance", "O4_tether")
            for view in ("primary", "swap_01", "swap_02")
        }
        if (
            {str(row["sample_id"]) for row in rows} == expected_ids
            and len(rows) == 6
            and {str(row["scene_cluster"]) for row in rows}
            == {str(expected["scene_cluster"])}
            and {str(row["cluster_material"]) for row in rows}
            == {str(expected["cluster_material"])}
        ):
            valid.append(case_id)
    invalid = [
        {
            "cell": "T2_vision_decisive",
            "case_id": case_id,
            "scene_cluster": str(row["scene_cluster"]),
            "cluster_material": str(row["cluster_material"]),
            "reason_code": (
                "missing_or_incomplete_six_record_feature_case"
                if case_id not in valid
                else "valid"
            ),
        }
        for case_id, row in sorted(planned.items())
        if case_id not in valid
    ]
    return valid, invalid


def _validate_t3(
    *,
    design_dir: Path,
    corpus: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    list[dict[str, Any]],
]:
    schedule_path = design_dir / "schedule.jsonl"
    case_path = design_dir / "case_schedule.jsonl"
    audit_path = design_dir / "audit.json"
    audit = _json(audit_path)
    if (
        audit.get("passed") is not True
        or audit.get("schedule_sha256") != _sha256(schedule_path)
        or audit.get("case_schedule_sha256") != _sha256(case_path)
    ):
        raise RuntimeError("invalid frozen T3 design")
    schedule = _jsonl(schedule_path)
    cases = _jsonl(case_path)
    if (
        len(cases) != PLANNED_CASES_PER_CELL
        or len({str(row["case_id"]) for row in cases}) != len(cases)
        or len(schedule) != 4 * len(cases)
    ):
        raise RuntimeError("T3 design is not 1,500 cases/four episodes per case")
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in schedule:
        by_group.setdefault(str(row["counterfactual_group_id"]), []).append(row)

    implementation = _load_feature_implementation()
    valid_cases: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        try:
            for key in (
                "o7_source_physics_group_id",
                "o8_source_physics_group_id",
            ):
                implementation._validated_anomaly(
                    by_group,
                    str(case[key]),
                    corpus,
                )
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            invalid.append(
                {
                    "cell": "T3_proprio_decisive",
                    "case_id": case_id,
                    "scene_cluster": str(case["scene_cluster"]),
                    "cluster_material": str(case["cluster_material"]),
                    "reason_code": "runtime_artifact_validation_failed",
                    "exception_type": type(exc).__name__,
                    "diagnostic": str(exc)[:240],
                }
            )
        else:
            valid_cases.append(case)
    valid_ids = {str(case["case_id"]) for case in valid_cases}
    valid_groups = {
        str(case[key])
        for case in valid_cases
        for key in ("o7_source_physics_group_id", "o8_source_physics_group_id")
    }
    valid_schedule = [
        row
        for row in schedule
        if str(row["counterfactual_group_id"]) in valid_groups
    ]
    if (
        len(valid_schedule) != 4 * len(valid_cases)
        or len(valid_ids) != len(valid_cases)
    ):
        raise RuntimeError("T3 valid-case filtering lost physical episodes")
    return valid_schedule, valid_cases, sorted(valid_ids), invalid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2-schedule", type=Path, required=True)
    parser.add_argument("--t2-features", type=Path, required=True)
    parser.add_argument("--t3-design", type=Path, required=True)
    parser.add_argument("--t3-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    t2_schedule = args.t2_schedule.resolve()
    t2_features = args.t2_features.resolve()
    t3_design = args.t3_design.resolve()
    t3_corpus = args.t3_corpus.resolve()

    valid_t2, invalid_t2 = _validate_t2(
        schedule_path=t2_schedule,
        features_dir=t2_features,
    )
    t3_schedule, t3_cases, valid_t3, invalid_t3 = _validate_t3(
        design_dir=t3_design,
        corpus=t3_corpus,
    )
    counts = {
        "T2_vision_decisive": {
            "planned_cases": PLANNED_CASES_PER_CELL,
            "valid_cases": len(valid_t2),
            "invalid_cases": len(invalid_t2),
            "attrition_rate": len(invalid_t2) / PLANNED_CASES_PER_CELL,
        },
        "T3_proprio_decisive": {
            "planned_cases": PLANNED_CASES_PER_CELL,
            "valid_cases": len(valid_t3),
            "invalid_cases": len(invalid_t3),
            "attrition_rate": len(invalid_t3) / PLANNED_CASES_PER_CELL,
        },
    }
    if (
        len(valid_t2) < MINIMUM_VALID_CASES_PER_CELL
        or len(valid_t3) < MINIMUM_VALID_CASES_PER_CELL
    ):
        raise RuntimeError(f"frozen five-percent attrition gate failed: {counts}")

    output.mkdir(parents=True, exist_ok=False)
    schedule_out = output / "schedule.jsonl"
    cases_out = output / "case_schedule.jsonl"
    attrition_out = output / "attrition_ledger.jsonl"
    _write_jsonl(schedule_out, t3_schedule)
    _write_jsonl(cases_out, t3_cases)
    _write_jsonl(
        attrition_out,
        sorted(
            invalid_t2 + invalid_t3,
            key=lambda row: (str(row["cell"]), str(row["case_id"])),
        ),
    )
    audit = {
        "schema_version": "kinofail.unified-confirmatory-valid-conflict-design.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "selection_uses_model_predictions": False,
        "selection_uses_outcome_strength": False,
        "case_level_complete_case_rule": True,
        "maximum_attrition_rate": 0.05,
        "counts": counts,
        "schedule_sha256": _sha256(schedule_out),
        "case_schedule_sha256": _sha256(cases_out),
        "attrition_ledger_sha256": _sha256(attrition_out),
        "source_sha256": {
            "t2_schedule": _sha256(t2_schedule),
            "t2_feature_manifest": _sha256(
                t2_features / "feature_manifest.json"
            ),
            "t3_schedule": _sha256(t3_design / "schedule.jsonl"),
            "t3_case_schedule": _sha256(t3_design / "case_schedule.jsonl"),
            "t3_design_audit": _sha256(t3_design / "audit.json"),
        },
    }
    _write_json(output / "audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
