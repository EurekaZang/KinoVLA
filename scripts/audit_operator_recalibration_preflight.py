#!/usr/bin/env python3
"""Fail-closed preflight for frozen realistic-operator recalibration batches."""

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


def _check_locked(record: dict[str, Any]) -> dict[str, Any]:
    path = _resolve(record["path"])
    exists = path.is_file()
    actual_hash = _sha256(path) if exists else None
    json_passed = None
    if exists and "json_passed" in record:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            json_passed = payload.get("passed") is record["json_passed"]
        except (OSError, json.JSONDecodeError, AttributeError):
            json_passed = False
    passed = exists and actual_hash == record["sha256"]
    if json_passed is not None:
        passed = passed and json_passed
    return {
        "path": str(path),
        "exists": exists,
        "expected_sha256": record["sha256"],
        "actual_sha256": actual_hash,
        "hash_matches": actual_hash == record["sha256"],
        "json_passed_matches": json_passed,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    locked = [_check_locked(item) for item in config.get("locked_files", [])]
    cases = []
    for case in config.get("cases", []):
        if "prerequisites" in case:
            prerequisites = {
                name: _check_locked(record)
                for name, record in case["prerequisites"].items()
            }
        else:
            prerequisites = {
                name: _check_locked(case[name])
                for name in ("compiled_audit", "stack_admission", "nominal_manifest")
            }
        episode = _resolve(case["episode_usd"])
        output_values = case.get("outputs", {"default": case.get("output")})
        outputs = {
            name: _resolve(value)
            for name, value in output_values.items()
            if value is not None
        }
        outputs_absent = {name: not path.exists() for name, path in outputs.items()}
        passed = (
            episode.is_file()
            and bool(outputs)
            and all(outputs_absent.values())
            and all(item["passed"] for item in prerequisites.values())
        )
        cases.append(
            {
                "scene_id": case["scene_id"],
                "operator": case.get("operator"),
                "episode_usd": str(episode),
                "episode_exists": episode.is_file(),
                "outputs": {name: str(path) for name, path in outputs.items()},
                "outputs_absent_before_execution": outputs_absent,
                "prerequisites": prerequisites,
                "passed": passed,
            }
        )

    passed = (
        config.get("freeze_status") == "frozen_before_first_anomaly_execution"
        and config.get("evidence_policy", {}).get("counts_as_a0_a7_evidence") is False
        and bool(locked)
        and bool(cases)
        and all(item["passed"] for item in locked)
        and all(item["passed"] for item in cases)
    )
    audit = {
        "schema_version": "kinofail.operator-recalibration-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "locked_files": locked,
        "cases": cases,
        "passed": passed,
        "counts_as_a0_a7_evidence": False,
        "admission_state": (
            "recalibration_execution_authorized" if passed else "recalibration_execution_rejected"
        ),
    }
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "passed": passed}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
