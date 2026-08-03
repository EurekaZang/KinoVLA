#!/usr/bin/env python3
"""Seal the three-case calibration of source-geometry preflight v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "kinofail.embodiedgen-source-preflight-validation-audit.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else REPO_ROOT / path


def _artifact_matches(row: dict[str, Any]) -> bool:
    path = _repo_path(str(row["path"]))
    return path.is_file() and _sha256(path) == row["sha256"]


def audit(plan_path: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    code_checks = {
        name: _artifact_matches(row) for name, row in plan["frozen_artifacts"].items()
    }
    cases = []
    for case in plan["validation_cases"]:
        manifest_path = _repo_path(case["source_manifest"]["path"])
        audit_path = _repo_path(case["preflight_audit"]["path"])
        manifest_hash_ok = _artifact_matches(case["source_manifest"])
        audit_hash_ok = _artifact_matches(case["preflight_audit"])
        record = json.loads(audit_path.read_text(encoding="utf-8")) if audit_hash_ok else {}
        checks = {
            "source_manifest_hash": manifest_hash_ok,
            "preflight_audit_hash": audit_hash_ok,
            "schema": record.get("schema_version")
            == "kinofail.embodiedgen-source-geometry-preflight.v1",
            "scene_id": record.get("scene_id") == case["scene_id"],
            "manifest_binding": record.get("source_manifest_sha256")
            == case["source_manifest"]["sha256"],
            "expected_passed": record.get("passed") is case["expected_passed"],
            "expected_primary_gate": record.get("primary_failed_gate")
            == case["expected_primary_failed_gate"],
        }
        cases.append(
            {
                "scene_id": case["scene_id"],
                "expected_role": case["expected_role"],
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    integrity = all(code_checks.values()) and all(case["passed"] for case in cases)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "sealed": True,
        "frozen_artifact_checks": code_checks,
        "validation_cases": cases,
        "audit_integrity_passed": integrity,
        "validation_passed": integrity,
        "interpretation": (
            "Cost-saving precompile gate calibrated on a transform failure, a low-complexity "
            "failure, and an admitted control; it is not benchmark confirmation evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit(Path(args.plan))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
