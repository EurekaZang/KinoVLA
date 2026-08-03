#!/usr/bin/env python3
"""Seal every v31 O4 result without selecting scenes by outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_o4_action_consequence_v27_development_postrun import (  # noqa: E402
    _audit_case as _audit_v27_case,
    _locked,
)


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
    output_root = _resolve(config["operator_output_root"])
    runner_path = output_root / "runner_audit.json"
    runner = _json(runner_path)
    launchers = {row.get("scene_id"): row for row in runner.get("cases", [])}

    cases = []
    for frozen_case in config["cases"]:
        case = _audit_v27_case(frozen_case, config["runtime"])
        launcher = launchers.get(frozen_case["scene_id"], {})
        manifest_path = Path(case["manifest"])
        manifest = _json(manifest_path)
        launcher_checks = {
            "launcher_terminal": launcher.get("collector_terminal_returncode") is True,
            "launcher_manifest_hash_matches": manifest_path.is_file()
            and launcher.get("manifest_sha256") == _sha256(manifest_path),
        }
        case["integrity_checks"].update(launcher_checks)
        case["integrity_passed"] = all(case["integrity_checks"].values())
        case["sealed"] = case["integrity_passed"]
        case["passed"] = case["integrity_passed"] and case["outcome_passed"]
        take = manifest.get("takes", {}).get("continue", {})
        case["metrics"].update(
            {
                "decision_trigger_reason": take.get("decision_trigger_reason"),
                "predecision_step": take.get("predecision_step"),
                "action_branch_step": take.get("action_branch_step"),
            }
        )
        cases.append(case)

    frozen_ids = [case["scene_id"] for case in config["cases"]]
    integrity_checks = {
        "preflight_passed_and_bound": preflight.get("passed") is True
        and preflight.get("config_sha256") == _sha256(config_path),
        "locked_files_unchanged": all(_locked(item) for item in config["locked_files"]),
        "runner_bound_to_config_and_preflight": runner.get("config_sha256")
        == _sha256(config_path)
        and runner.get("preflight_sha256") == _sha256(preflight_path),
        "every_frozen_case_attempted_in_order": runner.get("attempted_scene_ids")
        == frozen_ids
        and runner.get("all_cases_attempted") is True,
        "every_case_terminal_with_manifest": runner.get("all_cases_terminal") is True
        and runner.get("all_manifests_present") is True,
        "all_case_integrity_passed": all(case["integrity_passed"] for case in cases),
    }
    integrity_passed = all(integrity_checks.values())
    outcome_passed = all(case["outcome_passed"] for case in cases)
    passed_count = sum(case["passed"] for case in cases)
    result = {
        "schema_version": "kinofail.o4-action-consequence-v31-heldout-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "preflight": str(preflight_path),
        "preflight_sha256": _sha256(preflight_path),
        "runner_audit": str(runner_path),
        "runner_audit_sha256": _sha256(runner_path) if runner_path.is_file() else None,
        "integrity_checks": integrity_checks,
        "integrity_passed": integrity_passed,
        "outcome_passed": outcome_passed,
        "passed": integrity_passed and outcome_passed,
        "sealed": integrity_passed,
        "total_scenes": len(cases),
        "passed_scenes": passed_count,
        "room_families": sorted({case["room_family"] for case in cases}),
        "cases": cases,
        "counts_as_realistic_o4_operator_confirmation": integrity_passed,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "interpretation": (
            "This seals the never-calibrated realistic O4 confirmation cohort. "
            "It does not complete any A0-A7 experiment and cannot be mixed with legacy metrics."
        ),
        "next_gate": (
            "Freeze and confirm O5-O11 on realistic scenes, then registry-bind the corpus, "
            "retrain/infer, and rerun A0-A7 independently."
        ),
    }
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite postrun: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "passed": result["passed"],
                "sealed": result["sealed"],
                "passed_scenes": passed_count,
                "total_scenes": len(cases),
            },
            indent=2,
        )
    )
    return 0 if result["sealed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
