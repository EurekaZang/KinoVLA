#!/usr/bin/env python3
"""Seal and diagnose the frozen O4 v16 realistic paired-recalibration batch."""

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


def _locked(record: dict[str, Any]) -> dict[str, Any]:
    path = _resolve(record["path"])
    actual = _sha256(path) if path.is_file() else None
    json_ok = True
    if path.is_file() and "json_passed" in record:
        json_ok = _json(path).get("passed") is record["json_passed"]
    passed = path.is_file() and actual == record["sha256"] and json_ok
    return {
        "path": str(path),
        "expected_sha256": record["sha256"],
        "actual_sha256": actual,
        "hash_matches": actual == record["sha256"],
        "json_passed_matches": json_ok,
        "passed": passed,
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
    required = config["acceptance"]["required_manifest_checks"]
    rows = []
    for case in config["cases"]:
        path = _resolve(case["output"]) / "pair_manifest.json"
        manifest = _json(path)
        checks = {name: manifest.get("checks", {}).get(name) is True for name in required}
        nominal = manifest.get("takes", {}).get("nominal", {})
        anomaly = manifest.get("takes", {}).get("o4_adhesion", {})
        consequence = manifest.get("paired_consequence", {})
        identity = {
            "manifest_exists": path.is_file(),
            "scene_matches": manifest.get("scene_id") == case["scene_id"],
            "seed_matches": manifest.get("seed") == case["runtime_seed"],
            "material_matches": manifest.get("material_id") == case["material_id"],
            "camera_matches": manifest.get("camera_profile")
            == config["runtime"]["camera_profile"],
            "development_only": manifest.get("development_only") is True,
            "not_a0_a7_evidence": manifest.get("counts_as_a0_a7_evidence") is False,
        }
        mechanism = {
            "nominal_completed_without_fall": nominal.get("fell") is False
            and nominal.get("steps") == nominal.get("fixed_horizon_steps"),
            "adhesion_attached": int(anomaly.get("attachment_events", 0)) >= 1,
            "adhesion_force_reached": float(anomaly.get("peak_adhesion_force_n", -1.0))
            >= 10.0,
            "posture_effect_observed": consequence.get("available") is True
            and (
                float(consequence.get("max_tilt_increase_rad", -1.0)) >= 0.10
                or float(consequence.get("min_base_height_drop_m", -1.0)) >= 0.03
            ),
            "anomaly_terminal_fall": anomaly.get("fell") is True,
            "fall_precedes_frozen_peel_phase": anomaly.get("fell") is True
            and int(anomaly.get("first_fall_step", 10**9))
            < round(float(config["runtime"]["forward_s"]) / 0.02),
        }
        prerequisites = {
            name: _locked(record) for name, record in case["prerequisites"].items()
        }
        rows.append(
            {
                "scene_id": case["scene_id"],
                "room_family": case["room_family"],
                "material_id": case["material_id"],
                "manifest": {
                    "path": str(path),
                    "sha256": _sha256(path) if path.is_file() else None,
                    "reported_passed": manifest.get("passed"),
                },
                "required_acceptance_checks": checks,
                "identity_checks": identity,
                "mechanism_diagnostics": mechanism,
                "failed_acceptance_checks": [name for name, value in checks.items() if not value],
                "prerequisites": prerequisites,
                "accepted": all(checks.values()) and all(identity.values()),
            }
        )

    accepted_count = sum(row["accepted"] for row in rows)
    mechanism_count = sum(all(row["mechanism_diagnostics"].values()) for row in rows)
    integrity_checks = {
        "preflight_passed": preflight.get("passed") is True,
        "preflight_config_hash_matches": preflight.get("config_sha256") == _sha256(config_path),
        "locked_files_unchanged": all(_locked(item)["passed"] for item in config["locked_files"]),
        "all_three_frozen_cases_present": len(rows) == 3,
        "all_three_identity_checks_pass": all(all(row["identity_checks"].values()) for row in rows),
        "all_prerequisites_unchanged": all(
            all(item["passed"] for item in row["prerequisites"].values()) for row in rows
        ),
    }
    audit = {
        "schema_version": "kinofail.o4-realistic-pair-recalibration-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "preflight": str(preflight_path),
        "preflight_sha256": _sha256(preflight_path),
        "integrity_checks": integrity_checks,
        "integrity_passed": all(integrity_checks.values()),
        "primary_target": "all three frozen O4 pairs pass the shared recoverable-peel contract",
        "primary_target_passed": accepted_count == 3,
        "accepted_pair_count": accepted_count,
        "pair_count": len(rows),
        "mechanism_diagnostic_pass_count": mechanism_count,
        "cases": rows,
        "passed": accepted_count == 3 and all(integrity_checks.values()),
        "sealed": all(integrity_checks.values()),
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "diagnosis": (
            "All three severe 35 N O4 interventions attached and caused a terminal posture "
            "failure before the frozen peel phase. The shared recoverable-peel endpoint is "
            "therefore structurally incompatible with this severe lane; the 0/3 result is retained."
        ),
        "next_gate": (
            "Calibrate on train-only scenes, then preregister separate moderate/recoverable and "
            "severe/terminal O4 endpoint classes before any new held-out execution."
        ),
    }
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "passed": audit["passed"],
                "sealed": audit["sealed"],
                "accepted_pair_count": accepted_count,
                "mechanism_diagnostic_pass_count": mechanism_count,
            },
            indent=2,
        )
    )
    return 0 if audit["sealed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
