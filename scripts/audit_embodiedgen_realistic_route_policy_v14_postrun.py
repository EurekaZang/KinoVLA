#!/usr/bin/env python3
"""Audit the successful Kitchen screen of realistic-route locomotion v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    base = (
        ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v14_screen_v1"
    )
    parser.add_argument("--preflight", type=Path, default=base / "preflight_audit.json")
    parser.add_argument(
        "--manifest", type=Path, default=base / "kitchen31/o2_nominal/lane_manifest.json"
    )
    parser.add_argument("--out", type=Path, default=base / "postrun_audit.json")
    args = parser.parse_args()
    preflight_path = args.preflight.resolve()
    manifest_path = args.manifest.resolve()
    preflight = _json(preflight_path)
    manifest = _json(manifest_path)
    policy_path = Path(manifest["provenance"]["locomotion_policy"])
    maximum_deviation = float(
        manifest["measurements"]["maximum_absolute_route_lateral_offset_m"]
    )
    checks = {
        "preflight_passed": preflight.get("passed") is True,
        "all_frozen_runtime_checks_passed": manifest.get("passed") is True
        and all(value is True for value in manifest.get("checks", {}).values()),
        "policy_hash_recorded_and_verified": manifest.get("provenance", {}).get(
            "locomotion_policy_sha256"
        )
        == "fb40283b56ec260fd14d8943684107557f2e49c3372c2fca7804f0879642b61f"
        and _sha256(policy_path) == manifest["provenance"]["locomotion_policy_sha256"],
        "strict_tracking_gate_passed": maximum_deviation <= 0.30,
        "route_completed_without_fall": manifest["checks"]["nominal_route_completed"]
        is True
        and manifest["checks"]["nominal_robot_not_fallen"] is True,
        "operator_region_coverage_passed": manifest["checks"][
            "lane_appropriate_operator_region_coverage"
        ]
        is True,
        "no_anomaly_lane": not list(manifest_path.parents[2].glob("**/o2_anomaly*/lane_manifest.json")),
        "not_registry_or_a0_a7_evidence": manifest.get("scene_registry_eligible") is False
        and manifest.get("counts_as_a0_a7_evidence") is False,
    }
    result = {
        "schema_version": "kinofail.embodiedgen-realistic-route-policy-v14-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "audit_passed": all(checks.values()),
        "kitchen_screen_passed": True,
        "policy_admitted_for_formal_collection": False,
        "checks": checks,
        "measurements": {
            "route_progress_m": manifest["measurements"]["final_route_progress_m"],
            "maximum_absolute_route_lateral_offset_m": maximum_deviation,
            "tracking_margin_m": 0.30 - maximum_deviation,
            "region_sample_count": manifest["measurements"]["region_sample_count"],
            "load_bearing_region_sample_count": manifest["measurements"][
                "load_bearing_region_sample_count"
            ],
            "fallen": manifest["measurements"]["fallen"],
        },
        "next_required_action": (
            "Replicate the identical frozen policy and runtime protocol on LivingRoom30 and "
            "DiningRoom26 development scenes before unseen confirmation or anomaly lanes."
        ),
        "evidence_boundary": {
            "counts_as_a0_a7_evidence": False,
            "scene_registry_eligible": False,
            "realistic_a0_a7_readiness": "0/8",
            "old_a0_a7_results_are_reference_only": True,
            "completion_requires_new_realistic_corpus_training_inference_and_a0_a7": True,
        },
        "inputs": {
            "preflight": {"path": str(preflight_path), "sha256": _sha256(preflight_path)},
            "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "telemetry": {
                "path": str(manifest_path.parent / manifest["telemetry"]),
                "sha256": _sha256(manifest_path.parent / manifest["telemetry"]),
            },
            "policy": {"path": str(policy_path), "sha256": _sha256(policy_path)},
        },
    }
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite postrun audit: {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(output),
                "audit_passed": result["audit_passed"],
                "kitchen_screen_passed": True,
                "policy_admitted_for_formal_collection": False,
                "tracking_margin_m": result["measurements"]["tracking_margin_m"],
            },
            indent=2,
        )
    )
    return 0 if result["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
