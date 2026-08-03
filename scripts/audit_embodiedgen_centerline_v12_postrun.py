#!/usr/bin/env python3
"""Audit the stopped v12 centerline O2 development matrix."""

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
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_centerline_v12_development_matrix_v1"
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
    batch_root = manifest_path.parents[2]
    checks = {
        "preflight_passed": preflight.get("passed") is True,
        "kitchen_was_first_scene": manifest.get("scene_id") == "indoor_kitchen_31",
        "centerline_tracking_stayed_within_budget": manifest.get("checks", {}).get(
            "full_lane_tracking_within_admitted_budget"
        )
        is True,
        "robot_fell_before_operator_region": manifest.get("measurements", {}).get("fallen")
        is True
        and manifest.get("checks", {}).get("operator_region_reached") is False,
        "matrix_scientifically_failed": manifest.get("passed") is False,
        "livingroom_stopped": not (batch_root / "livingroom30/o2_nominal").exists(),
        "diningroom_stopped": not (batch_root / "diningroom26/o2_nominal").exists(),
        "no_anomaly_lane": not list(batch_root.glob("**/o2_anomaly*/lane_manifest.json")),
        "not_registry_evidence": manifest.get("scene_registry_eligible") is False,
        "not_a0_a7_evidence": manifest.get("counts_as_a0_a7_evidence") is False,
    }
    result = {
        "schema_version": "kinofail.embodiedgen-centerline-v12-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "audit_passed": all(checks.values()),
        "development_matrix_passed": False,
        "checks": checks,
        "measurements": {
            "failure_step": manifest["measurements"]["steps"],
            "failure_route_progress_m": manifest["measurements"][
                "final_route_progress_m"
            ],
            "failure_route_lateral_offset_m": manifest["measurements"][
                "final_route_lateral_offset_m"
            ],
            "maximum_absolute_route_lateral_offset_m": manifest["measurements"][
                "maximum_absolute_route_lateral_offset_m"
            ],
        },
        "scientific_outcome": (
            "Removing the lateral transfer did not solve the failure: the shipped low-level "
            "policy fell before O2 contact while route tracking remained inside budget."
        ),
        "decision": (
            "Reject further outer-loop gain search; evaluate the separately trained direct-"
            "velocity policy, then train a realistic-route locomotion policy if needed."
        ),
        "evidence_boundary": {
            "counts_as_a0_a7_evidence": False,
            "scene_registry_eligible": False,
            "realistic_a0_a7_readiness": "0/8",
            "old_a0_a7_results_are_reference_only": True,
            "completion_requires_new_realistic_corpus_training_inference_and_a0_a7": True,
        },
        "inputs": {
            "preflight": {
                "path": str(preflight_path),
                "sha256": _sha256(preflight_path),
            },
            "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "telemetry": {
                "path": str(manifest_path.parent / manifest["telemetry"]),
                "sha256": _sha256(manifest_path.parent / manifest["telemetry"]),
            },
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
                "development_matrix_passed": False,
            },
            indent=2,
        )
    )
    return 0 if result["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
