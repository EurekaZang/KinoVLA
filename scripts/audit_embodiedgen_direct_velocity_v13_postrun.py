#!/usr/bin/env python3
"""Audit the failed v13 direct-velocity policy screen."""

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
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_direct_velocity_v13_screen_v1"
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
    progress = float(manifest["measurements"]["final_route_progress_m"])
    checks = {
        "preflight_passed": preflight.get("passed") is True,
        "candidate_policy_hash_recorded": manifest.get("provenance", {}).get(
            "locomotion_policy_sha256"
        )
        == "730a57fb0ea53ab35e63d2bf37cac59789756a6bfdfcda00300b87895e258490"
        and _sha256(policy_path) == manifest["provenance"]["locomotion_policy_sha256"],
        "robot_remained_upright": manifest.get("measurements", {}).get("fallen") is False,
        "tracking_stayed_within_budget": manifest.get("checks", {}).get(
            "full_lane_tracking_within_admitted_budget"
        )
        is True,
        "route_not_completed": manifest.get("checks", {}).get("nominal_route_completed")
        is False,
        "low_speed_nonresponse": progress < 0.10,
        "screen_scientifically_failed": manifest.get("passed") is False,
        "no_anomaly_lane": not list(manifest_path.parents[2].glob("**/o2_anomaly*/lane_manifest.json")),
        "not_registry_or_a0_a7_evidence": manifest.get("scene_registry_eligible") is False
        and manifest.get("counts_as_a0_a7_evidence") is False,
    }
    result = {
        "schema_version": "kinofail.embodiedgen-direct-velocity-v13-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "audit_passed": all(checks.values()),
        "policy_screen_passed": False,
        "checks": checks,
        "measurements": {
            "steps": manifest["measurements"]["steps"],
            "commanded_forward_speed_mps": manifest["route_controller"][
                "forward_speed_mps"
            ],
            "final_route_progress_m": progress,
            "maximum_absolute_route_lateral_offset_m": manifest["measurements"][
                "maximum_absolute_route_lateral_offset_m"
            ],
        },
        "scientific_outcome": (
            "The direct-velocity candidate is stable but effectively treats the 0.18 m/s "
            "command as standing, so it is rejected for realistic route collection."
        ),
        "decision": (
            "Train a separate low-speed realistic-route policy; preserve both legacy policies "
            "and recalibrate the new benchmark stack against the new artifact."
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
            "locomotion_policy": {
                "path": str(policy_path),
                "sha256": _sha256(policy_path),
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
                "policy_screen_passed": False,
                "final_route_progress_m": progress,
            },
            indent=2,
        )
    )
    return 0 if result["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
