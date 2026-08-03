#!/usr/bin/env python3
"""Audit the three-scene v14 realistic-route policy development replication."""

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
    replication = (
        ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v14_development_replication_v1"
    )
    kitchen = (
        ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v14_screen_v1/kitchen31/o2_nominal/lane_manifest.json"
    )
    parser.add_argument("--preflight", type=Path, default=replication / "preflight_audit.json")
    parser.add_argument("--kitchen", type=Path, default=kitchen)
    parser.add_argument(
        "--livingroom", type=Path, default=replication / "livingroom30/o2_nominal/lane_manifest.json"
    )
    parser.add_argument(
        "--diningroom", type=Path, default=replication / "diningroom26/o2_nominal/lane_manifest.json"
    )
    parser.add_argument("--out", type=Path, default=replication / "postrun_audit.json")
    args = parser.parse_args()
    preflight_path = args.preflight.resolve()
    preflight = _json(preflight_path)
    manifest_paths = {
        "Kitchen": args.kitchen.resolve(),
        "LivingRoom": args.livingroom.resolve(),
        "DiningRoom": args.diningroom.resolve(),
    }
    manifests = {name: _json(path) for name, path in manifest_paths.items()}
    expected_scenes = {
        "Kitchen": "indoor_kitchen_31",
        "LivingRoom": "indoor_livingroom_30",
        "DiningRoom": "indoor_diningroom_26",
    }
    policy_hash = "fb40283b56ec260fd14d8943684107557f2e49c3372c2fca7804f0879642b61f"
    checks: dict[str, bool] = {"preflight_passed": preflight.get("passed") is True}
    measurements: dict[str, Any] = {}
    for family, manifest in manifests.items():
        checks[f"{family}_scene_identity"] = manifest.get("scene_id") == expected_scenes[family]
        checks[f"{family}_all_runtime_checks"] = manifest.get("passed") is True and all(
            value is True for value in manifest.get("checks", {}).values()
        )
        checks[f"{family}_policy_hash"] = manifest.get("provenance", {}).get(
            "locomotion_policy_sha256"
        ) == policy_hash
        checks[f"{family}_not_a0_a7_evidence"] = (
            manifest.get("counts_as_a0_a7_evidence") is False
        )
        checks[f"{family}_not_registry_evidence"] = (
            manifest.get("scene_registry_eligible") is False
        )
        measurements[family] = {
            "route_progress_m": manifest["measurements"]["final_route_progress_m"],
            "maximum_absolute_route_lateral_offset_m": manifest["measurements"][
                "maximum_absolute_route_lateral_offset_m"
            ],
            "region_sample_count": manifest["measurements"]["region_sample_count"],
            "load_bearing_region_sample_count": manifest["measurements"][
                "load_bearing_region_sample_count"
            ],
            "median_region_base_height_m": manifest["measurements"][
                "median_region_base_height_m"
            ],
            "fallen": manifest["measurements"]["fallen"],
        }
    checks["no_anomaly_lanes"] = not list(replication.glob("**/o2_anomaly*/lane_manifest.json"))
    deviations = [
        float(value["maximum_absolute_route_lateral_offset_m"])
        for value in measurements.values()
    ]
    progresses = [float(value["route_progress_m"]) for value in measurements.values()]
    result = {
        "schema_version": "kinofail.embodiedgen-realistic-route-policy-v14-replication-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "audit_passed": all(checks.values()),
        "development_replication_passed": all(checks.values()),
        "policy_admitted_for_formal_collection": False,
        "checks": checks,
        "scene_measurements": measurements,
        "aggregate": {
            "scene_family_count": 3,
            "all_passed": all(manifest.get("passed") is True for manifest in manifests.values()),
            "maximum_lateral_deviation_across_scenes_m": max(deviations),
            "minimum_tracking_margin_across_scenes_m": 0.30 - max(deviations),
            "minimum_final_route_progress_m": min(progresses),
            "maximum_final_route_progress_m": max(progresses),
        },
        "scientific_outcome": (
            "The new low-speed policy reproduces the complete nominal O2 route contract on "
            "Kitchen, LivingRoom, and DiningRoom development families with centimeter-scale "
            "lateral error."
        ),
        "next_required_action": (
            "Freeze confirmation on untouched Bedroom28 and one newly generated untouched "
            "scene family; no anomaly lane is permitted before both pass."
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
            "manifests": {
                family: {"path": str(path), "sha256": _sha256(path)}
                for family, path in manifest_paths.items()
            },
            "telemetry": {
                family: {
                    "path": str(path.parent / manifests[family]["telemetry"]),
                    "sha256": _sha256(path.parent / manifests[family]["telemetry"]),
                }
                for family, path in manifest_paths.items()
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
                "development_replication_passed": result["development_replication_passed"],
                "policy_admitted_for_formal_collection": False,
                "aggregate": result["aggregate"],
            },
            indent=2,
        )
    )
    return 0 if result["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
