#!/usr/bin/env python3
"""Fail-closed audit for the 11 registered Fig. 2 multiview cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kino_vla.data.o9_semantics import evaluate_o9_high_centering


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit registered Fig. 2 multiview sources")
    parser.add_argument(
        "--root",
        default="outputs/kinofail_realistic/fig2_registered_multiview_v1",
    )
    args = parser.parse_args()
    source_root = (ROOT / args.root).resolve()
    cases = []
    all_checks: dict[str, bool] = {}

    for index in range(1, 12):
        operator = f"O{index}"
        manifest_path = source_root / operator / "capture_manifest.json"
        manifest = load(manifest_path)
        camera = manifest["camera"]
        simulation = manifest["simulation"]
        external = list(camera["registered_external_views"])
        expected_paths = [camera["front"], *external]
        checks = {
            "manifest_operator_matches_directory": manifest["operator"]["short_id"] == operator,
            "manifest_validation_passed": manifest["validation"]["passed"] is True,
            "same_case_declared": manifest["validation"]["same_case_id_all_views"] is True,
            "same_scene_declared": manifest["validation"]["same_scene_all_views"] is True,
            "same_seed_declared": manifest["validation"]["same_seed_all_views"] is True,
            "same_parameters_declared": (
                manifest["validation"]["same_operator_parameters_all_views"] is True
            ),
            "same_timestamp_declared": manifest["validation"]["same_timestamp_all_views"] is True,
            "camera_only_change_declared": (
                manifest["validation"]["only_camera_extrinsics_changed"] is True
            ),
            "single_sensor_declared": camera["single_sensor_moved_after_physics_pause"] is True,
            "state_hashes_equal": (
                simulation["physics_state_sha256_before"]
                == simulation["physics_state_sha256_after"]
            ),
            "state_delta_exactly_zero": (
                float(simulation["max_abs_state_delta_across_views"]) == 0.0
            ),
            "state_frozen_declared": simulation["state_frozen_across_views"] is True,
            "three_external_views": len(external) == 3,
            "has_scene_overview": any(row["role"] == "registered_scene_overview" for row in external),
            "has_two_contact_views": (
                sum(row["role"] == "registered_low_contact_view" for row in external) == 2
            ),
            "all_images_exist": all((ROOT / row["path"]).is_file() for row in expected_paths),
            "all_image_hashes_match": all(
                sha256(ROOT / row["path"]) == row["sha256"] for row in expected_paths
            ),
        }
        if operator == "O9":
            position = simulation.get("robot_position_xy_m")
            semantic = evaluate_o9_high_centering(
                manifest.get("operator_telemetry_at_capture", {}),
                robot_position_xy_m=position,
                route_axis="x",
            )
            checks.update(
                {
                    "high_centering_semantics_recomputed": semantic["passed"] is True,
                    "high_centering_semantics_declared": (
                        manifest.get("operator_semantic_validation", {}).get("passed")
                        is True
                    ),
                    "high_centering_semantics_match": (
                        manifest.get("operator_semantic_validation") == semantic
                    ),
                }
            )
        passed = all(checks.values())
        for name, value in checks.items():
            all_checks[f"{operator}.{name}"] = bool(value)
        cases.append(
            {
                "operator": operator,
                "case_id": manifest["case_id"],
                "scene_id": manifest["scene"]["scene_id"],
                "seed": simulation["seed"],
                "timestamp_s": simulation["timestamp_s"],
                "physics_state_sha256": simulation["physics_state_sha256_before"],
                "manifest": str(manifest_path.relative_to(ROOT)),
                "manifest_sha256": sha256(manifest_path),
                "checks": checks,
                "passed": passed,
            }
        )

    aggregate = {
        "schema_version": "kinofail.fig2-registered-multiview-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_root": str(source_root.relative_to(ROOT)),
        "operators_expected": 11,
        "operators_found": len(cases),
        "all_views_from_one_frozen_state_per_operator": all(all_checks.values()),
        "passed": len(cases) == 11 and all(all_checks.values()),
        "cases": cases,
    }
    output = source_root / "aggregate_audit.json"
    output.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "passed": aggregate["passed"]}, sort_keys=True))
    if not aggregate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
