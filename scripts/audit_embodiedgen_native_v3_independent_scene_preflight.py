#!/usr/bin/env python3
"""Audit the freeze boundary for two native-v3 independent scene preflights."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kino_vla.sim.realistic_route_protocol_v2 import scene_route_binding


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_embodiedgen_native_v3_independent_scene_preflight_v1.json",
    )
    parser.add_argument("--phase", choices=("preflight", "postrun"), required=True)
    parser.add_argument(
        "--preflight-audit",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_native_v3_preflight_v1/preflight_audit.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_native_v3_preflight_v1/preflight_audit.json",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _json(config_path)
    boundary = config.get("evidence_boundary", {})
    policy = config.get("execution_policy", {})
    checks: dict[str, bool] = {
        "supported_schema": config.get("schema_version")
        in {
            "kinofail.embodiedgen-native-v3-independent-scene-preflight.v1",
            "kinofail.embodiedgen-native-v4-tracking-confirmation.v2",
        },
        "development_only": boundary.get("development_only") is True,
        "not_registry_evidence": boundary.get("scene_registry_eligible") is False,
        "not_a0_a7_evidence": boundary.get("counts_as_a0_a7_evidence") is False,
        "old_a0_a7_are_protocol_only": boundary.get("old_a0_a7_role")
        == "protocol_and_target_conclusions_only",
        "new_a0_a7_replication_required": boundary.get(
            "completion_requires_new_realistic_corpus_a0_a7_replication"
        )
        is True,
        "readiness_explicitly_zero_of_eight": boundary.get(
            "realistic_a0_a7_readiness_expected_before_corpus"
        )
        == "0/8",
        "nominal_before_anomaly": policy.get("nominal_preflight_before_any_anomaly")
        is True,
        "one_attempt_policy": policy.get("one_attempt_per_scene") is True,
        "no_retry_policy": policy.get("no_retry_or_scene_replacement") is True,
        "fresh_process_policy": policy.get("fresh_isaac_process_per_lane") is True,
    }
    if args.phase == "postrun":
        preflight = _json(args.preflight_audit.resolve()) if args.preflight_audit.is_file() else {}
        checks["preflight_passed"] = preflight.get("passed") is True
        checks["same_config_as_preflight"] = preflight.get("config", {}).get(
            "sha256"
        ) == _sha256(config_path)

    frozen_inputs: list[dict[str, Any]] = []
    for name, spec in config.get("source_freeze", {}).items():
        path = _resolve(spec["path"])
        valid = path.is_file() and _sha256(path) == spec["sha256"]
        checks[f"source_{name}_hash"] = valid
        frozen_inputs.append(
            {"name": name, "path": str(path), "sha256": spec["sha256"], "valid": valid}
        )
        if name == "controller_calibration":
            calibration = _json(path) if path.is_file() else {}
            checks["controller_calibration_passed"] = calibration.get("passed") is True
            checks["controller_calibration_not_a0_a7_evidence"] = calibration.get(
                "counts_as_a0_a7_evidence"
            ) is False
            checks["controller_contract_matches_calibration"] = calibration.get(
                "candidate_controller_contract"
            ) == {
                "forward_speed_mps": 0.20,
                "target_lateral_offset_m": -0.25,
                "operator_region_lateral_offset_m": -0.15,
                "operator_region_half_width_m": 0.40,
                "maximum_absolute_route_lateral_offset_m": 0.30,
                "minimum_operator_region_samples": 40,
            }

    contract = config.get("frozen_nominal_contract", {})
    checks["tracking_gated_contract_frozen"] = all(
        (
            abs(float(contract.get("forward_speed_mps", -1.0)) - 0.20) <= 1.0e-9,
            abs(float(contract.get("target_lateral_offset_m", 1.0)) + 0.25) <= 1.0e-9,
            abs(float(contract.get("region_lateral_offset_m", 1.0)) + 0.15) <= 1.0e-9,
            abs(float(contract.get("region_half_width_m", -1.0)) - 0.40) <= 1.0e-9,
            abs(float(contract.get("maximum_route_deviation_m", -1.0)) - 0.30)
            <= 1.0e-9,
            int(contract.get("minimum_operator_region_samples", -1)) == 40,
        )
    )

    scenes = config.get("scenes", [])
    checks["exactly_two_scenes"] = len(scenes) == 2
    checks["independent_scene_families"] = len(
        {str(scene.get("scene_family")) for scene in scenes}
    ) == len(scenes)
    checks["independent_scene_ids"] = len({str(scene.get("scene_id")) for scene in scenes}) == len(
        scenes
    )
    checks["distinct_visual_materials"] = len(
        {str(scene.get("material_id")) for scene in scenes}
    ) == len(scenes)

    scene_results: list[dict[str, Any]] = []
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        payloads: dict[str, dict[str, Any]] = {}
        for key in ("route_surface_admission", "terrain_compiled_audit", "offline_usd_audit"):
            spec = scene[key]
            path = _resolve(spec["path"])
            valid = path.is_file() and _sha256(path) == spec["sha256"]
            checks[f"{scene_id}_{key}_hash"] = valid
            payloads[key] = _json(path) if path.is_file() else {}
        episode_spec = scene["terrain_episode"]
        episode_path = _resolve(episode_spec["path"])
        checks[f"{scene_id}_terrain_episode_hash"] = (
            episode_path.is_file() and _sha256(episode_path) == episode_spec["sha256"]
        )
        compiled = payloads["terrain_compiled_audit"]
        offline = payloads["offline_usd_audit"]
        admission = payloads["route_surface_admission"]
        checks[f"{scene_id}_scene_identity"] = all(
            payload.get("scene_id") == scene_id
            for payload in (compiled, offline, admission)
        )
        checks[f"{scene_id}_prior_route_surface_admitted"] = admission.get("passed") is True
        checks[f"{scene_id}_terrain_compiled"] = compiled.get("passed") is True
        checks[f"{scene_id}_offline_usd_audited"] = offline.get("passed") is True
        try:
            binding = scene_route_binding(compiled)
            binding_ok = (
                binding.source_kind == "embodiedgen_terrain_v2"
                and binding.frame.route_length_m >= 1.80 - 1.0e-6
                and binding.collector_start_waypoint_index >= 0
            )
        except (KeyError, TypeError, ValueError):
            binding = None
            binding_ok = False
        checks[f"{scene_id}_operator_route_binding"] = binding_ok
        output = _resolve(scene["planned_nominal_output"])
        max_abs_lateral = None
        tracking_budget = None
        if args.phase == "preflight":
            checks[f"{scene_id}_planned_output_unused"] = not output.exists()
        else:
            manifest_path = output / "lane_manifest.json"
            manifest = _json(manifest_path) if manifest_path.is_file() else {}
            checks[f"{scene_id}_nominal_manifest_present"] = manifest_path.is_file()
            checks[f"{scene_id}_nominal_passed"] = manifest.get("passed") is True
            checks[f"{scene_id}_nominal_completed_without_fall"] = (
                manifest.get("checks", {}).get("nominal_route_completed") is True
                and manifest.get("measurements", {}).get("fallen") is False
            )
            checks[f"{scene_id}_native_collector_hash"] = manifest.get(
                "provenance", {}
            ).get("script_sha256") == config["source_freeze"]["native_collector"]["sha256"]
            checks[f"{scene_id}_route_protocol_hash"] = manifest.get(
                "provenance", {}
            ).get("route_protocol_sha256") == config["source_freeze"]["route_protocol"]["sha256"]
            checks[f"{scene_id}_still_not_a0_a7_evidence"] = manifest.get(
                "counts_as_a0_a7_evidence"
            ) is False
            route_controller = manifest.get("route_controller", {})
            checks[f"{scene_id}_frozen_controller_contract_applied"] = all(
                (
                    abs(float(route_controller.get("forward_speed_mps", -1.0)) - 0.20)
                    <= 1.0e-9,
                    abs(
                        float(route_controller.get("target_lateral_offset_m", 1.0))
                        + 0.25
                    )
                    <= 1.0e-9,
                    abs(
                        float(
                            route_controller.get("operator_region_lateral_offset_m", 1.0)
                        )
                        + 0.15
                    )
                    <= 1.0e-9,
                    abs(
                        float(route_controller.get("operator_region_half_width_m", -1.0))
                        - 0.40
                    )
                    <= 1.0e-9,
                    abs(
                        float(
                            route_controller.get(
                                "admitted_maximum_route_deviation_m", -1.0
                            )
                        )
                        - 0.30
                    )
                    <= 1.0e-9,
                    int(route_controller.get("minimum_operator_region_samples", -1)) == 40,
                )
            )
            telemetry_path = manifest_path.parent / str(manifest.get("telemetry", ""))
            telemetry_hash_valid = telemetry_path.is_file() and _sha256(
                telemetry_path
            ) == manifest.get("telemetry_sha256")
            checks[f"{scene_id}_telemetry_hash"] = telemetry_hash_valid
            rows = _jsonl(telemetry_path) if telemetry_hash_valid else []
            max_abs_lateral = max(
                (abs(float(row["route_lateral_offset_m"])) for row in rows),
                default=float("inf"),
            )
            tracking_budget = float(
                compiled.get("route", {}).get(
                    "maximum_admissible_tracking_error_m", float("-inf")
                )
            )
            checks[f"{scene_id}_full_lane_tracking_within_admitted_budget"] = (
                max_abs_lateral <= tracking_budget + 1.0e-9
            )
            checks[f"{scene_id}_operator_region_has_at_least_40_samples"] = int(
                manifest.get("measurements", {}).get("region_sample_count", 0)
            ) >= 40
        scene_results.append(
            {
                "scene_id": scene_id,
                "scene_family": scene["scene_family"],
                "material_id": scene["material_id"],
                "planned_nominal_output": str(output),
                "route_binding": (
                    {
                        "origin_xy_m": list(binding.frame.origin_xy_m),
                        "direction_xy": list(binding.frame.direction_xy),
                        "available_route_length_m": binding.frame.route_length_m,
                    }
                    if binding is not None
                    else None
                ),
                "postrun_tracking": (
                    {
                        "maximum_absolute_lateral_offset_m": max_abs_lateral,
                        "admitted_budget_m": tracking_budget,
                    }
                    if args.phase == "postrun"
                    else None
                ),
            }
        )

    result = {
        "schema_version": "kinofail.embodiedgen-native-v3-independent-scene-preflight-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "phase": args.phase,
        "passed": all(checks.values()),
        "checks": checks,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "frozen_inputs": frozen_inputs,
        "scenes": scene_results,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "completion_contract": config.get("post_confirmation_sequence", []),
    }
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "phase": args.phase, "passed": result["passed"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
