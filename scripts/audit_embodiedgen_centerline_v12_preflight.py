#!/usr/bin/env python3
"""Fail-closed preflight for the v12 centerline O2 development matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.realistic_route_scene_contract import audit_realistic_route_scene_contract


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_embodiedgen_centerline_v12_development_matrix_v1.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_centerline_v12_development_matrix_v1/preflight_audit.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _json(config_path)
    boundary = config.get("evidence_boundary", {})
    policy = config.get("execution_policy", {})
    runtime = config.get("runtime", {})
    change = config.get("protocol_change", {})
    scenes = config.get("scenes", [])
    scene_ids = [str(scene.get("scene_id")) for scene in scenes]
    checks: dict[str, bool] = {
        "supported_schema": config.get("schema_version")
        == "kinofail.embodiedgen-centerline-v12-development-matrix.v1",
        "development_only": boundary.get("evidence_role")
        == "route_protocol_development_only"
        and boundary.get("development_only") is True,
        "not_registry_evidence": boundary.get("scene_registry_eligible") is False,
        "not_a0_a7_evidence": boundary.get("counts_as_a0_a7_evidence") is False,
        "a0_a7_readiness_zero": boundary.get("realistic_a0_a7_readiness") == "0/8",
        "old_a0_a7_reference_only": boundary.get("old_a0_a7_results_are_reference_only")
        is True,
        "full_realistic_replication_required": boundary.get(
            "completion_requires_new_realistic_corpus_training_inference_and_a0_a7"
        )
        is True,
        "single_centerline_change": change.get("single_change")
        == "target_lateral_offset_m_from_-0.25_to_0.0"
        and change.get("controller_code_changed") is False
        and change.get("tracking_threshold_changed") is False
        and change.get("forward_speed_changed") is False,
        "runtime_o2_nominal": runtime.get("operator") == "o2"
        and runtime.get("lane") == "nominal",
        "centerline_target": abs(float(runtime.get("target_lateral_offset_m", -1.0)))
        <= 1.0e-9
        and abs(float(runtime.get("start_lateral_offset_m", -1.0))) <= 1.0e-9,
        "full_width_operator_region": abs(
            float(runtime.get("region_half_width_m", -1.0)) - 0.60
        )
        <= 1.0e-9,
        "frozen_tracking_budget": abs(
            float(runtime.get("maximum_route_deviation_m", -1.0)) - 0.30
        )
        <= 1.0e-9,
        "frozen_v11_controller_parameters": all(
            (
                abs(float(runtime.get("forward_speed_mps", -1.0)) - 0.18) <= 1.0e-9,
                abs(float(runtime.get("lateral_limit_mps", -1.0)) - 0.20) <= 1.0e-9,
                abs(float(runtime.get("cross_track_gain_per_s", -1.0)) - 1.0)
                <= 1.0e-9,
                abs(float(runtime.get("lateral_velocity_damping", -1.0))) <= 1.0e-9,
                abs(float(runtime.get("heading_gain_per_s", -1.0)) - 2.0) <= 1.0e-9,
                abs(float(runtime.get("yaw_rate_limit_radps", -1.0)) - 0.60)
                <= 1.0e-9,
            )
        ),
        "fresh_process_per_scene": policy.get("fresh_isaac_process_per_scene") is True,
        "one_attempt_per_scene": policy.get("one_attempt_per_scene") is True,
        "nominal_only": policy.get("nominal_only") is True,
        "stop_on_first_failure": policy.get("stop_on_first_failure") is True,
        "anomaly_blocked": policy.get("no_o2_anomaly_during_development_matrix") is True,
        "three_development_scenes": len(scenes) == 3 and len(set(scene_ids)) == 3,
        "fixed_scene_order": scene_ids == policy.get("fixed_scene_order"),
        "office_excluded": "indoor_office_27" not in scene_ids
        and policy.get("office27_must_not_be_relabeled_as_unseen") is True,
        "bedroom_reserved": "indoor_bedroom_28" not in scene_ids
        and policy.get("bedroom28_must_not_be_run_during_development") is True,
    }
    rejected = config["rejected_confirmation_basis"]["postrun_audit"]
    rejected_path = _resolve(rejected["path"])
    checks["rejected_v11_audit_hash"] = rejected_path.is_file() and _sha256(
        rejected_path
    ) == rejected["sha256"]
    rejected_payload = _json(rejected_path) if rejected_path.is_file() else {}
    checks["v11_scientifically_rejected"] = (
        rejected_payload.get("audit_passed") is True
        and rejected_payload.get("controller_confirmation_passed") is False
    )
    for name, spec in config.get("source_freeze", {}).items():
        path = _resolve(spec["path"])
        checks[f"source_{name}_hash"] = path.is_file() and _sha256(path) == spec["sha256"]

    contracts: list[dict[str, Any]] = []
    scene_families: set[str] = set()
    material_ids: set[str] = set()
    for scene in scenes:
        slug = str(scene["scene_family"]).lower()
        scene_families.add(str(scene["scene_family"]))
        material_ids.add(str(scene["material_id"]))
        payloads: dict[str, dict[str, Any]] = {}
        for key in ("route_surface_admission", "terrain_compiled_audit", "offline_usd_audit"):
            spec = scene[key]
            path = _resolve(spec["path"])
            checks[f"{slug}_{key}_hash"] = path.is_file() and _sha256(path) == spec["sha256"]
            payloads[key] = _json(path) if path.is_file() else {}
        episode = _resolve(scene["terrain_episode"]["path"])
        checks[f"{slug}_terrain_episode_hash"] = episode.is_file() and _sha256(
            episode
        ) == scene["terrain_episode"]["sha256"]
        checks[f"{slug}_scene_identity"] = all(
            payload.get("scene_id") == scene["scene_id"] for payload in payloads.values()
        )
        checks[f"{slug}_scene_inputs_passed"] = all(
            payload.get("passed") is True for payload in payloads.values()
        )
        contract = audit_realistic_route_scene_contract(payloads["terrain_compiled_audit"])
        checks[f"{slug}_route_contract_passed"] = contract.get("passed") is True
        checks[f"{slug}_operator_spans_route_width"] = (
            abs(
                float(contract.get("binding", {}).get("surface_width_m", -1.0))
                - 2.0 * float(runtime["region_half_width_m"])
            )
            <= 1.0e-9
        )
        checks[f"{slug}_planned_output_unused"] = not _resolve(
            scene["planned_output"]
        ).exists()
        contracts.append(contract)
    checks["three_distinct_scene_families"] = len(scene_families) == 3
    checks["three_distinct_materials"] = len(material_ids) == 3

    result = {
        "schema_version": "kinofail.embodiedgen-centerline-v12-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "scene_contracts": contracts,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
    }
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "passed": result["passed"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
