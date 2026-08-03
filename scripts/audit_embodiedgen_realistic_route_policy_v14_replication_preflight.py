#!/usr/bin/env python3
"""Fail-closed preflight for v14 LivingRoom/DiningRoom replication."""

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


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _bound(spec: dict[str, Any]) -> bool:
    path = _resolve(spec["path"])
    return path.is_file() and _sha256(path) == spec["sha256"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_embodiedgen_realistic_route_policy_v14_development_replication_v1.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v14_development_replication_v1/preflight_audit.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _json(config_path)
    boundary = config.get("evidence_boundary", {})
    policy = config.get("execution_policy", {})
    runtime = config.get("runtime", {})
    scenes = config.get("scenes", [])
    scene_ids = [scene.get("scene_id") for scene in scenes]
    checks: dict[str, bool] = {
        "supported_schema": config.get("schema_version")
        == "kinofail.embodiedgen-realistic-route-policy-v14-development-replication.v1",
        "development_only": boundary.get("evidence_role")
        == "cross_scene_policy_development_replication_only"
        and boundary.get("development_only") is True,
        "not_registry_or_a0_a7_evidence": boundary.get("scene_registry_eligible") is False
        and boundary.get("counts_as_a0_a7_evidence") is False
        and boundary.get("realistic_a0_a7_readiness") == "0/8",
        "full_realistic_replication_required": boundary.get(
            "completion_requires_new_realistic_corpus_training_inference_and_a0_a7"
        )
        is True,
        "policy_hash": _bound(config["policy"]),
        "fresh_one_attempt_each": policy.get("fresh_isaac_process_per_scene") is True
        and policy.get("one_attempt_per_scene") is True,
        "fixed_order": scene_ids == policy.get("fixed_order"),
        "stop_on_failure": policy.get("stop_on_first_failure") is True,
        "nominal_no_anomaly": policy.get("nominal_only") is True
        and policy.get("no_anomaly") is True,
        "no_changes": policy.get("no_parameter_or_threshold_change") is True,
        "bedroom_reserved": "indoor_bedroom_28" not in scene_ids
        and policy.get("bedroom28_remains_untouched") is True,
        "office_excluded": "indoor_office_27" not in scene_ids
        and policy.get("office27_remains_exposed_diagnostic_only") is True,
        "runtime_frozen": all(
            (
                runtime.get("operator") == "o2",
                runtime.get("lane") == "nominal",
                abs(float(runtime.get("target_lateral_offset_m", -1.0))) <= 1.0e-9,
                abs(float(runtime.get("forward_speed_mps", -1.0)) - 0.18) <= 1.0e-9,
                abs(float(runtime.get("maximum_route_deviation_m", -1.0)) - 0.30)
                <= 1.0e-9,
                abs(float(runtime.get("lateral_limit_mps", -1.0)) - 0.20) <= 1.0e-9,
                abs(float(runtime.get("cross_track_gain_per_s", -1.0)) - 1.0)
                <= 1.0e-9,
                abs(float(runtime.get("lateral_velocity_damping", -1.0))) <= 1.0e-9,
                abs(float(runtime.get("heading_gain_per_s", -1.0)) - 2.0) <= 1.0e-9,
                abs(float(runtime.get("yaw_rate_limit_radps", -1.0)) - 0.60)
                <= 1.0e-9,
            )
        ),
    }
    for name, spec in config.get("frozen_kitchen_screen", {}).items():
        checks[f"kitchen_{name}_hash"] = _bound(spec)
    kitchen = _json(_resolve(config["frozen_kitchen_screen"]["postrun_audit"]["path"]))
    checks["kitchen_screen_passed_not_formally_admitted"] = (
        kitchen.get("audit_passed") is True
        and kitchen.get("kitchen_screen_passed") is True
        and kitchen.get("policy_admitted_for_formal_collection") is False
    )
    for name, spec in config.get("source_freeze", {}).items():
        checks[f"source_{name}_hash"] = _bound(spec)

    contracts: list[dict[str, Any]] = []
    for scene in scenes:
        slug = str(scene["scene_family"]).lower()
        payloads: dict[str, dict[str, Any]] = {}
        for key in ("terrain_compiled_audit", "offline_usd_audit"):
            checks[f"{slug}_{key}_hash"] = _bound(scene[key])
            payloads[key] = _json(_resolve(scene[key]["path"]))
        checks[f"{slug}_episode_hash"] = _bound(scene["terrain_episode"])
        checks[f"{slug}_identity"] = all(
            value.get("scene_id") == scene["scene_id"] for value in payloads.values()
        )
        checks[f"{slug}_inputs_passed"] = all(
            value.get("passed") is True for value in payloads.values()
        )
        contract = audit_realistic_route_scene_contract(payloads["terrain_compiled_audit"])
        checks[f"{slug}_route_contract"] = contract.get("passed") is True
        checks[f"{slug}_output_unused"] = not _resolve(scene["planned_output"]).exists()
        contracts.append(contract)

    result = {
        "schema_version": "kinofail.embodiedgen-realistic-route-policy-v14-replication-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "route_contracts": contracts,
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
