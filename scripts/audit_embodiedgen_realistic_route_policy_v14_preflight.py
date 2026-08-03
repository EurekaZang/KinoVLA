#!/usr/bin/env python3
"""Fail-closed preflight for the v14 realistic-route policy screen."""

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
        / "configs/data/kinofail_embodiedgen_realistic_route_policy_v14_screen_v1.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v14_screen_v1/preflight_audit.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _json(config_path)
    boundary = config.get("evidence_boundary", {})
    policy = config.get("execution_policy", {})
    runtime = config.get("runtime", {})
    scene = config.get("scene", {})
    training = config.get("training_freeze", {})
    checks: dict[str, bool] = {
        "supported_schema": config.get("schema_version")
        == "kinofail.embodiedgen-realistic-route-policy-v14-screen.v1",
        "development_only": boundary.get("evidence_role")
        == "post_training_route_screen_only"
        and boundary.get("development_only") is True,
        "not_registry_or_a0_a7_evidence": boundary.get("scene_registry_eligible") is False
        and boundary.get("counts_as_a0_a7_evidence") is False
        and boundary.get("realistic_a0_a7_readiness") == "0/8",
        "full_realistic_replication_required": boundary.get(
            "completion_requires_new_realistic_corpus_training_inference_and_a0_a7"
        )
        is True,
        "one_nominal_attempt": policy.get("one_attempt") is True
        and policy.get("nominal_only") is True
        and policy.get("no_anomaly") is True,
        "policy_only_change": policy.get("policy_is_only_change_from_v13") is True,
        "all_runtime_checks_required": policy.get("pass_requires_every_runtime_check") is True,
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
        "training_policy_hash": _bound(training["policy"]),
        "training_manifest_hash": _bound(training["training_manifest"]),
        "training_postrun_hash": _bound(training["training_postrun_audit"]),
        "policy_not_pre_admitted": training.get("policy_admitted_before_this_screen") is False,
        "failed_predecessor_hash": _bound(config["failed_predecessor"]),
        "kitchen_development_scene": scene.get("scene_id") == "indoor_kitchen_31",
        "compiled_scene_hash": _bound(scene["terrain_compiled_audit"]),
        "episode_hash": _bound(scene["terrain_episode"]),
        "offline_audit_hash": _bound(scene["offline_usd_audit"]),
        "planned_output_unused": not _resolve(config["planned_output"]).exists(),
    }
    for name, spec in config.get("source_freeze", {}).items():
        checks[f"source_{name}_hash"] = _bound(spec)
    training_audit = _json(_resolve(training["training_postrun_audit"]["path"]))
    checks["training_audit_passed_but_not_admitted"] = (
        training_audit.get("audit_passed") is True
        and training_audit.get("policy_admitted_for_collection") is False
    )
    predecessor = _json(_resolve(config["failed_predecessor"]["path"]))
    checks["predecessor_rejected"] = predecessor.get("audit_passed") is True and predecessor.get(
        "policy_screen_passed"
    ) is False
    compiled = _json(_resolve(scene["terrain_compiled_audit"]["path"]))
    offline = _json(_resolve(scene["offline_usd_audit"]["path"]))
    route_contract = audit_realistic_route_scene_contract(compiled)
    checks["scene_inputs_passed"] = compiled.get("passed") is True and offline.get(
        "passed"
    ) is True
    checks["scene_identity"] = compiled.get("scene_id") == offline.get("scene_id") == scene.get(
        "scene_id"
    )
    checks["route_contract_passed"] = route_contract.get("passed") is True

    result = {
        "schema_version": "kinofail.embodiedgen-realistic-route-policy-v14-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "route_contract": route_contract,
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
