#!/usr/bin/env python3
"""Fail-closed preflight for v11 O2 nominal confirmation on unseen scenes."""

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
        / "configs/data/kinofail_embodiedgen_o2_v11_unseen_nominal_confirmation_v1.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_o2_v11_unseen_nominal_confirmation_v1/preflight_audit.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _json(config_path)
    boundary = config.get("evidence_boundary", {})
    policy = config.get("execution_policy", {})
    runtime = config.get("runtime", {})
    checks: dict[str, bool] = {
        "supported_schema": config.get("schema_version")
        == "kinofail.embodiedgen-o2-v11-unseen-nominal-confirmation.v1",
        "confirmation_only": boundary.get("evidence_role")
        == "unseen_scene_operator_controller_confirmation_only",
        "not_registry_evidence": boundary.get("scene_registry_eligible") is False,
        "not_a0_a7_evidence": boundary.get("counts_as_a0_a7_evidence") is False,
        "a0_a7_readiness_zero": boundary.get("realistic_a0_a7_readiness") == "0/8",
        "full_realistic_replication_required": boundary.get(
            "completion_requires_new_realistic_corpus_training_inference_and_a0_a7"
        )
        is True,
        "fresh_process_per_scene": policy.get("fresh_isaac_process_per_scene") is True,
        "one_attempt_per_scene": policy.get("one_attempt_per_scene") is True,
        "nominal_only": policy.get("nominal_only") is True,
        "anomaly_blocked_until_both_pass": policy.get(
            "both_scenes_must_pass_before_any_o2_anomaly"
        )
        is True,
        "controller_frozen_after_preflight": policy.get(
            "no_controller_or_threshold_change_after_preflight"
        )
        is True,
        "runtime_o2_nominal": runtime.get("operator") == "o2"
        and runtime.get("lane") == "nominal",
        "tracking_budget_frozen": abs(
            float(runtime.get("maximum_route_deviation_m", -1.0)) - 0.30
        )
        <= 1.0e-9,
        "v11_controller_frozen": all(
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
    }
    calibration_path = _resolve(config["calibration_freeze"]["path"])
    checks["calibration_hash"] = calibration_path.is_file() and _sha256(
        calibration_path
    ) == config["calibration_freeze"]["sha256"]
    checks["calibration_passed"] = (
        _json(calibration_path).get("passed") is True if calibration_path.is_file() else False
    )
    for name, spec in config.get("source_freeze", {}).items():
        path = _resolve(spec["path"])
        checks[f"source_{name}_hash"] = path.is_file() and _sha256(path) == spec["sha256"]

    scene_contracts: list[dict[str, Any]] = []
    scene_families: set[str] = set()
    for scene in config.get("scenes", []):
        slug = str(scene["scene_family"]).lower()
        scene_families.add(str(scene["scene_family"]))
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
        scene_contracts.append(contract)
        checks[f"{slug}_planned_output_unused"] = not _resolve(scene["planned_output"]).exists()
    checks["two_distinct_unseen_scene_families"] = len(scene_families) == 2

    result = {
        "schema_version": "kinofail.embodiedgen-o2-v11-unseen-nominal-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "scene_contracts": scene_contracts,
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
