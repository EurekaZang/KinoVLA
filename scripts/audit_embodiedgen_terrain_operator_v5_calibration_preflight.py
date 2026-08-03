#!/usr/bin/env python3
"""Fail-closed preflight for Kitchen31 terrain-operator v5 calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kino_vla.eval.realistic_route_scene_contract import (
    audit_realistic_route_scene_contract,
)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_embodiedgen_terrain_operator_v5_calibration_v1.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_terrain_operator_v5_calibration_v1/preflight_audit.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _json(config_path)
    boundary = config.get("evidence_boundary", {})
    policy = config.get("execution_policy", {})
    runtime = config.get("common_runtime", {})
    checks: dict[str, bool] = {
        "supported_schema": config.get("schema_version")
        == "kinofail.embodiedgen-terrain-operator-v5-calibration.v1",
        "calibration_only": boundary.get("evidence_role") == "kitchen31_calibration_only",
        "not_registry_evidence": boundary.get("scene_registry_eligible") is False,
        "not_a0_a7_evidence": boundary.get("counts_as_a0_a7_evidence") is False,
        "a0_a7_readiness_zero": boundary.get("realistic_a0_a7_readiness") == "0/8",
        "new_realistic_a0_a7_required": boundary.get(
            "completion_requires_new_realistic_corpus_a0_a7_replication"
        )
        is True,
        "one_attempt_policy": policy.get("one_attempt_per_operator_lane") is True,
        "no_post_anomaly_threshold_change": policy.get(
            "no_threshold_change_after_first_anomaly"
        )
        is True,
        "lane_aware_controller_frozen": all(
            (
                abs(float(runtime.get("forward_speed_mps", -1.0)) - 0.20) <= 1.0e-9,
                abs(float(runtime.get("target_lateral_offset_m", 1.0)) + 0.25) <= 1.0e-9,
                abs(float(runtime.get("maximum_route_deviation_m", -1.0)) - 0.30)
                <= 1.0e-9,
                int(runtime.get("minimum_nominal_region_samples", -1)) == 40,
            )
        ),
    }
    frozen_inputs: list[dict[str, Any]] = []
    for name, spec in config.get("source_freeze", {}).items():
        path = _resolve(spec["path"])
        valid = path.is_file() and _sha256(path) == spec["sha256"]
        checks[f"source_{name}_hash"] = valid
        if name == "controller_confirmation":
            checks["controller_confirmation_passed"] = (
                _json(path).get("passed") is True if path.is_file() else False
            )
        frozen_inputs.append({"name": name, "path": str(path), "valid": valid})

    scene_payloads: dict[str, dict[str, Any]] = {}
    for name in (
        "route_surface_admission",
        "terrain_compiled_audit",
        "offline_usd_audit",
    ):
        spec = config["scene"][name]
        path = _resolve(spec["path"])
        valid = path.is_file() and _sha256(path) == spec["sha256"]
        checks[f"scene_{name}_hash"] = valid
        scene_payloads[name] = _json(path) if path.is_file() else {}
    episode_spec = config["scene"]["terrain_episode"]
    episode = _resolve(episode_spec["path"])
    checks["scene_episode_hash"] = episode.is_file() and _sha256(
        episode
    ) == episode_spec["sha256"]
    checks["scene_identity_consistent"] = all(
        payload.get("scene_id") == config["scene"]["scene_id"]
        for payload in scene_payloads.values()
    )
    checks["scene_inputs_passed"] = all(
        payload.get("passed") is True for payload in scene_payloads.values()
    )
    scene_contract = audit_realistic_route_scene_contract(
        scene_payloads["terrain_compiled_audit"]
    )
    checks["scene_route_contract_passed"] = scene_contract.get("passed") is True

    planned_outputs: list[str] = []
    for operator, spec in config.get("operators", {}).items():
        for key in ("nominal_output", "anomaly_output", "pair_audit"):
            path = _resolve(spec[key])
            checks[f"{operator}_{key}_unused"] = not path.exists()
            planned_outputs.append(str(path))
    checks["three_operators_frozen"] = set(config.get("operators", {})) == {"o1", "o2", "o3"}

    result = {
        "schema_version": "kinofail.embodiedgen-terrain-operator-v5-calibration-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "frozen_inputs": frozen_inputs,
        "scene_contract": scene_contract,
        "planned_outputs": planned_outputs,
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
