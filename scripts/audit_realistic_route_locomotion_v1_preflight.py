#!/usr/bin/env python3
"""Fail-closed preflight for isolated realistic-route locomotion training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


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


def _bound_file(spec: dict[str, Any]) -> bool:
    path = _resolve(spec["path"])
    return path.is_file() and _sha256(path) == spec["sha256"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze",
        type=Path,
        default=ROOT
        / "configs/locomotion/go2_realistic_route_ppo_v1_freeze.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/eval/realistic_route_locomotion_v1_training_preflight.json",
    )
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    freeze = _json(freeze_path)
    boundary = freeze.get("evidence_boundary", {})
    contract = freeze.get("training_contract", {})
    config_path = _resolve(freeze["source_freeze"]["training_config"]["path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(config_path)
    command = config.get("command", {})
    dr = config.get("dr", {})
    checks: dict[str, bool] = {
        "supported_schema": freeze.get("schema_version")
        == "kinofail.realistic-route-locomotion-training-freeze.v1",
        "training_only": boundary.get("training_only") is True,
        "not_a0_a7_evidence": boundary.get("counts_as_a0_a7_evidence") is False
        and boundary.get("realistic_a0_a7_readiness") == "0/8",
        "full_realistic_replication_required": boundary.get(
            "completion_requires_new_realistic_corpus_training_inference_and_a0_a7"
        )
        is True,
        "training_script_hash": _bound_file(freeze["source_freeze"]["training_script"]),
        "training_config_hash": _bound_file(freeze["source_freeze"]["training_config"]),
        "shipped_policy_preserved": _bound_file(
            freeze["preserved_legacy_artifacts"]["shipped_policy"]
        ),
        "direct_velocity_policy_preserved": _bound_file(
            freeze["preserved_legacy_artifacts"]["direct_velocity_policy"]
        ),
        "output_dir_unused": not _resolve(freeze["planned_output_dir"]).exists(),
        "task_bound": config.get("task") == contract.get("task"),
        "scale_bound": int(config.get("num_envs", -1)) == int(contract.get("num_envs", -2))
        and int(config.get("max_iterations", -1))
        == int(contract.get("max_iterations", -2))
        and int(config.get("seed", -1)) == int(contract.get("seed", -2)),
        "direct_commands_bound": command.get("heading_command") is False
        and list(command.get("lin_vel_x", [])) == contract.get("forward_speed_range_mps")
        and list(command.get("lin_vel_y", [])) == contract.get("lateral_speed_range_mps")
        and list(command.get("ang_vel_z", [])) == contract.get("yaw_rate_range_radps")
        and float(command.get("rel_standing_envs", -1.0))
        == float(contract.get("standing_fraction", -2.0)),
        "domain_randomization_bound": bool(dr.get("enable_push"))
        and float(dr.get("static_friction_range", [1.0])[0]) <= 0.08
        and float(dr.get("add_base_mass_range", [0.0, 0.0])[0]) < 0.0
        and float(dr.get("add_base_mass_range", [0.0, 0.0])[1]) > 0.0,
        "post_training_gates_required": contract.get("post_training_route_gates_required")
        is True,
    }
    failure_payloads: dict[str, Any] = {}
    for name, spec in freeze.get("failure_basis", {}).items():
        checks[f"{name}_hash"] = _bound_file(spec)
        payload = _json(_resolve(spec["path"]))
        failure_payloads[name] = payload
        checks[f"{name}_audit_passed"] = payload.get("audit_passed") is True
    checks["both_prior_policies_rejected"] = (
        failure_payloads["shipped_policy_screen"].get("development_matrix_passed") is False
        and failure_payloads["direct_velocity_policy_screen"].get("policy_screen_passed")
        is False
    )

    result = {
        "schema_version": "kinofail.realistic-route-locomotion-training-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "freeze": {"path": str(freeze_path), "sha256": _sha256(freeze_path)},
        "planned_command": [
            sys.executable,
            "scripts/train_locomotion.py",
            "--config",
            "locomotion/go2_realistic_route_ppo_v1.yaml",
            "--headless",
        ],
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
