#!/usr/bin/env python3
"""v30 USD launcher fix, preserving the sealed v28/v29 implementations."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_o4_scene_candidate_v29 as predecessor


implementation = predecessor.implementation
KIT_SITE_PACKAGES = Path(
    "/home/eureka/nvidia/isaacsim/kit/python/lib/python3.11/site-packages"
)


def _expand_config(value: dict[str, Any]) -> dict[str, Any]:
    base_spec = value["base_config"]
    base_path = implementation._resolve(base_spec["path"])
    if implementation._sha256(base_path) != base_spec["sha256"]:
        raise RuntimeError("v30 base config hash mismatch")
    base = copy.deepcopy(predecessor._original_json(base_path))
    candidate_map = {item["scene_id"]: item for item in base["candidate_stream"]}
    candidate_map.update(
        {item["scene_id"]: item for item in value.get("new_candidates", [])}
    )
    candidate_ids = value["candidate_ids"]
    base["candidate_stream"] = [candidate_map[item] for item in candidate_ids]
    base["selection_policy"].update(value["selection_policy_overrides"])
    base["selection_policy"]["fixed_order"] = candidate_ids
    base["pipeline"].update(value["pipeline_overrides"])
    base["operator_output_root"] = value["operator_output_root"]
    base["locked_files"] = [
        *base["locked_files"],
        *value["additional_locked_files"],
    ]
    base["purpose"] = value["purpose"]
    base["evidence_policy"] = value["evidence_policy"]
    base["schema_version"] = "kinofail.o4-scene-stream-v28-freeze.v1"
    base["freeze_status"] = (
        "frozen_before_any_v28_candidate_generation_or_o4_execution"
    )
    return base


def _json_v30(path: Path) -> dict[str, Any]:
    value = predecessor._original_json(path)
    if value.get("schema_version") == "kinofail.o4-scene-stream-v30-freeze.v1":
        return _expand_config(value)
    return value


def _usd_command(script: Path, arguments: list[str]) -> list[str]:
    return [
        str(implementation.SHELL),
        "-lc",
        'source "$1"; export PYTHONPATH="$2${PYTHONPATH:+:$PYTHONPATH}"; shift 2; exec "$@"',
        "kinofail-v30-scene-candidate",
        str(implementation.ISAAC_SETUP),
        str(KIT_SITE_PACKAGES),
        str(implementation.ISAAC_PYTHON),
        str(script),
        *arguments,
    ]


def _run_stage_v30(
    *,
    name: str,
    command: list[str],
    audit_path: Path,
    environment: dict[str, str],
):
    if name in predecessor.PURE_USD_STAGES:
        command = _usd_command(Path(command[1]), command[2:])
    return predecessor._original_run_stage(
        name=name,
        command=command,
        audit_path=audit_path,
        environment=environment,
    )


def _write_receipt_v30(path: Path, payload: dict[str, Any]) -> None:
    corrected_path = Path(
        str(path).replace("_v28_candidate_receipt.json", "_v30_candidate_receipt.json")
    )
    corrected_payload = {
        **payload,
        "schema_version": "kinofail.o4-scene-candidate-v30-receipt.v1",
        "launcher_revision": "v30_explicit_isaac_kit_usd_python_path",
        "kit_site_packages": str(KIT_SITE_PACKAGES),
        "v28_candidate_runner_implementation_reused": True,
    }
    predecessor._original_write_receipt(corrected_path, corrected_payload)


def main() -> int:
    if not KIT_SITE_PACKAGES.joinpath("pxr").is_dir():
        raise FileNotFoundError(KIT_SITE_PACKAGES / "pxr")
    predecessor._json_v29 = _json_v30
    predecessor._run_stage_v29 = _run_stage_v30
    predecessor._write_receipt_v29 = _write_receipt_v30
    return predecessor.main()


if __name__ == "__main__":
    raise SystemExit(main())
