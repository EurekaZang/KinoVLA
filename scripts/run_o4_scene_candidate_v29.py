#!/usr/bin/env python3
"""v29 launcher correction for the frozen v28 candidate implementation.

The v28 runner is preserved byte-for-byte as invalidated evidence.  This wrapper
changes only two administrative/runtime details: USD-aware Python stages are
launched after Isaac Sim's environment setup, and receipts receive a v29 name.
All scene-generation, geometry, rendering, motion, and admission logic remains
in the locked v28 implementation.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_o4_scene_candidate_v28 as implementation


PURE_USD_STAGES = {
    "source_preflight",
    "base_compile",
    "corridor",
    "route",
    "terrain_compile",
    "terrain_offline",
    "stack_v16",
    "stack_v17",
}
_original_json = implementation._json
_original_run_stage = implementation._run_stage
_original_write_receipt = implementation._write_receipt


def _json_v29(path: Path) -> dict[str, Any]:
    value = _original_json(path)
    if value.get("schema_version") == "kinofail.o4-scene-stream-v29-freeze.v1":
        base_spec = value["base_config"]
        base_path = implementation._resolve(base_spec["path"])
        if implementation._sha256(base_path) != base_spec["sha256"]:
            raise RuntimeError("v29 base config hash mismatch")
        base = copy.deepcopy(_original_json(base_path))
        candidate_map = {
            item["scene_id"]: item for item in base["candidate_stream"]
        }
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
        value = base
    return value


def _run_stage_v29(
    *,
    name: str,
    command: list[str],
    audit_path: Path,
    environment: dict[str, str],
):
    if name in PURE_USD_STAGES:
        command = implementation._isaac_command(Path(command[1]), command[2:])
    return _original_run_stage(
        name=name,
        command=command,
        audit_path=audit_path,
        environment=environment,
    )


def _write_receipt_v29(path: Path, payload: dict[str, Any]) -> None:
    corrected_path = Path(
        str(path).replace("_v28_candidate_receipt.json", "_v29_candidate_receipt.json")
    )
    corrected_payload = {
        **payload,
        "schema_version": "kinofail.o4-scene-candidate-v29-receipt.v1",
        "launcher_revision": "v29_usd_stages_source_isaac_setup",
        "v28_candidate_runner_implementation_reused": True,
    }
    _original_write_receipt(corrected_path, corrected_payload)


def main() -> int:
    implementation._json = _json_v29
    implementation._run_stage = _run_stage_v29
    implementation._write_receipt = _write_receipt_v29
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
