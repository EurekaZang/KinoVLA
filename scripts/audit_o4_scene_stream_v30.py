#!/usr/bin/env python3
"""Audit adapter for the v30 explicit Isaac Kit USD launcher."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_o4_scene_stream_v29 as predecessor


implementation = predecessor.implementation


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
    base["schema_version"] = implementation.SCHEMA
    base["freeze_status"] = implementation.FREEZE_STATUS
    base["pipeline"]["candidate_runner"] = "scripts/run_o4_scene_candidate_v28.py"
    return base


def _json_v30(path: Path) -> dict[str, Any]:
    value = predecessor._original_json(path)
    schema = value.get("schema_version")
    if schema == "kinofail.o4-scene-stream-v30-freeze.v1":
        return _expand_config(value)
    if schema == "kinofail.o4-scene-candidate-v30-receipt.v1":
        value = copy.deepcopy(value)
        value["schema_version"] = "kinofail.o4-scene-candidate-v28-receipt.v1"
    return value


def _candidate_paths_v30(
    config: dict[str, Any], scene: dict[str, Any]
) -> dict[str, Path]:
    paths = predecessor._original_candidate_paths(config, scene)
    paths["receipt"] = Path(
        str(paths["receipt"]).replace(
            "_v28_candidate_receipt.json", "_v30_candidate_receipt.json"
        )
    )
    return paths


def main() -> int:
    predecessor._json_v29 = _json_v30
    predecessor._candidate_paths_v29 = _candidate_paths_v30
    code = predecessor.main()
    out = predecessor._out_path()
    payload = json.loads(out.read_text(encoding="utf-8"))
    payload["schema_version"] = str(payload["schema_version"]).replace(
        "v29", "v30"
    )
    payload["audit_adapter"] = "v30_explicit_isaac_kit_usd_python_path"
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
