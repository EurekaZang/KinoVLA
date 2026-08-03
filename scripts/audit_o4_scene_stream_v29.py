#!/usr/bin/env python3
"""Versioned audit adapter for the v29 scene stream launcher correction."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_o4_scene_stream_v28 as implementation


_original_json = implementation._json
_original_candidate_paths = implementation._candidate_paths


def _json_v29(path: Path) -> dict[str, Any]:
    value = _original_json(path)
    schema = value.get("schema_version")
    if schema == "kinofail.o4-scene-stream-v29-freeze.v1":
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
        base["schema_version"] = implementation.SCHEMA
        base["freeze_status"] = implementation.FREEZE_STATUS
        base["pipeline"]["candidate_runner"] = "scripts/run_o4_scene_candidate_v28.py"
        value = base
    elif schema == "kinofail.o4-scene-candidate-v29-receipt.v1":
        value = copy.deepcopy(value)
        value["schema_version"] = "kinofail.o4-scene-candidate-v28-receipt.v1"
    return value


def _candidate_paths_v29(
    config: dict[str, Any], scene: dict[str, Any]
) -> dict[str, Path]:
    paths = _original_candidate_paths(config, scene)
    paths["receipt"] = Path(
        str(paths["receipt"]).replace(
            "_v28_candidate_receipt.json", "_v29_candidate_receipt.json"
        )
    )
    return paths


def _out_path() -> Path:
    try:
        index = sys.argv.index("--out")
        return Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise RuntimeError("--out is required") from exc


def main() -> int:
    implementation._json = _json_v29
    implementation._candidate_paths = _candidate_paths_v29
    code = implementation.main()
    out = _out_path()
    payload = json.loads(out.read_text(encoding="utf-8"))
    payload["schema_version"] = str(payload["schema_version"]).replace(
        "v28", "v29"
    )
    payload["audit_adapter"] = "v29_usd_stage_launcher_correction"
    payload["v28_audit_implementation_reused"] = True
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
