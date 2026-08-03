#!/usr/bin/env python3
"""Fail-closed preflight for fresh O4 held-out realistic-scene generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _locked(record: dict[str, Any]) -> bool:
    path = _resolve(record["path"])
    if not path.is_file() or _sha256(path) != record["sha256"]:
        return False
    return "json_passed" not in record or _json(path).get("passed") is record["json_passed"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _json(config_path)
    source_root = _resolve(config["pipeline"]["source_root"])
    operator_root = _resolve(config["operator_output_root"])
    material_lock = _json(_resolve(config["material_lock"]["path"]))
    materials = {item["id"]: item for item in material_lock.get("materials", [])}
    scenes = config.get("requested_scenes", [])

    checks: dict[str, bool] = {
        "supported_schema": config.get("schema_version")
        == "kinofail.o4-heldout-scene-generation-v22-freeze.v1",
        "frozen_before_generation": config.get("freeze_status")
        == "frozen_before_any_requested_scene_generation_or_o4_execution",
        "strict_evidence_quarantine": config.get("evidence_policy", {}).get(
            "counts_as_a0_a7_evidence"
        )
        is False
        and config.get("evidence_policy", {}).get("development_only") is True
        and config.get("evidence_policy", {}).get("a8_in_scope") is False,
        "v21_is_required_and_passed": config.get("selection_policy", {}).get(
            "requires_sealed_v21_train_only_pass"
        )
        is True,
        "four_fixed_requests": len(scenes) == 4
        and [item.get("scene_id") for item in scenes]
        == config.get("selection_policy", {}).get("fixed_order"),
        "four_distinct_families": len({item.get("room_type") for item in scenes}) == 4,
        "one_attempt_no_replacement": config.get("selection_policy", {}).get(
            "one_generation_attempt_per_scene"
        )
        is True
        and config.get("selection_policy", {}).get("replacement_seed_allowed") is False
        and config.get("selection_policy", {}).get("source_repair_allowed") is False,
        "minimum_confirmation_is_multiscene": config.get("selection_policy", {}).get(
            "minimum_admitted_scenes"
        )
        >= 2
        and config.get("selection_policy", {}).get("minimum_admitted_families") >= 2,
        "no_o4_before_batch_adjudication": config.get("selection_policy", {}).get(
            "no_o4_execution_before_batch_adjudication"
        )
        is True,
        "operator_output_root_unused": not operator_root.exists(),
        "locked_files_match": all(_locked(item) for item in config.get("locked_files", [])),
        "material_lock_matches": _locked(config["material_lock"]),
    }
    for scene in scenes:
        slug = scene["scene_id"]
        output = source_root / scene["output_directory"]
        request = source_root / f"{scene['output_directory']}_request.json"
        material = materials.get(scene["material_id"], {})
        checks[f"{slug}_source_unused"] = not output.exists()
        checks[f"{slug}_request_unused"] = not request.exists()
        checks[f"{slug}_material_frozen"] = material.get("id") == scene[
            "material_id"
        ] and material.get("split") == scene["material_split"]
        checks[f"{slug}_seed_positive"] = int(scene.get("source_seed", 0)) > 0

    passed = all(checks.values())
    audit = {
        "schema_version": "kinofail.o4-heldout-scene-generation-v22-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": passed,
        "generation_authorized": passed,
        "o4_execution_authorized": False,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checks": checks,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
    }
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "passed": passed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
