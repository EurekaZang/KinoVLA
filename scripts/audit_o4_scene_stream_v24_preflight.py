#!/usr/bin/env python3
"""Fail-closed preflight for the deterministic v24 realistic-scene stream."""

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
        raise TypeError(path)
    return value


def _locked(record: dict[str, Any]) -> bool:
    path = _resolve(record["path"])
    if not path.is_file() or _sha256(path) != record["sha256"]:
        return False
    value = _json(path) if "json_passed" in record or "json_sealed" in record else {}
    return (
        ("json_passed" not in record or value.get("passed") is record["json_passed"])
        and ("json_sealed" not in record or value.get("sealed") is record["json_sealed"])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _json(config_path)
    pipeline = config["pipeline"]
    policy = config["selection_policy"]
    scenes = config["candidate_stream"]
    source_root = _resolve(pipeline["source_root"])
    operator_root = _resolve(config["operator_output_root"])
    material_lock = _json(_resolve(config["material_lock"]["path"]))
    materials = {item["id"]: item for item in material_lock.get("materials", [])}
    scene_ids = [item.get("scene_id") for item in scenes]
    seeds = [item.get("source_seed") for item in scenes]

    checks: dict[str, bool] = {
        "supported_schema": config.get("schema_version")
        == "kinofail.o4-scene-stream-v24-freeze.v1",
        "frozen_before_stream_execution": config.get("freeze_status")
        == "frozen_before_any_v24_candidate_generation_or_o4_execution",
        "strict_evidence_quarantine": config.get("evidence_policy", {}).get(
            "counts_as_a0_a7_evidence"
        )
        is False
        and config.get("evidence_policy", {}).get("development_only") is True
        and config.get("evidence_policy", {}).get("a8_in_scope") is False,
        "twelve_candidate_fixed_order": len(scenes) == 12
        and scene_ids == policy.get("fixed_order")
        and len(set(scene_ids)) == 12,
        "deterministic_prefix_stop": policy.get("processing_order")
        == "strict_candidate_stream_order"
        and policy.get("stop_at_first_prefix_meeting_quota") is True
        and policy.get("maximum_candidates") == 12,
        "publication_scale_acquisition_quota": policy.get("minimum_admitted_scenes") == 5
        and policy.get("minimum_admitted_families") == 4,
        "one_attempt_no_repair": policy.get("one_generation_attempt_per_candidate")
        is True
        and policy.get("replacement_seed_allowed") is False
        and policy.get("source_repair_allowed") is False,
        "operator_blind_acquisition": policy.get("o4_withheld_until_cohort_frozen") is True,
        "family_diversity": len({item.get("room_type") for item in scenes}) >= 5
        and all(item.get("room_type") != "Bathroom" for item in scenes),
        "unique_positive_seeds": len(set(seeds)) == len(seeds)
        and all(isinstance(seed, int) and seed > 0 for seed in seeds),
        "operator_output_root_unused": not operator_root.exists(),
        "locked_files_match": all(_locked(item) for item in config["locked_files"]),
        "material_lock_matches": _locked(config["material_lock"]),
    }
    for scene in scenes:
        source = source_root / scene["output_directory"]
        request = source_root / f"{scene['output_directory']}_request.json"
        exception = source_root / f"{scene['output_directory']}_generation_exception_audit.json"
        material = materials.get(scene["material_id"], {})
        checks[f"{scene['scene_id']}_unused"] = not source.exists() and not request.exists() and not exception.exists()
        checks[f"{scene['scene_id']}_material_frozen"] = (
            material.get("id") == scene["material_id"]
            and material.get("split") == scene["material_split"]
        )

    passed = all(checks.values())
    result = {
        "schema_version": "kinofail.o4-scene-stream-v24-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": passed,
        "scene_stream_execution_authorized": passed,
        "o4_execution_authorized": False,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checks": checks,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
    }
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "passed": passed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
