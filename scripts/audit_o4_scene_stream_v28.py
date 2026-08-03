#!/usr/bin/env python3
"""Preflight and seal the v28 operator-blind realistic-scene stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "kinofail.o4-scene-stream-v28-freeze.v1"
FREEZE_STATUS = "frozen_before_any_v28_candidate_generation_or_o4_execution"
STAGE_ORDER = [
    "generation",
    "source_preflight",
    "base_compile",
    "corridor",
    "route",
    "rtx",
    "motion_proxy",
    "terrain_compile",
    "terrain_offline",
    "go2",
    "stack_v16",
    "stack_v17",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _locked(record: dict[str, Any]) -> bool:
    path = _resolve(record["path"])
    if not path.is_file() or _sha256(path) != record["sha256"]:
        return False
    value = _json(path) if "json_passed" in record or "json_sealed" in record else {}
    return (
        ("json_passed" not in record or value.get("passed") is record["json_passed"])
        and ("json_sealed" not in record or value.get("sealed") is record["json_sealed"])
    )


def _candidate_paths(
    config: dict[str, Any], scene: dict[str, Any]
) -> dict[str, Path]:
    pipeline = config["pipeline"]
    source_root = _resolve(pipeline["source_root"])
    source = source_root / scene["output_directory"]
    route = source / pipeline["route_output_name"] / scene["material_id"]
    terrain = route / pipeline["terrain_output_name"]
    return {
        "source": source,
        "request": source_root / f"{scene['output_directory']}_request.json",
        "generation_exception": source_root
        / f"{scene['output_directory']}_generation_exception_audit.json",
        "receipt": source_root
        / f"{scene['output_directory']}_v28_candidate_receipt.json",
        "source_integrity": source / "source_integrity_audit.json",
        "source_preflight": source
        / "source_geometry_preflight_v1/source_geometry_preflight_audit.json",
        "base_compile": source
        / pipeline["base_output_name"]
        / "compiled_scene_audit.json",
        "corridor": source
        / pipeline["corridor_output_name"]
        / "compiled_scene_audit.json",
        "route": route / "compiled_scene_audit.json",
        "rtx": route / "rtx_qa/rtx_scene_audit.json",
        "motion_proxy": route
        / pipeline["motion_proxy_output_name"]
        / "route_motion_rtx_audit.json",
        "terrain_compile": terrain / "compiled_scene_audit.json",
        "terrain_offline": terrain / "offline_usd_audit.json",
        "go2_launcher": terrain / "go2_qa/go2_launcher_audit.json",
        "go2_native": terrain / "go2_qa/go2_scene_audit.json",
        "go2_exception": terrain / "go2_qa/go2_scene_exception_audit.json",
        "stack_v16": terrain / "realistic_stack_v16_admission.json",
        "stack_v17": terrain / "realistic_stack_v17_admission.json",
        "episode_usd": terrain / "episode_terrain_v2.usda",
    }


def _preflight(config_path: Path, out: Path) -> int:
    config = _json(config_path)
    pipeline = config.get("pipeline", {})
    policy = config.get("selection_policy", {})
    scenes = config.get("candidate_stream", [])
    material_lock = _json(_resolve(config.get("material_lock", {}).get("path", "")))
    materials = {item.get("id"): item for item in material_lock.get("materials", [])}
    ids = [scene.get("scene_id") for scene in scenes]
    families = Counter(scene.get("room_type") for scene in scenes)
    checks: dict[str, bool] = {
        "supported_schema": config.get("schema_version") == SCHEMA,
        "frozen_before_stream_execution": config.get("freeze_status")
        == FREEZE_STATUS,
        "strict_evidence_quarantine": config.get("evidence_policy", {}).get(
            "counts_as_a0_a7_evidence"
        )
        is False
        and config.get("evidence_policy", {}).get("development_only") is True
        and config.get("evidence_policy", {}).get("a8_in_scope") is False,
        "twenty_four_candidate_fixed_order": len(scenes) == 24
        and ids == policy.get("fixed_order")
        and len(set(ids)) == 24,
        "balanced_six_family_stream": set(families)
        == {"Office", "LivingRoom", "Kitchen", "House", "Bedroom", "DiningRoom"}
        and all(count == 4 for count in families.values()),
        "strict_prefix_stop": policy.get("processing_order")
        == "strict_candidate_stream_order"
        and policy.get("stop_at_first_prefix_meeting_quota") is True
        and policy.get("maximum_candidates") == 24,
        "publication_scale_acquisition_quota": policy.get(
            "minimum_admitted_scenes"
        )
        == 5
        and policy.get("minimum_admitted_families") == 4,
        "one_attempt_no_repair": policy.get("one_generation_attempt_per_candidate")
        is True
        and policy.get("replacement_seed_allowed") is False
        and policy.get("source_repair_allowed") is False,
        "operator_blind_acquisition": policy.get("o4_withheld_until_cohort_frozen")
        is True,
        "simple_only": all(scene.get("complexity") == "simple" for scene in scenes),
        "unique_positive_seeds": len(
            {scene.get("source_seed") for scene in scenes}
        )
        == len(scenes)
        and all(isinstance(scene.get("source_seed"), int) and scene["source_seed"] > 0 for scene in scenes),
        "locked_files_match": bool(config.get("locked_files"))
        and all(_locked(item) for item in config.get("locked_files", [])),
        "material_lock_matches": _locked(config.get("material_lock", {})),
        "operator_output_root_unused": not _resolve(
            config.get("operator_output_root", "")
        ).exists(),
        "audited_candidate_runner_only": pipeline.get("candidate_runner")
        == "scripts/run_o4_scene_candidate_v28.py",
    }
    for scene in scenes:
        paths = _candidate_paths(config, scene)
        material = materials.get(scene.get("material_id"), {})
        checks[f"{scene['scene_id']}_unused"] = not any(
            paths[name].exists()
            for name in ("source", "request", "generation_exception", "receipt")
        )
        checks[f"{scene['scene_id']}_material_frozen"] = (
            material.get("id") == scene.get("material_id")
            and material.get("split") == scene.get("material_split")
        )
    passed = all(checks.values())
    result = {
        "schema_version": "kinofail.o4-scene-stream-v28-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checks": checks,
        "passed": passed,
        "scene_stream_execution_authorized": passed,
        "o4_execution_authorized": False,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
    }
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "passed": passed}, indent=2))
    return 0 if passed else 2


def _receipt_integrity(
    *,
    config_hash: str,
    scene: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[bool, dict[str, bool]]:
    receipt = _json(paths["receipt"])
    stages = receipt.get("stages", [])
    names = [item.get("stage") for item in stages if isinstance(item, dict)]
    prefix = names == STAGE_ORDER[: len(names)] and len(names) == len(set(names))
    artifact_hashes = bool(stages) and all(
        isinstance(item, dict)
        and Path(item.get("audit", "")).is_file()
        and _sha256(Path(item["audit"])) == item.get("audit_sha256")
        for item in stages
    )
    admitted = receipt.get("fully_admitted") is True
    terminal_semantics = (
        admitted
        and receipt.get("passed") is True
        and names == STAGE_ORDER
        and all(item.get("passed") is True for item in stages)
        and _json(paths["stack_v17"]).get("passed") is True
    ) or (
        not admitted
        and receipt.get("passed") is False
        and bool(stages)
        and (
            receipt.get("terminal_stage") == "runner_exception"
            or receipt.get("terminal_stage") == names[-1]
        )
    )
    checks = {
        "receipt_exists": paths["receipt"].is_file(),
        "receipt_schema": receipt.get("schema_version")
        == "kinofail.o4-scene-candidate-v28-receipt.v1",
        "receipt_identity": receipt.get("scene_id") == scene.get("scene_id")
        and receipt.get("room_type") == scene.get("room_type")
        and receipt.get("source_seed") == scene.get("source_seed")
        and receipt.get("runtime_seed") == scene.get("runtime_seed")
        and receipt.get("material_id") == scene.get("material_id"),
        "receipt_binds_config": receipt.get("config_sha256") == config_hash,
        "receipt_sealed": receipt.get("sealed") is True,
        "operator_blind": receipt.get("o4_outcomes_observed") is False
        and receipt.get("counts_as_a0_a7_evidence") is False,
        "stages_form_frozen_prefix": prefix,
        "recorded_artifact_hashes_match": artifact_hashes,
        "terminal_semantics": terminal_semantics,
    }
    return all(checks.values()), checks


def _postrun(config_path: Path, preflight_path: Path, out: Path) -> int:
    config = _json(config_path)
    preflight = _json(preflight_path)
    config_hash = _sha256(config_path)
    rows: list[dict[str, Any]] = []
    for scene in config.get("candidate_stream", []):
        paths = _candidate_paths(config, scene)
        processed = any(
            paths[name].exists()
            for name in ("source", "request", "generation_exception", "receipt")
        )
        receipt = _json(paths["receipt"])
        admitted = processed and receipt.get("fully_admitted") is True
        integrity, receipt_checks = (
            _receipt_integrity(
                config_hash=config_hash, scene=scene, paths=paths
            )
            if processed
            else (True, {})
        )
        rows.append(
            {
                "scene_id": scene["scene_id"],
                "room_type": scene["room_type"],
                "source_seed": scene["source_seed"],
                "runtime_seed": scene["runtime_seed"],
                "material_id": scene["material_id"],
                "processed": processed,
                "terminal": processed and receipt.get("sealed") is True,
                "receipt_integrity_passed": integrity,
                "receipt_checks": receipt_checks,
                "first_failure_stage": None
                if admitted
                else receipt.get("terminal_stage"),
                "fully_admitted": admitted,
                "receipt": str(paths["receipt"]),
                "receipt_sha256": _sha256(paths["receipt"])
                if paths["receipt"].is_file()
                else None,
                "episode_usd": str(paths["episode_usd"])
                if admitted
                else None,
                "terrain_compiled_audit": {
                    "path": str(paths["terrain_compile"]),
                    "sha256": _sha256(paths["terrain_compile"])
                    if admitted
                    else None,
                    "passed": _json(paths["terrain_compile"]).get("passed"),
                },
                "stack_v17_admission": {
                    "path": str(paths["stack_v17"]),
                    "sha256": _sha256(paths["stack_v17"]) if admitted else None,
                    "passed": _json(paths["stack_v17"]).get("passed"),
                },
            }
        )

    processed_count = sum(row["processed"] for row in rows)
    prefix_shape = [row["processed"] for row in rows] == [
        index < processed_count for index in range(len(rows))
    ]
    policy = config["selection_policy"]
    first_quota_prefix: int | None = None
    admitted_count = 0
    admitted_families: set[str] = set()
    for index, row in enumerate(rows):
        if row["fully_admitted"]:
            admitted_count += 1
            admitted_families.add(row["room_type"])
        if (
            admitted_count >= policy["minimum_admitted_scenes"]
            and len(admitted_families) >= policy["minimum_admitted_families"]
        ):
            first_quota_prefix = index + 1
            break
    quota_reached = first_quota_prefix is not None
    exact_stop = (
        quota_reached and processed_count == first_quota_prefix
    ) or (
        not quota_reached and processed_count == policy["maximum_candidates"]
    )
    processed_rows = rows[:processed_count]
    suffix_rows = rows[processed_count:]
    checks = {
        "preflight_passed_and_bound": preflight.get("passed") is True
        and preflight.get("config_sha256") == config_hash,
        "locked_files_unchanged": all(_locked(item) for item in config["locked_files"]),
        "processed_candidates_form_prefix": prefix_shape,
        "processed_prefix_is_terminal": bool(processed_rows)
        and all(row["terminal"] for row in processed_rows),
        "all_processed_receipts_integrity_passed": all(
            row["receipt_integrity_passed"] for row in processed_rows
        ),
        "unprocessed_suffix_untouched": all(
            not row["processed"] for row in suffix_rows
        ),
        "exact_registered_stopping_rule": exact_stop,
        "no_o4_outcomes_exist": not _resolve(config["operator_output_root"]).exists(),
    }
    passed = all(checks.values()) and quota_reached
    sealed = all(checks.values())
    admitted_rows = [row for row in processed_rows if row["fully_admitted"]]
    result = {
        "schema_version": "kinofail.o4-scene-stream-v28-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "config_sha256": config_hash,
        "preflight": str(preflight_path),
        "preflight_sha256": _sha256(preflight_path),
        "checks": checks,
        "passed": passed,
        "sealed": sealed,
        "maximum_candidates": len(rows),
        "processed_candidates": processed_count,
        "first_quota_prefix": first_quota_prefix,
        "admitted": len(admitted_rows),
        "admitted_scene_ids": [row["scene_id"] for row in admitted_rows],
        "admitted_families": sorted({row["room_type"] for row in admitted_rows}),
        "scenes": rows,
        "cohort_frozen": passed,
        "v27_heldout_confirmation_authorized": passed,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "next_gate": "Freeze the unchanged v27 runtime against every admitted scene and execute exactly once per scene.",
    }
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "passed": passed,
                "sealed": sealed,
                "processed": processed_count,
                "admitted": len(admitted_rows),
            },
            indent=2,
        )
    )
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "postrun"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = args.config.resolve()
    out = args.out.resolve()
    if args.mode == "preflight":
        if args.preflight is not None:
            parser.error("--preflight is valid only in postrun mode")
        return _preflight(config, out)
    if args.preflight is None:
        parser.error("--preflight is required in postrun mode")
    return _postrun(config, args.preflight.resolve(), out)


if __name__ == "__main__":
    raise SystemExit(main())
