#!/usr/bin/env python3
"""Seal the first terminal prefix of the v24 realistic-scene stream."""

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


def _artifact(path: Path) -> dict[str, Any]:
    value = _json(path)
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": _sha256(path) if path.is_file() else None,
        "passed": value.get("passed"),
        "schema_version": value.get("schema_version"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    preflight_path = args.preflight.resolve()
    config = _json(config_path)
    preflight = _json(preflight_path)
    pipeline = config["pipeline"]
    policy = config["selection_policy"]
    source_root = _resolve(pipeline["source_root"])
    rows = []
    stage_order = [
        "generation_exception",
        "source_integrity",
        "source_preflight",
        "base_compile",
        "corridor",
        "route",
        "rtx",
        "motion_proxy",
        "terrain_compile",
        "terrain_offline",
        "go2_exception",
        "go2",
        "stack_v16",
        "stack_v17",
    ]
    for scene in config["candidate_stream"]:
        source = source_root / scene["output_directory"]
        request = source_root / f"{scene['output_directory']}_request.json"
        generation_exception = source_root / f"{scene['output_directory']}_generation_exception_audit.json"
        route = source / pipeline["route_output_name"] / scene["material_id"]
        terrain = route / pipeline["terrain_output_name"]
        artifacts = {
            "request": _artifact(request),
            "generation_exception": _artifact(generation_exception),
            "source_integrity": _artifact(source / "source_integrity_audit.json"),
            "source_preflight": _artifact(source / "source_geometry_preflight_v1/source_geometry_preflight_audit.json"),
            "base_compile": _artifact(source / pipeline["base_output_name"] / "compiled_scene_audit.json"),
            "corridor": _artifact(source / pipeline["corridor_output_name"] / "compiled_scene_audit.json"),
            "route": _artifact(route / "compiled_scene_audit.json"),
            "rtx": _artifact(route / "rtx_qa/rtx_scene_audit.json"),
            "motion_proxy": _artifact(route / "motion_rtx_proxy_v2_extrinsics/route_motion_rtx_audit.json"),
            "terrain_compile": _artifact(terrain / "compiled_scene_audit.json"),
            "terrain_offline": _artifact(terrain / "offline_usd_audit.json"),
            "go2_exception": _artifact(terrain / "go2_qa/go2_scene_exception_audit.json"),
            "go2": _artifact(terrain / "go2_qa/go2_scene_audit.json"),
            "stack_v16": _artifact(terrain / "realistic_stack_v16_admission.json"),
            "stack_v17": _artifact(terrain / "realistic_stack_v17_admission.json"),
        }
        processed = artifacts["request"]["exists"] or source.exists() or artifacts["generation_exception"]["exists"]
        manifest = _json(source / "source_manifest.json")
        generation_failed = artifacts["generation_exception"]["passed"] is False
        identity_ok = generation_failed or (
            manifest.get("scene_id") == scene["scene_id"]
            and manifest.get("generation", {}).get("room_type") == scene["room_type"]
            and manifest.get("generation", {}).get("seed") == scene["source_seed"]
            and manifest.get("generation", {}).get("complexity") == scene["complexity"]
        )
        first_failure = next(
            (name for name in stage_order if artifacts[name]["passed"] is False), None
        )
        admitted = artifacts["stack_v17"]["passed"] is True
        terminal = processed and (admitted or first_failure is not None)
        failure_closed = True
        if first_failure is not None:
            failure_index = stage_order.index(first_failure)
            failure_closed = not any(
                artifacts[name]["exists"] for name in stage_order[failure_index + 1 :]
            )
        if admitted:
            failure_closed = all(
                artifacts[name]["passed"] is True
                for name in stage_order
                if name not in {"generation_exception", "go2_exception"}
            ) and not artifacts["generation_exception"]["exists"] and not artifacts["go2_exception"]["exists"]
        rows.append(
            {
                "scene_id": scene["scene_id"],
                "room_type": scene["room_type"],
                "source_seed": scene["source_seed"],
                "runtime_seed": scene["runtime_seed"],
                "material_id": scene["material_id"],
                "processed": processed,
                "source_identity_matches": identity_ok if processed else True,
                "terminal": terminal,
                "first_failure_stage": first_failure,
                "failure_closed": failure_closed,
                "fully_admitted": admitted,
                "artifacts": artifacts,
            }
        )

    processed_flags = [row["processed"] for row in rows]
    processed_count = sum(processed_flags)
    prefix_shape = processed_flags == [index < processed_count for index in range(len(rows))]
    first_quota_prefix = None
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
    ) or (not quota_reached and processed_count == policy["maximum_candidates"])
    processed_rows = rows[:processed_count]
    suffix_rows = rows[processed_count:]
    operator_root = _resolve(config["operator_output_root"])
    checks = {
        "preflight_passed": preflight.get("passed") is True,
        "preflight_binds_config": preflight.get("config_sha256") == _sha256(config_path),
        "locked_files_unchanged": all(_locked(item) for item in config["locked_files"]),
        "processed_candidates_form_prefix": prefix_shape,
        "processed_prefix_is_terminal": bool(processed_rows)
        and all(row["terminal"] for row in processed_rows),
        "all_processed_source_identities_match": all(
            row["source_identity_matches"] for row in processed_rows
        ),
        "failure_closed_stage_order": all(row["failure_closed"] for row in processed_rows),
        "unprocessed_suffix_untouched": all(
            not any(item["exists"] for item in row["artifacts"].values())
            for row in suffix_rows
        ),
        "exact_registered_stopping_rule": exact_stop,
        "no_o4_outcomes_exist": not operator_root.exists(),
    }
    passed = all(checks.values()) and quota_reached
    sealed = all(checks.values())
    admitted_rows = [row for row in processed_rows if row["fully_admitted"]]
    result = {
        "schema_version": "kinofail.o4-scene-stream-v24-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": passed,
        "sealed": sealed,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "preflight": str(preflight_path),
        "preflight_sha256": _sha256(preflight_path),
        "checks": checks,
        "maximum_candidates": len(rows),
        "processed_candidates": processed_count,
        "first_quota_prefix": first_quota_prefix,
        "admitted": len(admitted_rows),
        "admitted_scene_ids": [row["scene_id"] for row in admitted_rows],
        "admitted_families": sorted({row["room_type"] for row in admitted_rows}),
        "scenes": rows,
        "cohort_frozen": passed,
        "o4_confirmation_authorized": passed,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "next_gate": (
            "Freeze and execute one v21 Continue/Backstep O4 attempt on every scene in the "
            "admitted cohort; no scene may be selected or removed using O4 outcomes."
        ),
    }
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "passed": passed, "sealed": sealed, "processed": processed_count, "admitted": len(admitted_rows)}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
