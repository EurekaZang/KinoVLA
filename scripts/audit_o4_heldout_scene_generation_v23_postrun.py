#!/usr/bin/env python3
"""Adjudicate the stratified v23 O4 scene-acquisition batch before O4."""

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
    source_root = _resolve(pipeline["source_root"])
    operator_root = _resolve(config["operator_output_root"])
    rows = []
    for scene in config["requested_scenes"]:
        source = source_root / scene["output_directory"]
        route = source / pipeline["route_output_name"] / scene["material_id"]
        terrain = route / pipeline["terrain_output_name"]
        artifacts = {
            "generation_exception": _artifact(
                source_root
                / f"{scene['output_directory']}_generation_exception_audit.json"
            ),
            "source_integrity": _artifact(source / "source_integrity_audit.json"),
            "source_preflight": _artifact(
                source / "source_geometry_preflight_v1/source_geometry_preflight_audit.json"
            ),
            "base_compile": _artifact(
                source / pipeline["base_output_name"] / "compiled_scene_audit.json"
            ),
            "corridor": _artifact(
                source / pipeline["corridor_output_name"] / "compiled_scene_audit.json"
            ),
            "route": _artifact(route / "compiled_scene_audit.json"),
            "rtx": _artifact(route / "rtx_qa/rtx_scene_audit.json"),
            "terrain_compile": _artifact(terrain / "compiled_scene_audit.json"),
            "terrain_offline": _artifact(terrain / "offline_usd_audit.json"),
            "go2": _artifact(terrain / "go2_qa/go2_scene_audit.json"),
            "go2_exception": _artifact(
                terrain / "go2_qa/go2_scene_exception_audit.json"
            ),
            "stack_admission": _artifact(
                terrain / "realistic_stack_v16_admission.json"
            ),
        }
        manifest = _json(source / "source_manifest.json")
        identity_ok = (
            manifest.get("scene_id") == scene["scene_id"]
            and manifest.get("generation", {}).get("room_type") == scene["room_type"]
            and manifest.get("generation", {}).get("seed") == scene["source_seed"]
            and manifest.get("generation", {}).get("complexity") == scene["complexity"]
        )
        generation_failed = artifacts["generation_exception"]["passed"] is False
        admitted = artifacts["stack_admission"]["passed"] is True
        failed = any(item["passed"] is False for item in artifacts.values())
        rows.append(
            {
                "scene_id": scene["scene_id"],
                "room_type": scene["room_type"],
                "source_seed": scene["source_seed"],
                "runtime_seed": scene["runtime_seed"],
                "material_id": scene["material_id"],
                "source_identity_matches": identity_ok or generation_failed,
                "artifacts": artifacts,
                "fully_admitted": admitted,
                "terminal_failure_observed": failed,
                "pipeline_terminal": admitted or failed,
            }
        )
    admitted = [row for row in rows if row["fully_admitted"]]
    policy = config["selection_policy"]
    checks = {
        "preflight_passed": preflight.get("passed") is True,
        "preflight_binds_config": preflight.get("config_sha256") == _sha256(config_path),
        "locked_files_unchanged": all(_locked(item) for item in config["locked_files"]),
        "all_requests_accounted": len(rows) == len(config["requested_scenes"]),
        "all_source_identities_match": all(row["source_identity_matches"] for row in rows),
        "all_pipelines_terminal": all(row["pipeline_terminal"] for row in rows),
        "minimum_admitted_scenes": len(admitted) >= policy["minimum_admitted_scenes"],
        "minimum_admitted_families": len({row["room_type"] for row in admitted})
        >= policy["minimum_admitted_families"],
        "no_o4_outcomes_exist": not operator_root.exists(),
    }
    passed = all(checks.values())
    sealed = checks["all_requests_accounted"] and checks["all_pipelines_terminal"]
    audit = {
        "schema_version": "kinofail.o4-heldout-scene-generation-v23-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": passed,
        "sealed": sealed,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "preflight": str(preflight_path),
        "preflight_sha256": _sha256(preflight_path),
        "checks": checks,
        "requested": len(rows),
        "admitted": len(admitted),
        "admitted_scene_ids": [row["scene_id"] for row in admitted],
        "admitted_families": sorted({row["room_type"] for row in admitted}),
        "scenes": rows,
        "o4_confirmation_authorized": passed,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "next_gate": (
            "Freeze one v21 Continue/Backstep attempt on every admitted scene before opening "
            "any O4 outcome."
        ),
    }
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite postrun audit: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "passed": passed,
                "sealed": sealed,
                "admitted_scene_ids": audit["admitted_scene_ids"],
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
