#!/usr/bin/env python3
"""Preflight a frozen O1 independent-metric-geometry confirmation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(record: dict) -> Path:
    return (ROOT / record["path"]).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-heldout-unused", action="store_true")
    parser.add_argument("--postrun", action="store_true")
    args = parser.parse_args()
    protocol_path = args.config.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    frozen_records = list(protocol["frozen_sources"].values())
    geometry_records = [
        protocol[key]
        for key in ("reference_geometry", "development_calibration", "heldout_confirmation")
    ]
    file_records = frozen_records + [
        item[subkey]
        for item in geometry_records
        for subkey in ("config", "compiled_audit")
    ]
    resolved_files = [
        {
            "path": str(_resolve(record)),
            "expected_sha256": record["sha256"],
            "actual_sha256": _sha256(_resolve(record)) if _resolve(record).is_file() else None,
        }
        for record in file_records
    ]
    compiled = [
        json.loads(_resolve(item["compiled_audit"]).read_text(encoding="utf-8"))
        for item in geometry_records
    ]
    physical_hash_sets = [
        tuple(row["files"][name] for name in ("collision.usda", "prop_collision.usda", "route.usda"))
        for row in compiled
    ]
    heldout_output = (ROOT / protocol["heldout_confirmation"]["output_directory"]).resolve()
    pair_path = heldout_output / "pair_audit.json"
    pair = json.loads(pair_path.read_text(encoding="utf-8")) if pair_path.is_file() else None
    nominal_path = heldout_output / "nominal/lane_manifest.json"
    anomaly_path = heldout_output / "anomaly/lane_manifest.json"
    nominal = json.loads(nominal_path.read_text(encoding="utf-8")) if nominal_path.is_file() else None
    anomaly = json.loads(anomaly_path.read_text(encoding="utf-8")) if anomaly_path.is_file() else None
    checks = {
        "supported_schema": protocol.get("schema_version")
        == "kinofail.forest-metric-geometry-o1-confirmation.v1-development",
        "all_frozen_files_exist_and_match_hash": all(
            row["actual_sha256"] == row["expected_sha256"] for row in resolved_files
        ),
        "three_scene_ids_are_distinct": len({row["scene_id"] for row in compiled}) == 3,
        "all_compiled_scenes_passed": all(row.get("passed") is True for row in compiled),
        "all_three_physical_layer_hash_tuples_are_distinct": len(set(physical_hash_sets)) == 3,
        "each_collision_hash_is_distinct": len({row[0] for row in physical_hash_sets}) == 3,
        "each_prop_collision_hash_is_distinct": len({row[1] for row in physical_hash_sets}) == 3,
        "each_route_hash_is_distinct": len({row[2] for row in physical_hash_sets}) == 3,
        "heldout_is_explicitly_single_attempt": protocol["heldout_confirmation"].get(
            "single_runtime_attempt"
        )
        is True,
        "heldout_output_state_matches_phase": (
            (not heldout_output.exists())
            if args.require_heldout_unused
            else (
                pair is not None and nominal is not None and anomaly is not None
                if args.postrun
                else True
            )
        ),
        "postrun_pair_passes_frozen_v2_contract": (
            True
            if not args.postrun
            else pair is not None
            and pair.get("schema_version")
            == "kinofail.realistic-forest-o1-pair-audit.v2-development"
            and pair.get("evidence_role") == "heldout_geometry_confirmation"
            and pair.get("passed") is True
            and pair.get("counts_as_a0_a7_evidence") is False
        ),
        "postrun_lanes_are_exactly_one_passed_nominal_anomaly_pair": (
            True
            if not args.postrun
            else nominal is not None
            and anomaly is not None
            and nominal.get("lane") == "nominal"
            and anomaly.get("lane") == "anomaly"
            and nominal.get("passed") is True
            and anomaly.get("passed") is True
            and sorted(path.name for path in heldout_output.iterdir() if path.is_dir())
            == ["anomaly", "nominal"]
        ),
        "still_not_a0_a7_evidence": protocol["evidence_boundary"].get(
            "counts_as_a0_a7_evidence"
        )
        is False
        and all(row.get("counts_as_a0_a7_evidence") is False for row in compiled),
    }
    result = {
        "schema_version": "kinofail.forest-metric-geometry-o1-protocol-audit.v1-development",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "checks": checks,
        "resolved_files": resolved_files,
        "geometry_realizations": [
            {
                "realization_id": spec["realization_id"],
                "scene_id": audit["scene_id"],
                "route": audit["route"],
                "minimum_route_clearance_m": audit["near_field_contract"][
                    "prop_collision"
                ]["minimum_route_clearance_m"],
                "physical_file_hashes": {
                    name: audit["files"][name]
                    for name in ("collision.usda", "prop_collision.usda", "route.usda")
                },
            }
            for spec, audit in zip(geometry_records, compiled, strict=True)
        ],
        "heldout_output": str(heldout_output),
        "heldout_pair_audit": str(pair_path) if pair is not None else None,
        "passed": all(checks.values()),
        "scene_registry_eligible": False,
        "counts_as_a0_a7_evidence": False,
    }
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
