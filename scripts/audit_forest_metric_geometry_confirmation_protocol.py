#!/usr/bin/env python3
"""Audit a generic frozen forest metric-geometry confirmation protocol."""

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


def _path(record: dict) -> Path:
    return (ROOT / record["path"]).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "postrun"), required=True)
    args = parser.parse_args()
    protocol_path = args.config.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    geometry_specs = [
        protocol[name]
        for name in ("reference_geometry", "development_calibration", "heldout_confirmation")
    ]
    records = list(protocol["frozen_sources"].values()) + [
        spec[key] for spec in geometry_specs for key in ("config", "compiled_audit")
    ]
    resolved = [
        {
            "path": str(_path(record)),
            "expected_sha256": record["sha256"],
            "actual_sha256": _sha256(_path(record)) if _path(record).is_file() else None,
        }
        for record in records
    ]
    compiled = [
        json.loads(_path(spec["compiled_audit"]).read_text(encoding="utf-8"))
        for spec in geometry_specs
    ]
    physical = [
        tuple(row["files"][name] for name in ("collision.usda", "prop_collision.usda", "route.usda"))
        for row in compiled
    ]
    output = (ROOT / protocol["heldout_confirmation"]["output_directory"]).resolve()
    pair_path = output / "pair_audit.json"
    nominal_path = output / "nominal/lane_manifest.json"
    anomaly_path = output / "anomaly/lane_manifest.json"
    pair = json.loads(pair_path.read_text(encoding="utf-8")) if pair_path.is_file() else None
    nominal = json.loads(nominal_path.read_text(encoding="utf-8")) if nominal_path.is_file() else None
    anomaly = json.loads(anomaly_path.read_text(encoding="utf-8")) if anomaly_path.is_file() else None
    preflight = args.phase == "preflight"
    checks = {
        "supported_schema": protocol.get("schema_version")
        == "kinofail.forest-metric-geometry-confirmation.v1-development",
        "all_frozen_files_match": all(
            row["actual_sha256"] == row["expected_sha256"] for row in resolved
        ),
        "all_compiled_scenes_passed": all(row.get("passed") is True for row in compiled),
        "three_scene_ids_distinct": len({row["scene_id"] for row in compiled}) == 3,
        "collision_hashes_distinct": len({row[0] for row in physical}) == 3,
        "prop_collision_hashes_distinct": len({row[1] for row in physical}) == 3,
        "route_hashes_distinct": len({row[2] for row in physical}) == 3,
        "single_attempt_frozen": protocol["heldout_confirmation"].get(
            "single_runtime_attempt"
        )
        is True,
        "output_state_matches_phase": (
            not output.exists()
            if preflight
            else pair is not None and nominal is not None and anomaly is not None
        ),
        "postrun_pair_matches_frozen_schema_and_passes": (
            True
            if preflight
            else pair is not None
            and pair.get("schema_version") == protocol["expected_pair_schema"]
            and pair.get("evidence_role") == "heldout_geometry_confirmation"
            and pair.get("passed") is True
            and pair.get("counts_as_a0_a7_evidence") is False
        ),
        "postrun_exactly_one_nominal_anomaly_directory": (
            True
            if preflight
            else nominal is not None
            and anomaly is not None
            and nominal.get("passed") is True
            and anomaly.get("passed") is True
            and sorted(path.name for path in output.iterdir() if path.is_dir())
            == ["anomaly", "nominal"]
        ),
        "not_a0_a7_evidence": protocol["evidence_boundary"].get(
            "counts_as_a0_a7_evidence"
        )
        is False,
    }
    result = {
        "schema_version": "kinofail.forest-metric-geometry-confirmation-audit.v1-development",
        "created_utc": datetime.now(UTC).isoformat(),
        "phase": args.phase,
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "operator_id": protocol["operator_id"],
        "checks": checks,
        "resolved_files": resolved,
        "physical_hashes": [
            {
                "realization_id": spec["realization_id"],
                "scene_id": audit["scene_id"],
                "collision_usda": hashes[0],
                "prop_collision_usda": hashes[1],
                "route_usda": hashes[2],
            }
            for spec, audit, hashes in zip(geometry_specs, compiled, physical, strict=True)
        ],
        "heldout_pair_audit": str(pair_path) if pair is not None else None,
        "passed": all(checks.values()),
        "scene_registry_eligible": False,
        "counts_as_a0_a7_evidence": False,
    }
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
