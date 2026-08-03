#!/usr/bin/env python3
"""Audit direct under-body high-centering evidence in accepted F25 O9 pairs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "outputs/eval/unified_moe_v3_replenishment_f25/accepted_overlay"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_replenishment_f25/corpus")
OUTPUT = ROOT / "outputs/eval/unified_moe_v3_replenishment_f25/o9_semantic_audit_expanded_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    overlay_audit_path = OVERLAY / "overlay_audit.json"
    overlay_audit = read_json(overlay_audit_path)
    if (
        overlay_audit.get("passed") is not True
        or overlay_audit.get("model_prediction_or_score_read") is not False
    ):
        raise RuntimeError("accepted Scale overlay is invalid")

    records: list[dict[str, Any]] = []
    record_paths = sorted(
        (OVERLAY / "shards").glob("*/scale/accepted_snapshots/snapshot_records.jsonl")
    )
    for path in record_paths:
        records.extend(read_jsonl(path))
    o9_primary = {
        (str(row["counterfactual_group_id"]), str(row["condition"])): row
        for row in records
        if row["target_operator"] == "O9_high_centering"
        and row["appearance_intervention_id"] == "primary"
    }
    group_ids = sorted({group for group, _ in o9_primary})
    if len(group_ids) != 24 or any(
        (group, condition) not in o9_primary
        for group in group_ids
        for condition in ("nominal_counterfactual", "anomaly")
    ):
        raise RuntimeError("expanded accepted O9 pair registry is incomplete")

    manifest_index = {
        path.parent.name: path for path in CORPUS.rglob("manifest.json")
    }
    cases: list[dict[str, Any]] = []
    for group_id in group_ids:
        nominal_record = o9_primary[(group_id, "nominal_counterfactual")]
        anomaly_record = o9_primary[(group_id, "anomaly")]
        pair_evidence: dict[str, Any] = {}
        for condition, record in (
            ("nominal_counterfactual", nominal_record),
            ("anomaly", anomaly_record),
        ):
            replacement_episode = str(record["source_replacement_physical_episode_id"])
            manifest_path = manifest_index.get(replacement_episode)
            if manifest_path is None or sha256(manifest_path) != record["source_manifest_sha256"]:
                raise RuntimeError(f"O9 source manifest missing or hash-mismatched: {replacement_episode}")
            telemetry_path = manifest_path.parent / "privileged.jsonl"
            telemetry = read_jsonl(telemetry_path)
            operator_rows = [row.get("operator", {}) for row in telemetry]
            regions = [
                region
                for operator in operator_rows
                for region in operator.get("regions", [])
                if isinstance(region, dict)
            ]
            pair_evidence[condition] = {
                "replacement_episode_id": replacement_episode,
                "manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "telemetry": str(telemetry_path),
                "telemetry_sha256": sha256(telemetry_path),
                "operator_enabled_observed": any(operator.get("enabled") is True for operator in operator_rows),
                "rounded_ridge_observed": any(region.get("geometry_kind") == "rounded_ridge" for region in regions),
                "ridge_prim_observed": any(str(region.get("prim_path", "")).startswith("/World/ridge_") for region in regions),
                "max_belly_contact_steps": max([int(region.get("belly_contact_steps", 0)) for region in regions] or [0]),
                "max_consecutive_belly_contact_steps": max([int(region.get("max_consecutive_belly_contact_steps", 0)) for region in regions] or [0]),
                "max_belly_contact_force_n": max([float(region.get("max_belly_contact_force_n", 0.0)) for region in regions] or [0.0]),
                "max_base_contact_force_n": max(
                    [
                        float(region.get("max_nonfoot_contact_by_body_n", {}).get("base", 0.0))
                        for region in regions
                    ]
                    or [0.0]
                ),
                "ridge_height_m": sorted({float(region["ridge_height_m"]) for region in regions if "ridge_height_m" in region}),
                "ridge_width_m": sorted({float(region["ridge_width_m"]) for region in regions if "ridge_width_m" in region}),
            }
        nominal = pair_evidence["nominal_counterfactual"]
        anomaly = pair_evidence["anomaly"]
        checks = {
            "nominal_operator_disabled": nominal["operator_enabled_observed"] is False,
            "nominal_no_belly_contact": nominal["max_belly_contact_steps"] == 0,
            "anomaly_operator_enabled": anomaly["operator_enabled_observed"] is True,
            "anomaly_rounded_ridge_geometry": anomaly["rounded_ridge_observed"] is True,
            "anomaly_ridge_prim_present": anomaly["ridge_prim_observed"] is True,
            "anomaly_direct_base_contact": anomaly["max_base_contact_force_n"] > 0.0,
            "anomaly_sustained_belly_contact": anomaly["max_belly_contact_steps"] > 0,
            "anomaly_positive_belly_contact_force": anomaly["max_belly_contact_force_n"] > 0.0,
        }
        cases.append(
            {
                "original_counterfactual_group_id": group_id,
                "scene_id": str(anomaly_record["scene_family"]),
                "domain": str(anomaly_record["domain"]),
                "severity_id": str(anomaly_record["severity_id"]),
                "passed": all(checks.values()),
                "checks": checks,
                "evidence": pair_evidence,
            }
        )

    passed_cases = sum(row["passed"] for row in cases)
    audit = {
        "schema_version": "kinofail.expanded-o9-semantic-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "passed": passed_cases == len(cases),
        "model_prediction_feature_label_or_score_read": False,
        "semantic_definition": (
            "High-centering requires a rounded transverse ridge and direct "
            "non-foot contact on the Go2 base; a ridge merely in front of the robot is insufficient."
        ),
        "counts": {
            "accepted_o9_pairs": len(cases),
            "passed_semantic_pairs": passed_cases,
            "failed_semantic_pairs": len(cases) - passed_cases,
            "scenes": len({row["scene_id"] for row in cases}),
            "domains": len({row["domain"] for row in cases}),
            "severities": dict(sorted(Counter(row["severity_id"] for row in cases).items())),
        },
        "aggregate_evidence": {
            "minimum_positive_base_contact_force_n": min(
                row["evidence"]["anomaly"]["max_base_contact_force_n"] for row in cases
            ),
            "minimum_positive_belly_contact_steps": min(
                row["evidence"]["anomaly"]["max_belly_contact_steps"] for row in cases
            ),
            "all_nominal_controls_operator_disabled_and_contact_free": all(
                row["checks"]["nominal_operator_disabled"]
                and row["checks"]["nominal_no_belly_contact"]
                for row in cases
            ),
        },
        "source_sha256": {
            "overlay_audit": sha256(overlay_audit_path),
            "snapshot_record_shards": {str(path.relative_to(ROOT)): sha256(path) for path in record_paths},
        },
        "cases": cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: audit[key] for key in ("status", "passed", "counts", "aggregate_evidence")}, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
