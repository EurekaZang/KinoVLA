#!/usr/bin/env python3
"""Freeze the route-axis O9 correction for only scale-v7 blocking pairs."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def main() -> None:
    schedule = ROOT / "outputs/kinofail_realistic/design_scale_v2/pilot_schedule.jsonl"
    registry = ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json"
    collector = ROOT / "scripts/isaac_collect_kinofail_realistic_o9_repair_v2.py"
    base_corpus = ROOT / "outputs/kinofail_realistic/corpus_scale_v7"
    output = ROOT / "configs/data/kinofail_realistic_o9_repair_formal_v2.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rows = [json.loads(line) for line in schedule.read_text(encoding="utf-8").splitlines() if line]
    blocking_pairs: set[str] = set()
    evidence: list[dict] = []
    for row in rows:
        manifest_path = base_corpus / row["required_outputs"]["episode_manifest"]
        manifest = _json(manifest_path)
        if not manifest:
            raise RuntimeError("base scale collection is not complete")
        validation = manifest.get("runtime_validation", {})
        if validation.get("passed") is True:
            continue
        issues = [str(value) for value in validation.get("issues", [])]
        remaining = [
            value for value in issues
            if not value.endswith("appearance_effect_too_small")
            and value != "rgb_spatial_contrast_too_low"
        ]
        if not remaining:
            continue
        if (
            row["condition"] == "anomaly"
            and row["target_operator"] == "O4_tether"
            and remaining == ["operator_local_qa_failed"]
        ):
            telemetry = manifest.get("operator_readback", {}).get("telemetry", {})
            if (
                int(telemetry.get("total_attachment_cycles", 0)) > 0
                and float(telemetry.get("total_applied_force_n", 0.0)) > 0.0
                and float(telemetry.get("total_tangential_work_j", 0.0)) > 0.0
            ):
                continue
        if (
            row["condition"] != "anomaly"
            or row["target_operator"] != "O9_high_centering"
            or remaining != ["operator_local_qa_failed"]
        ):
            raise RuntimeError(
                f"repair v2 only admits the known O9 exposure failure: "
                f"{row['episode_id']} {remaining}"
            )
        operator_region = manifest.get("geometry_readback", {}).get("operator_region", {})
        if not float(operator_region.get("hx", 0.0)) > float(operator_region.get("hy", 0.0)):
            raise RuntimeError(
                f"blocking O9 pair is not a world-Y route-axis case: {row['episode_id']}"
            )
        telemetry = manifest.get("operator_readback", {}).get("telemetry", {})
        if int(telemetry.get("scale_region_exposure_steps", -1)) != 0:
            raise RuntimeError(f"unexpected O9 blocker evidence: {row['episode_id']}")
        blocking_pairs.add(str(row["counterfactual_group_id"]))
        evidence.append(
            {
                "episode_id": row["episode_id"],
                "issues": issues,
                "operator_region": operator_region,
                "scale_region_exposure_steps": 0,
                "source_manifest_sha256": _sha(manifest_path),
            }
        )
    if not blocking_pairs:
        raise RuntimeError("no O9 blocking pairs require repair")
    repair_rows = [row for row in rows if row["counterfactual_group_id"] in blocking_pairs]
    allowed = {
        "counterfactual_group_ids": sorted(blocking_pairs),
        "target_operators": ["O9_high_centering"],
        "conditions": sorted({str(row["condition"]) for row in repair_rows}),
        "scene_families": sorted({str(row["scene_family"]) for row in repair_rows}),
        "severity_ids": sorted({str(row["severity_id"]) for row in repair_rows}),
        "physical_realizations": sorted(
            {str(row["physical_realization"]) for row in repair_rows}
        ),
        "geometry_profiles": sorted({str(row["geometry_profile"]) for row in repair_rows}),
    }
    protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": "kinofail_realistic_scale_v7_o9_route_axis_repair_v2",
        "status": "frozen",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": "kinofail_realistic_scale_v2",
        "scope": (
            "Targeted repair of the six scale-v7 O9 pairs on world-Y routes whose fixed-axis "
            "cylinder blocked mechanism exposure; no other pair is admitted."
        ),
        "base_protocol": "configs/data/kinofail_realistic_scale_formal_pilot_v7.json",
        "schedule_path": str(schedule.relative_to(ROOT)),
        "schedule_sha256": _sha(schedule),
        "scene_registry_path": str(registry.relative_to(ROOT)),
        "scene_registry_sha256": _sha(registry),
        "collector_path": str(collector.relative_to(ROOT)),
        "collector_sha256": _sha(collector),
        "runtime_manifest_path": "kino_vla/data/runtime_manifest.py",
        "runtime_manifest_sha256": _sha(ROOT / "kino_vla/data/runtime_manifest.py"),
        "allowed": allowed,
        "amendment": {
            "changed": (
                "For world-Y routes only, rotate the rounded-ridge cylinder axis Y -> X so it "
                "remains transverse to locomotion."
            ),
            "unchanged": [
                "scheduled ridge height, width, and residual-support target",
                "early region center and extent",
                "physics material",
                "controller",
                "camera and appearance interventions",
                "10 Hz capture",
                "counterfactual pairing",
            ],
            "selection_basis": evidence,
            "selection_is_runtime_geometry_qa_not_outcome_strength": True,
            "model_predictions_available_at_freeze": False,
        },
        "collection_contract": {
            "physical_episodes": len(repair_rows),
            "counterfactual_pairs": len(blocking_pairs),
            "appearance_views_per_episode": 3,
            "separate_repair_root_required": True,
            "a8_in_scope": False,
        },
    }
    output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha(output),
                "repair_pairs": sorted(blocking_pairs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
