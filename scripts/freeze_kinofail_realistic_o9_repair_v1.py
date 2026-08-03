#!/usr/bin/env python3
"""Freeze the predeclared O9 down-route repair for only scale-v7 blocking pairs."""

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
    collector = ROOT / "scripts/isaac_collect_kinofail_realistic_o9_repair_v1.py"
    base_corpus = ROOT / "outputs/kinofail_realistic/corpus_scale_v7"
    output = ROOT / "configs/data/kinofail_realistic_o9_repair_formal_v1.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rows = [json.loads(line) for line in schedule.read_text(encoding="utf-8").splitlines() if line]
    blocking_pairs = set()
    evidence = []
    for row in rows:
        manifest = _json(base_corpus / row["required_outputs"]["episode_manifest"])
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
        if row["condition"] == "anomaly" and row["target_operator"] == "O4_tether" and remaining == ["operator_local_qa_failed"]:
            telemetry = manifest.get("operator_readback", {}).get("telemetry", {})
            if int(telemetry.get("total_attachment_cycles", 0)) > 0 and float(telemetry.get("total_applied_force_n", 0.0)) > 0.0 and float(telemetry.get("total_tangential_work_j", 0.0)) > 0.0:
                continue
        if row["condition"] != "anomaly" or row["target_operator"] != "O9_high_centering" or remaining != ["operator_local_qa_failed"]:
            raise RuntimeError(f"repair v1 only admits the known O9 exposure failure: {row['episode_id']} {remaining}")
        blocking_pairs.add(row["counterfactual_group_id"])
        evidence.append({"episode_id": row["episode_id"], "issues": issues, "source_manifest_sha256": _sha(base_corpus / row["required_outputs"]["episode_manifest"])})
    if not blocking_pairs:
        raise RuntimeError("no O9 blocking pairs require repair")
    repair_rows = [row for row in rows if row["counterfactual_group_id"] in blocking_pairs]
    protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": "kinofail_realistic_scale_v7_o9_repair_v1",
        "status": "frozen",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": "kinofail_realistic_scale_v2",
        "scope": "Targeted repair of scale-v7 O9 pairs whose front feet stopped at the early ridge before base-region/belly-contact exposure; no other pair is admitted.",
        "base_protocol": "configs/data/kinofail_realistic_scale_formal_pilot_v7.json",
        "schedule_path": str(schedule.relative_to(ROOT)),
        "schedule_sha256": _sha(schedule),
        "scene_registry_path": str(registry.relative_to(ROOT)),
        "scene_registry_sha256": _sha(registry),
        "collector_path": str(collector.relative_to(ROOT)),
        "collector_sha256": _sha(collector),
        "runtime_manifest_path": "kino_vla/data/runtime_manifest.py",
        "runtime_manifest_sha256": _sha(ROOT / "kino_vla/data/runtime_manifest.py"),
        "allowed": {
            "counterfactual_group_ids": sorted(blocking_pairs),
            "target_operators": ["O9_high_centering"],
            "conditions": ["anomaly", "nominal_counterfactual"],
            "scene_families": sorted({row["scene_family"] for row in repair_rows}),
            "severity_ids": sorted({row["severity_id"] for row in repair_rows}),
        },
        "amendment": {
            "changed": "O9 ridge route center 0.35 m -> 0.50 m",
            "unchanged": ["scheduled severity parameters", "geometry kind", "controller", "camera", "appearance", "10 Hz capture", "counterfactual pairing"],
            "selection_basis": evidence,
            "selection_is_runtime_qa_not_outcome_strength": True,
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
    print(json.dumps({"output": str(output), "sha256": _sha(output), "repair_pairs": sorted(blocking_pairs)}, indent=2))


if __name__ == "__main__":
    main()
