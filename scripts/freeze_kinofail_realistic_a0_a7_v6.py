#!/usr/bin/env python3
"""Freeze the A0--A7 contract for scale-v8 paired physical nuisance collection."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONTRACT = ROOT / "configs/eval/kinofail_realistic_a0_a7_v5.json"
CONTRACT = ROOT / "configs/eval/kinofail_realistic_a0_a7_v6.json"
PROTOCOL = ROOT / "configs/data/kinofail_realistic_scale_v8_replication_formal_v3.json"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v8.py"
NUISANCE_AUDIT = (
    ROOT
    / "outputs/kinofail_realistic/design_scale_v8_replication_v1/"
    "scale_v8_physical_nuisance_audit.json"
)
SNAPSHOT_BASE = ROOT / "configs/eval/kinofail_realistic_snapshot_scale_v8_v1.json"
SNAPSHOT = ROOT / "configs/eval/kinofail_realistic_snapshot_scale_v8_v2.json"
MULTIMODAL_BASE = (
    ROOT / "configs/eval/kinofail_realistic_multimodal_scale_v8_v1.json"
)
MULTIMODAL = ROOT / "configs/eval/kinofail_realistic_multimodal_scale_v8_v2.json"
CORPUS = ROOT / "outputs/kinofail_realistic/corpus_scale_v8_replication_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    existing_pairs = list((CORPUS / "pair_summaries").glob("*.json"))
    if existing_pairs:
        raise RuntimeError(
            f"contract must precede scale-v8 collection; found {len(existing_pairs)} pairs"
        )
    base = _json(BASE_CONTRACT)
    protocol = _json(PROTOCOL)
    nuisance = _json(NUISANCE_AUDIT)
    if protocol.get("status") != "frozen" or nuisance.get("passed") is not True:
        raise RuntimeError("physical nuisance protocol is not frozen and audited")

    contract = json.loads(json.dumps(base))
    contract.update(
        {
            "schema_version": "kinofail.realistic-a0-a7-replication-contract.v6",
            "contract_id": (
                "kinofail-realistic-a0-a7-scale-v8-five-physical-profile-20260723"
            ),
            "status": "frozen_before_scale_v8_physical_collection",
            "scope": (
                "A0-A7 on the hash-frozen scale-v8 corpus with five balanced, pair-shared "
                "physical nuisance profiles per scene/operator/severity cell; A8 excluded."
            ),
        }
    )
    contract["claim_policy"]["corpus_protocol_id"] = protocol["protocol_id"]
    contract["evidence_roots"] = {
        "inference_manifest": "outputs/eval/realistic_a0_a7_v6/inference_manifest.json",
        "predictions": "outputs/eval/realistic_a0_a7_v6/predictions.jsonl",
        "runtime_audit": (
            "outputs/kinofail_realistic/runtime_audit/formal_scale_v8_nuisance/"
            "runtime_audit.json"
        ),
        "runtime_coverage": (
            "outputs/kinofail_realistic/runtime_audit/formal_scale_v8_nuisance/"
            "coverage.json"
        ),
        "scene_registry": (
            "outputs/kinofail_realistic/runtime_audit/formal_scale_v8_nuisance/"
            "scene_registry_audit.json"
        ),
        "snapshot_development_audit": (
            "outputs/eval/realistic_a0_a7_v6/snapshots/extraction_audit.json"
        ),
        "training_manifest": "outputs/eval/realistic_a0_a7_v6/training_manifest.json",
    }
    for experiment_id, experiment in contract["experiments"].items():
        filename = Path(str(experiment["output"])).name
        experiment["output"] = f"outputs/eval/realistic_a0_a7_v6/{filename}"
    contract["freeze_provenance"] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "base_contract": str(BASE_CONTRACT.relative_to(ROOT)),
        "base_contract_sha256": _sha(BASE_CONTRACT),
        "collection_protocol": str(PROTOCOL.relative_to(ROOT)),
        "collection_protocol_sha256": _sha(PROTOCOL),
        "collector": str(COLLECTOR.relative_to(ROOT)),
        "collector_sha256": _sha(COLLECTOR),
        "physical_nuisance_audit": str(NUISANCE_AUDIT.relative_to(ROOT)),
        "physical_nuisance_audit_sha256": _sha(NUISANCE_AUDIT),
        "schedule": protocol["schedule_path"],
        "schedule_sha256": protocol["schedule_sha256"],
        "model_outcomes_available_at_freeze": False,
        "physical_collection_started_at_freeze": False,
        "amendment_reason": (
            "Distinct seed integers were not sufficient because the inference reset freezes "
            "initial joints; v6 explicitly varies paired route-entry state and forward speed."
        ),
    }
    _write_new(CONTRACT, contract)

    snapshot = _json(SNAPSHOT_BASE)
    snapshot.update(
        {
            "protocol_id": "realistic_snapshot_scale_v8_physical_nuisance_v2",
            "status": "frozen_before_scale_v8_physical_collection",
            "output_dir": "outputs/eval/realistic_a0_a7_v6/snapshots",
        }
    )
    snapshot["selection"]["required_collection_protocol_id"] = protocol["protocol_id"]
    snapshot["publication_guard"]["reason"] = (
        "Five balanced, pair-shared route-entry nuisance profiles per "
        "scene/operator/severity cell on the frozen scale-v8 corpus."
    )
    snapshot["freeze_provenance"] = {
        "contract": str(CONTRACT.relative_to(ROOT)),
        "contract_sha256": _sha(CONTRACT),
        "collection_protocol": str(PROTOCOL.relative_to(ROOT)),
        "collection_protocol_sha256": _sha(PROTOCOL),
        "physical_nuisance_audit": str(NUISANCE_AUDIT.relative_to(ROOT)),
        "physical_nuisance_audit_sha256": _sha(NUISANCE_AUDIT),
        "model_outcomes_available_at_freeze": False,
        "physical_collection_started_at_freeze": False,
    }
    _write_new(SNAPSHOT, snapshot)

    multimodal = _json(MULTIMODAL_BASE)
    multimodal.update(
        {
            "protocol_id": (
                "kinofail_realistic_multimodal_scale_v8_physical_nuisance_v2"
            ),
            "status": "frozen_before_scale_v8_physical_collection",
            "freeze_context": (
                "Scale-v7 architecture, thresholds, splits and statistics are unchanged. "
                "This version binds the five balanced physical nuisance profiles before data."
            ),
            "source_snapshot_protocol": str(SNAPSHOT.relative_to(ROOT)),
            "source_feature_manifest": (
                "outputs/eval/realistic_a0_a7_v6/features/feature_manifest.json"
            ),
            "output_root": "outputs/eval/realistic_a0_a7_v6",
        }
    )
    _write_new(MULTIMODAL, multimodal)
    print(
        json.dumps(
            {
                "contract": str(CONTRACT),
                "contract_sha256": _sha(CONTRACT),
                "snapshot": str(SNAPSHOT),
                "snapshot_sha256": _sha(SNAPSHOT),
                "multimodal": str(MULTIMODAL),
                "multimodal_sha256": _sha(MULTIMODAL),
                "passed": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
