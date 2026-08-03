#!/usr/bin/env python3
"""Freeze the realistic A0--A7 contract against the final scale-v5 corpus paths."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/eval/kinofail_realistic_a0_a7_v1.json"
OUT = ROOT / "configs/eval/kinofail_realistic_a0_a7_v4.json"


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    contract = json.loads(SOURCE.read_text(encoding="utf-8"))
    contract.update({
        "schema_version": "kinofail.realistic-a0-a7-replication-contract.v4",
        "contract_id": "kinofail-realistic-a0-a7-scale-v7-20260723",
        "status": "frozen_before_scale_v7_outcomes",
        "scope": "A0-A7 on the hash-frozen realistic scale-v7 corpus; A8 excluded",
    })
    base = "outputs/eval/realistic_a0_a7_v4"
    audit = "outputs/kinofail_realistic/runtime_audit/formal_scale_v7"
    contract["evidence_roots"] = {
        "scene_registry": f"{audit}/scene_registry_audit.json",
        "runtime_audit": f"{audit}/runtime_audit.json",
        "runtime_coverage": f"{audit}/coverage.json",
        "snapshot_development_audit": f"{base}/snapshots/extraction_audit.json",
        "training_manifest": f"{base}/training_manifest.json",
        "inference_manifest": f"{base}/inference_manifest.json",
        "predictions": f"{base}/predictions.jsonl",
    }
    names = {
        "A0": "a0_certificate.json", "A1": "a1_construct_validity.json",
        "A2": "a2_headline.json", "A3": "a3_publication_report.json",
        "A4": "a4_consequence.json", "A5": "a5_selective.json",
        "A6": "a6_boundary.json", "A7": "a7_ablation.json",
    }
    for experiment_id, name in names.items():
        contract["experiments"][experiment_id]["output"] = f"{base}/{name}"
    contract["claim_policy"]["a8_in_scope"] = False
    contract["claim_policy"]["corpus_protocol_id"] = "kinofail_realistic_scale_396_v7"
    OUT.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "contract_id": contract["contract_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
