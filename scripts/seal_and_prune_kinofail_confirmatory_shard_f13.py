#!/usr/bin/env python3
"""F13 launcher-ledger preflight around the sealed F4/F9 pruner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from kinofail_reconfirmation_attrition_f13 import (  # noqa: E402
    F13,
    load_ledger,
    read_json,
    sha256,
)

F4 = ROOT / "scripts/seal_and_prune_kinofail_confirmatory_shard_f4.py"
VALIDATION_ROOT = (
    ROOT / "outputs/kinofail_reconfirmation_v2/f13_validations"
)


def _record_pair_ids(path: Path) -> set[str]:
    values = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            values.add(str(row["counterfactual_group_id"]))
    return values


def validate_scene(
    *,
    scene_id: str,
    derived_root: Path,
    receipt_path: Path,
    capsule_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    receipt = read_json(receipt_path)
    capsule = read_json(capsule_path)
    battery_rows = {}
    source_hashes: dict[str, str] = {
        "receipt": sha256(receipt_path),
        "capsule": sha256(capsule_path),
        "f13": sha256(F13),
    }
    for battery, receipt_key in (("scale", "scale"), ("c2_t3", "t3")):
        ledger, ledger_path = load_ledger(
            scene_id=scene_id,
            battery=battery,
        )
        snapshot_dir = derived_root / scene_id / battery / "snapshots"
        records_path = snapshot_dir / "snapshot_records.jsonl"
        audit_path = snapshot_dir / "extraction_audit.json"
        audit = read_json(audit_path)
        selected = _record_pair_ids(records_path)
        temporal = {
            str(row["counterfactual_group_id"])
            for row in audit.get("temporal_alignment_exclusions", [])
        }
        eligible = set(ledger["eligible_pair_ids"])
        attrited = {
            str(row["counterfactual_group_id"])
            for row in ledger["attrition"]
        }
        if selected | temporal != eligible or selected & temporal:
            raise RuntimeError(
                f"F13 {scene_id}/{battery} selected set differs from ledger"
            )
        if int(audit.get("skipped_incomplete_pairs", -1)) != len(attrited):
            raise RuntimeError(
                f"F13 {scene_id}/{battery} snapshot attrition mismatch"
            )
        if (
            audit.get("launcher_eligibility_ledger_sha256")
            != sha256(ledger_path)
            or audit.get(
                "model_or_prediction_loaded_for_launcher_filter"
            )
            is not False
            or audit.get("selection_uses_outcome_strength") is not False
        ):
            raise RuntimeError(
                f"F13 {scene_id}/{battery} snapshot provenance mismatch"
            )
        receipt_row = receipt[receipt_key]
        if int(receipt_row["skipped_incomplete_pairs"]) != len(attrited):
            raise RuntimeError(
                f"F13 {scene_id}/{battery} receipt attrition mismatch"
            )
        battery_rows[battery] = {
            "planned_pairs": int(ledger["counts"]["planned_pairs"]),
            "eligible_pairs": len(eligible),
            "selected_pairs": len(selected),
            "temporally_excluded_pairs": len(temporal),
            "attrited_pairs": len(attrited),
            "ledger": str(ledger_path),
            "ledger_sha256": sha256(ledger_path),
            "snapshot_audit_sha256": sha256(audit_path),
            "snapshot_records_sha256": sha256(records_path),
        }
        source_hashes[f"{battery}_ledger"] = sha256(ledger_path)
        source_hashes[f"{battery}_snapshot_audit"] = sha256(audit_path)
        source_hashes[f"{battery}_snapshot_records"] = sha256(records_path)
    t3_ledger, t3_ledger_path = load_ledger(
        scene_id=scene_id,
        battery="c2_t3",
    )
    if (
        int(capsule["t3"]["retained_anomaly_episodes"])
        != int(t3_ledger["counts"]["eligible_pairs"])
        or int(capsule["t3"]["excluded_anomaly_episodes"])
        != int(t3_ledger["counts"]["attrited_pairs"])
        or capsule["t3"].get("launcher_eligibility_ledger_sha256")
        != sha256(t3_ledger_path)
        or capsule.get("model_or_prediction_loaded") is not False
        or capsule.get("selection_uses_outcome_strength") is not False
    ):
        raise RuntimeError(f"F13 capsule differs from launcher ledger: {scene_id}")
    validation = {
        "schema_version": (
            "kinofail.reconfirmation-f13-scene-validation.v1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "passed": True,
        "scene_id": scene_id,
        "model_feature_label_outcome_or_score_used_for_filter": False,
        "result_dependent_retry_or_selection": False,
        "operational_amendment": str(F13),
        "operational_amendment_sha256": sha256(F13),
        "batteries": battery_rows,
        "capsule": {
            "retained_anomaly_episodes": capsule["t3"][
                "retained_anomaly_episodes"
            ],
            "excluded_anomaly_episodes": capsule["t3"][
                "excluded_anomaly_episodes"
            ],
            "manifest": str(capsule_path),
            "manifest_sha256": sha256(capsule_path),
        },
        "source_sha256": source_hashes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    known, _ = parser.parse_known_args()
    if not F13.is_file():
        raise FileNotFoundError(F13)
    completed = subprocess.run(
        [sys.executable, str(F4), *sys.argv[1:]],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        return int(completed.returncode)
    scene_id = str(known.scene_id)
    receipt_path = (
        known.receipt_root.resolve() / scene_id / "completed.json"
    )
    capsule_path = (
        ROOT
        / "outputs/kinofail_reconfirmation_v2/conflict_capsules_ext4"
        / scene_id
        / "capsule_manifest.json"
    )
    validation = validate_scene(
        scene_id=scene_id,
        derived_root=known.derived_root.resolve(),
        receipt_path=receipt_path,
        capsule_path=capsule_path,
        output_path=VALIDATION_ROOT / scene_id / "validation.json",
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
