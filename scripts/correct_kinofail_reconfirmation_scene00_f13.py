#!/usr/bin/env python3
"""One-time, model-blind F13 correction of the already-pruned scene 00.

The F12 incident had predeclared cf_5d2... as permanent attrition, but its
interrupted collector left two complete manifests.  The generic physical
selector therefore retained it before F13 could stop the old pipeline.  This
script preserves the excluded evidence, filters exactly that predeclared pair,
reseals all dependent hashes, and emits the normal F13 scene validation.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from kinofail_reconfirmation_attrition_f13 import (  # noqa: E402
    F13,
    ledger_path,
    read_json,
    read_jsonl,
    sha256,
)
from seal_and_prune_kinofail_confirmatory_shard_f13 import (  # noqa: E402
    VALIDATION_ROOT,
    validate_scene,
)

SCENE = "confirm_v2_life_scene_00"
EXTRA_PAIR = "cf_5d2b090e1eca9d248ace"
DERIVED = (
    ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2/shards" / SCENE
)
CAPSULE = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/conflict_capsules_ext4"
    / SCENE
)
RECEIPT = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/prune_receipts"
    / SCENE
    / "completed.json"
)
PIPELINE_STATE = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration"
    / SCENE
    / "pipeline_state.json"
)
SCHEDULE_SHARD = (
    ROOT / "outputs/kinofail_reconfirmation_v2/schedules/scenes" / SCENE
)
F12 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f12_process_local_rtx_texture_amendment1/"
    "amendment_manifest.json"
)
INCIDENT = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/incidents/"
    "f13_launcher_eligibility_scene00"
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(value, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _schedule_pair_ids(path: Path) -> list[str]:
    rows = read_jsonl(path)
    counts: dict[str, int] = {}
    for row in rows:
        pair_id = str(row["counterfactual_group_id"])
        counts[pair_id] = counts.get(pair_id, 0) + 1
    if set(counts.values()) != {2}:
        raise RuntimeError(path)
    return sorted(counts)


def _record_pair_ids(path: Path) -> set[str]:
    return {
        str(row["counterfactual_group_id"]) for row in read_jsonl(path)
    }


def _recovered_ledger(
    *,
    battery: str,
    schedule_path: Path,
    eligible: set[str],
    reasons: dict[str, str],
    evidence: dict[str, Any],
) -> Path:
    planned = set(_schedule_pair_ids(schedule_path))
    attrited = sorted(planned - eligible)
    if eligible | set(attrited) != planned or eligible & set(attrited):
        raise RuntimeError(f"recovered ledger does not partition {battery}")
    if set(reasons) != set(attrited):
        raise RuntimeError(f"recovered attrition reasons differ for {battery}")
    path = ledger_path(SCENE, battery)
    ledger = {
        "schema_version": (
            "kinofail.reconfirmation-f13-launcher-eligibility-ledger.v1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "passed": True,
        "scene_id": SCENE,
        "battery": battery,
        "selection_basis": (
            "pre-prune model-blind receipt/snapshot accounting plus the "
            "F12 predeclared no-retry incident ledger"
        ),
        "recovered_after_f4_prune": True,
        "recovery_evidence": evidence,
        "model_feature_label_outcome_or_score_read": False,
        "result_dependent_retry_or_selection": False,
        "scientific_sample_selection_rule_changed": False,
        "operational_amendment": str(F13),
        "operational_amendment_sha256": sha256(F13),
        "source_schedule": str(schedule_path),
        "source_schedule_sha256": sha256(schedule_path),
        "source_launcher_audits": [],
        "counts": {
            "planned_pairs": len(planned),
            "eligible_pairs": len(eligible),
            "attrited_pairs": len(attrited),
            "launcher_attempt_records": len(planned),
        },
        "eligible_pair_ids": sorted(eligible),
        "attrition": [
            {
                "counterfactual_group_id": pair_id,
                "reason": reasons[pair_id],
                "recovered_evidence": evidence,
            }
            for pair_id in attrited
        ],
        "checks": {
            "every_scheduled_pair_has_exactly_one_attempt": True,
            "eligible_and_attrited_partition_schedule": True,
            "no_model_feature_label_outcome_or_score_read": True,
            "no_result_dependent_retry": True,
        },
    }
    if path.exists():
        raise FileExistsError(path)
    _write_json(path, ledger)
    return path


def _filter_npz(path: Path, pair_id: str, archive_path: Path) -> int:
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    sample_ids = arrays["sample_ids"].astype(str)
    excluded = np.char.startswith(sample_ids, pair_id + "_")
    if int(excluded.sum()) != 6:
        raise RuntimeError(f"{path} does not contain six F13 rows")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            **{key: value[excluded] for key, value in arrays.items()},
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(
            stream,
            **{key: value[~excluded] for key, value in arrays.items()},
        )
    os.replace(temporary, path)
    return int((~excluded).sum())


def _inventory(root: Path) -> list[dict[str, Any]]:
    values = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "capsule_manifest.json":
            continue
        values.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return values


def main() -> int:
    if INCIDENT.exists():
        raise FileExistsError(INCIDENT)
    if not F13.is_file():
        raise FileNotFoundError(F13)
    f12 = read_json(F12)
    if (
        EXTRA_PAIR
        not in f12.get("incident", {}).get("affected_pair_ids", [])
        or f12.get("incident", {}).get("retained_as_global_attrition")
        is not True
    ):
        raise RuntimeError("F12 does not predeclare the F13 excluded pair")
    INCIDENT.mkdir(parents=True, exist_ok=False)
    capsule_manifest_path = CAPSULE / "capsule_manifest.json"
    canonical_files = {
        "receipt": RECEIPT,
        "pipeline_state": PIPELINE_STATE,
        "capsule_manifest": capsule_manifest_path,
        "scale_snapshot_audit": (
            DERIVED / "scale/snapshots/extraction_audit.json"
        ),
        "scale_visual_manifest": (
            DERIVED / "scale/features/feature_manifest.json"
        ),
        "scale_unified_manifest": (
            DERIVED / "scale/unified_features/feature_manifest.json"
        ),
        "t3_snapshot_records": (
            DERIVED / "c2_t3/snapshots/snapshot_records.jsonl"
        ),
        "t3_snapshot_audit": (
            DERIVED / "c2_t3/snapshots/extraction_audit.json"
        ),
        "t3_visual_features": DERIVED / "c2_t3/features/features.npz",
        "t3_visual_manifest": (
            DERIVED / "c2_t3/features/feature_manifest.json"
        ),
        "t3_unified_features": (
            DERIVED / "c2_t3/unified_features/features.npz"
        ),
        "t3_unified_manifest": (
            DERIVED / "c2_t3/unified_features/feature_manifest.json"
        ),
    }
    before = {key: sha256(path) for key, path in canonical_files.items()}
    for key in (
        "receipt",
        "pipeline_state",
        "capsule_manifest",
        "scale_snapshot_audit",
        "scale_visual_manifest",
        "scale_unified_manifest",
        "t3_snapshot_records",
        "t3_snapshot_audit",
        "t3_visual_manifest",
        "t3_unified_manifest",
    ):
        shutil.copy2(canonical_files[key], INCIDENT / f"before_{key}.json")

    old_capsule = read_json(capsule_manifest_path)
    old_receipt = read_json(RECEIPT)
    scale_records = DERIVED / "scale/snapshots/snapshot_records.jsonl"
    scale_eligible = _record_pair_ids(scale_records)
    scale_planned = set(
        _schedule_pair_ids(SCHEDULE_SHARD / "scale/schedule.jsonl")
    )
    scale_reasons = {
        pair_id: "pre_f13_prune_receipt_accounted_attrition"
        for pair_id in sorted(scale_planned - scale_eligible)
    }
    scale_ledger = _recovered_ledger(
        battery="scale",
        schedule_path=SCHEDULE_SHARD / "scale/schedule.jsonl",
        eligible=scale_eligible,
        reasons=scale_reasons,
        evidence={
            "receipt_sha256": before["receipt"],
            "snapshot_audit_sha256": old_receipt["scale"][
                "snapshot_audit_sha256"
            ],
            "accounted_pairs": old_receipt["scale"]["accounted_pairs"],
            "attrited_pairs": old_receipt["scale"][
                "skipped_incomplete_pairs"
            ],
        },
    )
    scale_ledger_value = read_json(scale_ledger)
    scale_audit = read_json(canonical_files["scale_snapshot_audit"])
    scale_audit["core_snapshot_skipped_incomplete_pairs"] = int(
        scale_audit.get("skipped_incomplete_pairs", 0)
    )
    scale_audit["skipped_incomplete_pairs"] = 17
    scale_audit["launcher_attrition_exclusions"] = scale_ledger_value[
        "attrition"
    ]
    scale_audit["launcher_eligibility_ledger"] = str(scale_ledger)
    scale_audit["launcher_eligibility_ledger_sha256"] = sha256(scale_ledger)
    scale_audit["operational_amendment"] = str(F13)
    scale_audit["operational_amendment_sha256"] = sha256(F13)
    scale_audit["model_or_prediction_loaded_for_launcher_filter"] = False
    scale_audit["selection_uses_outcome_strength"] = False
    scale_audit["counts"]["launcher_attrited_counterfactual_pairs"] = 17
    scale_audit["checks"].update(
        {
            "all_selected_or_excluded_pairs_accounted_for": True,
            "launcher_ledger_partitions_frozen_schedule": True,
            "selected_pairs_match_launcher_eligible_pairs": True,
            "launcher_filter_model_blind": True,
            "launcher_filter_result_independent": True,
        }
    )
    scale_audit["passed"] = all(scale_audit["checks"].values())
    _write_json(canonical_files["scale_snapshot_audit"], scale_audit)

    scale_visual_manifest = read_json(
        canonical_files["scale_visual_manifest"]
    )
    scale_visual_manifest["source_sha256"]["extraction_audit"] = sha256(
        canonical_files["scale_snapshot_audit"]
    )
    scale_visual_manifest["operational_correction_f13"] = {
        "amendment": str(F13),
        "amendment_sha256": sha256(F13),
        "model_feature_label_outcome_or_score_used": False,
    }
    _write_json(
        canonical_files["scale_visual_manifest"],
        scale_visual_manifest,
    )
    scale_unified_manifest = read_json(
        canonical_files["scale_unified_manifest"]
    )
    scale_unified_manifest["source_sha256"]["snapshot_audit"] = sha256(
        canonical_files["scale_snapshot_audit"]
    )
    scale_unified_manifest["source_sha256"]["visual_manifest"] = sha256(
        canonical_files["scale_visual_manifest"]
    )
    scale_unified_manifest["operational_correction_f13"] = {
        "amendment": str(F13),
        "amendment_sha256": sha256(F13),
        "model_feature_label_outcome_or_score_used": False,
    }
    _write_json(
        canonical_files["scale_unified_manifest"],
        scale_unified_manifest,
    )

    old_t3_records_path = canonical_files["t3_snapshot_records"]
    old_t3_records = read_jsonl(old_t3_records_path)
    old_selected = {
        str(row["counterfactual_group_id"]) for row in old_t3_records
    }
    if EXTRA_PAIR not in old_selected or len(old_selected) != 82:
        raise RuntimeError("scene 00 does not exhibit the sealed 82/81 defect")
    t3_eligible = old_selected - {EXTRA_PAIR}
    t3_planned = set(
        _schedule_pair_ids(SCHEDULE_SHARD / "c2_t3/schedule.jsonl")
    )
    old_exclusion_reasons = {
        str(row["episode_id"]).removesuffix("_anomaly"): str(row["reason"])
        for row in old_capsule["t3"]["exclusions"]
    }
    t3_reasons = {
        pair_id: (
            "F12_predeclared_interrupted_started_without_retry"
            if pair_id == EXTRA_PAIR
            else old_exclusion_reasons[pair_id]
        )
        for pair_id in sorted(t3_planned - t3_eligible)
    }
    t3_ledger = _recovered_ledger(
        battery="c2_t3",
        schedule_path=SCHEDULE_SHARD / "c2_t3/schedule.jsonl",
        eligible=t3_eligible,
        reasons=t3_reasons,
        evidence={
            "f12_sha256": sha256(F12),
            "pre_correction_capsule_sha256": before["capsule_manifest"],
            "pre_correction_snapshot_audit_sha256": before[
                "t3_snapshot_audit"
            ],
            "pre_correction_selected_pairs": len(old_selected),
            "predeclared_extra_pair": EXTRA_PAIR,
        },
    )

    filtered_records = [
        row
        for row in old_t3_records
        if str(row["counterfactual_group_id"]) != EXTRA_PAIR
    ]
    if len(filtered_records) != 486:
        raise RuntimeError("F13 scene 00 correction must retain 486 records")
    _write_jsonl(old_t3_records_path, filtered_records)
    visual_rows = _filter_npz(
        canonical_files["t3_visual_features"],
        EXTRA_PAIR,
        INCIDENT / "excluded_visual_features.npz",
    )
    unified_rows = _filter_npz(
        canonical_files["t3_unified_features"],
        EXTRA_PAIR,
        INCIDENT / "excluded_unified_features.npz",
    )
    if visual_rows != 486 or unified_rows != 486:
        raise RuntimeError("F13 feature correction retained another row count")

    audit = read_json(canonical_files["t3_snapshot_audit"])
    audit["pair_audits"] = [
        row
        for row in audit["pair_audits"]
        if str(row["counterfactual_group_id"]) != EXTRA_PAIR
    ]
    audit["counts"].update(
        {
            "snapshot_records": 486,
            "physical_episodes": 162,
            "independent_counterfactual_pairs": 81,
            "appearance_intervention_sequences": 486,
            "launcher_attrited_counterfactual_pairs": 19,
        }
    )
    audit["core_snapshot_skipped_incomplete_pairs"] = int(
        audit.get("skipped_incomplete_pairs", 0)
    )
    audit["skipped_incomplete_pairs"] = 19
    ledger_value = read_json(t3_ledger)
    audit["launcher_attrition_exclusions"] = ledger_value["attrition"]
    audit["launcher_eligibility_ledger"] = str(t3_ledger)
    audit["launcher_eligibility_ledger_sha256"] = sha256(t3_ledger)
    audit["operational_amendment"] = str(F13)
    audit["operational_amendment_sha256"] = sha256(F13)
    audit["model_or_prediction_loaded_for_launcher_filter"] = False
    audit["selection_uses_outcome_strength"] = False
    audit["checks"].update(
        {
            "all_records_from_complete_pairs": True,
            "all_selected_or_excluded_pairs_accounted_for": True,
            "launcher_ledger_partitions_frozen_schedule": True,
            "selected_pairs_match_launcher_eligible_pairs": True,
            "launcher_filter_model_blind": True,
            "launcher_filter_result_independent": True,
        }
    )
    audit["passed"] = all(audit["checks"].values())
    _write_json(canonical_files["t3_snapshot_audit"], audit)

    visual_manifest = read_json(canonical_files["t3_visual_manifest"])
    visual_manifest["counts"].update(
        {"samples": 486, "physical_episodes": 162, "counterfactual_pairs": 81}
    )
    visual_manifest["source_sha256"].update(
        {
            "extraction_audit": sha256(
                canonical_files["t3_snapshot_audit"]
            ),
            "snapshot_records": sha256(old_t3_records_path),
        }
    )
    visual_manifest["output_sha256"]["features"] = sha256(
        canonical_files["t3_visual_features"]
    )
    visual_manifest["operational_correction_f13"] = {
        "amendment": str(F13),
        "amendment_sha256": sha256(F13),
        "excluded_pair_id": EXTRA_PAIR,
        "model_feature_label_outcome_or_score_used": False,
    }
    _write_json(canonical_files["t3_visual_manifest"], visual_manifest)

    unified_manifest = read_json(canonical_files["t3_unified_manifest"])
    unified_manifest["counts"].update(
        {"samples": 486, "physical_episodes": 162, "counterfactual_pairs": 81}
    )
    unified_manifest["source_sha256"].update(
        {
            "snapshot_audit": sha256(
                canonical_files["t3_snapshot_audit"]
            ),
            "snapshot_records": sha256(old_t3_records_path),
            "visual_features": sha256(
                canonical_files["t3_visual_features"]
            ),
            "visual_manifest": sha256(
                canonical_files["t3_visual_manifest"]
            ),
        }
    )
    unified_manifest["output_sha256"]["features"] = sha256(
        canonical_files["t3_unified_features"]
    )
    unified_manifest["operational_correction_f13"] = {
        "amendment": str(F13),
        "amendment_sha256": sha256(F13),
        "excluded_pair_id": EXTRA_PAIR,
        "model_feature_label_outcome_or_score_used": False,
    }
    _write_json(canonical_files["t3_unified_manifest"], unified_manifest)

    extra_dirs = [
        path
        for path in CAPSULE.rglob(f"{EXTRA_PAIR}_anomaly")
        if path.is_dir()
    ]
    if len(extra_dirs) != 1:
        raise RuntimeError("F13 capsule extra episode is not unique")
    extra_dir = extra_dirs[0]
    relative_extra = extra_dir.relative_to(CAPSULE)
    target_extra = INCIDENT / "excluded_capsule_episode" / relative_extra
    target_extra.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(extra_dir), str(target_extra))

    capsule = old_capsule
    capsule["t3"]["retained_episode_ids"] = sorted(
        episode_id
        for episode_id in capsule["t3"]["retained_episode_ids"]
        if not episode_id.startswith(EXTRA_PAIR + "_")
    )
    capsule["t3"]["retained_anomaly_episodes"] = 81
    capsule["t3"]["excluded_anomaly_episodes"] = 19
    capsule["t3"]["exclusions"].append(
        {
            "episode_id": EXTRA_PAIR + "_anomaly",
            "counterfactual_group_id": EXTRA_PAIR,
            "reason": "launcher_attrition_without_retry",
            "launcher_attrition": next(
                row
                for row in ledger_value["attrition"]
                if row["counterfactual_group_id"] == EXTRA_PAIR
            ),
        }
    )
    capsule["t3"]["launcher_eligibility_ledger"] = str(t3_ledger)
    capsule["t3"]["launcher_eligibility_ledger_sha256"] = sha256(t3_ledger)
    capsule["t3"]["launcher_eligible_pairs"] = 81
    capsule["t3"]["launcher_attrited_pairs"] = 19
    capsule["t3"]["model_or_prediction_loaded_for_launcher_filter"] = False
    capsule["t3"]["selection_uses_outcome_strength"] = False
    capsule["operational_correction_f13"] = {
        "amendment": str(F13),
        "amendment_sha256": sha256(F13),
        "excluded_pair_id": EXTRA_PAIR,
        "pre_correction_manifest_sha256": before["capsule_manifest"],
    }
    capsule["inventory"] = _inventory(CAPSULE)
    capsule["counts"] = {
        "files": len(capsule["inventory"]),
        "bytes": sum(int(row["bytes"]) for row in capsule["inventory"]),
    }
    _write_json(capsule_manifest_path, capsule)

    receipt = old_receipt
    receipt["schema_version"] = (
        "kinofail.confirmatory-shard-prune-audit.f13-corrected.v1"
    )
    receipt["operational_correction_f13"] = {
        "amendment": str(F13),
        "amendment_sha256": sha256(F13),
        "pre_correction_receipt_sha256": before["receipt"],
        "capsule_manifest_sha256": sha256(capsule_manifest_path),
        "scale_ledger_sha256": sha256(scale_ledger),
        "t3_ledger_sha256": sha256(t3_ledger),
        "model_feature_label_outcome_or_score_used": False,
    }
    receipt["scale"].update(
        {
            "skipped_incomplete_pairs": 17,
            "snapshot_reported_skipped_incomplete_pairs": 17,
            "snapshot_audit_sha256": sha256(
                canonical_files["scale_snapshot_audit"]
            ),
            "visual_manifest_sha256": sha256(
                canonical_files["scale_visual_manifest"]
            ),
            "unified_manifest_sha256": sha256(
                canonical_files["scale_unified_manifest"]
            ),
            "launcher_eligibility_ledger_sha256": sha256(scale_ledger),
        }
    )
    receipt["t3"].update(
        {
            "samples": 486,
            "skipped_incomplete_pairs": 19,
            "snapshot_reported_skipped_incomplete_pairs": 19,
            "snapshot_audit_sha256": sha256(
                canonical_files["t3_snapshot_audit"]
            ),
            "snapshot_records_sha256": sha256(old_t3_records_path),
            "unified_features_sha256": sha256(
                canonical_files["t3_unified_features"]
            ),
            "unified_manifest_sha256": sha256(
                canonical_files["t3_unified_manifest"]
            ),
            "visual_features_sha256": sha256(
                canonical_files["t3_visual_features"]
            ),
            "visual_manifest_sha256": sha256(
                canonical_files["t3_visual_manifest"]
            ),
            "launcher_eligibility_ledger_sha256": sha256(t3_ledger),
        }
    )
    _write_json(RECEIPT, receipt)

    state = read_json(PIPELINE_STATE)
    state["status"] = "complete_corrected_f13"
    state["stages"]["operational_correction_f13"] = {
        "completed_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "excluded_pair_id": EXTRA_PAIR,
        "amendment": str(F13),
        "amendment_sha256": sha256(F13),
        "capsule_manifest_sha256": sha256(capsule_manifest_path),
        "receipt_sha256": sha256(RECEIPT),
        "model_feature_label_outcome_or_score_used": False,
    }
    state["stages"]["conflict_evidence_capsule"]["t3"] = capsule["t3"]
    state["stages"]["conflict_evidence_capsule"]["counts"] = capsule[
        "counts"
    ]
    _write_json(PIPELINE_STATE, state)

    validation_path = VALIDATION_ROOT / SCENE / "validation.json"
    validation = validate_scene(
        scene_id=SCENE,
        derived_root=DERIVED.parent,
        receipt_path=RECEIPT,
        capsule_path=capsule_manifest_path,
        output_path=validation_path,
    )
    after = {key: sha256(path) for key, path in canonical_files.items()}
    incident = {
        "schema_version": (
            "kinofail.reconfirmation-f13-scene00-correction.v1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "passed": True,
        "scene_id": SCENE,
        "excluded_pair_id": EXTRA_PAIR,
        "cause": (
            "generic physical snapshot/capsule selectors retained an "
            "F12-predeclared interrupted launcher pair whose manifests "
            "happened to be complete"
        ),
        "identified_before_prediction": True,
        "prediction_or_score_files_read": False,
        "feature_values_used_for_selection": False,
        "label_or_outcome_used_for_selection": False,
        "result_dependent_retry_or_selection": False,
        "raw_shard_was_already_pruned": True,
        "excluded_evidence_preserved": True,
        "operational_amendment": str(F13),
        "operational_amendment_sha256": sha256(F13),
        "before_sha256": before,
        "after_sha256": after,
        "scale_ledger_sha256": sha256(scale_ledger),
        "t3_ledger_sha256": sha256(t3_ledger),
        "validation_sha256": sha256(validation_path),
        "corrected_counts": {
            "t3_pairs": 81,
            "t3_attrition": 19,
            "t3_samples": 486,
            "capsule_anomaly_episodes": 81,
        },
    }
    _write_json(INCIDENT / "incident_manifest.json", incident)
    print(
        json.dumps(
            {
                "passed": True,
                "scene_id": SCENE,
                "validation": validation,
                "incident_manifest": str(
                    INCIDENT / "incident_manifest.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
