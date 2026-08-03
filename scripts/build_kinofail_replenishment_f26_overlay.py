#!/usr/bin/env python3
"""Build the model-blind F25 replacement overlay before any prediction.

F25 is a post-hoc, one-to-one replenishment of the 1,417 Scale slots that
attrited in the independent reconfirmation.  This script admits only the
1,005 replacements accepted by the sealed F25 completion audit, maps each
replacement back to its original frozen design slot, and emits observable
feature shards that can be supplied beside the retained original shards.

The original 10,560-group schedule remains the truth ledger.  Replacement
source identifiers and hashes are retained in every remapped record and in the
root audit.  Rejected replacements never enter the model-facing overlay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_F25_ROOT = ROOT / "outputs/kinofail_replenishment_f25"
DEFAULT_F25_EVAL = ROOT / "outputs/eval/unified_moe_v3_replenishment_f25"
DEFAULT_ORIGINAL = ROOT / "outputs/kinofail_reconfirmation_v2"
DEFAULT_OUTPUT = DEFAULT_F25_EVAL / "accepted_overlay"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(path)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prediction_or_score_artifacts(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and ("prediction" in path.name.lower() or "score" in path.name.lower())
    ]


def _schedule_by_group(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(path):
        grouped[str(row["counterfactual_group_id"])].append(row)
    if len(grouped) != 10_560:
        raise RuntimeError(f"unexpected original Scale group count: {len(grouped)}")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for group_id, rows in grouped.items():
        by_condition = {str(row["condition"]): row for row in rows}
        if len(rows) != 2 or set(by_condition) != {
            "nominal_counterfactual",
            "anomaly",
        }:
            raise RuntimeError(f"invalid original Scale pair: {group_id}")
        result[group_id] = by_condition
    return result


def _original_attrition_ids(original_root: Path) -> set[str]:
    ids: set[str] = set()
    for path in sorted((original_root / "attrition_ledgers_f13").glob("*/scale.json")):
        for row in _json(path)["attrition"]:
            pair_id = str(row["counterfactual_group_id"])
            if pair_id in ids:
                raise RuntimeError(f"duplicate original attrition id: {pair_id}")
            ids.add(pair_id)
    if len(ids) != 1_417:
        raise RuntimeError(f"unexpected original Scale attrition: {len(ids)}")
    return ids


def _load_feature_shard(
    snapshot_dir: Path, unified_dir: Path
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, str]]:
    records_path = snapshot_dir / "snapshot_records.jsonl"
    features_path = unified_dir / "features.npz"
    manifest_path = unified_dir / "feature_manifest.json"
    manifest = _json(manifest_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("output_sha256", {}).get("features") != _sha256(features_path)
        or manifest.get("source_sha256", {}).get("snapshot_records")
        != _sha256(records_path)
    ):
        raise RuntimeError(f"invalid F25 unified feature shard: {unified_dir}")
    rows = _jsonl(records_path)
    with np.load(features_path, allow_pickle=False) as archive:
        sample_ids = archive["sample_ids"].astype(str)
        visual = np.asarray(archive["visual"], dtype=np.float32)
        proprio = np.asarray(archive["proprio"], dtype=np.float32)
    if (
        sample_ids.tolist() != [str(row["sample_id"]) for row in rows]
        or visual.shape != (len(rows), 1536)
        or proprio.shape != (len(rows), 80)
        or not np.isfinite(visual).all()
        or not np.isfinite(proprio).all()
    ):
        raise RuntimeError(f"misaligned F25 feature shard: {unified_dir}")
    return rows, visual, proprio, {
        "snapshot_records": _sha256(records_path),
        "features": _sha256(features_path),
        "feature_manifest": _sha256(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f25-root", type=Path, default=DEFAULT_F25_ROOT)
    parser.add_argument("--f25-eval-root", type=Path, default=DEFAULT_F25_EVAL)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    f25_root = args.f25_root.resolve()
    f25_eval = args.f25_eval_root.resolve()
    original = args.original_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite F26 overlay: {output}")
    forbidden = _prediction_or_score_artifacts(f25_eval)
    if forbidden:
        raise RuntimeError(
            "prediction/score artifact exists before F26 overlay seal: "
            f"{forbidden[:3]}"
        )

    final_audit_path = f25_root / "final_audit.json"
    physical_freeze_path = f25_root / "freeze_manifest.json"
    postprocess_freeze_path = f25_root / "postprocess_freeze_manifest.json"
    schedule_path = original / "schedules/scale_schedule.jsonl"
    final_audit = _json(final_audit_path)
    physical_freeze = _json(physical_freeze_path)
    postprocess_freeze = _json(postprocess_freeze_path)
    if (
        final_audit.get("passed") is not True
        or final_audit.get("status") != "final"
        or final_audit.get("model_prediction_feature_label_or_score_read") is not False
        or final_audit.get("freeze_sha256") != _sha256(physical_freeze_path)
        or physical_freeze.get("status") != "sealed_before_collection"
        or physical_freeze.get("model_prediction_or_score_read") is not False
        or postprocess_freeze.get("status")
        != "sealed_before_model_blind_feature_extraction"
        or postprocess_freeze.get("model_prediction_label_outcome_or_score_read")
        is not False
    ):
        raise RuntimeError("invalid F25 freeze/audit chain")

    accepted = final_audit["accepted"]
    rejected = final_audit["rejected"]
    if len(accepted) != 1_005 or len(rejected) != 412:
        raise RuntimeError("unexpected F25 accepted/rejected counts")
    accepted_by_replacement = {
        str(row["replacement_counterfactual_group_id"]): row for row in accepted
    }
    if len(accepted_by_replacement) != len(accepted):
        raise RuntimeError("duplicate accepted replacement id")
    accepted_original_ids = {
        str(row["original_counterfactual_group_id"]) for row in accepted
    }
    if len(accepted_original_ids) != len(accepted):
        raise RuntimeError("accepted replacements are not one-to-one")
    if not accepted_original_ids <= _original_attrition_ids(original):
        raise RuntimeError("F25 accepted mapping is outside original attrition")

    schedule = _schedule_by_group(schedule_path)
    by_scene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scene_visual: dict[str, list[np.ndarray]] = defaultdict(list)
    by_scene_proprio: dict[str, list[np.ndarray]] = defaultdict(list)
    source_hashes: dict[str, dict[str, str]] = {}
    seen_replacements: set[str] = set()

    for scene_dir in sorted((f25_eval / "shards").glob("*")):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        snapshot_dir = scene_dir / "scale/snapshots"
        unified_dir = scene_dir / "scale/unified_features"
        rows, visual, proprio, hashes = _load_feature_shard(
            snapshot_dir, unified_dir
        )
        source_hashes[scene_id] = hashes
        for index, source in enumerate(rows):
            replacement_id = str(source["counterfactual_group_id"])
            mapping = accepted_by_replacement.get(replacement_id)
            if mapping is None:
                continue
            original_id = str(mapping["original_counterfactual_group_id"])
            condition = str(source["condition"])
            view_id = str(source["appearance_intervention_id"])
            original_episode = schedule[original_id][condition]
            if (
                str(mapping["scene_id"]) != scene_id
                or str(mapping["target_operator"]) != str(source["target_operator"])
                or str(mapping["severity_id"]) != str(source["severity_id"])
                or str(original_episode["scene_family"]) != str(source["scene_family"])
                or str(original_episode["target_operator"])
                != str(source["target_operator"])
                or str(original_episode["attribution_category"])
                != str(source["attribution_category"])
                or view_id
                not in {
                    str(view["appearance_view_id"])
                    for view in original_episode["appearance_views"]
                }
            ):
                raise RuntimeError(f"replacement/original slot mismatch: {replacement_id}")
            original_episode_id = str(original_episode["episode_id"])
            remapped = dict(source)
            remapped.update(
                {
                    "sample_id": f"{original_episode_id}__{view_id}",
                    "physical_episode_id": original_episode_id,
                    "counterfactual_group_id": original_id,
                    "statistical_unit_id": original_id,
                    "source_replacement_sample_id": str(source["sample_id"]),
                    "source_replacement_physical_episode_id": str(
                        source["physical_episode_id"]
                    ),
                    "source_replacement_counterfactual_group_id": replacement_id,
                    "replenishment_protocol": "F25-v1",
                    "replenishment_analysis_role": "posthoc_supplementary_overlay",
                    "replenishment_attempt_sha256": str(mapping["attempt_sha256"]),
                    "replenishment_summary_sha256": str(mapping["summary_sha256"]),
                }
            )
            by_scene_rows[scene_id].append(remapped)
            by_scene_visual[scene_id].append(visual[index])
            by_scene_proprio[scene_id].append(proprio[index])
            seen_replacements.add(replacement_id)

    if seen_replacements != set(accepted_by_replacement):
        missing = sorted(set(accepted_by_replacement) - seen_replacements)
        raise RuntimeError(f"accepted F25 replacements missing from features: {missing[:8]}")

    output.mkdir(parents=True, exist_ok=False)
    emitted_ids: set[str] = set()
    per_scene_counts: dict[str, int] = {}
    output_hashes: dict[str, dict[str, str]] = {}
    for scene_id in sorted(by_scene_rows):
        rows = by_scene_rows[scene_id]
        visual = np.stack(by_scene_visual[scene_id]).astype(np.float32)
        proprio = np.stack(by_scene_proprio[scene_id]).astype(np.float32)
        order = np.argsort(
            np.asarray([str(row["sample_id"]) for row in rows]), kind="stable"
        )
        rows = [rows[int(index)] for index in order]
        visual = visual[order]
        proprio = proprio[order]
        sample_ids = np.asarray([str(row["sample_id"]) for row in rows])
        if len(set(sample_ids.tolist())) != len(sample_ids):
            raise RuntimeError(f"duplicate remapped sample id in {scene_id}")
        if emitted_ids & set(sample_ids.tolist()):
            raise RuntimeError("duplicate remapped sample id across scenes")
        emitted_ids |= set(sample_ids.tolist())

        snapshot_out = output / "shards" / scene_id / "scale/accepted_snapshots"
        unified_out = output / "shards" / scene_id / "scale/unified_features"
        snapshot_out.mkdir(parents=True)
        unified_out.mkdir(parents=True)
        records_out = snapshot_out / "snapshot_records.jsonl"
        features_out = unified_out / "features.npz"
        _write_jsonl(records_out, rows)
        np.savez_compressed(
            features_out,
            sample_ids=sample_ids,
            visual=visual,
            proprio=proprio,
        )
        pair_count = len({str(row["counterfactual_group_id"]) for row in rows})
        if len(rows) != 6 * pair_count:
            raise RuntimeError(f"non-complete remapped pair in {scene_id}")
        manifest = {
            "schema_version": "kinofail.f26-replenishment-overlay-shard.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "complete",
            "model_prediction_or_score_read": False,
            "selection": "F25 final_audit accepted=true only",
            "slot_mapping": "replacement physical pair to original frozen Scale slot",
            "counts": {
                "samples": len(rows),
                "counterfactual_pairs": pair_count,
                "physical_episodes": 2 * pair_count,
            },
            "dimensions": {"visual": 1536, "proprio": 80},
            "source_sha256": {
                "snapshot_records": _sha256(records_out),
                "f25_source_snapshot_records": source_hashes[scene_id][
                    "snapshot_records"
                ],
                "f25_source_features": source_hashes[scene_id]["features"],
                "f25_source_feature_manifest": source_hashes[scene_id][
                    "feature_manifest"
                ],
                "f25_final_audit": _sha256(final_audit_path),
                "original_scale_schedule": _sha256(schedule_path),
            },
            "output_sha256": {"features": _sha256(features_out)},
        }
        manifest_out = unified_out / "feature_manifest.json"
        _write_json(manifest_out, manifest)
        per_scene_counts[scene_id] = pair_count
        output_hashes[scene_id] = {
            "records": _sha256(records_out),
            "features": _sha256(features_out),
            "manifest": _sha256(manifest_out),
        }

    expected_samples = 6 * len(accepted)
    if len(emitted_ids) != expected_samples or sum(per_scene_counts.values()) != len(
        accepted
    ):
        raise RuntimeError("F26 overlay count mismatch")
    rejected_ids = {
        str(row["replacement_counterfactual_group_id"]) for row in rejected
    }
    if seen_replacements & rejected_ids:
        raise RuntimeError("rejected F25 replacement entered overlay")

    root_audit = {
        "schema_version": "kinofail.f26-replenishment-overlay.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_after_model_blind_features_before_prediction",
        "passed": True,
        "posthoc_supplementary_replenishment": True,
        "original_confirmation_must_be_reported_separately": True,
        "model_prediction_or_score_read": False,
        "selection_uses_model_outcome_or_label_strength": False,
        "one_to_one_original_slot_mapping": True,
        "all_rejected_replacements_excluded": True,
        "counts": {
            "original_scale_planned": 10_560,
            "original_scale_attrited": 1_417,
            "accepted_replacements": len(accepted),
            "remaining_scale_attrition": 412,
            "replenished_valid_scale_groups": 10_148,
            "overlay_samples": len(emitted_ids),
            "scenes": len(per_scene_counts),
        },
        "attrition": {
            "original_scale_rate": 1_417 / 10_560,
            "replenished_scale_rate": 412 / 10_560,
        },
        "source_sha256": {
            "f25_physical_freeze": _sha256(physical_freeze_path),
            "f25_postprocess_freeze": _sha256(postprocess_freeze_path),
            "f25_final_audit": _sha256(final_audit_path),
            "original_scale_schedule": _sha256(schedule_path),
            "builder": _sha256(Path(__file__).resolve()),
            "feature_shards": source_hashes,
        },
        "per_scene_accepted_pairs": per_scene_counts,
        "output_sha256": output_hashes,
    }
    audit_out = output / "overlay_audit.json"
    _write_json(audit_out, root_audit)
    print(json.dumps(root_audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
