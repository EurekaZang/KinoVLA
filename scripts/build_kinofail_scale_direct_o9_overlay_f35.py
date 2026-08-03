#!/usr/bin/env python3
"""Build a complete Scale feature root with legacy O9 removed and F33 inserted.

The output is model blind.  It retains original valid non-O9 groups, adds only
accepted non-O9 F25 replenishments, and maps direct-event F34 O9 observations
back to their original 960 frozen Scale slots.  Legacy O9 features and all F25
O9 features are excluded because their direct-contact semantics were not
proven.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.build_kinofail_confirmatory_o9_pilot_f32 import ROOT, read_jsonl, sha256


ORIGINAL = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"
ORIGINAL_SCHEDULE = ROOT / "outputs/kinofail_reconfirmation_v2/schedules/scale_schedule.jsonl"
F25 = ROOT / "outputs/eval/unified_moe_v3_replenishment_f25/accepted_overlay"
F25_AUDIT = ROOT / "outputs/kinofail_replenishment_f25/final_audit.json"
F33 = ROOT / "outputs/kinofail_confirmatory_o9_final_f33"
F34 = ROOT / "outputs/kinofail_confirmatory_o9_postprocess_f34"
F34_EVAL = ROOT / "outputs/eval/unified_moe_v3_o9_direct_f34"
OUTPUT = ROOT / "outputs/eval/unified_moe_v3_scale_direct_o9_f35"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def load_features(snapshot_dir: Path, unified_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, str]]:
    records_path = snapshot_dir / "snapshot_records.jsonl"
    features_path = unified_dir / "features.npz"
    manifest_path = unified_dir / "feature_manifest.json"
    manifest = read_json(manifest_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("output_sha256", {}).get("features") != sha256(features_path)
        or manifest.get("source_sha256", {}).get("snapshot_records") != sha256(records_path)
    ):
        raise RuntimeError(f"invalid feature shard: {unified_dir}")
    rows = read_jsonl(records_path)
    with np.load(features_path, allow_pickle=False) as archive:
        ids = archive["sample_ids"].astype(str)
        visual = np.asarray(archive["visual"], dtype=np.float32)
        proprio = np.asarray(archive["proprio"], dtype=np.float32)
    if (
        ids.tolist() != [str(row["sample_id"]) for row in rows]
        or visual.shape != (len(rows), 1536)
        or proprio.shape != (len(rows), 80)
        or not np.isfinite(visual).all()
        or not np.isfinite(proprio).all()
    ):
        raise RuntimeError(f"misaligned feature shard: {unified_dir}")
    return rows, visual, proprio, {
        "records": sha256(records_path),
        "features": sha256(features_path),
        "manifest": sha256(manifest_path),
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite F35: {OUTPUT}")
    forbidden = [
        path
        for root in (F34_EVAL, OUTPUT)
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and any(token in path.name.lower() for token in ("prediction", "score"))
    ]
    if forbidden:
        raise RuntimeError("prediction or score artifacts exist before F35 overlay")
    f25_overlay_audit = read_json(F25 / "overlay_audit.json")
    f25_audit = read_json(F25_AUDIT)
    f33_audit = read_json(F33 / "final_audit.json")
    f34_audit = read_json(F34 / "final_audit.json")
    if not all(row.get("passed") is True for row in (f25_overlay_audit, f25_audit, f33_audit, f34_audit)):
        raise RuntimeError("F35 source audit failed")
    if (
        f25_audit.get("model_prediction_feature_label_or_score_read") is not False
        or f25_overlay_audit.get("model_prediction_or_score_read") is not False
        or f25_overlay_audit.get("selection_uses_model_outcome_or_label_strength")
        is not False
        or f25_overlay_audit.get("one_to_one_original_slot_mapping") is not True
        or f25_overlay_audit.get("all_rejected_replacements_excluded") is not True
        or f33_audit.get("model_prediction_feature_label_outcome_or_score_read")
        is not False
        or f33_audit.get("result_dependent_retry_or_selection") is not False
        or f34_audit.get("model_prediction_feature_label_outcome_or_score_read")
        is not False
    ):
        raise RuntimeError("F35 source selection is not fully certified model blind")
    valid_f33_ids = set(str(value) for value in f34_audit["valid_pair_ids"])
    if len(valid_f33_ids) < 750:
        raise RuntimeError("F34 direct-event O9 count is below the frozen gate")
    accepted_f33 = {
        str(row["pair_id"]): str(row["attempt_sha256"])
        for row in f33_audit.get("accepted", [])
    }
    if set(accepted_f33) != valid_f33_ids:
        raise RuntimeError("F33 accepted registry differs from F34 direct-event registry")
    allowed_visual_suffixes = tuple(
        str(value)
        for value in f33_audit.get("allowed_nonphysical_runtime_issue_suffixes", [])
    )
    visual_warning_counts: Counter[str] = Counter()
    visual_warning_pairs = 0
    for pair_id, expected_hash in accepted_f33.items():
        attempt_path = F33 / "attempts" / f"{pair_id}.json"
        if not attempt_path.is_file() or sha256(attempt_path) != expected_hash:
            raise RuntimeError(f"F33 accepted attempt hash mismatch: {pair_id}")
        attempt = read_json(attempt_path)
        runtime_issues = attempt.get("evidence", {}).get("runtime_issues", {})
        flattened = [
            str(issue)
            for values in runtime_issues.values()
            if isinstance(values, list)
            for issue in values
        ]
        if any(
            not any(issue.endswith(suffix) for suffix in allowed_visual_suffixes)
            for issue in flattened
        ):
            raise RuntimeError(f"F33 accepted pair has an undeclared runtime issue: {pair_id}")
        visual_warning_pairs += int(bool(flattened))
        visual_warning_counts.update(flattened)

    schedule_rows = read_jsonl(ORIGINAL_SCHEDULE)
    original_by_group: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in schedule_rows:
        original_by_group[str(row["counterfactual_group_id"])][str(row["condition"])] = row
    if len(original_by_group) != 10560:
        raise RuntimeError("original Scale schedule count mismatch")
    f33_rows = read_jsonl(F33 / "schedule.jsonl")
    f33_anomaly = {
        str(row["counterfactual_group_id"]): row
        for row in f33_rows
        if row["condition"] == "anomaly"
    }
    f33_to_original = {
        pair_id: str(row["o9_semantic_recollection"]["source_counterfactual_group_id"])
        for pair_id, row in f33_anomaly.items()
    }
    if len(f33_to_original) != 960 or len(set(f33_to_original.values())) != 960:
        raise RuntimeError("F33-to-original O9 mapping is not one-to-one")

    by_scene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scene_visual: dict[str, list[np.ndarray]] = defaultdict(list)
    by_scene_proprio: dict[str, list[np.ndarray]] = defaultdict(list)
    source_hashes: dict[str, Any] = {"original": {}, "f25": {}, "f34": {}}

    def append(scene: str, row: dict[str, Any], visual: np.ndarray, proprio: np.ndarray) -> None:
        by_scene_rows[scene].append(row)
        by_scene_visual[scene].append(visual)
        by_scene_proprio[scene].append(proprio)

    for scene_dir in sorted((ORIGINAL / "shards").glob("*")):
        scene = scene_dir.name
        rows, visual, proprio, hashes = load_features(
            scene_dir / "scale/snapshots",
            scene_dir / "scale/unified_features",
        )
        source_hashes["original"][scene] = hashes
        for index, row in enumerate(rows):
            if row["target_operator"] != "O9_high_centering":
                append(scene, dict(row), visual[index], proprio[index])

    for scene_dir in sorted((F25 / "shards").glob("*")):
        scene = scene_dir.name
        rows, visual, proprio, hashes = load_features(
            scene_dir / "scale/accepted_snapshots",
            scene_dir / "scale/unified_features",
        )
        source_hashes["f25"][scene] = hashes
        for index, row in enumerate(rows):
            if row["target_operator"] != "O9_high_centering":
                append(scene, dict(row), visual[index], proprio[index])

    f34_rows, f34_visual, f34_proprio, f34_hashes = load_features(
        F34_EVAL / "snapshots",
        F34_EVAL / "unified_features",
    )
    source_hashes["f34"] = f34_hashes
    seen_f33: set[str] = set()
    for index, source in enumerate(f34_rows):
        f33_id = str(source["counterfactual_group_id"])
        if f33_id not in valid_f33_ids:
            raise RuntimeError(f"F34 record outside valid pair registry: {f33_id}")
        original_id = f33_to_original[f33_id]
        condition = str(source["condition"])
        view_id = str(source["appearance_intervention_id"])
        original = original_by_group[original_id][condition]
        if (
            original["target_operator"] != "O9_high_centering"
            or str(original["scene_id"]) != str(source["scene_family"])
            or str(original["severity_id"]) != str(source["severity_id"])
        ):
            raise RuntimeError(f"F33 O9 slot mismatch: {f33_id}")
        original_episode_id = str(original["episode_id"])
        row = dict(source)
        row.update(
            {
                "sample_id": f"{original_episode_id}__{view_id}",
                "physical_episode_id": original_episode_id,
                "counterfactual_group_id": original_id,
                "statistical_unit_id": original_id,
                "source_f33_sample_id": str(source["sample_id"]),
                "source_f33_physical_episode_id": str(source["physical_episode_id"]),
                "source_f33_counterfactual_group_id": f33_id,
                "o9_replacement_protocol": "F33/F34 direct-contact confirmation",
                "o9_legacy_feature_used": False,
            }
        )
        append(str(source["scene_family"]), row, f34_visual[index], f34_proprio[index])
        seen_f33.add(f33_id)
    if seen_f33 != valid_f33_ids:
        raise RuntimeError("F34 valid O9 pairs are missing from F35 mapping")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    all_group_ids: set[str] = set()
    output_hashes: dict[str, dict[str, str]] = {}
    per_scene_pairs: dict[str, int] = {}
    per_operator: Counter[str] = Counter()
    for scene in sorted(by_scene_rows):
        rows = by_scene_rows[scene]
        visual = np.stack(by_scene_visual[scene]).astype(np.float32)
        proprio = np.stack(by_scene_proprio[scene]).astype(np.float32)
        order = np.argsort(np.asarray([str(row["sample_id"]) for row in rows]), kind="stable")
        rows = [rows[int(index)] for index in order]
        visual = visual[order]
        proprio = proprio[order]
        ids = np.asarray([str(row["sample_id"]) for row in rows])
        groups = {str(row["counterfactual_group_id"]) for row in rows}
        if len(rows) != 6 * len(groups) or len(set(ids.tolist())) != len(ids):
            raise RuntimeError(f"F35 incomplete or duplicate group in {scene}")
        if all_group_ids & groups:
            raise RuntimeError("F35 duplicate group across scenes")
        all_group_ids |= groups
        for group_id in groups:
            exemplar = next(row for row in rows if str(row["counterfactual_group_id"]) == group_id)
            per_operator[str(exemplar["target_operator"])] += 1
        snapshot_dir = OUTPUT / "shards" / scene / "scale/snapshots"
        unified_dir = OUTPUT / "shards" / scene / "scale/unified_features"
        snapshot_dir.mkdir(parents=True)
        unified_dir.mkdir(parents=True)
        records_path = snapshot_dir / "snapshot_records.jsonl"
        features_path = unified_dir / "features.npz"
        write_jsonl(records_path, rows)
        np.savez_compressed(features_path, sample_ids=ids, visual=visual, proprio=proprio)
        manifest = {
            "schema_version": "kinofail.scale-direct-o9-f35-shard.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "complete",
            "model_prediction_or_score_read": False,
            "counts": {
                "samples": len(rows),
                "counterfactual_pairs": len(groups),
                "physical_episodes": 2 * len(groups),
            },
            "dimensions": {"visual": 1536, "proprio": 80},
            "source_sha256": {"snapshot_records": sha256(records_path)},
            "output_sha256": {"features": sha256(features_path)},
        }
        manifest_path = unified_dir / "feature_manifest.json"
        write_json(manifest_path, manifest)
        per_scene_pairs[scene] = len(groups)
        output_hashes[scene] = {
            "records": sha256(records_path),
            "features": sha256(features_path),
            "manifest": sha256(manifest_path),
        }

    valid_pairs = len(all_group_ids)
    expected = 9283 + len(valid_f33_ids)
    if valid_pairs != expected or per_operator["O9_high_centering"] != len(valid_f33_ids):
        raise RuntimeError("F35 effective Scale count mismatch")
    attrition = (10560 - valid_pairs) / 10560
    audit = {
        "schema_version": "kinofail.scale-direct-o9-f35-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_after_model_blind_features_before_prediction",
        "passed": attrition < 0.05 and len(per_scene_pairs) == 30,
        "model_prediction_or_score_read": False,
        "selection_uses_model_outcome_label_or_score": False,
        "legacy_o9_features_excluded": True,
        "f25_o9_features_excluded": True,
        "f33_allowed_visual_runtime_warnings": {
            "role": "reported descriptive QA; never used for result-dependent selection",
            "pairs_with_any_warning": visual_warning_pairs,
            "fraction_of_direct_o9_pairs": visual_warning_pairs / len(valid_f33_ids),
            "issue_counts": dict(sorted(visual_warning_counts.items())),
            "allowed_suffixes": list(allowed_visual_suffixes),
            "all_warning_pairs_retained": True,
        },
        "counts": {
            "planned_scale_pairs": 10560,
            "original_valid_non_o9_pairs": 8302,
            "accepted_f25_non_o9_pairs": 981,
            "direct_event_f34_o9_pairs": len(valid_f33_ids),
            "effective_valid_scale_pairs": valid_pairs,
            "remaining_attrition_pairs": 10560 - valid_pairs,
            "scenes": len(per_scene_pairs),
        },
        "effective_scale_attrition_rate": attrition,
        "pairs_by_operator": dict(sorted(per_operator.items())),
        "pairs_by_scene": per_scene_pairs,
        "source_sha256": {
            "original_schedule": sha256(ORIGINAL_SCHEDULE),
            "f25_overlay_audit": sha256(F25 / "overlay_audit.json"),
            "f25_final_audit": sha256(F25_AUDIT),
            "f33_final_audit": sha256(F33 / "final_audit.json"),
            "f34_final_audit": sha256(F34 / "final_audit.json"),
            "feature_sources": source_hashes,
            "builder": sha256(Path(__file__).resolve()),
        },
        "output_sha256": output_hashes,
    }
    write_json(OUTPUT / "overlay_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
