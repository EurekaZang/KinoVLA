#!/usr/bin/env python3
"""Preserve the exact raw evidence needed by frozen conflict-v5 features.

The capsule is built and hash-verified before a scene's large raw shard is
pruned.  It retains no extra trajectories: T2 keeps only its observable
arrays and referenced images; T3 keeps validated anomaly manifests, proprio,
telemetry, and the last five RGB frames per frozen appearance view.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from kinofail_reconfirmation_attrition_f13 import (  # noqa: E402
    load_ledger,
    sha256 as _ledger_sha256,
)


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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _copy(source: Path, target: Path) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_hash = _sha256(source)
    target_hash = _sha256(target)
    if source_hash != target_hash:
        raise RuntimeError(f"capsule copy hash mismatch: {source}")
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": target_hash,
    }


def _t2_capsule(
    *,
    scene: str,
    corpus_scene: Path,
    feature_dir: Path,
    output_scene: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feature_manifest_path = feature_dir / "feature_manifest.json"
    records_path = feature_dir / "records.jsonl"
    features_path = feature_dir / "features.npz"
    feature_manifest = _json(feature_manifest_path)
    if (
        feature_manifest.get("passed") is not True
        or feature_manifest.get("scene_id") != scene
        or feature_manifest.get("output_sha256", {}).get("records")
        != _sha256(records_path)
        or feature_manifest.get("output_sha256", {}).get("features")
        != _sha256(features_path)
    ):
        raise RuntimeError("invalid scene T2 features before capsule")
    records = _jsonl(records_path)
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    if any(
        len(rows) != 6
        or {str(row["scene_cluster"]) for row in rows} != {scene}
        for rows in by_case.values()
    ):
        raise RuntimeError("T2 capsule source contains an incomplete feature case")

    inventory = []
    for case_id, rows in sorted(by_case.items()):
        source_case = corpus_scene / "c2_t2" / case_id
        target_case = output_scene / "c2_t2" / case_id
        manifest_path = source_case / "manifest.json"
        observables_path = source_case / "observables.npz"
        manifest = _json(manifest_path)
        if (
            manifest.get("passed") is not True
            or str(manifest.get("case_id")) != case_id
            or str(manifest.get("scene_cluster")) != scene
            or manifest.get("artifacts", {}).get("observables", {}).get(
                "sha256"
            )
            != _sha256(observables_path)
        ):
            raise RuntimeError(f"invalid T2 capsule case: {case_id}")
        required = {Path("manifest.json"), Path("observables.npz")}
        for row in rows:
            required.update(Path(str(value)) for value in row["rgb_paths"])
        case_inventory = [
            _copy(source_case / relative, target_case / relative)
            for relative in sorted(required)
        ]
        inventory.extend(case_inventory)
    return inventory, {
        "cases": len(by_case),
        "samples": len(records),
        "source_feature_manifest": str(feature_manifest_path),
        "source_feature_manifest_sha256": _sha256(feature_manifest_path),
        "source_records_sha256": _sha256(records_path),
        "source_features_sha256": _sha256(features_path),
    }


def _t3_capsule(
    *,
    scene: str,
    corpus_scene: Path,
    schedule_path: Path,
    output_scene: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schedule = _jsonl(schedule_path)
    anomaly_rows = [
        row for row in schedule if str(row["condition"]) == "anomaly"
    ]
    if len(schedule) != 200 or len(anomaly_rows) != 100:
        raise RuntimeError("scene T3 schedule is not the frozen 100-pair design")
    ledger, ledger_path = load_ledger(scene_id=scene, battery="c2_t3")
    eligible_pair_ids = set(ledger["eligible_pair_ids"])
    launcher_attrition = {
        str(row["counterfactual_group_id"]): row
        for row in ledger["attrition"]
    }
    if (
        len(eligible_pair_ids) + len(launcher_attrition) != 100
        or eligible_pair_ids & set(launcher_attrition)
    ):
        raise RuntimeError("F13 T3 launcher ledger does not partition 100 pairs")
    inventory = []
    retained = []
    excluded = []
    for record in anomaly_rows:
        pair_id = str(record["counterfactual_group_id"])
        if pair_id in launcher_attrition:
            excluded.append(
                {
                    "episode_id": str(record["episode_id"]),
                    "counterfactual_group_id": pair_id,
                    "reason": "launcher_attrition_without_retry",
                    "launcher_attrition": launcher_attrition[pair_id],
                }
            )
            continue
        if pair_id not in eligible_pair_ids:
            raise RuntimeError(
                f"T3 pair is absent from F13 launcher ledger: {pair_id}"
            )
        relative_episode = Path(
            str(record["required_outputs"]["episode_manifest"])
        ).parent
        source_episode = corpus_scene / relative_episode
        manifest_path = source_episode / "manifest.json"
        if not manifest_path.is_file():
            excluded.append(
                {
                    "episode_id": str(record["episode_id"]),
                    "reason": "missing_manifest",
                }
            )
            continue
        manifest = _json(manifest_path)
        if (
            manifest.get("artifact_state") != "validated"
            or manifest.get("evaluation_eligible") is not True
            or manifest.get("operator_readback", {}).get("qa_passed")
            is not True
            or manifest.get("runtime_validation", {}).get("passed") is not True
            or manifest.get("episode_id") != record["episode_id"]
            or manifest.get("counterfactual_group_id")
            != record["counterfactual_group_id"]
        ):
            excluded.append(
                {
                    "episode_id": str(record["episode_id"]),
                    "reason": "runtime_validation_failed",
                }
            )
            continue
        required = {Path("manifest.json")}
        proprio = manifest["artifacts"]["proprio"]
        telemetry = manifest["artifacts"]["telemetry"]
        required.add(Path(str(proprio["path"])))
        required.add(Path(str(telemetry["path"])))
        views = manifest["artifacts"]["rgb_views"]
        if set(views) != {"primary", "swap_01", "swap_02"}:
            raise RuntimeError("T3 capsule has another appearance-view contract")
        for entries in views.values():
            if len(entries) < 5:
                raise RuntimeError("T3 capsule source has fewer than five frames")
            required.update(
                Path(str(entry["path"])) for entry in entries[-5:]
            )
        target_episode = output_scene / relative_episode
        episode_inventory = [
            _copy(source_episode / relative, target_episode / relative)
            for relative in sorted(required)
        ]
        inventory.extend(episode_inventory)
        retained.append(str(record["episode_id"]))
    return inventory, {
        "planned_anomaly_episodes": len(anomaly_rows),
        "retained_anomaly_episodes": len(retained),
        "excluded_anomaly_episodes": len(excluded),
        "per_scene_attrition_gate_enforced": False,
        "global_battery_attrition_gate": 0.05,
        "global_gate_delegated_to_finalizer": True,
        "retained_episode_ids": sorted(retained),
        "exclusions": excluded,
        "launcher_eligibility_ledger": str(ledger_path),
        "launcher_eligibility_ledger_sha256": _ledger_sha256(ledger_path),
        "launcher_eligible_pairs": len(eligible_pair_ids),
        "launcher_attrited_pairs": len(launcher_attrition),
        "model_or_prediction_loaded_for_launcher_filter": False,
        "selection_uses_outcome_strength": False,
        "source_schedule": str(schedule_path),
        "source_schedule_sha256": _sha256(schedule_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--corpus-scene", type=Path, required=True)
    parser.add_argument("--schedule-shard", type=Path, required=True)
    parser.add_argument("--t2-feature-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    corpus_scene = args.corpus_scene.resolve()
    schedule_shard = args.schedule_shard.resolve()
    feature_dir = args.t2_feature_dir.resolve()
    output_root = args.output_root.resolve()
    output_scene = output_root / args.scene_id
    if output_scene.exists() or output_scene.is_symlink():
        raise FileExistsError(output_scene)
    if corpus_scene.name != args.scene_id or corpus_scene.is_symlink():
        raise RuntimeError("capsule source must be the physical scene shard")
    if schedule_shard.name != args.scene_id:
        raise RuntimeError("capsule schedule shard does not match the scene")

    output_scene.mkdir(parents=True, exist_ok=False)
    try:
        t2_inventory, t2 = _t2_capsule(
            scene=args.scene_id,
            corpus_scene=corpus_scene,
            feature_dir=feature_dir,
            output_scene=output_scene,
        )
        t3_inventory, t3 = _t3_capsule(
            scene=args.scene_id,
            corpus_scene=corpus_scene,
            schedule_path=schedule_shard / "c2_t3/schedule.jsonl",
            output_scene=output_scene,
        )
        inventory = t2_inventory + t3_inventory
        if len({str(row["path"]) for row in inventory}) != len(inventory):
            raise RuntimeError("conflict capsule output paths are duplicated")
        manifest = {
            "schema_version": "kinofail.reconfirmation-conflict-capsule.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "complete",
            "passed": True,
            "scene_id": args.scene_id,
            "purpose": (
                "minimum hash-preserved evidence for the F0-frozen "
                "conflict-v5 visual/HOG/post-interaction proprio features"
            ),
            "model_or_prediction_loaded": False,
            "selection_uses_outcome_strength": False,
            "t2": t2,
            "t3": t3,
            "counts": {
                "files": len(inventory),
                "bytes": sum(int(row["bytes"]) for row in inventory),
            },
            "inventory": inventory,
        }
        manifest_path = output_scene / "capsule_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = {
            "passed": True,
            "scene_id": args.scene_id,
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "counts": manifest["counts"],
            "t2_cases": t2["cases"],
            "t3_anomaly_episodes": t3["retained_anomaly_episodes"],
        }
    except BaseException:
        # Preserve the incomplete directory for audit; never silently replace it.
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
