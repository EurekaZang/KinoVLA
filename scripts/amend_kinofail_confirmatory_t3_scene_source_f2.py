#!/usr/bin/env python3
"""Create an append-only F2 schedule amendment for a T3 provenance omission.

F1-sealed T3 records accidentally omitted ``scene_source`` although the same
field is present in every F1-sealed Scale record and in the scene registry as
``source``.  The frozen collector requires the field only when serializing
runtime provenance, after the physical rollout.  This utility copies the F1
schedule tree, adds that one registry-derived field to T3 physical records,
updates only dependent hashes/paths, and records the two interrupted
pre-amendment groups as terminal attrition so they cannot be retried.

No seed, operator, parameter, nuisance, scene assignment, material assignment,
case ID, output path, model, feature, threshold, or statistical rule changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/kinofail_confirmatory_v1/schedules"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/kinofail_confirmatory_v1"
    / "schedules_f2_scene_source_amendment"
)
DEFAULT_F2 = (
    ROOT
    / "outputs/freeze"
    / "unified_moe_v3_confirmatory_f2_scene_source_amendment"
)
DEFAULT_CORPUS = ROOT / "outputs/kinofail_confirmatory_v1/corpus"
FAILED_GROUPS = {
    0: "cf_027bbc592efd57b1d98e",
    1: "cf_045268df68296eb2516a",
}


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _verify_f1(f1_path: Path, source_root: Path) -> dict[str, Any]:
    f1 = _json(f1_path)
    sidecar = f1_path.with_name("seal_manifest.sha256")
    if (
        f1.get("status")
        != "sealed_before_anomaly_collection_and_model_inference"
        or f1.get("confirmatory") is not True
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split()[0] != _sha256(f1_path)
    ):
        raise RuntimeError("invalid F1 seal")
    frozen = {
        str(row["path"]): row
        for row in f1.get("frozen_realized_files", [])
        if str(row["path"]).startswith(
            "outputs/kinofail_confirmatory_v1/schedules/"
        )
    }
    actual_files = sorted(path for path in source_root.rglob("*") if path.is_file())
    if len(frozen) != len(actual_files):
        raise RuntimeError("F1 schedule inventory count differs")
    for path in actual_files:
        key = _display(path)
        row = frozen.get(key)
        if (
            row is None
            or int(row["bytes"]) != path.stat().st_size
            or str(row["sha256"]) != _sha256(path)
        ):
            raise RuntimeError(f"F1 schedule artifact changed before F2: {path}")
    return f1


def _amend_rows(
    path: Path,
    *,
    source_by_scene: dict[str, str],
) -> dict[str, Any]:
    before_hash = _sha256(path)
    rows = _jsonl(path)
    before = [dict(row) for row in rows]
    for row in rows:
        if "scene_source" in row:
            raise RuntimeError(f"T3 row already has scene_source: {path}")
        scene_id = str(row["scene_family"])
        row["scene_source"] = source_by_scene[scene_id]
    for old, new in zip(before, rows, strict=True):
        restored = dict(new)
        restored.pop("scene_source")
        if restored != old:
            raise RuntimeError("F2 changed a field other than scene_source")
    _write_jsonl(path, rows)
    return {
        "path": _display(path),
        "records": len(rows),
        "before_sha256": before_hash,
        "after_sha256": _sha256(path),
        "only_added_field": "scene_source",
    }


def _replace_root_path(value: str, source_root: Path, output_root: Path) -> str:
    source = _display(source_root)
    output = _display(output_root)
    if not value.startswith(source + "/"):
        raise RuntimeError(f"protocol path is outside F1 schedule root: {value}")
    return output + value[len(source) :]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1-manifest", type=Path, required=True)
    parser.add_argument("--scene-registry", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--f2-out", type=Path, default=DEFAULT_F2)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()

    f1_path = args.f1_manifest.resolve()
    registry_path = args.scene_registry.resolve()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    f2_out = args.f2_out.resolve()
    corpus_root = args.corpus_root.resolve()
    for target in (output_root, f2_out):
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
    f1 = _verify_f1(f1_path, source_root)
    registry = _json(registry_path)
    scenes = {
        str(row["scene_id"]): row for row in registry.get("scenes", [])
    }
    source_by_scene = {
        scene_id: str(row["source"]) for scene_id, row in scenes.items()
    }
    if len(source_by_scene) != 30 or any(not value for value in source_by_scene.values()):
        raise RuntimeError("scene registry lacks thirty source descriptors")
    if (ROOT / "outputs/eval/unified_moe_v3_confirmatory_v1").exists():
        raise RuntimeError("F2 must precede confirmatory feature/model inference")

    shutil.copytree(source_root, output_root, symlinks=False)
    changes: list[dict[str, Any]] = []
    global_t3 = output_root / "c2_t3/schedule.jsonl"
    changes.append(_amend_rows(global_t3, source_by_scene=source_by_scene))
    global_audit_path = output_root / "c2_t3/audit.json"
    global_audit = _json(global_audit_path)
    global_audit["schedule_sha256"] = _sha256(global_t3)
    global_audit["operational_amendment"] = (
        "F2 adds registry-derived scene_source provenance only"
    )
    _write_json(global_audit_path, global_audit)

    for scene_id in sorted(scenes):
        scene_root = output_root / "scenes" / scene_id
        schedule_path = scene_root / "c2_t3/schedule.jsonl"
        changes.append(_amend_rows(schedule_path, source_by_scene=source_by_scene))
        collection_path = scene_root / "c2_t3/collection_protocol.json"
        collection = _json(collection_path)
        collection["schedule_path"] = _replace_root_path(
            str(collection["schedule_path"]), source_root, output_root
        )
        collection["schedule_sha256"] = _sha256(schedule_path)
        collection["operational_amendment"] = (
            "F2 scene_source provenance completion; scientific design unchanged"
        )
        _write_json(collection_path, collection)
        snapshot_path = scene_root / "c2_t3/snapshot_protocol.json"
        snapshot = _json(snapshot_path)
        snapshot["source_schedule"] = _replace_root_path(
            str(snapshot["source_schedule"]), source_root, output_root
        )
        snapshot["source_schedule_sha256"] = _sha256(schedule_path)
        snapshot["operational_amendment"] = (
            "F2 scene_source provenance completion; selection unchanged"
        )
        _write_json(snapshot_path, snapshot)
        shard_path = scene_root / "shard_manifest.json"
        shard = _json(shard_path)
        shard["artifacts"]["t3_schedule"] = _sha256(schedule_path)
        shard["artifacts"]["t3_collection_protocol"] = _sha256(collection_path)
        shard["artifacts"]["t3_snapshot_protocol"] = _sha256(snapshot_path)
        shard["operational_amendment"] = (
            "F2 scene_source provenance completion"
        )
        _write_json(shard_path, shard)

    root_manifest_path = output_root / "manifest.json"
    root_manifest = _json(root_manifest_path)
    root_manifest["global_artifacts"]["t3_schedule"]["sha256"] = _sha256(global_t3)
    root_manifest["global_artifacts"]["t3_design_audit"]["sha256"] = _sha256(
        global_audit_path
    )
    for row in root_manifest["shards"]:
        shard_path = output_root / str(row["manifest"])
        row["manifest_sha256"] = _sha256(shard_path)
    root_manifest["operational_amendment"] = {
        "id": "F2_scene_source_provenance_completion",
        "scientific_design_changed": False,
        "model_or_endpoint_outcomes_used": False,
    }
    _write_json(root_manifest_path, root_manifest)

    failed_scene = "confirm_v1_life_scene_00"
    failed_schedule = (
        output_root / "scenes" / failed_scene / "c2_t3/schedule.jsonl"
    )
    failed_protocol = (
        output_root
        / "scenes"
        / failed_scene
        / "c2_t3/collection_protocol.json"
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in _jsonl(failed_schedule):
        groups.setdefault(str(row["counterfactual_group_id"]), []).append(row)
    pair_ids = sorted(groups)
    launcher_root = corpus_root / failed_scene / "launcher_audits"
    launcher_root.mkdir(parents=True, exist_ok=True)
    terminal_attempts = []
    for partition_index, group_id in FAILED_GROUPS.items():
        selected = [
            pair_id
            for index, pair_id in enumerate(pair_ids)
            if index % 2 == partition_index
        ]
        if not selected or selected[0] != group_id:
            raise RuntimeError("interrupted group is not the first frozen partition item")
        attempt = {
            "counterfactual_group_id": group_id,
            "returncode": 1,
            "summary_exists": False,
            "passed": False,
            "completed_utc": datetime.now(UTC).isoformat(),
            "retry_authorized": False,
            "failure_reason": "pre-F2 missing scene_source provenance field",
        }
        audit = {
            "schema_version": "kinofail.confirmatory-shard-launcher.v1",
            "updated_utc": datetime.now(UTC).isoformat(),
            "scene_id": failed_scene,
            "partition_index": partition_index,
            "partition_count": 2,
            "schedule": str(failed_schedule),
            "schedule_sha256": _sha256(failed_schedule),
            "protocol": str(failed_protocol),
            "protocol_sha256": _sha256(failed_protocol),
            "collector_sha256": str(
                _json(failed_protocol)["collector_sha256"]
            ),
            "material_lock_sha256": str(
                _json(failed_protocol)["material_lock_sha256"]
            ),
            "selected_pair_ids": selected,
            "attempts": [attempt],
            "operational_amendment": "F2; interrupted group retained as attrition",
        }
        audit_path = launcher_root / f"partition_{partition_index}_of_2.json"
        if audit_path.exists():
            raise FileExistsError(audit_path)
        _write_json(audit_path, audit)
        terminal_attempts.append(
            {
                "partition_index": partition_index,
                "group_id": group_id,
                "launcher_audit": _display(audit_path),
                "launcher_audit_sha256": _sha256(audit_path),
            }
        )

    scientific_keys = {
        "episode_id",
        "counterfactual_group_id",
        "case_id",
        "condition",
        "target_operator",
        "operator_seed",
        "physical_seed",
        "physics_parameters",
        "physical_nuisance",
        "scene_family",
        "source_scene_id",
        "scene_seed",
        "material_id",
        "cluster_material",
        "appearance_views",
        "required_outputs",
    }
    manifest = {
        "schema_version": "kinofail.unified-confirmatory-f2-amendment.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_resuming_t3_and_before_model_inference",
        "confirmatory": True,
        "amendment_reason": (
            "T3 schedule rows omitted the mandatory provenance-only scene_source "
            "field already present in Scale rows and the F1 scene registry."
        ),
        "change": {
            "only_added_field": "scene_source",
            "value_source": "F1 scene_registry.scenes[].source",
            "scientific_design_changed": False,
            "architecture_changed": False,
            "features_changed": False,
            "threshold_eta_changed": False,
            "statistics_changed": False,
            "seeds_changed": False,
            "operators_or_parameters_changed": False,
            "scenes_or_materials_changed": False,
            "case_or_group_ids_changed": False,
            "protected_scientific_keys": sorted(scientific_keys),
        },
        "interrupted_groups": terminal_attempts,
        "retry_policy": "two interrupted incomplete groups are terminal attrition",
        "model_predictions_available_at_amendment": False,
        "source_sha256": {
            "f1_manifest": _sha256(f1_path),
            "f1_schedule_manifest": f1["input_sha256"]["schedule_manifest"],
            "scene_registry": _sha256(registry_path),
            "amendment_script": _sha256(Path(__file__).resolve()),
        },
        "amended_schedule_root": {
            "path": _display(output_root),
            "manifest_sha256": _sha256(root_manifest_path),
        },
        "changed_schedules": changes,
        "counts": {
            "global_t3_records": len(_jsonl(global_t3)),
            "scene_t3_schedules": len(scenes),
            "interrupted_groups_excluded": len(terminal_attempts),
        },
    }
    f2_out.mkdir(parents=True, exist_ok=False)
    f2_manifest = f2_out / "amendment_manifest.json"
    _write_json(f2_manifest, manifest)
    (f2_out / "amendment_manifest.sha256").write_text(
        f"{_sha256(f2_manifest)}  amendment_manifest.json\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "amended_schedule_root": str(output_root),
                "f2_manifest": str(f2_manifest),
                "f2_sha256": _sha256(f2_manifest),
                "t3_records": manifest["counts"]["global_t3_records"],
                "interrupted_groups_excluded": len(terminal_attempts),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
