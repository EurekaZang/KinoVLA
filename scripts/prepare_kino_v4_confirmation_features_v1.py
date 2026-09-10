#!/usr/bin/env python3
"""Prepare frozen observable features after the F4 observation seal.

No classifier or prediction artifact is opened here.  Runtime-valid physical
units are converted to the exact CLIP, 190-D full-body, 80-D invariant-body,
and DINOv2-L/14 descriptors declared by F0.  Any temporal-window attrition is
accounted at its physical unit before an evaluation checkpoint can be loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.data.realistic_snapshots import (  # noqa: E402
    build_event_aligned_snapshots,
    write_snapshot_bundle,
)


PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
HF_HOME = Path("/data/eureka/huggingface")
DESIGN = ROOT / "outputs/kinofail_kino_v4_confirmation_v1"
CORPUS = Path("/data/eureka/kinofail_kino_v4_confirmation_v1/corpus")
F4 = ROOT / "outputs/freeze/kino_v4_confirmation_observations_f4/observation_seal.json"
DEFAULT_OUTPUT = ROOT / "outputs/eval/kino_v4_confirmation_v1"
ATTRITION_LIMIT = 0.05


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _run(arguments: list[str]) -> None:
    subprocess.run(
        [str(PYTHON), *arguments],
        cwd=ROOT,
        check=True,
        env={
            **os.environ,
            "HF_HOME": str(HF_HOME),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )


def _scale_protocol(
    *,
    scene: str,
    schedule_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    original = _json(
        DESIGN / "schedules/scenes" / scene / "scale/snapshot_protocol.json"
    )
    return {
        **original,
        "protocol_id": original["protocol_id"] + "-f4-retained",
        "status": "derived_from_model_blind_f4_observation_seal",
        "source_schedule": str(schedule_path),
        "source_schedule_sha256": _sha256(schedule_path),
        "source_corpus_root": str(CORPUS / scene),
        "output_dir": str(output_dir),
        "development_only": False,
        "selection": {
            **original["selection"],
            "require_evaluation_eligible": True,
            "require_runtime_validation_passed": True,
            "required_collection_protocol_id": (
                f"kinofail-kino-v4-independent-confirmation-v1-{scene}-scale-f3"
            ),
            "on_temporal_alignment_failure": "exclude_complete_pair_and_audit",
        },
        "publication_guard": {
            "may_satisfy_realistic_a0_a7": True,
            "condition": "F4-retained independent confirmation only",
        },
    }


def _prepare_scale(
    output: Path,
    seal: dict[str, Any],
    scenes: list[str],
) -> dict[str, Any]:
    retained = set(seal["retained"]["scale_pair_ids"])
    planned_rows = _jsonl(DESIGN / "schedules/global/scale_schedule.jsonl")
    planned_by_pair = {
        pair_id: rows
        for pair_id, rows in (
            (
                pair_id,
                [
                    row
                    for row in planned_rows
                    if str(row["counterfactual_group_id"]) == pair_id
                ],
            )
            for pair_id in retained
        )
    }
    all_records: list[dict[str, Any]] = []
    all_visual: list[np.ndarray] = []
    all_full: list[np.ndarray] = []
    all_invariant: list[np.ndarray] = []
    temporal_exclusions: list[dict[str, Any]] = []
    source_hashes: dict[str, Any] = {}
    for scene in scenes:
        scene_rows = [
            row
            for row in planned_rows
            if str(row["scene_id"]) == scene
            and str(row["counterfactual_group_id"]) in retained
        ]
        schedule_path = output / "derived_design/scale" / scene / "schedule.jsonl"
        _write_jsonl(schedule_path, scene_rows)
        snapshot_dir = output / "shards" / scene / "scale/snapshots"
        protocol = _scale_protocol(
            scene=scene, schedule_path=schedule_path, output_dir=snapshot_dir
        )
        protocol_path = output / "derived_design/scale" / scene / "snapshot_protocol.json"
        _write_json(protocol_path, protocol)
        records, arrays, audit = build_event_aligned_snapshots(
            protocol, repo_root=ROOT
        )
        planned_lookup = {
            str(row["counterfactual_group_id"]): row for row in scene_rows
        }
        for row in records:
            planned = planned_lookup[str(row["counterfactual_group_id"])]
            row.update(
                {
                    "scene_cluster": scene,
                    "cluster_material": str(planned["cluster_material"]),
                    "material_id": str(planned["material_id"]),
                    "split": "independent_confirmation",
                }
            )
        audit["f4_observation_seal_sha256"] = _sha256(F4)
        hashes = write_snapshot_bundle(
            records, arrays, audit, output_dir=snapshot_dir
        )
        temporal_exclusions.extend(audit["temporal_alignment_exclusions"])
        feature_dir = output / "shards" / scene / "scale/features"
        invariant_dir = output / "shards" / scene / "scale/invariant_features"
        _run(
            [
                "scripts/extract_kinofail_realistic_features.py",
                "--snapshot-dir",
                str(snapshot_dir),
                "--output-dir",
                str(feature_dir),
                "--batch-size",
                "128",
            ]
        )
        _run(
            [
                "scripts/build_kinofail_unified_invariant_features_v1.py",
                "--snapshot-dir",
                str(snapshot_dir),
                "--visual-feature-dir",
                str(feature_dir),
                "--output-dir",
                str(invariant_dir),
            ]
        )
        scene_records = _jsonl(snapshot_dir / "snapshot_records.jsonl")
        with np.load(feature_dir / "features.npz", allow_pickle=False) as archive:
            ids = archive["sample_ids"].astype(str)
            visual = np.asarray(archive["visual"], dtype=np.float32)
            full = np.asarray(archive["proprio"], dtype=np.float32)
        with np.load(invariant_dir / "features.npz", allow_pickle=False) as archive:
            invariant_ids = archive["sample_ids"].astype(str)
            invariant = np.asarray(archive["proprio"], dtype=np.float32)
        expected = [str(row["sample_id"]) for row in scene_records]
        if ids.tolist() != expected or not np.array_equal(ids, invariant_ids):
            raise RuntimeError(f"Scale feature alignment failed: {scene}")
        all_records.extend(scene_records)
        all_visual.append(visual)
        all_full.append(full)
        all_invariant.append(invariant)
        source_hashes[scene] = {
            "schedule": _sha256(schedule_path),
            "snapshot_protocol": _sha256(protocol_path),
            "snapshot_bundle": hashes,
            "features": _sha256(feature_dir / "features.npz"),
            "invariant": _sha256(invariant_dir / "features.npz"),
        }

    order = sorted(
        range(len(all_records)), key=lambda index: str(all_records[index]["sample_id"])
    )
    visual = np.concatenate(all_visual, axis=0)[order]
    full = np.concatenate(all_full, axis=0)[order]
    invariant = np.concatenate(all_invariant, axis=0)[order]
    records = [all_records[index] for index in order]
    ids = np.asarray([str(row["sample_id"]) for row in records])
    retained_pairs = {str(row["counterfactual_group_id"]) for row in records}
    final_attrition = (2_112 - len(retained_pairs)) / 2_112
    if (
        len(records) != 6 * len(retained_pairs)
        or final_attrition >= ATTRITION_LIMIT
        or visual.shape != (len(records), 1536)
        or full.shape != (len(records), 190)
        or invariant.shape != (len(records), 80)
    ):
        raise RuntimeError("final Scale feature or attrition contract failed")
    merged = output / "scale"
    merged.mkdir(parents=True, exist_ok=False)
    records_path = merged / "records.jsonl"
    features_path = merged / "features.npz"
    _write_jsonl(records_path, records)
    np.savez_compressed(
        features_path,
        sample_ids=ids,
        visual=visual,
        full_proprio=full,
        invariant_proprio=invariant,
    )
    report = {
        "planned_pairs": 2_112,
        "retained_pairs_after_runtime_and_temporal_validity": len(retained_pairs),
        "excluded_pairs": 2_112 - len(retained_pairs),
        "attrition_rate": final_attrition,
        "samples": len(records),
        "temporal_exclusions": temporal_exclusions,
        "source_sha256": source_hashes,
        "output_sha256": {
            "records": _sha256(records_path),
            "features": _sha256(features_path),
        },
    }
    _write_json(merged / "feature_manifest.json", {"passed": True, **report})
    return report


def _prepare_t2(
    output: Path,
    seal: dict[str, Any],
    t2_plan: dict[str, dict[str, Any]],
    scenes: list[str],
) -> tuple[Path, dict[str, Any]]:
    retained = set(seal["retained"]["t2_case_ids"])
    all_rows: list[dict[str, Any]] = []
    all_ids: list[np.ndarray] = []
    all_visual: list[np.ndarray] = []
    all_proprio: list[np.ndarray] = []
    sources: dict[str, Any] = {}
    for scene in scenes:
        scene_cases = sorted(
            case_id
            for case_id in retained
            if str(t2_plan[case_id]["scene_cluster"]) == scene
        )
        valid_corpus = output / "derived_design/t2_valid_corpus" / scene
        valid_corpus.mkdir(parents=True, exist_ok=False)
        for case_id in scene_cases:
            os.symlink(
                CORPUS / scene / "c2_t2" / case_id,
                valid_corpus / case_id,
                target_is_directory=True,
            )
        feature_dir = output / "shards" / scene / "conflict/t2_features"
        _run(
            [
                "scripts/extract_kinofail_confirmatory_t2_features_v1.py",
                "--corpus",
                str(valid_corpus),
                "--output",
                str(feature_dir),
                "--scene-id",
                scene,
                "--batch-size",
                "128",
            ]
        )
        rows = _jsonl(feature_dir / "records.jsonl")
        with np.load(feature_dir / "features.npz", allow_pickle=False) as archive:
            ids = archive["sample_ids"].astype(str)
            visual = np.asarray(archive["visual"], dtype=np.float32)
            proprio = np.asarray(archive["proprio"], dtype=np.float32)
        if ids.tolist() != [str(row["sample_id"]) for row in rows]:
            raise RuntimeError(f"T2 feature alignment failed: {scene}")
        for row in rows:
            case_id = str(row["case_id"])
            plan = t2_plan[case_id]
            row.update(
                {
                    "cell": "T2_vision_decisive",
                    "cluster_material": str(plan["cluster_material"]),
                    "material_id": str(plan["material_id"]),
                    "source_case_dir": str(CORPUS / scene / "c2_t2" / case_id),
                    "split": "independent_confirmation",
                }
            )
        all_rows.extend(rows)
        all_ids.append(ids)
        all_visual.append(visual)
        all_proprio.append(proprio)
        sources[scene] = {
            "manifest": _sha256(feature_dir / "feature_manifest.json"),
            "records": _sha256(feature_dir / "records.jsonl"),
            "features": _sha256(feature_dir / "features.npz"),
        }
    order = sorted(
        range(len(all_rows)), key=lambda index: str(all_rows[index]["sample_id"])
    )
    rows = [all_rows[index] for index in order]
    ids = np.concatenate(all_ids)[order]
    visual = np.concatenate(all_visual)[order]
    proprio = np.concatenate(all_proprio)[order]
    cases = {str(row["case_id"]) for row in rows}
    attrition = (288 - len(cases)) / 288
    if len(rows) != 6 * len(cases) or attrition >= ATTRITION_LIMIT:
        raise RuntimeError("final T2 feature or attrition contract failed")
    merged = output / "conflict/t2"
    merged.mkdir(parents=True, exist_ok=False)
    records_path = merged / "records.jsonl"
    features_path = merged / "features.npz"
    _write_jsonl(records_path, rows)
    np.savez_compressed(
        features_path, sample_ids=ids, visual=visual, proprio=proprio
    )
    manifest = {
        "schema_version": "kinofail.kino-v4-confirmation-t2-features.v1",
        "passed": True,
        "counts": {"cases": len(cases), "samples": len(rows)},
        "attrition_rate": attrition,
        "source_sha256": sources,
        "output_sha256": {
            "records": _sha256(records_path),
            "features": _sha256(features_path),
        },
    }
    _write_json(merged / "feature_manifest.json", manifest)
    return merged, manifest


def _prepare_conflict(
    output: Path,
    seal: dict[str, Any],
    scenes: list[str],
) -> dict[str, Any]:
    t2_plan_rows = _jsonl(DESIGN / "schedules/c2_t2/schedule.jsonl")
    t2_plan = {str(row["case_id"]): row for row in t2_plan_rows}
    t2_dir, t2_manifest = _prepare_t2(output, seal, t2_plan, scenes)

    all_t3_cases = _jsonl(DESIGN / "schedules/global/t3_cases.jsonl")
    retained_t3 = set(seal["retained"]["t3_case_ids"])
    t3_cases = [
        row for row in all_t3_cases if str(row["case_id"]) in retained_t3
    ]
    required_groups = {
        str(row[key])
        for row in t3_cases
        for key in ("o7_source_physics_group_id", "o8_source_physics_group_id")
    }
    all_t3_rows: list[dict[str, Any]] = []
    for scene in scenes:
        all_t3_rows.extend(
            _jsonl(DESIGN / "schedules/scenes" / scene / "c2_t3/schedule.jsonl")
        )
    t3_rows = [
        row
        for row in all_t3_rows
        if str(row["counterfactual_group_id"]) in required_groups
    ]
    t3_design = output / "derived_design/t3_valid"
    schedule_path = t3_design / "schedule.jsonl"
    cases_path = t3_design / "case_schedule.jsonl"
    _write_jsonl(schedule_path, t3_rows)
    _write_jsonl(cases_path, t3_cases)
    audit = {
        "schema_version": "kinofail.kino-v4-confirmation-t3-valid-design.v1",
        "passed": True,
        "selection_is_model_blind": True,
        "f4_observation_seal_sha256": _sha256(F4),
        "schedule_sha256": _sha256(schedule_path),
        "case_schedule_sha256": _sha256(cases_path),
        "counts": {"cases": len(t3_cases), "physical_schedule_rows": len(t3_rows)},
    }
    _write_json(t3_design / "audit.json", audit)
    if (
        len(t3_rows) != 4 * len(t3_cases)
        or (288 - len(t3_cases)) / 288 >= ATTRITION_LIMIT
    ):
        raise RuntimeError("final T3 design or attrition contract failed")

    base = output / "conflict/base_features"
    _run(
        [
            "scripts/build_kino_v4_confirmation_conflict_base_features_v1.py",
            "--c1-features",
            str(t2_dir),
            "--c1-corpus",
            str(CORPUS),
            "--t3-design",
            str(t3_design),
            "--t3-corpus",
            str(CORPUS),
            "--output",
            str(base),
            "--batch-size",
            "128",
        ]
    )
    invariant = output / "conflict/invariant_features"
    command = [
        "scripts/build_kinofail_realistic_c2_v5_features.py",
        "--base-features",
        str(base),
        "--t3-corpus",
        str(CORPUS),
        "--output",
        str(invariant),
    ]
    for scene in scenes:
        command.extend(["--t2-corpus", str(CORPUS / scene / "c2_t2")])
    _run(command)
    dino = output / "conflict/dinov2"
    _run(
        [
            "scripts/extract_kino_v4_confirmation_dinov2_v1.py",
            "--records",
            str(base / "records.jsonl"),
            "--output",
            str(dino),
            "--batch-size",
            "64",
        ]
    )
    rows = _jsonl(base / "records.jsonl")
    cases_by_cell = defaultdict(set)
    for row in rows:
        cases_by_cell[str(row["cell"])].add(str(row["case_id"]))
    report = {
        "planned_cases_per_cell": 288,
        "retained_cases": {
            "T2_vision_decisive": len(cases_by_cell["T2_vision_decisive"]),
            "T3_proprio_decisive": len(cases_by_cell["T3_proprio_decisive"]),
        },
        "attrition_rates": {
            "T2_vision_decisive": t2_manifest["attrition_rate"],
            "T3_proprio_decisive": (288 - len(t3_cases)) / 288,
        },
        "samples": len(rows),
        "output_sha256": {
            "records": _sha256(base / "records.jsonl"),
            "base_features": _sha256(base / "features.npz"),
            "invariant_features": _sha256(invariant / "features.npz"),
            "dinov2": _sha256(dino / "features.npz"),
        },
    }
    if any(value >= ATTRITION_LIMIT for value in report["attrition_rates"].values()):
        raise RuntimeError("Conflict attrition exceeds the frozen gate")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    seal = _json(F4)
    if (
        seal.get("passed") is not True
        or seal.get("status") != "sealed_before_representation_or_inference"
        or seal.get("model_checkpoint_loaded") is not False
        or seal.get("prediction_artifact_loaded") is not False
    ):
        raise RuntimeError("invalid model-blind F4 observation seal")
    scenes = [
        str(row["scene_id"])
        for row in _json(DESIGN / "scene_registry.json")["scenes"]
    ]
    if len(scenes) != 12:
        raise RuntimeError("confirmation scene registry drift")
    output.mkdir(parents=True, exist_ok=False)
    scale = _prepare_scale(output, seal, scenes)
    conflict = _prepare_conflict(output, seal, scenes)
    checks = {
        "f4_observation_seal_precedes_features": True,
        "scale_attrition_below_five_percent": scale["attrition_rate"] < ATTRITION_LIMIT,
        "t2_attrition_below_five_percent": conflict["attrition_rates"]["T2_vision_decisive"] < ATTRITION_LIMIT,
        "t3_attrition_below_five_percent": conflict["attrition_rates"]["T3_proprio_decisive"] < ATTRITION_LIMIT,
        "classifier_or_prediction_loaded": False,
        "selection_uses_representation_values": False,
    }
    manifest = {
        "schema_version": "kinofail.kino-v4-confirmation-feature-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_f0_checkpoint_inference",
        "passed": all(checks.values()),
        "checks": checks,
        "scale": scale,
        "conflict": conflict,
        "source_sha256": {"f4_observation_seal": _sha256(F4)},
        "code_sha256": {
            "preparer": _sha256(Path(__file__).resolve()),
            "snapshot_builder": _sha256(
                ROOT / "kino_vla/data/realistic_snapshots.py"
            ),
            "conflict_base_wrapper": _sha256(
                ROOT / "scripts/build_kino_v4_confirmation_conflict_base_features_v1.py"
            ),
            "dino_extractor": _sha256(
                ROOT / "scripts/extract_kino_v4_confirmation_dinov2_v1.py"
            ),
        },
    }
    _write_json(output / "feature_seal.json", manifest)
    if not manifest["passed"]:
        raise RuntimeError("confirmation feature seal failed")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
