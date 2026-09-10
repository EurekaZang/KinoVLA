#!/usr/bin/env python3
"""Prepare and merge all-191 Scale/T2 features before frozen-model scoring."""

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
from kino_vla.eval.c2_temporal_v5 import invariant_summary  # noqa: E402


PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
HF_HOME = Path("/data/eureka/huggingface")
DESIGN = ROOT / "outputs/kinofail_kino_v4_all191_scale_t2_extension_v2"
REGISTRY = DESIGN / "design/scene_registry.json"
SCALE_SCHEDULE = DESIGN / (
    "schedules/operational_supplement_v1/global/"
    "evaluation_scale_schedule.jsonl"
)
REPLACEMENT_MAP = DESIGN / (
    "schedules/operational_supplement_v1/replacement_map.json"
)
T2_SCHEDULE = DESIGN / "schedules/c2_t2/schedule.jsonl"
OBSERVATION_SEAL = ROOT / "outputs/freeze/kino_v4_all191_scale_t2_observations_v1/observation_seal.json"
CORPUS = Path("/data/eureka/kinofail_kino_v4_all191_scale_t2_extension_v2/corpus")
CORE = Path("/data/eureka/kinovla_outputs/kino_v4_confirmation_v1_f5d")
DEFAULT_OUTPUT = Path("/data/eureka/kinovla_outputs/kino_v4_all191_v1")
ATTRITION_LIMIT = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values), encoding="utf-8")


def run(arguments: list[str]) -> None:
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


def load_scale_shard(shard: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    snapshot = shard / "snapshots"
    feature = shard / "features"
    invariant = shard / "invariant_features"
    snapshot_rows = rows(snapshot / "snapshot_records.jsonl")
    if load(snapshot / "extraction_audit.json").get("passed") is not True:
        raise RuntimeError(f"failed snapshot shard: {shard}")
    if load(feature / "feature_manifest.json").get("status") != "complete":
        raise RuntimeError(f"failed visual feature shard: {shard}")
    if load(invariant / "feature_manifest.json").get("status") != "complete":
        raise RuntimeError(f"failed invariant feature shard: {shard}")
    with np.load(feature / "features.npz", allow_pickle=False) as archive:
        ids = archive["sample_ids"].astype(str)
        visual = np.asarray(archive["visual"], dtype=np.float32)
        full = np.asarray(archive["proprio"], dtype=np.float32)
    with np.load(invariant / "features.npz", allow_pickle=False) as archive:
        invariant_ids = archive["sample_ids"].astype(str)
        invariant_values = np.asarray(archive["proprio"], dtype=np.float32)
    expected = np.asarray([str(value["sample_id"]) for value in snapshot_rows])
    if not np.array_equal(ids, expected) or not np.array_equal(ids, invariant_ids):
        raise RuntimeError(f"Scale shard feature alignment failed: {shard}")
    return snapshot_rows, visual, full, invariant_values


def prepare_scale(output: Path, seal: dict[str, Any], scene_ids: list[str]) -> dict[str, Any]:
    retained = set(seal["retained"]["scale_pair_ids"])
    replacement = load(REPLACEMENT_MAP)
    replacement_pair_ids = sorted(replacement["old_to_new_pair_id"].values())
    schedule_rows = rows(SCALE_SCHEDULE)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in schedule_rows:
        if str(value["counterfactual_group_id"]) in retained:
            by_scene[str(value["scene_id"])].append(value)
    extension_rows: list[dict[str, Any]] = []
    extension_visual = []
    extension_full = []
    extension_invariant = []
    sources = {}
    temporal_exclusions = []
    for position, scene_id in enumerate(scene_ids, start=1):
        shard = output / "shards" / scene_id / "scale"
        snapshot_dir = shard / "snapshots"
        feature_dir = shard / "features"
        invariant_dir = shard / "invariant_features"
        if not (feature_dir / "feature_manifest.json").is_file():
            scene_rows = by_scene[scene_id]
            schedule_path = output / "derived_design/scale" / scene_id / "schedule.jsonl"
            write_jsonl(schedule_path, scene_rows)
            original = load(DESIGN / "schedules/scenes" / scene_id / "scale/snapshot_protocol.json")
            protocol = {
                **original,
                "protocol_id": str(original["protocol_id"]) + "-model-blind-retained",
                "status": "derived_from_model_blind_all191_observation_seal",
                "source_schedule": str(schedule_path),
                "source_schedule_sha256": sha256(schedule_path),
                "source_corpus_root": str(CORPUS / scene_id),
                "output_dir": str(snapshot_dir),
                "development_only": False,
                "selection": {
                    **original["selection"],
                    "require_evaluation_eligible": False,
                    "require_runtime_validation_passed": True,
                    "required_collection_protocol_id": (
                        f"kinofail-kino-v4-all191-scale-t2-extension-v2-{scene_id}-scale"
                    ),
                },
                "publication_guard": {
                    "may_satisfy_realistic_a0_a7": True,
                    "condition": "model-blind retained all191 extension only",
                },
            }
            if scene_id == replacement["scene_id"]:
                protocol["source_corpus_overlays"] = [
                    {
                        "corpus_root": str(CORPUS / scene_id),
                        "required_collection_protocol_id": (
                            "kinofail-kino-v4-all191-operational-"
                            f"supplement-v1-{scene_id}"
                        ),
                        "counterfactual_group_ids": replacement_pair_ids,
                    }
                ]
            protocol_path = output / "derived_design/scale" / scene_id / "snapshot_protocol.json"
            write_json(protocol_path, protocol)
            snapshot_rows, arrays, audit = build_event_aligned_snapshots(protocol, repo_root=ROOT)
            plan = {str(value["counterfactual_group_id"]): value for value in scene_rows}
            for value in snapshot_rows:
                source = plan[str(value["counterfactual_group_id"])]
                value.update(
                    {
                        "scene_cluster": scene_id,
                        "cluster_material": str(source["cluster_material"]),
                        "material_id": str(source["material_id"]),
                        "split": "all191_scale_t2_extension",
                    }
                )
            audit["observation_seal_sha256"] = sha256(OBSERVATION_SEAL)
            if audit.get("passed") is not True:
                raise RuntimeError(f"Scale snapshot audit failed: {scene_id}")
            write_snapshot_bundle(snapshot_rows, arrays, audit, output_dir=snapshot_dir)
            run([
                "scripts/extract_kinofail_realistic_features.py",
                "--snapshot-dir", str(snapshot_dir),
                "--output-dir", str(feature_dir),
                "--batch-size", "128",
            ])
            run([
                "scripts/build_kinofail_unified_invariant_features_v1.py",
                "--snapshot-dir", str(snapshot_dir),
                "--visual-feature-dir", str(feature_dir),
                "--output-dir", str(invariant_dir),
            ])
            print(json.dumps({"stage": "scale_features", "scene": scene_id, "progress": f"{position}/{len(scene_ids)}"}), flush=True)
        scene_rows, visual, full, invariant = load_scale_shard(shard)
        extension_rows.extend(scene_rows)
        extension_visual.append(visual)
        extension_full.append(full)
        extension_invariant.append(invariant)
        audit = load(snapshot_dir / "extraction_audit.json")
        temporal_exclusions.extend(audit.get("temporal_alignment_exclusions", []))
        sources[scene_id] = {
            "snapshot_records": sha256(snapshot_dir / "snapshot_records.jsonl"),
            "snapshots": sha256(snapshot_dir / "snapshots.npz"),
            "snapshot_audit": sha256(snapshot_dir / "extraction_audit.json"),
            "visual_features": sha256(feature_dir / "features.npz"),
            "invariant_features": sha256(invariant_dir / "features.npz"),
        }

    extension_ids = np.asarray([str(value["sample_id"]) for value in extension_rows])
    extension_visual_array = np.concatenate(extension_visual)
    extension_full_array = np.concatenate(extension_full)
    extension_invariant_array = np.concatenate(extension_invariant)
    if not (
        extension_visual_array.shape == (len(extension_rows), 1536)
        and extension_full_array.shape == (len(extension_rows), 190)
        and extension_invariant_array.shape == (len(extension_rows), 80)
    ):
        raise RuntimeError("extension Scale feature shape drift")

    core_rows = rows(CORE / "scale/records.jsonl")
    with np.load(CORE / "scale/features.npz", allow_pickle=False) as archive:
        core_ids = archive["sample_ids"].astype(str)
        core_visual = np.asarray(archive["visual"], dtype=np.float32)
        core_full = np.asarray(archive["full_proprio"], dtype=np.float32)
        core_invariant = np.asarray(archive["invariant_proprio"], dtype=np.float32)
    if core_ids.tolist() != [str(value["sample_id"]) for value in core_rows]:
        raise RuntimeError("core Scale feature drift")
    combined_rows = core_rows + extension_rows
    combined_ids = np.concatenate([core_ids, extension_ids])
    combined_visual = np.concatenate([core_visual, extension_visual_array])
    combined_full = np.concatenate([core_full, extension_full_array])
    combined_invariant = np.concatenate([core_invariant, extension_invariant_array])
    order = np.argsort(combined_ids)
    combined_rows = [combined_rows[int(index)] for index in order]
    combined_ids = combined_ids[order]
    combined_visual = combined_visual[order]
    combined_full = combined_full[order]
    combined_invariant = combined_invariant[order]
    physical_pairs = {str(value["counterfactual_group_id"]) for value in extension_rows}
    final_scenes = {str(value.get("scene_cluster") or value["scene_family"]) for value in combined_rows}
    attrition = (3938 - len(physical_pairs) - len(temporal_exclusions)) / 3938
    # `physical_pairs` contains snapshot-retained pairs only; temporal exclusions
    # have no rows and therefore are already absent.  Use the direct complement.
    attrition = (3938 - len(physical_pairs)) / 3938
    if attrition >= ATTRITION_LIMIT or len(final_scenes) != 191:
        raise RuntimeError("final Scale attrition or scene coverage failed")
    merged = output / "scale"
    merged.mkdir(parents=True, exist_ok=False)
    records_path = merged / "records.jsonl"
    features_path = merged / "features.npz"
    write_jsonl(records_path, combined_rows)
    np.savez_compressed(
        features_path,
        sample_ids=combined_ids,
        visual=combined_visual,
        full_proprio=combined_full,
        invariant_proprio=combined_invariant,
    )
    report = {
        "schema_version": "kinofail.kino-v4-all191-scale-features.v1",
        "passed": True,
        "counts": {
            "samples": len(combined_rows),
            "extension_snapshot_pairs": len(physical_pairs),
            "extension_temporal_exclusions": len(temporal_exclusions),
            "scenes": len(final_scenes),
        },
        "extension_attrition_rate_after_temporal_alignment": attrition,
        "source_sha256": {"core_feature_seal": sha256(CORE / "feature_seal.json"), "shards": sources},
        "output_sha256": {"records": sha256(records_path), "features": sha256(features_path)},
    }
    write_json(merged / "feature_manifest.json", report)
    return report


def prepare_t2(output: Path, seal: dict[str, Any], scene_ids: list[str]) -> tuple[Path, dict[str, Any]]:
    retained = set(seal["retained"]["t2_case_ids"])
    plan_rows = rows(T2_SCHEDULE)
    plan = {str(value["case_id"]): value for value in plan_rows}
    valid_corpus = output / "derived_design/t2_valid_corpus"
    valid_corpus.mkdir(parents=True, exist_ok=False)
    for case_id in sorted(retained):
        source = CORPUS / str(plan[case_id]["scene_cluster"]) / "c2_t2" / case_id
        os.symlink(source, valid_corpus / case_id, target_is_directory=True)
    feature_dir = output / "extension_t2/features"
    run([
        "scripts/extract_kino_v4_all191_t2_features_v1.py",
        "--corpus", str(valid_corpus),
        "--output", str(feature_dir),
        "--batch-size", "128",
    ])
    t2_rows = rows(feature_dir / "records.jsonl")
    with np.load(feature_dir / "features.npz", allow_pickle=False) as archive:
        ids = archive["sample_ids"].astype(str)
        visual = np.asarray(archive["visual"], dtype=np.float32)
    if ids.tolist() != [str(value["sample_id"]) for value in t2_rows]:
        raise RuntimeError("extension T2 feature alignment failed")
    invariant_values = []
    for value in t2_rows:
        case_id = str(value["case_id"])
        case = plan[case_id]
        source = CORPUS / str(case["scene_cluster"]) / "c2_t2" / case_id
        key = f"{case_id}__{value['target_operator']}__{value['appearance_view_id']}__proprio"
        with np.load(source / "observables.npz", allow_pickle=False) as archive:
            invariant_values.append(invariant_summary(np.asarray(archive[key], dtype=np.float32)))
        value.update(
            {
                "cell": "T2_vision_decisive",
                "cluster_material": str(case["cluster_material"]),
                "material_id": str(case["material_id"]),
                "source_case_dir": str(source),
                "split": "all191_scale_t2_extension",
            }
        )
    invariant_array = np.stack(invariant_values).astype(np.float32)
    if invariant_array.shape != (len(t2_rows), 80):
        raise RuntimeError("extension T2 invariant feature shape drift")
    base = output / "extension_t2/base_features"
    invariant = output / "extension_t2/invariant_features"
    base.mkdir(parents=True, exist_ok=False)
    invariant.mkdir(parents=True, exist_ok=False)
    write_jsonl(base / "records.jsonl", t2_rows)
    write_jsonl(invariant / "records.jsonl", t2_rows)
    np.savez_compressed(base / "features.npz", sample_ids=ids, visual=visual, proprio=np.zeros((len(ids), 190), dtype=np.float32))
    np.savez_compressed(invariant / "features.npz", sample_ids=ids, visual=visual, proprio=invariant_array)
    dino = output / "extension_t2/dinov2"
    run([
        "scripts/extract_kino_v4_confirmation_dinov2_v1.py",
        "--records", str(base / "records.jsonl"),
        "--output", str(dino),
        "--batch-size", "64",
    ])
    cases = {str(value["case_id"]) for value in t2_rows}
    scenes = {str(value["scene_cluster"]) for value in t2_rows}
    if len(t2_rows) != 6 * len(cases) or len(scenes) != 179:
        raise RuntimeError("extension T2 case or scene coverage failed")
    report = {
        "schema_version": "kinofail.kino-v4-all191-t2-features.v1",
        "passed": True,
        "counts": {"cases": len(cases), "samples": len(t2_rows), "scenes": len(scenes)},
        "attrition_rate": (716 - len(cases)) / 716,
        "output_sha256": {
            "records": sha256(base / "records.jsonl"),
            "base_features": sha256(base / "features.npz"),
            "invariant_features": sha256(invariant / "features.npz"),
            "dinov2": sha256(dino / "features.npz"),
        },
    }
    if report["attrition_rate"] >= ATTRITION_LIMIT:
        raise RuntimeError("extension T2 feature attrition failed")
    write_json(output / "extension_t2/feature_manifest.json", report)
    return base, report


def merge_conflict(output: Path, extension_base: Path, t2_report: dict[str, Any]) -> dict[str, Any]:
    core_base = CORE / "conflict/base_features"
    core_invariant = CORE / "conflict/invariant_features"
    core_dino = CORE / "conflict/dinov2"
    core_rows = rows(core_base / "records.jsonl")
    extension_rows = rows(extension_base / "records.jsonl")
    with np.load(core_base / "features.npz", allow_pickle=False) as archive:
        core_ids = archive["sample_ids"].astype(str)
        core_visual = np.asarray(archive["visual"], dtype=np.float32)
        core_full = np.asarray(archive["proprio"], dtype=np.float32)
    with np.load(CORE / "conflict/invariant_features/features.npz", allow_pickle=False) as archive:
        core_invariant_ids = archive["sample_ids"].astype(str)
        core_invariant_values = np.asarray(archive["proprio"], dtype=np.float32)
    with np.load(core_dino / "features.npz", allow_pickle=False) as archive:
        core_dino_ids = archive["sample_ids"].astype(str)
        core_dino_values = np.asarray(archive["visual_ground_mean"], dtype=np.float32)
    with np.load(extension_base / "features.npz", allow_pickle=False) as archive:
        extension_ids = archive["sample_ids"].astype(str)
        extension_visual = np.asarray(archive["visual"], dtype=np.float32)
        extension_full = np.asarray(archive["proprio"], dtype=np.float32)
    with np.load(output / "extension_t2/invariant_features/features.npz", allow_pickle=False) as archive:
        extension_invariant_ids = archive["sample_ids"].astype(str)
        extension_invariant_values = np.asarray(archive["proprio"], dtype=np.float32)
    with np.load(output / "extension_t2/dinov2/features.npz", allow_pickle=False) as archive:
        extension_dino_ids = archive["sample_ids"].astype(str)
        extension_dino_values = np.asarray(archive["visual_ground_mean"], dtype=np.float32)
    if not (
        np.array_equal(core_ids, core_invariant_ids)
        and np.array_equal(core_ids, core_dino_ids)
        and core_ids.tolist() == [str(value["sample_id"]) for value in core_rows]
        and np.array_equal(extension_ids, extension_invariant_ids)
        and np.array_equal(extension_ids, extension_dino_ids)
        and extension_ids.tolist() == [str(value["sample_id"]) for value in extension_rows]
    ):
        raise RuntimeError("Conflict feature block alignment failed")
    combined_rows = core_rows + extension_rows
    combined_ids = np.concatenate([core_ids, extension_ids])
    visual = np.concatenate([core_visual, extension_visual])
    full = np.concatenate([core_full, extension_full])
    invariant_values = np.concatenate([core_invariant_values, extension_invariant_values])
    dino_values = np.concatenate([core_dino_values, extension_dino_values])
    order = np.argsort(combined_ids)
    combined_rows = [combined_rows[int(index)] for index in order]
    combined_ids = combined_ids[order]
    visual = visual[order]
    full = full[order]
    invariant_values = invariant_values[order]
    dino_values = dino_values[order]
    base_out = output / "conflict/base_features"
    invariant_out = output / "conflict/invariant_features"
    dino_out = output / "conflict/dinov2"
    for path in (base_out, invariant_out, dino_out):
        path.mkdir(parents=True, exist_ok=False)
    write_jsonl(base_out / "records.jsonl", combined_rows)
    write_jsonl(invariant_out / "records.jsonl", combined_rows)
    np.savez_compressed(base_out / "features.npz", sample_ids=combined_ids, visual=visual, proprio=full)
    np.savez_compressed(invariant_out / "features.npz", sample_ids=combined_ids, visual=visual, proprio=invariant_values)
    np.savez_compressed(dino_out / "features.npz", sample_ids=combined_ids, visual_ground_mean=dino_values)
    cases = defaultdict(set)
    scenes = defaultdict(set)
    for value in combined_rows:
        cases[str(value["cell"])].add(str(value["case_id"]))
        scenes[str(value["cell"])].add(str(value["scene_cluster"]))
    checks = {
        "t2_scene_count_is_191": len(scenes["T2_vision_decisive"]) == 191,
        "t3_scene_count_is_191": len(scenes["T3_proprio_decisive"]) == 191,
        "extension_t2_attrition_below_five_percent": t2_report["attrition_rate"] < ATTRITION_LIMIT,
        "all_feature_blocks_aligned": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"final Conflict coverage failed: {checks}")
    report = {
        "schema_version": "kinofail.kino-v4-all191-conflict-features.v1",
        "passed": True,
        "checks": checks,
        "counts": {
            "samples": len(combined_rows),
            "T2_cases": len(cases["T2_vision_decisive"]),
            "T3_cases": len(cases["T3_proprio_decisive"]),
            "T2_scenes": len(scenes["T2_vision_decisive"]),
            "T3_scenes": len(scenes["T3_proprio_decisive"]),
        },
        "source_sha256": {"core_feature_seal": sha256(CORE / "feature_seal.json")},
        "output_sha256": {
            "records": sha256(base_out / "records.jsonl"),
            "base_features": sha256(base_out / "features.npz"),
            "invariant_features": sha256(invariant_out / "features.npz"),
            "dinov2": sha256(dino_out / "features.npz"),
        },
    }
    write_json(output / "conflict/feature_manifest.json", report)
    return report


def verify_existing_feature_reports(
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify a fully materialized run before writing only its final seal."""
    report_paths = {
        "scale": output / "scale/feature_manifest.json",
        "t2": output / "extension_t2/feature_manifest.json",
        "conflict": output / "conflict/feature_manifest.json",
    }
    if any(not path.is_file() for path in report_paths.values()):
        raise RuntimeError("existing all191 feature tree is incomplete")
    scale = load(report_paths["scale"])
    t2 = load(report_paths["t2"])
    conflict = load(report_paths["conflict"])
    if not all(value.get("passed") is True for value in (scale, t2, conflict)):
        raise RuntimeError("existing all191 feature report did not pass")

    expected_hashes = {
        output / "scale/records.jsonl": scale["output_sha256"]["records"],
        output / "scale/features.npz": scale["output_sha256"]["features"],
        output / "extension_t2/base_features/records.jsonl": (
            t2["output_sha256"]["records"]
        ),
        output / "extension_t2/base_features/features.npz": (
            t2["output_sha256"]["base_features"]
        ),
        output / "extension_t2/invariant_features/features.npz": (
            t2["output_sha256"]["invariant_features"]
        ),
        output / "extension_t2/dinov2/features.npz": (
            t2["output_sha256"]["dinov2"]
        ),
        output / "conflict/base_features/records.jsonl": (
            conflict["output_sha256"]["records"]
        ),
        output / "conflict/base_features/features.npz": (
            conflict["output_sha256"]["base_features"]
        ),
        output / "conflict/invariant_features/features.npz": (
            conflict["output_sha256"]["invariant_features"]
        ),
        output / "conflict/dinov2/features.npz": (
            conflict["output_sha256"]["dinov2"]
        ),
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"existing all191 feature artifact mismatch: {path}")
    return scale, t2, conflict


def seal_features(
    output: Path,
    scale: dict[str, Any],
    t2: dict[str, Any],
    conflict: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "observation_seal_precedes_features": True,
        "scale_has_191_scenes": scale["counts"]["scenes"] == 191,
        "t2_has_191_scenes": conflict["counts"]["T2_scenes"] == 191,
        "t3_has_191_scenes": conflict["counts"]["T3_scenes"] == 191,
        "all_attrition_gates_pass": (
            scale["extension_attrition_rate_after_temporal_alignment"]
            < ATTRITION_LIMIT
            and t2["attrition_rate"] < ATTRITION_LIMIT
        ),
        # Positive gate: False for a prohibited-state fact is success, so the
        # key consumed by all() must encode the desired state positively.
        "no_classifier_or_prediction_loaded": True,
    }
    feature_seal = {
        "schema_version": "kinofail.kino-v4-all191-feature-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_f0_checkpoint_inference",
        "passed": all(checks.values()),
        "checks": checks,
        "scale": scale,
        "t2_extension": t2,
        "conflict": conflict,
        "source_sha256": {
            "observation_seal": sha256(OBSERVATION_SEAL),
            "core_feature_seal": sha256(CORE / "feature_seal.json"),
            "script": sha256(Path(__file__).resolve()),
        },
    }
    if feature_seal["passed"] is not True:
        raise RuntimeError(f"all191 feature seal failed: {checks}")
    write_json(output / "feature_seal.json", feature_seal)
    return feature_seal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.finalize_existing:
        raise FileExistsError(output)
    if not output.exists() and args.finalize_existing:
        raise FileNotFoundError(output)
    for path in (
        OBSERVATION_SEAL,
        REGISTRY,
        SCALE_SCHEDULE,
        T2_SCHEDULE,
        REPLACEMENT_MAP,
        CORE / "feature_seal.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    seal = load(OBSERVATION_SEAL)
    if not (
        seal.get("passed") is True
        and seal.get("status") == "sealed_before_representation_or_inference"
        and seal.get("model_checkpoint_loaded") is False
        and seal.get("prediction_artifact_loaded") is False
    ):
        raise RuntimeError("invalid all191 observation seal")
    if seal.get("source_sha256", {}).get("scale_schedule") != sha256(
        SCALE_SCHEDULE
    ):
        raise RuntimeError("observation-sealed Scale schedule hash drift")
    scene_ids = [str(value["scene_id"]) for value in load(REGISTRY)["scenes"]]
    if len(scene_ids) != 179:
        raise RuntimeError("all191 extension registry drift")
    if args.finalize_existing:
        if (output / "feature_seal.json").exists():
            raise FileExistsError(output / "feature_seal.json")
        scale, t2, conflict = verify_existing_feature_reports(output)
    else:
        output.mkdir(parents=True, exist_ok=False)
        scale = prepare_scale(output, seal, scene_ids)
        t2_base, t2 = prepare_t2(output, seal, scene_ids)
        conflict = merge_conflict(output, t2_base, t2)
    feature_seal = seal_features(output, scale, t2, conflict)
    print(json.dumps({"passed": True, "output": str(output), "checks": feature_seal["checks"], "scale": scale["counts"], "conflict": conflict["counts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
