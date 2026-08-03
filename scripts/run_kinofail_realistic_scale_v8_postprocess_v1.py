#!/usr/bin/env python3
"""Finalize scale-v8 and reproduce the frozen A2/A3/A5 model experiments."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
AUDIT = ROOT / "outputs/eval/realistic_a0_a7_v6/scale_v8_postprocess.json"
COLLECTION_PREFIX_STAGES = {
    "repair_structurally_incomplete_pairs",
    "repair_complete_o3_severe_and_o8_blocks",
    "collect_a7_legacy_surrogate_18_pairs",
    "collect_a7_visual_camera_replay_9_pairs",
}
THROUGH_REVERSE_STAGES = COLLECTION_PREFIX_STAGES | {
    "freeze_a0",
    "build_snapshots",
    "extract_features",
    "build_a7_matched_snapshots",
    "extract_a7_legacy_surrogate_features",
    "extract_a7_visual_replay_features",
    "reproduce_a2_a3_a5",
    "reproduce_a3_reverse_conflict",
}


def _write_audit(stages: list[dict], status: str) -> None:
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(
        json.dumps(
            {
                "schema_version": "kinofail.scale-v8-postprocess.v1",
                "updated_utc": datetime.now(UTC).isoformat(),
                "status": status,
                "stages": stages,
                "a8_in_scope": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _run(
    stages: list[dict],
    name: str,
    args: list[str],
    *,
    accepted_returncodes: tuple[int, ...] = (0,),
) -> None:
    if (
        "--resume-after-collections" in sys.argv[1:]
        and name in COLLECTION_PREFIX_STAGES
    ):
        stages.append(
            {
                "name": name,
                "command": [],
                "returncode": 0,
                "skipped": "resume_after_verified_collections",
            }
        )
        _write_audit(stages, "running")
        return
    if (
        "--resume-after-snapshots" in sys.argv[1:]
        and name
        in COLLECTION_PREFIX_STAGES | {"freeze_a0", "build_snapshots"}
    ):
        stages.append(
            {
                "name": name,
                "command": [],
                "returncode": 0,
                "skipped": "resume_after_verified_main_snapshot_bundle",
            }
        )
        _write_audit(stages, "running")
        return
    if (
        "--resume-after-reverse" in sys.argv[1:]
        and name in THROUGH_REVERSE_STAGES
    ):
        stages.append(
            {
                "name": name,
                "command": [],
                "returncode": 0,
                "skipped": "resume_after_verified_reverse_extension",
            }
        )
        _write_audit(stages, "running")
        return
    if (
        "--skip-a5-collection" in sys.argv[1:]
        and name == "collect_a5_physical_realization_parallel"
    ):
        stages.append(
            {
                "name": name,
                "command": [],
                "returncode": 0,
                "skipped": "managed_by_external_frozen_parallel_runner",
            }
        )
        _write_audit(stages, "running")
        return
    command = [str(PYTHON), *args]
    print(json.dumps({"stage": name, "running": command}), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    stages.append({"name": name, "command": command, "returncode": completed.returncode})
    accepted = completed.returncode in accepted_returncodes
    _write_audit(stages, "running" if accepted else "blocked")
    if not accepted:
        raise RuntimeError(f"{name} failed with return code {completed.returncode}")


def main() -> int:
    stages: list[dict] = []
    _write_audit(stages, "running")
    _run(
        stages,
        "repair_structurally_incomplete_pairs",
        ["scripts/repair_kinofail_realistic_scale_v8_structural_v1.py"],
    )
    _run(
        stages,
        "repair_complete_o3_severe_and_o8_blocks",
        ["scripts/run_kinofail_realistic_temporal_repair_v4.py"],
    )
    _run(
        stages,
        "collect_a7_legacy_surrogate_18_pairs",
        [
            "scripts/run_kinofail_realistic_scale_v2.py",
            "--schedule",
            "outputs/kinofail_realistic/design_a7_legacy_surrogate_v1/schedule.jsonl",
            "--scene-registry",
            "configs/data/kinofail_realistic_scale_scene_registry_v1.json",
            "--protocol",
            "configs/data/kinofail_realistic_a7_legacy_surrogate_formal_v1.json",
            "--corpus-root",
            "outputs/kinofail_realistic/corpus_a7_legacy_surrogate_v1",
            "--collector",
            "scripts/isaac_collect_kinofail_realistic_a7_legacy_surrogate_v1.py",
            "--audit-name",
            "full_collection.json",
        ],
        accepted_returncodes=(0, 2),
    )
    _run(
        stages,
        "collect_a7_visual_camera_replay_9_pairs",
        [
            "scripts/run_kinofail_realistic_scale_v2.py",
            "--schedule",
            "outputs/kinofail_realistic/design_a7_visual_replay_v1/schedule.jsonl",
            "--scene-registry",
            "configs/data/kinofail_realistic_scale_scene_registry_v1.json",
            "--protocol",
            "configs/data/kinofail_realistic_a7_visual_replay_formal_v1.json",
            "--corpus-root",
            "outputs/kinofail_realistic/corpus_a7_visual_replay_v1",
            "--collector",
            "scripts/isaac_collect_kinofail_realistic_a7_visual_replay_v1.py",
            "--audit-name",
            "full_collection.json",
        ],
        accepted_returncodes=(0, 2),
    )
    _run(
        stages,
        "freeze_a0",
        [
            "scripts/finalize_kinofail_realistic_scale_v5.py",
            "--schedule",
            "outputs/kinofail_realistic/design_scale_v8_replication_v1/full_schedule.jsonl",
            "--static-audit",
            "outputs/kinofail_realistic/design_scale_v8_replication_v1/scale_v8_replication_audit.json",
            "--registry",
            "configs/data/kinofail_realistic_scale_scene_registry_v1.json",
            "--protocol",
            "configs/data/kinofail_realistic_scale_v8_replication_formal_v3.json",
            "--contract",
            "configs/eval/kinofail_realistic_a0_a7_v6.json",
            "--corpus-root",
            "outputs/kinofail_realistic/corpus_scale_v8_replication_v1",
            "--repair-corpus-root",
            "outputs/kinofail_realistic/corpus_scale_v8_temporal_repair_v4",
            "--repair-protocol",
            "configs/data/kinofail_realistic_temporal_repair_formal_v4.json",
            "--out",
            "outputs/kinofail_realistic/runtime_audit/formal_scale_v8_nuisance",
        ],
    )
    _run(
        stages,
        "build_snapshots",
        [
            "scripts/build_kinofail_realistic_snapshot_dev.py",
            "--protocol",
            "configs/eval/kinofail_realistic_snapshot_scale_v8_v6.json",
        ],
    )
    _run(
        stages,
        "extract_features",
        [
            "scripts/extract_kinofail_realistic_features.py",
            "--snapshot-dir",
            "outputs/eval/realistic_a0_a7_v6/snapshots",
            "--output-dir",
            "outputs/eval/realistic_a0_a7_v6/features",
        ],
    )
    _run(
        stages,
        "build_a7_matched_snapshots",
        [
            "scripts/build_kinofail_realistic_a7_matched_snapshots_v1.py",
            "--config",
            "configs/eval/kinofail_realistic_a7_matched_snapshot_v1.json",
        ],
    )
    _run(
        stages,
        "extract_a7_legacy_surrogate_features",
        [
            "scripts/extract_kinofail_realistic_features.py",
            "--snapshot-dir",
            "outputs/eval/realistic_a0_a7_v6/a7_matched/legacy_surrogate/snapshots",
            "--output-dir",
            "outputs/eval/realistic_a0_a7_v6/a7_matched/legacy_surrogate/features",
        ],
    )
    _run(
        stages,
        "extract_a7_visual_replay_features",
        [
            "scripts/extract_kinofail_realistic_features.py",
            "--snapshot-dir",
            "outputs/eval/realistic_a0_a7_v6/a7_matched/visual_replay/snapshots",
            "--output-dir",
            "outputs/eval/realistic_a0_a7_v6/a7_matched/visual_replay/features",
        ],
    )
    _run(
        stages,
        "reproduce_a2_a3_a5",
        [
            "scripts/run_kinofail_realistic_multimodal_v1.py",
            "--config",
            "configs/eval/kinofail_realistic_multimodal_scale_v8_v6.json",
            "--snapshot-dir",
            "outputs/eval/realistic_a0_a7_v6/snapshots",
            "--feature-dir",
            "outputs/eval/realistic_a0_a7_v6/features",
        ],
    )
    _run(
        stages,
        "reproduce_a3_reverse_conflict",
        [
            "scripts/analyze_kinofail_realistic_a3_reverse_v1.py",
            "--snapshot-dir",
            "outputs/eval/realistic_a0_a7_v4/a3_reverse/snapshots",
            "--feature-dir",
            "outputs/eval/realistic_a0_a7_v4/a3_reverse/features",
            "--base-report",
            "outputs/eval/realistic_a0_a7_v6/a3_publication_report.json",
            "--training-manifest",
            "outputs/eval/realistic_a0_a7_v6/training_manifest.json",
            "--checkpoint-root",
            "outputs/eval/realistic_a0_a7_v6/checkpoints",
            "--output-predictions",
            "outputs/eval/realistic_a0_a7_v6/a3_reverse_predictions.jsonl",
        ],
    )
    _run(
        stages,
        "freeze_a7_texture_ablation",
        [
            "scripts/freeze_kinofail_realistic_a7_texture_scale_v8_v1.py",
            "--evidence-root",
            "outputs/eval/realistic_a0_a7_v6",
            "--out",
            "configs/eval/kinofail_realistic_a7_texture_ablation_scale_v8_v2.json",
            "--protocol-id",
            "kinofail_realistic_a7_texture_ablation_scale_v8_nuisance_v2",
        ],
    )
    _run(
        stages,
        "reproduce_a7_texture_ablation",
        [
            "scripts/run_kinofail_realistic_a7_texture_ablation_v1.py",
            "--config",
            "configs/eval/kinofail_realistic_a7_texture_ablation_scale_v8_v2.json",
        ],
        accepted_returncodes=(0, 2),
    )
    _run(
        stages,
        "reproduce_a7_five_matched_ablations",
        [
            "scripts/analyze_kinofail_realistic_a7_matched_v1.py",
            "--config",
            "configs/eval/kinofail_realistic_a7_matched_analysis_v1.json",
        ],
        accepted_returncodes=(0, 2),
    )
    _run(
        stages,
        "collect_a5_physical_realization_parallel",
        ["scripts/run_kinofail_realistic_a5_realization_parallel_v1.py"],
        accepted_returncodes=(0, 2),
    )
    _write_audit(stages, "complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
