#!/usr/bin/env python3
"""Wait for frozen conflict acquisition and build all model-blind features.

No checkpoint or prediction code is imported.  The driver executes only the
F0-frozen snapshot/feature/complete-case tools, then performs the append-only
F3 launcher-audit namespace handoff required before Scale acquisition.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
REGISTRY = ROOT / "outputs/kinofail_confirmatory_v1/scene_registry.json"
SCHEDULE_ROOT = (
    ROOT
    / "outputs/kinofail_confirmatory_v1/"
    "schedules_f2_scene_source_amendment"
)
T2_CORPUS = ROOT / "outputs/kinofail_confirmatory_v1/c2_t2"
T3_CORPUS = ROOT / "outputs/kinofail_confirmatory_v1/corpus"
STORAGE_RECEIPTS = (
    ROOT / "outputs/kinofail_confirmatory_v1/storage_receipts"
)
EVAL_ROOT = ROOT / "outputs/eval/unified_moe_v3_confirmatory_v1"
CONFLICT_ROOT = EVAL_ROOT / "conflict"
LOG_ROOT = CONFLICT_ROOT / "logs"


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


def _scenes() -> list[str]:
    scenes = [str(row["scene_id"]) for row in _json(REGISTRY)["scenes"]]
    if len(scenes) != 30 or len(set(scenes)) != 30:
        raise RuntimeError("confirmatory registry must contain 30 scenes")
    return scenes


def _t2_complete(scenes: list[str]) -> bool:
    for partition in (0, 1):
        path = (
            T2_CORPUS
            / "launcher_audits"
            / f"t2_partition_{partition}_of_2.json"
        )
        if not path.is_file():
            return False
        audit = _json(path)
        if (
            len(audit.get("selected_scenes", [])) != 15
            or len(audit.get("attempts", [])) != 15
        ):
            return False
    for scene_id in scenes:
        path = (
            T2_CORPUS
            / scene_id
            / "c2_t2"
            / "scene_summaries"
            / f"{scene_id}.json"
        )
        if not path.is_file() or _json(path).get("passed") is not True:
            return False
    return True


def _t3_complete(scenes: list[str]) -> bool:
    return all(
        (STORAGE_RECEIPTS / f"{scene_id}.json").is_file()
        and _json(STORAGE_RECEIPTS / f"{scene_id}.json").get("passed") is True
        for scene_id in scenes
    )


def _active_isaac_collectors() -> list[str]:
    active = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "isaac_collect_kinofail_confirmatory_" in command:
            active.append(command)
    return active


def _wait_for_acquisition(scenes: list[str], poll_seconds: int) -> None:
    last_report = 0.0
    while True:
        t2 = _t2_complete(scenes)
        t3 = _t3_complete(scenes)
        active = _active_isaac_collectors()
        if t2 and t3 and not active:
            return
        now = time.monotonic()
        if now - last_report >= 600:
            print(
                json.dumps(
                    {
                        "stage": "waiting_for_model_blind_conflict_acquisition",
                        "t2_complete": t2,
                        "t3_complete": t3,
                        "active_isaac_collectors": len(active),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_report = now
        time.sleep(poll_seconds)


def _run(command: list[str], log_name: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{log_name}.log"
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    with log_path.open("wb") as stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(
            f"stage failed ({completed.returncode}): {log_name}\n{tail}"
        )


def _parallel(
    commands: Iterable[tuple[list[str], str]], *, workers: int
) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_run, command, log_name)
            for command, log_name in commands
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def _require_manifest(path: Path) -> dict[str, Any]:
    manifest = _json(path)
    if (
        manifest.get("passed") is not True
        and manifest.get("status") != "complete"
    ):
        raise RuntimeError(f"derived manifest did not pass: {path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--feature-workers", type=int, default=4)
    args = parser.parse_args()
    if args.poll_seconds < 5 or not 1 <= args.feature_workers <= 4:
        raise ValueError("invalid wait/worker configuration")
    completion_path = CONFLICT_ROOT / "postprocess_audit.json"
    if completion_path.exists():
        raise FileExistsError(completion_path)
    predictions = list(EVAL_ROOT.glob("**/*prediction*"))
    if predictions:
        raise RuntimeError(
            "confirmatory predictions exist before conflict postprocessing"
        )

    scenes = _scenes()
    _wait_for_acquisition(scenes, args.poll_seconds)
    CONFLICT_ROOT.mkdir(parents=True, exist_ok=True)

    t2_scene_root = CONFLICT_ROOT / "t2_scene_features"
    t2_commands = []
    for scene_id in scenes:
        output = t2_scene_root / scene_id
        if output.exists():
            raise FileExistsError(output)
        t2_commands.append(
            (
                [
                    str(PYTHON),
                    "scripts/extract_kinofail_confirmatory_t2_features_v1.py",
                    "--corpus",
                    str(T2_CORPUS / scene_id / "c2_t2"),
                    "--output",
                    str(output),
                    "--scene-id",
                    scene_id,
                    "--batch-size",
                    "64",
                ],
                f"t2_features_{scene_id}",
            )
        )
    _parallel(t2_commands, workers=args.feature_workers)
    for scene_id in scenes:
        _require_manifest(
            t2_scene_root / scene_id / "feature_manifest.json"
        )

    merged_t2 = CONFLICT_ROOT / "t2_merged"
    merge_command = [
        str(PYTHON),
        "scripts/merge_kinofail_confirmatory_t2_features_v1.py",
    ]
    for scene_id in scenes:
        merge_command.extend(
            [
                "--scene-feature-dir",
                str(t2_scene_root / scene_id),
                "--scene-corpus-dir",
                str(T2_CORPUS / scene_id / "c2_t2"),
            ]
        )
    merge_command.extend(
        [
            "--t2-schedule",
            str(SCHEDULE_ROOT / "c2_t2/schedule.jsonl"),
            "--output",
            str(merged_t2),
        ]
    )
    _run(merge_command, "t2_merge")
    merged_manifest = _require_manifest(
        merged_t2 / "feature_manifest.json"
    )

    snapshot_commands = []
    for scene_id in scenes:
        snapshot_dir = (
            EVAL_ROOT / "shards" / scene_id / "c2_t3" / "snapshots"
        )
        if snapshot_dir.exists():
            raise FileExistsError(snapshot_dir)
        snapshot_commands.append(
            (
                [
                    str(PYTHON),
                    "scripts/build_kinofail_realistic_snapshot_dev.py",
                    "--protocol",
                    str(
                        SCHEDULE_ROOT
                        / "scenes"
                        / scene_id
                        / "c2_t3"
                        / "snapshot_protocol.json"
                    ),
                    "--repo-root",
                    str(ROOT),
                ],
                f"t3_snapshots_{scene_id}",
            )
        )
    _parallel(snapshot_commands, workers=args.feature_workers)

    feature_commands = []
    for scene_id in scenes:
        shard = EVAL_ROOT / "shards" / scene_id / "c2_t3"
        feature_commands.append(
            (
                [
                    str(PYTHON),
                    "scripts/extract_kinofail_realistic_features.py",
                    "--snapshot-dir",
                    str(shard / "snapshots"),
                    "--output-dir",
                    str(shard / "features"),
                    "--batch-size",
                    "64",
                ],
                f"t3_visual_features_{scene_id}",
            )
        )
    _parallel(feature_commands, workers=args.feature_workers)

    unified_commands = []
    for scene_id in scenes:
        shard = EVAL_ROOT / "shards" / scene_id / "c2_t3"
        unified_commands.append(
            (
                [
                    str(PYTHON),
                    "scripts/build_kinofail_unified_invariant_features_v1.py",
                    "--snapshot-dir",
                    str(shard / "snapshots"),
                    "--visual-feature-dir",
                    str(shard / "features"),
                    "--output-dir",
                    str(shard / "unified_features"),
                ],
                f"t3_unified_features_{scene_id}",
            )
        )
    _parallel(unified_commands, workers=args.feature_workers)

    valid_design = CONFLICT_ROOT / "valid_design"
    _run(
        [
            str(PYTHON),
            "scripts/prepare_kinofail_confirmatory_valid_conflict_design_v1.py",
            "--t2-schedule",
            str(SCHEDULE_ROOT / "c2_t2/schedule.jsonl"),
            "--t2-features",
            str(merged_t2),
            "--t3-design",
            str(SCHEDULE_ROOT / "c2_t3"),
            "--t3-corpus",
            str(T3_CORPUS),
            "--output",
            str(valid_design),
        ],
        "valid_conflict_design",
    )
    valid_audit = _require_manifest(valid_design / "audit.json")

    base_features = CONFLICT_ROOT / "c2_base"
    _run(
        [
            str(PYTHON),
            "scripts/build_kinofail_confirmatory_c2_base_features_v1.py",
            "--c1-features",
            str(merged_t2),
            "--c1-corpus",
            str(T2_CORPUS),
            "--t3-design",
            str(valid_design),
            "--t3-corpus",
            str(T3_CORPUS),
            "--output",
            str(base_features),
            "--batch-size",
            "64",
        ],
        "conflict_c2_base_features",
    )
    _require_manifest(base_features / "feature_manifest.json")
    _require_manifest(base_features / "geometry_manifest.json")

    final_features = CONFLICT_ROOT / "features"
    v5_command = [
        str(PYTHON),
        "scripts/build_kinofail_realistic_c2_v5_features.py",
        "--base-features",
        str(base_features),
    ]
    for scene_id in scenes:
        v5_command.extend(
            [
                "--t2-corpus",
                str(T2_CORPUS / scene_id / "c2_t2"),
            ]
        )
    v5_command.extend(
        [
            "--t3-corpus",
            str(T3_CORPUS),
            "--output",
            str(final_features),
        ]
    )
    _run(v5_command, "conflict_c2_v5_features")
    final_manifest = _require_manifest(
        final_features / "feature_manifest.json"
    )

    f3_path = (
        ROOT
        / "outputs/freeze/"
        "unified_moe_v3_confirmatory_f3_runtime_namespace/"
        "amendment_manifest.json"
    )
    _run(
        [
            str(PYTHON),
            "scripts/archive_kinofail_confirmatory_t3_launcher_state_f3.py",
        ],
        "f3_archive_t3_launcher_state",
    )
    f3 = _json(f3_path)
    if f3.get("status") != "sealed":
        raise RuntimeError("F3 runtime namespace amendment did not seal")

    audit = {
        "schema_version":
        "kinofail.confirmatory-conflict-postprocess.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "model_or_checkpoint_loaded": False,
        "predictions_generated": False,
        "scientific_design_changed": False,
        "scene_registry_sha256": _sha256(REGISTRY),
        "schedule_manifest_sha256": _sha256(
            SCHEDULE_ROOT / "manifest.json"
        ),
        "t2_merged_manifest_sha256": _sha256(
            merged_t2 / "feature_manifest.json"
        ),
        "valid_design_audit_sha256": _sha256(
            valid_design / "audit.json"
        ),
        "base_feature_manifest_sha256": _sha256(
            base_features / "feature_manifest.json"
        ),
        "base_geometry_manifest_sha256": _sha256(
            base_features / "geometry_manifest.json"
        ),
        "final_feature_manifest_sha256": _sha256(
            final_features / "feature_manifest.json"
        ),
        "f3_manifest_sha256": _sha256(f3_path),
        "counts": {
            "t2": merged_manifest.get("counts"),
            "valid_conflict": valid_audit.get("counts"),
            "final_features": final_manifest.get("counts"),
        },
    }
    completion_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
