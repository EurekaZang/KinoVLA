#!/usr/bin/env python3
"""Publish the paper-facing Kino-Fail benchmark snapshot to Hugging Face.

The collector roots contain long simulator trajectories, launcher logs, local
paths, and superseded observations.  This publisher instead exports the exact
event-window observations used by the all-191 benchmark:

* Scale: the already frozen RGB/proprioception snapshot shards;
* T2/T3: clean per-scene snapshot bundles rebuilt from the final scored index;
* frozen benchmark features and official evaluation metadata; and
* the four formal action-study corpora.

Uploads are serial and resumable.  Generated conflict shards are staged one
scene at a time so the nearly-full data volume never needs a second full copy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "EurekaZang123/Kino-Fail"
HF = Path("/home/eureka/.local/bin/hf")
RELEASE_ROOT = Path("/data/eureka/kinofail_hf_release_v1")
SPOOL = RELEASE_ROOT / "spool"
STATE_PATH = RELEASE_ROOT / "upload_state.json"

ALL191 = Path("/data/eureka/kinovla_outputs/kino_v4_all191_v1")
CORE_FEATURES = Path("/data/eureka/kinovla_outputs/kino_v4_confirmation_v1_f5d")
CONFLICT_RECORDS = ALL191 / "conflict/base_features/records.jsonl"
BENCHMARK = ROOT / "outputs/eval/kino_v4_all191_benchmark_v1"
COMPLETION = ROOT / "outputs/eval/kino_v4_all191_completion_audit_v1"
SCENE_REGISTRY = (
    ROOT
    / "outputs/kinofail_kino_v4_all191_scale_t2_extension_v2"
    / "design/scene_registry.json"
)

ACTION_ROOTS = {
    "full_matrix": Path("/data/eureka/kinofail_action_multiscene_v1_formal_v4"),
    "o6_confirmation": Path("/data/eureka/kinofail_o6_stabilization_v3_confirmation"),
    "severity": Path("/data/eureka/kinofail_action_severity_v3"),
    "registry": Path("/data/eureka/kinofail_registry_physical_v2"),
}

TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml", ".md"}
EXCLUDED_PARTS = {
    ".cache",
    "__pycache__",
    ".pytest_cache",
    "incidents",
    "operational_quarantine",
    "launcher_logs",
    "logs",
}
EXCLUDED_SUFFIXES = {".log", ".pid", ".tmp", ".dmp", ".pyc"}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sanitized_text(value: str) -> str:
    replacements = (
        ("/data/eureka/", "${KINOFAIL_DATA_ROOT}/"),
        ("/home/eureka/KinoVLA/", "${KINOVLA_ROOT}/"),
        ("/home/eureka/", "${LOCAL_HOME}/"),
    )
    for old, new in replacements:
        value = value.replace(old, new)
    return value


def sanitized_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitized_text(value)
    if isinstance(value, list):
        return [sanitized_value(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitized_value(item) for key, item in value.items()}
    return value


def copy_clean_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in TEXT_SUFFIXES:
        target.write_text(
            sanitized_text(source.read_text(encoding="utf-8", errors="strict")),
            encoding="utf-8",
        )
    else:
        shutil.copy2(source, target)


def should_copy(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if path.is_symlink() or any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if path.name.startswith("attempt_") or path.name in {"core", "campaign_supervisor.log"}:
        return False
    return path.is_file()


def copy_clean_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    for path in source.rglob("*"):
        if should_copy(path, source):
            copy_clean_file(path, target / path.relative_to(source))


def assert_public_safe_tree(root: Path) -> None:
    """Reject links, private absolute paths, logs, and crash-like artifacts."""

    forbidden_strings = ("/home/eureka", "/data/eureka")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"release tree contains a symbolic link: {path}")
        if not path.is_file():
            continue
        if not should_copy(path, root):
            raise RuntimeError(f"release tree contains an excluded artifact: {path}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="strict")
            if any(value in text for value in forbidden_strings):
                raise RuntimeError(f"release metadata contains a local absolute path: {path}")


def load_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return {"schema_version": "kinofail.hf-upload-state.v1", "completed": []}
    return load_json(STATE_PATH)


def mark_complete(key: str, *, details: dict[str, Any] | None = None) -> None:
    state = load_state()
    completed = set(str(item) for item in state.get("completed", []))
    completed.add(key)
    state["completed"] = sorted(completed)
    if details is not None:
        state.setdefault("details", {})[key] = details
    state["last_completed"] = key
    state["updated_unix_s"] = time.time()
    write_json(STATE_PATH, state)


def is_complete(key: str) -> bool:
    return key in set(str(item) for item in load_state().get("completed", []))


def run_upload(source: Path, remote: str, key: str) -> None:
    if is_complete(key):
        print(json.dumps({"stage": "skip_uploaded", "key": key}), flush=True)
        return
    if not source.exists():
        raise FileNotFoundError(source)
    command = [
        str(HF),
        "upload",
        REPO_ID,
        str(source),
        remote,
        "--repo-type",
        "dataset",
        "--revision",
        "main",
        "--commit-message",
        f"Upload {key}",
        "--format",
        "agent",
    ]
    # This host reaches the Hub through a constrained, persistent TUN route.
    # Xet high-performance mode starts with 16 upload streams and keeps at
    # least four active; on this link that produced dozens of CAS connections
    # and repeated whole-command retries.  Keep Xet's adaptive controller, cap
    # its ceiling, and let transient requests retry in place.  These settings
    # affect only this child upload process and never alter the host's proxy,
    # route table, or remote-access services.
    upload_env = dict(os.environ)
    upload_env.pop("HF_XET_HIGH_PERFORMANCE", None)
    upload_env.pop("HF_XET_HP", None)
    upload_env.update(
        {
            "HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY": "true",
            "HF_XET_CLIENT_AC_INITIAL_UPLOAD_CONCURRENCY": "2",
            "HF_XET_CLIENT_AC_MIN_UPLOAD_CONCURRENCY": "1",
            "HF_XET_CLIENT_AC_MAX_UPLOAD_CONCURRENCY": "8",
            "HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION": "4",
            "HF_XET_CLIENT_RETRY_MAX_ATTEMPTS": "12",
            "HF_XET_CLIENT_RETRY_MAX_DURATION": "900s",
        }
    )
    delay = 30
    while True:
        print(json.dumps({"stage": "upload_start", "key": key, "remote": remote}), flush=True)
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=upload_env,
            check=False,
        )
        if result.returncode == 0:
            mark_complete(key)
            print(json.dumps({"stage": "upload_complete", "key": key}), flush=True)
            return
        print(
            json.dumps(
                {"stage": "upload_retry", "key": key, "returncode": result.returncode, "delay_s": delay}
            ),
            flush=True,
        )
        time.sleep(delay)
        delay = min(delay * 2, 900)


def dataset_card(*, complete: bool) -> str:
    status = "complete" if complete else "uploading"
    return f"""---
license: mit
task_categories:
- image-classification
- tabular-classification
tags:
- robotics
- quadruped
- multimodal
- failure-attribution
- isaac-sim
pretty_name: Kino-Fail
size_categories:
- 10K<n<100K
---

# Kino-Fail

Kino-Fail is a counterfactual benchmark for recovery-relevant failure attribution in quadrupedal navigation. The ICRA paper-facing release contains 11 physics and sensing interventions, 191 RTX/PBR scenes, 11,418 matched groups, and 22,836 accepted physical units.

Release status: **{status}**.

## Data organization

- `data/scale/<scene>/`: five-frame front-camera RGB and 21x19 proprioception windows.
- `data/conflict/<scene>/`: T2 visual-decisive and T3 proprioception-decisive snapshots.
- `features/`: the frozen all-191 feature matrices used for the reported benchmark.
- `actions/`: formal branched recovery trajectories.
- `metadata/`: official unit index, scene registry, audits, and benchmark reports.

Each appearance view belongs to its underlying physical episode and is not an independent physical sample. Evaluation should use the official physical-unit index in `metadata/benchmark/physical_units.jsonl`.

## Scope

This release contains generated sensor observations, proprioceptive trajectories, labels, and experiment metadata. It excludes launcher logs, crash material, local filesystem paths, simulator caches, source `.blend`/`.usd` assets, and superseded development corpora. Unitree and Isaac Sim assets are not redistributed.

## Code

Generation and evaluation code: https://github.com/EurekaZang/KinoVLA
"""


def prepare_metadata(*, complete: bool = False) -> Path:
    target = SPOOL / "release_metadata"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    (target / "README.md").write_text(dataset_card(complete=complete), encoding="utf-8")
    notices = """# Third-party notices

This benchmark release contains generated sensor observations and numerical trajectories only. It does not redistribute EmbodiedGen/Infinigen scene-source files, NVIDIA Isaac Sim assets, or Unitree robot assets. Terrain appearance sources used during generation include CC0 materials from ambientCG; those source textures are not included in this release.
"""
    (target / "THIRD_PARTY_NOTICES.md").write_text(notices, encoding="utf-8")

    benchmark_target = target / "metadata/benchmark"
    for path in BENCHMARK.glob("*"):
        if path.is_file():
            copy_clean_file(path, benchmark_target / path.name)
    for path in COMPLETION.glob("*"):
        if path.is_file():
            copy_clean_file(path, target / "metadata/completion" / path.name)
    copy_clean_file(SCENE_REGISTRY, target / "metadata/scene_registry.json")
    return target


def upload_metadata(*, complete: bool = False) -> None:
    key = "metadata-final" if complete else "metadata-initial"
    target = prepare_metadata(complete=complete)
    assert_public_safe_tree(target)
    run_upload(target, ".", key)
    shutil.rmtree(target)


def upload_features() -> None:
    target = SPOOL / "features"
    if target.exists():
        shutil.rmtree(target)
    for relative in (
        Path("scale/feature_manifest.json"),
        Path("scale/features.npz"),
        Path("scale/records.jsonl"),
        Path("conflict/base_features/features.npz"),
        Path("conflict/base_features/records.jsonl"),
        Path("conflict/dinov2/features.npz"),
        Path("conflict/feature_manifest.json"),
        Path("conflict/invariant_features/features.npz"),
        Path("conflict/invariant_features/records.jsonl"),
        Path("feature_seal.json"),
    ):
        copy_clean_file(ALL191 / relative, target / relative)
    assert_public_safe_tree(target)
    run_upload(target, "features", "features-all191")
    shutil.rmtree(target)


def upload_actions() -> None:
    for name, source in ACTION_ROOTS.items():
        key = f"actions-{name}"
        if is_complete(key):
            continue
        target = SPOOL / "actions" / name
        copy_clean_tree(source, target)
        assert_public_safe_tree(target)
        run_upload(target, f"actions/{name}", key)
        shutil.rmtree(target)


def scale_sources() -> list[tuple[str, Path]]:
    values: dict[str, Path] = {}
    for root in (CORE_FEATURES, ALL191):
        for path in root.glob("shards/*/scale/snapshots"):
            values[path.parents[1].name] = path
    if len(values) != 191:
        raise RuntimeError(f"expected 191 Scale scenes, found {len(values)}")
    return sorted(values.items())


def upload_scale(*, only_scene: str | None = None) -> None:
    for scene, source in scale_sources():
        if only_scene is not None and scene != only_scene:
            continue
        key = f"scale-{scene}"
        assert_public_safe_tree(source)
        run_upload(source, f"data/scale/{scene}", key)


def conflict_by_scene() -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in load_jsonl(CONFLICT_RECORDS):
        grouped[str(row["scene_cluster"])][str(row["cell"])].append(row)
    if len(grouped) != 191:
        raise RuntimeError(f"expected 191 Conflict scenes, found {len(grouped)}")
    return grouped


def clean_record(row: dict[str, Any]) -> dict[str, Any]:
    clean = sanitized_value(dict(row))
    clean.pop("source_case_dir", None)
    clean.pop("rgb_paths", None)
    return clean


def build_t2_scene(scene: str, rows: list[dict[str, Any]], target: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    for case_id, case_rows in sorted(by_case.items()):
        case_dir = Path(str(case_rows[0]["source_case_dir"]))
        manifest = load_json(case_dir / "manifest.json")
        observables = case_dir / str(manifest["artifacts"]["observables"]["path"])
        with np.load(observables, allow_pickle=False) as archive:
            for row in sorted(case_rows, key=lambda value: str(value["sample_id"])):
                sample_id = str(row["sample_id"])
                rgb_key = f"{sample_id}__rgb"
                proprio_key = f"{sample_id}__proprio"
                rgb = np.asarray(archive[rgb_key], dtype=np.uint8)
                proprio = np.asarray(archive[proprio_key], dtype=np.float32)
                if rgb.shape[0] != 5 or proprio.shape != (21, 19):
                    raise RuntimeError(f"T2 observation drift: {sample_id} {rgb.shape} {proprio.shape}")
                arrays[rgb_key] = rgb
                arrays[proprio_key] = proprio
                clean = clean_record(row)
                clean.update(
                    {
                        "rgb_key": rgb_key,
                        "proprio_key": proprio_key,
                        "rgb_shape": list(rgb.shape),
                        "proprio_shape": list(proprio.shape),
                    }
                )
                records.append(clean)
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "t2_snapshots.npz", **arrays)
    write_jsonl(target / "t2_records.jsonl", records)
    write_json(
        target / "t2_audit.json",
        {
            "passed": len(records) == len(rows) and len(records) == 6 * len(by_case),
            "scene": scene,
            "cases": len(by_case),
            "appearance_records": len(records),
            "physical_units": 2 * len(by_case),
            "rgb_frames_per_record": 5,
            "proprio_shape": [21, 19],
        },
    )


def locate_episode(scene_root: Path, operator: str, episode_id: str) -> Path:
    matches = list(scene_root.glob(f"c2_t3/*/{operator}/{episode_id}"))
    if len(matches) != 1:
        raise RuntimeError(f"episode lookup failed: {scene_root} {operator} {episode_id} {len(matches)}")
    return matches[0]


def t3_window(episode_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float]]:
    from kino_vla.eval.c2_temporal import _distance_to_region
    from kino_vla.eval.c2_temporal_v5 import (
        FOOTPRINT_MARGIN_M,
        MAX_END_SKEW_S,
        POST_ENCOUNTER_DELAY_S,
        WINDOW_SAMPLES,
    )

    manifest = load_json(episode_dir / "manifest.json")
    telemetry_path = episode_dir / str(manifest["artifacts"]["telemetry"]["path"])
    telemetry = load_jsonl(telemetry_path)
    region = manifest["geometry_readback"]["operator_region"]
    encounters = [
        float(row["timestamp_s"])
        for row in telemetry
        if _distance_to_region(
            float(row["position_xy_m"][0]), float(row["position_xy_m"][1]), region
        )
        <= FOOTPRINT_MARGIN_M
    ]
    if not encounters:
        raise RuntimeError(f"no T3 encounter: {episode_dir}")
    encounter_time = encounters[0]
    decision_time = encounter_time + POST_ENCOUNTER_DELAY_S
    artifact = manifest["artifacts"]["proprio"]
    with np.load(episode_dir / str(artifact["path"]), allow_pickle=False) as archive:
        features = np.asarray(archive[str(artifact["features_key"])], dtype=np.float32)
        timestamps = np.asarray(archive[str(artifact["timestamps_key"])], dtype=np.float64)
        names_key = str(artifact.get("feature_names_key", "feature_names"))
        names = archive[names_key].astype(str).tolist()
    candidates = np.flatnonzero(timestamps <= decision_time + MAX_END_SKEW_S)
    if len(candidates) < WINDOW_SAMPLES:
        raise RuntimeError(f"short T3 window: {episode_dir}")
    selected = candidates[-WINDOW_SAMPLES:]
    end_skew = abs(float(timestamps[selected[-1]]) - decision_time)
    if end_skew > MAX_END_SKEW_S:
        raise RuntimeError(f"T3 end skew: {episode_dir} {end_skew}")
    return (
        features[selected],
        timestamps[selected],
        names,
        {
            "encounter_time_s": encounter_time,
            "decision_time_s": decision_time,
            "end_skew_s": end_skew,
        },
    )


def rgb_sequence(scene_root: Path, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    paths = [scene_root / str(relative) for relative in row["rgb_paths"]]
    frames = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    rgb = np.stack(frames).astype(np.uint8)
    if rgb.shape[0] != 5:
        raise RuntimeError(f"T3 RGB frame drift: {row['sample_id']} {rgb.shape}")
    visual_episode = locate_episode(
        scene_root,
        "O7_visual_remap",
        str(row["visual_source_episode_id"]),
    )
    manifest = load_json(visual_episode / "manifest.json")
    entries = manifest["artifacts"]["rgb_views"][str(row["appearance_view_id"])]
    times_by_path = {str(entry["path"]): float(entry["timestamp_s"]) for entry in entries}
    relative_to_episode = [str(path.relative_to(visual_episode)) for path in paths]
    timestamps = np.asarray([times_by_path[path] for path in relative_to_episode], dtype=np.float64)
    return rgb, timestamps


def build_t3_scene(scene: str, rows: list[dict[str, Any]], target: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    rgb_cache: dict[tuple[str, str], tuple[str, str]] = {}
    proprio_cache: dict[str, tuple[str, str, dict[str, float], list[str]]] = {}
    scene_roots = {Path(str(row["source_case_dir"])).resolve() for row in rows}
    if len(scene_roots) != 1:
        raise RuntimeError(f"T3 scene source drift: {scene} {scene_roots}")
    scene_root = next(iter(scene_roots))

    for row in sorted(rows, key=lambda value: str(value["sample_id"])):
        case_id = str(row["case_id"])
        view = str(row["appearance_view_id"])
        rgb_cache_key = (case_id, view)
        if rgb_cache_key not in rgb_cache:
            rgb, rgb_times = rgb_sequence(scene_root, row)
            rgb_key = f"{case_id}__{view}__rgb"
            rgb_time_key = f"{case_id}__{view}__rgb_timestamp_s"
            arrays[rgb_key] = rgb
            arrays[rgb_time_key] = rgb_times
            rgb_cache[rgb_cache_key] = (rgb_key, rgb_time_key)

        episode_id = str(row["proprio_source_episode_id"])
        if episode_id not in proprio_cache:
            episode_dir = locate_episode(scene_root, str(row["target_operator"]), episode_id)
            window, timestamps, names, alignment = t3_window(episode_dir)
            proprio_key = f"{episode_id}__proprio"
            proprio_time_key = f"{episode_id}__proprio_timestamp_s"
            arrays[proprio_key] = window
            arrays[proprio_time_key] = timestamps
            proprio_cache[episode_id] = (proprio_key, proprio_time_key, alignment, names)

        rgb_key, rgb_time_key = rgb_cache[rgb_cache_key]
        proprio_key, proprio_time_key, alignment, names = proprio_cache[episode_id]
        clean = clean_record(row)
        clean.update(
            {
                "rgb_key": rgb_key,
                "rgb_timestamp_key": rgb_time_key,
                "proprio_key": proprio_key,
                "proprio_timestamp_key": proprio_time_key,
                "rgb_shape": list(arrays[rgb_key].shape),
                "proprio_shape": list(arrays[proprio_key].shape),
                "proprio_feature_names": names,
                **alignment,
            }
        )
        records.append(clean)

    cases = {str(row["case_id"]) for row in rows}
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "t3_snapshots.npz", **arrays)
    write_jsonl(target / "t3_records.jsonl", records)
    write_json(
        target / "t3_audit.json",
        {
            "passed": len(records) == len(rows) and len(records) == 6 * len(cases),
            "scene": scene,
            "cases": len(cases),
            "appearance_records": len(records),
            "physical_units": 2 * len(cases),
            "unique_rgb_sequences": len(rgb_cache),
            "unique_proprioception_windows": len(proprio_cache),
            "rgb_frames_per_sequence": 5,
            "proprio_shape": [21, 19],
        },
    )


def frozen_invariant_lookup() -> dict[str, np.ndarray]:
    feature_root = ALL191 / "conflict/invariant_features"
    records = load_jsonl(feature_root / "records.jsonl")
    with np.load(feature_root / "features.npz", allow_pickle=False) as archive:
        sample_ids = archive["sample_ids"].astype(str)
        proprio = np.asarray(archive["proprio"], dtype=np.float32)
    expected = [str(row["sample_id"]) for row in records]
    if sample_ids.tolist() != expected:
        raise RuntimeError("frozen Conflict record/feature alignment drift")
    return {sample_id: proprio[index] for index, sample_id in enumerate(sample_ids)}


def validate_conflict_scene(target: Path, expected: dict[str, np.ndarray]) -> None:
    from kino_vla.eval.c2_temporal_v5 import invariant_summary

    for battery in ("t2", "t3"):
        audit = load_json(target / f"{battery}_audit.json")
        if audit.get("passed") is not True:
            raise RuntimeError(f"failed {battery.upper()} release audit: {target}")
        rows = load_jsonl(target / f"{battery}_records.jsonl")
        with np.load(target / f"{battery}_snapshots.npz", allow_pickle=False) as archive:
            for row in rows:
                sample_id = str(row["sample_id"])
                raw_window = np.asarray(archive[str(row["proprio_key"])], dtype=np.float32)
                observed = invariant_summary(raw_window)
                if not np.array_equal(observed, expected[sample_id]):
                    delta = float(np.max(np.abs(observed - expected[sample_id])))
                    raise RuntimeError(
                        f"{battery.upper()} frozen-input mismatch: {sample_id} max_delta={delta}"
                    )
    assert_public_safe_tree(target)


def upload_conflict(*, only_scene: str | None = None) -> None:
    grouped = conflict_by_scene()
    expected = frozen_invariant_lookup()
    for scene in sorted(grouped):
        if only_scene is not None and scene != only_scene:
            continue
        key = f"conflict-{scene}"
        if is_complete(key):
            continue
        target = SPOOL / "conflict" / scene
        if target.exists():
            shutil.rmtree(target)
        build_t2_scene(scene, grouped[scene]["T2_vision_decisive"], target)
        build_t3_scene(scene, grouped[scene]["T3_proprio_decisive"], target)
        validate_conflict_scene(target, expected)
        run_upload(target, f"data/conflict/{scene}", key)
        shutil.rmtree(target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("all", "metadata", "features", "actions", "scale", "conflict"),
        default="all",
    )
    parser.add_argument("--scene", help="restrict Scale/Conflict to one scene")
    args = parser.parse_args()
    RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
    SPOOL.mkdir(parents=True, exist_ok=True)

    if args.stage in {"all", "metadata"}:
        upload_metadata(complete=False)
    if args.stage in {"all", "features"}:
        upload_features()
    if args.stage in {"all", "actions"}:
        upload_actions()
    if args.stage in {"all", "conflict"}:
        upload_conflict(only_scene=args.scene)
    if args.stage in {"all", "scale"}:
        upload_scale(only_scene=args.scene)
    if args.stage == "all":
        upload_metadata(complete=True)
        mark_complete("release-v1-complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
