#!/usr/bin/env python3
"""Publish the complete, formal all-191 Kino-Fail raw observations.

The paper-facing benchmark publisher exports the exact event windows consumed
by the reported evaluation.  This companion publisher preserves the complete
accepted Scale, T2, and T3 observation trees so future users can choose other
time windows or re-run feature extraction.

The source contains millions of small files and cannot be uploaded as a flat
Hub folder reliably.  Files are therefore streamed into deterministic,
scene-and-battery-local ``tar.zst`` parts.  Text metadata is sanitized while it
is streamed; operational logs, crash material, simulator source assets, and
superseded roots are outside the release scope.  Only one part is staged at a
time, uploaded, recorded, and removed, which keeps the nearly-full data volume
from needing a second local copy.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "EurekaZang123/Kino-Fail"
HF = Path("/home/eureka/.local/bin/hf")

BENCHMARK_RELEASE_ROOT = Path("/data/eureka/kinofail_hf_release_v1")
BENCHMARK_STATE = BENCHMARK_RELEASE_ROOT / "upload_state.json"
RAW_RELEASE_ROOT = Path("/data/eureka/kinofail_hf_raw_v1")
RAW_SPOOL = RAW_RELEASE_ROOT / "spool"
RAW_STATE = RAW_RELEASE_ROOT / "upload_state.json"

TARGET_PART_BYTES = 12 * 1024**3
MIN_FREE_AFTER_STAGE_BYTES = 20 * 1024**3

TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml", ".md", ".usda"}
EXCLUDED_DIR_NAMES = {
    ".cache",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "incidents",
    "launcher_audits",
    "launcher_logs",
    "logs",
    "operational_quarantine",
}
EXCLUDED_SUFFIXES = {
    ".blend",
    ".dmp",
    ".hdr",
    ".log",
    ".pid",
    ".pyc",
    ".tmp",
    ".usd",
    ".usdc",
}


@dataclass(frozen=True)
class SourceSpec:
    cohort: str
    corpus: Path
    source_battery: str
    public_battery: str


@dataclass(frozen=True)
class RawUnit:
    cohort: str
    scene: str
    battery: str
    source: Path


@dataclass(frozen=True)
class PlannedFile:
    path: Path
    relative: Path
    size: int


SOURCES = (
    SourceSpec(
        "core",
        Path("/data/eureka/kinofail_kino_v4_confirmation_v1/corpus"),
        "scale",
        "scale",
    ),
    SourceSpec(
        "core",
        Path("/data/eureka/kinofail_kino_v4_confirmation_v1/corpus"),
        "c2_t2",
        "t2",
    ),
    SourceSpec(
        "core",
        Path("/data/eureka/kinofail_kino_v4_confirmation_t3_runin_f4e/corpus"),
        "c2_t3",
        "t3",
    ),
    SourceSpec(
        "all191_extension",
        Path("/data/eureka/kinofail_kino_v4_all191_scale_t2_extension_v2/corpus"),
        "scale",
        "scale",
    ),
    SourceSpec(
        "all191_extension",
        Path("/data/eureka/kinofail_kino_v4_all191_scale_t2_extension_v2/corpus"),
        "c2_t2",
        "t2",
    ),
    SourceSpec(
        "all191_extension",
        Path("/data/eureka/kinofail_kino_v4_confirmation_t3_extension_f4h/corpus"),
        "c2_t3",
        "t3",
    ),
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_state() -> dict[str, Any]:
    if not RAW_STATE.is_file():
        return {
            "schema_version": "kinofail.hf-raw-upload-state.v1",
            "completed": [],
            "details": {},
        }
    return load_json(RAW_STATE)


def is_complete(key: str) -> bool:
    return key in set(str(value) for value in load_state().get("completed", []))


def mark_complete(key: str, details: dict[str, Any]) -> None:
    state = load_state()
    completed = set(str(value) for value in state.get("completed", []))
    completed.add(key)
    state["completed"] = sorted(completed)
    state.setdefault("details", {})[key] = details
    state["last_completed"] = key
    state["updated_unix_s"] = time.time()
    write_json_atomic(RAW_STATE, state)


def benchmark_is_complete() -> bool:
    if not BENCHMARK_STATE.is_file():
        return False
    state = load_json(BENCHMARK_STATE)
    return "release-v1-complete" in set(str(value) for value in state.get("completed", []))


def wait_for_benchmark() -> None:
    announced = False
    while not benchmark_is_complete():
        if not announced:
            print(
                json.dumps(
                    {
                        "stage": "waiting_for_benchmark_release",
                        "state": str(BENCHMARK_STATE),
                    }
                ),
                flush=True,
            )
            announced = True
        time.sleep(60)
    print(json.dumps({"stage": "benchmark_release_complete"}), flush=True)


def raw_units() -> list[RawUnit]:
    values: dict[tuple[str, str], RawUnit] = {}
    for spec in SOURCES:
        if not spec.corpus.is_dir():
            raise FileNotFoundError(spec.corpus)
        for scene_root in sorted(path for path in spec.corpus.iterdir() if path.is_dir()):
            source = scene_root / spec.source_battery
            if not source.is_dir():
                continue
            key = (spec.public_battery, scene_root.name)
            if key in values:
                raise RuntimeError(f"duplicate formal raw source: {key}")
            values[key] = RawUnit(
                cohort=spec.cohort,
                scene=scene_root.name,
                battery=spec.public_battery,
                source=source,
            )
    counts = Counter(unit.battery for unit in values.values())
    if counts != Counter({"scale": 191, "t2": 191, "t3": 191}):
        raise RuntimeError(f"formal raw scope drift: {dict(counts)}")
    return [values[key] for key in sorted(values)]


def exclusion_reason(path: Path) -> str | None:
    name = path.name
    if name.startswith("attempt_"):
        return "attempt_artifact"
    if name in {"core", "campaign_supervisor.log"}:
        return "crash_or_supervisor_artifact"
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return f"excluded_suffix:{path.suffix.lower()}"
    return None


def plan_unit(unit: RawUnit) -> tuple[list[PlannedFile], Counter[str]]:
    planned: list[PlannedFile] = []
    excluded: Counter[str] = Counter()
    for current, dir_names, file_names in os.walk(unit.source, followlinks=False):
        current_path = Path(current)
        kept_dirs: list[str] = []
        for name in sorted(dir_names):
            path = current_path / name
            if path.is_symlink():
                raise RuntimeError(f"formal raw tree contains directory symlink: {path}")
            if name in EXCLUDED_DIR_NAMES:
                excluded[f"excluded_directory:{name}"] += 1
            else:
                kept_dirs.append(name)
        dir_names[:] = kept_dirs
        for name in sorted(file_names):
            path = current_path / name
            if path.is_symlink():
                raise RuntimeError(f"formal raw tree contains file symlink: {path}")
            if not path.is_file():
                raise RuntimeError(f"formal raw tree contains non-file: {path}")
            reason = exclusion_reason(path)
            if reason is not None:
                excluded[reason] += 1
                continue
            planned.append(
                PlannedFile(
                    path=path,
                    relative=path.relative_to(unit.source),
                    size=path.stat().st_size,
                )
            )
    planned.sort(key=lambda value: value.relative.as_posix())
    if not planned:
        raise RuntimeError(f"empty formal raw unit: {unit}")
    return planned, excluded


def split_parts(files: list[PlannedFile]) -> list[list[PlannedFile]]:
    parts: list[list[PlannedFile]] = []
    current: list[PlannedFile] = []
    current_bytes = 0
    for value in files:
        if current and current_bytes + value.size > TARGET_PART_BYTES:
            parts.append(current)
            current = []
            current_bytes = 0
        current.append(value)
        current_bytes += value.size
    if current:
        parts.append(current)
    return parts


def sanitized_text(value: str) -> str:
    replacements = (
        ("/data/eureka/", "${KINOFAIL_DATA_ROOT}/"),
        ("/home/eureka/KinoVLA/", "${KINOVLA_ROOT}/"),
        ("/home/eureka/", "${LOCAL_HOME}/"),
    )
    for old, new in replacements:
        value = value.replace(old, new)
    return value


def normalized_tarinfo(archive: tarfile.TarFile, path: Path, arcname: str) -> tarfile.TarInfo:
    info = archive.gettarinfo(str(path), arcname=arcname)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def add_file(archive: tarfile.TarFile, unit: RawUnit, value: PlannedFile) -> None:
    arcname = Path("Kino-Fail-raw-v1") / unit.battery / unit.scene / value.relative
    info = normalized_tarinfo(archive, value.path, arcname.as_posix())
    if value.path.suffix.lower() in TEXT_SUFFIXES:
        raw = value.path.read_bytes()
        text = raw.decode("utf-8", errors="surrogateescape")
        public = sanitized_text(text).encode("utf-8", errors="surrogateescape")
        if b"/home/eureka" in public or b"/data/eureka" in public:
            raise RuntimeError(f"unsanitized local path remains in {value.path}")
        info.size = len(public)
        archive.addfile(info, io.BytesIO(public))
    else:
        with value.path.open("rb") as handle:
            archive.addfile(info, handle)


def ensure_stage_capacity(max_archive_bytes: int) -> None:
    required = max_archive_bytes + MIN_FREE_AFTER_STAGE_BYTES
    while shutil.disk_usage(RAW_SPOOL).free < required:
        print(
            json.dumps(
                {
                    "stage": "waiting_for_stage_space",
                    "required_bytes": required,
                    "free_bytes": shutil.disk_usage(RAW_SPOOL).free,
                }
            ),
            flush=True,
        )
        time.sleep(300)


def build_archive(unit: RawUnit, files: list[PlannedFile], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.partial")
    if partial.exists():
        partial.unlink()
    ensure_stage_capacity(sum(value.size for value in files))
    print(
        json.dumps(
            {
                "stage": "raw_archive_start",
                "scene": unit.scene,
                "battery": unit.battery,
                "files": len(files),
                "source_bytes": sum(value.size for value in files),
            }
        ),
        flush=True,
    )
    with partial.open("wb") as output:
        process = subprocess.Popen(
            ["zstd", "-T4", "-1", "--no-progress", "-c"],
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None or process.stderr is None:
            raise RuntimeError("failed to open zstd streams")
        try:
            with tarfile.open(fileobj=process.stdin, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for value in files:
                    add_file(archive, unit, value)
            process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            returncode = process.wait()
            if returncode != 0:
                raise RuntimeError(f"zstd failed ({returncode}): {stderr[-2000:]}")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            if partial.exists():
                partial.unlink()
            raise
    os.replace(partial, target)
    print(
        json.dumps(
            {
                "stage": "raw_archive_complete",
                "scene": unit.scene,
                "battery": unit.battery,
                "archive_bytes": target.stat().st_size,
            }
        ),
        flush=True,
    )


def upload_environment() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("HF_XET_HIGH_PERFORMANCE", None)
    env.pop("HF_XET_HP", None)
    env.update(
        {
            "HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY": "true",
            "HF_XET_CLIENT_AC_INITIAL_UPLOAD_CONCURRENCY": "2",
            "HF_XET_CLIENT_AC_MIN_UPLOAD_CONCURRENCY": "1",
            "HF_XET_CLIENT_AC_MAX_UPLOAD_CONCURRENCY": "8",
            "HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION": "2",
            "HF_XET_CLIENT_RETRY_MAX_ATTEMPTS": "12",
            "HF_XET_CLIENT_RETRY_MAX_DURATION": "900s",
        }
    )
    return env


def run_upload(source: Path, remote: str, key: str) -> None:
    delay = 30
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
    while True:
        print(json.dumps({"stage": "upload_start", "key": key, "remote": remote}), flush=True)
        result = subprocess.run(command, cwd=ROOT, env=upload_environment(), check=False)
        if result.returncode == 0:
            print(json.dumps({"stage": "upload_complete", "key": key}), flush=True)
            return
        print(
            json.dumps(
                {
                    "stage": "upload_retry",
                    "key": key,
                    "returncode": result.returncode,
                    "delay_s": delay,
                }
            ),
            flush=True,
        )
        time.sleep(delay)
        delay = min(delay * 2, 900)


def raw_readme() -> str:
    return """# Kino-Fail complete raw observations

This directory contains the complete accepted Scale, T2, and T3 observation
trees for all 191 formal scenes. Each archive is local to one battery and one
scene; large units are split into numbered parts. Extract all parts under the
same destination to reconstruct the hierarchy rooted at `Kino-Fail-raw-v1/`.

The archives retain full RGB streams, proprioception, telemetry, manifests,
and accepted simulator readbacks. Public metadata replaces machine-local path
prefixes with portable placeholders. Operational logs, crash artifacts,
simulator caches, source `.blend`/`.usd` assets, and superseded collections are
not observations and are not distributed.

The compact tensors used directly by the paper tables remain under `data/`;
the raw tier supports alternative temporal windows, feature extractors, and
event-boundary studies.
"""


def upload_raw_readme() -> None:
    key = "raw-v1-readme"
    if is_complete(key):
        return
    target = RAW_SPOOL / "RAW_README.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(raw_readme(), encoding="utf-8")
    run_upload(target, "raw/v1/README.md", key)
    mark_complete(key, {"remote": "raw/v1/README.md", "bytes": target.stat().st_size})
    target.unlink()


def upload_final_manifest() -> None:
    key = "raw-v1-manifest"
    if is_complete(key):
        return
    state = load_state()
    rows = [
        value
        for part_key, value in sorted((state.get("details") or {}).items())
        if str(part_key).startswith("raw-") and isinstance(value, dict) and "scene" in value
    ]
    target = RAW_SPOOL / "raw_manifest.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    run_upload(target, "raw/v1/manifest.jsonl", key)
    mark_complete(
        key,
        {
            "remote": "raw/v1/manifest.jsonl",
            "archives": len(rows),
            "bytes": target.stat().st_size,
        },
    )
    target.unlink()


def publish_raw(*, only_scene: str | None = None, only_battery: str | None = None) -> None:
    upload_raw_readme()
    for unit in raw_units():
        if only_scene is not None and unit.scene != only_scene:
            continue
        if only_battery is not None and unit.battery != only_battery:
            continue
        files, excluded = plan_unit(unit)
        parts = split_parts(files)
        for index, part in enumerate(parts):
            part_name = f"part-{index:05d}-of-{len(parts):05d}.tar.zst"
            key = f"raw-{unit.battery}-{unit.scene}-{index:05d}-of-{len(parts):05d}"
            if is_complete(key):
                continue
            target = RAW_SPOOL / unit.battery / unit.scene / part_name
            if not target.is_file():
                build_archive(unit, part, target)
            remote = f"raw/v1/{unit.battery}/{unit.scene}/{part_name}"
            run_upload(target, remote, key)
            details = {
                "archive_bytes": target.stat().st_size,
                "battery": unit.battery,
                "cohort": unit.cohort,
                "excluded_entries_in_unit": dict(sorted(excluded.items())),
                "files": len(part),
                "part_index": index,
                "parts_in_unit": len(parts),
                "remote": remote,
                "scene": unit.scene,
                "source_bytes": sum(value.size for value in part),
            }
            mark_complete(key, details)
            target.unlink()
            try:
                target.parent.rmdir()
            except OSError:
                pass
    if only_scene is None and only_battery is None:
        upload_final_manifest()
        mark_complete(
            "raw-v1-complete",
            {
                "formal_scenes": 191,
                "batteries": ["scale", "t2", "t3"],
                "completed_unix_s": time.time(),
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-benchmark", action="store_true")
    parser.add_argument("--scene")
    parser.add_argument("--battery", choices=("scale", "t2", "t3"))
    parser.add_argument("--scope-only", action="store_true")
    args = parser.parse_args()

    RAW_SPOOL.mkdir(parents=True, exist_ok=True)
    units = raw_units()
    if args.scope_only:
        counts = Counter(unit.battery for unit in units)
        print(json.dumps({"units": len(units), "counts": dict(counts)}, sort_keys=True))
        return 0
    if args.wait_for_benchmark:
        wait_for_benchmark()
    publish_raw(only_scene=args.scene, only_battery=args.battery)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
