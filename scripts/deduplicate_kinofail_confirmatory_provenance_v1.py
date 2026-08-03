#!/usr/bin/env python3
"""Hard-link duplicate immutable provenance assets after a scene is complete.

Only copied provenance assets are deduplicated.  RGB observations,
proprioception, manifests, schedules, summaries, and audits are untouched.
Every replacement is atomic and byte-checked before and after linking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "outputs/kinofail_confirmatory_v1/scene_registry.json"
DEFAULT_CORPUS = ROOT / "outputs/kinofail_confirmatory_v1/corpus"
DEFAULT_RECEIPTS = (
    ROOT / "outputs/kinofail_confirmatory_v1/storage_receipts"
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


def _scene_complete(scene: Path) -> bool:
    for partition in (0, 1):
        audit_path = (
            scene
            / "launcher_audits"
            / f"partition_{partition}_of_2.json"
        )
        if not audit_path.is_file():
            return False
        audit = _json(audit_path)
        selected = [str(value) for value in audit.get("selected_pair_ids", [])]
        attempts = [
            str(row["counterfactual_group_id"])
            for row in audit.get("attempts", [])
        ]
        if (
            len(selected) != 50
            or len(attempts) != 50
            or set(attempts) != set(selected)
        ):
            return False
    return True


def _scene_active(scene_id: str) -> bool:
    marker = f"/{scene_id}/"
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
        if marker in command and (
            "isaac_collect_kinofail_confirmatory_pair_v1.py" in command
            or "run_kinofail_confirmatory_scale_shard_v1.py" in command
        ):
            return True
    return False


def _is_provenance_asset(path: Path) -> bool:
    parts = path.parts
    try:
        index = parts.index("provenance")
    except ValueError:
        return False
    return (
        index + 1 < len(parts)
        and parts[index + 1] in {"appearance", "scene_source"}
        and path.is_file()
        and not path.is_symlink()
    )


def _deduplicate_scene(
    *,
    scene_id: str,
    corpus_root: Path,
    receipt_root: Path,
) -> dict[str, Any] | None:
    logical_scene = corpus_root / scene_id
    receipt_path = receipt_root / f"{scene_id}.json"
    if receipt_path.is_file():
        return None
    if not _scene_complete(logical_scene) or _scene_active(scene_id):
        return None

    physical_scene = logical_scene.resolve()
    candidates = sorted(
        path for path in physical_scene.rglob("*") if _is_provenance_asset(path)
    )
    by_size: dict[int, list[Path]] = defaultdict(list)
    for path in candidates:
        by_size[path.stat().st_size].append(path)
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for paths in by_size.values():
        if len(paths) < 2:
            continue
        for path in paths:
            by_hash[_sha256(path)].append(path)

    replacement_count = 0
    logical_bytes_saved = 0
    duplicate_groups = 0
    for digest, paths in sorted(by_hash.items()):
        if len(paths) < 2:
            continue
        duplicate_groups += 1
        canonical = paths[0]
        canonical_stat = canonical.stat()
        for duplicate in paths[1:]:
            stat = duplicate.stat()
            if (
                stat.st_dev == canonical_stat.st_dev
                and stat.st_ino == canonical_stat.st_ino
            ):
                continue
            if stat.st_dev != canonical_stat.st_dev:
                raise RuntimeError("cross-device provenance group")
            if stat.st_size != canonical_stat.st_size:
                raise RuntimeError("same-hash provenance size mismatch")
            temporary = duplicate.with_name(
                f".{duplicate.name}.kinofail-link-{os.getpid()}"
            )
            if temporary.exists():
                raise FileExistsError(temporary)
            os.link(canonical, temporary)
            os.replace(temporary, duplicate)
            if _sha256(duplicate) != digest:
                raise RuntimeError("post-link provenance hash mismatch")
            replacement_count += 1
            logical_bytes_saved += stat.st_size

    aggregate = hashlib.sha256()
    for path in candidates:
        relative = str(path.relative_to(physical_scene))
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(_sha256(path).encode("ascii"))
        aggregate.update(b"\n")
    receipt = {
        "schema_version": "kinofail.confirmatory-provenance-dedup.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "scene_id": scene_id,
        "logical_scene": str(logical_scene),
        "physical_scene": str(physical_scene),
        "scene_complete_before_operation": True,
        "scene_inactive_before_operation": True,
        "scope": ["provenance/appearance", "provenance/scene_source"],
        "untouched": [
            "RGB observations",
            "proprioception",
            "manifests",
            "schedules",
            "pair summaries",
            "launcher audits",
        ],
        "candidate_files": len(candidates),
        "duplicate_content_groups": duplicate_groups,
        "hardlink_replacements": replacement_count,
        "logical_duplicate_bytes_eliminated": logical_bytes_saved,
        "post_operation_path_content_aggregate_sha256": aggregate.hexdigest(),
        "scientific_content_changed": False,
    }
    receipt_root.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--receipt-root", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--scene-id")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    registry = _json(args.registry.resolve())
    scenes = [str(row["scene_id"]) for row in registry["scenes"]]
    if len(scenes) != 30 or len(set(scenes)) != 30:
        raise RuntimeError("confirmatory registry must contain 30 scenes")
    if args.scene_id is not None:
        if args.scene_id not in scenes:
            raise ValueError("scene is not in the frozen registry")
        scenes = [args.scene_id]
    corpus_root = args.corpus_root.absolute()
    receipt_root = args.receipt_root.absolute()

    while True:
        for scene_id in scenes:
            _deduplicate_scene(
                scene_id=scene_id,
                corpus_root=corpus_root,
                receipt_root=receipt_root,
            )
        receipts = sum(
            (receipt_root / f"{scene_id}.json").is_file()
            for scene_id in scenes
        )
        if not args.monitor or receipts == len(scenes):
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
