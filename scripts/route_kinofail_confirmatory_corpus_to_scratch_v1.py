#!/usr/bin/env python3
"""Route not-yet-started confirmatory scene corpora to isolated NVMe scratch.

The workspace keeps the frozen logical corpus paths as symlinks.  This script
refuses to replace any existing path and never touches an active scene.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "outputs/kinofail_confirmatory_v1/scene_registry.json"
DEFAULT_LOGICAL_ROOT = ROOT / "outputs/kinofail_confirmatory_v1/corpus"
DEFAULT_SCRATCH_ROOT = Path(
    "/media/eureka/FC28565528560ED0/tmp/"
    "KinoVLA_confirmatory_20260725/corpus"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--logical-root", type=Path, default=DEFAULT_LOGICAL_ROOT)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--start-scene-index", type=int, default=1)
    args = parser.parse_args()

    registry_path = args.registry.resolve()
    logical_root = args.logical_root.resolve()
    # Do not resolve the scratch root before it exists.
    scratch_root = args.scratch_root.absolute()
    registry = _json(registry_path)
    scenes = [str(row["scene_id"]) for row in registry["scenes"]]
    if len(scenes) != 30 or len(set(scenes)) != 30:
        raise RuntimeError("confirmatory registry must contain 30 scenes")
    selected = scenes[args.start_scene_index :]
    if not selected:
        raise ValueError("empty scratch-routing scene slice")

    scratch_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(scratch_root)
    if usage.free < 100 * 1024**3:
        raise RuntimeError("scratch volume has less than 100 GiB free")

    routed = []
    for scene_id in selected:
        logical = logical_root / scene_id
        physical = scratch_root / scene_id
        if logical.exists() or logical.is_symlink():
            raise FileExistsError(
                f"refusing to replace existing scene corpus path: {logical}"
            )
        physical.mkdir(parents=True, exist_ok=False)
        logical.symlink_to(physical, target_is_directory=True)
        if logical.resolve() != physical.resolve():
            raise RuntimeError(f"scratch route did not resolve: {logical}")
        routed.append(
            {
                "scene_id": scene_id,
                "logical_path": str(logical),
                "physical_path": str(physical),
            }
        )

    audit = {
        "schema_version": "kinofail.confirmatory-corpus-scratch-routing.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "scientific_design_changed": False,
        "schedule_or_record_changed": False,
        "scene_registry_sha256": _sha256(registry_path),
        "logical_root": str(logical_root),
        "scratch_root": str(scratch_root),
        "scratch_device": os.stat(scratch_root).st_dev,
        "free_bytes_before_routing": usage.free,
        "routed_scenes": routed,
        "excluded_active_scenes": scenes[: args.start_scene_index],
    }
    audit_path = (
        logical_root.parent / "scratch_routing_audit_20260725.json"
    )
    if audit_path.exists():
        raise FileExistsError(audit_path)
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
