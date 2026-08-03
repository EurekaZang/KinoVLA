#!/usr/bin/env python3
"""Download and freeze Poly Haven forest appearance maps as a terrain lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/data/kinofail_forest_texture_assets_polyhaven_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/assets/forest_textures_polyhaven_v1",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "kinofail.forest-texture-assets.v1":
        raise ValueError("unsupported forest texture config")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "terrain_assets.lock.json"
    if lock_path.exists():
        raise FileExistsError(f"refusing to overwrite forest texture lock: {lock_path}")

    locked = []
    for material in config["materials"]:
        maps = {}
        for map_name, spec in material["maps"].items():
            relative = Path(material["id"]) / spec["filename"]
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                temporary = destination.with_suffix(destination.suffix + ".partial")
                if temporary.exists():
                    temporary.unlink()
                urllib.request.urlretrieve(spec["url"], temporary)
                temporary.replace(destination)
            if destination.stat().st_size != int(spec["bytes"]):
                raise ValueError(f"unexpected bytes for {destination}")
            if _digest(destination, "md5") != spec["md5"]:
                raise ValueError(f"provider MD5 mismatch for {destination}")
            maps[map_name] = {
                "path": str(relative),
                "bytes": destination.stat().st_size,
                "provider_md5": spec["md5"],
                "sha256": _digest(destination, "sha256"),
                "url": spec["url"],
            }
        locked.append(
            {
                **{key: value for key, value in material.items() if key != "maps"},
                "license": "CC0",
                "maps": maps,
            }
        )

    lock = {
        "schema_version": "kinofail.terrain-assets.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "asset_root": str(output.relative_to(ROOT)),
        "catalog_config": {
            "path": str(config_path),
            "sha256": _digest(config_path, "sha256"),
        },
        "provider": config["provider"],
        "selection_policy": config["selection_policy"],
        "materials": locked,
        "audit": {
            "passed": True,
            "n_materials": len(locked),
            "split_counts": {
                split: sum(item["split"] == split for item in locked)
                for split in ("train", "val", "test")
            },
            "appearance_only": True,
            "counts_as_a0_a7_evidence": False,
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "materials": len(locked), "lock": str(lock_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
