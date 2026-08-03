#!/usr/bin/env python3
"""Download and freeze the CC0 photographic HDRI used by the forest hybrid."""

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
        default=ROOT / "configs/data/kinofail_forest_hdri_assets_polyhaven_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/assets/forest_hdri_polyhaven_v1",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "kinofail.forest-hdri-assets.v1":
        raise ValueError("unsupported forest HDRI asset config")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "forest_hdri.lock.json"
    if lock_path.exists():
        raise FileExistsError(f"refusing to overwrite HDRI lock: {lock_path}")

    locked_assets = []
    for asset in config["assets"]:
        file_spec = asset["file"]
        destination = output / file_spec["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".partial")
            if temporary.exists():
                temporary.unlink()
            urllib.request.urlretrieve(file_spec["url"], temporary)
            temporary.replace(destination)
        if destination.stat().st_size != int(file_spec["bytes"]):
            raise ValueError(f"unexpected byte count for {destination}")
        if _digest(destination, "md5") != file_spec["md5"]:
            raise ValueError(f"provider MD5 mismatch for {destination}")
        locked_assets.append(
            {
                **{key: value for key, value in asset.items() if key != "file"},
                "file": {
                    **file_spec,
                    "path": str(destination),
                    "sha256": _digest(destination, "sha256"),
                },
            }
        )

    lock = {
        "schema_version": "kinofail.forest-hdri-lock.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": {
            "path": str(config_path),
            "sha256": _digest(config_path, "sha256"),
        },
        "provider": config["provider"],
        "selection_policy": config["selection_policy"],
        "assets": locked_assets,
        "interpretation_policy": config["interpretation_policy"],
        "passed": True,
        "counts_as_a0_a7_evidence": False,
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "lock": str(lock_path), "assets": len(locked_assets)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
