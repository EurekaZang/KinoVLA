#!/usr/bin/env python3
"""Fetch a small, frozen CC0 forest-prop bundle through Poly Haven's API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _request_json(url: str, *, user_agent: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _safe_relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe provider-relative path: {value!r}")
    return Path(*path.parts)


def _download(
    *,
    url: str,
    destination: Path,
    expected_size: int,
    expected_md5: str,
    user_agent: str,
) -> dict[str, Any]:
    if destination.is_file():
        if destination.stat().st_size != expected_size:
            raise ValueError(f"existing asset has wrong size: {destination}")
        if _digest(destination, "md5") != expected_md5:
            raise ValueError(f"existing asset has wrong MD5: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".part", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    shutil.copyfileobj(response, temporary, length=1024 * 1024)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
        if temporary_path.stat().st_size != expected_size:
            temporary_path.unlink(missing_ok=True)
            raise ValueError(f"downloaded asset has wrong size: {destination}")
        if _digest(temporary_path, "md5") != expected_md5:
            temporary_path.unlink(missing_ok=True)
            raise ValueError(f"downloaded asset has wrong MD5: {destination}")
        os.replace(temporary_path, destination)
    return {
        "path": str(destination),
        "url": url,
        "bytes": destination.stat().st_size,
        "provider_md5": expected_md5,
        "sha256": _digest(destination, "sha256"),
    }


def sync(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "kinofail.forest-prop-source.v1":
        raise ValueError("unsupported forest-prop source schema")
    provider = config["provider"]
    api_base = str(provider["api_base"]).rstrip("/")
    user_agent = str(provider["user_agent"])
    output = (ROOT / config["output"]["directory"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for requested in config["assets"]:
        asset_id = str(requested["id"])
        metadata = _request_json(f"{api_base}/info/{asset_id}", user_agent=user_agent)
        if int(metadata.get("type", -1)) != 2:
            raise ValueError(f"provider asset is not a 3D model: {asset_id}")
        files = _request_json(f"{api_base}/files/{asset_id}", user_agent=user_agent)
        try:
            selected = files[config["format"]][config["resolution"]][config["format"]]
        except KeyError as error:
            raise ValueError(f"missing requested USD resolution for {asset_id}") from error
        main_name = PurePosixPath(urllib.parse.urlparse(selected["url"]).path).name
        if not main_name:
            raise ValueError(f"provider returned an invalid primary URL for {asset_id}")
        asset_root = output / asset_id
        downloaded = [
            _download(
                url=str(selected["url"]),
                destination=asset_root / main_name,
                expected_size=int(selected["size"]),
                expected_md5=str(selected["md5"]),
                user_agent=user_agent,
            )
        ]
        for relative, spec in sorted(selected.get("include", {}).items()):
            downloaded.append(
                _download(
                    url=str(spec["url"]),
                    destination=asset_root / _safe_relative_path(relative),
                    expected_size=int(spec["size"]),
                    expected_md5=str(spec["md5"]),
                    user_agent=user_agent,
                )
            )
        records.append(
            {
                "id": asset_id,
                "name": metadata["name"],
                "role": requested["role"],
                "authors": metadata.get("authors", {}),
                "provider_dimensions_mm": metadata.get("dimensions"),
                "provider_polycount": metadata.get("polycount"),
                "provider_date_published_unix": metadata.get("date_published"),
                "tags": metadata.get("tags", []),
                "format": config["format"],
                "resolution": config["resolution"],
                "primary_path": str(asset_root / main_name),
                "files": downloaded,
            }
        )
    lock = {
        "schema_version": "kinofail.forest-prop-lock.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": {
            "path": str(config_path),
            "sha256": _digest(config_path, "sha256"),
        },
        "provider": provider,
        "assets": records,
        "development_only": True,
        "visual_only_by_default": True,
        "counts_as_scene_registry_admission": False,
        "counts_as_a0_a7_evidence": False,
    }
    lock_path = output / config["output"]["lock_file"]
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"lock": str(lock_path), "asset_count": len(records), "assets": records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/data/kinofail_forest_prop_assets_polyhaven_v1.json",
    )
    args = parser.parse_args()
    result = sync(args.config)
    print(json.dumps({"lock": result["lock"], "asset_count": result["asset_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
