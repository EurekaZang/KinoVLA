#!/usr/bin/env python3
"""Extract one scene's T2 features and bind the output to that scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import extract_kinofail_realistic_c1_causal_features_v1 as source  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    output = args.output.resolve()
    previous = sys.argv
    try:
        sys.argv = [
            str(Path(source.__file__).resolve()),
            "--corpus",
            str(args.corpus.resolve()),
            "--output",
            str(output),
            "--batch-size",
            str(args.batch_size),
        ]
        result = int(source.main())
    finally:
        sys.argv = previous
    manifest_path = output / "feature_manifest.json"
    records_path = output / "records.jsonl"
    features_path = output / "features.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if (
        result != 0
        or manifest.get("passed") is not True
        or {str(row["scene_cluster"]) for row in records} != {args.scene_id}
        or manifest.get("output_sha256", {}).get("records")
        != _sha256(records_path)
        or manifest.get("output_sha256", {}).get("features")
        != _sha256(features_path)
    ):
        raise RuntimeError("T2 scene feature binding failed")
    manifest["schema_version"] = "kinofail.confirmatory-t2-scene-features.v1"
    manifest["scene_id"] = args.scene_id
    manifest["scene_cluster"] = args.scene_id
    manifest["source_extractor_sha256"] = _sha256(Path(source.__file__).resolve())
    manifest["wrapper_sha256"] = _sha256(Path(__file__).resolve())
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "scene_id": args.scene_id,
                "samples": len(records),
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
