#!/usr/bin/env python3
"""Run the frozen T2 feature extractor on the mirrored /data scratch tree.

Only path provenance is adapted.  The frozen wrapper, source extractor,
feature code, encoder, input arrays, ordering, and output schema are reused
unchanged.  The /home and /data trees must have the same repo-relative path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_REPO_ROOT = Path("/data/eureka/KinoVLA")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import extract_kinofail_confirmatory_t2_features_v1 as frozen


def mirrored_storage_mapping(corpus: Path) -> tuple[Path, Path, Path]:
    logical = Path(os.path.abspath(corpus))
    physical = logical.resolve(strict=True)
    try:
        logical_relative = logical.relative_to(ROOT)
        physical_relative = physical.relative_to(DATA_REPO_ROOT)
    except ValueError as error:
        raise RuntimeError("T2 corpus is outside the sealed mirror roots") from error
    if logical_relative != physical_relative:
        raise RuntimeError("logical and physical T2 corpus paths are not mirrored")
    return logical, physical, logical_relative


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    logical, physical, relative = mirrored_storage_mapping(args.corpus)
    previous_root = frozen.source.ROOT
    previous_argv = sys.argv
    try:
        # The physical tree mirrors the repo layout.  This changes only the
        # base used by Path.relative_to() in the frozen provenance writer.
        frozen.source.ROOT = DATA_REPO_ROOT
        sys.argv = [
            str(Path(frozen.__file__).resolve()),
            "--corpus",
            str(logical),
            "--output",
            str(args.output),
            "--scene-id",
            args.scene_id,
            "--batch-size",
            str(args.batch_size),
        ]
        result = int(frozen.main())
    finally:
        sys.argv = previous_argv
        frozen.source.ROOT = previous_root

    manifest_path = args.output.resolve() / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        result != 0
        or manifest.get("corpus") != relative.as_posix()
        or manifest.get("source_extractor_sha256")
        != frozen._sha256(Path(frozen.source.__file__).resolve())
        or manifest.get("wrapper_sha256")
        != frozen._sha256(Path(frozen.__file__).resolve())
    ):
        raise RuntimeError("storage-aware T2 provenance binding failed")
    print(
        json.dumps(
            {
                "passed": True,
                "scientific_content_changed": False,
                "logical_corpus": str(logical),
                "physical_corpus": str(physical),
                "repo_relative_corpus": relative.as_posix(),
                "frozen_wrapper": str(Path(frozen.__file__).resolve()),
                "frozen_source": str(Path(frozen.source.__file__).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
