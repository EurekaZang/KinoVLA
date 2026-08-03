#!/usr/bin/env python3
"""Add one already-frozen transitive dependency to the reconfirmation F0.

The v2 F0 binds the complete original F0 manifest, which had already frozen
``kino_vla/sim/c1_causal_visuals.py`` by content hash.  The v2 schedule
compiler also requires that dependency to appear in the v2 manifest's flat
file index.  This amendment copies the sealed F0 and adds only that redundant
flat-index record.  It cannot change any scientific design field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_SHA256 = (
    "128c9a5c3feac0bf7696310734782162c8c3080318b03c71076a485c2f0ed4ac"
)
ORIGINAL_F0_SHA256 = (
    "8eced1a4fa1ea7babbc6e6d5d3852a3a5c85896594dfd160b3279979820c80e5"
)
DEPENDENCY = ROOT / "kino_vla/sim/c1_causal_visuals.py"
DEPENDENCY_SHA256 = (
    "9a0c72b682e757198e1b40dedf281a7be42325ba041137de78e23dee9cc23c78"
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--original-f0", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base_path = args.base.resolve()
    original_path = args.original_f0.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    if _sha256(base_path) != BASE_SHA256:
        raise RuntimeError("base reconfirmation F0 hash mismatch")
    if _sha256(original_path) != ORIGINAL_F0_SHA256:
        raise RuntimeError("original F0 hash mismatch")
    if _sha256(DEPENDENCY) != DEPENDENCY_SHA256:
        raise RuntimeError("transitive dependency hash mismatch")
    base = _json(base_path)
    original = _json(original_path)
    original_records = {
        str(row["path"]): dict(row)
        for row in original.get("frozen_files", [])
    }
    dependency_record = original_records.get(
        "kino_vla/sim/c1_causal_visuals.py"
    )
    expected = {
        "path": "kino_vla/sim/c1_causal_visuals.py",
        "bytes": DEPENDENCY.stat().st_size,
        "sha256": DEPENDENCY_SHA256,
    }
    if dependency_record != expected:
        raise RuntimeError("original F0 transitive record mismatch")
    if any(
        row["path"] == expected["path"] for row in base.get("frozen_files", [])
    ):
        raise RuntimeError("base already has the direct flat-index record")
    if list(
        (ROOT / "outputs/kinofail_reconfirmation_v2/corpus").glob("**/*")
    ):
        raise RuntimeError("anomaly corpus exists before F0 binding amendment")
    if list(
        (ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2").glob("**/*")
    ):
        raise RuntimeError("model/evaluation artifacts exist before amendment")

    amended = deepcopy(base)
    amended["schema_version"] = (
        "kinofail.unified-reconfirmation-f0-transitive-amendment.v1"
    )
    amended["amendment"] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "base_manifest": str(base_path.relative_to(ROOT)),
        "base_manifest_sha256": BASE_SHA256,
        "reason": (
            "The schedule compiler requires a direct flat-index entry for the "
            "T2 visual helper. The helper was already frozen transitively by "
            "the original F0 bound in the base manifest."
        ),
        "added_path": expected["path"],
        "added_sha256": expected["sha256"],
        "original_f0_record_verified": True,
        "scientific_design_changed": False,
        "architecture_threshold_features_or_statistics_changed": False,
        "scene_or_material_selection_changed": False,
        "seed_or_operator_parameter_changed": False,
        "anomaly_episodes_available_at_amendment": 0,
        "model_predictions_available_at_amendment": 0,
        "amendment_builder": str(Path(__file__).resolve().relative_to(ROOT)),
        "amendment_builder_sha256": _sha256(Path(__file__).resolve()),
    }
    amended["frozen_files"].append(expected)
    amended["frozen_files"] = sorted(
        amended["frozen_files"], key=lambda row: str(row["path"])
    )
    out.mkdir(parents=True, exist_ok=False)
    file_text = "".join(
        f"{row['sha256']}  {row['bytes']}  {row['path']}\n"
        for row in amended["frozen_files"]
    )
    files_path = out / "files.sha256"
    files_path.write_text(file_text, encoding="utf-8")
    amended["files_manifest_sha256"] = _sha256(files_path)
    manifest_path = out / "freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(amended, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "freeze_manifest.sha256").write_text(
        f"{_sha256(manifest_path)}  freeze_manifest.json\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "manifest": str(manifest_path),
                "sha256": _sha256(manifest_path),
                "scientific_design_changed": False,
                "added_path": expected["path"],
                "added_sha256": expected["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
