#!/usr/bin/env python3
"""Archive completed T3 launcher state before Scale uses the same namespace.

The frozen per-scene launcher writes ``launcher_audits`` independently of the
battery name.  T3 and Scale therefore need an append-only namespace handoff.
This operation moves only completed launcher audit directories, preserving
every byte and leaving episodes, summaries, schedules, and scientific inputs
unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "outputs/kinofail_confirmatory_v1/scene_registry.json"
DEFAULT_CORPUS = ROOT / "outputs/kinofail_confirmatory_v1/corpus"
DEFAULT_RECEIPTS = (
    ROOT / "outputs/kinofail_confirmatory_v1/storage_receipts"
)
DEFAULT_F2 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_confirmatory_f2_scene_source_amendment/"
    "amendment_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_confirmatory_f3_runtime_namespace/"
    "amendment_manifest.json"
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


def _validate_partition(path: Path, partition: int) -> dict[str, Any]:
    audit = _json(path)
    selected = [str(value) for value in audit.get("selected_pair_ids", [])]
    attempts = [
        str(row["counterfactual_group_id"])
        for row in audit.get("attempts", [])
    ]
    if (
        audit.get("partition_index") != partition
        or len(selected) != 50
        or len(attempts) != 50
        or set(selected) != set(attempts)
    ):
        raise RuntimeError(f"incomplete T3 launcher audit: {path}")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--storage-receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--f2-manifest", type=Path, default=DEFAULT_F2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.absolute()
    if output.exists():
        raise FileExistsError(output)
    registry_path = args.registry.resolve()
    corpus_root = args.corpus_root.absolute()
    receipt_root = args.storage_receipts.absolute()
    f2_path = args.f2_manifest.resolve()
    registry = _json(registry_path)
    scenes = [str(row["scene_id"]) for row in registry["scenes"]]
    if len(scenes) != 30 or len(set(scenes)) != 30:
        raise RuntimeError("confirmatory registry must contain 30 scenes")

    prepared = []
    for scene_id in scenes:
        receipt_path = receipt_root / f"{scene_id}.json"
        if _json(receipt_path).get("passed") is not True:
            raise RuntimeError(f"storage receipt is incomplete: {scene_id}")
        scene = corpus_root / scene_id
        source = scene / "launcher_audits"
        destination = scene / "c2_t3_launcher_audits_f2"
        if not source.is_dir() or destination.exists() or destination.is_symlink():
            raise RuntimeError(f"invalid T3 audit handoff state: {scene_id}")
        files = []
        for partition in (0, 1):
            path = source / f"partition_{partition}_of_2.json"
            audit = _validate_partition(path, partition)
            files.append(
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "selected_pairs": len(audit["selected_pair_ids"]),
                    "terminal_attempts": len(audit["attempts"]),
                    "passed_pairs": sum(
                        row.get("passed") is True
                        for row in audit["attempts"]
                    ),
                    "failed_pairs": sum(
                        row.get("passed") is not True
                        for row in audit["attempts"]
                    ),
                }
            )
        extra = sorted(
            str(path.relative_to(source))
            for path in source.rglob("*")
            if path.is_file()
            and path.name
            not in {"partition_0_of_2.json", "partition_1_of_2.json"}
        )
        if extra:
            raise RuntimeError(f"unexpected T3 launcher audit files: {extra[:3]}")
        prepared.append(
            {
                "scene_id": scene_id,
                "source": str(source),
                "destination": str(destination),
                "files": files,
            }
        )

    for row in prepared:
        source = Path(row["source"])
        destination = Path(row["destination"])
        source.rename(destination)
        if source.exists() or not destination.is_dir():
            raise RuntimeError("T3 launcher namespace handoff failed")
        for file_row in row["files"]:
            archived = destination / Path(file_row["path"]).name
            if _sha256(archived) != file_row["sha256"]:
                raise RuntimeError("archived T3 launcher audit hash changed")
            file_row["archived_path"] = str(archived)

    amendment = {
        "schema_version":
        "kinofail.confirmatory-f3-runtime-namespace-amendment.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed",
        "confirmatory": True,
        "reason": (
            "The frozen generic launcher used the battery-independent "
            "launcher_audits basename for both T3 and Scale."
        ),
        "scientific_design_changed": False,
        "architecture_or_checkpoint_changed": False,
        "threshold_or_feature_changed": False,
        "statistics_changed": False,
        "sample_or_schedule_record_changed": False,
        "retry_authorized": False,
        "episodes_or_pair_summaries_moved": False,
        "change": (
            "Completed T3 launcher audit directories were byte-preservingly "
            "renamed to c2_t3_launcher_audits_f2 before Scale acquisition."
        ),
        "source_sha256": {
            "f2_manifest": _sha256(f2_path),
            "scene_registry": _sha256(registry_path),
        },
        "scenes": prepared,
        "counts": {
            "scenes": len(prepared),
            "planned_t3_pairs": 3_000,
            "terminal_t3_attempts": sum(
                file_row["terminal_attempts"]
                for row in prepared
                for file_row in row["files"]
            ),
            "passed_t3_pairs": sum(
                file_row["passed_pairs"]
                for row in prepared
                for file_row in row["files"]
            ),
            "failed_t3_pairs": sum(
                file_row["failed_pairs"]
                for row in prepared
                for file_row in row["files"]
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(
        json.dumps(amendment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(amendment, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
