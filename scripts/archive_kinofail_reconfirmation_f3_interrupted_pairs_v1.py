#!/usr/bin/env python3
"""Recoverably archive exact partial pair directories interrupted for F3."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENE = "confirm_v2_life_scene_00"
CORPUS = Path(
    "/media/eureka/FC28565528560ED0/tmp/"
    f"KinoVLA_reconfirmation_v2/corpus/{SCENE}"
)
SCHEDULE = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/schedules/scenes"
    / SCENE
    / "scale/schedule.jsonl"
)
INCIDENT_ROOT = CORPUS / "incidents/f3_nuisance_contract_interruption"
RECEIPT = INCIDENT_ROOT / "archive_receipt.json"
INTERRUPTED_PAIR_IDS = {
    "cf_025108cfb719cb716b63",
    "cf_04678ef990b4ff57de94",
    "cf_07365e997113eacd4de6",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(path: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(item.relative_to(path)),
            "bytes": item.stat().st_size,
            "sha256": _sha256(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]


def main() -> int:
    if RECEIPT.exists():
        raise FileExistsError(RECEIPT)
    records = [
        json.loads(line)
        for line in SCHEDULE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = [
        row
        for row in records
        if str(row["counterfactual_group_id"]) in INTERRUPTED_PAIR_IDS
    ]
    if (
        {str(row["counterfactual_group_id"]) for row in selected}
        != INTERRUPTED_PAIR_IDS
        or len(selected) != 2 * len(INTERRUPTED_PAIR_IDS)
    ):
        raise RuntimeError("interrupted pair schedule binding is incomplete")

    rows = []
    for record in selected:
        relative = Path(str(record["required_outputs"]["episode_manifest"])).parent
        source = CORPUS / relative
        target = INCIDENT_ROOT / "partial_pairs" / relative
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        if not source.exists():
            rows.append(
                {
                    "counterfactual_group_id": record[
                        "counterfactual_group_id"
                    ],
                    "condition": record["condition"],
                    "source": str(source),
                    "artifact_existed": False,
                }
            )
            continue
        if source.is_symlink() or not source.is_dir():
            raise RuntimeError(f"unsafe partial source: {source}")
        before = _inventory(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        after = _inventory(target)
        if before != after or source.exists():
            raise RuntimeError("recoverable partial archive verification failed")
        rows.append(
            {
                "counterfactual_group_id": record[
                    "counterfactual_group_id"
                ],
                "condition": record["condition"],
                "source": str(source),
                "archive": str(target),
                "artifact_existed": True,
                "files": len(after),
                "bytes": sum(int(row["bytes"]) for row in after),
                "inventory": after,
            }
        )

    value = {
        "schema_version": "kinofail.reconfirmation-f3-partial-archive.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "operation": "recoverable_move",
        "interrupted_pair_ids": sorted(INTERRUPTED_PAIR_IDS),
        "schedule": str(SCHEDULE),
        "schedule_sha256": _sha256(SCHEDULE),
        "rows": rows,
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    temporary = RECEIPT.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, RECEIPT)
    print(
        json.dumps(
            {
                "passed": True,
                "receipt": str(RECEIPT),
                "receipt_sha256": _sha256(RECEIPT),
                "archived_directories": sum(
                    row["artifact_existed"] is True for row in rows
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
