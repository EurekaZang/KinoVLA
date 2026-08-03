#!/usr/bin/env python3
"""Archive the F3 pre-cleanup attempt and its exact leftover directories."""

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
F3 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f3_nuisance_contract_amendment1/"
    "amendment_manifest.json"
)
EXPECTED_F3_SHA256 = (
    "c17bd336f93953f95eb7b5ff0aa53242b9572d2b139246b79d13646c7f4036f0"
)
INCIDENT = CORPUS / "incidents/f3_nuisance_contract_interruption"
SUPPLEMENT = INCIDENT / "precleanup_v7_attempt"
RECEIPT = SUPPLEMENT / "archive_receipt.json"
AUDIT = CORPUS / "launcher_audits/scale_f3v7_partition_3_of_4.json"
LOG_ROOT = CORPUS / "launcher_logs/scale_f3v7"
PAIR_IDS = {
    "cf_0181579aad8dd1e2599c",
    "cf_07365e997113eacd4de6",
}
SCHEDULE = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/schedules/scenes"
    / SCENE
    / "scale/schedule.jsonl"
)


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


def _move(source: Path, target: Path) -> dict[str, object]:
    if not source.exists() or source.is_symlink():
        raise FileNotFoundError(source)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    before = _inventory(source) if source.is_dir() else None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    if source.exists():
        raise RuntimeError("source remained after recoverable move")
    after = _inventory(target) if target.is_dir() else None
    if before != after:
        raise RuntimeError("archive inventory changed")
    return {
        "source": str(source),
        "archive": str(target),
        "kind": "directory" if target.is_dir() else "file",
        "sha256": _sha256(target) if target.is_file() else None,
        "inventory": after,
    }


def main() -> int:
    if _sha256(F3) != EXPECTED_F3_SHA256:
        raise RuntimeError("F3 amendment hash mismatch")
    if RECEIPT.exists():
        raise FileExistsError(RECEIPT)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    attempts = {
        str(row["counterfactual_group_id"]): row
        for row in audit["attempts"]
    }
    if set(attempts) != PAIR_IDS:
        raise RuntimeError("pre-cleanup attempt pair IDs changed")
    if (
        attempts["cf_0181579aad8dd1e2599c"].get("state") != "terminal"
        or attempts["cf_0181579aad8dd1e2599c"].get("summary_exists") is not False
        or attempts["cf_07365e997113eacd4de6"].get("state") != "started"
    ):
        raise RuntimeError("pre-cleanup attempt state changed")

    moves = [
        _move(AUDIT, SUPPLEMENT / "launcher_audit.json"),
        _move(LOG_ROOT, SUPPLEMENT / "launcher_logs"),
    ]
    records = [
        json.loads(line)
        for line in SCHEDULE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for record in records:
        if str(record["counterfactual_group_id"]) not in PAIR_IDS:
            continue
        relative = Path(str(record["required_outputs"]["episode_manifest"])).parent
        source = CORPUS / relative
        if source.exists():
            moves.append(
                _move(
                    source,
                    SUPPLEMENT / "partial_pairs" / relative,
                )
            )

    value = {
        "schema_version": "kinofail.reconfirmation-f3-precleanup-archive.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "reason": (
            "F3-authorized reexecution encountered directories created before "
            "the F0 guard exception; one following pair was interrupted."
        ),
        "model_or_endpoint_outcomes_used": False,
        "f3_amendment": str(F3),
        "f3_amendment_sha256": _sha256(F3),
        "pair_ids": sorted(PAIR_IDS),
        "moves": moves,
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
                "moves": len(moves),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
