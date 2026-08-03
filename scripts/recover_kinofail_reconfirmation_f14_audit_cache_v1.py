#!/usr/bin/env python3
"""Recover the byte-identical pruned F14 audit from retained provenance.

The scene-01 raw inventory authenticates the original byte count and SHA-256.
The complete JSON was also retained in the local execution transcript.  This
script canonicalizes that recorded JSON exactly as the original F14 writer did
and refuses to write unless both independent checks match.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SESSION = Path(
    "/home/eureka/.codex/sessions/2026/07/17/"
    "rollout-2026-07-17T05-01-28-019f6f4f-2fe8-7f63-96a9-446c8cede6d6.jsonl"
)
CALL_ID = "call_8G6jWgutxxhM3m6ucBeLg8OZ"
RAW_INVENTORY = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/prune_receipts/"
    "confirm_v2_life_scene_01/raw_inventory.jsonl"
)
ORIGINAL_RELATIVE_PATH = (
    "c2_t2/launcher_audits/confirm_v2_life_scene_01_f14_resume.json"
)
CACHE = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/recovery_audit_cache/"
    "confirm_v2_life_scene_01_f14_resume.json"
)
EVAL_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_record() -> dict[str, Any]:
    for line in RAW_INVENTORY.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if value.get("path") == ORIGINAL_RELATIVE_PATH:
            return value
    raise RuntimeError("F14 audit is absent from the sealed raw inventory")


def transcript_record() -> tuple[dict[str, Any], str]:
    for raw_line in SESSION.open("rb"):
        row = json.loads(raw_line)
        payload = row.get("payload", {})
        if (
            payload.get("type") == "function_call_output"
            and payload.get("call_id") == CALL_ID
        ):
            output = str(payload.get("output", ""))
            marker = "Output:\n"
            if marker not in output:
                raise RuntimeError("recorded F14 output marker is absent")
            body = output.split(marker, 1)[1]
            start = body.find("{")
            end = body.rfind("}") + 1
            if start < 0 or end <= start:
                raise RuntimeError("recorded F14 JSON is absent")
            value = json.loads(body[start:end])
            if not isinstance(value, dict):
                raise TypeError("recorded F14 audit is not an object")
            return value, sha256_bytes(raw_line)
    raise RuntimeError("recorded F14 terminal output was not found")


def recover_canonical() -> tuple[bytes, dict[str, Any], str]:
    expected = inventory_record()
    value, transcript_line_sha256 = transcript_record()
    canonical = (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if (
        len(canonical) != int(expected.get("bytes", -1))
        or sha256_bytes(canonical) != expected.get("sha256")
        or value.get("state") != "terminal"
        or value.get("passed") is not True
        or value.get("all_scheduled_cases_accounted") is not True
        or int(value.get("summary_case_count", -1)) != 50
    ):
        raise RuntimeError("recovered F14 audit fails sealed authentication")
    return canonical, expected, transcript_line_sha256


def main() -> int:
    if list(EVAL_ROOT.glob("**/*prediction*")):
        raise RuntimeError("prediction artifact exists before audit recovery")
    canonical, expected, transcript_line_sha256 = recover_canonical()
    if CACHE.exists():
        if CACHE.read_bytes() != canonical:
            raise RuntimeError("an incompatible F14 audit cache already exists")
    else:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        temporary = CACHE.with_suffix(".json.tmp")
        temporary.write_bytes(canonical)
        os.replace(temporary, CACHE)
    print(
        json.dumps(
            {
                "cache": str(CACHE),
                "bytes": CACHE.stat().st_size,
                "sha256": sha256(CACHE),
                "raw_inventory_record": expected,
                "transcript_call_id": CALL_ID,
                "transcript_record_sha256": transcript_line_sha256,
                "byte_identical_to_pruned_original": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
