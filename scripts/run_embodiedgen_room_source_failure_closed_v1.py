#!/usr/bin/env python3
"""Run one frozen RoomGen request and preserve a terminal failure artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate_embodiedgen_room_source.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embodiedgen-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--room-type", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--complexity", required=True)
    parser.add_argument("--exception-out", type=Path, required=True)
    args = parser.parse_args()

    command = [
        sys.executable,
        str(GENERATOR),
        "--embodiedgen-root",
        str(args.embodiedgen_root.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--scene-id",
        args.scene_id,
        "--room-type",
        args.room_type,
        "--seed",
        str(args.seed),
        "--complexity",
        args.complexity,
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode == 0:
        return 0

    out = args.exception_out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite generation failure: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "kinofail.embodiedgen-generation-exception.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": False,
        "scene_id": args.scene_id,
        "room_type": args.room_type,
        "seed": args.seed,
        "complexity": args.complexity,
        "generator": str(GENERATOR),
        "generator_sha256": _sha256(GENERATOR),
        "returncode": completed.returncode,
        "replacement_authorized": False,
        "formal_operator_run_authorized": False,
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "passed": False}, indent=2))
    return completed.returncode or 2


if __name__ == "__main__":
    raise SystemExit(main())
